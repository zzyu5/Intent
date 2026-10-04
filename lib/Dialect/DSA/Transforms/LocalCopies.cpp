#include "Intent/Dialect/DSA/Transforms/StoragePatterns.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/IR/MemoryEffects.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::dsa {
namespace {

struct LocalCopy {
  Operation *operation;
  Value source, destination, sourceOwner;
};

std::optional<LocalCopy> fullCopy(Operation *operation, StorageAnalysis &storage,
                                 func::FuncOp function) {
  Value source, destination;
  if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
    source = copy.getSource(); destination = copy.getTarget();
  } else {
    auto load = cast<LoadTileOp>(operation);
    if (load.getAsynchronous()) return std::nullopt;
    source = load.getSource(); destination = load.getOutput();
    auto shape = cast<MemRefType>(destination.getType());
    auto exact = [&](Value value, int64_t expected) {
      auto interval = integerInterval(value, function);
      return interval && interval->first == expected && interval->second == expected;
    };
    if (!exact(load.getOffset(), 0) || !exact(load.getRows(), shape.getDimSize(0)) ||
        !exact(load.getColumns(), shape.getDimSize(1)) ||
        (shape.getDimSize(0) > 1 && !exact(load.getRowStride(), shape.getDimSize(1))) ||
        (shape.getDimSize(1) > 1 && !exact(load.getColumnStride(), 1))) return std::nullopt;
  }
  auto input = cast<MemRefType>(source.getType());
  auto output = cast<MemRefType>(destination.getType());
  Value origin = storage.uniqueOrigin(source);
  if (!input.hasStaticShape() || !output.hasStaticShape() ||
      input.getMemorySpaceAsInt() != nramSpace || output.getMemorySpaceAsInt() != nramSpace ||
      !input.getLayout().isIdentity() || !output.getLayout().isIdentity() ||
      input.getElementType() != output.getElementType() ||
      input.getNumElements() != output.getNumElements() || !origin ||
      !origin.getDefiningOp<memref::AllocaOp>() ||
      !destination.getDefiningOp<memref::AllocaOp>() || origin == destination ||
      !isCompleteLocalStorageView(source) || !isCompleteLocalStorageView(destination) ||
      !isCompleteStorageViewOf(source, origin) ||
      !storage.aliases(origin).complete || !storage.aliases(destination).complete)
    return std::nullopt;
  return LocalCopy{operation, source, destination, origin};
}

bool observes(const BufferStorageEffects &effects, Value memory,
              StorageAnalysis &storage) {
  return llvm::any_of(effects.entries, [&](const auto &entry) {
    return isa<MemoryEffects::Read>(entry.effect.getEffect()) &&
           !storage.disjoint(memory, entry.effect.getValue());
  });
}

bool modifies(const BufferStorageEffects &effects, Value memory,
              StorageAnalysis &storage) {
  return llvm::any_of(effects.entries, [&](const auto &entry) {
    return !isa<MemoryEffects::Read, MemoryEffects::Allocate>(entry.effect.getEffect()) &&
           !storage.disjoint(memory, entry.effect.getValue());
  });
}

// The copy defines one complete version. A later overwrite can terminate it;
// other writes to the allocation outside that interval are not its consumers.
bool forwardVersion(const LocalCopy &copy, StorageAnalysis &storage,
                    DominanceInfo &dominance) {
  auto aliases = storage.aliases(copy.destination);
  llvm::SetVector<OpOperand *> reads;
  llvm::SmallPtrSet<Operation *, 16> readers;
  bool overwritten = false;
  for (Operation *next = copy.operation->getNextNode();
       next && !next->hasTrait<OpTrait::IsTerminator>(); next = next->getNextNode()) {
    auto effects = storage.effects(next);
    if (!effects.complete || effects.ordered) break;
    if (modifies(effects, copy.destination, storage)) {
      OpOperand *output = detail::completeLocalOutput(next);
      overwritten = output &&
          isCompleteStorageViewOf(output->get(), copy.destination) &&
          !observes(effects, copy.destination, storage);
      break;
    }
    for (const auto &entry : effects.entries) {
      if (!isa<MemoryEffects::Read>(entry.effect.getEffect()) ||
          storage.disjoint(copy.destination, entry.effect.getValue())) continue;
      Operation *reader = entry.operation;
      if (!readers.insert(reader).second) continue;
      auto completion = storage.completionOfUse(reader);
      if (!dominance.properlyDominates(copy.operation, reader) ||
          failed(completion) || *completion != reader ||
          !storage.preserves(reader, copy.destination) ||
          !storage.readStable(copy.source, copy.operation, reader)) return false;
      bool actual = false;
      for (OpOperand &operand : reader->getOpOperands()) {
        if (!llvm::is_contained(aliases.values, operand.get())) continue;
        if (!isCompleteStorageViewOf(operand.get(), copy.destination)) return false;
        reads.insert(&operand);
        actual = true;
      }
      if (!actual) return false;
    }
  }
  if (!overwritten) {
    // A local instance dies with its block. For an enclosing carry allocation,
    // every observation must belong to this version's proved read interval;
    // static dominance alone does not authorize crossing its backedge.
    bool local = copy.destination.getDefiningOp()->getBlock() == copy.operation->getBlock();
    for (Operation *user : aliases.users) {
      if (isBufferStorageAliasOperation(user)) continue;
      auto effects = storage.effects(user);
      if (!effects.complete) return false;
      if (!observes(effects, copy.destination, storage) || readers.contains(user)) continue;
      Operation *point = copy.operation->getBlock()->findAncestorOpInBlock(*user);
      if (!local || !point || !point->isBeforeInBlock(copy.operation)) return false;
    }
  }
  OpBuilder builder(copy.operation);
  DenseMap<Type, Value> views;
  for (OpOperand *read : reads) {
    Type type = read->get().getType();
    Value &replacement = views[type];
    if (!replacement) {
      auto view = materializeCollectiveView(builder, copy.operation->getLoc(),
                                            copy.source, cast<MemRefType>(type));
      assert(succeeded(view) && "complete local copies preserve storage capacity");
      replacement = *view;
    }
    read->set(replacement);
  }
  copy.operation->erase();
  return true;
}

// Publish directly into the copied-to version, without consuming an earlier
// snapshot. Only the complete output slot changes; all source reads remain the
// original operands, and no new input/output alias is introduced.
bool writeThrough(const LocalCopy &copy, StorageAnalysis &storage,
                  DominanceInfo &dominance) {
  Operation *writer = storage.lastWriterBefore(copy.source, copy.operation);
  if (!writer || writer->getBlock() != copy.operation->getBlock() ||
      writer->getNumRegions() || writer->getNumResults() ||
      copy.sourceOwner.getDefiningOp()->getBlock() != writer->getBlock()) return false;
  OpOperand *output = detail::completeLocalOutput(writer);
  if (!output || !isCompleteStorageViewOf(output->get(), copy.sourceOwner) ||
      !dominance.properlyDominates(copy.destination, writer)) return false;
  auto from = cast<memref::AllocaOp>(copy.sourceOwner.getDefiningOp());
  auto to = cast<memref::AllocaOp>(copy.destination.getDefiningOp());
  if (from.getAlignment().value_or(0) > to.getAlignment().value_or(0)) return false;
  auto writing = storage.effects(writer);
  if (!writing.complete || writing.ordered ||
      requiredCompletion(writer) != CompletionScope::Immediate) return false;
  for (OpOperand &operand : writer->getOpOperands())
    if (&operand != output && isa<MemRefType>(operand.get().getType()) &&
        !storage.disjoint(operand.get(), copy.destination)) return false;
  for (Operation *operation = writer->getNextNode(); operation != copy.operation;
       operation = operation->getNextNode()) {
    auto effects = storage.effects(operation);
    if (!effects.complete || effects.ordered ||
        observes(effects, copy.destination, storage) ||
        modifies(effects, copy.destination, storage)) return false;
  }
  for (Value memory : {copy.sourceOwner, copy.destination}) {
    for (Operation *user : storage.aliases(memory).users) {
      if (isBufferStorageAliasOperation(user)) continue;
      // No pending use can extend through the earlier publication point.
      if (requiredCompletion(user) != CompletionScope::Immediate) return false;
      if (memory != copy.sourceOwner || user == writer || user == copy.operation) continue;
      auto effects = storage.effects(user);
      if (!effects.complete || effects.ordered) return false;
      if (!observes(effects, memory, storage) && !modifies(effects, memory, storage)) continue;
      Operation *local = writer->getBlock()->findAncestorOpInBlock(*user);
      if (!local || local == writer || !local->isBeforeInBlock(writer)) return false;
    }
  }
  OpBuilder builder(writer);
  auto destination = materializeCollectiveView(builder, writer->getLoc(),
      copy.destination, cast<MemRefType>(output->get().getType()));
  assert(succeeded(destination) && "complete local copies preserve output schema");
  output->set(*destination);
  copy.operation->erase();
  return true;
}

} // namespace

bool forwardFullLocalCopies(func::FuncOp function) {
  SmallVector<Operation *> copies;
  function.walk([&](Operation *operation) {
    if (isa<memref::CopyOp, LoadTileOp>(operation)) copies.push_back(operation);
  });
  bool changed = false;
  for (Operation *operation : copies) {
    StorageAnalysis storage(function);
    auto copy = fullCopy(operation, storage, function);
    if (!copy) continue;
    DominanceInfo dominance(function);
    changed |= forwardVersion(*copy, storage, dominance) ||
               writeThrough(*copy, storage, dominance);
  }
  return changed;
}

} // namespace intent::dsa

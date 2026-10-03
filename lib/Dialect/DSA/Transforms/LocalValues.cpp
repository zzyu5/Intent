#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/IR/ExecutionRelations.h"
#include "Intent/Dialect/DSA/IR/Views.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

using namespace mlir;
namespace intent::dsa {
namespace {
bool supportedScalarBinary(BinaryOperator kind) {
  return kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply;
}
} // namespace
void bindUniformOperands(func::FuncOp function) {
  SmallVector<dsa::BinaryOp> binaries;
  function.walk([&](dsa::BinaryOp binary) { binaries.push_back(binary); });
  bool changed = false;
  for (auto binary : binaries) {
    if (!supportedScalarBinary(binary.getKind()) ||
        binary.getApproximate() || binary.getFlushToZero()) continue;
    Type element = cast<MemRefType>(binary.getLhs().getType()).getElementType();
    if (!element.isF16() && !element.isF32()) continue;
    Value previous = binary.getRhs();
    auto fill = uniformFillBefore(previous, binary);
    if (!fill) continue;
    binary.getRhsMutable().assign(fill.getValue());
    changed = true;
  }
  function.walk([&](dsa::SelectOp select) {
    if (!isa<MemRefType>(select.getFalseValue().getType())) return;
    auto fill = uniformFillBefore(select.getFalseValue(), select);
    FloatAttr constant;
    if (fill && matchPattern(fill.getValue(), m_Constant(&constant))) {
      select.getFalseValueMutable().assign(fill.getValue());
      changed = true;
    }
  });
  if (changed)
    eliminateUnreadLocalWrites(function);
}

void eliminateOverwrittenFills(func::FuncOp function) {
  SmallVector<dsa::FillOp> fills;
  function.walk([&](dsa::FillOp fill) { fills.push_back(fill); });
  for (auto fill : fills) {
    Value output = fill.getOutput();
    if (!output.getDefiningOp<memref::AllocaOp>()) continue;
    StorageAnalysis storage(function);
    if (!storage.aliases(output).complete)
      continue;
    for (Operation *next = fill->getNextNode(); next;
         next = next->getNextNode()) {
      auto effects = storage.effects(next);
      if (!effects.complete || effects.ordered)
        break;
      bool read = false, written = false;
      for (const auto &entry : effects.entries) {
        const auto &effect = entry.effect;
        if (isa<MemoryEffects::Allocate>(effect.getEffect()) ||
            storage.disjoint(output, effect.getValue()))
          continue;
        written |= isa<MemoryEffects::Write>(effect.getEffect());
        read |= !isa<MemoryEffects::Write>(effect.getEffect());
        if (!effect.getValue()) read = true;
      }
      if (read) break;
      if (!written) continue;
      bool complete = isa<dsa::FillOp, dsa::LoadTileOp, dsa::GatherRowsOp, dsa::GroupGatherRowsOp, dsa::TransposeOp, dsa::UnaryOp, dsa::BinaryOp, dsa::CompareOp,
                          dsa::CastOp, dsa::SelectOp, memref::CopyOp>(next);
      if (auto divide = dyn_cast<dsa::DivideCastOp>(next)) complete = divide.getOutput() == output;
      // Aliasing alone does not prove that a write covers the original tile.
      if (complete) {
        for (const auto &entry : effects.entries)
          if (isa<MemoryEffects::Write>(entry.effect.getEffect()) &&
              !storage.disjoint(output, entry.effect.getValue()))
            complete &=
                isCompleteStorageViewOf(entry.effect.getValue(), output);
      }
      if (complete) fill.erase();
      break;
    }
  }
}

bool eliminateUnreadLocalWrites(func::FuncOp function) {
  SmallVector<memref::AllocaOp> allocations;
  function.walk([&](memref::AllocaOp allocation) {
    if (allocation.getType().getMemorySpaceAsInt() == dsa::nramSpace) allocations.push_back(allocation);
  });
  bool changed = false;
  for (auto allocation : allocations) {
    StorageAnalysis storage(function);
    auto aliases = storage.aliases(allocation);
    if (!aliases.complete)
      continue;
    SmallVector<Operation *> writes;
    bool unread = true;
    for (Operation *user : aliases.users) {
      if (isBufferStorageAliasOperation(user))
        continue;
      bool write = false;
      if (auto fill = dyn_cast<dsa::FillOp>(user))
        write = storage.uniqueOrigin(fill.getOutput()) == allocation;
      if (auto store = dyn_cast<memref::StoreOp>(user))
        write = storage.uniqueOrigin(store.getMemref()) == allocation;
      if (auto copy = dyn_cast<memref::CopyOp>(user))
        write = storage.uniqueOrigin(copy.getTarget()) == allocation &&
                storage.disjoint(allocation, copy.getSource());
      if (auto load = dyn_cast<dsa::LoadTileOp>(user))
        write = !load.getAsynchronous() &&
                storage.uniqueOrigin(load.getOutput()) == allocation &&
                storage.disjoint(allocation, load.getSource());
      if (auto cast = dyn_cast<dsa::CastOp>(user))
        write = storage.uniqueOrigin(cast.getOutput()) == allocation &&
                storage.disjoint(allocation, cast.getInput());
      if (!write) unread = false;
      else if (!llvm::is_contained(writes, user)) writes.push_back(user);
    }
    if (!unread || writes.empty()) continue;
    for (Operation *write : writes) write->erase();
    for (Value alias : llvm::reverse(aliases.values))
      if (alias.use_empty() && alias.getDefiningOp())
        alias.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}

bool forwardFullLocalCopies(func::FuncOp function) {
  SmallVector<Operation *> copies;
  function.walk([&](Operation *op) {
    if (isa<memref::CopyOp, dsa::LoadTileOp>(op)) copies.push_back(op);
  });
  auto exact = [&](Value value, int64_t expected) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == expected && interval->second == expected;
  };
  bool changed = false;
  for (Operation *copy : copies) {
    StorageAnalysis storage(function);
    Value source, destination;
    if (auto local = dyn_cast<memref::CopyOp>(copy)) {
      source = local.getSource(); destination = local.getTarget();
    } else {
      auto load = cast<dsa::LoadTileOp>(copy);
      if (load.getAsynchronous()) continue;
      source = load.getSource(); destination = load.getOutput();
      auto shape = cast<MemRefType>(destination.getType());
      if (!exact(load.getOffset(), 0) || !exact(load.getRows(), shape.getDimSize(0)) ||
          !exact(load.getColumns(), shape.getDimSize(1)) ||
          (shape.getDimSize(0) > 1 && !exact(load.getRowStride(), shape.getDimSize(1))) ||
          (shape.getDimSize(1) > 1 && !exact(load.getColumnStride(), 1))) continue;
    }
    auto input = dyn_cast<MemRefType>(source.getType());
    auto output = cast<MemRefType>(destination.getType());
    Value sourceOwner = storage.uniqueOrigin(source);
    if (!input || !input.hasStaticShape() || !output.hasStaticShape() ||
        input.getMemorySpaceAsInt() != dsa::nramSpace ||
        output.getMemorySpaceAsInt() != dsa::nramSpace ||
        !input.getLayout().isIdentity() || !output.getLayout().isIdentity() ||
        input.getElementType() != output.getElementType() ||
        input.getNumElements() != output.getNumElements() || !sourceOwner ||
        !sourceOwner.getDefiningOp<memref::AllocaOp>() ||
        !destination.getDefiningOp<memref::AllocaOp>() ||
        sourceOwner == destination)
      continue;
    if (!isCompleteStorageViewOf(source, sourceOwner))
      continue;
    auto destinationAliases = storage.aliases(destination);
    if (!destinationAliases.complete || !storage.aliases(sourceOwner).complete)
      continue;
    Operation *last = copy;
    bool eligible = true;
    // A copied snapshot can be forwarded only to read-only consumers. Views
    // retain the entire owned allocation, as required by the DSA verifier.
    for (Operation *user : destinationAliases.users) {
      if (user == copy) continue;
      if (user->getBlock() != copy->getBlock() || !copy->isBeforeInBlock(user)) { eligible = false; break; }
      if (last->isBeforeInBlock(user)) last = user;
      auto completion = storage.completionOfUse(user);
      if (failed(completion) || *completion != user ||
          !storage.preserves(user, destination))
        eligible = false;
    }
    if (!eligible || last == copy) continue;
    // Preserve the source snapshot through its final redirected read, including
    // writes nested in intervening control flow and writes by that last user.
    if (!storage.unchangedBetween(source, copy, last) ||
        !storage.preserves(last, source))
      continue;
    OpBuilder builder(copy);
    auto replacement =
        materializeCollectiveView(builder, copy->getLoc(), source, output);
    assert(succeeded(replacement) &&
           "complete copies preserve local storage capacity");
    destination.replaceAllUsesExcept(*replacement, copy);
    copy->erase();
    if (destination.use_empty()) destination.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}

bool forwardUniformScalarLoads(func::FuncOp function) {
  SmallVector<memref::LoadOp> loads;
  function.walk([&](memref::LoadOp load) { loads.push_back(load); });
  bool changed = false;
  for (auto load : loads) {
    auto initialization = uniformFillBefore(load.getMemref(), load);
    if (!initialization)
      continue;
    load.getResult().replaceAllUsesWith(initialization.getValue());
    load.erase();
    changed = true;
  }
  if (changed)
    eliminateUnreadLocalWrites(function);
  return changed;
}

bool batchPointwiseTasks(func::FuncOp function) {
  auto interface = intent::getPublicInterface(function);
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  auto argumentWithAccess = [&](Value value, unsigned access) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = intent::getPublicView(interface, argument.getArgNumber());
    return view && view.getAccess() == access;
  };
  auto constant = [](Value value) -> int64_t {
    auto index = value.getDefiningOp<arith::ConstantIndexOp>();
    return index ? index.value() : -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() ||
        !loop.getStep().getDefiningOp<dsa::TaskCountOp>() || !loop.getInitArgs().empty()) continue;
    Block *body = loop.getBody();
    BlockArgument induction = cast<BlockArgument>(loop.getInductionVar());
    ExecutionRelations execution(function);
    SmallVector<memref::AllocaOp> allocations;
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    int64_t capacity = 0;
    bool eligible = true;
    for (Operation &operation : *body) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(operation)) {
        auto type = allocation.getType();
        if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 ||
            !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
            (capacity && capacity != type.getDimSize(1)) ||
            llvm::any_of(allocation.getResult().getUsers(), [&](Operation *user) { return user->getBlock() != body; })) {
          eligible = false; break;
        }
        capacity = type.getDimSize(1);
        allocations.push_back(allocation);
      } else if (auto transfer = dyn_cast<dsa::LoadTileOp>(operation)) {
        if (load || !argumentWithAccess(transfer.getSource(), 0)) { eligible = false; break; }
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(operation)) {
        if (store || !argumentWithAccess(transfer.getDestination(), 1)) { eligible = false; break; }
        store = transfer;
      } else if (isa<dsa::FillOp, dsa::UnaryOp, dsa::BinaryOp, dsa::CastOp, dsa::CompareOp, dsa::SelectOp>(operation)) {
        if (auto unary = dyn_cast<dsa::UnaryOp>(operation); unary && unary.getScratch()) eligible = false;
        if (auto select = dyn_cast<dsa::SelectOp>(operation); select && select.getScratch()) eligible = false;
        for (Value operand : operation.getOperands()) {
          if (isa<MemRefType>(operand.getType())) {
            auto allocation = operand.getDefiningOp<memref::AllocaOp>();
            if (!allocation || allocation->getBlock() != body) eligible = false;
          } else if (execution.coordinateDependency(operand, induction) !=
                     CoordinateDependency::Independent) eligible = false;
        }
        if (!eligible) break;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(operation)) {
        auto argument = dyn_cast<BlockArgument>(stride.getSource());
        if (!argument || argument.getOwner() != &function.front()) { eligible = false; break; }
      } else if (!isa<scf::YieldOp>(operation) &&
                 (operation.getNumRegions() || !isMemoryEffectFree(&operation) ||
                  operation.getName().getDialectNamespace() != "arith")) {
        eligible = false; break;
      }
    }
    if (!eligible || !load || !store || allocations.empty()) continue;
    int64_t columns = constant(load.getColumns());
    if (columns <= 0 || constant(store.getColumns()) != columns ||
        constant(load.getRows()) != 1 || constant(store.getRows()) != 1 ||
        capacity <= columns || capacity % columns) continue;
    auto matchesStride = [&](Value value, Value resource, int64_t axis) {
      if (auto stride = value.getDefiningOp<dsa::StrideOp>())
        return stride.getSource() == resource && stride.getAxis() == uint64_t(axis);
      auto argument = dyn_cast<BlockArgument>(resource);
      if (!argument || argument.getOwner() != &function.front()) return false;
      auto view = intent::getPublicView(interface, argument.getArgNumber());
      if (!view || !view.getConstraints().getHasStrides()) return false;
      auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]);
      APInt bits;
      return fixed && matchPattern(value, m_ConstantInt(&bits)) && bits.getSExtValue() == fixed.getInt();
    };
    auto rowCoefficient = [&](Value view, Value offset, Value columnStride) -> Value {
      auto type = cast<MemRefType>(view.getType());
      if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(1) != columns ||
          type.getDimSize(0) != constant(loop.getUpperBound())) return {};
      if (!matchesStride(columnStride, view, 1)) return {};
      auto product = offset.getDefiningOp<arith::MulIOp>();
      if (!product) return {};
      Value coefficient = product.getLhs() == induction ? product.getRhs() :
          product.getRhs() == induction ? product.getLhs() : Value{};
      return coefficient && matchesStride(coefficient, view, 0) ? coefficient : Value{};
    };
    Value inputPitch = rowCoefficient(load.getSource(), load.getOffset(), load.getColumnStride());
    Value outputPitch = rowCoefficient(store.getDestination(), store.getOffset(), store.getColumnStride());
    auto owned = [&](Value value) {
      auto allocation = value.getDefiningOp<memref::AllocaOp>();
      return allocation && llvm::is_contained(allocations, allocation);
    };
    if (!inputPitch || !outputPitch || !owned(load.getOutput()) || !owned(store.getInput())) continue;
    // The runtime rejects overlapping writable view arguments. With read-only
    // input, write-only output and no carried/local scalar state, these rows
    // can share one supply/compute/store while retaining each task's row order.
    int64_t rows = capacity / columns;
    int64_t tasks = config.getTasks(), upper = constant(loop.getUpperBound());
    if (tasks <= 0 || rows > (std::numeric_limits<int64_t>::max() - upper) / tasks) continue;
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    Value factor = builder.create<arith::ConstantIndexOp>(loc, rows);
    Value oldStep = loop.getStep();
    loop.getStepMutable().assign(builder.create<arith::MulIOp>(loc, oldStep, factor));
    builder.setInsertionPointToStart(body);
    Value remaining = builder.create<arith::SubIOp>(loc, loop.getUpperBound(), induction);
    Value count = builder.create<arith::CeilDivSIOp>(loc, remaining, oldStep);
    count = builder.create<arith::MinSIOp>(loc, count, factor);
    for (auto allocation : allocations) {
      auto type = allocation.getType();
      allocation.getResult().setType(MemRefType::get({rows, columns}, type.getElementType(),
          MemRefLayoutAttrInterface{}, type.getMemorySpace()));
    }
    builder.setInsertionPoint(load);
    load.getRowStrideMutable().assign(builder.create<arith::MulIOp>(loc, inputPitch, oldStep));
    load.getRowsMutable().assign(count);
    builder.setInsertionPoint(store);
    store.getRowStrideMutable().assign(builder.create<arith::MulIOp>(loc, outputPitch, oldStep));
    store.getRowsMutable().assign(count);
    changed = true;
  }
  return changed;
}
} // namespace intent::dsa

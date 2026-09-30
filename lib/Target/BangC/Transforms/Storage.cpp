#include "PassDetail.h"

using namespace mlir;
namespace intent::bangc {
namespace {
StorageUsage visitStorage(func::FuncOp function,
    llvm::function_ref<void(memref::AllocaOp, int64_t, int64_t)> binding) {
  struct Live { uint64_t finish; int64_t offset, bytes, space; };
  SmallVector<Live> live;
  StorageUsage usage;
  for (const auto &range : dsa::analyzeStorageLifetimes(function)) {
    auto allocation = range.allocation;
    auto type = allocation.getType();
    int64_t space = type.getMemorySpaceAsInt();
    bool matrix = space == dsa::matrixSpace;
    int64_t payload = type.getNumElements() * llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
    // MTP372 WRAM addresses are per-bank offsets across sixteen LT banks.
    int64_t bytes = matrix ? llvm::alignTo(llvm::divideCeil(payload, int64_t(16)), int64_t(64))
                           : llvm::alignTo(payload, int64_t(128));
    llvm::erase_if(live, [&](const Live &value) { return value.finish < range.start; });
    llvm::sort(live, [](const Live &a, const Live &b) { return a.offset < b.offset; });
    int64_t offset = 0;
    for (const Live &value : live) {
      if (value.space != space) continue;
      if (offset + bytes <= value.offset) break;
      offset = std::max(offset, value.offset + value.bytes);
    }
    live.push_back({range.finish, offset, bytes, space});
    int64_t &peak = matrix ? usage.wram : space == dsa::sharedSpace ? usage.sram : usage.nram;
    peak = std::max(peak, (offset + bytes) * (matrix ? 16 : 1));
    binding(allocation, offset, bytes * (matrix ? 16 : 1));
  }
  return usage;
}
} // namespace

StorageUsage measureStorage(func::FuncOp function) {
  return visitStorage(function, [](memref::AllocaOp, int64_t, int64_t) {});
}

void measureStorage(func::FuncOp function, int64_t &nram, int64_t &wram) {
  auto usage = measureStorage(function);
  nram = usage.nram; wram = usage.wram;
}

StorageUsage bindStorage(func::FuncOp function) {
  Builder builder(function.getContext());
  auto usage = visitStorage(function, [&](memref::AllocaOp allocation, int64_t offset, int64_t bytes) {
    allocation->setAttr("bangc.offset", builder.getI64IntegerAttr(offset));
    allocation->setAttr("bangc.allocation_bytes", builder.getI64IntegerAttr(bytes));
  });
  function->setAttr("bangc.sram_bytes", builder.getI64IntegerAttr(usage.sram));
  return usage;
}

Value allocate(OpBuilder &b, Location loc, Type element, ArrayRef<int64_t> shape, int64_t space) {
  auto type = MemRefType::get(shape, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
  auto result = b.create<memref::AllocaOp>(loc, type);
  result.setAlignment(128);
  return result;
}

bool reuseConsumedBinaryInputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> operations;
  function.walk([&](dsa::BinaryOp operation) { operations.push_back(operation); });
  int64_t currentNram = 0, currentWram = 0;
  measureStorage(function, currentNram, currentWram);
  bool changed = false;
  for (auto operation : operations) {
    // These same-dtype native operations support exact output/input aliasing.
    // Division and operations with additional workspace retain separate storage.
    if (operation.getApproximate() || operation.getFlushToZero()) continue;
    if (!isa<MemRefType>(operation.getRhs().getType()) && !supportedScalarBinary(operation.getKind())) continue;
    switch (operation.getKind()) {
    case BinaryOperator::Add: case BinaryOperator::Subtract: case BinaryOperator::Multiply:
    case BinaryOperator::Maximum: case BinaryOperator::Minimum:
    case BinaryOperator::MaximumNum: case BinaryOperator::MinimumNum: break;
    default: continue;
    }
    Value input = operation.getLhs(), output = operation.getOutput();
    auto source = input.getDefiningOp<memref::AllocaOp>();
    auto destination = output.getDefiningOp<memref::AllocaOp>();
    if (!source || !destination || input == output || output == operation.getRhs() ||
        source->getBlock() != operation->getBlock() || destination->getBlock() != operation->getBlock() ||
        source.getType() != destination.getType() || !source.getType().getLayout().isIdentity() ||
        source.getType().getMemorySpaceAsInt() != dsa::nramSpace ||
        source.getAlignment().value_or(0) < destination.getAlignment().value_or(0)) continue;
    Type element = source.getType().getElementType();
    if (!element.isF16() && !element.isF32()) continue;
    if (auto loop = dyn_cast<scf::ForOp>(operation->getParentOp()))
      if (auto uniform = uniformFillBefore(input, operation);
          uniform && loop.isDefinedOutsideOfLoop(uniform.getValue())) continue;
    if (llvm::any_of(input.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !scope->isBeforeInBlock(operation) || user->getNumRegions() ||
              user->hasTrait<OpTrait::IsTerminator>() ||
              llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); });
        }) || llvm::any_of(output.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !operation->isBeforeInBlock(scope) || user->hasTrait<OpTrait::IsTerminator>();
        })) continue;
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : output.getUses()) uses.push_back(&use);
    for (OpOperand *use : uses) use->set(input);
    Operation *next = destination->getNextNode();
    destination->remove();
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > currentNram || wram > currentWram || nram > config.getLocalBytes() ||
        nram > 768 * 1024 || wram > 1024 * 1024) {
      OpBuilder restore(next);
      restore.insert(destination.getOperation());
      for (OpOperand *use : uses) use->set(output);
    } else {
      destination->destroy();
      currentNram = nram;
      currentWram = wram;
      changed = true;
    }
  }
  return changed;
}

bool reuseConsumedExp2Inputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::UnaryOp> operations;
  function.walk([&](dsa::UnaryOp op) {
    if (op.getKind() == UnaryOperator::Exp2 && op.getApproximate() && op.getFlushToZero()) operations.push_back(op);
  });
  bool changed = false;
  for (auto operation : operations) {
    Value input = operation.getInput(), output = operation.getOutput();
    auto source = input.getDefiningOp<memref::AllocaOp>(), destination = output.getDefiningOp<memref::AllocaOp>();
    if (!source || !destination || input == output || source.getType() != destination.getType() ||
        source->getBlock() != operation->getBlock() || destination->getBlock() != operation->getBlock() ||
        !source.getType().getElementType().isF32() || !source.getType().getLayout().isIdentity() ||
        source.getType().getMemorySpaceAsInt() != dsa::nramSpace ||
        source.getAlignment().value_or(0) < destination.getAlignment().value_or(0) ||
        (operation.getScratch() && !cast<MemRefType>(operation.getScratch().getType()).getElementType().isInteger(32))) continue;
    if (llvm::any_of(input.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !scope->isBeforeInBlock(operation) || user->getNumRegions() ||
              user->hasTrait<OpTrait::IsTerminator>() ||
              llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); });
        }) || llvm::any_of(output.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !operation->isBeforeInBlock(scope) || user->hasTrait<OpTrait::IsTerminator>();
        })) continue;
    // compute_30 exp2 is one pow2.nram instruction. Its FTZ implementation
    // already supports exact input/output aliasing; the old input is dead here.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : output.getUses()) uses.push_back(&use);
    for (OpOperand *use : uses) use->set(input);
    Operation *next = destination->getNextNode();
    destination->remove();
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      OpBuilder restore(next); restore.insert(destination.getOperation());
      for (OpOperand *use : uses) use->set(output);
    } else {
      destination->destroy(); changed = true;
    }
  }
  return changed;
}

void hoistInvariantFills(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<memref::AllocaOp> allocations;
  function.walk([&](memref::AllocaOp allocation) { allocations.push_back(allocation); });
  DominanceInfo dominance(function);
  int64_t currentNram = 0, currentWram = 0;
  measureStorage(function, currentNram, currentWram);
  for (auto allocation : allocations) {
    auto loop = dyn_cast<scf::ForOp>(allocation->getParentOp());
    if (!loop || allocation->getBlock() != loop.getBody()) continue;
    Value buffer = allocation.getResult();
    dsa::FillOp initialization;
    bool immutable = true;
    for (Operation *user : buffer.getUsers()) {
      if (auto fill = dyn_cast<dsa::FillOp>(user)) {
        if (initialization || fill->getBlock() != loop.getBody()) { immutable = false; break; }
        initialization = fill;
        continue;
      }
      if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) {
        immutable = false; break;
      }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == buffer)) immutable = false;
      } else if (!isMemoryEffectFree(user)) immutable = false;
      if (!immutable) break;
    }
    if (!immutable || !initialization || buffer.hasOneUse() || !loop.isDefinedOutsideOfLoop(initialization.getValue()) ||
        llvm::any_of(buffer.getUsers(), [&](Operation *user) {
          return user != initialization && !dominance.dominates(initialization, user);
        })) continue;
    Operation *allocationNext = allocation->getNextNode(), *initializationNext = initialization->getNextNode();
    allocation->moveBefore(loop);
    initialization->moveBefore(loop);
    // Sharing the immutable tile across iterations extends its live range.
    // Retain a move only when it does not increase the program's peak storage.
    int64_t nram = 0, wram = 0;
    measureStorage(function, nram, wram);
    if (nram > currentNram || wram > currentWram ||
        nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      initialization->moveBefore(initializationNext);
      allocation->moveBefore(allocationNext);
    } else { currentNram = nram; currentWram = wram; }
  }
}

LogicalResult bindProgramStorage(ModuleOp module) {
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  StorageUsage usage = bindStorage(function);
  int64_t nram = usage.nram, wram = usage.wram, shared = usage.sram;
  if (shared > 3968 * 1024) return function.emitError("selected shared supply exceeds the MLU370 SRAM budget");
  int64_t internalNram = 0;
  function.walk([&](Operation *op) {
    if (auto bytes = op->getAttrOfType<IntegerAttr>("bangc.internal_nram_bytes"))
      internalNram = std::max(internalNram, bytes.getInt());
  });
  if (nram + internalNram > config.getLocalBytes() || nram + internalNram > 768 * 1024 || wram > 1024 * 1024)
    return function.emitError("selected local implementation exceeds the MLU370 per-task storage budget")
        << "; NRAM=" << nram << ", WRAM=" << wram;
  Builder builder(module.getContext());
  function->setAttr("bangc.nram_bytes", builder.getI64IntegerAttr(nram));
  function->setAttr("bangc.wram_bytes", builder.getI64IntegerAttr(wram));
  function->setAttr("bangc.wram_align", builder.getI64IntegerAttr(16));
  return success();
}
} // namespace intent::bangc

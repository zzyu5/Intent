#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

void forwardDestinations(func::FuncOp function) {
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    auto allocation = copy.getSource().getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != copy->getBlock()) continue;
    Value target = copy.getTarget();
    if (allocation.getType().getRank() != copy.getTarget().getType().getRank()) continue;
    auto targetOp = target.getDefiningOp();
    DominanceInfo dominance(function);
    if (targetOp && !dominance.dominates(targetOp, allocation)) {
      if (!isMemoryEffectFree(targetOp) ||
          llvm::any_of(targetOp->getOperands(), [&](Value value) {
            return !dominance.dominates(value, allocation);
          })) continue;
      targetOp->moveBefore(allocation);
    }
    bool legal = true;
    SmallVector<memref::DeallocOp> deallocations;
    for (Operation *user : allocation->getUsers()) {
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
        deallocations.push_back(dealloc);
        continue;
      }
      Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor || (ancestor != copy && !ancestor->isBeforeInBlock(copy))) legal = false;
    }
    if (!legal) continue;
    for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
    allocation.getResult().replaceAllUsesExcept(target, copy);
    copy.erase();
    allocation.erase();
  }
}

bool canReplay(Value value, Operation *root, llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *operation = value.getDefiningOp();
  if (!operation || !root->isAncestor(operation)) return true;
  if (!seen.insert(operation).second) return true;
  if (operation->getNumRegions() || operation->getNumResults() != 1 ||
      (!isMemoryEffectFree(operation) && !isa<memref::LoadOp>(operation))) return false;
  return llvm::all_of(operation->getOperands(), [&](Value input) { return canReplay(input, root, seen); });
}

bool stableRead(memref::LoadOp load, Operation *producer, func::FuncOp function) {
  Value base = load.getMemref();
  while (true) {
    if (auto view = base.getDefiningOp<memref::SubViewOp>()) base = view.getSource();
    else if (auto cast = base.getDefiningOp<memref::CastOp>()) base = cast.getSource();
    else break;
  }
  if (auto argument = dyn_cast<BlockArgument>(base)) {
    if (argument.getOwner() != &function.front()) return false;
    auto abi = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
    auto field = dyn_cast<ViewArgumentAttr>(abi.getArguments()[argument.getArgNumber()]);
    return field && field.getAccess() == 0;
  }
  auto allocation = base.getDefiningOp<memref::AllocOp>();
  if (!allocation) return false;
  for (Operation *user : base.getUsers()) {
    if (isa<memref::LoadOp, memref::DimOp, memref::DeallocOp>(user)) continue;
    if (!isa<memref::StoreOp>(user)) return false;
    Operation *write = producer->getBlock()->findAncestorOpInBlock(*user);
    if (!write || write == producer || !write->isBeforeInBlock(producer)) return false;
  }
  return true;
}

Value replay(Value value, Operation *root, OpBuilder &builder, IRMapping &mapping) {
  if (mapping.contains(value)) return mapping.lookup(value);
  Operation *operation = value.getDefiningOp();
  if (!operation || !root->isAncestor(operation)) return value;
  for (Value input : operation->getOperands())
    mapping.map(input, replay(input, root, builder, mapping));
  builder.clone(*operation, mapping);
  return mapping.lookup(value);
}

bool fuse(memref::AllocOp allocation) {
  memref::StoreOp store;
  SmallVector<memref::LoadOp> loads;
  SmallVector<memref::DeallocOp> deallocations;
  for (Operation *user : allocation->getUsers()) {
    if (auto write = dyn_cast<memref::StoreOp>(user)) {
      if (store) return false;
      store = write;
    } else if (auto read = dyn_cast<memref::LoadOp>(user)) {
      loads.push_back(read);
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocations.push_back(dealloc);
    } else return false;
  }
  if (!store || loads.empty()) return false;
  llvm::SmallPtrSet<Operation *, 4> consumers;
  for (memref::LoadOp load : loads) {
    Operation *stage = allocation->getBlock()->findAncestorOpInBlock(*load);
    if (!stage) return false;
    consumers.insert(stage);
  }
  if (consumers.size() != 1) return false;
  SmallVector<scf::ForOp> loops;
  Operation *root = store;
  while (auto parent = dyn_cast<scf::ForOp>(root->getParentOp())) {
    if (parent.getNumResults() || !matchPattern(parent.getLowerBound(), m_Zero()) ||
        !matchPattern(parent.getStep(), m_One())) return false;
    loops.push_back(parent);
    root = parent;
  }
  std::reverse(loops.begin(), loops.end());
  if (root->getBlock() != allocation->getBlock() ||
      loops.size() != store.getIndices().size()) return false;
  unsigned dynamicAxis = 0;
  for (auto [axis, loop] : llvm::enumerate(loops)) {
    if (store.getIndices()[axis] != loop.getInductionVar()) return false;
    if (allocation.getType().isDynamicDim(axis)) {
      if (loop.getUpperBound() != allocation.getDynamicSizes()[dynamicAxis++]) return false;
    } else {
      auto extent = getConstantIntValue(loop.getUpperBound());
      if (!extent || *extent != allocation.getType().getDimSize(axis)) return false;
    }
  }
  bool otherEffect = false;
  root->walk([&](Operation *operation) {
    if (operation == store || isa<scf::ForOp, scf::YieldOp, memref::LoadOp>(operation)) return;
    if (!isMemoryEffectFree(operation)) otherEffect = true;
  });
  if (otherEffect) return false;
  llvm::SmallPtrSet<Operation *, 16> seen;
  if (!canReplay(store.getValue(), root, seen)) return false;
  auto function = allocation->getParentOfType<func::FuncOp>();
  for (Operation *operation : seen)
    if (auto read = dyn_cast<memref::LoadOp>(operation); read && !stableRead(read, root, function)) return false;
  DominanceInfo dominance(function);
  for (memref::LoadOp load : loads)
    if (root->isAncestor(load) || !dominance.dominates(root, load)) return false;
  for (memref::LoadOp load : loads) {
    IRMapping mapping;
    for (auto [loop, coordinate] : llvm::zip(loops, load.getIndices()))
      mapping.map(loop.getInductionVar(), coordinate);
    OpBuilder builder(load);
    load.getResult().replaceAllUsesWith(replay(store.getValue(), root, builder, mapping));
    load.erase();
  }
  root->erase();
  for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
  allocation.erase();
  return true;
}

}

LogicalResult fuseIntermediateBuffers(func::FuncOp function) {
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
  if (!interface || !interface.getDisjointOutputs())
    return function.emitError("CPU buffer fusion requires established external alias legality");
  forwardDestinations(function);
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    OpBuilder builder(copy);
    Location loc = copy.getLoc();
    SmallVector<Value> indices;
    std::function<void(int64_t)> materialize = [&](int64_t axis) {
      if (axis == copy.getSource().getType().getRank()) {
        Value value = builder.create<memref::LoadOp>(loc, copy.getSource(), indices);
        builder.create<memref::StoreOp>(loc, value, copy.getTarget(), indices);
        return;
      }
      Value extent = builder.create<memref::DimOp>(loc, copy.getSource(), axis);
      loop(builder, loc, index(builder, loc, 0), extent, 1, [&](Value coordinate) {
        indices.push_back(coordinate);
        materialize(axis + 1);
        indices.pop_back();
      });
    };
    materialize(0);
    copy.erase();
  }
  bool changed;
  do {
    changed = false;
    SmallVector<memref::AllocOp> allocations;
    function.walk([&](memref::AllocOp op) { allocations.push_back(op); });
    for (memref::AllocOp allocation : llvm::reverse(allocations))
      changed |= fuse(allocation);
  } while (changed);
  return success();
}

void forwardCPUOutputs(func::FuncOp function) { forwardDestinations(function); }

}

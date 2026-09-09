#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool sameBound(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto left = getConstantIntValue(lhs), right = getConstantIntValue(rhs);
  return left && right && *left == *right;
}

bool readOnlyReduction(scf::ForOp loop) {
  auto order = loop->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (!order || !order.getAdjacentReassociation() || !loop.getNumResults()) return false;
  for (Operation &operation : loop.getBody()->without_terminator())
    if (operation.getNumRegions() ||
        (!isMemoryEffectFree(&operation) && !isa<memref::LoadOp>(operation))) return false;
  return true;
}

memref::StoreOp pointwiseStore(scf::ForOp loop) {
  if (loop.getNumResults()) return {};
  memref::StoreOp store;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions()) return {};
    if (auto write = dyn_cast<memref::StoreOp>(operation)) {
      if (store) return {};
      store = write;
    } else if (!isMemoryEffectFree(&operation) && !isa<memref::LoadOp>(operation)) return {};
  }
  if (!store || store.getIndices().size() != 1 ||
      store.getIndices()[0] != loop.getInductionVar()) return {};
  auto function = loop->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(function);
  Value root = analysis.storageRoot(store.getMemref());
  auto external = analysis.externalView(root);
  if (!root.getDefiningOp<memref::AllocOp>() &&
      (!external || external.getAccess() != 1)) return {};
  for (auto access : analysis.accesses(function))
    if (analysis.storageRoot(access.memory) == root &&
        (access.memory != store.getMemref() ||
         (access.read && loop->isAncestor(access.operation)))) return {};
  return store;
}

bool fuse(scf::ForOp first, scf::ForOp second, DominanceInfo &dominance) {
  if (!readOnlyReduction(second) ||
      !sameBound(first.getLowerBound(), second.getLowerBound()) ||
      !sameBound(first.getUpperBound(), second.getUpperBound()) ||
      !sameBound(first.getStep(), second.getStep())) return false;
  auto store = pointwiseStore(first);
  if (!readOnlyReduction(first) && !store) return false;
  if (store) {
    bool consumes = false;
    for (Operation &operation : second.getBody()->without_terminator()) {
      auto load = dyn_cast<memref::LoadOp>(operation);
      if (!load || load.getMemref() != store.getMemref()) continue;
      if (load.getIndices().size() != 1 || load.getIndices()[0] != second.getInductionVar())
        return false;
      consumes = true;
    }
    if (!consumes) return false;
  }
  for (Value operand : second.getInitArgs())
    if (!dominance.dominates(operand, first)) return false;
  for (Operation &operation : second.getBody()->without_terminator())
    for (Value operand : operation.getOperands())
      if (!second.getRegion().isAncestor(operand.getParentRegion()) &&
          !dominance.dominates(operand, first)) return false;

  OpBuilder b(first);
  SmallVector<Value> initial(first.getInitArgs());
  llvm::append_range(initial, second.getInitArgs());
  auto joined = b.create<scf::ForOp>(first.getLoc(), first.getLowerBound(),
      first.getUpperBound(), first.getStep(), initial);
  joined->setAttr("intent_cpu.reduction_order", second->getAttr("intent_cpu.reduction_order"));
  b.setInsertionPointToStart(joined.getBody());
  SmallVector<Value> yields;
  auto append = [&](scf::ForOp source, unsigned begin) {
    IRMapping mapping;
    mapping.map(source.getInductionVar(), joined.getInductionVar());
    for (auto [oldValue, newValue] : llvm::zip(source.getRegionIterArgs(),
             joined.getRegionIterArgs().drop_front(begin))) mapping.map(oldValue, newValue);
    for (Operation &operation : source.getBody()->without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(operation)) {
        Value memory = mapping.lookupOrDefault(load.getMemref());
        SmallVector<Value> indices;
        for (Value index : load.getIndices()) indices.push_back(mapping.lookupOrDefault(index));
        Value reused;
        for (Operation &previous : llvm::reverse(*joined.getBody())) {
          auto write = dyn_cast<memref::StoreOp>(previous);
          if (write && write.getMemref() == memory && llvm::equal(write.getIndices(), indices)) {
            reused = write.getValue();
            break;
          }
          auto read = dyn_cast<memref::LoadOp>(previous);
          if (read && read.getMemref() == memory && llvm::equal(read.getIndices(), indices)) {
            reused = read.getResult();
            break;
          }
        }
        if (reused) {
          mapping.map(load.getResult(), reused);
          continue;
        }
      }
      b.clone(operation, mapping);
    }
    for (Value value : source.getBody()->getTerminator()->getOperands())
      yields.push_back(mapping.lookupOrDefault(value));
  };
  append(first, 0);
  append(second, first.getNumResults());
  b.create<scf::YieldOp>(first.getLoc(), yields);
  first.replaceAllUsesWith(joined.getResults().take_front(first.getNumResults()));
  second.replaceAllUsesWith(joined.getResults().drop_front(first.getNumResults()));
  first.erase();
  second.erase();
  return true;
}

}

LogicalResult fuseReductionTraversals(func::FuncOp function) {
  bool changed;
  do {
    changed = false;
    SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) {
      if (readOnlyReduction(loop) || pointwiseStore(loop)) loops.push_back(loop);
    });
    DominanceInfo dominance(function);
    for (scf::ForOp first : loops) {
      for (Operation *next = first->getNextNode(); next; next = next->getNextNode()) {
        if (auto second = dyn_cast<scf::ForOp>(next)) {
          changed = fuse(first, second, dominance);
          break;
        }
        if (!isMemoryEffectFree(next)) break;
      }
      if (changed) break;
    }
  } while (changed);
  return success();
}

}

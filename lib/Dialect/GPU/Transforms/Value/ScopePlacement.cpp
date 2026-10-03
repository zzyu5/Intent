#include "ScopePlacement.h"

#include "Intent/Dialect/GPU/Analysis/MemoryEffects.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;

namespace intent::gpu::placement {
namespace {

bool hasUnorderedEffects(Operation *operation) {
  return !operation->walk([](Operation *nested) {
    auto access = dyn_cast<AccessOpInterface>(nested);
    if (!access || access.getAccessKind() == AccessKind::Store ||
        hasOnlyReadEffects(nested))
      return WalkResult::advance();
    return WalkResult::interrupt();
  }).wasInterrupted();
}

} // namespace

bool independentMemoryEffects(Operation *first, Operation *second,
                              ResourceAliasAnalysis &aliases) {
  if (!hasUnorderedEffects(first) || !hasUnorderedEffects(second))
    return false;
  auto firstEffects = getEffectsRecursively(first);
  auto secondEffects = getEffectsRecursively(second);
  if (!firstEffects || !secondEffects)
    return false;
  auto understood = [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Read, MemoryEffects::Write,
               MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect());
  };
  if (!llvm::all_of(*firstEffects, understood) ||
      !llvm::all_of(*secondEffects, understood))
    return false;
  for (const auto &lhs : *firstEffects) {
    for (const auto &rhs : *secondEffects) {
      if (lhs.getResource() != rhs.getResource() ||
          (isa<MemoryEffects::Read>(lhs.getEffect()) &&
           isa<MemoryEffects::Read>(rhs.getEffect())))
        continue;
      if (!lhs.getValue() || !rhs.getValue() ||
          !aliases.alias(lhs.getValue(), rhs.getValue()).isNo())
        return false;
    }
  }
  return true;
}

bool isMovableValueOperation(Operation *operation) {
  return operation->getNumRegions() == 0 && operation->getNumResults() != 0 &&
         isMemoryEffectFree(operation) && isSpeculatable(operation);
}

bool canMoveBefore(Operation *operation, Operation *before) {
  if (!operation || !before || operation == before ||
      operation->getBlock() != before->getBlock())
    return false;
  if (isMovableValueOperation(operation))
    return true;
  auto load = dyn_cast<LoadOp>(operation);
  return load && canReplayReadAt(load, before);
}

bool moveInputsBefore(Operation *first, Operation *second,
                      func::FuncOp kernel) {
  DominanceInfo dominance(kernel);
  SmallVector<Operation *> hoist;
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> available = [&](Value value) {
    if (dominance.properlyDominates(value, first))
      return true;
    if (!visited.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getBlock() != first->getBlock() ||
        !first->isBeforeInBlock(producer) ||
        !producer->isBeforeInBlock(second) ||
        !canMoveBefore(producer, first) ||
        !llvm::all_of(producer->getOperands(), available))
      return false;
    if (!llvm::is_contained(hoist, producer))
      hoist.push_back(producer);
    return true;
  };
  llvm::SetVector<Value> captures;
  for (Region &region : second->getRegions())
    getUsedValuesDefinedAbove(region, region, captures);
  if (!llvm::all_of(second->getOperands(), available) ||
      !llvm::all_of(captures, available))
    return false;
  for (Operation *operation : hoist)
    operation->moveBefore(first);
  return true;
}

} // namespace intent::gpu::placement

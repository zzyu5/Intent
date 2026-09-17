#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

bool sameBound(Value lhs, Value rhs, ArrayAttr tuples) {
  if (samePhysicalScalarExpression(lhs, rhs))
    return true;
  PhysicalExprAttr left = queryLaunchExpression(lhs);
  PhysicalExprAttr right = queryLaunchExpression(rhs);
  if (left && left == right)
    return true;
  auto first = lhs.getDefiningOp<ParameterOp>();
  auto second = rhs.getDefiningOp<ParameterOp>();
  if (!first || !second || !tuples || tuples.empty())
    return false;
  return llvm::all_of(tuples, [&](Attribute attribute) {
    auto tuple = cast<DictionaryAttr>(attribute);
    auto a = tuple.getAs<IntegerAttr>(first.getParameter().getName().getValue());
    auto b = tuple.getAs<IntegerAttr>(second.getParameter().getName().getValue());
    return a && b && a == b;
  });
}

bool collectViewReads(scf::ForOp loop, bool allowScratchWrites,
                      llvm::DenseSet<Value> &reads) {
  return !loop.walk([&](Operation *operation) {
    if (operation == loop)
      return WalkResult::advance();
    if (auto load = dyn_cast<LoadOp>(operation)) {
      auto view = dyn_cast<ViewType>(load.getResource().getType());
      if (!view || view.getAccess() != 0)
        return WalkResult::interrupt();
      reads.insert(load.getResource());
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<StoreOp>(operation))
      return allowScratchWrites && isa<BufferType>(store.getResource().getType())
                 ? WalkResult::advance() : WalkResult::interrupt();
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(operation))
      return WalkResult::interrupt();
    return isMemoryEffectFree(operation) ? WalkResult::advance()
                                        : WalkResult::interrupt();
  }).wasInterrupted();
}

bool tryFuse(scf::ForOp first, scf::ForOp second, func::FuncOp kernel,
             ArrayAttr tuples) {
  if (first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) ||
      second.getNumResults() != 0 ||
      first->hasAttr(executionGroupAttr) || second->hasAttr(executionGroupAttr) ||
      !sameBound(first.getLowerBound(), second.getLowerBound(), tuples) ||
      !sameBound(first.getUpperBound(), second.getUpperBound(), tuples) ||
      !sameBound(first.getStep(), second.getStep(), tuples))
    return false;
  for (NamedAttribute attribute : first->getDiscardableAttrs())
    if (attribute.getName() != originAttr &&
        attribute.getName() != reductionSourcesAttr &&
        attribute.getName() != independentIterationAttr)
      return false;
  for (NamedAttribute attribute : second->getDiscardableAttrs())
    if (attribute.getName() != originAttr &&
        attribute.getName() != independentIterationAttr)
      return false;
  llvm::DenseSet<Value> firstReads, secondReads;
  if (!collectViewReads(first, false, firstReads) ||
      !collectViewReads(second, true, secondReads) ||
      !llvm::any_of(firstReads, [&](Value value) {
        return secondReads.contains(value);
      }))
    return false;
  for (Operation *operation = first->getNextNode(); operation != second;
       operation = operation->getNextNode())
    if (!isMemoryEffectFree(operation))
      return false;

  DominanceInfo dominance(kernel);
  SmallVector<Operation *> hoist;
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> availableBeforeFirst = [&](Value value) {
    if (dominance.properlyDominates(value, first))
      return true;
    if (!visited.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getBlock() != first->getBlock() ||
        !first->isBeforeInBlock(producer) ||
        !producer->isBeforeInBlock(second) || producer->getNumRegions() ||
        !isMemoryEffectFree(producer) ||
        !llvm::all_of(producer->getOperands(), availableBeforeFirst))
      return false;
    if (!llvm::is_contained(hoist, producer))
      hoist.push_back(producer);
    return true;
  };
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(second.getRegion(), second.getRegion(), captures);
  if (!llvm::all_of(second->getOperands(), availableBeforeFirst) ||
      !llvm::all_of(captures, availableBeforeFirst))
    return false;
  for (Operation *operation : hoist)
    operation->moveBefore(first);

  // The complete candidate set proves equal granularity even when the two
  // traversals retain distinct parameter roles in their fragment schemas.
  second.setLowerBound(first.getLowerBound());
  second.setUpperBound(first.getUpperBound());
  second.setStep(first.getStep());
  auto attributes = first->getAttrDictionary();
  bool independent = first->hasAttr(independentIterationAttr) &&
                     second->hasAttr(independentIterationAttr);
  IRRewriter rewriter(kernel.getContext());
  second->moveBefore(first);
  scf::ForOp fused = mlir::fuseIndependentSiblingForLoops(first, second, rewriter);
  fused->setAttrs(attributes);
  if (!independent || !fused.getInitArgs().empty())
    fused->removeAttr(independentIterationAttr);
  return true;
}

} // namespace

LogicalResult fuseIndependentTraversals(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  auto tuples = kernel->getAttrOfType<ArrayAttr>(sharedConfigTuplesAttr);
  bool changed;
  do {
    changed = false;
    SmallVector<scf::ForOp> loops;
    kernel.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp first : loops) {
      for (scf::ForOp second : loops)
        if (first != second && tryFuse(first, second, kernel, tuples)) {
          changed = true;
          break;
        }
      if (changed)
        break;
    }
  } while (changed);
  return eliminateCommonValues(module);
}

} // namespace intent::gpu

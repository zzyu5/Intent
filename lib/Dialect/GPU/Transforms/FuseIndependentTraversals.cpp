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

bool hoistInputsBefore(Operation *first, Operation *second,
                       func::FuncOp kernel) {
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
  for (Region &region : second->getRegions())
    getUsedValuesDefinedAbove(region, region, captures);
  if (!llvm::all_of(second->getOperands(), availableBeforeFirst) ||
      !llvm::all_of(captures, availableBeforeFirst))
    return false;
  for (Operation *operation : hoist)
    operation->moveBefore(first);
  return true;
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

  if (!hoistInputsBefore(first, second, kernel))
    return false;

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

ReduceOp tryFuse(ReduceOp first, ReduceOp second, func::FuncOp kernel) {
  if (first->getBlock() != second->getBlock() ||
      !first->isBeforeInBlock(second) || first.getAxes() != second.getAxes())
    return {};
  auto shape = dyn_cast<FragmentType>(first.getInputs().front().getType());
  if (!shape)
    return {};
  for (ReduceOp reduce : {first, second}) {
    for (NamedAttribute attribute : reduce->getDiscardableAttrs())
      if (attribute.getName() != originAttr)
        return {};
    for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
      auto type = dyn_cast<FragmentType>(source.getType());
      if (!type || type.getShape() != shape.getShape() ||
          type.getAxisMaps() != shape.getAxisMaps() ||
          type.getValidity() != shape.getValidity() ||
          type.getOwner() != shape.getOwner())
        return {};
    }
  }
  if (!hoistInputsBefore(first, second, kernel))
    return {};

  unsigned count = first.getSourceCount() + second.getSourceCount();
  SmallVector<Value> sources, identities, captures;
  SmallVector<Type> results;
  for (ReduceOp reduce : {first, second}) {
    llvm::append_range(sources,
                       reduce.getInputs().take_front(reduce.getSourceCount()));
    llvm::append_range(identities, reduce.getInputs().slice(
                                      reduce.getSourceCount(),
                                      reduce.getIdentityCount()));
    llvm::append_range(captures, reduce.getInputs().take_back(
                                    reduce.getCaptureCount()));
    llvm::append_range(results, reduce.getResultTypes());
  }
  OpBuilder builder(first);
  Location location = builder.getFusedLoc({first.getLoc(), second.getLoc()});
  OperationState state(location, ReduceOp::getOperationName());
  state.addOperands(sources);
  state.addOperands(identities);
  state.addOperands(captures);
  state.addTypes(results);
  state.addAttribute("axes", first.getAxesAttr());
  state.addAttribute("source_count", builder.getI64IntegerAttr(count));
  state.addAttribute("identity_count", builder.getI64IntegerAttr(count));
  state.addAttribute("capture_count",
                     builder.getI64IntegerAttr(captures.size()));
  if (Attribute origin = first->getAttr(originAttr))
    state.addAttribute(originAttr, origin);
  state.addRegion();
  auto fused = cast<ReduceOp>(builder.create(state));
  SmallVector<Type> arguments(results);
  llvm::append_range(arguments, results);
  for (Value capture : captures)
    arguments.push_back(capture.getType());
  Block *body = builder.createBlock(&fused.getCombine(), {}, arguments,
                                    SmallVector<Location>(arguments.size(),
                                                          location));
  SmallVector<Value> yields;
  unsigned componentOffset = 0, captureOffset = 0;
  for (ReduceOp reduce : {first, second}) {
    Block &combine = reduce.getCombine().front();
    unsigned size = reduce.getSourceCount();
    IRMapping mapping;
    for (unsigned index = 0; index < size; ++index) {
      mapping.map(combine.getArgument(index),
                  body->getArgument(componentOffset + index));
      mapping.map(combine.getArgument(size + index),
                  body->getArgument(count + componentOffset + index));
    }
    for (unsigned index = 0; index < reduce.getCaptureCount(); ++index)
      mapping.map(combine.getArgument(2 * size + index),
                  body->getArgument(2 * count + captureOffset + index));
    for (Operation &operation : combine.without_terminator())
      builder.clone(operation, mapping);
    for (Value value : combine.getTerminator()->getOperands())
      yields.push_back(mapping.lookupOrDefault(value));
    componentOffset += size;
    captureOffset += reduce.getCaptureCount();
  }
  builder.create<YieldOp>(location, yields);
  first->replaceAllUsesWith(fused.getResults().take_front(first.getNumResults()));
  second->replaceAllUsesWith(fused.getResults().take_back(second.getNumResults()));
  first.erase();
  second.erase();
  return fused;
}

} // namespace

LogicalResult fuseIndependentReductions(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  if (!kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr)
           .getNativeTupleReductions())
    return success();
  kernel.walk<WalkOrder::PostOrder>([&](Block *block) {
    SmallVector<ReduceOp> reductions(block->getOps<ReduceOp>());
    for (unsigned first = 0; first < reductions.size(); ++first) {
      if (!reductions[first])
        continue;
      for (unsigned second = first + 1; second < reductions.size(); ++second) {
        if (!reductions[second])
          continue;
        if (ReduceOp fused =
                tryFuse(reductions[first], reductions[second], kernel)) {
          reductions[first] = fused;
          reductions[second] = {};
        }
      }
    }
  });
  return success();
}

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

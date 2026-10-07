#include "Intent/Dialect/GPU/Analysis/IterationDependencies.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "ScalarExpressions.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

bool isOutsideIterationRange(Value coordinate, scf::ForOp loop) {
  if (!coordinate.getType().isIndex() ||
      !loop.isDefinedOutsideOfLoop(coordinate))
    return false;
  IndexRelations relations;
  if (relations.same(coordinate, loop.getUpperBound())) return true;
  auto index = relations.constant(coordinate);
  auto lower = relations.constant(loop.getLowerBound());
  auto upper = relations.constant(loop.getUpperBound());
  if (index && ((lower && *index < *lower) || (upper && *index >= *upper)))
    return true;
  auto successor = loop.getLowerBound().getDefiningOp<BinaryOp>();
  if (!successor || successor.getOperatorKind() != BinaryOperator::Add)
    return false;
  auto induction = dyn_cast<BlockArgument>(coordinate);
  auto owner = induction
                   ? dyn_cast<scf::ForOp>(induction.getOwner()->getParentOp())
                   : scf::ForOp();
  if (!owner || coordinate != owner.getInductionVar() ||
      relations.constant(owner.getStep()) != 1 ||
      !owner->isProperAncestor(loop))
    return false;
  // An active unit-step induction value is below its signed upper bound, so
  // its successor is representable even at the last iteration.
  for (auto [base, offset] :
       {std::pair{successor.getLhs(), successor.getRhs()},
        std::pair{successor.getRhs(), successor.getLhs()}})
    if (base == coordinate && relations.constant(offset) == 1) return true;
  return false;
}

bool isUnitIterationCoordinate(Value coordinate, scf::ForOp loop) {
  llvm::DenseMap<Value, IntegerDifference> differences;
  llvm::DenseSet<Value> active;
  std::function<IntegerDifference(Value)> difference =
      [&](Value value) -> IntegerDifference {
    if (value == loop.getInductionVar()) return 1;
    if (loop.isDefinedOutsideOfLoop(value)) return 0;
    if (auto known = differences.find(value); known != differences.end())
      return known->second;
    if (!value.getType().isIndex() || !active.insert(value).second)
      return std::nullopt;
    Operation *producer = value.getDefiningOp();
    IntegerDifference result;
    if (producer && !producer->getNumRegions() && isMemoryEffectFree(producer)) {
      UniformExpression expression = describeUniformValue(value);
      if (expression.kind == UniformKind::Cast) {
        // A signed i32 IV may be widened to an address without changing the
        // body arithmetic. The shared modular difference query is i64-only.
        if (expression.operands.size() == 1 &&
            isValuePreservingIntegerCast(expression.operands.front(), value.getType()))
          result = difference(expression.operands.front());
      } else {
        bool preservesInteger = true;
        if (expression.kind == UniformKind::Add ||
            expression.kind == UniformKind::Subtract ||
            expression.kind == UniformKind::Multiply)
          preservesInteger = integerOperationDoesNotWrap(value);
        if (preservesInteger)
          result = foldIntegerDifference(expression, difference,
              [](Value operand) { return IndexRelations().constant(operand); }, 64);
      }
    }
    active.erase(value);
    differences[value] = result;
    return result;
  };
  return coordinate.getType().isIndex() &&
         difference(coordinate) == IntegerDifference(1);
}

// These operations only replicate or rearrange the same coordinate values.
// Stripping them here proves set membership, not an axis/layout equivalence.
Value coordinateValues(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (!isa<SplatOp, BroadcastOp, ReshapeOp, TransposeOp>(producer)) break;
    value = producer->getOperand(0);
  }
  return value;
}

bool ownsCoordinate(Value value, ExecutionGroupOp group, unsigned ordinal,
                    func::FuncOp kernel) {
  IndexRelations relations;
  value = coordinateValues(value);
  auto range = value.getDefiningOp<MakeRangeOp>();
  Value start = range ? range.getStart() : value;
  if (auto workset = start.getDefiningOp<WorksetCoordinateOp>()) {
    if (relations.constant(workset.getStep()) != 1) return false;
    start = workset.getCoordinate();
  }
  if (!range) return relations.same(start, group.getCoordinates()[ordinal]);
  if (relations.constant(range.getStep()) != 1 ||
      !relations.alignedUnitWindow(range.getStart(), range.getExtent()))
    return false;
  auto product = detail::stripScalarIdentity(start).getDefiningOp<BinaryOp>();
  if (!product || product.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate = group.getCoordinates()[ordinal];
  if (!((relations.same(product.getLhs(), coordinate) &&
         relations.same(product.getRhs(), range.getExtent())) ||
        (relations.same(product.getRhs(), coordinate) &&
         relations.same(product.getLhs(), range.getExtent()))))
    return false;
  auto launch = cast<PhysicalExprAttr>(group.getLaunchExtents()[ordinal]);
  if (launch.getKind() != PhysicalExprKind::CeilDiv ||
      launch.getOperands().size() != 2 ||
      launch.getOperands()[1] != queryLaunchExpression(range.getExtent()))
    return false;
  auto bound = queryPhysicalExpressionRange(
      cast<PhysicalExprAttr>(launch.getOperands()[0]), kernel);
  // 0 <= program < ceil(stop/width), stop >= 0 proves program*width <=
  // stop-1 in every nonempty launch. Together with the aligned-window lemma,
  // whole represented intervals are disjoint without signed modular wrap.
  return bound && !bound->smin().isNegative();
}

using OwnerCoordinates = SmallVector<std::pair<int64_t, Value>>;

FailureOr<OwnerCoordinates> programCoordinates(StoreOp store, scf::ForOp loop,
                                               func::FuncOp kernel) {
  auto group = loop->getParentOfType<ExecutionGroupOp>();
  if (!group) return failure();
  OwnerCoordinates result;
  llvm::SmallDenseSet<int64_t> axes;
  IndexRelations relations;
  for (auto [ordinal, coordinate] : llvm::enumerate(group.getCoordinates())) {
    if (relations.constant(group.getExtents()[ordinal]) == 1) continue;
    std::optional<std::pair<int64_t, Value>> selected;
    for (auto [axis, value] : llvm::zip(store.getSourceAxes(), store.getCoordinates())) {
      if (loop.isDefinedOutsideOfLoop(value) &&
          ownsCoordinate(value, group, ordinal, kernel)) {
        selected = std::pair<int64_t, Value>{axis, coordinateValues(value)};
        break;
      }
    }
    if (!selected || !axes.insert(selected->first).second) return failure();
    result.push_back(*selected);
  }
  return result;
}

bool preservesProgramCoordinates(AccessOpInterface access,
                                 ArrayRef<std::pair<int64_t, Value>> owner) {
  for (auto [axis, coordinate] : owner) {
    bool found = false;
    for (auto [accessAxis, value] : llvm::zip(access.getAccessSourceAxes(), access.getAccessCoordinates()))
      if (accessAxis == axis && samePhysicalScalarExpression(
              coordinateValues(value), coordinate)) found = true;
    if (!found) return false;
  }
  return true;
}

} // namespace

FailureOr<IndependentIterationAccesses>
queryIndependentIterationAccesses(scf::ForOp loop) {
  auto kernel = loop->getParentOfType<func::FuncOp>();
  IndexRelations relations;
  Type inductionType = loop.getInductionVar().getType();
  auto integer = dyn_cast<IntegerType>(inductionType);
  bool signedI32 = integer && !integer.isUnsigned() && integer.getWidth() == 32;
  if (!kernel || loop.getNumResults() || relations.constant(loop.getStep()) != 1 ||
      (!inductionType.isIndex() && !signedI32)) return failure();
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  bool singleInstance = space && llvm::all_of(space, [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() == PhysicalExprKind::Constant && extent.getValue() == 1;
  });
  IndependentIterationAccesses result;
  struct WrittenSlice { unsigned axis; Value coordinate; };
  llvm::DenseMap<Value, WrittenSlice> writtenAxes;
  llvm::DenseMap<Value, OwnerCoordinates> owners;
  llvm::DenseMap<Value, bool> varying;
  bool independent = true;
  loop.walk([&](Operation *operation) {
    if (!isa<LoadOp, StoreOp, scf::ForOp, scf::IfOp, scf::YieldOp>(operation) &&
        !isMemoryEffectFree(operation)) independent = false;
  });
  loop.walk([&](StoreOp store) {
    auto buffer = store.getResource().getDefiningOp<BufferOp>();
    auto type = dyn_cast<BufferType>(store.getResource().getType());
    auto view = dyn_cast<ViewType>(store.getResource().getType());
    auto group = buffer ? dyn_cast<ExecutionGroupOp>(buffer->getParentOp())
                        : ExecutionGroupOp{};
    bool privateBuffer = group && group->getBlock() == &kernel.front() &&
        buffer->getBlock() == &group.getBody().front() && type &&
        type.getScope().getValue() == BufferScope::ProgramPrivate;
    if (!privateBuffer && !view) { independent = false; return; }
    if (view && !llvm::is_contained(result.guardedViews, store.getResource()))
      result.guardedViews.push_back(store.getResource());
    if (view && !singleInstance) {
      auto owner = programCoordinates(store, loop, kernel);
      if (failed(owner)) { independent = false; return; }
      auto [previous, inserted] = owners.try_emplace(store.getResource(), *owner);
      if (!inserted && !preservesProgramCoordinates(
              cast<AccessOpInterface>(store.getOperation()), previous->second)) {
        independent = false;
        return;
      }
    }
    std::optional<unsigned> selected;
    Value selectedCoordinate;
    for (auto [axis, coordinate] : llvm::zip(store.getSourceAxes(), store.getCoordinates())) {
      if (!selected && isUnitIterationCoordinate(coordinate, loop)) {
        selected = axis;
        selectedCoordinate = coordinate;
      } else if (variesWithIteration(coordinate, loop, varying)) independent = false;
    }
    if (!selected) { independent = false; return; }
    auto [previous, inserted] = writtenAxes.try_emplace(
        store.getResource(), WrittenSlice{*selected, selectedCoordinate});
    independent &= inserted ||
        (previous->second.axis == *selected && samePhysicalScalarExpression(
            previous->second.coordinate, selectedCoordinate));
  });
  if (!independent || writtenAxes.empty()) return failure();

  SmallVector<Value> accessedViews(result.guardedViews.begin(), result.guardedViews.end());
  loop.walk([&](LoadOp load) {
    if (isa<ViewType>(load.getResource().getType()) &&
        !llvm::is_contained(accessedViews, load.getResource()))
      accessedViews.push_back(load.getResource());
    if (auto owner = owners.find(load.getResource()); owner != owners.end())
      independent &= preservesProgramCoordinates(
          cast<AccessOpInterface>(load.getOperation()), owner->second);
    auto written = writtenAxes.find(load.getResource());
    if (written == writtenAxes.end()) return;
    bool independentRead = false;
    for (auto [axis, coordinate] : llvm::zip(load.getSourceAxes(), load.getCoordinates()))
      if (axis == written->second.axis)
        independentRead = samePhysicalScalarExpression(coordinate, written->second.coordinate) ||
            (written->second.coordinate == loop.getInductionVar() &&
             isOutsideIterationRange(coordinate, loop));
    independent &= independentRead;
  });
  if (!independent) return failure();
  for (Value view : accessedViews) {
    auto argument = dyn_cast<BlockArgument>(view);
    if (!argument || argument.getOwner() != &kernel.front()) return failure();
  }
  for (Value written : result.guardedViews)
    for (Value other : accessedViews) {
      if (written == other) continue;
      auto pair = std::pair{written, other};
      if (cast<BlockArgument>(written).getArgNumber() > cast<BlockArgument>(other).getArgNumber())
        std::swap(pair.first, pair.second);
      if (!llvm::is_contained(result.disjointViews, pair)) result.disjointViews.push_back(pair);
    }
  return result;
}

} // namespace intent::gpu

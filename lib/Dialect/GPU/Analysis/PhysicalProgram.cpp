#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isCoordinateReplayNode(Operation *operation) {
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp,
             MakeRecordOp, ExtractOp>(operation);
}

bool isValueReplayNode(Operation *operation) {
  return isCoordinateReplayNode(operation) ||
         isa<ContractOp, ScaledContractOp, SparseContractOp, ReduceOp>(
             operation);
}

bool isAccessNode(Operation *operation) {
  return isa<LoadOp, GatherOp>(operation);
}

std::optional<int64_t> integerConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

Value stripBroadcast(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  while (auto splat = value.getDefiningOp<SplatOp>())
    value = splat.getValue();
  return value;
}

Value stripScalarIdentity(Value value) {
  value = stripBroadcast(value);
  while (true) {
    if (auto cast = value.getDefiningOp<CastOp>()) {
      value = stripBroadcast(cast.getValue());
      continue;
    }
    auto binary = value.getDefiningOp<BinaryOp>();
    if (!binary)
      return value;
    auto isInteger = [](Value operand, int64_t expected) {
      auto constant = operand.getDefiningOp<arith::ConstantOp>();
      auto integer =
          constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
      return integer && integer.getInt() == expected;
    };
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      if (isInteger(binary.getLhs(), 0)) {
        value = stripBroadcast(binary.getRhs());
        continue;
      }
      if (isInteger(binary.getRhs(), 0)) {
        value = stripBroadcast(binary.getLhs());
        continue;
      }
    }
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      if (isInteger(binary.getLhs(), 0))
        return binary.getLhs();
      if (isInteger(binary.getRhs(), 0))
        return binary.getRhs();
      if (isInteger(binary.getLhs(), 1)) {
        value = stripBroadcast(binary.getRhs());
        continue;
      }
      if (isInteger(binary.getRhs(), 1)) {
        value = stripBroadcast(binary.getLhs());
        continue;
      }
    }
    return value;
  }
}

void appendUnique(SmallVectorImpl<MakeRangeOp> &destination, MakeRangeOp range) {
  if (!llvm::is_contained(destination, range))
    destination.push_back(range);
}

void appendUnique(SmallVectorImpl<Operation *> &destination,
                  Operation *operation) {
  if (operation && !llvm::is_contained(destination, operation))
    destination.push_back(operation);
}

} // namespace

bool isPhysicalReplayNode(Operation *operation, PhysicalReplayScope scope,
                          bool allowAccesses) {
  if (!operation || operation->getNumResults() != 1)
    return false;
  if (isAccessNode(operation))
    return allowAccesses && operation->getNumRegions() == 0;
  if (operation->getNumRegions() != 0 && !isa<ReduceOp>(operation))
    return false;
  return scope == PhysicalReplayScope::Coordinate
             ? isCoordinateReplayNode(operation)
             : isValueReplayNode(operation);
}

FailureOr<unsigned> queryFragmentAxis(Type type, uint64_t sourceId) {
  PhysicalAxisProjection result = queryUniqueSourceAxis(type, sourceId);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

PhysicalAxisProjection queryFragmentAxis(Type type,
                                         PhysicalSourceAxis source) {
  PhysicalAxisProjection result;
  result.source = source;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<unsigned> axis;
  std::optional<int64_t> dimension;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != source.sourceId ||
        mapping.getSourceAxis() != source.sourceAxis)
      continue;
    if ((axis && *axis != mapping.getFragmentAxis()) ||
        (dimension && *dimension != mapping.getDimensionId())) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    axis = mapping.getFragmentAxis();
    dimension = mapping.getDimensionId();
  }
  if (!axis || !dimension)
    return result;
  result.state = PhysicalFactState::Exact;
  result.fragmentAxis = *axis;
  result.dimensionId = *dimension;
  return result;
}

PhysicalDimensionProjection queryFragmentDimension(Type type,
                                                   int64_t dimensionId) {
  PhysicalDimensionProjection result;
  result.dimensionId = dimensionId;
  if (dimensionId <= 0)
    return result;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<unsigned> axis;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getDimensionId() != dimensionId)
      continue;
    if (axis && *axis != mapping.getFragmentAxis()) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    axis = mapping.getFragmentAxis();
  }
  if (!axis)
    return result;
  result.state = PhysicalFactState::Exact;
  result.fragmentAxis = *axis;
  return result;
}

FailureOr<int64_t> querySourceDimension(Type type, PhysicalSourceAxis source) {
  if (auto range = dyn_cast<RangeType>(type))
    return range.getSourceId() == source.sourceId &&
                   range.getSourceAxis() == source.sourceAxis &&
                   range.getDimensionId() > 0
               ? FailureOr<int64_t>(range.getDimensionId())
               : FailureOr<int64_t>(failure());
  PhysicalAxisProjection projection = queryFragmentAxis(type, source);
  return projection.isExact() && projection.dimensionId > 0
             ? FailureOr<int64_t>(projection.dimensionId)
             : FailureOr<int64_t>(failure());
}

PhysicalAxisProjection queryUniqueSourceAxis(Type type, uint64_t sourceId) {
  PhysicalAxisProjection result;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return result;
  std::optional<PhysicalSourceAxis> source;
  std::optional<unsigned> axis;
  std::optional<int64_t> dimension;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != sourceId)
      continue;
    PhysicalSourceAxis current{mapping.getSourceId(),
                               mapping.getSourceAxis()};
    if ((source && !(*source == current)) ||
        (axis && *axis != mapping.getFragmentAxis()) ||
        (dimension && *dimension != mapping.getDimensionId())) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    source = current;
    axis = mapping.getFragmentAxis();
    dimension = mapping.getDimensionId();
  }
  if (!source || !axis || !dimension)
    return result;
  result.state = PhysicalFactState::Exact;
  result.source = *source;
  result.fragmentAxis = *axis;
  result.dimensionId = *dimension;
  return result;
}

FailureOr<unsigned> queryCoordinateIndex(ValueRange coordinates,
                                         uint64_t sourceId) {
  std::optional<unsigned> result;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    if (failed(queryFragmentAxis(coordinate.getType(), sourceId)))
      continue;
    if (result)
      return failure();
    result = index;
  }
  return result ? FailureOr<unsigned>(*result)
                : FailureOr<unsigned>(failure());
}

PhysicalAxisProjection
queryCoordinateIndex(ValueRange coordinates, PhysicalSourceAxis source) {
  PhysicalAxisProjection result;
  result.source = source;
  std::optional<unsigned> coordinateIndex;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    PhysicalAxisProjection projection =
        queryFragmentAxis(coordinate.getType(), source);
    if (projection.state == PhysicalFactState::Ambiguous) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    if (!projection.isExact())
      continue;
    if (coordinateIndex) {
      result.state = PhysicalFactState::Ambiguous;
      return result;
    }
    coordinateIndex = index;
  }
  if (!coordinateIndex)
    return result;
  result.state = PhysicalFactState::Exact;
  PhysicalAxisProjection projection =
      queryFragmentAxis(coordinates[*coordinateIndex].getType(), source);
  result.dimensionId = projection.dimensionId;
  result.fragmentAxis = *coordinateIndex;
  return result;
}

PhysicalProgramAnalysis::PhysicalProgramAnalysis(func::FuncOp kernel)
    : kernel(kernel) {}

FailureOr<unsigned>
PhysicalProgramAnalysis::fragmentAxis(Type type,
                                      PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryFragmentAxis(type, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

FailureOr<unsigned>
PhysicalProgramAnalysis::fragmentAxis(Type type, uint64_t sourceId) const {
  return queryFragmentAxis(type, sourceId);
}

FailureOr<unsigned> PhysicalProgramAnalysis::coordinateIndex(
    ValueRange coordinates, PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

FailureOr<unsigned> PhysicalProgramAnalysis::coordinateIndex(
    ValueRange coordinates, uint64_t sourceId) const {
  return queryCoordinateIndex(coordinates, sourceId);
}

bool PhysicalProgramAnalysis::carriesSource(Type type,
                                            PhysicalSourceAxis source) const {
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return false;
  return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
    auto mapping = cast<AxisMapAttr>(attribute);
    return mapping.getSourceId() == source.sourceId &&
           mapping.getSourceAxis() == source.sourceAxis;
  });
}

Value PhysicalProgramAnalysis::structuredSourceForArgument(
    BlockArgument argument) const {
  Block *block = argument.getOwner();
  Operation *owner = block ? block->getParentOp() : nullptr;
  if (auto fold = dyn_cast_or_null<RegionFoldOp>(owner)) {
    if (block == &fold.getSummarize().front() &&
        argument.getArgNumber() < fold.getSourceCount())
      return fold.getInputs()[argument.getArgNumber()];
  }
  if (auto scan = dyn_cast_or_null<RegionScanOp>(owner)) {
    if (block == &scan.getSummarize().front() &&
        argument.getArgNumber() < scan.getSourceCount())
      return scan.getInputs()[argument.getArgNumber()];
  }
  if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
    if (argument == loop.getInductionVar())
      return {};
    unsigned offset = argument.getArgNumber() - 1;
    if (offset < loop.getInitArgs().size())
      return loop.getInitArgs()[offset];
  }
  return {};
}

void PhysicalProgramAnalysis::collectRanges(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalRangeFact &result,
    SmallPtrSetImpl<Operation *> &visited) {
  if (!value)
    return;
  if (source && !carriesSource(value.getType(), *source))
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Value outer = structuredSourceForArgument(argument);
    if (!outer) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    collectRanges(outer, source, result, visited);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (!source || (range.getSourceId() == source->sourceId &&
                    range.getSourceAxis() == source->sourceAxis))
      appendUnique(result.roots, range);
    return;
  }
  if (isAccessNode(operation))
    appendUnique(result.accesses, operation);
  if (!isValueReplayNode(operation) && !isAccessNode(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  for (Value operand : operation->getOperands())
    collectRanges(operand, source, result, visited);
}

PhysicalRangeFact PhysicalProgramAnalysis::sourceRanges(
    Value value, std::optional<PhysicalSourceAxis> source) {
  if (!source) {
    if (auto cached = unrestrictedRangeCache.find(value);
        cached != unrestrictedRangeCache.end())
      return cached->second;
  }
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  SmallPtrSet<Operation *, 32> visited;
  collectRanges(value, source, result, visited);
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty() ||
      result.roots.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1)
    result.state = PhysicalFactState::Ambiguous;
  if (result.isUnique()) {
    std::optional<int64_t> step = integerConstant(result.roots.front().getStep());
    result.unitStep = step && *step == 1;
  }
  if (!source)
    unrestrictedRangeCache.try_emplace(value, result);
  return result;
}

void PhysicalProgramAnalysis::analyzeReplay(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    PhysicalReplayFact &result,
    SmallPtrSetImpl<Operation *> &visited) {
  if (!value || (source && !carriesSource(value.getType(), *source)))
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Value outer = structuredSourceForArgument(argument);
    if (!outer) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    analyzeReplay(outer, source, scope, allowAccesses, result, visited);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (isa<MakeRangeOp>(operation))
    return;
  if (isAccessNode(operation)) {
    result.crossesAccess = true;
    if (!allowAccesses) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
  } else if (!isPhysicalReplayNode(operation, scope, allowAccesses)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  if (operation->getNumRegions() != 0 && !isa<ReduceOp>(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  for (Value operand : operation->getOperands())
    analyzeReplay(operand, source, scope, allowAccesses, result, visited);
}

PhysicalReplayFact PhysicalProgramAnalysis::replayability(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses) {
  PhysicalReplayFact result;
  result.state = PhysicalFactState::Exact;
  SmallPtrSet<Operation *, 32> visited;
  analyzeReplay(value, source, scope, allowAccesses, result, visited);
  return result;
}

bool PhysicalProgramAnalysis::isTailPredicate(
    Value value, ArrayRef<std::pair<MakeRangeOp, Value>> ranges) const {
  if (!value)
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    return integer && integer.getType().isInteger(1) && integer.getInt() != 0;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    return isTailPredicate(broadcast.getValue(), ranges);
  if (auto splat = value.getDefiningOp<SplatOp>())
    return isTailPredicate(splat.getValue(), ranges);
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return isTailPredicate(reshape.getValue(), ranges);
  if (auto transpose = value.getDefiningOp<TransposeOp>())
    return isTailPredicate(transpose.getValue(), ranges);
  if (auto conjunction = value.getDefiningOp<BinaryOp>()) {
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    return (logical || bitwiseI1) &&
           isTailPredicate(conjunction.getLhs(), ranges) &&
           isTailPredicate(conjunction.getRhs(), ranges);
  }
  auto comparison = value.getDefiningOp<CompareOp>();
  if (!comparison || comparison.getPredicate() != ComparePredicate::Lt)
    return false;
  Value lhs = stripBroadcast(comparison.getLhs());
  Value rhs = stripScalarIdentity(comparison.getRhs());
  MakeRangeOp predicateRange = lhs.getDefiningOp<MakeRangeOp>();
  if (predicateRange) {
    Value start = stripScalarIdentity(predicateRange.getStart());
    Value extent = stripScalarIdentity(predicateRange.getExtent());
    auto zero = integerConstant(start);
    if (zero && *zero == 0 && rhs == extent)
      return true;
    auto stop = rhs.getDefiningOp<BinaryOp>();
    if (stop && stop.getOperatorKind() == BinaryOperator::Add &&
        ((stop.getLhs() == predicateRange.getStart() &&
          stop.getRhs() == predicateRange.getExtent()) ||
         (stop.getRhs() == predicateRange.getStart() &&
          stop.getLhs() == predicateRange.getExtent())))
      return true;
  }
  return llvm::any_of(ranges, [&](const auto &entry) {
    MakeRangeOp expectedRange = entry.first;
    Value expectedEnd = stripScalarIdentity(entry.second);
    if (lhs == expectedRange.getResult())
      return rhs == expectedEnd;
    if (!predicateRange ||
        predicateRange.getSourceId() != expectedRange.getSourceId() ||
        predicateRange.getSourceAxis() != expectedRange.getSourceAxis())
      return false;
    FailureOr<int64_t> predicateDimension = querySourceDimension(
        predicateRange.getResult().getType(),
        PhysicalSourceAxis{predicateRange.getSourceId(),
                           predicateRange.getSourceAxis()});
    FailureOr<int64_t> expectedDimension = querySourceDimension(
        expectedRange.getResult().getType(),
        PhysicalSourceAxis{expectedRange.getSourceId(),
                           expectedRange.getSourceAxis()});
    if (!predicateRange->hasAttr(sourceSubregionAttr) &&
        !expectedRange->hasAttr(sourceSubregionAttr) &&
        succeeded(predicateDimension) && succeeded(expectedDimension) &&
        *predicateDimension == *expectedDimension)
      return true;
    if (!predicateRange->hasAttr(sourceSubregionAttr) &&
        !expectedRange->hasAttr(sourceSubregionAttr) &&
        predicateRange.getSourceId() == expectedRange.getSourceId() &&
        rhs == expectedEnd)
      return true;
    return rhs == expectedEnd &&
           predicateRange.getStart() == expectedRange.getStart() &&
           predicateRange.getExtent() == expectedRange.getExtent() &&
           predicateRange.getStep() == expectedRange.getStep();
  });
}

PhysicalAccessFootprint
PhysicalProgramAnalysis::footprint(Operation *access) {
  PhysicalAccessFootprint result;
  auto collect = [&](Value resource, ValueRange coordinates,
                     ArrayRef<int64_t> sourceAxes, Value validity, Value fill) {
    result.resource = resource;
    result.coordinates.append(coordinates.begin(), coordinates.end());
    result.sourceAxes.append(sourceAxes.begin(), sourceAxes.end());
    result.validity = validity;
    result.fill = fill;
    result.state = coordinates.size() == sourceAxes.size()
                       ? PhysicalFactState::Exact
                       : PhysicalFactState::Unknown;
    for (Value coordinate : coordinates) {
      PhysicalRangeFact ranges = sourceRanges(coordinate);
      if (ranges.state != PhysicalFactState::Exact)
        result.state = ranges.state;
      for (MakeRangeOp range : ranges.roots)
        appendUnique(result.ranges, range);
    }
  };
  if (auto load = dyn_cast<LoadOp>(access))
    collect(load.getResource(), load.getCoordinates(), load.getSourceAxes(),
            load.getValid(), load.getFill());
  else if (auto gather = dyn_cast<GatherOp>(access))
    collect({}, gather.getCoordinates(), gather.getSourceAxes(),
            gather.getValid(), gather.getFill());
  else if (auto store = dyn_cast<StoreOp>(access))
    collect(store.getResource(), store.getCoordinates(), store.getSourceAxes(),
            store.getValid(), {});
  else if (auto atomic = dyn_cast<AtomicLoadOp>(access))
    collect(atomic.getResource(), atomic.getCoordinates(), atomic.getSourceAxes(),
            atomic.getValid(), {});
  else if (auto atomic = dyn_cast<AtomicStoreOp>(access))
    collect(atomic.getResource(), atomic.getCoordinates(), atomic.getSourceAxes(),
            atomic.getValid(), {});
  else if (auto atomic = dyn_cast<AtomicRMWOp>(access))
    collect(atomic.getResource(), atomic.getCoordinates(), atomic.getSourceAxes(),
            atomic.getValid(), {});
  else if (auto atomic = dyn_cast<AtomicCompareExchangeOp>(access))
    collect(atomic.getResource(), atomic.getCoordinates(), atomic.getSourceAxes(),
            atomic.getValid(), {});
  else
    result.state = PhysicalFactState::Unknown;
  return result;
}

void PhysicalProgramAnalysis::invalidate() {
  unrestrictedRangeCache.clear();
}

} // namespace intent::gpu

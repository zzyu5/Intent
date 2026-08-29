#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

bool isCoordinateReplayNode(Operation *operation) {
  return isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
             BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp,
             MakeRecordOp, ExtractOp, RandomBitsOp>(operation);
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

bool sameScalarExpression(Value lhs, Value rhs, unsigned depth = 0) {
  if (lhs == rhs)
    return true;
  if (depth >= 32 || lhs.getType() != rhs.getType())
    return false;
  Value left = stripScalarIdentity(lhs);
  Value right = stripScalarIdentity(rhs);
  if (left != lhs || right != rhs)
    return sameScalarExpression(left, right, depth + 1);
  auto leftConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rightConstant = rhs.getDefiningOp<arith::ConstantOp>();
  if (leftConstant || rightConstant)
    return leftConstant && rightConstant &&
           leftConstant.getValue() == rightConstant.getValue();
  auto leftBinary = lhs.getDefiningOp<BinaryOp>();
  auto rightBinary = rhs.getDefiningOp<BinaryOp>();
  if (leftBinary || rightBinary)
    return leftBinary && rightBinary &&
           leftBinary.getOperatorKind() == rightBinary.getOperatorKind() &&
           sameScalarExpression(leftBinary.getLhs(), rightBinary.getLhs(),
                                depth + 1) &&
           sameScalarExpression(leftBinary.getRhs(), rightBinary.getRhs(),
                                depth + 1);
  auto leftCast = lhs.getDefiningOp<CastOp>();
  auto rightCast = rhs.getDefiningOp<CastOp>();
  return leftCast && rightCast &&
         sameScalarExpression(leftCast.getValue(), rightCast.getValue(),
                              depth + 1);
}

bool isUnitStepValue(Value value) {
  if (std::optional<int64_t> constant = integerConstant(value))
    return *constant == 1;
  if (auto cast = value.getDefiningOp<CastOp>())
    return isUnitStepValue(cast.getValue());
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    return range && bound.getBound() == 2 && isUnitStepValue(range.getStep());
  }
  return false;
}

bool reductionTypeConsumesSource(Type type, ArrayRef<int64_t> axes,
                                 PhysicalSourceAxis source) {
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return reductionTypeConsumesSource(cast<TypeAttr>(field).getValue(), axes,
                                         source);
    });
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return false;
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
      continue;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    if (PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()} == source)
      return true;
  }
  return false;
}

Value accessResource(Operation *operation) {
  if (auto load = dyn_cast<LoadOp>(operation))
    return load.getResource();
  if (auto store = dyn_cast<StoreOp>(operation))
    return store.getResource();
  if (auto scatter = dyn_cast<ScatterReduceOp>(operation))
    return scatter.getResource();
  if (auto atomic = dyn_cast<AtomicLoadOp>(operation))
    return atomic.getResource();
  if (auto atomic = dyn_cast<AtomicStoreOp>(operation))
    return atomic.getResource();
  if (auto atomic = dyn_cast<AtomicRMWOp>(operation))
    return atomic.getResource();
  if (auto atomic = dyn_cast<AtomicCompareExchangeOp>(operation))
    return atomic.getResource();
  return {};
}

bool readsResource(Operation *operation) {
  return isa<LoadOp, ScatterReduceOp, AtomicLoadOp, AtomicRMWOp,
             AtomicCompareExchangeOp>(operation);
}

bool writesResourceWithoutReading(Operation *operation) {
  return isa<StoreOp, AtomicStoreOp>(operation);
}

bool valueMatchesExtent(Value value, PhysicalExprAttr extent) {
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return physical.getExpression() == extent;
  auto kind = static_cast<PhysicalExprKind>(extent.getKind());
  if (kind == PhysicalExprKind::Constant) {
    std::optional<int64_t> constant = integerConstant(value);
    return constant && *constant == extent.getValue();
  }
  if (kind == PhysicalExprKind::Parameter) {
    auto parameter = value.getDefiningOp<ParameterOp>();
    return parameter &&
           parameter.getParameter().getName() == extent.getSymbol();
  }
  return false;
}

struct LoopCoordinate {
  Value induction;
  Value lower;
  Value upper;
  Value step;
};

bool unconditionalStoreCoversBuffer(Operation *operation, BufferOp buffer,
                                    ArrayRef<LoopCoordinate> loops) {
  ValueRange coordinates;
  ArrayRef<int64_t> sourceAxes;
  Value valid;
  if (auto store = dyn_cast<StoreOp>(operation)) {
    if (store.getResource() != buffer.getResult())
      return false;
    coordinates = store.getCoordinates();
    sourceAxes = store.getSourceAxes();
    valid = store.getValid();
  } else if (auto atomic = dyn_cast<AtomicStoreOp>(operation)) {
    if (atomic.getResource() != buffer.getResult())
      return false;
    coordinates = atomic.getCoordinates();
    sourceAxes = atomic.getSourceAxes();
    valid = atomic.getValid();
  } else {
    return false;
  }
  BufferType type = buffer.getResult().getType();
  if (valid || coordinates.size() != type.getShape().size() ||
      sourceAxes.size() != coordinates.size())
    return false;
  llvm::DenseSet<int64_t> coveredAxes;
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes)) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(type.getShape().size()) ||
        !coveredAxes.insert(sourceAxis).second)
      return false;
    auto loop = llvm::find_if(loops, [&](const LoopCoordinate &candidate) {
      return sameScalarExpression(coordinate, candidate.induction);
    });
    if (loop == loops.end() || integerConstant(loop->lower) != 0 ||
        integerConstant(loop->step) != 1 ||
        !valueMatchesExtent(
            loop->upper,
            cast<PhysicalExprAttr>(type.getShape()[sourceAxis])))
      return false;
  }
  return coveredAxes.size() == type.getShape().size();
}

bool regionInitializesBuffer(Region &region, BufferOp buffer,
                             SmallVectorImpl<LoopCoordinate> &loops);

bool operationInitializesBuffer(Operation *operation, BufferOp buffer,
                                SmallVectorImpl<LoopCoordinate> &loops) {
  if (unconditionalStoreCoversBuffer(operation, buffer, loops))
    return true;
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    loops.push_back(
        {loop.getInductionVar(), loop.getLowerBound(), loop.getUpperBound(),
         loop.getStep()});
    bool initializes = regionInitializesBuffer(loop.getRegion(), buffer, loops);
    loops.pop_back();
    return initializes;
  }
  if (auto branch = dyn_cast<scf::IfOp>(operation)) {
    if (branch.getElseRegion().empty())
      return false;
    SmallVector<LoopCoordinate> thenLoops(loops.begin(), loops.end());
    SmallVector<LoopCoordinate> elseLoops(loops.begin(), loops.end());
    return regionInitializesBuffer(branch.getThenRegion(), buffer, thenLoops) &&
           regionInitializesBuffer(branch.getElseRegion(), buffer, elseLoops);
  }
  return false;
}

bool regionInitializesBuffer(Region &region, BufferOp buffer,
                             SmallVectorImpl<LoopCoordinate> &loops) {
  if (!llvm::hasSingleElement(region))
    return false;
  for (Operation &operation : region.front())
    if (operationInitializesBuffer(&operation, buffer, loops))
      return true;
  return false;
}

bool precedingRegionInitializesBuffer(Operation *read, BufferOp buffer,
                                      ArrayRef<Operation *> writes,
                                      DominanceInfo &dominance) {
  llvm::SmallPtrSet<Operation *, 8> candidates;
  for (Operation *write : writes) {
    for (Operation *candidate = write->getParentOp(); candidate;
         candidate = candidate->getParentOp()) {
      if (!isa<scf::ForOp, scf::IfOp>(candidate))
        continue;
      candidates.insert(candidate);
      if (candidate->getBlock() == buffer->getBlock())
        break;
    }
  }
  for (Operation *candidate : candidates) {
    if (!dominance.properlyDominates(candidate, read))
      continue;
    SmallVector<LoopCoordinate> loops;
    if (operationInitializesBuffer(candidate, buffer, loops))
      return true;
  }
  return false;
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

PhysicalParameterBinding queryParameterBinding(ParameterOp parameter) {
  PhysicalParameterBinding result;
  if (!parameter)
    return result;
  auto dimension = parameter->getAttrOfType<IntegerAttr>(dimensionAttr);
  auto coverage = parameter->getAttrOfType<IntegerAttr>(coverageDimensionAttr);
  if (dimension && dimension.getInt() <= 0)
    return result;
  if (coverage && coverage.getInt() <= 0)
    return result;
  if (dimension && coverage && dimension.getInt() != coverage.getInt()) {
    result.state = PhysicalFactState::Ambiguous;
    return result;
  }
  if (coverage)
    result.dimension = coverage.getInt();
  else if (dimension)
    result.dimension = dimension.getInt();
  if (auto source =
          parameter->getAttrOfType<PhysicalSourceAttr>(parameterSourceAttr))
    result.source =
        PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis(),
                           source.getDerived()};
  result.state = result.dimension || result.source ? PhysicalFactState::Exact
                                                   : PhysicalFactState::Unknown;
  return result;
}

FailureOr<ParameterOp> queryParameterBySymbol(func::FuncOp kernel,
                                              StringAttr symbol) {
  ParameterOp result;
  bool ambiguous = false;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName() != symbol)
      return;
    if (result && result != parameter)
      ambiguous = true;
    else
      result = parameter;
  });
  return result && !ambiguous ? FailureOr<ParameterOp>(result)
                              : FailureOr<ParameterOp>(failure());
}

FailureOr<ParameterOp> queryBlockingParameter(func::FuncOp kernel,
                                              MakeRangeOp range) {
  auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
  auto extent = fragment && fragment.getShape().size() == 1
                    ? dyn_cast<PhysicalExprAttr>(fragment.getShape()[0])
                    : PhysicalExprAttr();
  if (extent && extent.getKind() ==
                    static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return queryParameterBySymbol(kernel, extent.getSymbol());

  PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                            range.getDerived()};
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  ParameterOp sourceMatch;
  ParameterOp dimensionMatch;
  bool sourceAmbiguous = false;
  bool dimensionAmbiguous = false;
  kernel.walk([&](ParameterOp parameter) {
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact())
      return;
    if (binding.source && *binding.source == source) {
      if (sourceMatch && sourceMatch != parameter)
        sourceAmbiguous = true;
      else
        sourceMatch = parameter;
    }
    if (succeeded(dimension) && binding.dimension &&
        *binding.dimension == *dimension) {
      if (dimensionMatch && dimensionMatch != parameter)
        dimensionAmbiguous = true;
      else
        dimensionMatch = parameter;
    }
  });
  if (sourceMatch && !sourceAmbiguous)
    return sourceMatch;
  return dimensionMatch && !dimensionAmbiguous
             ? FailureOr<ParameterOp>(dimensionMatch)
             : FailureOr<ParameterOp>(failure());
}

PhysicalSourceAxis sourceAxisIdentity(AxisMapAttr mapping) {
  return {mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived()};
}

PhysicalSourceAxis sourceAxisIdentity(MakeRangeOp range) {
  return {range.getSourceId(), range.getSourceAxis(), range.getDerived()};
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
        mapping.getSourceAxis() != source.sourceAxis ||
        mapping.getDerived() != source.derived)
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

FailureOr<AxisMapAttr> queryAxisMap(Type type, unsigned fragmentAxis) {
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return failure();
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == fragmentAxis)
      return mapping;
  }
  return failure();
}

FailureOr<int64_t> queryRangeDimension(MakeRangeOp range) {
  return querySourceDimension(range.getResult().getType(),
                              sourceAxisIdentity(range));
}

bool samePhysicalScalarExpression(Value lhs, Value rhs) {
  return sameScalarExpression(lhs, rhs);
}

bool sameLogicalRange(MakeRangeOp lhs, MakeRangeOp rhs) {
  if (!lhs || !rhs || lhs.getSourceId() != rhs.getSourceId() ||
      lhs.getSourceAxis() != rhs.getSourceAxis() ||
      lhs.getDerived() != rhs.getDerived())
    return false;
  FailureOr<int64_t> leftDimension = queryRangeDimension(lhs);
  FailureOr<int64_t> rightDimension = queryRangeDimension(rhs);
  if (!lhs->hasAttr(sourceSubregionAttr) &&
      !rhs->hasAttr(sourceSubregionAttr) && succeeded(leftDimension) &&
      succeeded(rightDimension) && *leftDimension == *rightDimension)
    return true;
  return sameScalarExpression(lhs.getStart(), rhs.getStart()) &&
         sameScalarExpression(lhs.getExtent(), rhs.getExtent()) &&
         sameScalarExpression(lhs.getStep(), rhs.getStep());
}

bool isUnitStepRange(MakeRangeOp range) {
  return range && isUnitStepValue(range.getStep());
}

FailureOr<MakeRangeOp>
queryExactLogicalRange(const PhysicalRangeFact &fact) {
  if (fact.state == PhysicalFactState::Unknown || fact.roots.empty())
    return failure();
  MakeRangeOp first = fact.roots.front();
  return llvm::all_of(fact.roots, [&](MakeRangeOp range) {
           return sameLogicalRange(first, range);
         })
             ? FailureOr<MakeRangeOp>(first)
             : FailureOr<MakeRangeOp>(failure());
}

SmallVector<PhysicalAxisProjection, 2>
queryFragmentAxes(Type type, PhysicalSourceAxis source) {
  SmallVector<PhysicalAxisProjection, 2> results;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return results;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != source.sourceId ||
        mapping.getSourceAxis() != source.sourceAxis ||
        mapping.getDerived() != source.derived)
      continue;
    results.push_back(PhysicalAxisProjection{
        PhysicalFactState::Exact, source, mapping.getDimensionId(),
        mapping.getFragmentAxis()});
  }
  return results;
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
                   range.getDerived() == source.derived &&
                   range.getDimensionId() > 0
               ? FailureOr<int64_t>(range.getDimensionId())
               : FailureOr<int64_t>(failure());
  PhysicalAxisProjection projection = queryFragmentAxis(type, source);
  return projection.isExact() && projection.dimensionId > 0
             ? FailureOr<int64_t>(projection.dimensionId)
             : FailureOr<int64_t>(failure());
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

FailureOr<unsigned> queryCoordinatePosition(ValueRange coordinates,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
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

FailureOr<unsigned> PhysicalProgramAnalysis::coordinateIndex(
    ValueRange coordinates, PhysicalSourceAxis source) const {
  PhysicalAxisProjection result = queryCoordinateIndex(coordinates, source);
  return result.isExact() ? FailureOr<unsigned>(result.fragmentAxis)
                          : FailureOr<unsigned>(failure());
}

bool PhysicalProgramAnalysis::carriesSource(Type type,
                                            PhysicalSourceAxis source) const {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      auto mapping = cast<AxisMapAttr>(attribute);
      return mapping.getSourceId() == source.sourceId &&
             mapping.getSourceAxis() == source.sourceAxis &&
             mapping.getDerived() == source.derived;
    });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return carriesSource(cast<TypeAttr>(field).getValue(), source);
    });
  return false;
}

SmallVector<Value, 2> PhysicalProgramAnalysis::structuredSourcesForArgument(
    BlockArgument argument) const {
  SmallVector<Value, 2> sources;
  Block *block = argument.getOwner();
  Operation *owner = block ? block->getParentOp() : nullptr;
  if (auto fold = dyn_cast_or_null<RegionFoldOp>(owner)) {
    if (block == &fold.getSummarize().front() &&
        argument.getArgNumber() < fold.getSourceCount())
      sources.push_back(fold.getInputs()[argument.getArgNumber()]);
    return sources;
  }
  if (auto scan = dyn_cast_or_null<RegionScanOp>(owner)) {
    if (block == &scan.getSummarize().front() &&
        argument.getArgNumber() < scan.getSourceCount())
      sources.push_back(scan.getInputs()[argument.getArgNumber()]);
    return sources;
  }
  if (auto loop = dyn_cast_or_null<scf::ForOp>(owner)) {
    if (argument == loop.getInductionVar())
      return sources;
    unsigned offset = argument.getArgNumber() - 1;
    if (offset >= loop.getInitArgs().size())
      return sources;
    sources.push_back(loop.getInitArgs()[offset]);
    if (auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        yield && offset < yield.getResults().size())
      sources.push_back(yield.getResults()[offset]);
  }
  return sources;
}

void PhysicalProgramAnalysis::collectRanges(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalRangeFact &result,
    SmallPtrSetImpl<Operation *> &visited) {
  if (!value)
    return;
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      uint64_t field = extract.getField();
      if (field >= record.getFields().size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, extract);
        return;
      }
      collectRanges(record.getFields()[field], source, result, visited);
      return;
    }
  }
  if (source && !carriesSource(value.getType(), *source))
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    if (outer.empty()) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value related : outer)
      collectRanges(related, source, result, visited);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (!source || sourceAxisIdentity(range) == *source)
      appendUnique(result.roots, range);
    return;
  }
  // These operations are typed coordinate leaves.  They do not contribute a
  // fragment range root, but reaching one is an exact end of provenance rather
  // than an unknown operation in the producer graph.
  if (isa<arith::ConstantOp, PhysicalExprOp, ProgramIdOp, DelinearizeOp,
          WorksetCoordinateOp, DimOp, RangeOp, RangeBoundOp>(operation))
    return;
  if (auto scan = dyn_cast<ScanOp>(operation)) {
    bool followed = false;
    for (Value scanSource : scan.getInputs().take_front(scan.getSourceCount())) {
      if (source && !carriesSource(scanSource.getType(), *source))
        continue;
      followed = true;
      collectRanges(scanSource, source, result, visited);
    }
    if (!followed) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
    }
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
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  if (!source)
    unrestrictedRangeCache.try_emplace(value, result);
  return result;
}

PhysicalRangeFact
PhysicalProgramAnalysis::programRanges(PhysicalSourceAxis source) {
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  kernel.walk([&](MakeRangeOp range) {
    if (!(sourceAxisIdentity(range) == source))
      return;
    appendUnique(result.roots, range);
  });
  if (result.roots.empty()) {
    result.state = PhysicalFactState::Unknown;
    return result;
  }
  MakeRangeOp authority = result.roots.front();
  if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
        return sameLogicalRange(authority, range);
      }))
    result.state = PhysicalFactState::Ambiguous;
  result.unitStep = llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

void PhysicalProgramAnalysis::collectAxisRanges(
    Value value, unsigned fragmentAxis, PhysicalRangeFact &result,
    SmallPtrSetImpl<Operation *> &visited) {
  if (!value)
    return;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size()) {
    result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    bool followed = false;
    for (Value related : outer) {
      auto outerFragment = dyn_cast<FragmentType>(related.getType());
      if (!outerFragment || fragmentAxis >= outerFragment.getShape().size())
        continue;
      followed = true;
      collectAxisRanges(related, fragmentAxis, result, visited);
    }
    if (!followed) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (fragmentAxis == 0)
      appendUnique(result.roots, range);
    else
      result.state = PhysicalFactState::Unknown;
    return;
  }
  if (auto scan = dyn_cast<ScanOp>(operation)) {
    auto expected =
        cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    bool followed = false;
    for (Value scanSource : scan.getInputs().take_front(scan.getSourceCount())) {
      auto sourceType = dyn_cast<FragmentType>(scanSource.getType());
      if (!sourceType || fragmentAxis >= sourceType.getShape().size())
        continue;
      auto mapping =
          cast<AxisMapAttr>(sourceType.getAxisMaps()[fragmentAxis]);
      if (!(sourceAxisIdentity(mapping) == sourceAxisIdentity(expected)) ||
          mapping.getDimensionId() != expected.getDimensionId())
        continue;
      followed = true;
      collectAxisRanges(scanSource, fragmentAxis, result, visited);
    }
    if (!followed) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
    }
    return;
  }
  if (auto extract = dyn_cast<ExtractOp>(operation)) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      if (extract.getField() >= record.getFields().size()) {
        result.state = PhysicalFactState::Unknown;
        return;
      }
      collectAxisRanges(record.getFields()[extract.getField()], fragmentAxis,
                        result, visited);
      return;
    }
  }
  if (auto splat = dyn_cast<SplatOp>(operation))
    return;
  if (auto broadcast = dyn_cast<BroadcastOp>(operation)) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (!input)
      return;
    if (input.getShape().size() > fragment.getShape().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    if (input.getShape().size() != fragment.getShape().size()) {
      auto expected =
          cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
      std::optional<unsigned> inputAxis;
      for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
        auto mapping = cast<AxisMapAttr>(input.getAxisMaps()[axis]);
        if (!(sourceAxisIdentity(mapping) == sourceAxisIdentity(expected)) ||
            mapping.getDimensionId() != expected.getDimensionId())
          continue;
        if (inputAxis) {
          result.state = PhysicalFactState::Ambiguous;
          return;
        }
        inputAxis = axis;
      }
      if (!inputAxis)
        return;
      collectAxisRanges(broadcast.getValue(), *inputAxis, result, visited);
      return;
    }
    unsigned offset = fragment.getShape().size() - input.getShape().size();
    if (fragmentAxis < offset)
      return;
    unsigned inputAxis = fragmentAxis - offset;
    auto inputExtent = cast<PhysicalExprAttr>(input.getShape()[inputAxis]);
    auto outputExtent = cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
    if (inputExtent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        inputExtent.getValue() == 1 && inputExtent != outputExtent)
      return;
    collectAxisRanges(broadcast.getValue(), inputAxis, result, visited);
    return;
  }
  if (auto transpose = dyn_cast<TransposeOp>(operation)) {
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    if (fragmentAxis >= permutation.size() || permutation[fragmentAxis] < 0) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    collectAxisRanges(transpose.getValue(),
                      static_cast<unsigned>(permutation[fragmentAxis]), result,
                      visited);
    return;
  }
  if (auto load = dyn_cast<LoadOp>(operation)) {
    appendUnique(result.accesses, operation);
    auto expected = cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    bool found = false;
    for (Value coordinate : load.getCoordinates()) {
      auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
      if (!coordinateType)
        continue;
      std::optional<unsigned> coordinateAxis;
      if (fragmentAxis < coordinateType.getAxisMaps().size()) {
        auto candidate = cast<AxisMapAttr>(
            coordinateType.getAxisMaps()[fragmentAxis]);
        if (sourceAxisIdentity(candidate) == sourceAxisIdentity(expected) &&
            candidate.getDimensionId() == expected.getDimensionId())
          coordinateAxis = fragmentAxis;
      }
      PhysicalAxisProjection sourceAxis = queryFragmentAxis(
          coordinateType,
          PhysicalSourceAxis{expected.getSourceId(), expected.getSourceAxis(),
                             expected.getDerived()});
      PhysicalDimensionProjection dimensionAxis =
          queryFragmentDimension(coordinateType, expected.getDimensionId());
      if (!coordinateAxis && sourceAxis.isExact())
        coordinateAxis = sourceAxis.fragmentAxis;
      else if (!coordinateAxis && dimensionAxis.isExact())
        coordinateAxis = dimensionAxis.fragmentAxis;
      if (!coordinateAxis)
        continue;
      if (found) {
        result.state = PhysicalFactState::Ambiguous;
        return;
      }
      found = true;
      collectAxisRanges(coordinate, *coordinateAxis, result, visited);
    }
    if (!found) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
    }
    return;
  }
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    if (!opResult || opResult.getResultNumber() >= reduce.getSourceCount()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    auto source = dyn_cast<FragmentType>(
        reduce.getInputs()[opResult.getResultNumber()].getType());
    if (!source) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    llvm::SmallDenseSet<int64_t> reduced(reduce.getAxes().begin(),
                                         reduce.getAxes().end());
    SmallVector<unsigned> freeAxes;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      if (!reduced.contains(axis))
        freeAxes.push_back(axis);
    if (fragmentAxis >= freeAxes.size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    collectAxisRanges(reduce.getInputs()[opResult.getResultNumber()],
                      freeAxes[fragmentAxis], result, visited);
    return;
  }
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    auto opResult = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!opResult || !yield ||
        opResult.getResultNumber() >= yield.getResults().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    collectAxisRanges(yield.getResults()[opResult.getResultNumber()],
                      fragmentAxis, result, visited);
    return;
  }
  if (auto select = dyn_cast<SelectOp>(operation)) {
    for (Value selected : {select.getTrueValue(), select.getFalseValue()}) {
      auto selectedType = dyn_cast<FragmentType>(selected.getType());
      if (!selectedType ||
          selectedType.getShape().size() != fragment.getShape().size() ||
          fragmentAxis >= selectedType.getShape().size())
        continue;
      collectAxisRanges(selected, fragmentAxis, result, visited);
    }
    return;
  }
  if (!isCoordinateReplayNode(operation)) {
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
    return;
  }
  bool followed = false;
  for (Value operand : operation->getOperands()) {
    auto operandType = dyn_cast<FragmentType>(operand.getType());
    if (!operandType || operandType.getShape().size() != fragment.getShape().size() ||
        fragmentAxis >= operandType.getShape().size())
      continue;
    followed = true;
    collectAxisRanges(operand, fragmentAxis, result, visited);
  }
  if (!followed && !operation->getOperands().empty()) {
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
  }
}

PhysicalRangeFact PhysicalProgramAnalysis::axisRanges(Value value,
                                                      unsigned fragmentAxis) {
  PhysicalRangeFact result;
  result.state = PhysicalFactState::Exact;
  SmallPtrSet<Operation *, 32> visited;
  collectAxisRanges(value, fragmentAxis, result, visited);
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty() ||
      result.roots.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1)
    result.state = PhysicalFactState::Ambiguous;
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

void PhysicalProgramAnalysis::analyzeReplay(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    PhysicalReplayFact &result,
    SmallPtrSetImpl<Operation *> &visited) {
  if (!value)
    return;
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      uint64_t field = extract.getField();
      if (field >= record.getFields().size()) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, extract);
        return;
      }
      analyzeReplay(record.getFields()[field], source, scope, allowAccesses,
                    result, visited);
      return;
    }
  }
  if (!isa<FragmentType, RecordType>(value.getType()))
    return;
  if (source && !carriesSource(value.getType(), *source))
    return;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    if (auto loop = dyn_cast_or_null<scf::ForOp>(owner);
        loop && argument != loop.getInductionVar()) {
      appendUnique(result.blockers, loop);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    SmallVector<Value, 2> outer = structuredSourcesForArgument(argument);
    if (outer.empty()) {
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value related : outer)
      analyzeReplay(related, source, scope, allowAccesses, result, visited);
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
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    WalkResult helper = reduce.getCombine().walk([&](Operation *nested) {
      if (isa<YieldOp, arith::ConstantOp>(nested))
        return WalkResult::advance();
      if (!isPhysicalReplayNode(nested, PhysicalReplayScope::Coordinate,
                                /*allowAccesses=*/false)) {
        appendUnique(result.blockers, nested);
        result.state = PhysicalFactState::Unknown;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (helper.wasInterrupted())
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

PhysicalReductionDependencyFact PhysicalProgramAnalysis::reductionDependency(
    Value value, PhysicalSourceAxis source,
    std::optional<int64_t> sourceDimension) {
  SmallPtrSet<Operation *, 32> visited;
  std::function<PhysicalReductionDependencyFact(Value)> analyze =
      [&](Value current) -> PhysicalReductionDependencyFact {
    PhysicalReductionDependencyFact exact;
    exact.state = PhysicalFactState::Exact;
    Operation *operation = current.getDefiningOp();
    if (!operation || !visited.insert(operation).second)
      return exact;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      for (Value input :
           reduce.getInputs().take_front(reduce.getSourceCount())) {
        if (reductionTypeConsumesSource(input.getType(), reduce.getAxes(),
                                        source)) {
          exact.depends = true;
          return exact;
        }
        if (!sourceDimension)
          continue;
        auto fragment = dyn_cast<FragmentType>(input.getType());
        if (!fragment)
          continue;
        for (int64_t axis : reduce.getAxes()) {
          if (axis < 0 ||
              axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
            continue;
          auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
          PhysicalRangeFact ranges = sourceRanges(
              input, sourceAxisIdentity(mapping));
          if (llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
                FailureOr<int64_t> dimension = queryRangeDimension(range);
                return succeeded(dimension) &&
                       *dimension == *sourceDimension;
              })) {
            exact.depends = true;
            return exact;
          }
        }
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      auto traversal =
          loop->getAttrOfType<PhysicalSourceAttr>(reductionTraversalSourceAttr);
      if (traversal &&
          PhysicalSourceAxis{traversal.getSourceId(),
                             traversal.getSourceAxis(),
                             traversal.getDerived()} == source) {
        exact.depends = true;
        return exact;
      }
      auto result = dyn_cast<OpResult>(current);
      auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= loop.getInitArgs().size()) {
        exact.state = PhysicalFactState::Unknown;
        appendUnique(exact.blockers, operation);
        return exact;
      }
      unsigned index = result.getResultNumber();
      for (Value related :
           {loop.getInitArgs()[index], yield.getResults()[index]}) {
        PhysicalReductionDependencyFact nested = analyze(related);
        if (nested.depends)
          return nested;
        if (!nested.isExact()) {
          exact.state = PhysicalFactState::Unknown;
          llvm::append_range(exact.blockers, nested.blockers);
        }
      }
      return exact;
    }
    if (operation->getNumRegions() != 0) {
      exact.state = PhysicalFactState::Unknown;
      appendUnique(exact.blockers, operation);
      return exact;
    }
    for (Value operand : operation->getOperands()) {
      PhysicalReductionDependencyFact nested = analyze(operand);
      if (nested.depends)
        return nested;
      if (!nested.isExact()) {
        exact.state = PhysicalFactState::Unknown;
        llvm::append_range(exact.blockers, nested.blockers);
      }
    }
    return exact;
  };
  return analyze(value);
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
  return llvm::any_of(ranges, [&](const auto &entry) {
    MakeRangeOp expectedRange = entry.first;
    Value expectedEnd = stripScalarIdentity(entry.second);
    if (lhs == expectedRange.getResult())
      return rhs == expectedEnd;
    if (!predicateRange ||
        !(sourceAxisIdentity(predicateRange) ==
          sourceAxisIdentity(expectedRange)))
      return false;
    FailureOr<int64_t> predicateDimension = querySourceDimension(
        predicateRange.getResult().getType(), sourceAxisIdentity(predicateRange));
    FailureOr<int64_t> expectedDimension = querySourceDimension(
        expectedRange.getResult().getType(), sourceAxisIdentity(expectedRange));
    if (!predicateRange->hasAttr(sourceSubregionAttr) &&
        !expectedRange->hasAttr(sourceSubregionAttr) &&
        succeeded(predicateDimension) && succeeded(expectedDimension) &&
        *predicateDimension == *expectedDimension &&
        sameScalarExpression(rhs, expectedEnd))
      return true;
    return sameScalarExpression(rhs, expectedEnd) &&
           sameScalarExpression(predicateRange.getStart(),
                                expectedRange.getStart()) &&
           sameScalarExpression(predicateRange.getExtent(),
                                expectedRange.getExtent()) &&
           sameScalarExpression(predicateRange.getStep(),
                                expectedRange.getStep());
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
    result.state = resource && coordinates.size() == sourceAxes.size()
                       ? PhysicalFactState::Exact
                       : PhysicalFactState::Unknown;
    result.rangeState = PhysicalFactState::Exact;
    for (Value coordinate : coordinates) {
      PhysicalRangeFact ranges = sourceRanges(coordinate);
      // A footprint records the complete set of ranges, not a request for one
      // unique range.  Multiple roots are therefore exact here.  A coordinate
      // with no range roots is also exact when it is a scalar/broadcast-only
      // expression.  Only an operation that blocks provenance makes the
      // address footprint unknown.
      if (!ranges.blockers.empty())
        result.rangeState = PhysicalFactState::Unknown;
      for (Operation *blocker : ranges.blockers)
        appendUnique(result.blockers, blocker);
      for (MakeRangeOp range : ranges.roots)
        appendUnique(result.ranges, range);
    }
  };
  if (auto load = dyn_cast<LoadOp>(access))
    collect(load.getResource(), load.getCoordinates(), load.getSourceAxes(),
            load.getValid(), load.getFill());
  else if (auto gather = dyn_cast<GatherOp>(access))
    collect(gather.getSource(), gather.getCoordinates(), gather.getSourceAxes(),
            gather.getValid(), gather.getFill());
  else if (auto store = dyn_cast<StoreOp>(access))
    collect(store.getResource(), store.getCoordinates(), store.getSourceAxes(),
            store.getValid(), {});
  else if (auto scatter = dyn_cast<ScatterReduceOp>(access))
    collect(scatter.getResource(), scatter.getCoordinates(),
            scatter.getSourceAxes(), scatter.getValid(), {});
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

PhysicalBufferDataflowFact
PhysicalProgramAnalysis::bufferDataflow(BufferOp buffer) {
  PhysicalBufferDataflowFact result;
  if (!buffer || buffer->getParentOfType<func::FuncOp>() != kernel)
    return result;
  result.state = PhysicalFactState::Exact;
  SmallVector<Operation *> reads;
  SmallVector<Operation *> initializingWrites;
  for (OpOperand &use : buffer.getResult().getUses()) {
    Operation *user = use.getOwner();
    if (isa<AssumeInBoundsOp>(user))
      continue;
    if (accessResource(user) != buffer.getResult()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, user);
      continue;
    }
    if (readsResource(user))
      reads.push_back(user);
    if (writesResourceWithoutReading(user))
      initializingWrites.push_back(user);
  }
  if (buffer.getResult().getType().getInitialization().getValue() ==
      BufferInitialization::FullValue)
    return result;

  DominanceInfo dominance(kernel);
  for (Operation *read : reads) {
    bool initialized = llvm::any_of(initializingWrites, [&](Operation *write) {
      return dominance.properlyDominates(write, read);
    });
    if (!initialized)
      initialized = precedingRegionInitializesBuffer(
          read, buffer, initializingWrites, dominance);
    if (!initialized) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, read);
    }
  }
  return result;
}

} // namespace intent::gpu

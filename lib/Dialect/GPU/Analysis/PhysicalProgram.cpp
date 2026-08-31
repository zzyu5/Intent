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
         isa<ContractOp, ScaledContractOp, SparseContractOp, ReduceOp, ScanOp>(
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

Value stripCombineProjection(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
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

bool isProvablySingletonLogicalRange(MakeRangeOp range) {
  std::optional<int64_t> start = integerConstant(range.getLogicalStart());
  std::optional<int64_t> stop = integerConstant(range.getLogicalStop());
  std::optional<int64_t> step = integerConstant(range.getStep());
  if (start && stop && step && *step > 0)
    return *stop > *start && *stop - *start <= *step;

  Value logicalStop = stripScalarIdentity(range.getLogicalStop());
  auto add = logicalStop.getDefiningOp<BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  return (sameScalarExpression(add.getLhs(), range.getLogicalStart()) &&
          sameScalarExpression(add.getRhs(), range.getStep())) ||
         (sameScalarExpression(add.getRhs(), range.getLogicalStart()) &&
          sameScalarExpression(add.getLhs(), range.getStep()));
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

std::optional<BinaryOperator> queryBinaryCombineKind(Region &region) {
  if (!llvm::hasSingleElement(region))
    return std::nullopt;
  Block &block = region.front();
  if (block.getNumArguments() != 2)
    return std::nullopt;
  auto yield = dyn_cast_or_null<YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return std::nullopt;
  Value result = stripCombineProjection(yield.getValues().front());
  auto binary = result.getDefiningOp<BinaryOp>();
  if (!binary)
    return std::nullopt;
  Value lhs = stripCombineProjection(binary.getLhs());
  Value rhs = stripCombineProjection(binary.getRhs());
  if (!((lhs == block.getArgument(0) && rhs == block.getArgument(1)) ||
        (lhs == block.getArgument(1) && rhs == block.getArgument(0))))
    return std::nullopt;
  return binary.getOperatorKind();
}

bool isPhysicalReplayNode(Operation *operation, PhysicalReplayScope scope,
                          bool allowAccesses) {
  if (!operation || operation->getNumResults() == 0)
    return false;
  if (isAccessNode(operation))
    return allowAccesses && operation->getNumRegions() == 0;
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation))
    return false;
  return scope == PhysicalReplayScope::Coordinate
             ? isCoordinateReplayNode(operation)
             : isValueReplayNode(operation);
}

bool typeCarriesTraversal(Type type, PhysicalSourceAxis source,
                          int64_t dimension) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return llvm::any_of(queryFragmentAxes(fragment, source),
                        [&](const PhysicalAxisProjection &projection) {
                          return projection.dimensionId == dimension;
                        });
  if (auto record = dyn_cast<RecordType>(type))
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return typeCarriesTraversal(cast<TypeAttr>(field).getValue(), source,
                                  dimension);
    });
  return false;
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

FailureOr<int64_t> querySubregionParentDimension(MakeRangeOp range) {
  if (!range)
    return failure();
  auto parent = range->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
  return parent && parent.getInt() > 0
             ? FailureOr<int64_t>(parent.getInt())
             : FailureOr<int64_t>(failure());
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

SmallVector<PhysicalAxisProjection, 2>
queryRangeProjections(Type type, MakeRangeOp range) {
  SmallVector<PhysicalAxisProjection, 2> sourceResults =
      queryFragmentAxes(type, sourceAxisIdentity(range));
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  if (succeeded(dimension)) {
    SmallVector<PhysicalAxisProjection, 2> dimensionResults = sourceResults;
    llvm::erase_if(dimensionResults,
                   [&](const PhysicalAxisProjection &projection) {
      return projection.dimensionId != *dimension;
    });
    if (!dimensionResults.empty())
      return dimensionResults;
    PhysicalDimensionProjection projection =
        queryFragmentDimension(type, *dimension);
    if (projection.isExact())
      return {PhysicalAxisProjection{
          PhysicalFactState::Exact, sourceAxisIdentity(range), *dimension,
          projection.fragmentAxis}};
  }
  return sourceResults.size() == 1
             ? sourceResults
             : SmallVector<PhysicalAxisProjection, 2>{};
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

SmallVector<PhysicalDimensionProjection, 2>
queryFragmentDimensions(Type type, int64_t dimensionId) {
  SmallVector<PhysicalDimensionProjection, 2> results;
  if (dimensionId <= 0)
    return results;
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return results;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getDimensionId() != dimensionId)
      continue;
    results.push_back(PhysicalDimensionProjection{
        PhysicalFactState::Exact, dimensionId, mapping.getFragmentAxis()});
  }
  return results;
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
    unsigned index = argument.getArgNumber();
    if (block == &fold.getSummarize().front()) {
      if (index < fold.getSourceCount())
        sources.push_back(fold.getInputs()[index]);
      else {
        unsigned capture = index - fold.getSourceCount();
        unsigned input = fold.getSourceCount() + fold.getIdentityCount() +
                         capture;
        if (input < fold.getInputs().size())
          sources.push_back(fold.getInputs()[input]);
      }
      return sources;
    }
    if (block == &fold.getCombine().front() && fold.getIdentityCount() > 0) {
      unsigned component = index % fold.getIdentityCount();
      unsigned identity = fold.getSourceCount() + component;
      if (identity < fold.getInputs().size())
        sources.push_back(fold.getInputs()[identity]);
      if (auto yield =
              dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
          yield && component < yield.getValues().size())
        sources.push_back(yield.getValues()[component]);
    }
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
  // Reshape preserves the row-major coordinate relation carried by its
  // reassociation groups.  Source-specific range queries follow that typed
  // relation through the input instead of treating reshape as an opaque value
  // producer.  Any genuinely incompatible roots remain ambiguous in
  // sourceRanges(), where all collected ranges are compared.
  if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    collectRanges(reshape.getValue(), source, result, visited);
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
  else if (result.roots.size() > 1) {
    MakeRangeOp authority = result.roots.front();
    if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      result.state = PhysicalFactState::Ambiguous;
  }
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
  if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
    if (!input) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    unsigned logicalSourceRank = 0;
    unsigned logicalResultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = dyn_cast<ReshapeGroupAttr>(attribute);
      if (!group) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, operation);
        return;
      }
      if (!group.getSourceAxes().empty())
        logicalSourceRank =
            std::max(logicalSourceRank,
                     static_cast<unsigned>(
                         group.getSourceAxes().asArrayRef().back() + 1));
      if (!group.getResultAxes().empty())
        logicalResultRank =
            std::max(logicalResultRank,
                     static_cast<unsigned>(
                         group.getResultAxes().asArrayRef().back() + 1));
    }
    if (logicalSourceRank > input.getShape().size() ||
        logicalResultRank > fragment.getShape().size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    unsigned sourcePrefix = input.getShape().size() - logicalSourceRank;
    unsigned resultPrefix = fragment.getShape().size() - logicalResultRank;
    if (fragmentAxis < resultPrefix) {
      if (sourcePrefix != resultPrefix || fragmentAxis >= sourcePrefix) {
        result.state = PhysicalFactState::Unknown;
        appendUnique(result.blockers, operation);
        return;
      }
      collectAxisRanges(reshape.getValue(), fragmentAxis, result, visited);
      return;
    }
    unsigned logicalResultAxis = fragmentAxis - resultPrefix;
    auto expected =
        cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
    std::optional<unsigned> sourceAxis;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      if (!llvm::is_contained(group.getResultAxes().asArrayRef(),
                              logicalResultAxis))
        continue;
      for (int64_t logicalSourceAxis : group.getSourceAxes().asArrayRef()) {
        if (logicalSourceAxis < 0 ||
            sourcePrefix + static_cast<unsigned>(logicalSourceAxis) >=
                input.getShape().size())
          continue;
        unsigned physicalSourceAxis = sourcePrefix + logicalSourceAxis;
        auto mapping =
            cast<AxisMapAttr>(input.getAxisMaps()[physicalSourceAxis]);
        if (mapping.getDimensionId() != expected.getDimensionId())
          continue;
        if (sourceAxis) {
          result.state = PhysicalFactState::Ambiguous;
          appendUnique(result.blockers, operation);
          return;
        }
        sourceAxis = physicalSourceAxis;
      }
      break;
    }
    if (sourceAxis) {
      collectAxisRanges(reshape.getValue(), *sourceAxis, result, visited);
      return;
    }
    auto extent =
        cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
    if (extent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        extent.getValue() == 1)
      return;
    result.state = PhysicalFactState::Unknown;
    appendUnique(result.blockers, operation);
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
    bool followed = false;
    llvm::SmallPtrSet<Value, 8> visitedRecords;
    std::function<void(Value)> collectRecordField = [&](Value recordValue) {
      if (!recordValue || !visitedRecords.insert(recordValue).second)
        return;
      if (auto record = recordValue.getDefiningOp<MakeRecordOp>()) {
        if (extract.getField() >= record.getFields().size())
          return;
        Value field = record.getFields()[extract.getField()];
        auto type = dyn_cast<FragmentType>(field.getType());
        if (!type || fragmentAxis >= type.getShape().size())
          return;
        followed = true;
        collectAxisRanges(field, fragmentAxis, result, visited);
        return;
      }
      if (auto argument = dyn_cast<BlockArgument>(recordValue)) {
        for (Value related : structuredSourcesForArgument(argument))
          collectRecordField(related);
        return;
      }
      auto opResult = dyn_cast<OpResult>(recordValue);
      if (!opResult)
        return;
      if (auto fold = dyn_cast<RegionFoldOp>(opResult.getOwner())) {
        unsigned component = opResult.getResultNumber();
        if (component >= fold.getIdentityCount())
          return;
        unsigned identity = fold.getSourceCount() + component;
        if (identity < fold.getInputs().size())
          collectRecordField(fold.getInputs()[identity]);
        if (auto yield =
                dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
            yield && component < yield.getValues().size())
          collectRecordField(yield.getValues()[component]);
        return;
      }
      if (auto loop = dyn_cast<scf::ForOp>(opResult.getOwner())) {
        unsigned component = opResult.getResultNumber();
        if (component < loop.getInitArgs().size())
          collectRecordField(loop.getInitArgs()[component]);
        if (auto yield =
                dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
            yield && component < yield.getResults().size())
          collectRecordField(yield.getResults()[component]);
      }
    };
    collectRecordField(extract.getRecord());
    if (!followed)
      result.state = PhysicalFactState::Unknown;
    return;
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
        inputExtent.getValue() == 1 && inputExtent != outputExtent) {
      auto inputMapping = cast<AxisMapAttr>(input.getAxisMaps()[inputAxis]);
      auto outputMapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
      if (!(sourceAxisIdentity(inputMapping) ==
                sourceAxisIdentity(outputMapping)) ||
          inputMapping.getDimensionId() != outputMapping.getDimensionId())
        return;
    }
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
    using CoordinateOccurrence = std::pair<Value, unsigned>;
    enum class OccurrencePriority {
      SourceAndDimension,
      Dimension,
      UniqueSource,
    };
    auto selectOccurrence = [&](OccurrencePriority priority)
        -> SmallVector<CoordinateOccurrence, 2> {
      SmallVector<CoordinateOccurrence, 2> occurrences;
      for (Value coordinate : load.getCoordinates()) {
        auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
        if (!coordinateType)
          continue;
        SmallVector<unsigned, 2> axes;
        if (priority == OccurrencePriority::Dimension) {
          for (PhysicalDimensionProjection projection : queryFragmentDimensions(
                   coordinateType, expected.getDimensionId()))
            axes.push_back(projection.fragmentAxis);
        } else {
          SmallVector<PhysicalAxisProjection, 2> projections =
              queryFragmentAxes(coordinateType, sourceAxisIdentity(expected));
          if (priority == OccurrencePriority::SourceAndDimension)
            llvm::erase_if(projections,
                           [&](const PhysicalAxisProjection &projection) {
              return projection.dimensionId != expected.getDimensionId();
            });
          else if (projections.size() != 1)
            projections.clear();
          for (PhysicalAxisProjection projection : projections)
            axes.push_back(projection.fragmentAxis);
        }
        SmallVector<unsigned, 2> sameOccurrence;
        llvm::copy_if(axes, std::back_inserter(sameOccurrence),
                      [&](unsigned axis) { return axis == fragmentAxis; });
        if (sameOccurrence.size() == 1)
          axes = std::move(sameOccurrence);
        for (unsigned axis : axes)
          occurrences.emplace_back(coordinate, axis);
      }
      return occurrences;
    };
    SmallVector<CoordinateOccurrence, 2> occurrences =
        selectOccurrence(OccurrencePriority::SourceAndDimension);
    if (occurrences.empty())
      occurrences = selectOccurrence(OccurrencePriority::Dimension);
    if (occurrences.empty())
      occurrences = selectOccurrence(OccurrencePriority::UniqueSource);
    if (occurrences.empty()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    for (const CoordinateOccurrence &occurrence : occurrences) {
      PhysicalRangeFact nested = axisRanges(occurrence.first, occurrence.second);
      if (nested.state == PhysicalFactState::Unknown ||
          !nested.blockers.empty()) {
        result.state = PhysicalFactState::Unknown;
        for (Operation *blocker : nested.blockers)
          appendUnique(result.blockers, blocker);
        continue;
      }
      for (MakeRangeOp range : nested.roots)
        appendUnique(result.roots, range);
      for (Operation *access : nested.accesses)
        appendUnique(result.accesses, access);
    }
    if (result.state != PhysicalFactState::Unknown && !result.roots.empty()) {
      MakeRangeOp authority = result.roots.front();
      if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
            return sameLogicalRange(authority, range);
          })) {
        result.state = PhysicalFactState::Ambiguous;
        appendUnique(result.blockers, operation);
      }
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
  if (auto contract = dyn_cast<ContractOp>(operation)) {
    llvm::SmallDenseSet<int64_t> lhsReduced(
        contract.getLhsReductionAxes().begin(),
        contract.getLhsReductionAxes().end());
    llvm::SmallDenseSet<int64_t> rhsReduced(
        contract.getRhsReductionAxes().begin(),
        contract.getRhsReductionAxes().end());
    llvm::SmallDenseSet<int64_t> rhsBatched(
        contract.getRhsBatchAxes().begin(), contract.getRhsBatchAxes().end());
    SmallVector<std::pair<Value, unsigned>> resultSources;
    auto lhs = dyn_cast<FragmentType>(contract.getLhs().getType());
    auto rhs = dyn_cast<FragmentType>(contract.getRhs().getType());
    if (!lhs || !rhs) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    for (unsigned axis = 0; axis < lhs.getShape().size(); ++axis)
      if (!lhsReduced.contains(axis))
        resultSources.emplace_back(contract.getLhs(), axis);
    for (unsigned axis = 0; axis < rhs.getShape().size(); ++axis)
      if (!rhsReduced.contains(axis) && !rhsBatched.contains(axis))
        resultSources.emplace_back(contract.getRhs(), axis);
    if (fragmentAxis >= resultSources.size()) {
      result.state = PhysicalFactState::Unknown;
      appendUnique(result.blockers, operation);
      return;
    }
    collectAxisRanges(resultSources[fragmentAxis].first,
                      resultSources[fragmentAxis].second, result, visited);
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
  if (result.state == PhysicalFactState::Unknown || !result.blockers.empty())
    result.state = PhysicalFactState::Unknown;
  else if (result.roots.size() > 1) {
    MakeRangeOp authority = result.roots.front();
    if (!llvm::all_of(result.roots, [&](MakeRangeOp range) {
          return sameLogicalRange(authority, range);
        }))
      result.state = PhysicalFactState::Ambiguous;
  }
  result.unitStep = !result.roots.empty() &&
                    llvm::all_of(result.roots, isUnitStepRange);
  return result;
}

PhysicalAxisRealizationFact
PhysicalProgramAnalysis::axisRealization(Value value, unsigned fragmentAxis) {
  PhysicalAxisRealizationFact result;
  result.fragmentAxis = fragmentAxis;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size() ||
      fragmentAxis >= fragment.getAxisMaps().size())
    return result;

  auto mapping = dyn_cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);
  if (!mapping || !extent)
    return result;
  result.source = sourceAxisIdentity(mapping);
  result.dimensionId = mapping.getDimensionId();

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    auto isSegmentSource = [&](uint64_t sourceCount, uint64_t axis,
                               Block &region) {
      return argument.getOwner() == &region &&
             argument.getArgNumber() < sourceCount && fragmentAxis == axis;
    };
    bool segmentSource = false;
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(owner))
      segmentSource = isSegmentSource(fold.getSourceCount(), fold.getAxis(),
                                      fold.getSummarize().front());
    else if (auto scan = dyn_cast_or_null<RegionScanOp>(owner))
      segmentSource =
          isSegmentSource(scan.getSourceCount(), scan.getAxis(),
                          scan.getSummarize().front()) ||
          isSegmentSource(scan.getSourceCount(), scan.getAxis(),
                          scan.getEmit().front());
    if (segmentSource) {
      // RegionFoldOp/RegionScanOp verification binds this exact block argument
      // axis to the operation's segment parameter.  The slice extent is a
      // first-class physical relation, not something to rediscover from the
      // unsliced outer source range.
      result.state = PhysicalFactState::Exact;
      result.extentAuthority =
          PhysicalAxisRealizationFact::ExtentAuthority::Structural;
      return result;
    }
  }

  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (source) {
      BroadcastProjection projection = queryAxisProjection(source, fragment);
      if (projection.isExact() &&
          fragmentAxis < projection.targetToSource.size())
        if (std::optional<unsigned> sourceAxis =
                projection.targetToSource[fragmentAxis];
            sourceAxis && source.getShape()[*sourceAxis] == extent) {
          PhysicalAxisRealizationFact input =
              axisRealization(broadcast.getValue(), *sourceAxis);
          if (input.hasExtentAuthority()) {
            result.state = PhysicalFactState::Exact;
            result.physicalized = input.physicalized;
            result.extentAuthority =
                PhysicalAxisRealizationFact::ExtentAuthority::Structural;
            return result;
          }
        }
    }
  }

  if (auto reduce = value.getDefiningOp<ReduceOp>()) {
    auto opResult = dyn_cast<OpResult>(value);
    if (opResult && opResult.getResultNumber() < reduce.getSourceCount()) {
      Value sourceValue = reduce.getInputs()[opResult.getResultNumber()];
      auto source = dyn_cast<FragmentType>(sourceValue.getType());
      if (source) {
        llvm::SmallDenseSet<int64_t> reduced(reduce.getAxes().begin(),
                                             reduce.getAxes().end());
        SmallVector<unsigned> freeAxes;
        for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
          if (!reduced.contains(axis))
            freeAxes.push_back(axis);
        if (fragmentAxis < freeAxes.size()) {
          unsigned sourceAxis = freeAxes[fragmentAxis];
          PhysicalAxisRealizationFact input =
              axisRealization(sourceValue, sourceAxis);
          if (input.hasExtentAuthority() &&
              source.getShape()[sourceAxis] == extent) {
            result.state = PhysicalFactState::Exact;
            result.physicalized = input.physicalized;
            result.extentAuthority =
                PhysicalAxisRealizationFact::ExtentAuthority::Structural;
            return result;
          }
        }
      }
    }
  }

  if (auto contract = value.getDefiningOp<ContractOp>()) {
    llvm::SmallDenseSet<int64_t> lhsReduced(
        contract.getLhsReductionAxes().begin(),
        contract.getLhsReductionAxes().end());
    llvm::SmallDenseSet<int64_t> rhsReduced(
        contract.getRhsReductionAxes().begin(),
        contract.getRhsReductionAxes().end());
    llvm::SmallDenseSet<int64_t> rhsBatched(
        contract.getRhsBatchAxes().begin(), contract.getRhsBatchAxes().end());
    SmallVector<std::pair<Value, unsigned>> resultSources;
    auto lhs = dyn_cast<FragmentType>(contract.getLhs().getType());
    auto rhs = dyn_cast<FragmentType>(contract.getRhs().getType());
    if (lhs && rhs) {
      for (unsigned axis = 0; axis < lhs.getShape().size(); ++axis)
        if (!lhsReduced.contains(axis))
          resultSources.emplace_back(contract.getLhs(), axis);
      for (unsigned axis = 0; axis < rhs.getShape().size(); ++axis)
        if (!rhsReduced.contains(axis) && !rhsBatched.contains(axis))
          resultSources.emplace_back(contract.getRhs(), axis);
      if (fragmentAxis < resultSources.size()) {
        Value sourceValue = resultSources[fragmentAxis].first;
        unsigned sourceAxis = resultSources[fragmentAxis].second;
        auto source = cast<FragmentType>(sourceValue.getType());
        PhysicalAxisRealizationFact input =
            axisRealization(sourceValue, sourceAxis);
        if (input.hasExtentAuthority() &&
            source.getShape()[sourceAxis] == extent) {
          result.state = PhysicalFactState::Exact;
          result.physicalized = input.physicalized;
          result.extentAuthority =
              PhysicalAxisRealizationFact::ExtentAuthority::Structural;
          return result;
        }
      }
    }
  }

  if (value.getDefiningOp<ExtractOp>()) {
    // ExtractOp::verify requires the result type to equal the selected typed
    // record field.  MakeRecord and structured fold/scan verifiers in turn
    // require their field/result schemas to match the executable values at the
    // region boundary.  The projected extent is therefore already a current-IR
    // structural fact; treating it as unknown would make consumers rediscover
    // the record schema from nearby arithmetic.
    result.state = PhysicalFactState::Exact;
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Structural;
    return result;
  }

  PhysicalRangeFact ranges = axisRanges(value, fragmentAxis);
  result.roots.append(ranges.roots.begin(), ranges.roots.end());
  result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
  result.constructionScalarSeed =
      extent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Constant) &&
      extent.getValue() == 1 && !ranges.roots.empty() &&
      llvm::any_of(ranges.roots, [](MakeRangeOp range) {
        return !isProvablySingletonLogicalRange(range);
      });
  result.physicalized =
      !result.constructionScalarSeed && !ranges.roots.empty() &&
      llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
         return valueMatchesExtent(range.getExtent(), extent);
       });
  if (value.getDefiningOp<ReshapeOp>()) {
    // A verified reshape carries its own row-major physical reassociation.  Its
    // result extent remains exact even when no single pre-reshape range can be
    // projected to one split/merged result axis.  Keep that extent fact
    // separate from range provenance rather than turning a legal reshape into
    // analysis unknown.
    result.state = PhysicalFactState::Exact;
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Structural;
    return result;
  }
  if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
    return result;
  if (!ranges.roots.empty() && failed(queryExactLogicalRange(ranges))) {
    result.state = PhysicalFactState::Ambiguous;
    return result;
  }

  result.state = PhysicalFactState::Exact;
  if (result.physicalized)
    result.extentAuthority =
        PhysicalAxisRealizationFact::ExtentAuthority::Range;
  return result;
}

PhysicalRangeAxisFact
PhysicalProgramAnalysis::rangeAxes(Value value,
                                   ArrayRef<MakeRangeOp> selectedRoots) {
  PhysicalRangeAxisFact result;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || selectedRoots.empty())
    return result;
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    (void)splat;
    result.state = PhysicalFactState::Exact;
    return result;
  }
  if (auto broadcast = value.getDefiningOp<BroadcastOp>();
      broadcast && !isa<FragmentType>(broadcast.getValue().getType())) {
    result.state = PhysicalFactState::Exact;
    return result;
  }
  result.state = PhysicalFactState::Exact;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = axisRanges(value, axis);
    bool selected = llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
      return llvm::is_contained(selectedRoots, range);
    });
    if (selected) {
      if (failed(queryExactLogicalRange(ranges))) {
        result.state = PhysicalFactState::Ambiguous;
        result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
        return result;
      }
      result.fragmentAxes.push_back(axis);
      continue;
    }
    if (ranges.state != PhysicalFactState::Unknown)
      continue;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    bool mayCarrySelectedRoot = llvm::any_of(selectedRoots, [&](MakeRangeOp root) {
      FailureOr<int64_t> dimension = queryRangeDimension(root);
      return sourceAxisIdentity(mapping) == sourceAxisIdentity(root) &&
             succeeded(dimension) && mapping.getDimensionId() == *dimension;
    });
    if (!mayCarrySelectedRoot)
      continue;
    result.state = PhysicalFactState::Unknown;
    result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
    return result;
  }
  return result;
}

PhysicalLockstepTraversalFact PhysicalProgramAnalysis::lockstepTraversal(
    ValueRange sources, ArrayRef<unsigned> fragmentAxes) {
  PhysicalLockstepTraversalFact result;
  if (sources.empty() || sources.size() != fragmentAxes.size())
    return result;
  SmallVector<MakeRangeOp> authorities;
  for (auto [source, fragmentAxis] : llvm::zip(sources, fragmentAxes)) {
    PhysicalRangeFact ranges = axisRanges(source, fragmentAxis);
    PhysicalLockstepTraversalFact sourceFact = lockstepRanges(ranges.roots);
    if (!sourceFact.isExact()) {
      result.state = sourceFact.state;
      result.blockers.append(ranges.blockers.begin(), ranges.blockers.end());
      result.blockers.append(sourceFact.blockers.begin(),
                             sourceFact.blockers.end());
      return result;
    }
    authorities.push_back(sourceFact.authority);
  }
  return lockstepRanges(authorities);
}

PhysicalLockstepTraversalFact
PhysicalProgramAnalysis::lockstepRanges(ArrayRef<MakeRangeOp> ranges) {
  PhysicalLockstepTraversalFact result;
  if (ranges.empty())
    return result;
  auto sameLogicalTraversal = [](MakeRangeOp lhs, MakeRangeOp rhs) {
    FailureOr<int64_t> lhsDimension = queryRangeDimension(lhs);
    FailureOr<int64_t> rhsDimension = queryRangeDimension(rhs);
    if (!lhs->hasAttr(sourceSubregionAttr) &&
        !rhs->hasAttr(sourceSubregionAttr) && succeeded(lhsDimension) &&
        succeeded(rhsDimension) && *lhsDimension == *rhsDimension)
      return true;
    return samePhysicalScalarExpression(lhs.getLogicalStart(),
                                        rhs.getLogicalStart()) &&
           samePhysicalScalarExpression(lhs.getLogicalStop(),
                                        rhs.getLogicalStop());
  };
  auto sameTraversal = [](MakeRangeOp lhs, MakeRangeOp rhs) {
    return samePhysicalScalarExpression(lhs.getStart(), rhs.getStart()) &&
           samePhysicalScalarExpression(lhs.getExtent(), rhs.getExtent()) &&
           samePhysicalScalarExpression(lhs.getStep(), rhs.getStep());
  };
  result.authority = ranges.front();
  for (MakeRangeOp range : llvm::drop_begin(ranges))
    if (!sameTraversal(result.authority, range) ||
        !sameLogicalTraversal(result.authority, range)) {
      result.state = PhysicalLockstepState::Inconsistent;
      return result;
    }
  result.state = PhysicalLockstepState::Exact;
  return result;
}

void PhysicalProgramAnalysis::analyzeReplay(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    Operation *insertionAnchor, std::optional<int64_t> sourceDimension,
    DominanceInfo *dominance,
    PhysicalReplayFact &result,
    SmallPtrSetImpl<Operation *> &visited) {
  if (!value)
    return;
  bool carriesRequestedTraversal = false;
  if (source) {
    if (sourceDimension) {
      carriesRequestedTraversal = llvm::any_of(
          queryFragmentAxes(value.getType(), *source),
          [&](const PhysicalAxisProjection &projection) {
            return projection.dimensionId == *sourceDimension;
          });
    } else {
      carriesRequestedTraversal = carriesSource(value.getType(), *source);
    }
  }
  if (insertionAnchor && dominance &&
      dominance->dominates(value, insertionAnchor) &&
      !carriesRequestedTraversal)
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
                    insertionAnchor, sourceDimension, dominance, result,
                    visited);
      return;
    }
  }
  bool physicalValue = isa<FragmentType, RecordType>(value.getType());
  if (!physicalValue && !insertionAnchor)
    return;
  if (source && physicalValue && !carriesSource(value.getType(), *source) &&
      !insertionAnchor)
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
      analyzeReplay(related, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, result, visited);
    return;
  }
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (insertionAnchor && source && sourceDimension) {
      FailureOr<int64_t> dimension = queryRangeDimension(range);
      if (!(sourceAxisIdentity(range) == *source) || failed(dimension) ||
          *dimension != *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
      }
    }
    return;
  }
  if (!physicalValue) {
    if (isa<arith::ConstantOp>(operation))
      return;
    if (!isPhysicalReplayNode(operation, scope, allowAccesses) ||
        operation->getNumRegions() != 0 || operation->getNumResults() != 1) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value operand : operation->getOperands())
      analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                    sourceDimension, dominance, result, visited);
    return;
  }
  if (isAccessNode(operation)) {
    result.crossesAccess = true;
    appendUnique(result.accesses, operation);
    if (!allowAccesses) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
  } else if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
    if (!source || !sourceDimension) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value segmentSource :
         fold.getInputs().take_front(fold.getSourceCount())) {
      auto fragment = dyn_cast<FragmentType>(segmentSource.getType());
      if (!fragment || fold.getAxis() >= fragment.getAxisMaps().size()) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
      auto mapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[fold.getAxis()]);
      if (sourceAxisIdentity(mapping) == *source &&
          mapping.getDimensionId() == *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : fold.getInputs())
      if (typeCarriesTraversal(operand.getType(), *source, *sourceDimension))
        analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                      sourceDimension, dominance, result, visited);
    return;
  } else if (auto scan = dyn_cast<RegionScanOp>(operation)) {
    if (!source || !sourceDimension) {
      appendUnique(result.blockers, operation);
      result.state = PhysicalFactState::Unknown;
      return;
    }
    for (Value segmentSource :
         scan.getInputs().take_front(scan.getSourceCount())) {
      auto fragment = dyn_cast<FragmentType>(segmentSource.getType());
      if (!fragment || scan.getAxis() >= fragment.getAxisMaps().size()) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
      auto mapping =
          cast<AxisMapAttr>(fragment.getAxisMaps()[scan.getAxis()]);
      if (sourceAxisIdentity(mapping) == *source &&
          mapping.getDimensionId() == *sourceDimension) {
        appendUnique(result.blockers, operation);
        result.state = PhysicalFactState::Unknown;
        return;
      }
    }
    for (Value operand : scan.getInputs())
      if (typeCarriesTraversal(operand.getType(), *source, *sourceDimension))
        analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                      sourceDimension, dominance, result, visited);
    return;
  } else if (!isPhysicalReplayNode(operation, scope, allowAccesses)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  Region *combine = nullptr;
  if (auto reduce = dyn_cast<ReduceOp>(operation))
    combine = &reduce.getCombine();
  else if (auto scan = dyn_cast<ScanOp>(operation))
    combine = &scan.getCombine();
  if (combine) {
    WalkResult helper = combine->walk([&](Operation *nested) {
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
  if (operation->getNumRegions() != 0 && !isa<ReduceOp, ScanOp>(operation)) {
    appendUnique(result.blockers, operation);
    result.state = PhysicalFactState::Unknown;
    return;
  }
  for (Value operand : operation->getOperands())
    analyzeReplay(operand, source, scope, allowAccesses, insertionAnchor,
                  sourceDimension, dominance, result, visited);
}

PhysicalReplayFact PhysicalProgramAnalysis::replayability(
    Value value, std::optional<PhysicalSourceAxis> source,
    PhysicalReplayScope scope, bool allowAccesses,
    Operation *insertionAnchor,
    std::optional<int64_t> sourceDimension) {
  PhysicalReplayFact result;
  result.state = PhysicalFactState::Exact;
  SmallPtrSet<Operation *, 32> visited;
  std::optional<DominanceInfo> dominance;
  if (insertionAnchor)
    dominance.emplace(kernel);
  analyzeReplay(value, source, scope, allowAccesses, insertionAnchor,
                sourceDimension, dominance ? &*dominance : nullptr, result,
                visited);
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
    if (auto range = dyn_cast<MakeRangeOp>(operation)) {
      // Reaching a coordinate proves an ordinary value dependency, not a
      // reduction dependency.  Only an operation that removes or carries this
      // axis may turn the fact into `depends=true` below.  Treating every range
      // leaf as a reduction made pointwise ownership axes both program-mapped
      // and internally traversed, so different programs replayed overlapping
      // writeback domains.
      return exact;
    }
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
      return exact;
    }
    if (auto scan = dyn_cast<ScanOp>(operation)) {
      for (Value input : scan.getInputs().take_front(scan.getSourceCount()))
        if (reductionTypeConsumesSource(
                input.getType(),
                ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())},
                source)) {
          exact.depends = true;
          return exact;
        }
      return exact;
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
    if (comparison->hasAttr(physicalTailAttr) && predicateRange &&
        sameLogicalRange(predicateRange, expectedRange) &&
        sameScalarExpression(rhs, expectedEnd))
      return true;
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

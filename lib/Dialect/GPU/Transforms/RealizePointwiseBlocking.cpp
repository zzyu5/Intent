#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, sourceSubregionBoundAttr,
        worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

uint32_t physicalElementBitWidth(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    type = fragment.getElementType();
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto record = dyn_cast<RecordType>(type)) {
    uint32_t width = 0;
    for (Attribute field : record.getFieldTypes())
      width = std::max(
          width,
          physicalElementBitWidth(cast<TypeAttr>(field).getValue()));
    return width;
  }
  return isa<IntegerType, FloatType>(type)
             ? type.getIntOrFloatBitWidth()
             : 0;
}

void collectPhysicalDimensions(Type type,
                               llvm::SmallDenseSet<uint64_t> &dimensions) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (Attribute attribute : fragment.getAxisMaps()) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getDimensionId() > 0)
        dimensions.insert(mapping.getDimensionId());
    }
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    for (Attribute field : record.getFieldTypes())
      collectPhysicalDimensions(cast<TypeAttr>(field).getValue(), dimensions);
}

void collectPartiallyCarriedDimensions(
    Type type, llvm::SmallDenseSet<uint64_t> &dimensions) {
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return;

  llvm::DenseMap<uint64_t, unsigned> fieldCounts;
  for (Attribute field : record.getFieldTypes()) {
    Type fieldType = cast<TypeAttr>(field).getValue();
    llvm::SmallDenseSet<uint64_t> fieldDimensions;
    collectPhysicalDimensions(fieldType, fieldDimensions);
    for (uint64_t dimension : fieldDimensions)
      ++fieldCounts[dimension];
    collectPartiallyCarriedDimensions(fieldType, dimensions);
  }
  for (auto [dimension, count] : fieldCounts)
    if (count < record.getFieldTypes().size())
      dimensions.insert(dimension);
}

Attribute dimensionAxisKey(MLIRContext *context, uint64_t dimension) {
  return IntegerAttr::get(IntegerType::get(context, 64), dimension);
}

Attribute sourceAxisKey(MLIRContext *context, PhysicalSourceAxis source) {
  return PhysicalSourceAttr::get(context, source.sourceId, source.sourceAxis,
                                 source.derived);
}

bool isSourceAxisKey(Attribute axis) { return isa<PhysicalSourceAttr>(axis); }

FailureOr<uint64_t> axisDimension(Attribute axis) {
  auto dimension = dyn_cast<IntegerAttr>(axis);
  return dimension && dimension.getInt() > 0
             ? FailureOr<uint64_t>(dimension.getInt())
             : FailureOr<uint64_t>(failure());
}

FailureOr<PhysicalSourceAxis> axisSource(Attribute axis) {
  auto source = dyn_cast<PhysicalSourceAttr>(axis);
  return source ? FailureOr<PhysicalSourceAxis>(PhysicalSourceAxis{
                      source.getSourceId(), source.getSourceAxis(),
                      source.getDerived()})
                : FailureOr<PhysicalSourceAxis>(failure());
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {}) {
  return PhysicalExprAttr::get(context, static_cast<uint32_t>(kind), value,
                               StringAttr::get(context, symbol),
                               ArrayAttr::get(context, {}));
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(kind), 0, StringAttr::get(context),
      ArrayAttr::get(context, {lhs, rhs}));
}

PhysicalExprAttr launchRangeExtent(MakeRangeOp range) {
  if (auto count = constantLogicalRangeCardinality(range))
    return expression(range.getContext(), PhysicalExprKind::Constant, *count);
  PhysicalExprAttr start = queryLaunchExpression(range.getLogicalStart());
  PhysicalExprAttr stop = queryLaunchExpression(range.getLogicalStop());
  PhysicalExprAttr step = queryLaunchExpression(range.getStep());
  if (!start || !stop || !step)
    return {};
  if (step.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
      step.getValue() <= 0)
    return {};
  MLIRContext *context = range.getContext();
  PhysicalExprAttr distance = binaryExpression(
      context, PhysicalExprKind::Subtract, stop, start);
  distance = binaryExpression(
      context, PhysicalExprKind::Maximum, distance,
      expression(context, PhysicalExprKind::Constant, 0));
  return binaryExpression(context, PhysicalExprKind::CeilDiv, distance, step);
}

bool hasCompileTimeExtent(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() ||
         value.getDefiningOp<ParameterOp>() ||
         value.getDefiningOp<PhysicalExprOp>();
}

FailureOr<Value> dimensionArgument(func::FuncOp kernel, uint64_t dimension) {
  for (BlockArgument argument : kernel.getArguments()) {
    auto attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == static_cast<int64_t>(dimension))
      return Value(argument);
  }
  return failure();
}

FailureOr<uint64_t> rangeDimension(MakeRangeOp range) {
  FailureOr<int64_t> dimension = querySourceDimension(
      range.getResult().getType(),
      PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()});
  return succeeded(dimension) && *dimension > 0
             ? FailureOr<uint64_t>(*dimension)
             : FailureOr<uint64_t>(failure());
}

bool hasAccessDependentSubregionBounds(func::FuncOp kernel, MakeRangeOp range) {
  if (!range->hasAttr(sourceSubregionAttr))
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value bound : {range.getLogicalStart(), range.getLogicalStop()}) {
    PhysicalRangeFact provenance = analysis.sourceRanges(bound);
    if (!provenance.accesses.empty())
      return true;
  }
  return false;
}

FailureOr<uint64_t> ownershipDimension(func::FuncOp kernel,
                                       MakeRangeOp range) {
  FailureOr<int64_t> parent = querySubregionParentDimension(range);
  if (succeeded(parent) && !hasAccessDependentSubregionBounds(kernel, range))
    return static_cast<uint64_t>(*parent);
  return rangeDimension(range);
}

FailureOr<ParameterOp> queryOwnershipBlockingParameter(func::FuncOp kernel,
                                                       MakeRangeOp range) {
  FailureOr<ParameterOp> direct = queryBlockingParameter(kernel, range);
  if (succeeded(direct))
    return *direct;
  FailureOr<int64_t> parent = querySubregionParentDimension(range);
  if (failed(parent))
    return failure();
  ParameterOp result;
  bool ambiguous = false;
  kernel.walk([&](ParameterOp parameter) {
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.dimension ||
        *binding.dimension != *parent ||
        parameter.getParameter().getRole() !=
            static_cast<uint32_t>(ParameterRole::OwnershipN))
      return;
    if (result && result != parameter)
      ambiguous = true;
    else
      result = parameter;
  });
  return result && !ambiguous ? FailureOr<ParameterOp>(result)
                              : FailureOr<ParameterOp>(failure());
}

FailureOr<Attribute> parameterAxis(ParameterOp parameter) {
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (!binding.isExact())
    return failure();
  if (binding.source)
    return sourceAxisKey(parameter.getContext(), *binding.source);
  if (binding.dimension)
    return dimensionAxisKey(parameter.getContext(), *binding.dimension);
  return failure();
}

PhysicalExprAttr fragmentExtent(ParameterOp parameter) {
  ParameterAttr schema = parameter.getParameter();
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (binding.source && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    schema.getName().getValue());
}

struct ProductConstraint {
  int64_t constant = 1;
  StringAttr parameter;
  unsigned parameterCount = 0;
};

bool collectProductConstraint(PhysicalExprAttr extent,
                              ProductConstraint &constraint) {
  auto kind = static_cast<PhysicalExprKind>(extent.getKind());
  if (kind == PhysicalExprKind::Constant) {
    if (extent.getValue() <= 0)
      return false;
    if (constraint.constant >
        std::numeric_limits<int64_t>::max() / extent.getValue())
      return false;
    constraint.constant *= extent.getValue();
    return true;
  }
  if (kind == PhysicalExprKind::Parameter) {
    ++constraint.parameterCount;
    if (constraint.parameter && constraint.parameter != extent.getSymbol())
      return false;
    constraint.parameter = extent.getSymbol();
    return true;
  }
  if (kind != PhysicalExprKind::Multiply || extent.getOperands().size() != 2)
    return false;
  return collectProductConstraint(
             cast<PhysicalExprAttr>(extent.getOperands()[0]), constraint) &&
         collectProductConstraint(
             cast<PhysicalExprAttr>(extent.getOperands()[1]), constraint);
}

bool collectProductConstraint(FragmentType fragment, ArrayRef<int64_t> axes,
                              ProductConstraint &constraint) {
  for (int64_t axis : axes)
    if (!collectProductConstraint(
            cast<PhysicalExprAttr>(fragment.getShape()[axis]), constraint))
      return false;
  return true;
}

LogicalResult bindStructurallyRequiredStaticFragments(func::FuncOp kernel) {
  llvm::MapVector<ParameterOp, int64_t> required;
  auto requireExtent = [&](Operation *operation, StringAttr symbol,
                           int64_t candidate) -> LogicalResult {
    FailureOr<ParameterOp> declaration = queryParameterBySymbol(kernel, symbol);
    if (failed(declaration))
      return success();
    ParameterOp parameter = *declaration;
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.source)
      return success();
    if (!llvm::is_contained(
            parameter.getParameter().getCandidates().asArrayRef(), candidate))
      return operation->emitOpError(
          "reshape requires a static fragment extent outside its legal domain");
    auto found = required.find(parameter);
    if (found != required.end() && found->second != candidate)
      return parameter.emitOpError(
          "one static fragment has incompatible structural extent requirements");
    required[parameter] = candidate;
    return success();
  };
  SmallVector<ReshapeOp> reshapes;
  SmallVector<std::pair<Value, unsigned>> fixedAxes;
  kernel.walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
  for (ReshapeOp reshape : reshapes) {
    auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
    auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
    if (!source || !result)
      continue;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      auto sourceAxes = group.getSourceAxes().asArrayRef();
      auto resultAxes = group.getResultAxes().asArrayRef();
      // Unmerged axes carry their tile extent through the reshape. They do
      // not become full-coverage dimensions because other axes are reshaped.
      if (sourceAxes.size() <= 1 && resultAxes.size() <= 1)
        continue;
      ProductConstraint sourceProduct;
      ProductConstraint resultProduct;
      if (!collectProductConstraint(source, sourceAxes, sourceProduct) ||
          !collectProductConstraint(result, resultAxes, resultProduct))
        continue;
      if (sourceAxes.size() > 1 && resultAxes.size() == 1 &&
          sourceProduct.parameterCount == 0 &&
          resultProduct.parameterCount == 0 && resultProduct.constant > 1 &&
          sourceProduct.constant == resultProduct.constant)
        fixedAxes.emplace_back(reshape.getResult(), resultAxes.front());
      if (sourceAxes.size() == 1 && resultAxes.size() > 1 &&
          sourceProduct.parameterCount == 0 && resultProduct.parameterCount == 0 &&
          sourceProduct.constant == resultProduct.constant)
        for (int64_t axis : resultAxes)
          fixedAxes.emplace_back(reshape.getResult(), axis);
      if (sourceProduct.parameterCount != 1 ||
          resultProduct.parameterCount != 0 ||
          resultProduct.constant % sourceProduct.constant != 0)
        continue;
      int64_t candidate = resultProduct.constant / sourceProduct.constant;
      if (failed(requireExtent(reshape, sourceProduct.parameter, candidate)))
        return failure();
    }
  }

  // A retained static merge fixes its lane count through explicit pointwise
  // and one-to-one reshape relations, including positional source rebinding.
  // Propagate that structural constraint, not equality of unrelated shapes.
  for (auto [seed, seedAxis] : fixedAxes) {
    int64_t extent = cast<PhysicalExprAttr>(
        cast<FragmentType>(seed.getType()).getShape()[seedAxis]).getValue();
    SmallVector<std::pair<Value, unsigned>> pending{{seed, seedAxis}};
    llvm::DenseSet<std::pair<Value, unsigned>> visited;
    while (!pending.empty()) {
      auto [value, axis] = pending.pop_back_val();
      if (!visited.insert({value, axis}).second)
        continue;
      auto source = cast<FragmentType>(value.getType());
      for (Operation *user : value.getUsers()) {
        if (user->getNumResults() != 1)
          continue;
        auto result = dyn_cast<FragmentType>(user->getResult(0).getType());
        if (!result)
          continue;
        std::optional<unsigned> resultAxis;
        if (auto reshape = dyn_cast<ReshapeOp>(user)) {
          for (Attribute attribute : reshape.getReassociation()) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            if (group.getSourceAxes().size() == 1 &&
                group.getResultAxes().size() == 1 &&
                group.getSourceAxes()[0] == static_cast<int64_t>(axis))
              resultAxis = group.getResultAxes()[0];
          }
        } else if (isa<BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
                       CastOp, BitcastOp>(user)) {
          BroadcastProjection projection = queryAxisProjection(source, result);
          if (projection.isExact())
            for (auto [position, inputAxis] :
                 llvm::enumerate(projection.targetToSource))
              if (inputAxis && *inputAxis == axis)
                resultAxis = position;
        }
        if (!resultAxis)
          continue;
        auto constrain = [&](PhysicalExprAttr expression,
                             bool mayBroadcast = false) -> LogicalResult {
          if (expression.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Parameter))
            return requireExtent(user, expression.getSymbol(), extent);
          if (expression.getKind() ==
                  static_cast<uint32_t>(PhysicalExprKind::Constant) &&
              expression.getValue() != extent &&
              !(mayBroadcast && expression.getValue() == 1))
            return user->emitOpError(
                "pointwise relation conflicts with a static reshape extent");
          return success();
        };
        if (failed(constrain(
                cast<PhysicalExprAttr>(result.getShape()[*resultAxis]))))
          return failure();
        if (!isa<ReshapeOp>(user))
          for (Value operand : user->getOperands()) {
            auto input = dyn_cast<FragmentType>(operand.getType());
            if (!input)
              continue;
            BroadcastProjection projection = queryAxisProjection(input, result);
            if (projection.isExact() && projection.targetToSource[*resultAxis] &&
                failed(constrain(cast<PhysicalExprAttr>(input.getShape()[
                    *projection.targetToSource[*resultAxis]]), true)))
              return failure();
          }
        pending.emplace_back(user->getResult(0), *resultAxis);
      }
    }
  }

  for (auto [parameter, candidate] : required) {
    StringAttr parameterName = parameter.getParameter().getName();
    PhysicalExprAttr fixedExtent =
        expression(kernel.getContext(), PhysicalExprKind::Constant, candidate);
    // A parameter binding applies to every typed occurrence of its symbol,
    // including values reached through positional axis rebinding.
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](PhysicalExprAttr current) -> std::optional<Attribute> {
          if (current.getKind() ==
                  static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
              current.getSymbol() == parameterName)
            return fixedExtent;
          return std::nullopt;
        });
    replacer.recursivelyReplaceElementsIn(kernel.getOperation(),
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
    OpBuilder builder(parameter);
    Value fixed =
        builder.create<arith::ConstantIndexOp>(parameter.getLoc(), candidate);
    parameter.getResult().replaceAllUsesWith(fixed);
    parameter.erase();
  }
  return success();
}

bool hasExactStaticFullCoverage(func::FuncOp kernel, Value source,
                                uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return false;
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  if (!extent || extent.getKind() !=
                     static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      extent.getValue() <= 0)
    return false;
  PhysicalAxisRealizationFact fact =
      PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
  if (!fact.isExact() || !fact.physicalized || fact.constructionScalarSeed ||
      fact.roots.empty())
    return false;
  return llvm::all_of(fact.roots, [&](MakeRangeOp range) {
    if (!samePhysicalScalarExpression(range.getStart(),
                                      range.getLogicalStart()))
      return false;
    auto start = range.getLogicalStart().getDefiningOp<arith::ConstantIndexOp>();
    auto stop = range.getLogicalStop().getDefiningOp<arith::ConstantIndexOp>();
    auto step = range.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!step || step.value() <= 0)
      return false;
    if (start && stop)
      return stop.value() >= start.value() &&
             static_cast<__int128>(extent.getValue()) * step.value() >=
                 static_cast<__int128>(stop.value()) - start.value();
    // A nonnegative dynamic start needs at most the bounded stop's lanes.
    // The original tail predicate still selects the subregion's members.
    PhysicalExprAttr startBound =
        queryNonNegativeIndexUpperBound(range.getLogicalStart());
    PhysicalExprAttr stopBound =
        queryNonNegativeIndexUpperBound(range.getLogicalStop());
    return startBound && stopBound &&
           stopBound.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           static_cast<__int128>(extent.getValue()) * step.value() >=
               stopBound.getValue();
  });
}

LogicalResult requireFullDimensionCoverage(func::FuncOp kernel, Value source,
                                           uint64_t axis) {
  if (hasExactStaticFullCoverage(kernel, source, axis))
    return success();
  return realizeFullCoverageDimension(kernel, source, axis);
}

bool isZeroScanIdentity(Value value) {
  while (true) {
    if (auto splat = value.getDefiningOp<SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto cast = value.getDefiningOp<CastOp>()) {
      value = cast.getValue();
      continue;
    }
    break;
  }
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

FailureOr<int64_t>
exactSubregionStaticBound(const PhysicalRangeFact &ranges,
                          PhysicalSourceAxis source) {
  if (!ranges.isExact() || ranges.roots.empty())
    return failure();
  std::optional<int64_t> bound;
  for (MakeRangeOp range : ranges.roots) {
    auto current = range->getAttrOfType<IntegerAttr>(sourceSubregionBoundAttr);
    if (!range->hasAttr(sourceSubregionAttr) || !current ||
        current.getInt() <= 0 || !isUnitStepRange(range) ||
        !(sourceAxisIdentity(range) == source) ||
        (bound && *bound != current.getInt()))
      return failure();
    bound = current.getInt();
  }
  return *bound;
}

FailureOr<int64_t>
exactStaticTraversalExtent(const PhysicalRangeFact &ranges) {
  if (!ranges.isExact() || ranges.roots.empty() ||
      failed(queryExactLogicalRange(ranges)))
    return failure();
  std::optional<int64_t> extent;
  for (MakeRangeOp range : ranges.roots) {
    auto current = constantLogicalRangeCardinality(range);
    if (!current || (extent && *extent != *current))
      return failure();
    extent = *current;
  }
  return extent ? FailureOr<int64_t>(*extent)
                : FailureOr<int64_t>(failure());
}

LogicalResult requireScanFullCoverage(func::FuncOp kernel, ScanOp scan,
                                      Value source, uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  if (hasExactStaticFullCoverage(kernel, source, axis))
    return success();
  auto sourceMapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
  int64_t dimension = sourceMapping.getDimensionId();
  if (dimension <= 0)
    return requireFullDimensionCoverage(kernel, source, axis);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact sourceRanges = analysis.axisRanges(source, axis);
  if (sourceRanges.state == PhysicalFactState::Unknown ||
      sourceRanges.roots.empty()) {
    PhysicalReplayFact replay = analysis.replayability(
        source, sourceAxisIdentity(sourceMapping),
        PhysicalReplayScope::ValueGraph, /*allowAccesses=*/false);
    if (!sourceRanges.roots.empty() || !replay.isReplayable()) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "scan source has no exact full-coverage range authority");
      diagnostic << "; source_type=" << source.getType();
      for (Operation *blocker : sourceRanges.blockers)
        diagnostic << ", blocker=" << blocker->getName();
      return failure();
    }
  }
  bool subregion = llvm::any_of(sourceRanges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  if (subregion) {
    if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
        scan.getCaptureCount() != 0 ||
        queryBinaryCombineKind(scan.getCombine()) != BinaryOperator::Add ||
        !isZeroScanIdentity(
            scan.getInputs()[scan.getSourceCount()]))
      return scan.emitOpError(
          "dynamic subregion scan has no proven tail-neutral combine");
    auto chunkExtent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<int64_t> bound =
        exactSubregionStaticBound(sourceRanges,
                                  sourceAxisIdentity(sourceMapping));
    if (chunkExtent.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        failed(bound) || chunkExtent.getValue() < *bound)
      return scan.emitOpError(
          "dynamic subregion scan has no static source bound");
    PhysicalSourceAxis sourceAxis = sourceAxisIdentity(sourceMapping);
    for (MakeRangeOp range : sourceRanges.roots) {
      if (!range->hasAttr(sourceSubregionAttr) || !isUnitStepRange(range) ||
          !(sourceAxisIdentity(range) == sourceAxis))
        return range.emitOpError(
            "subregion scan source has incompatible physical ranges");
      retargetSourceExtent(range.getResult(), sourceAxis, chunkExtent);
    }
    retargetDimensionExtent(source, dimension, chunkExtent);
    return success();
  }
  if (failed(dimensionArgument(kernel, dimension))) {
    FailureOr<int64_t> staticExtent =
        exactStaticTraversalExtent(sourceRanges);
    if (failed(staticExtent))
      return scan.emitOpError(
                 "scan full-coverage dimension has neither runtime ABI nor exact static range authority")
             << "; dimension=" << dimension
             << "; source=" << source.getType();
    PhysicalExprAttr covered = expression(
        kernel.getContext(), PhysicalExprKind::Constant, *staticExtent);
    for (MakeRangeOp range : sourceRanges.roots) {
      FailureOr<uint64_t> rangeIdentity = rangeDimension(range);
      if (failed(rangeIdentity) ||
          *rangeIdentity != static_cast<uint64_t>(dimension) ||
          range->hasAttr(sourceSubregionAttr))
        return range.emitOpError(
            "static scan source range does not match its logical dimension");
      retargetDimensionExtent(range.getResult(), dimension, covered);
    }
    retargetDimensionExtent(source, dimension, covered);
    return success();
  }
  std::string parameterName = ("FULL_D" + Twine(dimension)).str();
  ParameterOp parameter;
  kernel.walk([&](ParameterOp candidate) {
    if (!parameter &&
        candidate.getParameter().getName().getValue() == parameterName)
      parameter = candidate;
  });
  if (!parameter) {
    static constexpr int64_t candidates[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    OpBuilder builder(&kernel.getBody().front(),
                      kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(parameterName),
        static_cast<uint32_t>(ParameterRole::ScanChunk),
        static_cast<uint32_t>(ParameterCategory::Coverage),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    parameter = builder.create<ParameterOp>(source.getLoc(),
                                            builder.getIndexType(), schema);
  }
  parameter->setAttr(
      dimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
  PhysicalExprAttr covered = fragmentExtent(parameter);
  for (MakeRangeOp range : sourceRanges.roots) {
    FailureOr<uint64_t> rangeIdentity = rangeDimension(range);
    if (failed(rangeIdentity) ||
        *rangeIdentity != static_cast<uint64_t>(dimension) ||
        range->hasAttr(sourceSubregionAttr))
      return range.emitOpError(
          "scan source range does not match its full-coverage dimension");
    retargetDimensionExtent(range.getResult(), dimension, covered);
  }
  retargetDimensionExtent(source, dimension, covered);
  if (failed(
          bindFullCoverageDimension(kernel, dimension, parameter.getResult())))
    return kernel.emitError(
        "scan source could not bind its exact full-coverage parameter");
  return success();
}

LogicalResult requireStructuredReductionFullCoverage(func::FuncOp kernel,
                                                      Value source,
                                                      uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  if (hasExactStaticFullCoverage(kernel, source, axis))
    return success();
  auto sourceMapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, axis);
  bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  if (subregion) {
    PhysicalSourceAxis sourceAxis = sourceAxisIdentity(sourceMapping);
    FailureOr<int64_t> bound = exactSubregionStaticBound(ranges, sourceAxis);
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    if (failed(bound) || extent.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        extent.getValue() < *bound)
      return realizeFullCoverageDimension(kernel, source, axis);
    for (MakeRangeOp range : ranges.roots)
      retargetSourceExtent(range.getResult(), sourceAxis, extent);
    retargetSourceExtent(source, sourceAxis, extent);
    return success();
  }
  return realizeFullCoverageDimension(kernel, source, axis);
}

FragmentType predicateType(FragmentType source) {
  return FragmentType::get(source.getContext(), IntegerType::get(source.getContext(), 1),
                           source.getShape(), source.getAxisMaps(),
                           source.getValidity(),
                           source.getOwner());
}

FailureOr<unsigned> tailPredicateAxis(FragmentType target, MakeRangeOp range) {
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  auto source = queryFragmentAxis(target, sourceAxisIdentity(range));
  if (source.isExact() && succeeded(dimension) && source.dimensionId == *dimension)
    return source.fragmentAxis;
  if (range->hasAttr(sourceSubregionAttr))
    return failure();
  auto axis = succeeded(dimension) ? queryFragmentDimension(target, *dimension)
                                   : PhysicalDimensionProjection{};
  return axis.isExact() ? FailureOr<unsigned>(axis.fragmentAxis)
                        : FailureOr<unsigned>(failure());
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges);
void collectProducerRanges(Value value, PhysicalSourceAxis source,
                           llvm::SmallPtrSetImpl<Operation *> &ranges);

bool hasTailPredicate(
    Value coordinate,
    const llvm::DenseMap<Value, Value> &rangePredicates) {
  llvm::SmallPtrSet<Operation *, 8> ranges;
  collectCoordinateRanges(coordinate, ranges);
  return llvm::any_of(ranges, [&](Operation *operation) {
    auto range = dyn_cast<MakeRangeOp>(operation);
    return range && rangePredicates.contains(range.getResult());
  });
}

FailureOr<Value> accessValidity(OpBuilder &builder, Location location,
                                ValueRange coordinates,
                                llvm::DenseMap<Value, Value> &rangePredicates,
                                FragmentType valueType, Value existing) {
  FragmentType target = predicateType(valueType);
  Value result;
  if (existing) {
    FailureOr<Value> broadcast =
        materializeBroadcastToFragment(builder, location, existing, target);
    if (failed(broadcast))
      return emitError(location, "pointwise access existing validity has incompatible schema")
             << "; predicate=" << existing.getType() << "; target=" << target;
    result = *broadcast;
  }
  SmallVector<Value> fragmentCoordinates;
  for (Value coordinate : coordinates)
    if (isa<FragmentType>(coordinate.getType()))
      fragmentCoordinates.push_back(coordinate);
  bool cartesian = fragmentCoordinates.size() == target.getShape().size() &&
      llvm::all_of(fragmentCoordinates, [](Value coordinate) {
        return cast<FragmentType>(coordinate.getType()).getShape().size() == 1;
      });
  unsigned fragmentAxis = 0;
  for (Value coordinate : coordinates) {
    std::optional<unsigned> positionalAxis;
    if (isa<FragmentType>(coordinate.getType())) {
      if (cartesian)
        positionalAxis = fragmentAxis;
      ++fragmentAxis;
    }
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    for (Operation *operation : ranges) {
      auto range = dyn_cast<MakeRangeOp>(operation);
      if (!range)
        continue;
      auto found = rangePredicates.find(range.getResult());
      if (found == rangePredicates.end())
        continue;
      FailureOr<unsigned> axis = tailPredicateAxis(target, range);
      if (failed(axis) && positionalAxis)
        axis = *positionalAxis;
      if (failed(axis))
        continue;
      FailureOr<Value> broadcast = projectPredicateToFragmentAxis(
          builder, location, found->second, target, *axis);
      if (failed(broadcast))
        return emitError(location, "pointwise access tail has incompatible schema")
               << "; tail=" << found->second.getType() << "; target=" << target;
      result = result ? Value(builder.create<BinaryOp>(
                            location, target, result, *broadcast,
                            BinaryOperator::LogicalAnd))
                      : *broadcast;
    }
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

bool hasFullRangeReductionCapture(Value value, MakeRangeOp range,
                                  Operation *anchor) {
  auto kernel = anchor->getParentOfType<func::FuncOp>();
  if (!samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()))
    return false;
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  FailureOr<int64_t> dimension = queryRangeDimension(range);
  if (!kernel || failed(dimension))
    return false;
  DominanceInfo dominance(kernel);
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    Operation *producer = current.getDefiningOp();
    if (!producer)
      continue;
    if (auto loop = dyn_cast<scf::ForOp>(producer)) {
      auto resultType = dyn_cast<FragmentType>(current.getType());
      auto sources = loop->getAttrOfType<ArrayAttr>(reductionSourcesAttr);
      bool consumesSource = sources && llvm::any_of(sources, [&](Attribute attr) {
        auto identity = dyn_cast<PhysicalSourceAttr>(attr);
        return identity &&
               PhysicalSourceAxis{identity.getSourceId(), identity.getSourceAxis(),
                                  identity.getDerived()} == source;
      });
      if (!consumesSource || !resultType ||
          !dominance.dominates(current, anchor) ||
          llvm::any_of(resultType.getAxisMaps(), [&](Attribute attr) {
            return cast<AxisMapAttr>(attr).getDimensionId() == *dimension;
          }) ||
          !samePhysicalScalarExpression(loop.getLowerBound(),
                                         range.getLogicalStart()) ||
          !samePhysicalScalarExpression(loop.getUpperBound(),
                                         range.getLogicalStop()))
        continue;
      bool readsOnly = !loop->walk([](Operation *operation) {
        return isa<LoadOp, GatherOp, scf::ForOp, scf::IfOp, scf::WhileOp>(operation) ||
                       isMemoryEffectFree(operation)
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      }).wasInterrupted();
      if (!readsOnly)
        continue;
      llvm::DenseSet<Value> dependencies;
      std::function<bool(Value)> usesOutputRange = [&](Value dependency) {
        if (!dependencies.insert(dependency).second)
          return false;
        if (auto argument = dyn_cast<BlockArgument>(dependency)) {
          Operation *owner = argument.getOwner()->getParentOp();
          return owner != kernel && owner != loop && !loop->isAncestor(owner);
        }
        Operation *definition = dependency.getDefiningOp();
        if (!definition)
          return true;
        if (auto candidate = dyn_cast<MakeRangeOp>(definition))
          if (analysis.lockstepRanges({range, candidate}).isExact())
            return true;
        if (llvm::any_of(definition->getOperands(), usesOutputRange))
          return true;
        for (Region &region : definition->getRegions())
          for (Block &block : region)
            for (Operation &nested : block)
              if (llvm::any_of(nested.getOperands(), usesOutputRange))
                return true;
        return false;
      };
      bool dependent = usesOutputRange(current);
      if (!dependent)
        return true;
      continue;
    }
    if (producer->getNumRegions() == 0)
      pending.append(producer->operand_begin(), producer->operand_end());
  }
  return false;
}

FailureOr<Value> replayPointwiseValue(OpBuilder &builder, Value value,
                                      PhysicalSourceAxis source,
                                      ArrayRef<int64_t> traversalDimensions,
                                      PhysicalExprAttr blockedExtent,
                                      Value blockedRange, Value blockedValidity,
                                      Operation *insertionAnchor,
                                      IRMapping &mapping);

HistogramOp histogramSource(Value value);

LogicalResult addTailValidity(func::FuncOp kernel,
                              llvm::DenseMap<Value, Value> &rangePredicates,
                              bool includeStores) {
  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  SmallVector<HistogramOp> histograms;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  kernel.walk([&](HistogramOp histogram) { histograms.push_back(histogram); });

  for (LoadOp load : loads) {
    bool affected = llvm::any_of(load.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      continue;
    auto valueType = dyn_cast<FragmentType>(load.getResult().getType());
    if (!valueType)
      return load.emitOpError(
          "pointwise blocked load must produce a physical fragment");
    OpBuilder builder(load);
    FailureOr<Value> valid = accessValidity(
        builder, load.getLoc(), load.getCoordinates(), rangePredicates, valueType,
        load.getValid());
    if (failed(valid)) {
      InFlightDiagnostic diagnostic =
          load.emitOpError("could not form pointwise tail validity");
      diagnostic << "; value=" << valueType;
      if (load.getValid())
        diagnostic << "; existing=" << load.getValid().getType();
      else
        diagnostic << "; existing=<none>";
      return failure();
    }
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> broadcast =
          materializeBroadcastToFragment(builder, load.getLoc(), load.getFill(), valueType);
      if (failed(broadcast))
        return load.emitOpError("could not broadcast the existing load fill");
      fill = *broadcast;
    } else {
      FailureOr<Value> zero = materializeZeroFragment(builder, load.getLoc(), valueType);
      if (failed(zero))
        return load.emitOpError("pointwise load element type has no zero fill");
      fill = *zero;
    }
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), valueType, load.getResource(), load.getCoordinates(),
        *valid, fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }

  for (HistogramOp histogram : histograms) {
    auto valueType = dyn_cast<FragmentType>(histogram.getValues().getType());
    if (!valueType)
      return histogram.emitOpError(
          "pointwise blocked histogram must consume a physical fragment");
    FragmentType validType = predicateType(valueType);
    OpBuilder builder(histogram);
    FailureOr<Value> existing = materializeBroadcastToFragment(
        builder, histogram.getLoc(), histogram.getValid(), validType);
    if (failed(existing))
      return histogram.emitOpError(
          "could not broadcast the existing histogram validity");
    Value valid = *existing;
    bool affected = false;
    for (auto [rangeValue, predicate] : rangePredicates) {
      auto range = rangeValue.getDefiningOp<MakeRangeOp>();
      if (!range)
        continue;
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectProducerRanges(
          histogram.getValues(),
          PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()},
          ranges);
      if (!ranges.contains(range.getOperation()))
        continue;
      FailureOr<Value> broadcast =
          materializeBroadcastToFragment(builder, histogram.getLoc(), predicate, validType);
      if (failed(broadcast))
        return histogram.emitOpError(
            "could not project pointwise tail validity onto histogram values");
      valid = builder.create<BinaryOp>(histogram.getLoc(), validType, valid,
                                       *broadcast, BinaryOperator::LogicalAnd);
      affected = true;
    }
    if (!affected)
      continue;
    auto replacement = builder.create<HistogramOp>(
        histogram.getLoc(), histogram.getResult().getType(),
        histogram.getValues(), histogram.getBins(), valid);
    if (Attribute origin = histogram->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    histogram.getResult().replaceAllUsesWith(replacement.getResult());
    histogram.erase();
  }

  if (!includeStores)
    return success();
  SmallVector<std::pair<MakeRangeOp, IRMapping>> storeReplays;
  for (StoreOp store : stores) {
    bool affected = llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      continue;
    OpBuilder builder(store);
    Value payload = store.getValue();
    Value existingValidity = store.getValid();
    auto valueType = dyn_cast<FragmentType>(payload.getType());
    if (valueType) {
      SmallVector<MakeRangeOp> blockedRanges;
      for (Value coordinate : store.getCoordinates()) {
        llvm::SmallPtrSet<Operation *, 8> roots;
        collectCoordinateRanges(coordinate, roots);
        for (Operation *root : roots) {
          auto range = dyn_cast<MakeRangeOp>(root);
          if (range && rangePredicates.contains(range.getResult()) &&
              !llvm::is_contained(blockedRanges, range))
            blockedRanges.push_back(range);
        }
      }
      for (MakeRangeOp range : blockedRanges) {
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        auto sourceProjection = queryFragmentAxis(valueType, sourceAxisIdentity(range));
        PhysicalDimensionProjection dimensionProjection =
            succeeded(dimension)
                ? queryFragmentDimension(valueType, *dimension)
                : PhysicalDimensionProjection{};
        if (failed(dimension) ||
            (!sourceProjection.isExact() && !dimensionProjection.isExact()))
          continue;
        unsigned payloadAxis = sourceProjection.isExact()
                                   ? sourceProjection.fragmentAxis
                                   : dimensionProjection.fragmentAxis;
        auto rangeType = cast<FragmentType>(range.getResult().getType());
        auto blockedExtent =
            cast<PhysicalExprAttr>(rangeType.getShape()[0]);
        if (HistogramOp histogram = histogramSource(payload)) {
          retargetDimensionExtent(histogram.getResult(), *dimension,
                                  blockedExtent);
          payload = store.getValue();
          valueType = dyn_cast<FragmentType>(payload.getType());
          if (!valueType)
            return store.emitOpError(
                "histogram writeback lost its physical result schema");
        }
        if (valueType.getShape()[payloadAxis] == blockedExtent)
          continue;
        IRMapping mapping;
        DominanceInfo dominance(kernel);
        for (auto &entry : storeReplays) {
          MakeRangeOp previous = entry.first;
          auto previousDimension = queryRangeDimension(previous);
          if (failed(previousDimension) || *previousDimension != *dimension ||
              !samePhysicalScalarExpression(previous.getStart(), range.getStart()) ||
              !samePhysicalScalarExpression(previous.getExtent(), range.getExtent()) ||
              !samePhysicalScalarExpression(previous.getLogicalStart(), range.getLogicalStart()) ||
              !samePhysicalScalarExpression(previous.getLogicalStop(), range.getLogicalStop()) ||
              !samePhysicalScalarExpression(previous.getStep(), range.getStep()))
            continue;
          for (auto [original, replayed] : entry.second.getValueMap())
            if (dominance.dominates(replayed, store.getOperation()))
              mapping.map(original, replayed);
        }
        auto payloadMapping =
            cast<AxisMapAttr>(valueType.getAxisMaps()[payloadAxis]);
        SmallVector<int64_t> traversalDimensions{payloadMapping.getDimensionId()};
        FailureOr<Value> replayed = replayPointwiseValue(
            builder, payload, sourceAxisIdentity(range), traversalDimensions,
            blockedExtent,
            range.getResult(), rangePredicates.lookup(range.getResult()),
            store.getOperation(), mapping);
        if (failed(replayed))
          return store.emitOpError(
                     "pointwise store payload cannot be replayed to its blocked coordinate")
                 << "; source_id=" << range.getSourceId()
                 << ", source_axis=" << range.getSourceAxis()
                 << ", dimension="
                 << (succeeded(dimension) ? *dimension : -1)
                 << ", payload=" << payload.getType()
                 << ", coordinate=" << range.getResult().getType();
        if (existingValidity) {
          FailureOr<Value> replayedValidity = replayPointwiseValue(
              builder, existingValidity, sourceAxisIdentity(range),
              traversalDimensions, blockedExtent, range.getResult(),
              rangePredicates.lookup(range.getResult()), store.getOperation(),
              mapping);
          if (failed(replayedValidity))
            return store.emitOpError(
                "pointwise store validity cannot be replayed with its payload");
          existingValidity = *replayedValidity;
        }
        storeReplays.emplace_back(range, mapping);
        payload = *replayed;
        valueType = dyn_cast<FragmentType>(payload.getType());
        if (!valueType)
          return store.emitOpError(
              "pointwise replay lost the store payload fragment schema");
      }
    }
    if (!valueType) {
      if (!isa<IntegerType, FloatType, IndexType>(payload.getType()))
        return store.emitOpError(
            "pointwise blocked store has no projectable value schema");
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      std::optional<uint64_t> owner;
      for (Value coordinate : store.getCoordinates()) {
        auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        if (!fragment)
          continue;
        if (owner && *owner != fragment.getOwner())
          return store.emitOpError(
              "pointwise store coordinates have conflicting ownership");
        owner = fragment.getOwner();
        for (auto [extent, attribute] :
             llvm::zip(fragment.getShape(), fragment.getAxisMaps())) {
          auto mapping = cast<AxisMapAttr>(attribute);
          auto found = llvm::find_if(mappings, [&](Attribute existing) {
            auto axis = cast<AxisMapAttr>(existing);
            return sourceAxisIdentity(axis) == sourceAxisIdentity(mapping);
          });
          if (found != mappings.end())
            continue;
          shape.push_back(extent);
          mappings.push_back(AxisMapAttr::get(
              store.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
        }
      }
      if (shape.empty())
        return store.emitOpError(
            "pointwise blocked store has no coordinate fragment authority");
      auto target = FragmentType::get(
          store.getContext(), payload.getType(),
          ArrayAttr::get(store.getContext(), shape),
          ArrayAttr::get(store.getContext(), mappings), /*validity=*/1,
          owner.value_or(1));
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, store.getLoc(), payload, target);
      if (failed(projected))
        return store.emitOpError(
            "pointwise store value cannot be projected to its coordinate schema");
      payload = *projected;
      valueType = target;
    }
    FailureOr<Value> valid = accessValidity(
        builder, store.getLoc(), store.getCoordinates(), rangePredicates,
        valueType, existingValidity);
    if (failed(valid))
      return store.emitOpError("could not form pointwise store validity");
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        payload, *valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

void collectProducerRanges(Value value, PhysicalSourceAxis source,
                           llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return;
  if (!queryFragmentAxis(value.getType(), source).isExact())
    return;
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value, source);
  if (fact.state == PhysicalFactState::Unknown)
    return;
  for (MakeRangeOp range : fact.roots)
    if (sourceAxisIdentity(range) == source)
      ranges.insert(range.getOperation());
}

Type replaceTraversalExtent(Type type, PhysicalSourceAxis source,
                            ArrayRef<int64_t> dimensions,
                            PhysicalExprAttr extent) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    SmallVector<Attribute> shape(fragment.getShape().begin(),
                                 fragment.getShape().end());
    bool changed = false;
    for (const PhysicalAxisProjection &projection :
         queryFragmentAxes(fragment, source)) {
      if (!llvm::is_contained(dimensions, projection.dimensionId))
        continue;
      shape[projection.fragmentAxis] = extent;
      changed = true;
    }
    return changed ? Type(FragmentType::get(
                         fragment.getContext(), fragment.getElementType(),
                         ArrayAttr::get(fragment.getContext(), shape),
                         fragment.getAxisMaps(), fragment.getValidity(),
                         fragment.getOwner()))
                   : type;
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return type;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute field : record.getFieldTypes()) {
    Type current = cast<TypeAttr>(field).getValue();
    Type replacement =
        replaceTraversalExtent(current, source, dimensions, extent);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != current;
  }
  return changed ? Type(RecordType::get(
                       type.getContext(), record.getFieldNames(),
                       ArrayAttr::get(type.getContext(), fields),
                       record.getOwner()))
                 : type;
}

bool containsSource(Value value, PhysicalSourceAxis source) {
  return queryFragmentAxis(value.getType(), source).isExact();
}

std::optional<int64_t> estimatedFragmentRegisters(func::FuncOp kernel, Value value) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || !isa<IntegerType, FloatType>(fragment.getElementType()))
    return std::nullopt;
  PhysicalProgramAnalysis analysis(kernel);
  int64_t limit = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr).getRegistersPerUnit();
  int64_t footprint = std::max(1u,
      (fragment.getElementType().getIntOrFloatBitWidth() + 31) / 32);
  for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    int64_t minimum;
    if (analysis.axisRealization(value, axis).constructionScalarSeed) {
      auto range = queryExactLogicalRange(analysis.axisRanges(value, axis));
      auto capacity = succeeded(range) ? queryLogicalRangeCapacity(*range)
                                       : PhysicalExprAttr();
      auto count = capacity ? constantPhysicalExpression(capacity) : std::nullopt;
      if (!count || *count <= 0)
        return std::nullopt;
      // Construction starts runtime subregions with extent one. Their eventual
      // complete fragments must cover the logical capacity, not that seed.
      minimum = *count > limit ? limit + 1
                              : llvm::PowerOf2Ceil(static_cast<uint64_t>(*count));
    } else if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant)) {
      minimum = extent.getValue();
    } else if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      auto parameter = queryParameterBySymbol(kernel, extent.getSymbol());
      if (failed(parameter))
        return std::nullopt;
      minimum = *llvm::min_element(parameter->getParameter().getCandidates().asArrayRef());
    } else {
      return std::nullopt;
    }
    if (minimum <= 0)
      return std::nullopt;
    footprint = std::min<__int128>(static_cast<__int128>(footprint) * minimum,
                                   static_cast<__int128>(limit) + 1);
  }
  return footprint;
}

bool containsTraversal(Value value, PhysicalSourceAxis source,
                       ArrayRef<int64_t> dimensions) {
  std::function<bool(Type)> contains = [&](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type))
      return llvm::any_of(queryFragmentAxes(fragment, source),
                          [&](const PhysicalAxisProjection &projection) {
                            return llvm::is_contained(dimensions,
                                                      projection.dimensionId);
                          });
    auto record = dyn_cast<RecordType>(type);
    return record && llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
             return contains(cast<TypeAttr>(field).getValue());
           });
  };
  return contains(value.getType());
}

void collectTraversalDimensions(Type type, PhysicalSourceAxis source,
                                SmallVectorImpl<int64_t> &dimensions) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (const PhysicalAxisProjection &projection :
         queryFragmentAxes(fragment, source))
      if (!llvm::is_contained(dimensions, projection.dimensionId))
        dimensions.push_back(projection.dimensionId);
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    for (Attribute field : record.getFieldTypes())
      collectTraversalDimensions(cast<TypeAttr>(field).getValue(), source,
                                 dimensions);
}

FailureOr<FragmentType> coordinateValueSchema(Type element,
                                              ValueRange coordinates) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  std::optional<uint64_t> owner;
  for (Value coordinate : coordinates) {
    auto fragment = dyn_cast<FragmentType>(coordinate.getType());
    if (!fragment)
      continue;
    if (owner && *owner != fragment.getOwner())
      return failure();
    owner = fragment.getOwner();
    for (auto [extent, attribute] :
         llvm::zip(fragment.getShape(), fragment.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      auto found = llvm::find_if(mappings, [&](Attribute existing) {
        return sourceAxisIdentity(cast<AxisMapAttr>(existing)) ==
               sourceAxisIdentity(mapping);
      });
      if (found != mappings.end()) {
        unsigned axis = std::distance(mappings.begin(), found);
        if (shape[axis] != extent)
          return failure();
        continue;
      }
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          element.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    }
  }
  if (shape.empty())
    return failure();
  return FragmentType::get(element.getContext(), element,
                           ArrayAttr::get(element.getContext(), shape),
                           ArrayAttr::get(element.getContext(), mappings),
                           /*validity=*/1, owner.value_or(1));
}

FailureOr<Value> replayPointwiseValue(OpBuilder &builder, Value value,
                                      PhysicalSourceAxis source,
                                      ArrayRef<int64_t> traversalDimensions,
                                      PhysicalExprAttr blockedExtent,
                                      Value blockedRange, Value blockedValidity,
                                      Operation *insertionAnchor,
                                      IRMapping &mapping) {
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  auto blocked = blockedRange.getDefiningOp<MakeRangeOp>();
  FailureOr<int64_t> blockedDimension =
      blocked ? queryRangeDimension(blocked) : FailureOr<int64_t>(failure());
  if (failed(blockedDimension))
    return insertionAnchor->emitOpError(
        "pointwise replay has no exact blocked-dimension authority");
  if (!containsTraversal(value, source, traversalDimensions)) {
    DominanceInfo dominance(kernel);
    if (dominance.dominates(value, insertionAnchor))
      return value;
  }
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return insertionAnchor->emitOpError(
        "pointwise replay cannot rematerialize a non-dominating block argument");
  auto selectedResult = dyn_cast<OpResult>(value);
  if (!selectedResult ||
      selectedResult.getResultNumber() >= producer->getNumResults())
    return producer->emitOpError(
        "pointwise replay has no exact producer result occurrence");
  if (auto original = dyn_cast<FragmentType>(value.getType());
      original && isa<ReshapeOp, scf::ForOp>(producer)) {
    auto target = cast<FragmentType>(replaceTraversalExtent(
        original, source, traversalDimensions, blockedExtent));
    SmallVector<unsigned> changedAxes;
    for (auto [axis, extent] : llvm::enumerate(original.getShape()))
      if (extent != target.getShape()[axis])
        changedAxes.push_back(axis);
    if (changedAxes.size() == 1) {
      unsigned axis = changedAxes.front();
      bool retain = false;
      if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
        unsigned resultRank = 0;
        for (Attribute attribute : reshape.getReassociation())
          resultRank += cast<ReshapeGroupAttr>(attribute).getResultAxes().size();
        unsigned prefix = original.getShape().size() - resultRank;
        retain = llvm::any_of(reshape.getReassociation(), [&](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getResultAxes().size() > 1 &&
                 llvm::is_contained(group.getResultAxes().asArrayRef(),
                                    static_cast<int64_t>(axis) - prefix);
        });
      } else {
        auto relation = cast<AxisMapAttr>(original.getAxisMaps()[axis]);
        auto dependency = PhysicalProgramAnalysis(kernel).reductionDependency(
            value, sourceAxisIdentity(relation), relation.getDimensionId());
        retain = !dependency.isExact() || dependency.depends;
      }
      auto extent = cast<PhysicalExprAttr>(original.getShape()[axis]);
      auto size = constantLogicalRangeCardinality(blocked);
      auto begin = queryLaunchExpression(blocked.getLogicalStart());
      if (retain && size && isUnitStepRange(blocked) &&
          extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          extent.getValue() == *size && begin &&
          begin.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          begin.getValue() == 0 && DominanceInfo(kernel).dominates(value, insertionAnchor)) {
        // Preserve split shapes and complete loop-carried reduction inputs.
        // Only the already-computed result is projected to the writeback tile.
        Location location = producer->getLoc();
        auto axisMap = cast<AxisMapAttr>(original.getAxisMaps()[axis]);
        auto coordinate = FragmentType::get(
            kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({blockedExtent}),
            builder.getArrayAttr({AxisMapAttr::get(
                kernel.getContext(), axisMap.getSourceId(), axisMap.getSourceAxis(),
                axisMap.getDimensionId(), 0, axisMap.getDerived())}),
            original.getValidity(), original.getOwner());
        auto identity = builder.getArrayAttr({ReshapeGroupAttr::get(
            kernel.getContext(), builder.getDenseI64ArrayAttr({0}),
            builder.getDenseI64ArrayAttr({0}))});
        Value indices = builder.create<ReshapeOp>(
            location, coordinate, blockedRange, identity);
        Value stop = builder.create<arith::ConstantIndexOp>(location, *size);
        Value valid = builder.create<CompareOp>(
            location, predicateType(coordinate), indices,
            builder.create<BroadcastOp>(location, coordinate, stop),
            ComparePredicate::Lt);
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value nonnegative = builder.create<CompareOp>(
            location, predicateType(coordinate), indices,
            builder.create<BroadcastOp>(location, coordinate, zero),
            ComparePredicate::Ge);
        valid = builder.create<BinaryOp>(location, predicateType(coordinate),
                                          valid, nonnegative, BinaryOperator::LogicalAnd);
        auto projected = projectPredicateToFragmentAxis(builder, location, valid, target, axis);
        auto fill = materializeZeroFragment(builder, location, target);
        if (failed(projected) || failed(fill))
          return failure();
        auto selected = builder.create<GatherOp>(
            location, target, value, ValueRange{indices}, *projected, *fill,
            ArrayRef<int64_t>{static_cast<int64_t>(axis)});
        if (Attribute origin = producer->getAttr(originAttr))
          selected->setAttr(originAttr, origin);
        mapping.map(value, selected.getResult());
        return selected.getResult();
      }
    }
  }
  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    if (sourceAxisIdentity(range) == source &&
        sourceAxisIdentity(blocked) == source && succeeded(rangeDimension) &&
        *rangeDimension == *blockedDimension &&
        samePhysicalScalarExpression(range.getLogicalStart(), blocked.getLogicalStart()) &&
        samePhysicalScalarExpression(range.getStep(), blocked.getStep())) {
      mapping.map(value, blockedRange);
      return blockedRange;
    }
    auto originalType = dyn_cast<FragmentType>(range.getResult().getType());
    bool sameBounds =
        sourceAxisIdentity(range) == source &&
        samePhysicalScalarExpression(range.getLogicalStart(), blocked.getLogicalStart()) &&
        samePhysicalScalarExpression(range.getLogicalStop(), blocked.getLogicalStop()) &&
        samePhysicalScalarExpression(range.getStep(), blocked.getStep());
    auto sourceSize = constantLogicalRangeCardinality(range);
    auto targetSize = constantLogicalRangeCardinality(blocked);
    bool positional = sourceAxisIdentity(range) == source &&
        succeeded(rangeDimension) &&
        llvm::is_contained(traversalDimensions, *rangeDimension) &&
        isUnitStepRange(range) && isUnitStepRange(blocked) &&
        sourceSize && targetSize && *sourceSize == *targetSize;
    if (!blocked || failed(rangeDimension) || failed(blockedDimension) ||
        (*rangeDimension != *blockedDimension && !sameBounds && !positional) || !originalType ||
        originalType.getShape().size() != 1) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "pointwise replay reached a range without an exact shared dimension relation");
      if (succeeded(rangeDimension))
        diagnostic << "; source_dimension=" << *rangeDimension;
      if (succeeded(blockedDimension))
        diagnostic << "; blocked_dimension=" << *blockedDimension;
      return failure();
    }
    auto projectedType = FragmentType::get(
        originalType.getContext(), originalType.getElementType(),
        ArrayAttr::get(originalType.getContext(), {blockedExtent}),
        originalType.getAxisMaps(), originalType.getValidity(),
        originalType.getOwner());
    Value offset = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), blocked.getStart(),
        blocked.getLogicalStart(), BinaryOperator::Subtract);
    Value start = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getLogicalStart(), offset,
        BinaryOperator::Add);
    Value projected = builder.create<MakeRangeOp>(
        range.getLoc(), projectedType, start, blocked.getExtent(),
        blocked.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(projected.getDefiningOp(), range);
    mapping.map(value, projected);
    return projected;
  }
  SmallVector<int64_t> valueDimensions;
  collectTraversalDimensions(value.getType(), source, valueDimensions);
  for (int64_t dimension : valueDimensions) {
    if (!llvm::is_contained(traversalDimensions, dimension))
      continue;
    // Pure scalar/pointwise nodes are proved by replaying their operands below.
    // A bound operand can already hold a snapshot made before an intervening
    // store; following its old graph would incorrectly require another load.
    if (producer->getNumRegions() == 0 &&
        isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                             /*allowAccesses=*/false))
      continue;
    PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
        value, source, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true, insertionAnchor, dimension);
    if (!replay.isReplayable()) {
      InFlightDiagnostic diagnostic = producer->emitOpError(
          "pointwise value has no exact insertion-point replay fact");
      diagnostic << "; dimension=" << dimension;
      for (Operation *blocker : replay.blockers)
        diagnostic << "; blocker=" << blocker->getName();
      return failure();
    }
  }
  SmallVector<Value> replayOperands(producer->getOperands());
  if (isa<scf::IfOp, scf::ForOp>(producer)) {
    llvm::SetVector<Value> captures;
    for (Region &region : producer->getRegions())
      getUsedValuesDefinedAbove(region, captures);
    replayOperands.append(captures.begin(), captures.end());
  }
  for (Value operand : replayOperands) {
    PhysicalSourceAxis operandSource = source;
    SmallVector<int64_t> operandDimensions(traversalDimensions);
    if (isa<BroadcastOp, ReshapeOp>(producer)) {
      auto input = dyn_cast<FragmentType>(operand.getType());
      auto output = dyn_cast<FragmentType>(value.getType());
      PhysicalAxisProjection requested = queryFragmentAxis(output, source);
      if (input && requested.isExact() &&
          llvm::is_contained(traversalDimensions, requested.dimensionId)) {
        std::optional<unsigned> inputAxis;
        if (isa<BroadcastOp>(producer)) {
          BroadcastProjection projection = queryAxisProjection(input, output);
          if (projection.isExact())
            inputAxis = projection.targetToSource[requested.fragmentAxis];
        } else {
          auto reassociation = cast<ReshapeOp>(producer).getReassociation();
          unsigned sourceRank = 0, resultRank = 0;
          for (Attribute attribute : reassociation) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            for (int64_t axis : group.getSourceAxes().asArrayRef())
              sourceRank = std::max(sourceRank, static_cast<unsigned>(axis + 1));
            for (int64_t axis : group.getResultAxes().asArrayRef())
              resultRank = std::max(resultRank, static_cast<unsigned>(axis + 1));
          }
          unsigned sourcePrefix = input.getShape().size() - sourceRank;
          unsigned resultPrefix = output.getShape().size() - resultRank;
          for (Attribute attribute : reassociation) {
            auto group = cast<ReshapeGroupAttr>(attribute);
            if (group.getSourceAxes().size() == 1 &&
                group.getResultAxes().size() == 1 &&
                resultPrefix + group.getResultAxes()[0] ==
                    static_cast<int64_t>(requested.fragmentAxis) &&
                input.getShape()[sourcePrefix + group.getSourceAxes()[0]] ==
                    output.getShape()[requested.fragmentAxis])
              inputAxis = sourcePrefix + group.getSourceAxes()[0];
          }
        }
        if (auto axis = inputAxis) {
            auto axisMap = cast<AxisMapAttr>(input.getAxisMaps()[*axis]);
            PhysicalRangeFact ranges =
                PhysicalProgramAnalysis(kernel).axisRanges(operand, *axis);
            bool renamedAxis =
                input.getShape()[*axis] ==
                    output.getShape()[requested.fragmentAxis];
            if (renamedAxis ||
                (axisMap.getDimensionId() == *blockedDimension &&
                ranges.state != PhysicalFactState::Unknown &&
                ranges.blockers.empty() &&
                PhysicalProgramAnalysis(kernel).lockstepRanges(ranges.roots).isExact() &&
                llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
                  auto dimension = queryRangeDimension(range);
                  bool sameExtent = succeeded(dimension) &&
                                    *dimension == *blockedDimension;
                  bool sameBounds = samePhysicalScalarExpression(
                      range.getLogicalStart(), blocked.getLogicalStart()) &&
                      samePhysicalScalarExpression(range.getLogicalStop(), blocked.getLogicalStop());
                  auto sourceSize = constantLogicalRangeCardinality(range);
                  auto targetSize = constantLogicalRangeCardinality(blocked);
                  bool sameCardinality = renamedAxis && sourceSize && targetSize &&
                                         *sourceSize == *targetSize;
                  return (sameExtent || sameBounds || sameCardinality) &&
                         samePhysicalScalarExpression(range.getStep(), blocked.getStep());
                }))) {
              // An explicit projection can rename an exact coordinate traversal.
              operandSource = sourceAxisIdentity(axisMap);
              for (int64_t &dimension : operandDimensions)
                if (dimension == requested.dimensionId)
                  dimension = axisMap.getDimensionId();
            }
        }
      }
    }
    FailureOr<Value> replacement = replayPointwiseValue(
        builder, operand, operandSource, operandDimensions, blockedExtent,
        blockedRange, blockedValidity, insertionAnchor, mapping);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  auto result = dyn_cast<FragmentType>(value.getType());
  FragmentType resultType =
      result ? dyn_cast<FragmentType>(replaceTraversalExtent(
                   result, source, traversalDimensions, blockedExtent))
             : FragmentType();
  auto mapped = [&](Value operand) {
    if (!operand)
      return Value();
    Value replacement = mapping.lookupOrNull(operand);
    return replacement ? replacement : operand;
  };
  auto accessValidity = [&](Value existing) -> FailureOr<Value> {
    Value valid = mapped(existing);
    for (Value operand : producer->getOperands()) {
      Value coordinate = mapped(operand);
      if (coordinate == operand || !coordinate.getDefiningOp<MakeRangeOp>())
        continue;
      auto type = cast<FragmentType>(coordinate.getType());
      Value zero = builder.create<arith::ConstantIndexOp>(producer->getLoc(), 0);
      Value lower = builder.create<CompareOp>(
          producer->getLoc(), predicateType(type), coordinate,
          builder.create<SplatOp>(producer->getLoc(), type, zero), ComparePredicate::Ge);
      FailureOr<Value> bounded = materializeValidityConjunction(
          builder, producer->getLoc(), valid, lower, predicateType(resultType));
      if (failed(bounded))
        return failure();
      valid = *bounded;
    }
    if (!blockedValidity)
      return valid;
    auto axes = queryFragmentAxes(resultType, source);
    llvm::erase_if(axes, [&](const PhysicalAxisProjection &axis) {
      return !llvm::is_contained(traversalDimensions, axis.dimensionId);
    });
    FailureOr<Value> projected = axes.size() == 1
        ? projectPredicateToFragmentAxis(builder, producer->getLoc(),
                                         blockedValidity, resultType,
                                         axes.front().fragmentAxis)
        : materializeBroadcastToFragment(builder, producer->getLoc(),
                                          blockedValidity, predicateType(resultType));
    if (failed(projected))
      return producer->emitOpError("pointwise tile validity lost its access-axis projection");
    if (!valid)
      return *projected;
    FailureOr<Value> original =
        materializeBroadcastToFragment(builder, producer->getLoc(), valid, predicateType(resultType));
    if (failed(original))
      return failure();
    return Value(builder.create<BinaryOp>(producer->getLoc(),
                                          predicateType(resultType), *original,
                                          *projected,
                                          BinaryOperator::LogicalAnd));
  };
  Value replayed;
  if (auto load = dyn_cast<LoadOp>(producer)) {
    if (!resultType)
      return failure();
    FailureOr<Value> valid = accessValidity(load.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(load.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, producer->getLoc(), resultType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates())
      coordinates.push_back(mapped(coordinate));
    auto clone = builder.create<LoadOp>(
        producer->getLoc(), resultType, mapped(load.getResource()), coordinates,
        *valid, fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      clone->setAttr(originAttr, origin);
    replayed = clone.getResult();
  } else if (auto gather = dyn_cast<GatherOp>(producer)) {
    if (!resultType)
      return failure();
    FailureOr<Value> valid = accessValidity(gather.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(gather.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, producer->getLoc(), resultType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : gather.getCoordinates())
      coordinates.push_back(mapped(coordinate));
    auto clone = builder.create<GatherOp>(
        producer->getLoc(), resultType, mapped(gather.getSource()), coordinates,
        *valid, fill, gather.getSourceAxes());
    if (Attribute origin = gather->getAttr(originAttr))
      clone->setAttr(originAttr, origin);
    replayed = clone.getResult();
  } else if (auto reshape = dyn_cast<ReshapeOp>(producer);
             reshape && !containsSource(reshape.getValue(), source) &&
             mapped(reshape.getValue()) == reshape.getValue()) {
    if (!resultType)
      return failure();
    FailureOr<Value> projected = materializeBroadcastToFragment(
        builder, producer->getLoc(), value, resultType);
    if (failed(projected))
      return producer->emitOpError("retained reshape cannot adopt the pointwise tile")
             << "; value=" << value.getType() << "; target=" << resultType;
    replayed = *projected;
  } else {
    SmallVector<MakeRangeOp> controlSourceRanges;
    if (isa<scf::IfOp, scf::ForOp>(producer) && result) {
      PhysicalProgramAnalysis analysis(kernel);
      for (auto [axis, attribute] : llvm::enumerate(result.getAxisMaps())) {
        auto relation = cast<AxisMapAttr>(attribute);
        if (!(sourceAxisIdentity(relation) == source) ||
            !llvm::is_contained(traversalDimensions, relation.getDimensionId()))
          continue;
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (!ranges.isExact())
          continue;
        for (MakeRangeOp range : ranges.roots) {
          auto sourceSize = constantLogicalRangeCardinality(range);
          auto targetSize = constantLogicalRangeCardinality(blocked);
          if (producer->isProperAncestor(range) && sourceSize && targetSize &&
              *sourceSize == *targetSize &&
              samePhysicalScalarExpression(range.getStep(), blocked.getStep()) &&
              !llvm::is_contained(controlSourceRanges, range))
            controlSourceRanges.push_back(range);
        }
      }
    }
    Operation *clone = builder.clone(*producer, mapping);
    for (auto [original, cloned] :
         llvm::zip(producer->getResults(), clone->getResults())) {
      cloned.setType(replaceTraversalExtent(
          original.getType(), source, traversalDimensions, blockedExtent));
      if (!mapping.lookupOrNull(original))
        mapping.map(original, cloned);
    }
    bool structuredControl = isa<scf::IfOp, scf::ForOp>(clone);
    llvm::SmallPtrSet<Operation *, 8> controlRanges;
    for (MakeRangeOp sourceRange : controlSourceRanges)
      if (Value mappedRange = mapping.lookupOrNull(sourceRange.getResult()))
        if (auto range = mappedRange.getDefiningOp<MakeRangeOp>())
          controlRanges.insert(range);
    llvm::DenseMap<Value, Value> controlTails;
    if (isa<ReduceOp, ScanOp, RegionFoldOp, RegionScanOp>(clone) || structuredControl)
      for (Region &region : clone->getRegions())
        for (Block &block : region) {
          for (BlockArgument argument : block.getArguments())
            argument.setType(replaceTraversalExtent(
                argument.getType(), source, traversalDimensions, blockedExtent));
          block.walk([&](Operation *nested) {
            for (Region &region : nested->getRegions())
              for (Block &block : region)
                for (BlockArgument argument : block.getArguments())
                  argument.setType(replaceTraversalExtent(
                      argument.getType(), source, traversalDimensions, blockedExtent));
            for (Value nestedResult : nested->getResults())
              nestedResult.setType(replaceTraversalExtent(
                  nestedResult.getType(), source, traversalDimensions,
                  blockedExtent));
            auto range = dyn_cast<MakeRangeOp>(nested);
            FailureOr<int64_t> dimension =
                range ? queryRangeDimension(range)
                      : FailureOr<int64_t>(failure());
            if (!range ||
                (!controlRanges.contains(range) &&
                 (!(sourceAxisIdentity(range) == source) || failed(dimension) ||
                  !llvm::is_contained(traversalDimensions, *dimension))))
              return;
            if (structuredControl) {
              retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                                   blockedExtent);
              OpBuilder nestedBuilder(range);
              Value offset = nestedBuilder.create<BinaryOp>(range.getLoc(), nestedBuilder.getIndexType(),
                  blocked.getStart(), blocked.getLogicalStart(), BinaryOperator::Subtract);
              Value start = nestedBuilder.create<BinaryOp>(range.getLoc(), nestedBuilder.getIndexType(),
                  range.getLogicalStart(), offset, BinaryOperator::Add);
              range->setOperand(0, start);
              range->setOperand(1, blocked.getExtent());
              nestedBuilder.setInsertionPointAfter(range);
              auto type = range.getResult().getType();
              Value zero = nestedBuilder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
              Value lower = nestedBuilder.create<CompareOp>(range.getLoc(), predicateType(type),
                  range, nestedBuilder.create<SplatOp>(range.getLoc(), type, zero), ComparePredicate::Ge);
              Value upper = nestedBuilder.create<CompareOp>(range.getLoc(), predicateType(type),
                  range, nestedBuilder.create<SplatOp>(range.getLoc(), type, range.getLogicalStop()), ComparePredicate::Lt);
              controlTails[range] = nestedBuilder.create<BinaryOp>(range.getLoc(), predicateType(type),
                  lower, upper, BinaryOperator::LogicalAnd);
              return;
            }
            // Helper-local ranges are complete physical values too.  When a
            // structured producer is replayed for a wider ownership fragment,
            // its local coordinate carrier must use that same physical extent;
            // changing only the result type leaves an illegal one-lane range.
            bool coveredItsLocalDomain =
                samePhysicalScalarExpression(range.getStart(),
                                             range.getLogicalStart()) &&
                samePhysicalScalarExpression(range.getExtent(),
                                             range.getLogicalStop());
            range->setOperand(1, blocked.getExtent());
            if (coveredItsLocalDomain)
              range->setOperand(4, blocked.getExtent());
          });
        }
    if (!controlTails.empty() && failed(addTailValidity(kernel, controlTails,
                                                       /*includeStores=*/false)))
      return failure();
    replayed = clone->getResult(selectedResult.getResultNumber());
  }
  if (!mapping.lookupOrNull(value))
    mapping.map(value, replayed);
  return replayed;
}

bool storeUsesRange(StoreOp store, MakeRangeOp range) {
  return llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    return ranges.contains(range.getOperation());
  });
}

std::optional<int64_t> storeAxisForRange(StoreOp store, MakeRangeOp range) {
  std::optional<int64_t> result;
  for (auto [coordinateIndex, coordinate] :
       llvm::enumerate(store.getCoordinates())) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    if (!ranges.contains(range.getOperation()))
      continue;
    int64_t sourceAxis = store.getSourceAxes()[coordinateIndex];
    if (!result || sourceAxis > *result)
      result = sourceAxis;
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
traversalDimensionsForStores(ArrayRef<StoreOp> stores, MakeRangeOp range) {
  FailureOr<int64_t> coordinateDimension = queryRangeDimension(range);
  if (failed(coordinateDimension))
    return failure();
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  SmallVector<int64_t> dimensions{*coordinateDimension};
  for (StoreOp store : stores) {
    collectTraversalDimensions(store.getValue().getType(), source, dimensions);
    if (store.getValid())
      collectTraversalDimensions(store.getValid().getType(), source, dimensions);
    for (Value coordinate : store.getCoordinates())
      collectTraversalDimensions(coordinate.getType(), source, dimensions);
  }
  return dimensions;
}

FailureOr<int64_t> reuseTraversalDimension(func::FuncOp kernel,
                                           ArrayRef<StoreOp> stores,
                                           MakeRangeOp range) {
  FailureOr<SmallVector<int64_t>> dimensions =
      traversalDimensionsForStores(stores, range);
  if (failed(dimensions))
    return failure();
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  SmallVector<int64_t> selected;
  for (int64_t dimension : *dimensions) {
    bool depends = false;
    for (StoreOp store : stores) {
      PhysicalReductionDependencyFact fact =
          PhysicalProgramAnalysis(kernel).reductionDependency(
              store.getValue(), source, dimension);
      if (!fact.isExact())
        return failure();
      depends |= fact.depends;
    }
    if (depends)
      selected.push_back(dimension);
  }
  return selected.size() == 1 ? FailureOr<int64_t>(selected.front())
                              : FailureOr<int64_t>(failure());
}

bool reachesDifferentStore(Value value, StoreOp current,
                           PhysicalSourceAxis source, int64_t dimension,
                           llvm::SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (auto store = dyn_cast<StoreOp>(user);
        store && store != current && store.getValue() == value) {
      PhysicalAxisProjection payload = queryFragmentAxis(value.getType(), source);
      bool coordinateCarriesAxis =
          llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
            PhysicalAxisProjection projection =
                queryFragmentAxis(coordinate.getType(), source);
            return projection.isExact() && projection.dimensionId == dimension;
          });
      if (payload.isExact() && payload.dimensionId == dimension &&
          coordinateCarriesAxis)
        return true;
    }
    if (!visited.insert(user).second || user->getNumRegions() != 0 ||
        !isPhysicalReplayNode(user, PhysicalReplayScope::ValueGraph,
                              /*allowAccesses=*/false))
      continue;
    for (Value result : user->getResults())
      if (reachesDifferentStore(result, current, source, dimension, visited))
        return true;
  }
  return false;
}

bool reachesReduction(Value value, PhysicalSourceAxis source,
                      int64_t dimension,
                      llvm::SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (auto reduce = dyn_cast<ReduceOp>(user)) {
      for (Value input :
           reduce.getInputs().take_front(reduce.getSourceCount())) {
        if (input != value)
          continue;
        auto fragment = dyn_cast<FragmentType>(input.getType());
        if (!fragment)
          continue;
        for (int64_t axis : reduce.getAxes()) {
          if (axis < 0 ||
              axis >= static_cast<int64_t>(fragment.getShape().size()))
            continue;
          PhysicalAxisProjection projection =
              queryFragmentAxis(fragment, source);
          if (projection.isExact() &&
              projection.fragmentAxis == static_cast<unsigned>(axis) &&
              projection.dimensionId == dimension)
            return true;
        }
      }
    }
    if (!visited.insert(user).second || user->getNumRegions() != 0 ||
        !isPhysicalReplayNode(user, PhysicalReplayScope::ValueGraph,
                              /*allowAccesses=*/false))
      continue;
    for (Value result : user->getResults())
      if (reachesReduction(result, source, dimension, visited))
        return true;
  }
  return false;
}

bool isExpensiveReplayProducer(Value value) {
  if (isa_and_nonnull<ContractOp, ReduceOp>(value.getDefiningOp()))
    return true;
  auto unary = value.getDefiningOp<UnaryOp>();
  if (!unary)
    return false;
  switch (unary.getOperatorKind()) {
  case UnaryOperator::Exp:
  case UnaryOperator::Exp2:
  case UnaryOperator::Log:
  case UnaryOperator::Log1p:
  case UnaryOperator::Lgamma:
  case UnaryOperator::Sin:
  case UnaryOperator::Cos:
  case UnaryOperator::Erf:
  case UnaryOperator::Erfc:
  case UnaryOperator::I0:
  case UnaryOperator::Rsqrt:
  case UnaryOperator::Sigmoid:
  case UnaryOperator::Tanh:
  case UnaryOperator::Sqrt:
    return true;
  case UnaryOperator::Negate:
  case UnaryOperator::Not:
  case UnaryOperator::Floor:
  case UnaryOperator::Abs:
    return false;
  }
  llvm_unreachable("unknown unary replay cost");
}

bool hasMaterializedReductionStoreFork(
    Value value, StoreOp current, PhysicalSourceAxis source, int64_t dimension,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (fragment) {
    PhysicalAxisProjection projection = queryFragmentAxis(fragment, source);
    if (projection.isExact() && projection.dimensionId == dimension) {
      llvm::SmallPtrSet<Operation *, 16> storeVisited;
      llvm::SmallPtrSet<Operation *, 16> reductionVisited;
      // The reduction already consumes the full-axis producer. Preserve its
      // SSA reuse even for this store when tiling would duplicate costly math.
      StoreOp excluded = isExpensiveReplayProducer(value) ? StoreOp() : current;
      if (reachesDifferentStore(value, excluded, source, dimension,
                                storeVisited) &&
          reachesReduction(value, source, dimension, reductionVisited))
        return true;
    }
  }
  Operation *producer = value.getDefiningOp();
  if (!producer || !visited.insert(producer).second ||
      !isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/true))
    return false;
  return llvm::any_of(producer->getOperands(), [&](Value operand) {
    return hasMaterializedReductionStoreFork(operand, current, source,
                                             dimension, visited);
  });
}

std::optional<unsigned> repeatedReductionOutputAxis(
    Value value, MakeRangeOp range) {
  auto reduce = value.getDefiningOp<ReduceOp>();
  if (!reduce || reduce.getSourceCount() != 1 || reduce->getNumResults() != 1 ||
      !isUnitStepRange(range))
    return std::nullopt;
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  auto input = dyn_cast<FragmentType>(reduce.getInputs().front().getType());
  auto output = queryFragmentAxis(value.getType(), source);
  if (!input || !output.isExact())
    return std::nullopt;
  SmallVector<unsigned> freeAxes;
  bool reducesSource = false;
  for (auto [axis, attribute] : llvm::enumerate(input.getAxisMaps())) {
    bool reduced = llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis));
    if (!reduced)
      freeAxes.push_back(axis);
    else
      reducesSource |= sourceAxisIdentity(cast<AxisMapAttr>(attribute)) == source;
  }
  if (!reducesSource || output.fragmentAxis >= freeAxes.size() ||
      !(sourceAxisIdentity(cast<AxisMapAttr>(
            input.getAxisMaps()[freeAxes[output.fragmentAxis]])) == source))
    return std::nullopt;
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  auto ranges = analysis.axisRanges(reduce.getInputs().front(),
                                    freeAxes[output.fragmentAxis]);
  if (!ranges.isExact() || !ranges.blockers.empty() || ranges.roots.empty())
    return std::nullopt;
  SmallVector<MakeRangeOp> authorities(ranges.roots.begin(), ranges.roots.end());
  authorities.push_back(range);
  if (!analysis.lockstepRanges(authorities).isExact())
    return std::nullopt;
  return output.fragmentAxis;
}

LogicalResult separateReductionOutputOccurrences(func::FuncOp kernel) {
  // A source reused in two operand positions can be both reduced and retained.
  // Give the retained occurrence its own derived range before ownership sees it.
  // The positional replay leaves the original occurrence available to other uses.
  uint64_t nextSource = nextPhysicalAxisIdentities(kernel).first;
  SmallVector<StoreOp> stores;
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  for (StoreOp store : stores) {
    SmallVector<Value> coordinates(store.getCoordinates());
    for (auto [position, coordinate] : llvm::enumerate(coordinates)) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type || type.getShape().size() != 1)
        continue;
      PhysicalProgramAnalysis analysis(kernel);
      auto address = analysis.axisRanges(coordinate, 0);
      auto range = queryExactLogicalRange(address);
      if (failed(range))
        continue;
      auto axis = repeatedReductionOutputAxis(store.getValue(), *range);
      if (!axis)
        continue;
      auto source = sourceAxisIdentity(*range);
      auto replay = analysis.replayability(
          store.getValue(), source, PhysicalReplayScope::ValueGraph,
          /*allowAccesses=*/true, store);
      auto replayable = [&](Value value) {
        return !value || analysis.replayability(
            value, source, PhysicalReplayScope::ValueGraph,
            /*allowAccesses=*/true, store).isReplayable();
      };
      if (!replay.isReplayable() || replay.crossesStructuredProgram ||
          !llvm::all_of(replay.contractions, [](Operation *operation) {
            return isa<ContractOp>(operation);
          }) ||
          llvm::any_of(replay.accesses, [&](Operation *access) {
            return llvm::any_of(access->getResults(), [&](Value value) {
              return queryFragmentAxes(value.getType(), source).size() > 1;
            });
          }) ||
          !replayable(coordinate) || !replayable(store.getValid()))
        continue;
      auto roots = analysis.axisRanges(store.getValue(), *axis);
      for (MakeRangeOp root : address.roots)
        if (!llvm::is_contained(roots.roots, root))
          roots.roots.push_back(root);
      auto extent = cast<PhysicalExprAttr>(type.getShape()[0]);
      if (cast<FragmentType>(store.getValue().getType()).getShape()[*axis] !=
              extent ||
          llvm::any_of(roots.roots, [&](MakeRangeOp root) {
            return root.getResult().getType().getShape()[0] != extent;
          }))
        continue;
      OpBuilder builder(store);
      IRMapping mapping;
      auto original = cast<AxisMapAttr>(type.getAxisMaps()[0]);
      auto selected = AxisMapAttr::get(
          kernel.getContext(), nextSource++, original.getSourceAxis(),
          original.getDimensionId(), 0, true);
      for (MakeRangeOp root : roots.roots) {
        auto fragment = cast<FragmentType>(root.getResult().getType());
        auto replacementType = FragmentType::get(
            kernel.getContext(), fragment.getElementType(), fragment.getShape(),
            builder.getArrayAttr({selected}), fragment.getValidity(),
            fragment.getOwner());
        auto replacement = builder.create<MakeRangeOp>(
            root.getLoc(), replacementType, root.getStart(), root.getExtent(),
            root.getStep(), root.getLogicalStart(), root.getLogicalStop(),
            selected.getSourceId(), selected.getSourceAxis(), true);
        inheritRangeAuthority(replacement, root);
        mapping.map(root.getResult(), replacement.getResult());
      }
      ReplayMaterializationOptions options;
      options.fragmentAxis = *axis;
      options.traversalRanges = roots.roots;
      options.segmentMapping = selected;
      auto payload = materializeReplayedValue(
          builder, store.getLoc(), store.getValue(), source, extent, mapping,
          options);
      if (failed(payload))
        return store.emitOpError("reduction output occurrence cannot be separated");
      options.fragmentAxis = 0;
      auto addressValue = materializeReplayedValue(
          builder, store.getLoc(), coordinate, source, extent, mapping, options);
      if (failed(addressValue))
        return store.emitOpError("reduction output address cannot be separated");
      if (store.getValid()) {
        options.fragmentAxis = *axis;
        auto valid = materializeReplayedValue(
            builder, store.getLoc(), store.getValid(), source, extent, mapping,
            options);
        if (failed(valid))
          return store.emitOpError("reduction output validity cannot be separated");
        store.getValidMutable().assign(*valid);
      }
      store.getValueMutable().assign(*payload);
      coordinates[position] = *addressValue;
      store.getCoordinatesMutable().assign(coordinates);
    }
  }
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult realizeReusePointwiseTraversal(func::FuncOp kernel,
                                             MakeRangeOp range,
                                             ArrayRef<StoreOp> stores,
                                             bool effectLocal,
                                             ArrayRef<LoadOp> retainedReads) {
  if (stores.empty())
    return range.emitOpError(
        "reuse-sensitive pointwise traversal has no write effect");
  FailureOr<SmallVector<int64_t>> traversalDimensions =
      traversalDimensionsForStores(stores, range);
  if (failed(traversalDimensions))
    return range.emitOpError(
        "reuse-sensitive pointwise traversal has no typed coordinate/data relation");
  bool accessDependentSubregion =
      !effectLocal && hasAccessDependentSubregionBounds(kernel, range);
  SmallVector<int64_t> payloadTraversalDimensions(*traversalDimensions);
  std::optional<int64_t> payloadTraversalDimension;
  if (accessDependentSubregion) {
    FailureOr<int64_t> dimension =
        reuseTraversalDimension(kernel, stores, range);
    if (failed(dimension))
      return range.emitOpError(
          "reuse-sensitive pointwise traversal has no unique reduced output dimension");
    payloadTraversalDimension = *dimension;
    payloadTraversalDimensions.assign(1, *dimension);
  }

  PhysicalSourceAxis logicalSource{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
  std::string name = ("POINTWISE_CHUNK_S" + Twine(logicalSource.sourceId) +
                      "_A" + Twine(logicalSource.sourceAxis))
                         .str();
  ParameterOp chunk;
  bool ambiguousChunk = false;
  kernel.walk([&](ParameterOp parameter) {
    auto source =
        parameter->getAttrOfType<PhysicalSourceAttr>(parameterSourceAttr);
    if (!parameter->hasAttr(pointwiseChunkAttr) || !source ||
        !(PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis(),
                             source.getDerived()} == logicalSource))
      return;
    if (chunk && chunk != parameter)
      ambiguousChunk = true;
    else
      chunk = parameter;
  });
  if (ambiguousChunk)
    return range.emitOpError(
        "pointwise traversal has multiple parameters for one source axis");
  if (!chunk) {
    SmallVector<int64_t> candidates{1,   2,   4,    8,    16,   32, 64,
                                   128, 256, 512, 1024, 2048, 4096};
    FailureOr<int64_t> staticExtent = exactStaticTraversalExtent(
        PhysicalProgramAnalysis(kernel).axisRanges(range.getResult(), 0));
    if (succeeded(staticExtent)) {
      uint64_t paddedExtent = llvm::PowerOf2Ceil(
          static_cast<uint64_t>(std::max<int64_t>(*staticExtent, 1)));
      llvm::erase_if(candidates, [&](int64_t candidate) {
        return static_cast<uint64_t>(candidate) > paddedExtent;
      });
    }
    uint32_t elementBitWidth = 0;
    for (StoreOp store : stores)
      elementBitWidth =
          std::max(elementBitWidth,
                   physicalElementBitWidth(store.getValue().getType()));
    if (elementBitWidth == 0)
      return range.emitOpError(
          "pointwise traversal has no typed data width for its physical parameter");
    OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), entry.getStringAttr(name),
        static_cast<uint32_t>(ParameterRole::OwnershipN),
        static_cast<uint32_t>(ParameterCategory::Pointwise), elementBitWidth,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    chunk = entry.create<ParameterOp>(range.getLoc(), entry.getIndexType(), schema);
    chunk->setAttr(parameterSourceAttr,
                   PhysicalSourceAttr::get(kernel.getContext(),
                                           logicalSource.sourceId,
                                           logicalSource.sourceAxis,
                                           logicalSource.derived));
    chunk->setAttr(pointwiseChunkAttr, entry.getUnitAttr());
  }

  auto originalType = dyn_cast<FragmentType>(range.getResult().getType());
  if (!originalType || originalType.getShape().size() != 1)
    return range.emitOpError(
        "reuse-sensitive pointwise traversal requires one physical source axis");
  PhysicalExprAttr chunkExtent = boundedTraversalChunk(chunk, range);
  SmallVector<Attribute> blockedMappings(originalType.getAxisMaps().begin(),
                                         originalType.getAxisMaps().end());
  if (accessDependentSubregion) {
    auto mapping = cast<AxisMapAttr>(blockedMappings[0]);
    blockedMappings[0] = AxisMapAttr::get(
        kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        *payloadTraversalDimension, 0, mapping.getDerived());
  }
  auto blockedType = FragmentType::get(
      kernel.getContext(), originalType.getElementType(),
      ArrayAttr::get(kernel.getContext(), {chunkExtent}),
      ArrayAttr::get(kernel.getContext(), blockedMappings),
      originalType.getValidity(),
      originalType.getOwner());
  // The original range remains the authority for the full reduction
  // traversal.  The internal writeback loop below owns a distinct blocked
  // range and replays only the store-side value graph against it.

  llvm::MapVector<Block *, SmallVector<StoreOp>> storesByBlock;
  for (StoreOp store : stores)
    storesByBlock[store->getBlock()].push_back(store);
  SmallVector<SmallVector<StoreOp>> storeGroups;
  PhysicalProgramAnalysis replayAnalysis(kernel);
  auto canReplayAt = [&](StoreOp store, Operation *anchor) {
    DominanceInfo dominance(kernel);
    for (LoadOp read : retainedReads)
      if (dominance.dominates(read.getResult(), store) &&
          !dominance.dominates(read.getResult(), anchor))
        return false;
    auto replayable = [&](Value value, ArrayRef<int64_t> dimensions) {
      if (!value)
        return true;
      SmallVector<int64_t> valueDimensions;
      collectTraversalDimensions(value.getType(), logicalSource,
                                 valueDimensions);
      for (int64_t dimension : valueDimensions)
        if (llvm::is_contained(dimensions, dimension) &&
            !replayAnalysis
                 .replayability(value, logicalSource,
                                PhysicalReplayScope::ValueGraph,
                                /*allowAccesses=*/true, anchor, dimension)
                 .isReplayable())
          return false;
      return true;
    };
    if (!replayable(store.getValue(), payloadTraversalDimensions) ||
        !replayable(store.getValid(), payloadTraversalDimensions))
      return false;
    return llvm::all_of(store.getCoordinates(), [&](Value coordinate) {
      return replayable(coordinate, *traversalDimensions);
    });
  };
  for (auto &entry : storesByBlock) {
    SmallVector<StoreOp> &blockStores = entry.second;
    SmallVector<bool> assigned(blockStores.size(), false);
    for (unsigned first = 0; first < blockStores.size(); ++first) {
      if (assigned[first])
        continue;
      SmallVector<StoreOp> group{blockStores[first]};
      assigned[first] = true;
      Operation *anchor = blockStores[first].getOperation();
      for (unsigned next = first + 1; next < blockStores.size(); ++next) {
        if (assigned[next] || !canReplayAt(blockStores[next], anchor))
          continue;
        group.push_back(blockStores[next]);
        assigned[next] = true;
      }
      storeGroups.push_back(std::move(group));
    }
  }
  SmallVector<scf::ForOp> materializedLoops;
  for (SmallVector<StoreOp> &group : storeGroups) {
  OpBuilder builder(group.front());
  Operation *loopInsertionAnchor = group.front().getOperation();
  Value stop = range.getLogicalStop();
  Value chunkSize = builder.create<PhysicalExprOp>(
      range.getLoc(), builder.getIndexType(), chunkExtent);
  Value loopStep = builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), chunkSize, range.getStep(),
      BinaryOperator::Multiply);
  bool bodyFailed = false;
  std::string failureReason;
  auto loop = builder.create<scf::ForOp>(
      range.getLoc(), range.getStart(), stop, loopStep, ValueRange{},
      [&](OpBuilder &nested, Location location, Value tileStart, ValueRange) {
        Value blocked = nested.create<MakeRangeOp>(
            location, blockedType, tileStart, chunkSize, range.getStep(),
            range.getLogicalStart(), range.getLogicalStop(), range.getSourceId(),
            range.getSourceAxis(), range.getDerived());
        inheritRangeAuthority(blocked.getDefiningOp(), range);
        Value end = nested.create<BroadcastOp>(location, blockedType, stop);
        auto tailComparison = nested.create<CompareOp>(
            location, predicateType(blockedType), blocked, end,
            ComparePredicate::Lt);
        tailComparison->setAttr(physicalTailAttr, nested.getUnitAttr());
        Value tail = tailComparison.getResult();
        IRMapping mapping;
        mapping.map(range.getResult(), blocked);
        for (LoadOp read : retainedReads) {
          if (!DominanceInfo(kernel).dominates(read.getResult(), loopInsertionAnchor))
            continue;
          auto slice = materializeRetainedSlice(
              nested, location, read.getResult(), 0, chunkExtent, blocked,
              loopInsertionAnchor);
          if (failed(slice)) {
            bodyFailed = true;
            failureReason = "retained input cannot be sliced in the writeback loop";
            return;
          }
          mapping.map(read.getResult(), *slice);
        }
        for (StoreOp store : group) {
          FailureOr<Value> payload = replayPointwiseValue(
              nested, store.getValue(), logicalSource,
              payloadTraversalDimensions,
              chunkExtent,
              blocked, tail, loopInsertionAnchor, mapping);
          if (failed(payload)) {
            bodyFailed = true;
            failureReason = "write payload cannot be replayed in the internal tile loop";
            return;
          }
          SmallVector<Value> coordinates;
          for (Value coordinate : store.getCoordinates()) {
            FailureOr<Value> replayed = replayPointwiseValue(
                nested, coordinate, logicalSource, *traversalDimensions,
                chunkExtent, blocked,
                tail, loopInsertionAnchor, mapping);
            if (failed(replayed)) {
              bodyFailed = true;
              failureReason =
                  "write coordinate cannot be replayed in the internal tile loop";
              return;
            }
            coordinates.push_back(*replayed);
          }
          auto payloadType = dyn_cast<FragmentType>((*payload).getType());
          if (!payloadType) {
            FailureOr<FragmentType> schema =
                coordinateValueSchema((*payload).getType(), coordinates);
            if (failed(schema)) {
              bodyFailed = true;
              failureReason =
                  "write coordinates have no exact payload fragment schema";
              return;
            }
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                nested, location, *payload, *schema);
            if (failed(projected)) {
              bodyFailed = true;
              failureReason =
                  "write payload cannot adopt its coordinate fragment schema";
              return;
            }
            payload = *projected;
            payloadType = *schema;
          }
          FailureOr<Value> valid = materializeBroadcastToFragment(nested, location, tail,
                                               predicateType(payloadType));
          if (failed(valid)) {
            bodyFailed = true;
            failureReason = "tile tail cannot be projected to the write payload";
            return;
          }
          if (store.getValid()) {
            FailureOr<Value> existing = replayPointwiseValue(
                nested, store.getValid(), logicalSource,
                payloadTraversalDimensions,
                chunkExtent,
                blocked, tail, loopInsertionAnchor, mapping);
            if (failed(existing)) {
              bodyFailed = true;
              failureReason = "write validity cannot be replayed in the tile loop";
              return;
            }
            FailureOr<Value> projected = materializeBroadcastToFragment(
                nested, location, *existing, predicateType(payloadType));
            if (failed(projected)) {
              bodyFailed = true;
              failureReason = "write validity cannot be projected to the payload";
              return;
            }
            valid = Value(nested.create<BinaryOp>(
                location, predicateType(payloadType), *valid, *projected,
                BinaryOperator::LogicalAnd));
          }
          auto replacement = nested.create<StoreOp>(
              location, store.getResource(), coordinates, *payload, *valid,
              store.getSourceAxes());
          if (Attribute origin = store->getAttr(originAttr))
            replacement->setAttr(originAttr, origin);
        }
        if (!bodyFailed)
          nested.create<scf::YieldOp>(location);
      });
  if (bodyFailed) {
    loop.erase();
    for (scf::ForOp materialized : materializedLoops)
      materialized.erase();
    return range.emitOpError(
               "reuse-sensitive pointwise traversal could not be materialized: ")
           << failureReason;
  }
  materializedLoops.push_back(loop);
  }
  for (StoreOp store : stores)
    store.erase();
  return success();
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto kernel = coordinate.getParentRegion()->getParentOfType<func::FuncOp>();
  auto fragment = dyn_cast<FragmentType>(coordinate.getType());
  if (!kernel || !fragment)
    return;
  PhysicalProgramAnalysis analysis(kernel);
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(coordinate, axis);
    if (fact.state != PhysicalFactState::Exact)
      continue;
    for (MakeRangeOp range : fact.roots)
      ranges.insert(range.getOperation());
  }
}

void collectStoreRanges(Value value, llvm::SmallPtrSetImpl<Operation *> &ranges,
                        llvm::SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<CastOp, BroadcastOp>(user)) {
      collectStoreRanges(user->getResult(0), ranges, visited);
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getValue() != value)
      continue;
    for (Value coordinate : store.getCoordinates())
      collectCoordinateRanges(coordinate, ranges);
  }
}

HistogramOp histogramSource(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto histogram = dyn_cast<HistogramOp>(producer))
      return histogram;
    if (!isa<BroadcastOp, CastOp>(producer) || producer->getNumOperands() != 1 ||
        producer->getNumResults() != 1)
      return HistogramOp();
    value = producer->getOperand(0);
  }
  return HistogramOp();
}

LogicalResult alignHistogramOutputOwnership(func::FuncOp kernel) {
  SmallVector<HistogramOp> histograms;
  kernel.walk([&](HistogramOp histogram) { histograms.push_back(histogram); });
  for (HistogramOp histogram : histograms) {
    SmallVector<StoreOp> stores;
    kernel.walk([&](StoreOp store) {
      if (histogramSource(store.getValue()) == histogram)
        stores.push_back(store);
    });
    if (stores.empty())
      return histogram.emitOpError(
          "histogram result has no physical output ownership effect");
    FailureOr<FragmentType> outputType = coordinateValueSchema(
        histogram.getResult().getType().getElementType(),
        stores.front().getCoordinates());
    if (failed(outputType) || outputType->getShape().size() != 1)
      return histogram.emitOpError(
          "histogram output has no one-axis physical ownership schema");
    for (StoreOp store : llvm::drop_begin(stores)) {
      FailureOr<FragmentType> current = coordinateValueSchema(
          histogram.getResult().getType().getElementType(),
          store.getCoordinates());
      if (failed(current) || *current != *outputType)
        return histogram.emitOpError(
            "histogram output effects do not share one physical ownership schema");
    }
    histogram.getResult().setType(*outputType);
  }
  return success();
}

LogicalResult realizeOwnedHistograms(func::FuncOp kernel) {
  SmallVector<HistogramOp> histograms;
  kernel.walk([&](HistogramOp histogram) { histograms.push_back(histogram); });
  for (HistogramOp histogram : histograms) {
    SmallVector<StoreOp> stores;
    kernel.walk([&](StoreOp store) {
      if (histogramSource(store.getValue()) == histogram)
        stores.push_back(store);
    });
    if (stores.empty())
      return histogram.emitOpError(
          "histogram result has no physical output ownership effect");

    FailureOr<FragmentType> outputType = coordinateValueSchema(
        histogram.getResult().getType().getElementType(),
        stores.front().getCoordinates());
    if (failed(outputType) || outputType->getShape().size() != 1)
      return histogram.emitOpError(
          "histogram output has no one-axis physical ownership schema");
    for (StoreOp store : llvm::drop_begin(stores)) {
      FailureOr<FragmentType> current = coordinateValueSchema(
          histogram.getResult().getType().getElementType(),
          store.getCoordinates());
      if (failed(current) || *current != *outputType)
        return histogram.emitOpError(
            "histogram output effects do not share one physical ownership schema");
    }

    auto outputMapping = cast<AxisMapAttr>(outputType->getAxisMaps()[0]);
    PhysicalSourceAxis outputSource{outputMapping.getSourceId(),
                                    outputMapping.getSourceAxis(),
                                    outputMapping.getDerived()};
    llvm::SmallPtrSet<Operation *, 8> outputRoots;
    for (Value coordinate : stores.front().getCoordinates())
      collectCoordinateRanges(coordinate, outputRoots);
    SmallVector<MakeRangeOp> outputRanges;
    for (Operation *root : outputRoots)
      if (auto range = dyn_cast<MakeRangeOp>(root);
          range && sourceAxisIdentity(range) == outputSource)
        outputRanges.push_back(range);
    if (outputRanges.size() != 1 || !isUnitStepRange(outputRanges.front()))
      return histogram.emitOpError(
          "histogram output ownership has no unique unit-step bin range");
    MakeRangeOp outputRange = outputRanges.front();
    for (StoreOp store : llvm::drop_begin(stores)) {
      llvm::SmallPtrSet<Operation *, 8> currentRoots;
      for (Value coordinate : store.getCoordinates())
        collectCoordinateRanges(coordinate, currentRoots);
      SmallVector<MakeRangeOp> currentRanges;
      for (Operation *root : currentRoots)
        if (auto range = dyn_cast<MakeRangeOp>(root);
            range && sourceAxisIdentity(range) == outputSource)
          currentRanges.push_back(range);
      if (currentRanges.size() != 1 ||
          !PhysicalProgramAnalysis(kernel)
               .lockstepRanges({outputRange, currentRanges.front()})
               .isExact())
        return histogram.emitOpError(
            "histogram output effects do not share one physical bin traversal");
    }
    FailureOr<ParameterOp> outputParameter =
        queryBlockingParameter(kernel, outputRange);
    if (failed(outputParameter))
      return histogram.emitOpError(
          "histogram output ownership has no typed blocking parameter");
    ParameterAttr outputSchema = outputParameter->getParameter();
    SmallVector<int64_t> outputCandidates(
        outputSchema.getCandidates().asArrayRef());
    if (auto bins = histogram.getBins().getDefiningOp<arith::ConstantIndexOp>())
      llvm::erase_if(outputCandidates,
                     [&](int64_t candidate) { return candidate > bins.value(); });
    if (outputCandidates.empty())
      return histogram.emitOpError(
          "histogram output ownership has no legal bin-tile candidate");
    (*outputParameter)->setAttr(
        "parameter",
        ParameterAttr::get(
            kernel.getContext(), outputSchema.getName(), outputSchema.getRole(),
            outputSchema.getCategory(),
            outputSchema.getElementBitWidth(),
            DenseI64ArrayAttr::get(kernel.getContext(), outputCandidates)));

    auto valuesType = dyn_cast<FragmentType>(histogram.getValues().getType());
    if (!valuesType || valuesType.getShape().size() != 1)
      return histogram.emitOpError(
          "histogram input has no one-axis physical traversal schema");
    PhysicalRangeFact inputFact =
        PhysicalProgramAnalysis(kernel).axisRanges(histogram.getValues(), 0);
    if (!inputFact.isUnique() || !isUnitStepRange(inputFact.roots.front()))
      return histogram.emitOpError(
          "histogram input has no unique unit-step physical traversal");
    MakeRangeOp inputRange = inputFact.roots.front();
    FailureOr<int64_t> inputDimension = queryRangeDimension(inputRange);
    if (failed(inputDimension))
      return histogram.emitOpError(
          "histogram input traversal has no logical dimension identity");
    PhysicalSourceAxis inputSource = sourceAxisIdentity(inputRange);

    std::string chunkName =
        ("HISTOGRAM_CHUNK_S" + Twine(inputSource.sourceId) + "_A" +
         Twine(inputSource.sourceAxis) +
         (inputSource.derived ? "_DERIVED" : ""))
            .str();
    ParameterOp chunk = getOrCreatePhysicalParameter(
        kernel, chunkName, ParameterRole::Reduction,
        ParameterCategory::Histogram,
        valuesType.getElementType().getIntOrFloatBitWidth(),
        {256, 512, 1024, 2048, 4096, 8192, 16384, 32768});
    if (!chunk)
      return failure();
    chunk->setAttr(dimensionAttr,
                   IntegerAttr::get(IntegerType::get(kernel.getContext(), 64),
                                    *inputDimension));

    PhysicalExprAttr chunkExtent = expression(
        kernel.getContext(), PhysicalExprKind::Parameter, 0, chunkName);
    auto inputRangeType = cast<FragmentType>(inputRange.getResult().getType());
    auto blockedInputType = FragmentType::get(
        kernel.getContext(), inputRangeType.getElementType(),
        ArrayAttr::get(kernel.getContext(), {chunkExtent}),
        inputRangeType.getAxisMaps(), inputRangeType.getValidity(),
        inputRangeType.getOwner());

    Block *outputBlock = stores.front()->getBlock();
    if (llvm::any_of(stores, [&](StoreOp store) {
          return store->getBlock() != outputBlock;
        }))
      return histogram.emitOpError(
          "histogram output effects do not share one physical control block");
    OpBuilder builder(stores.front());
    auto countType = dyn_cast<IntegerType>(outputType->getElementType());
    if (!countType)
      return histogram.emitOpError(
          "histogram count type has no integer zero identity");
    Value zeroScalar = builder.create<arith::ConstantOp>(
        histogram.getLoc(), countType, builder.getIntegerAttr(countType, 0));
    Value zero = builder.create<SplatOp>(histogram.getLoc(), *outputType,
                                         zeroScalar);
    Value loopStep = builder.create<BinaryOp>(
        histogram.getLoc(), builder.getIndexType(), chunk.getResult(),
        inputRange.getStep(), BinaryOperator::Multiply);
    bool bodyFailed = false;
    std::string failureReason;
    auto loop = builder.create<scf::ForOp>(
        histogram.getLoc(), inputRange.getStart(), inputRange.getLogicalStop(),
        loopStep, ValueRange{zero},
        [&](OpBuilder &nested, Location location, Value tileStart,
            ValueRange carries) {
          Value blocked = nested.create<MakeRangeOp>(
              location, blockedInputType, tileStart, chunk.getResult(),
              inputRange.getStep(), inputRange.getLogicalStart(),
              inputRange.getLogicalStop(), inputRange.getSourceId(),
              inputRange.getSourceAxis(), inputRange.getDerived());
          inheritRangeAuthority(blocked.getDefiningOp(), inputRange);
          Value inputEnd = nested.create<BroadcastOp>(
              location, blockedInputType, inputRange.getLogicalStop());
          auto tailComparison = nested.create<CompareOp>(
              location, predicateType(blockedInputType), blocked, inputEnd,
              ComparePredicate::Lt);
          tailComparison->setAttr(physicalTailAttr, nested.getUnitAttr());
          Value tail = tailComparison.getResult();

          IRMapping mapping;
          mapping.map(inputRange.getResult(), blocked);
          SmallVector<int64_t> traversalDimensions{*inputDimension};
          FailureOr<Value> values = replayPointwiseValue(
              nested, histogram.getValues(), inputSource, traversalDimensions,
              chunkExtent, blocked, tail, histogram.getOperation(), mapping);
          FailureOr<Value> valid = replayPointwiseValue(
              nested, histogram.getValid(), inputSource, traversalDimensions,
              chunkExtent, blocked, tail, histogram.getOperation(), mapping);
          if (failed(values) || failed(valid)) {
            bodyFailed = true;
            failureReason =
                "input values cannot be replayed in the histogram traversal loop";
            return;
          }
          auto blockedValuesType = dyn_cast<FragmentType>((*values).getType());
          if (!blockedValuesType) {
            bodyFailed = true;
            failureReason =
                "replayed histogram values lost their physical fragment schema";
            return;
          }
          FragmentType predicate = predicateType(blockedValuesType);
          FailureOr<Value> projectedValid = materializeBroadcastToFragment(
              nested, location, *valid, predicate);
          FailureOr<Value> projectedTail =
              materializeBroadcastToFragment(nested, location, tail, predicate);
          if (failed(projectedValid) || failed(projectedTail)) {
            bodyFailed = true;
            failureReason =
                "histogram input validity cannot adopt its blocked traversal schema";
            return;
          }

          Value outputOffset = nested.create<BinaryOp>(
              location, nested.getIndexType(), outputRange.getStart(),
              outputRange.getLogicalStart(), BinaryOperator::Subtract);
          Value outputEnd = nested.create<BinaryOp>(
              location, nested.getIndexType(), outputOffset,
              outputRange.getExtent(), BinaryOperator::Add);
          Type inputElement = blockedValuesType.getElementType();
          Value typedStart = nested.create<CastOp>(
              location, inputElement, outputOffset);
          Value typedEnd =
              nested.create<CastOp>(location, inputElement, outputEnd);
          FailureOr<Value> start = materializeBroadcastToFragment(
              nested, location, typedStart, blockedValuesType);
          FailureOr<Value> end = materializeBroadcastToFragment(
              nested, location, typedEnd, blockedValuesType);
          if (failed(start) || failed(end)) {
            bodyFailed = true;
            failureReason =
                "histogram bin ownership cannot project onto the input values";
            return;
          }
          Value lower = nested.create<CompareOp>(
              location, predicate, *values, *start, ComparePredicate::Ge);
          Value upper = nested.create<CompareOp>(
              location, predicate, *values, *end, ComparePredicate::Lt);
          Value active = nested.create<BinaryOp>(
              location, predicate, *projectedValid, *projectedTail,
              BinaryOperator::LogicalAnd);
          active = nested.create<BinaryOp>(location, predicate, active, lower,
                                           BinaryOperator::LogicalAnd);
          active = nested.create<BinaryOp>(location, predicate, active, upper,
                                           BinaryOperator::LogicalAnd);
          Value localValues = nested.create<BinaryOp>(
              location, blockedValuesType, *values, *start,
              BinaryOperator::Subtract);
          auto partial = nested.create<HistogramOp>(
              location, *outputType, localValues, outputRange.getExtent(),
              active);
          if (Attribute origin = histogram->getAttr(originAttr))
            partial->setAttr(originAttr, origin);
          Value accumulated = nested.create<BinaryOp>(
              location, *outputType, carries.front(), partial.getResult(),
              BinaryOperator::Add);
          nested.create<scf::YieldOp>(location, accumulated);
        });
    if (bodyFailed) {
      loop.erase();
      return histogram.emitOpError(
                 "histogram ownership could not be materialized: ")
             << failureReason;
    }
    for (StoreOp store : stores)
      store.getValueMutable().assign(loop.getResult(0));
  }
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult alignContractAccumulatorTypes(func::FuncOp kernel) {
  auto align = [](Operation *owner, OpOperand &accumulatorOperand, Value result,
                  bool &changed) -> LogicalResult {
    Value accumulator = accumulatorOperand.get();
    if (accumulator.getType() == result.getType())
      return success();
    changed = true;
    auto source = dyn_cast<FragmentType>(accumulator.getType());
    auto target = dyn_cast<FragmentType>(result.getType());
    if (!source || !target || source.getElementType() != target.getElementType() ||
        source.getOwner() != target.getOwner())
      return owner->emitOpError(
          "pointwise ownership cannot preserve the contract accumulator relation");
    if (isLiteralZeroProjection(accumulator)) {
      OpBuilder builder(owner);
      auto projected = projectPhysicalValueToSchema(
          builder, owner->getLoc(), accumulator, target);
      if (failed(projected))
        return owner->emitOpError("contract zero accumulator cannot adopt its result schema");
      accumulatorOperand.set(*projected);
      return success();
    }
    if (source.getAxisMaps() != target.getAxisMaps())
      return owner->emitOpError(
          "pointwise ownership cannot preserve the contract accumulator relation");
    auto isUnit = [](Attribute attribute) {
      auto expression = cast<PhysicalExprAttr>(attribute);
      return expression.getKind() ==
                 static_cast<uint32_t>(PhysicalExprKind::Constant) &&
             expression.getValue() == 1;
    };
    SmallVector<Attribute> shape(target.getShape().begin(),
                                 target.getShape().end());
    for (unsigned axis = 0; axis < shape.size(); ++axis) {
      if (source.getShape()[axis] == target.getShape()[axis])
        continue;
      bool sourceUnit = isUnit(source.getShape()[axis]);
      bool targetUnit = isUnit(target.getShape()[axis]);
      if (sourceUnit == targetUnit)
        return owner->emitOpError(
            "pointwise ownership found two non-equivalent contract extents");
      if (targetUnit)
        shape[axis] = source.getShape()[axis];
    }
    auto aligned = FragmentType::get(
        target.getContext(), target.getElementType(),
        ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
        target.getValidity(), target.getOwner());
    for (auto [axis, mapping] : llvm::enumerate(aligned.getAxisMaps())) {
      if (source.getShape()[axis] == aligned.getShape()[axis] &&
          target.getShape()[axis] == aligned.getShape()[axis])
        continue;
      int64_t dimension = cast<AxisMapAttr>(mapping).getDimensionId();
      if (dimension <= 0)
        return owner->emitOpError(
            "contract accumulator alignment has no dimension authority");
      retargetDimensionExtent(
          result, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]));
      retargetDimensionExtent(
          accumulator, dimension,
          cast<PhysicalExprAttr>(aligned.getShape()[axis]));
    }
    accumulator.setType(aligned);
    result.setType(aligned);
    return success();
  };
  bool changed;
  do {
    changed = false;
    WalkResult result = kernel.walk([&](Operation *operation) {
      OpOperand *accumulator;
      Value output;
      if (auto contract = dyn_cast<ContractOp>(operation)) {
        accumulator = &contract.getAccumulatorMutable();
        output = contract.getResult();
      } else if (auto contract = dyn_cast<ScaledContractOp>(operation)) {
        accumulator = &contract.getAccumulatorMutable();
        output = contract.getResult();
      } else if (auto contract = dyn_cast<SparseContractOp>(operation)) {
        accumulator = &contract.getAccumulatorMutable();
        output = contract.getResult();
      } else {
        return WalkResult::advance();
      }
      return failed(align(operation, *accumulator, output, changed))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  } while (changed);
  return success();
}

LogicalResult alignOrdinaryContractOperandTypes(func::FuncOp kernel) {
  auto isUnit = [](Attribute attribute) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    return extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           extent.getValue() == 1;
  };
  auto isUniformBatch = [&](Value value, unsigned axis) {
    auto broadcast = value.getDefiningOp<BroadcastOp>();
    auto source = broadcast
                      ? dyn_cast<FragmentType>(broadcast.getValue().getType())
                      : FragmentType();
    auto target = cast<FragmentType>(value.getType());
    if (!source)
      return false;
    auto projection = queryBroadcastProjection(source, target);
    auto mapping = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
    if (!projection.isExact() || projection.targetToSource[axis] ||
        queryFragmentAxes(target, sourceAxisIdentity(mapping)).size() != 1)
      return false;
    auto ranges = PhysicalProgramAnalysis(kernel).axisRanges(value, axis);
    return ranges.isExact() && ranges.roots.empty() && ranges.blockers.empty();
  };
  WalkResult result = kernel.walk([&](ContractOp contract) {
    auto alignPairs = [&](Value lhs, Value rhs, ArrayRef<int64_t> lhsAxes,
                          ArrayRef<int64_t> rhsAxes, bool batch) -> LogicalResult {
      if (lhsAxes.size() != rhsAxes.size())
        return failure();
      for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsAxes, rhsAxes)) {
        auto lhsType = cast<FragmentType>(lhs.getType());
        auto rhsType = cast<FragmentType>(rhs.getType());
        if (lhsAxis < 0 || rhsAxis < 0 ||
            lhsAxis >= static_cast<int64_t>(lhsType.getShape().size()) ||
            rhsAxis >= static_cast<int64_t>(rhsType.getShape().size()))
          return failure();
        Attribute lhsExtent = lhsType.getShape()[lhsAxis];
        Attribute rhsExtent = rhsType.getShape()[rhsAxis];
        if (lhsExtent == rhsExtent)
          continue;
        bool lhsUnit = isUnit(lhsExtent);
        bool rhsUnit = isUnit(rhsExtent);
        bool rebindLhs = lhsUnit;
        bool lhsUniform = batch && isUniformBatch(lhs, lhsAxis);
        bool rhsUniform = batch && isUniformBatch(rhs, rhsAxis);
        if (lhsUnit == rhsUnit && lhsUniform == rhsUniform)
          return contract.emitOpError(
                     "ordinary contract paired axes have conflicting physical extents")
                 << "; lhs_axis=" << lhsAxis << "; lhs_extent=" << lhsExtent
                 << "; rhs_axis=" << rhsAxis << "; rhs_extent=" << rhsExtent;
        if (lhsUnit == rhsUnit)
          rebindLhs = lhsUniform;
        if (rebindLhs) {
          auto mapping = cast<AxisMapAttr>(lhsType.getAxisMaps()[lhsAxis]);
          retargetSourceExtent(lhs, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(rhsExtent));
        } else {
          auto mapping = cast<AxisMapAttr>(rhsType.getAxisMaps()[rhsAxis]);
          retargetSourceExtent(rhs, sourceAxisIdentity(mapping),
                               cast<PhysicalExprAttr>(lhsExtent));
        }
      }
      return success();
    };
    if (failed(alignPairs(contract.getLhs(), contract.getRhs(),
                          contract.getLhsReductionAxes(),
                          contract.getRhsReductionAxes(), false)) ||
        failed(alignPairs(contract.getLhs(), contract.getRhs(),
                          contract.getLhsBatchAxes(),
                          contract.getRhsBatchAxes(), true)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool isCartesianPointwiseValueOp(Operation *operation) {
  return isa<SplatOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
             CastOp, BitcastOp, LoadOp, GatherOp, RandomBitsOp>(operation);
}

bool isStructuredFreeAxisValueOp(Operation *operation) {
  return isCartesianPointwiseValueOp(operation) ||
         isa<ReshapeOp, TransposeOp, ContractOp, ReduceOp, MakeRecordOp,
             ExtractOp>(operation);
}

bool isStaticUnitExtent(Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  return extent &&
         extent.getKind() ==
             static_cast<uint32_t>(PhysicalExprKind::Constant) &&
         extent.getValue() == 1;
}

bool analyzeStructuredFreeBlock(Block &block,
                                ArrayRef<unsigned> dependentArguments,
                                SmallVectorImpl<bool> &dependentResults,
                                bool &sawContract) {
  llvm::SmallDenseSet<Value> dependent;
  for (unsigned index : dependentArguments) {
    if (index >= block.getNumArguments())
      return false;
    dependent.insert(block.getArgument(index));
  }
  for (Operation &operation : block) {
    if (auto yield = dyn_cast<YieldOp>(operation)) {
      dependentResults.clear();
      for (Value value : yield.getValues())
        dependentResults.push_back(dependent.contains(value));
      return true;
    }
    bool operationDepends =
        llvm::any_of(operation.getOperands(), [&](Value operand) {
          return dependent.contains(operand);
        });
    if (!operationDepends)
      continue;
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      bool lhs = dependent.contains(contract.getLhs());
      bool rhs = dependent.contains(contract.getRhs());
      if (lhs == rhs || dependent.contains(contract.getAccumulator()))
        return false;
      sawContract = true;
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      ValueRange sources =
          reduce.getInputs().take_front(reduce.getSourceCount());
      ValueRange boundaries =
          reduce.getInputs().drop_front(reduce.getSourceCount());
      if (!llvm::any_of(sources, [&](Value value) {
            return dependent.contains(value);
          }) ||
          llvm::any_of(boundaries, [&](Value value) {
            return dependent.contains(value);
          }))
        return false;
    } else if (auto gather = dyn_cast<GatherOp>(operation)) {
      auto source = dyn_cast<FragmentType>(gather.getSource().getType());
      if (!source || !dependent.contains(gather.getSource()) ||
          llvm::any_of(gather.getCoordinates(), [&](Value value) {
            return dependent.contains(value);
          }))
        return false;
      for (int64_t sourceAxis : gather.getSourceAxes())
        if (sourceAxis < 0 ||
            sourceAxis >= static_cast<int64_t>(source.getShape().size()) ||
            !isStaticUnitExtent(source.getShape()[sourceAxis]))
          return false;
    } else if (!isStructuredFreeAxisValueOp(&operation) ||
               operation.getNumRegions() != 0) {
      return false;
    }
    if (operation.getNumResults() == 0)
      return false;
    for (Value result : operation.getResults())
      dependent.insert(result);
  }
  return false;
}

bool analyzeStructuredRegionFold(
    RegionFoldOp fold, const std::function<bool(Value)> &depends,
    SmallVectorImpl<bool> &dependentResults, bool &sawContract) {
  if (!llvm::hasSingleElement(fold.getSummarize()) ||
      !llvm::hasSingleElement(fold.getCombine()))
    return false;
  unsigned sourceCount = fold.getSourceCount();
  unsigned identityCount = fold.getIdentityCount();
  unsigned captureCount = fold.getCaptureCount();
  ValueRange inputs = fold.getInputs();
  if (inputs.size() != sourceCount + identityCount + captureCount)
    return false;
  if (llvm::any_of(inputs.take_front(sourceCount + identityCount), depends))
    return false;

  SmallVector<unsigned> summarizeArguments;
  for (unsigned capture = 0; capture < captureCount; ++capture)
    if (depends(inputs[sourceCount + identityCount + capture]))
      summarizeArguments.push_back(sourceCount + capture);
  if (summarizeArguments.empty())
    return false;
  if (!analyzeStructuredFreeBlock(fold.getSummarize().front(),
                                  summarizeArguments, dependentResults,
                                  sawContract) ||
      dependentResults.size() != identityCount ||
      !llvm::any_of(dependentResults, [](bool value) { return value; }))
    return false;

  SmallVector<unsigned> combineArguments;
  for (auto [index, dependent] : llvm::enumerate(dependentResults))
    if (dependent) {
      combineArguments.push_back(index);
      combineArguments.push_back(identityCount + index);
    }
  SmallVector<bool> combinedResults;
  if (!analyzeStructuredFreeBlock(fold.getCombine().front(), combineArguments,
                                  combinedResults, sawContract) ||
      combinedResults != dependentResults)
    return false;
  return true;
}

FailureOr<bool> propagateOrderedCarryDependency(
    Value value, Operation *user, llvm::function_ref<void(Value)> enqueue,
    llvm::SmallPtrSetImpl<Operation *> &loops) {
  auto propagateWhile = [&](scf::WhileOp loop) -> FailureOr<bool> {
    if (!canPredicateScalarWhile(loop))
      return failure();
    loops.insert(loop);
    // A varying trip count makes even an initially uniform recurrence vary.
    for (Value argument : loop.getBeforeArguments())
      enqueue(argument);
    for (Value argument : loop.getAfterArguments())
      enqueue(argument);
    for (Value result : loop.getResults())
      enqueue(result);
    return true;
  };
  if (auto loop = dyn_cast<scf::WhileOp>(user))
    return propagateWhile(loop);
  if (auto condition = dyn_cast<scf::ConditionOp>(user))
    return propagateWhile(cast<scf::WhileOp>(condition->getParentOp()));
  if (auto yield = dyn_cast<scf::YieldOp>(user)) {
    if (auto loop = dyn_cast<scf::WhileOp>(yield->getParentOp()))
      return propagateWhile(loop);
    auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
    if (!loop)
      return failure();
    loops.insert(loop);
    for (auto [index, operand] : llvm::enumerate(yield.getResults()))
      if (operand == value) {
        enqueue(loop.getRegionIterArgs()[index]);
        enqueue(loop.getResult(index));
      }
    return true;
  }
  if (auto loop = dyn_cast<scf::ForOp>(user)) {
    if (value == loop.getLowerBound() || value == loop.getUpperBound() ||
        value == loop.getStep())
      return failure();
    loops.insert(loop);
    for (auto [index, operand] : llvm::enumerate(loop.getInitArgs()))
      if (operand == value) {
        enqueue(loop.getRegionIterArgs()[index]);
        enqueue(loop.getResult(index));
      }
    return true;
  }
  return false;
}

bool hasSupportedOrderedBodies(
    const llvm::SmallPtrSetImpl<Operation *> &loops,
    llvm::function_ref<bool(Value)> ownsCoordinate) {
  for (Operation *loop : loops) {
    WalkResult effects = loop->walk([&](Operation *operation) {
      if (isa<scf::ForOp, ReduceOp>(operation))
        return WalkResult::advance();
      if (auto whileLoop = dyn_cast<scf::WhileOp>(operation))
        return canPredicateScalarWhile(whileLoop) ? WalkResult::advance()
                                                 : WalkResult::interrupt();
      if (operation->getNumRegions() != 0)
        return WalkResult::interrupt();
      if (auto store = dyn_cast<StoreOp>(operation)) {
        if (llvm::none_of(store.getCoordinates(), ownsCoordinate))
          return WalkResult::interrupt();
        if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
            buffer && buffer.getScope().getValue() !=
                          BufferScope::InvocationWorkspace)
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      return isa<scf::YieldOp, scf::ConditionOp, LoadOp>(operation) ||
                     isMemoryEffectFree(operation)
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (effects.wasInterrupted())
      return false;
  }
  return true;
}

bool supportsCartesianPointwiseValueGraph(
    ArrayRef<WorksetCoordinateOp> coordinates, bool allowOrderedLoops = false) {
  llvm::SmallDenseSet<Value> dependent;
  SmallVector<Value> worklist;
  auto enqueue = [&](Value value) {
    if (dependent.insert(value).second)
      worklist.push_back(value);
  };
  for (WorksetCoordinateOp coordinate : coordinates) {
    enqueue(coordinate.getResult());
  }
  llvm::SmallPtrSet<Operation *, 32> visited;
  llvm::SmallPtrSet<Operation *, 8> loops;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (allowOrderedLoops) {
        FailureOr<bool> propagated =
            propagateOrderedCarryDependency(value, user, enqueue, loops);
        if (failed(propagated))
          return false;
        if (*propagated)
          continue;
      }
      if (!visited.insert(user).second)
        continue;
      if (user->getNumResults() == 0) {
        // A terminal pointwise store can consume the promoted fragment.  A
        // control-flow terminator or effectful region cannot: promoting the
        // predicate/carry would change scalar program structure into a lane
        // program without a physical control-flow realization.
        auto store = dyn_cast<StoreOp>(user);
        if (!store || user->getNumRegions() != 0)
          return false;
        // Coarsening program ownership does not replicate local allocation
        // instances. Keep those stores scalar until storage is made explicit.
        if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
            buffer && buffer.getScope().getValue() !=
                          BufferScope::InvocationWorkspace)
          return false;
        continue;
      }
      if (!isCartesianPointwiseValueOp(user) || user->getNumRegions() != 0)
        return false;
      for (Value result : user->getResults()) {
        Type type = result.getType();
        if (!isa<IntegerType, FloatType, IndexType, FragmentType>(type))
          return false;
        enqueue(result);
      }
    }
  }
  return hasSupportedOrderedBodies(loops, [&](Value value) {
    return dependent.contains(value);
  });
}

/// A structured free workset axis is still lane-wise: it may flow through
/// pointwise operations, remain free across reductions, and occupy exactly one
/// operand's free side of an ordinary contraction.  This is the physical form
/// needed to group independent rows/columns while keeping invariant matrix
/// operands shared by the group.  A region fold may carry the axis through
/// immutable captures and summaries, but its source traversal remains
/// independent. Ordered loops may carry the free axis and its owned stores
/// without changing scalar bounds. Paired/batched dependence, local allocation
/// replication and non-store effects remain excluded.
bool supportsStructuredFreeAxisValueGraph(
    ArrayRef<WorksetCoordinateOp> coordinates,
    llvm::SmallPtrSetImpl<Operation *> *ownedStores = nullptr,
    llvm::DenseMap<Operation *, bool> *contractSides = nullptr,
    bool *containsContraction = nullptr) {
  if (coordinates.size() > 1) {
    llvm::SmallPtrSet<Operation *, 4> stores;
    for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
      llvm::SmallPtrSet<Operation *, 4> currentStores;
      if (!supportsStructuredFreeAxisValueGraph({coordinate}, &currentStores))
        return false;
      if (index == 0)
        stores.insert(currentStores.begin(), currentStores.end());
      else if (stores.size() != currentStores.size() ||
               !llvm::all_of(stores, [&](Operation *store) {
                 return currentStores.contains(store);
               }))
        return false;
    }
  }
  llvm::SmallDenseSet<Value> dependent;
  SmallVector<Value> worklist;
  auto enqueue = [&](Value value) {
    if (dependent.insert(value).second)
      worklist.push_back(value);
  };
  for (WorksetCoordinateOp coordinate : coordinates)
    enqueue(coordinate.getResult());
  llvm::SmallPtrSet<Operation *, 32> visited;
  llvm::SmallPtrSet<Operation *, 8> loops;
  SmallVector<Operation *> operations;
  bool sawNestedContract = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      FailureOr<bool> propagated =
          propagateOrderedCarryDependency(value, user, enqueue, loops);
      if (failed(propagated))
        return false;
      if (*propagated)
        continue;
      if (!visited.insert(user).second)
        continue;
      operations.push_back(user);
      if (user->getNumResults() == 0) {
        auto store = dyn_cast<StoreOp>(user);
        if (!store || user->getNumRegions() != 0)
          return false;
        if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
            buffer && buffer.getScope().getValue() !=
                          BufferScope::InvocationWorkspace)
          return false;
        continue;
      }
      if (auto fold = dyn_cast<RegionFoldOp>(user)) {
        SmallVector<bool> resultDependencies;
        if (!analyzeStructuredRegionFold(
                fold,
                [&](Value operand) { return dependent.contains(operand); },
                resultDependencies, sawNestedContract))
          return false;
        for (auto [result, resultDepends] :
             llvm::zip(fold.getResults(), resultDependencies))
          if (resultDepends && dependent.insert(result).second)
            worklist.push_back(result);
        continue;
      }
      if (!isStructuredFreeAxisValueOp(user))
        return false;
      for (Value result : user->getResults())
        if (dependent.insert(result).second)
          worklist.push_back(result);
    }
  }

  auto depends = [&](Value value) { return dependent.contains(value); };
  bool sawContract = sawNestedContract;
  bool sawReduction = false;
  bool sawOwnedStore = false;
  for (Operation *operation : operations) {
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      // Another producer path may reach this fold after its first visit. Check
      // the completed dependence set before committing any ownership rewrite.
      SmallVector<bool> resultDependencies;
      if (!analyzeStructuredRegionFold(fold, depends, resultDependencies,
                                       sawContract))
        return false;
      continue;
    }
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      bool lhs = depends(contract.getLhs());
      bool rhs = depends(contract.getRhs());
      if (lhs == rhs || depends(contract.getAccumulator()))
        return false;
      if (contractSides)
        (*contractSides)[contract] = lhs;
      sawContract = true;
      continue;
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      ValueRange sources =
          reduce.getInputs().take_front(reduce.getSourceCount());
      ValueRange boundaries =
          reduce.getInputs().drop_front(reduce.getSourceCount());
      if (!llvm::any_of(sources, depends) || llvm::any_of(boundaries, depends))
        return false;
      sawReduction = true;
      continue;
    }
    if (auto gather = dyn_cast<GatherOp>(operation)) {
      auto source = dyn_cast<FragmentType>(gather.getSource().getType());
      if (!source || !depends(gather.getSource()) ||
          llvm::any_of(gather.getCoordinates(), depends))
        return false;
      for (int64_t sourceAxis : gather.getSourceAxes())
        if (sourceAxis < 0 ||
            sourceAxis >= static_cast<int64_t>(source.getShape().size()) ||
            !isStaticUnitExtent(source.getShape()[sourceAxis]))
          return false;
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      bool ownedCoordinate = llvm::any_of(store.getCoordinates(), depends);
      if (!ownedCoordinate || !depends(store.getValue()))
        return false;
      sawOwnedStore = true;
      if (ownedStores)
        ownedStores->insert(store.getOperation());
    }
  }
  if (containsContraction)
    *containsContraction = sawContract;
  return (sawContract || sawReduction) && sawOwnedStore &&
         hasSupportedOrderedBodies(loops, depends);
}

SmallVector<WorksetCoordinateOp> orthogonalContractCoordinates(
    ArrayRef<WorksetCoordinateOp> coordinates) {
  for (auto [index, coordinate] : llvm::enumerate(llvm::reverse(coordinates))) {
    WorksetCoordinateOp first = coordinate;
    llvm::SmallPtrSet<Operation *, 4> firstStores;
    llvm::DenseMap<Operation *, bool> firstSides;
    if (!supportsStructuredFreeAxisValueGraph({first}, &firstStores, &firstSides) ||
        firstSides.empty() ||
        !llvm::all_of(firstSides, [](const auto &entry) {
          auto contract = cast<ContractOp>(entry.first);
          return contract.getLhs().getType().getShape().size() == 1 &&
                 contract.getRhs().getType().getShape().size() == 1 &&
                 contract.getResult().getType().getShape().empty() &&
                 contract.getLhsReductionAxes().size() == 1 &&
                 contract.getRhsReductionAxes().size() == 1 &&
                 contract.getLhsBatchAxes().empty();
        }))
      continue;
    bool firstIsLhs = firstSides.begin()->second;
    if (!llvm::all_of(firstSides, [&](const auto &entry) {
          return entry.second == firstIsLhs;
        }))
      continue;
    for (WorksetCoordinateOp second :
         llvm::reverse(coordinates.drop_back(index + 1))) {
      if (first.getSourceId() == second.getSourceId() &&
          first.getSourceAxis() == second.getSourceAxis())
        continue;
      llvm::SmallPtrSet<Operation *, 4> secondStores;
      llvm::DenseMap<Operation *, bool> secondSides;
      if (!supportsStructuredFreeAxisValueGraph(
              {second}, &secondStores, &secondSides) ||
          firstStores.size() != secondStores.size() ||
          !llvm::all_of(firstStores, [&](Operation *store) {
            return secondStores.contains(store);
          }) ||
          firstSides.size() != secondSides.size() ||
          !llvm::all_of(firstSides, [&](const auto &entry) {
            auto side = secondSides.find(entry.first);
            return side != secondSides.end() && side->second != entry.second;
          }))
        continue;
      // Each lift prepends its free axis. Lift RHS first to retain the
      // contraction result order [lhs free axes, rhs free axes].
      return firstIsLhs ? SmallVector<WorksetCoordinateOp>{second, first}
                        : SmallVector<WorksetCoordinateOp>{first, second};
    }
  }
  return {};
}

LogicalResult rankLiftPointwiseValueGraph(
    func::FuncOp kernel, ArrayRef<MakeRangeOp> liftedRanges,
    llvm::SmallPtrSetImpl<Operation *> &laneReductions,
    llvm::DenseMap<Value, SmallVector<BroadcastOp>> &laneBounds) {
  if (liftedRanges.empty())
    return success();

  SmallVector<std::pair<PhysicalExprAttr, AxisMapAttr>> liftedAxes;
  for (MakeRangeOp range : llvm::reverse(liftedRanges)) {
    auto fragment = cast<FragmentType>(range.getResult().getType());
    liftedAxes.emplace_back(
        cast<PhysicalExprAttr>(fragment.getShape()[0]),
        cast<AxisMapAttr>(fragment.getAxisMaps()[0]));
  }
  auto liftedType = [&](FragmentType original) {
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    auto appendAxis = [&](PhysicalExprAttr extent, AxisMapAttr mapping) {
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    };
    for (auto [extent, mapping] : liftedAxes) {
      bool present = llvm::any_of(original.getAxisMaps(), [&](Attribute attribute) {
        auto axis = cast<AxisMapAttr>(attribute);
        return sourceAxisIdentity(axis) == sourceAxisIdentity(mapping);
      });
      if (!present)
        appendAxis(extent, mapping);
    }
    for (auto [extent, mapping] :
         llvm::zip(original.getShape(), original.getAxisMaps()))
      appendAxis(cast<PhysicalExprAttr>(extent), cast<AxisMapAttr>(mapping));
    return FragmentType::get(kernel.getContext(), original.getElementType(),
                             ArrayAttr::get(kernel.getContext(), shape),
                             ArrayAttr::get(kernel.getContext(), mappings),
                             original.getValidity(), original.getOwner());
  };

  auto scalarLiftedType = [&](Type element) {
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    for (auto [extent, mapping] : liftedAxes) {
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    }
    return FragmentType::get(
        kernel.getContext(), element, ArrayAttr::get(kernel.getContext(), shape),
        ArrayAttr::get(kernel.getContext(), mappings), /*validity=*/1,
        /*owner=*/1);
  };
  std::function<Type(Type)> liftedValueType = [&](Type original) -> Type {
    if (auto fragment = dyn_cast<FragmentType>(original))
      return liftedType(fragment);
    if (auto record = dyn_cast<RecordType>(original)) {
      SmallVector<Attribute> fields;
      fields.reserve(record.getFieldTypes().size());
      for (Attribute field : record.getFieldTypes())
        fields.push_back(TypeAttr::get(
            liftedValueType(cast<TypeAttr>(field).getValue())));
      return RecordType::get(kernel.getContext(), record.getFieldNames(),
                             ArrayAttr::get(kernel.getContext(), fields),
                             record.getOwner());
    }
    if (isa<IntegerType, FloatType, IndexType>(original))
      return scalarLiftedType(original);
    return original;
  };
  std::function<bool(Type)> carriesLiftedAxis = [&](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type))
      return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
        auto axis = cast<AxisMapAttr>(attribute);
        return llvm::any_of(liftedAxes, [&](const auto &lifted) {
          return sourceAxisIdentity(axis) == sourceAxisIdentity(lifted.second);
        });
      });
    if (auto record = dyn_cast<RecordType>(type))
      return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
        return carriesLiftedAxis(cast<TypeAttr>(field).getValue());
      });
    return false;
  };

  llvm::SmallDenseSet<Value> liftedValues;
  SmallVector<SplatOp> rankLiftedSplats;
  SmallVector<GatherOp> rankLiftedUnitGathers;
  auto rememberRankLiftedSplat = [&](SplatOp splat) {
    if (!llvm::is_contained(rankLiftedSplats, splat))
      rankLiftedSplats.push_back(splat);
  };
  for (MakeRangeOp range : liftedRanges)
    liftedValues.insert(range.getResult());
  auto dependsOnLiftedAxis = [&](Value value) {
    return liftedValues.contains(value) || carriesLiftedAxis(value.getType());
  };
  auto shiftedAxes = [&](ArrayRef<int64_t> axes) {
    SmallVector<int64_t> shifted;
    shifted.reserve(axes.size());
    for (int64_t axis : axes)
      shifted.push_back(axis + static_cast<int64_t>(liftedAxes.size()));
    return shifted;
  };
  llvm::SmallPtrSet<Operation *, 32> liftedOperations;
  auto liftLoopCarry = [&](scf::ForOp loop, unsigned index,
                           Type target) -> LogicalResult {
    OpBuilder initBuilder(loop);
    FailureOr<Value> init = projectPhysicalValueToSchema(
        initBuilder, loop.getLoc(), loop.getInitArgs()[index], target);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    OpBuilder yieldBuilder(yield);
    FailureOr<Value> yielded = projectPhysicalValueToSchema(
        yieldBuilder, loop.getLoc(), yield.getResults()[index], target);
    if (failed(init) || failed(yielded))
      return loop.emitOpError("ordered carry cannot adopt the lifted output axis");
    loop.getInitArgsMutable()[index].assign(*init);
    yield->setOperand(index, *yielded);
    loop.getRegionIterArgs()[index].setType(target);
    loop.getResult(index).setType(target);
    liftedValues.insert(loop.getRegionIterArgs()[index]);
    liftedValues.insert(loop.getResult(index));
    return success();
  };
  std::function<WalkResult(Operation *)> liftOperation;
  liftOperation = [&](Operation *operation) {
    if (isa<MakeRangeOp>(operation) || liftedOperations.contains(operation) ||
        operation->getParentOfType<scf::WhileOp>())
      return WalkResult::advance();
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      bool dependent = false;
      loop->walk([&](Operation *nested) {
        dependent |= llvm::any_of(nested->getOperands(), dependsOnLiftedAxis);
      });
      if (!dependent)
        return WalkResult::advance();
      if (!canPredicateScalarWhile(loop)) {
        loop.emitOpError("cannot predicate the lifted ordered recurrence");
        return WalkResult::interrupt();
      }
      OpBuilder builder(loop);
      Location location = loop.getLoc();
      auto shape = scalarLiftedType(builder.getI1Type());
      Value active;
      for (MakeRangeOp range : liftedRanges) {
        auto indexType = scalarLiftedType(builder.getIndexType());
        auto project = [&](Value value, Type type) -> Value {
          return builder.create<BroadcastOp>(location, type, value);
        };
        Value coordinate = project(range.getResult(), indexType);
        Value lower = builder.create<CompareOp>(
            location, shape, coordinate, project(range.getLogicalStart(), indexType),
            ComparePredicate::Ge);
        auto stop = builder.create<BroadcastOp>(location, indexType, range.getLogicalStop());
        laneBounds[range.getResult()].push_back(stop);
        Value upper = builder.create<CompareOp>(
            location, shape, coordinate, stop, ComparePredicate::Lt);
        Value bounded = builder.create<BinaryOp>(
            location, shape, lower, upper, BinaryOperator::LogicalAnd);
        active = active ? Value(builder.create<BinaryOp>(
                              location, shape, active, bounded, BinaryOperator::LogicalAnd))
                        : bounded;
      }
      IRMapping mapping;
      clonePredicatedScalarOperation(builder, loop, mapping, active, shape);
      Operation *replacement = mapping.lookup(loop.getResult(0)).getDefiningOp();
      // The cloned any reduction consumes the new lanes; they are not free
      // reduction axes to be lifted again in the next fixed-point iteration.
      replacement->walk([&](Operation *nested) {
        liftedOperations.insert(nested);
        if (isa<ReduceOp>(nested))
          laneReductions.insert(nested);
        for (Value result : nested->getResults())
          if (carriesLiftedAxis(result.getType()))
            liftedValues.insert(result);
      });
      for (Value result : loop.getResults())
        result.replaceAllUsesWith(mapping.lookup(result));
      loop.erase();
      return WalkResult::advance();
    }
    bool dependsOnLiftedRange =
        llvm::any_of(operation->getOperands(), [&](Value operand) {
          return dependsOnLiftedAxis(operand);
        });
    if (!dependsOnLiftedRange)
      return WalkResult::advance();
    if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
      auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!loop)
        return WalkResult::interrupt();
      for (auto [index, value] : llvm::enumerate(yield.getResults()))
        if (dependsOnLiftedAxis(value) &&
            failed(liftLoopCarry(loop, index, value.getType())))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      if (dependsOnLiftedAxis(loop.getLowerBound()) ||
          dependsOnLiftedAxis(loop.getUpperBound()) ||
          dependsOnLiftedAxis(loop.getStep()))
        return WalkResult::interrupt();
      for (auto [index, value] : llvm::enumerate(loop.getInitArgs()))
        if (dependsOnLiftedAxis(value) &&
            failed(liftLoopCarry(loop, index, value.getType())))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (!liftedOperations.insert(operation).second)
      return WalkResult::advance();
    if (operation->getNumResults() == 0)
      return WalkResult::advance();

    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      if (!llvm::hasSingleElement(fold.getSummarize()) ||
          !llvm::hasSingleElement(fold.getCombine()))
        return WalkResult::interrupt();
      unsigned sourceCount = fold.getSourceCount();
      unsigned identityCount = fold.getIdentityCount();
      unsigned captureCount = fold.getCaptureCount();
      ValueRange inputs = fold.getInputs();
      if (inputs.size() != sourceCount + identityCount + captureCount ||
          llvm::any_of(inputs.take_front(sourceCount + identityCount),
                       dependsOnLiftedAxis)) {
        fold.emitOpError("rank lifting requires independent sources and identities");
        return WalkResult::interrupt();
      }

      Block &summarize = fold.getSummarize().front();
      bool dependentCapture = false;
      for (unsigned capture = 0; capture < captureCount; ++capture) {
        unsigned operand = sourceCount + identityCount + capture;
        if (!dependsOnLiftedAxis(inputs[operand]))
          continue;
        dependentCapture = true;
        BlockArgument argument = summarize.getArgument(sourceCount + capture);
        argument.setType(inputs[operand].getType());
        liftedValues.insert(argument);
      }
      if (!dependentCapture)
        return WalkResult::interrupt();
      for (Operation &nested : summarize.without_terminator())
        if (liftOperation(&nested).wasInterrupted())
          return WalkResult::interrupt();
      auto summarizeYield = dyn_cast<YieldOp>(summarize.getTerminator());
      if (!summarizeYield ||
          summarizeYield.getValues().size() != identityCount)
        return WalkResult::interrupt();

      Block &combine = fold.getCombine().front();
      if (combine.getNumArguments() != 2 * identityCount)
        return WalkResult::interrupt();
      SmallVector<Type> resultTypes;
      SmallVector<bool> dependentResults;
      OpBuilder builder(fold);
      for (unsigned index = 0; index < identityCount; ++index) {
        Value summary = summarizeYield.getValues()[index];
        Type target = summary.getType();
        bool dependent = dependsOnLiftedAxis(summary);
        resultTypes.push_back(target);
        dependentResults.push_back(dependent);
        if (!dependent)
          continue;
        FailureOr<Value> identity = projectPhysicalValueToSchema(
            builder, fold.getLoc(), inputs[sourceCount + index], target);
        if (failed(identity))
          return WalkResult::interrupt();
        fold->setOperand(sourceCount + index, *identity);
        fold.getResult(index).setType(target);
        combine.getArgument(index).setType(target);
        combine.getArgument(identityCount + index).setType(target);
        liftedValues.insert(combine.getArgument(index));
        liftedValues.insert(combine.getArgument(identityCount + index));
        liftedValues.insert(fold.getResult(index));
      }
      if (!llvm::any_of(dependentResults, [](bool value) { return value; }))
        return WalkResult::interrupt();
      for (Operation &nested : combine.without_terminator())
        if (liftOperation(&nested).wasInterrupted())
          return WalkResult::interrupt();
      auto combineYield = dyn_cast<YieldOp>(combine.getTerminator());
      if (!combineYield || combineYield.getValues().size() != identityCount)
        return WalkResult::interrupt();
      for (auto [index, value] : llvm::enumerate(combineYield.getValues()))
        if (value.getType() != resultTypes[index])
          return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (auto contract = dyn_cast<ContractOp>(operation)) {
      bool lhs = dependsOnLiftedAxis(contract.getLhs());
      bool rhs = dependsOnLiftedAxis(contract.getRhs());
      if (lhs == rhs || dependsOnLiftedAxis(contract.getAccumulator())) {
        contract.emitOpError("cannot rank-lift contraction operand relation")
            << "; lhs=" << lhs << "; rhs=" << rhs
            << "; accumulator=" << dependsOnLiftedAxis(contract.getAccumulator());
        return WalkResult::interrupt();
      }
      OpBuilder builder(contract);
      if (lhs) {
        contract->setAttr(
            "lhs_reduction_axes",
            builder.getDenseI64ArrayAttr(
                shiftedAxes(contract.getLhsReductionAxes())));
        contract->setAttr(
            "lhs_batch_axes",
            builder.getDenseI64ArrayAttr(
                shiftedAxes(contract.getLhsBatchAxes())));
      } else {
        contract->setAttr(
            "rhs_reduction_axes",
            builder.getDenseI64ArrayAttr(
                shiftedAxes(contract.getRhsReductionAxes())));
        contract->setAttr(
            "rhs_batch_axes",
            builder.getDenseI64ArrayAttr(
                shiftedAxes(contract.getRhsBatchAxes())));
      }
      Type target = liftedValueType(contract.getResult().getType());
      FailureOr<Value> accumulator = projectPhysicalValueToSchema(
          builder, contract.getLoc(), contract.getAccumulator(), target);
      if (failed(accumulator))
        return WalkResult::interrupt();
      contract->setOperand(2, *accumulator);
      contract.getResult().setType(cast<FragmentType>(target));
      liftedValues.insert(contract.getResult());
      return WalkResult::advance();
    }
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      bool sourceDepends = llvm::any_of(
          reduce.getInputs().take_front(reduce.getSourceCount()),
          dependsOnLiftedAxis);
      if (!sourceDepends)
        return WalkResult::interrupt();
      OpBuilder builder(reduce);
      for (unsigned index = 0; index < reduce.getSourceCount(); ++index) {
        Value source = reduce.getInputs()[index];
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, reduce.getLoc(), source, liftedValueType(source.getType()));
        if (failed(projected)) {
          reduce.emitOpError("source cannot adopt the lifted free axes");
          return WalkResult::interrupt();
        }
        reduce->setOperand(index, *projected);
      }
      reduce->setAttr("axes", DenseI64ArrayAttr::get(
                                  kernel.getContext(),
                                  shiftedAxes(reduce.getAxes())));
      if (!llvm::hasSingleElement(reduce.getCombine()) ||
          reduce.getCombine().front().getNumArguments() <
              2 * reduce.getIdentityCount())
        return WalkResult::interrupt();
      Block &combine = reduce.getCombine().front();
      for (auto [index, value] : llvm::enumerate(reduce.getResults())) {
        value.setType(liftedValueType(value.getType()));
        combine.getArgument(index).setType(value.getType());
        combine.getArgument(reduce.getIdentityCount() + index)
            .setType(value.getType());
        liftedValues.insert(combine.getArgument(index));
        liftedValues.insert(
            combine.getArgument(reduce.getIdentityCount() + index));
        liftedValues.insert(value);
      }
      for (Operation &nested : combine.without_terminator())
        if (liftOperation(&nested).wasInterrupted())
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    if (auto record = dyn_cast<MakeRecordOp>(operation)) {
      auto target = cast<RecordType>(liftedValueType(record.getResult().getType()));
      OpBuilder builder(record);
      for (auto [index, field] : llvm::enumerate(record.getFields())) {
        Type fieldType =
            cast<TypeAttr>(target.getFieldTypes()[index]).getValue();
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, record.getLoc(), field, fieldType);
        if (failed(projected))
          return WalkResult::interrupt();
        record->setOperand(index, *projected);
      }
      record.getResult().setType(target);
      liftedValues.insert(record.getResult());
      return WalkResult::advance();
    }
    if (auto gather = dyn_cast<GatherOp>(operation)) {
      if (dependsOnLiftedAxis(gather.getSource())) {
        gather->setAttr(
            "source_axes",
            DenseI64ArrayAttr::get(
                kernel.getContext(), shiftedAxes(gather.getSourceAxes())));
        rankLiftedUnitGathers.push_back(gather);
      }
    } else if (auto transpose = dyn_cast<TransposeOp>(operation)) {
      SmallVector<int64_t> permutation;
      for (unsigned axis = 0; axis < liftedAxes.size(); ++axis)
        permutation.push_back(axis);
      for (int64_t axis : transpose.getPermutation())
        permutation.push_back(axis + liftedAxes.size());
      transpose->setAttr(
          "permutation",
          DenseI64ArrayAttr::get(kernel.getContext(), permutation));
    }
    if (!isStructuredFreeAxisValueOp(operation)) {
      operation->emitOpError("has no rank-lifting rule for a structured free axis");
      return WalkResult::interrupt();
    }
    if (auto splat = dyn_cast<SplatOp>(operation);
        splat && isa<FragmentType>(splat.getValue().getType()))
      rememberRankLiftedSplat(splat);
    for (Value value : operation->getResults()) {
      Type lifted = liftedValueType(value.getType());
      if (lifted == value.getType() &&
          !isa<FragmentType, RecordType>(lifted))
        return WalkResult::interrupt();
      value.setType(lifted);
      liftedValues.insert(value);
    }
    return WalkResult::advance();
  };
  // A carry can make earlier body operations lane-dependent. Reach a fixed
  // point through SSA uses without changing ordered loop control or traversal.
  size_t previousLiftedCount;
  do {
    previousLiftedCount = liftedValues.size() + liftedOperations.size();
    WalkResult result = kernel.walk(
        [&](Operation *operation) { return liftOperation(operation); });
    if (result.wasInterrupted())
      return failure();
  } while (previousLiftedCount != liftedValues.size() + liftedOperations.size());

  // Rank lifting can turn the scalar producer of an existing splat into a
  // fragment.  Preserve that newly explicit value relation as a fragment
  // broadcast; SplatOp remains the scalar-to-fragment boundary.
  for (SplatOp splat : rankLiftedSplats) {
    auto source = dyn_cast<FragmentType>(splat.getValue().getType());
    auto target = dyn_cast<FragmentType>(splat.getResult().getType());
    if (!source || !target ||
        source.getElementType() != target.getElementType() ||
        source.getOwner() != target.getOwner()) {
      splat.emitOpError(
          "rank-lifted splat has incompatible fragment value relation");
      return failure();
    }

    Value replacement = splat.getValue();
    if (source != target) {
      OpBuilder builder(splat);
      Value broadcastSource = splat.getValue();
      auto broadcastSourceType = source;

      // Lifted workset axes are a physical execution prefix.  If the old splat
      // had additional value axes, make their scalar expansion explicit as
      // reshape-inserted unit axes before applying ordinary trailing broadcast
      // semantics.  This keeps ownership axes out of the logical broadcast
      // suffix and lets later extent retargeting distinguish the two relations.
      bool sourceIsTargetPrefix =
          source.getShape().size() < target.getShape().size();
      for (unsigned axis = 0;
           sourceIsTargetPrefix && axis < source.getShape().size(); ++axis)
        sourceIsTargetPrefix =
            source.getShape()[axis] == target.getShape()[axis] &&
            source.getAxisMaps()[axis] == target.getAxisMaps()[axis];
      if (sourceIsTargetPrefix) {
        SmallVector<Attribute> shape(source.getShape().begin(),
                                     source.getShape().end());
        SmallVector<Attribute> mappings(source.getAxisMaps().begin(),
                                        source.getAxisMaps().end());
        PhysicalExprAttr unit = expression(
            kernel.getContext(), PhysicalExprKind::Constant, 1);
        for (unsigned axis = source.getShape().size();
             axis < target.getShape().size(); ++axis) {
          shape.push_back(unit);
          mappings.push_back(target.getAxisMaps()[axis]);
        }
        broadcastSourceType = FragmentType::get(
            kernel.getContext(), source.getElementType(),
            ArrayAttr::get(kernel.getContext(), shape),
            ArrayAttr::get(kernel.getContext(), mappings),
            source.getValidity(), source.getOwner());
        FailureOr<ArrayAttr> reassociation = inferReshapeReassociation(
            source, broadcastSourceType,
            /*sourcePrefix=*/source.getShape().size(),
            /*resultPrefix=*/source.getShape().size());
        if (failed(reassociation)) {
          splat.emitOpError(
              "rank-lifted splat cannot insert its broadcast suffix");
          return failure();
        }
        broadcastSource = builder.create<ReshapeOp>(
            splat.getLoc(), broadcastSourceType, splat.getValue(),
            *reassociation);
      }
      if (!queryBroadcastProjection(broadcastSourceType, target).isExact()) {
        splat.emitOpError(
            "rank-lifted splat has no exact fragment broadcast relation");
        return failure();
      }
      auto broadcast = builder.create<BroadcastOp>(
          splat.getLoc(), target, broadcastSource);
      if (Attribute origin = splat->getAttr(originAttr))
        broadcast->setAttr(originAttr, origin);
      replacement = broadcast.getResult();
    }
    splat.getResult().replaceAllUsesWith(replacement);
    splat.erase();
  }

  auto scalarIntegerConstant = [](Value value,
                                  int64_t expected) -> bool {
    while (true) {
      if (auto splat = value.getDefiningOp<SplatOp>()) {
        value = splat.getValue();
        continue;
      }
      if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
        value = broadcast.getValue();
        continue;
      }
      if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
        value = reshape.getValue();
        continue;
      }
      if (auto cast = value.getDefiningOp<CastOp>()) {
        value = cast.getValue();
        continue;
      }
      break;
    }
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    if (auto boolean =
            constant ? dyn_cast<BoolAttr>(constant.getValue()) : BoolAttr())
      return static_cast<int64_t>(boolean.getValue()) == expected;
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
    return integer && integer.getInt() == expected;
  };
  // Gathering every logical unit suffix from a rank-lifted value is a typed
  // squeeze of those suffix axes.  Keep that fact in shared IR as reshape so
  // providers do not need a special partial-tile extraction convention.
  for (GatherOp gather : rankLiftedUnitGathers) {
    auto source = dyn_cast<FragmentType>(gather.getSource().getType());
    auto target = dyn_cast<FragmentType>(gather.getResult().getType());
    if (!source || !target ||
        (gather.getValid() &&
         !scalarIntegerConstant(gather.getValid(), 1))) {
      InFlightDiagnostic diagnostic = gather.emitOpError(
          "rank-lifted unit gather has no unconditional squeeze relation");
      diagnostic << "; source=" << gather.getSource().getType()
                 << "; result=" << gather.getResult().getType();
      if (gather.getValid()) {
        diagnostic << "; valid=" << gather.getValid().getType();
        if (Operation *producer = gather.getValid().getDefiningOp())
          diagnostic << "; valid_producer=" << producer->getName();
      }
      return failure();
    }
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(source.getShape().size()) ||
          !isStaticUnitExtent(source.getShape()[sourceAxis]) ||
          !scalarIntegerConstant(coordinate, 0))
        return gather.emitOpError(
            "rank-lifted unit gather selects a non-unit source axis");
    FailureOr<ArrayAttr> reassociation =
        inferReshapeReassociation(source, target);
    if (failed(reassociation))
      return gather.emitOpError(
          "rank-lifted unit gather has no exact reshape projection");
    OpBuilder builder(gather);
    auto replacement = builder.create<ReshapeOp>(
        gather.getLoc(), target, gather.getSource(), *reassociation);
    if (Attribute origin = gather->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
  }
  // Ownership queries consume operand relations, not just the lifted result
  // types. Make scalar and coordinate broadcasts explicit before those queries.
  return alignPointwiseValueRelations(kernel);
}

enum ContractFreeAxisSide : unsigned {
  ContractFreeAxisNone = 0,
  ContractFreeAxisLhs = 1,
  ContractFreeAxisRhs = 2,
};

struct ContractFreeAxisFacts {
  unsigned sides = ContractFreeAxisNone;
  unsigned matrixSides = ContractFreeAxisNone;
  bool regionContraction = false;
  bool batchedContraction = false;
  unsigned operandElementBitWidth = 0;
};

ContractFreeAxisFacts contractFreeAxisFacts(func::FuncOp kernel,
                                            MakeRangeOp range) {
  ContractFreeAxisFacts facts;
  PhysicalProgramAnalysis analysis(kernel);
  kernel.walk([&](ContractOp contract) {
    PhysicalContractFreeAxisFact freeAxes =
        analysis.contractFreeAxes(contract);
    if (!freeAxes.isExact())
      return;
    bool matrixFreeAxes =
        llvm::any_of(freeAxes.axes, [&](const auto &axis) {
          return axis.operand == contract.getLhs();
        }) &&
        llvm::any_of(freeAxes.axes, [&](const auto &axis) {
          return axis.operand == contract.getRhs();
        });
    for (const PhysicalContractFreeAxis &axis : freeAxes.axes) {
      if (!llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range);
          }))
        continue;
      unsigned side = ContractFreeAxisNone;
      if (axis.operand == contract.getLhs())
        side |= ContractFreeAxisLhs;
      if (axis.operand == contract.getRhs())
        side |= ContractFreeAxisRhs;
      facts.sides |= side;
      if (matrixFreeAxes)
        facts.matrixSides |= side;
      facts.operandElementBitWidth = std::max(
          facts.operandElementBitWidth,
          std::max(physicalElementBitWidth(contract.getLhs().getType()),
                   physicalElementBitWidth(contract.getRhs().getType())));
      auto fold = contract->getParentOfType<RegionFoldOp>();
      if (!fold && !contract.getLhsBatchAxes().empty() &&
          llvm::count_if(freeAxes.axes, [&](const auto &free) {
            return free.operand == contract.getLhs();
          }) == 1 &&
          llvm::count_if(freeAxes.axes, [&](const auto &free) {
            return free.operand == contract.getRhs();
          }) == 1)
        facts.batchedContraction = true;
      if (fold &&
          fold.getSegment().getCategory() ==
              static_cast<uint32_t>(ParameterCategory::RegionContraction))
        facts.regionContraction = true;
    }
  });
  return facts;
}

unsigned contractFreeAxisSides(func::FuncOp kernel, MakeRangeOp range) {
  return contractFreeAxisFacts(kernel, range).sides;
}

} // namespace

LogicalResult alignContractValueRelations(func::FuncOp kernel) {
  if (failed(alignOrdinaryContractOperandTypes(kernel)) ||
      failed(alignContractAccumulatorTypes(kernel)))
    return failure();
  return success();
}

static LogicalResult realizePointwiseBlockingImpl(ModuleOp module,
                                                   bool ownershipOnly) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  llvm::SmallPtrSet<Operation *, 4> laneReductions;
  llvm::DenseMap<Value, SmallVector<BroadcastOp>> laneBounds;
  if (ownershipOnly && failed(separateReductionOutputOccurrences(kernel)))
    return failure();
  auto finalizeValueRelations = [&]() -> LogicalResult {
    eraseDeadPhysicalValues(kernel);
    if (failed(bindStructurallyRequiredStaticFragments(kernel)))
      return failure();
    if (failed(alignStructuredCaptureRelations(kernel)) ||
        failed(alignReductionResultRelations(kernel)) ||
        failed(alignReductionIdentityRelations(kernel)) ||
        failed(alignAggregateValueRelations(kernel)) ||
        failed(alignPointwiseValueRelations(kernel)) ||
        failed(alignReductionYieldRelations(kernel)) ||
        failed(alignAccessValueRelations(kernel)) ||
        failed(alignAggregateValueRelations(kernel)) ||
        failed(alignContractValueRelations(kernel)) ||
        failed(alignAggregateValueRelations(kernel)) ||
        failed(alignAccessValueRelations(kernel)) ||
        failed(alignPointwiseValueRelations(kernel)))
      return failure();
    PhysicalProgramAnalysis analysis(kernel);
    WalkResult localized = kernel.walk([&](GatherOp gather) {
      auto source = dyn_cast<FragmentType>(gather.getSource().getType());
      if (!source)
        return WalkResult::advance();
      for (auto [position, axis] : llvm::enumerate(gather.getSourceAxes())) {
        Value coordinate = gather.getCoordinates()[position];
        Value origin = coordinate;
        while (auto broadcast = origin.getDefiningOp<BroadcastOp>())
          origin = broadcast.getValue();
        auto range = origin.getDefiningOp<MakeRangeOp>();
        auto begin = range ? queryLaunchExpression(range.getLogicalStart())
                           : PhysicalExprAttr();
        if (!range || !isUnitStepRange(range) ||
            !begin || begin.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            begin.getValue() != 0 ||
            samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) ||
            range.getResult().getType().getShape()[0] != source.getShape()[axis])
          continue;
        auto roots = analysis.axisRanges(gather.getSource(), axis);
        if (!roots.isExact() || roots.roots.empty() || !roots.blockers.empty() ||
            !llvm::all_of(roots.roots, [&](MakeRangeOp root) {
              return sameLogicalRange(root, range) &&
                     samePhysicalScalarExpression(root.getStart(), range.getStart()) &&
                     samePhysicalScalarExpression(root.getExtent(), range.getExtent());
            }))
          continue;
        OpBuilder builder(gather);
        auto type = cast<FragmentType>(coordinate.getType());
        auto start = materializeBroadcastToFragment(builder, gather.getLoc(),
                                                    range.getStart(), type);
        if (failed(start))
          return WalkResult::interrupt();
        Value ordinal = builder.create<BinaryOp>(
            gather.getLoc(), type, coordinate, *start, BinaryOperator::Subtract);
        gather.getCoordinatesMutable().slice(position, 1).assign(ordinal);
      }
      return WalkResult::advance();
    });
    if (localized.wasInterrupted())
      return failure();
    eraseDeadPhysicalValues(kernel);
    return success();
  };

  if (ownershipOnly) {
    SmallVector<WorksetCoordinateOp> pointwiseCoordinates;
    kernel.walk([&](WorksetCoordinateOp coordinate) {
      pointwiseCoordinates.push_back(coordinate);
    });
    SmallVector<MakeRangeOp> existingRanges;
    kernel.walk([&](MakeRangeOp range) { existingRanges.push_back(range); });
    SmallVector<WorksetCoordinateOp> lifted;
    bool liftSeparately = false;
    bool preserveContractionOrder = false;
    if (existingRanges.empty()) {
      if (supportsCartesianPointwiseValueGraph(pointwiseCoordinates, true))
        lifted.append(pointwiseCoordinates.begin(), pointwiseCoordinates.end());
      else {
        // A coordinate used by ordered control can stay scalar without
        // excluding other independent coordinates from the ownership tile.
        for (WorksetCoordinateOp coordinate : llvm::reverse(pointwiseCoordinates)) {
          SmallVector<WorksetCoordinateOp> candidates(lifted);
          candidates.insert(candidates.begin(), coordinate);
          if (supportsCartesianPointwiseValueGraph(candidates, true))
            lifted = std::move(candidates);
        }
      }
    } else {
      SmallVector<std::pair<int64_t, WorksetCoordinateOp>> uncovered;
      for (WorksetCoordinateOp coordinate : pointwiseCoordinates) {
        PhysicalSourceAxis source{coordinate.getSourceId(),
                                  coordinate.getSourceAxis(), false};
        bool covered = llvm::any_of(existingRanges, [&](MakeRangeOp range) {
          return sourceAxisIdentity(range) == source;
        });
        auto worksetAxis =
            coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr);
        if (!covered && worksetAxis && worksetAxis.getInt() >= 0)
          uncovered.emplace_back(worksetAxis.getInt(), coordinate);
      }
      llvm::stable_sort(uncovered, [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });
      SmallVector<WorksetCoordinateOp> uncoveredCoordinates;
      for (auto [_, coordinate] : uncovered)
        uncoveredCoordinates.push_back(coordinate);
      lifted = orthogonalContractCoordinates(uncoveredCoordinates);
      liftSeparately = !lifted.empty();
      // Independent reductions may expose reuse on an outer output axis. Keep
      // all proven free axes available for the later ownership choice instead
      // of fixing the choice to the two innermost source loops.
      if (lifted.empty() && uncoveredCoordinates.size() >= 2) {
        llvm::SmallDenseSet<PhysicalSourceAxis> sources;
        bool distinctSources =
            llvm::all_of(uncoveredCoordinates, [&](auto coordinate) {
              return sources.insert({coordinate.getSourceId(),
                                     coordinate.getSourceAxis(), false}).second;
            });
        bool containsContraction = false;
        if (distinctSources &&
            supportsStructuredFreeAxisValueGraph(uncoveredCoordinates, nullptr,
                                                nullptr, &containsContraction) &&
            !containsContraction) {
          lifted = uncoveredCoordinates;
          preserveContractionOrder = true;
        }
      }
      // A blocked pointwise program can keep one existing local vector range
      // while tiling the two innermost independent workset axes (for example
      // sequence/head around a feature vector).  Lift only when both axes
      // exist: a lone workset coordinate should remain the scalar program owner
      // of the existing local range.
      if (lifted.empty() && uncovered.size() >= 2) {
        SmallVector<WorksetCoordinateOp> candidates{
            uncovered[uncovered.size() - 2].second,
            uncovered.back().second};
        llvm::DenseMap<Operation *, bool> contractSides;
        if (supportsCartesianPointwiseValueGraph(candidates)) {
          lifted = std::move(candidates);
        } else if (supportsStructuredFreeAxisValueGraph(
                       candidates, nullptr, &contractSides)) {
          lifted = std::move(candidates);
          preserveContractionOrder = !contractSides.empty();
        }
      }
      if (lifted.empty())
        for (auto [_, coordinate] : llvm::reverse(uncovered))
          if (supportsStructuredFreeAxisValueGraph({coordinate}) ||
              supportsCartesianPointwiseValueGraph({coordinate}, true)) {
            lifted.push_back(coordinate);
            break;
          }
    }

    SmallVector<MakeRangeOp> liftedRanges;
    for (WorksetCoordinateOp coordinate : lifted) {
      OpBuilder builder(coordinate);
      builder.setInsertionPointAfter(coordinate);
      Value extent = builder.create<arith::ConstantIndexOp>(coordinate.getLoc(), 1);
      Value step = coordinate.getStep();
      PhysicalExprAttr unit = expression(module.getContext(),
                                         PhysicalExprKind::Constant, 1);
      int64_t dimension = coordinate.getDimensionId();
      if (dimension <= 0)
        return coordinate.emitOpError(
            "workset coordinate has no logical dimension identity");
      auto type = FragmentType::get(
          module.getContext(), builder.getIndexType(), builder.getArrayAttr({unit}),
          builder.getArrayAttr({AxisMapAttr::get(
              module.getContext(), coordinate.getSourceId(),
              coordinate.getSourceAxis(), dimension,
              /*fragmentAxis=*/0, /*derived=*/false)}),
          /*validity=*/1, /*owner=*/1);
      Value logicalStop = builder.create<BinaryOp>(
          coordinate.getLoc(), builder.getIndexType(), coordinate.getResult(),
          step, BinaryOperator::Add);
      auto range = builder.create<MakeRangeOp>(
          coordinate.getLoc(), type, coordinate.getResult(), extent, step,
          coordinate.getResult(), logicalStop,
          coordinate.getSourceId(), coordinate.getSourceAxis(),
          /*derived=*/false);
      range->setAttr(worksetCoordinateRangeAttr, builder.getUnitAttr());
      Operation *logicalStopProducer = logicalStop.getDefiningOp();
      coordinate.getResult().replaceUsesWithIf(
          range.getResult(), [&](OpOperand &use) {
            return use.getOwner() != range.getOperation() &&
                   use.getOwner() != logicalStopProducer;
          });
      liftedRanges.push_back(range);
      if (liftSeparately) {
        if (failed(rankLiftPointwiseValueGraph(kernel, {range}, laneReductions,
                                               laneBounds)))
          return kernel.emitError(
              "failed to rank-lift an independent contraction axis");
        liftedRanges.clear();
      }
    }
    // Native matrix forms use the last free operand axis for M/N. Preserve
    // logical nesting on a proven free side so the innermost workset axis,
    // rather than an outer independent axis, occupies that matrix dimension.
    if (preserveContractionOrder)
      std::reverse(liftedRanges.begin(), liftedRanges.end());
    if (failed(rankLiftPointwiseValueGraph(kernel, liftedRanges, laneReductions,
                                           laneBounds)))
      return kernel.emitError(
          "failed to rank-lift a legal pointwise ownership graph");
    // Newly explicit free axes must reach the existing contraction recognizer
    // before ownership freezes any axis to scalar grid execution.
    if (!lifted.empty() && failed(fuseMultiplyReductions(module)))
      return failure();
  }

  SmallVector<MakeRangeOp> dynamicRanges;
  SmallVector<MakeRangeOp> allRanges;
  llvm::DenseMap<Value, Value> fixedRangePredicates;
  kernel.walk([&](MakeRangeOp range) {
    allRanges.push_back(range);
    // Helper-local ranges already describe one physical fold/scan slice.  The
    // enclosing structured operation owns their segment relation; pointwise
    // blocking may only revisit them after the helper has been materialized
    // into the executable loop.
    if (range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>())
      return;
    if (!hasCompileTimeExtent(range.getExtent())) {
      dynamicRanges.push_back(range);
      return;
    }
    if (PhysicalProgramAnalysis(kernel)
            .axisRealization(range.getResult(), 0)
            .constructionScalarSeed) {
      dynamicRanges.push_back(range);
      return;
    }
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    auto fixedExtent = constantPhysicalExpression(extent);
    if (!ownershipOnly && fixedExtent && *fixedExtent > 0 &&
        !llvm::isPowerOf2_64(*fixedExtent) && isUnitStepRange(range)) {
      dynamicRanges.push_back(range);
      return;
    }
    if (!ownershipOnly ||
        !range.getExtent().getDefiningOp<arith::ConstantIndexOp>())
      return;
    if (range->hasAttr(worksetCoordinateRangeAttr)) {
      dynamicRanges.push_back(range);
      return;
    }
    if (extent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        extent.getValue() > 1)
      dynamicRanges.push_back(range);
  });
  // Classify ranges consumed by structured operations before considering any
  // pointwise reblocking.  Their logical extent belongs to the structured
  // traversal; replacing it with a pointwise fragment extent would truncate
  // fold/scan/reduction/contract semantics before the owning pass sees them.
  llvm::SmallPtrSet<Operation *, 32> internalTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> structuredTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> reductionTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> reductionFreeRanges;
  llvm::SmallDenseSet<uint64_t> scanSegmentDimensions;
  llvm::SmallDenseSet<PhysicalSourceAxis> scanSegmentSources;
  auto collectAxisInto = [&](Value source, uint64_t axis,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis >= fragment.getAxisMaps().size())
      return;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    collectProducerRanges(
        source,
        PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
        ranges);
  };
  std::function<void(Value, llvm::SmallPtrSetImpl<Operation *> &)>
      collectAllAxesInto = [&](Value value,
                               llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment) {
      if (auto record = value.getDefiningOp<MakeRecordOp>())
        for (Value field : record.getFields())
          collectAllAxesInto(field, ranges);
      return;
    }
    PhysicalRangeFact all = PhysicalProgramAnalysis(kernel).sourceRanges(value);
    for (MakeRangeOp range : all.roots)
      ranges.insert(range.getOperation());
    for (Attribute attribute : fragment.getAxisMaps()) {
      auto mapping = cast<AxisMapAttr>(attribute);
      collectProducerRanges(
          value,
          PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
          ranges);
    }
  };
  auto collectStructuredRegionRanges = [&](Operation *structured,
                                           ValueRange sources, uint64_t axis) {
    SmallVector<PhysicalSourceAxis> segmentSources;
    for (Value source : sources) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment || axis >= fragment.getAxisMaps().size())
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      PhysicalSourceAxis physical{mapping.getSourceId(),
                                  mapping.getSourceAxis(),
                                  mapping.getDerived()};
      if (!llvm::is_contained(segmentSources, physical))
        segmentSources.push_back(physical);
    }
    for (Region &region : structured->getRegions())
      region.walk([&](MakeRangeOp range) {
        if (llvm::is_contained(
                segmentSources,
                PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()}))
          structuredTraversalRanges.insert(range.getOperation());
      });
  };
  kernel.walk([&](RegionFoldOp fold) {
    ValueRange sources = fold.getInputs().take_front(fold.getSourceCount());
    for (Value source : sources)
      collectAxisInto(source, fold.getAxis(), structuredTraversalRanges);
    collectStructuredRegionRanges(fold, sources, fold.getAxis());
  });
  auto recordScanTraversal = [&](Value source, uint64_t axis) {
    collectAxisInto(source, axis, structuredTraversalRanges);
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis >= fragment.getAxisMaps().size())
      return;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    int64_t dimension = mapping.getDimensionId();
    if (dimension > 0)
      scanSegmentDimensions.insert(static_cast<uint64_t>(dimension));
    scanSegmentSources.insert({mapping.getSourceId(), mapping.getSourceAxis(),
                               mapping.getDerived()});
  };
  kernel.walk([&](RegionScanOp scan) {
    ValueRange sources = scan.getInputs().take_front(scan.getSourceCount());
    for (Value source : sources)
      recordScanTraversal(source, scan.getAxis());
    collectStructuredRegionRanges(scan, sources, scan.getAxis());
  });
  auto collectReductionAxis = [&](Value source, int64_t axis) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalAxisRealizationFact fact = analysis.axisRealization(source, axis);
    for (MakeRangeOp range : allRanges) {
      if (!llvm::any_of(fact.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range) &&
                   analysis.lockstepRanges({root, range}).isExact();
          }))
        continue;
      structuredTraversalRanges.insert(range.getOperation());
      reductionTraversalRanges.insert(range.getOperation());
    }
  };
  kernel.walk([&](ReduceOp reduce) {
    // A newly formed loop guard reduces the physical ownership tile, not an
    // author traversal. Its cardinality follows the tile selected by this pass.
    if (laneReductions.contains(reduce))
      return;
    for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      for (int64_t axis : reduce.getAxes())
        collectReductionAxis(source, axis);
      for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
        if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          continue;
        PhysicalAxisRealizationFact fact =
            PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
        for (MakeRangeOp range : fact.roots) {
          structuredTraversalRanges.insert(range.getOperation());
          reductionFreeRanges.insert(range.getOperation());
        }
      }
    }
  });
  bool scanCoverageFailed = false;
  kernel.walk([&](ScanOp scan) {
    for (Value source : scan.getInputs().take_front(scan.getSourceCount()))
      recordScanTraversal(source, scan.getAxis());
    for (Value source : scan.getInputs().take_front(scan.getSourceCount()))
      scanCoverageFailed |=
          failed(requireScanFullCoverage(kernel, scan, source, scan.getAxis()));
  });
  if (scanCoverageFailed)
    return kernel.emitError(
        "dynamic scan axis has no launch-visible full-coverage realization");
  SmallVector<std::pair<Value, uint64_t>> structuredReductionSources;
  kernel.walk([&](ReduceOp reduce) {
    if (laneReductions.contains(reduce))
      return;
    for (Value source :
         reduce.getInputs().take_front(reduce.getSourceCount())) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      for (int64_t axis : reduce.getAxes()) {
        if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
          continue;
        auto mapping =
            cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
            source,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true);
        bool structuredBlocker = llvm::any_of(
            replay.blockers, [](Operation *blocker) {
              return isa<ScanOp, scf::ForOp>(blocker);
            });
        if (structuredBlocker)
          structuredReductionSources.emplace_back(source,
                                                   static_cast<uint64_t>(axis));
      }
    }
  });
  for (auto [source, axis] : structuredReductionSources)
    if (failed(requireStructuredReductionFullCoverage(kernel, source, axis)))
      return kernel.emitError(
          "structured reduction source has no exact full-coverage realization");
  kernel.walk([&](HistogramOp histogram) {
    llvm::SmallPtrSet<Operation *, 16> histogramRanges;
    collectAllAxesInto(histogram.getValues(), histogramRanges);
    structuredTraversalRanges.insert(histogramRanges.begin(),
                                     histogramRanges.end());
    reductionTraversalRanges.insert(histogramRanges.begin(),
                                    histogramRanges.end());
  });
  llvm::SmallPtrSet<Operation *, 32> contractionTraversalRanges;
  kernel.walk([&](ContractOp contract) {
    auto collectConsumedAxes = [&](Value operand, ArrayRef<int64_t> axes) {
      auto type = cast<FragmentType>(operand.getType());
      for (int64_t axis : axes) {
        auto source = sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[axis]));
        // A source can occupy both K and a retained operand position. Its
        // retained occurrence still needs the contract's free-axis mapping.
        if (!queryFragmentAxes(contract.getResult().getType(), source).empty())
          continue;
        collectReductionAxis(operand, axis);
      }
    };
    collectConsumedAxes(contract.getLhs(), contract.getLhsReductionAxes());
    collectConsumedAxes(contract.getRhs(), contract.getRhsReductionAxes());
    collectAllAxesInto(contract.getLhs(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhs(), contractionTraversalRanges);
    PhysicalContractFreeAxisFact freeAxes =
        PhysicalProgramAnalysis(kernel).contractFreeAxes(contract);
    bool directFreeCoordinates = freeAxes.isExact() &&
        llvm::all_of(freeAxes.axes, [&](const auto &axis) {
          auto load = axis.operand.template getDefiningOp<LoadOp>();
          FailureOr<AxisMapAttr> mapping =
              queryAxisMap(axis.operand.getType(), axis.operandAxis);
          return load && succeeded(mapping) &&
                 succeeded(queryCoordinatePosition(
                     load.getCoordinates(), sourceAxisIdentity(*mapping)));
        });
    if (directFreeCoordinates) {
      llvm::SmallPtrSet<Operation *, 16> visited;
      collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
    }
  });
  kernel.walk([&](ScaledContractOp contract) {
    collectAllAxesInto(contract.getLhs(), contractionTraversalRanges);
    collectAllAxesInto(contract.getLhsScale(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhs(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhsScale(), contractionTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  kernel.walk([&](SparseContractOp contract) {
    collectAllAxesInto(contract.getCompressed(), contractionTraversalRanges);
    collectAllAxesInto(contract.getMetadata(), contractionTraversalRanges);
    collectAllAxesInto(contract.getRhs(), contractionTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  SmallVector<MakeRangeOp> contractionAuthorities;
  for (Operation *operation : contractionTraversalRanges)
    if (auto range = dyn_cast<MakeRangeOp>(operation))
      contractionAuthorities.push_back(range);
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(contractionAuthorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(authority, range);
        }))
      contractionTraversalRanges.insert(range.getOperation());
  structuredTraversalRanges.insert(contractionTraversalRanges.begin(),
                                   contractionTraversalRanges.end());
  SmallVector<MakeRangeOp> structuredAuthorities;
  for (Operation *operation : structuredTraversalRanges)
    if (auto range = dyn_cast<MakeRangeOp>(operation))
      structuredAuthorities.push_back(range);
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(structuredAuthorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(authority, range);
        }))
      structuredTraversalRanges.insert(range.getOperation());
  llvm::SmallPtrSet<Operation *, 16> reuseTraversalRanges;
  llvm::MapVector<Attribute, SmallVector<int64_t>> effectLocalOrigins;
  llvm::MapVector<Attribute, MakeRangeOp> effectLocalAuthorities;
  auto effectLocalKey = [&](MakeRangeOp range) -> FailureOr<Attribute> {
    FailureOr<int64_t> dimension = queryRangeDimension(range);
    if (failed(dimension))
      return failure();
    return ArrayAttr::get(
        kernel.getContext(),
        {sourceAxisKey(kernel.getContext(), sourceAxisIdentity(range)),
         dimensionAxisKey(kernel.getContext(), *dimension)});
  };
  SmallVector<StoreOp> candidateStores;
  kernel.walk([&](StoreOp store) { candidateStores.push_back(store); });
  SmallVector<Attribute> postStructuredWritebackKeys;
  llvm::SmallPtrSet<Operation *, 16> reductionCaptureWritebackRanges;
  llvm::SmallPtrSet<Operation *, 16> boundedWritebackRanges;
  for (StoreOp store : candidateStores) {
    for (MakeRangeOp range : allRanges) {
      if (!storeAxisForRange(store, range))
        continue;
      FailureOr<int64_t> dimension = queryRangeDimension(range);
      if (failed(dimension))
        continue;
      PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
          store.getValue(), sourceAxisIdentity(range),
          PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
          store.getOperation(), *dimension);
      bool regionReduction = !replay.structuredPrograms.empty() &&
          llvm::all_of(replay.structuredPrograms, [](Operation *operation) {
            auto fold = dyn_cast<RegionFoldOp>(operation);
            return fold &&
                   static_cast<ParameterCategory>(
                       fold.getSegment().getCategory()) ==
                       ParameterCategory::RegionReduction;
          });
      if (!replay.isReplayable() || !regionReduction)
        continue;
      FailureOr<Attribute> key = effectLocalKey(range);
      if (succeeded(key) &&
          !llvm::is_contained(postStructuredWritebackKeys, *key))
        postStructuredWritebackKeys.push_back(*key);
    }
  }
  SmallVector<SmallVector<Value, 4>> writeCoordinates;
  SmallVector<Operation *> writeOperations;
  auto rememberWriteCoordinates = [&](Operation *operation,
                                      ValueRange coordinates) {
    writeOperations.push_back(operation);
    writeCoordinates.emplace_back(coordinates.begin(), coordinates.end());
  };
  kernel.walk([&](StoreOp store) {
    rememberWriteCoordinates(store, store.getCoordinates());
  });
  kernel.walk([&](AtomicStoreOp store) {
    rememberWriteCoordinates(store, store.getCoordinates());
  });
  kernel.walk([&](AtomicRMWOp store) {
    rememberWriteCoordinates(store, store.getCoordinates());
  });
  kernel.walk([&](AtomicCompareExchangeOp store) {
    rememberWriteCoordinates(store, store.getCoordinates());
  });
  kernel.walk([&](ScatterReduceOp scatter) {
    rememberWriteCoordinates(scatter, scatter.getCoordinates());
  });
  auto coordinatesUseRange = [&](ValueRange coordinates, MakeRangeOp range) {
    return llvm::any_of(coordinates, [&](Value coordinate) {
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectCoordinateRanges(coordinate, ranges);
      return ranges.contains(range.getOperation());
    });
  };
  auto coordinatesDirectlyUseRange = [&](ValueRange coordinates,
                                         MakeRangeOp range) {
    if (!coordinatesUseRange(coordinates, range))
      return false;
    PhysicalSourceAxis source = sourceAxisIdentity(range);
    return llvm::any_of(coordinates, [&](Value coordinate) {
      if (!queryFragmentAxis(coordinate.getType(), source).isExact())
        return false;
      PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
          coordinate, source, PhysicalReplayScope::Coordinate,
          /*allowAccesses=*/true);
      return replay.isReplayable() && !replay.crossesAccess;
    });
  };
  auto contractionFreeAxisNeedsRange = [&](Operation *operation,
                                           MakeRangeOp range) {
    PhysicalContractFreeAxisFact fact =
        PhysicalProgramAnalysis(kernel).contractFreeAxes(operation);
    return llvm::any_of(fact.axes, [&](const auto &axis) {
      return !axis.realization.physicalized && axis.ranges.isExact() &&
             llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
               return sameLogicalRange(root, range);
             });
    });
  };
  for (StoreOp store : candidateStores) {
    SmallVector<std::pair<MakeRangeOp, int64_t>> candidates;
    llvm::SmallPtrSet<Operation *, 4> capturedRanges;
    int64_t innermostSourceAxis = -1;
    for (MakeRangeOp range : allRanges) {
      bool structuredFreeAxis =
          structuredTraversalRanges.contains(range.getOperation()) &&
          !reductionTraversalRanges.contains(range.getOperation());
      std::optional<int64_t> sourceAxis = storeAxisForRange(store, range);
      if (!sourceAxis)
        continue;
      if (structuredFreeAxis) {
        bool capturesReduction =
            llvm::all_of(writeOperations, [&](Operation *operation) {
              return operation == store.getOperation();
            }) &&
            hasFullRangeReductionCapture(store.getValue(), range,
                                         store.getOperation());
        bool usedByEveryEffect = llvm::all_of(
            writeCoordinates, [&](ArrayRef<Value> coordinates) {
              return llvm::any_of(allRanges, [&](MakeRangeOp candidate) {
                return sameLogicalRange(range, candidate) &&
                       coordinatesUseRange(coordinates, candidate);
              });
            });
        FailureOr<uint64_t> dimension = rangeDimension(range);
        PhysicalReplayFact replay = failed(dimension)
                                        ? PhysicalReplayFact()
                                        : PhysicalProgramAnalysis(kernel)
                                              .replayability(
                                                  store.getValue(),
                                                  sourceAxisIdentity(range),
                                                  PhysicalReplayScope::ValueGraph,
                                                  /*allowAccesses=*/true,
                                                  store.getOperation(),
                                                  static_cast<int64_t>(*dimension));
        PhysicalAxisProjection valueProjection =
            queryFragmentAxis(store.getValue().getType(),
                              sourceAxisIdentity(range));
        bool contractRequiresRange = llvm::any_of(
            replay.contractions, [&](Operation *contract) {
              return contractionFreeAxisNeedsRange(contract, range);
            });
        auto effectOrigin = store->getAttrOfType<IntegerAttr>(originAttr);
        if ((usedByEveryEffect && !capturesReduction) || failed(dimension) ||
            !valueProjection.isExact() ||
            valueProjection.dimensionId != static_cast<int64_t>(*dimension) ||
            !replay.isReplayable() || replay.crossesStructuredProgram ||
            !contractRequiresRange || !effectOrigin)
          continue;
        if (capturesReduction) {
          capturedRanges.insert(range.getOperation());
          reductionCaptureWritebackRanges.insert(range.getOperation());
        }
      } else if (hasAccessDependentSubregionBounds(kernel, range)) {
        if (failed(reuseTraversalDimension(kernel, ArrayRef<StoreOp>{store},
                                           range)))
          continue;
      } else {
        std::optional<int64_t> sourceDimension;
        if (FailureOr<uint64_t> dimension = rangeDimension(range);
            succeeded(dimension))
          sourceDimension = *dimension;
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
            store.getValue(), sourceAxisIdentity(range),
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
            store.getOperation(), sourceDimension);
        FailureOr<Attribute> key = effectLocalKey(range);
        bool postStructuredWriteback =
            succeeded(key) &&
            llvm::is_contained(postStructuredWritebackKeys, *key) &&
            replay.isReplayable();
        bool boundedWriteback = false;
        if (!ownershipOnly && replay.isReplayable() && !replay.crossesStructuredProgram &&
            replay.structuredPrograms.empty() && replay.contractions.empty() &&
            llvm::any_of(writeCoordinates, [&](ArrayRef<Value> coordinates) {
              return !coordinatesUseRange(coordinates, range);
            })) {
          auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
          auto footprint = estimatedFragmentRegisters(kernel, store.getValue());
          // A value filling the entire register file leaves no registers for
          // the write address or other live state. Keep a tunable traversal.
          boundedWriteback = footprint && *footprint >= capabilities.getRegistersPerUnit();
        }
        if (boundedWriteback)
          boundedWritebackRanges.insert(range.getOperation());
        if (!postStructuredWriteback && !boundedWriteback) {
          PhysicalReductionDependencyFact dependency =
              PhysicalProgramAnalysis(kernel).reductionDependency(
                  store.getValue(), sourceAxisIdentity(range), sourceDimension);
          if (!dependency.isExact() || !dependency.depends ||
              dependency.throughStructuredReduction)
            continue;
          reductionCaptureWritebackRanges.insert(range.getOperation());
        }
      }
      candidates.emplace_back(range, *sourceAxis);
      innermostSourceAxis = std::max(innermostSourceAxis, *sourceAxis);
    }
    for (auto [range, sourceAxis] : candidates) {
      internalTraversalRanges.insert(range.getOperation());
      if (sourceAxis != innermostSourceAxis &&
          !capturedRanges.contains(range.getOperation()))
        continue;
      if (structuredTraversalRanges.contains(range.getOperation()) &&
          !reductionTraversalRanges.contains(range.getOperation())) {
        FailureOr<Attribute> key = effectLocalKey(range);
        if (failed(key))
          continue;
        MakeRangeOp authority = range;
        auto existing = effectLocalAuthorities.find(*key);
        if (existing != effectLocalAuthorities.end())
          authority = existing->second;
        else
          effectLocalAuthorities[*key] = authority;
        reuseTraversalRanges.insert(authority.getOperation());
        auto origin = store->getAttrOfType<IntegerAttr>(originAttr);
        if (!origin)
          continue;
        SmallVector<int64_t> &selected = effectLocalOrigins[*key];
        if (!llvm::is_contained(selected, origin.getInt()))
          selected.push_back(origin.getInt());
      } else {
        reuseTraversalRanges.insert(range.getOperation());
      }
    }
  }
  if (!ownershipOnly)
    for (MakeRangeOp range : allRanges)
      if (reuseTraversalRanges.contains(range.getOperation()) &&
          !llvm::is_contained(dynamicRanges, range))
        dynamicRanges.push_back(range);
  llvm::SmallPtrSet<Operation *, 16> writeTraversalRanges;
  auto sharesLogicalTraversal = [&](MakeRangeOp lhs, MakeRangeOp rhs) {
    if (sameLogicalRange(lhs, rhs))
      return true;
    return PhysicalProgramAnalysis(kernel).lockstepRanges({lhs, rhs}).isExact();
  };
  for (Operation *operation : structuredTraversalRanges) {
    auto range = dyn_cast<MakeRangeOp>(operation);
    bool writeOwnership = range &&
        llvm::any_of(writeCoordinates, [&](ArrayRef<Value> coordinates) {
          return llvm::any_of(allRanges, [&](MakeRangeOp candidate) {
            return sharesLogicalTraversal(range, candidate) &&
                   (candidate->hasAttr(sourceSubregionAttr)
                        ? coordinatesDirectlyUseRange(coordinates, candidate)
                        : coordinatesUseRange(coordinates, candidate));
          });
        });
    if (writeOwnership)
      writeTraversalRanges.insert(operation);
    if (reductionTraversalRanges.contains(operation) || !writeOwnership)
      internalTraversalRanges.insert(operation);
  }
  if (ownershipOnly)
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      FailureOr<int64_t> parent = querySubregionParentDimension(range);
      bool hasLaunchParent =
          succeeded(parent) &&
          !hasAccessDependentSubregionBounds(kernel, range) &&
          succeeded(dimensionArgument(kernel, static_cast<uint64_t>(*parent)));
      return range->hasAttr(sourceSubregionAttr) &&
             extent.getKind() ==
                 static_cast<uint32_t>(PhysicalExprKind::Constant) &&
             !hasLaunchParent;
    });
  {
    llvm::SmallPtrSet<Operation *, 32> seen;
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return !seen.insert(range.getOperation()).second;
    });
  }
  if (!ownershipOnly) {
    SmallVector<MakeRangeOp> unresolved;
    for (MakeRangeOp range : dynamicRanges) {
      if (internalTraversalRanges.contains(range.getOperation())) {
        unresolved.push_back(range);
        continue;
      }
      FailureOr<ParameterOp> parameter = queryBlockingParameter(kernel, range);
      if (failed(parameter) ||
          parameter->getParameter().getRole() !=
              static_cast<uint32_t>(ParameterRole::ScanChunk)) {
        unresolved.push_back(range);
        continue;
      }
      // Region fold/scan owns this logical traversal.  Its physical segment
      // parameter is the loop step, not a replacement for the logical source
      // extent; replacing the extent would silently truncate the program to
      // one segment before structured lowering sees it.
    }
    dynamicRanges = std::move(unresolved);
    unresolved.clear();
    for (MakeRangeOp range : dynamicRanges) {
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      const bool fixedSubregion =
          range->hasAttr(sourceSubregionAttr) &&
          physicalExtent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant);
      if (!fixedSubregion && boundedWritebackRanges.contains(range.getOperation()) &&
          reuseTraversalRanges.contains(range.getOperation())) {
        unresolved.push_back(range);
        continue;
      }
      if (PhysicalProgramAnalysis(kernel)
              .axisRealization(range.getResult(), 0)
              .constructionScalarSeed) {
        FailureOr<int64_t> staticExtent = exactStaticTraversalExtent(
            PhysicalProgramAnalysis(kernel).axisRanges(range.getResult(), 0));
        if (failed(staticExtent) ||
            physicalExtent.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            !samePhysicalScalarExpression(range.getStart(),
                                          range.getLogicalStart())) {
          unresolved.push_back(range);
          continue;
        }
        uint64_t covered = llvm::PowerOf2Ceil(
            static_cast<uint64_t>(std::max<int64_t>(*staticExtent, 1)));
        if (covered > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          return range.emitOpError("static traversal extent exceeds index range");
        physicalExtent = expression(kernel.getContext(),
                                    PhysicalExprKind::Constant, covered);
        retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                             physicalExtent);
        fragment = cast<FragmentType>(range.getResult().getType());
      }
      auto fixedExtent = constantPhysicalExpression(physicalExtent);
      if (fixedExtent && *fixedExtent > 0 &&
          !llvm::isPowerOf2_64(*fixedExtent) &&
          isUnitStepRange(range)) {
        uint64_t padded = llvm::PowerOf2Ceil(
            static_cast<uint64_t>(*fixedExtent));
        if (padded > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          return range.emitOpError("static fragment padding exceeds index range");
        physicalExtent = expression(kernel.getContext(),
                                    PhysicalExprKind::Constant, padded);
        auto dimension = queryRangeDimension(range);
        if (succeeded(dimension))
          retargetDimensionExtent(range.getResult(), *dimension, physicalExtent);
        else
          retargetSourceExtent(range.getResult(), sourceAxisIdentity(range), physicalExtent);
        fragment = cast<FragmentType>(range.getResult().getType());
      }
      if (!fixedSubregion && reuseTraversalRanges.contains(range.getOperation())) {
        unresolved.push_back(range);
        continue;
      }
      if (physicalExtent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant)) {
        unresolved.push_back(range);
        continue;
      }
      OpBuilder builder(range);
      Value extent = builder.create<arith::ConstantIndexOp>(
          range.getLoc(), physicalExtent.getValue());
      auto blocked = builder.create<MakeRangeOp>(
          range.getLoc(), fragment, range.getStart(), extent, range.getStep(),
          range.getLogicalStart(), range.getLogicalStop(), range.getSourceId(),
          range.getSourceAxis(), range.getDerived());
      inheritRangeAuthority(blocked, range);
      Value endFragment =
          builder.create<BroadcastOp>(range.getLoc(), fragment,
                                      range.getLogicalStop());
      auto validComparison = builder.create<CompareOp>(
          range.getLoc(), predicateType(fragment), blocked.getResult(),
          endFragment, ComparePredicate::Lt);
      validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
      Value valid = validComparison.getResult();
      range.getResult().replaceAllUsesWith(blocked.getResult());
      fixedRangePredicates[blocked.getResult()] = valid;
      range.erase();
    }
    // A fixed physical extent immediately requires executable tail validity.
    // Keeping the predicate only in a side map across later rewrites leaves it
    // without an IR use and lets dead-value cleanup invalidate the fact.
    if (!fixedRangePredicates.empty()) {
      if (failed(alignAccessResultRelations(kernel)) ||
          failed(addTailValidity(kernel, fixedRangePredicates,
                                 /*includeStores=*/true)))
        return failure();
      fixedRangePredicates.clear();
    }
    dynamicRanges = std::move(unresolved);
  }
  if (!ownershipOnly)
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return structuredTraversalRanges.contains(range.getOperation()) &&
             !writeTraversalRanges.contains(range.getOperation()) &&
             !reuseTraversalRanges.contains(range.getOperation());
    });
  if (!ownershipOnly) {
    SmallVector<MakeRangeOp> realized;
    SmallVector<MakeRangeOp> retainedFragments;
    for (MakeRangeOp range : dynamicRanges) {
      if (!reuseTraversalRanges.contains(range.getOperation()))
        continue;
      bool useReplayTraversal = true;
      SmallVector<StoreOp> currentStores;
      SmallVector<LoadOp> retainedReads;
      FailureOr<Attribute> effectKey = effectLocalKey(range);
      auto effectLocal = succeeded(effectKey)
                             ? effectLocalOrigins.find(*effectKey)
                             : effectLocalOrigins.end();
      if (effectLocal != effectLocalOrigins.end()) {
        for (int64_t origin : effectLocal->second) {
          StoreOp selected;
          bool ambiguous = false;
          kernel.walk([&](StoreOp store) {
            auto current = store->getAttrOfType<IntegerAttr>(originAttr);
            if (!current || current.getInt() != origin)
              return;
            if (selected)
              ambiguous = true;
            else
              selected = store;
          });
          if (!selected || ambiguous)
            return kernel.emitError(
                "effect-local traversal has no unique current write-effect occurrence");
          currentStores.push_back(selected);
        }
      } else {
        kernel.walk([&](StoreOp store) {
          if (storeUsesRange(store, range))
            currentStores.push_back(store);
        });
      }
      if (currentStores.empty())
        continue;
      FailureOr<SmallVector<int64_t>> traversalDimensions =
          traversalDimensionsForStores(currentStores, range);
      if (failed(traversalDimensions)) {
        retainedFragments.push_back(range);
        continue;
      }
      for (StoreOp store : currentStores) {
        PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                                  range.getDerived()};
        SmallVector<int64_t> valueDimensions;
        collectTraversalDimensions(store.getValue().getType(), source,
                                   valueDimensions);
        for (int64_t dimension : valueDimensions) {
          if (!llvm::is_contained(*traversalDimensions, dimension))
            continue;
          PhysicalProgramAnalysis analysis(kernel);
          PhysicalReplayFact replay = analysis.replayability(
              store.getValue(), source, PhysicalReplayScope::ValueGraph,
              /*allowAccesses=*/true, store.getOperation(), dimension);
          llvm::SmallPtrSet<Operation *, 32> materializationVisited;
          bool materializedFork =
              replay.crossesAccess && hasMaterializedReductionStoreFork(
                                          store.getValue(), store, source,
                                          dimension, materializationVisited);
          auto reduction = analysis.reductionDependency(store.getValue(), source, dimension);
          // Dominating contractions outside this traversal are captured SSA,
          // not work that a blocked writeback would recompute.
          bool retainedContraction = replay.isReplayable() &&
              !replay.contractions.empty() && reduction.isExact() && reduction.depends;
          if (retainedContraction) {
            auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
            for (Operation *contract : replay.contractions) {
              auto footprint = estimatedFragmentRegisters(kernel, contract->getResult(0));
              if (!footprint || *footprint >= capabilities.getRegistersPerUnit()) {
                retainedContraction = false;
                break;
              }
            }
          }
          // Replaying a read of the destination also crosses its writes from
          // earlier chunks. Preserve that snapshot unless all addresses agree.
          bool clobbersLaterChunk = llvm::any_of(
              replay.accesses, [&](Operation *access) {
                auto load = dyn_cast<LoadOp>(access);
                if (!load || load.getResource() != store.getResource() ||
                    canReplayReadAt(load, store->getNextNode()))
                  return false;
                if (load.getSourceAxes().size() != store.getSourceAxes().size())
                  return true;
                auto coordinate =
                    queryCoordinateIndex(store.getCoordinates(), source, dimension);
                if (!coordinate.isExact() || coordinate.dimensionId != dimension)
                  return true;
                bool sameCoordinates = llvm::all_of(llvm::enumerate(store.getSourceAxes()),
                    [&](auto axis) {
                      auto readAxis = llvm::find(load.getSourceAxes(), axis.value());
                      if (readAxis == load.getSourceAxes().end())
                        return false;
                      Value lhs = load.getCoordinates()[readAxis - load.getSourceAxes().begin()];
                      Value rhs = store.getCoordinates()[axis.index()];
                      if (samePhysicalScalarExpression(lhs, rhs))
                        return true;
                      auto left = lhs.getDefiningOp<MakeRangeOp>();
                      auto right = rhs.getDefiningOp<MakeRangeOp>();
                      return left && right && sameLogicalRange(left, right) &&
                          samePhysicalScalarExpression(left.getStart(), right.getStart()) &&
                          samePhysicalScalarExpression(left.getExtent(), right.getExtent());
                    });
                if (sameCoordinates)
                  return false;
                // Keep a small captured row/column as its original SSA snapshot.
                // Re-reading it after an earlier chunk writes could observe a
                // different value, including through an overlapping view layout.
                auto type = dyn_cast<FragmentType>(load.getType());
                auto selected = type ? queryFragmentAxis(type, source)
                                     : PhysicalAxisProjection{};
                auto footprint = estimatedFragmentRegisters(kernel, load.getResult());
                auto authority = selected.isExact()
                    ? queryExactLogicalRange(analysis.axisRanges(load.getResult(), selected.fragmentAxis))
                    : FailureOr<MakeRangeOp>(failure());
                if (type && type.getShape().size() == 1 && selected.isExact() &&
                    selected.dimensionId == dimension && succeeded(authority) &&
                    isUnitStepRange(*authority) && sameLogicalRange(*authority, range) &&
                    footprint && *footprint < kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr).getRegistersPerUnit()) {
                  if (!llvm::is_contained(retainedReads, load))
                    retainedReads.push_back(load);
                  return false;
                }
                return true;
              });
          if (clobbersLaterChunk) {
            for (PhysicalAxisProjection projection :
                 queryFragmentAxes(store.getValue().getType(), source))
              if (projection.dimensionId == dimension &&
                  failed(requireFullDimensionCoverage(
                      kernel, store.getValue(), projection.fragmentAxis)))
                return failure();
            useReplayTraversal = false;
            continue;
          }
          useReplayTraversal &=
              replay.isReplayable() && !materializedFork && !retainedContraction &&
              (effectLocal == effectLocalOrigins.end() ||
               !replay.crossesStructuredProgram);
        }
      }
      // Keep non-replayable control and shared reduction producers intact.
      // The internal-range path below binds their full-coverage extent,
      // avoiding control cloning and duplicate costly fragment computation.
      if (!useReplayTraversal) {
        retainedFragments.push_back(range);
        continue;
      }
      // Close snapshots before constructing slices with the same axis identity.
      for (LoadOp read : retainedReads)
        if (failed(realizeFullCoverageDimension(kernel, read.getResult(), 0)))
          return failure();
      if (failed(realizeReusePointwiseTraversal(
              kernel, range, currentStores,
              effectLocal != effectLocalOrigins.end(), retainedReads)))
        return failure();
      realized.push_back(range);
    }
    for (MakeRangeOp range : retainedFragments) {
      reuseTraversalRanges.erase(range.getOperation());
      internalTraversalRanges.insert(range.getOperation());
    }
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return llvm::is_contained(realized, range);
    });
    eraseDeadPhysicalValues(kernel);
    allRanges.clear();
    kernel.walk([&](MakeRangeOp range) { allRanges.push_back(range); });
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return !llvm::is_contained(allRanges, range) || range.getResult().use_empty();
    });
  }
  if (dynamicRanges.empty()) {
    if (ownershipOnly && failed(alignHistogramOutputOwnership(kernel)))
      return failure();
    if (!ownershipOnly && failed(realizeOwnedHistograms(kernel)))
      return failure();
    if (!ownershipOnly &&
        (failed(alignAccessResultRelations(kernel)) ||
         failed(addTailValidity(kernel, fixedRangePredicates,
                                /*includeStores=*/true))))
      return failure();
    return finalizeValueRelations();
  }

  SmallVector<DelinearizeOp> mappings;
  kernel.walk([&](DelinearizeOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return kernel.emitError(
        "dynamic pointwise blocking requires one explicit execution workset");
  DelinearizeOp mapping = mappings.front();

  struct WriteEffectFacts {
    SmallVector<Value> coordinates;
    SmallVector<Value> payloads;
  };
  SmallVector<WriteEffectFacts> writeEffects;
  uint32_t pointwiseElementBitWidth = 0;
  llvm::SmallDenseSet<PhysicalSourceAxis> ownershipSources;
  llvm::SmallDenseSet<PhysicalSourceAxis> directOwnershipSources;
  auto collectOwnership = [&](ValueRange coordinates, ValueRange payloads) {
    WriteEffectFacts effect;
    effect.coordinates.append(coordinates.begin(), coordinates.end());
    effect.payloads.append(payloads.begin(), payloads.end());
    writeEffects.push_back(std::move(effect));
    for (Value payload : payloads)
      pointwiseElementBitWidth =
          std::max(pointwiseElementBitWidth,
                   physicalElementBitWidth(payload.getType()));
    for (Value coordinate : coordinates) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment)
        continue;
      for (Attribute attribute : fragment.getAxisMaps()) {
        auto mapping = cast<AxisMapAttr>(attribute);
        ownershipSources.insert(
            {mapping.getSourceId(), mapping.getSourceAxis(),
             mapping.getDerived()});
      }
    }
  };
  kernel.walk([&](StoreOp store) {
    collectOwnership(store.getCoordinates(), ValueRange(store.getValue()));
  });
  kernel.walk([&](AtomicStoreOp store) {
    collectOwnership(store.getCoordinates(), ValueRange(store.getValue()));
  });
  kernel.walk([&](AtomicRMWOp store) {
    collectOwnership(store.getCoordinates(), ValueRange(store.getValue()));
  });
  kernel.walk([&](AtomicCompareExchangeOp store) {
    SmallVector<Value> payloads{store.getExpected(), store.getDesired()};
    collectOwnership(store.getCoordinates(), payloads);
  });
  kernel.walk([&](ScatterReduceOp scatter) {
    collectOwnership(scatter.getCoordinates(), ValueRange(scatter.getValue()));
  });
  for (const WriteEffectFacts &effect : writeEffects)
    for (Value coordinate : effect.coordinates) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment)
        continue;
      for (Attribute attribute : fragment.getAxisMaps()) {
        auto mapping = cast<AxisMapAttr>(attribute);
        PhysicalSourceAxis source{mapping.getSourceId(), mapping.getSourceAxis(),
                                  mapping.getDerived()};
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
            coordinate, source, PhysicalReplayScope::Coordinate,
            /*allowAccesses=*/true);
        if (replay.isReplayable() && !replay.crossesAccess)
          directOwnershipSources.insert(source);
      }
    }
  // Structured traversal ownership is attached to the exact range operations
  // above, not erased source-wide here.  One immutable source axis may have a
  // query ownership projection and an independent fold/scan segment
  // projection; conflating them would discard a real physical decision.

  if (auto roles =
          mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr)) {
    SmallVector<int64_t> updatedRoles(roles.asArrayRef());
    kernel.walk([&](WorksetCoordinateOp coordinate) {
      auto axis = coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr);
      if (!axis || axis.getInt() < 0 ||
          static_cast<size_t>(axis.getInt()) >= updatedRoles.size())
        return;
      auto crossesDataLoad = [&](Value root) {
        SmallVector<std::pair<Value, bool>> worklist{{root, false}};
        llvm::SmallDenseSet<Value> beforeLoad;
        llvm::SmallDenseSet<Value> afterLoad;
        while (!worklist.empty()) {
          auto [value, crossed] = worklist.pop_back_val();
          auto &visited = crossed ? afterLoad : beforeLoad;
          if (!visited.insert(value).second)
            continue;
          if (value == coordinate.getResult())
            return crossed;
          Operation *definition = value.getDefiningOp();
          if (!definition)
            continue;
          bool nextCrossed = crossed || isa<LoadOp, GatherOp>(definition);
          for (Value operand : definition->getOperands())
            worklist.emplace_back(operand, nextCrossed);
        }
        return false;
      };
      bool indirect = llvm::any_of(writeEffects, [&](const auto &effect) {
        return llvm::any_of(effect.coordinates, crossesDataLoad);
      });
      updatedRoles[axis.getInt()] = static_cast<int64_t>(
          indirect ? CoordinateRole::IndirectTraversal
                   : CoordinateRole::Workset);
    });
    mapping->setAttr(coordinateRolesAttr,
                     DenseI64ArrayAttr::get(module.getContext(), updatedRoles));
  }

  llvm::MapVector<Attribute, SmallVector<MakeRangeOp>> axes;
  llvm::DenseMap<Attribute, ParameterOp> parameters;
  llvm::SmallDenseSet<Attribute> ownershipAxes;
  llvm::SmallDenseSet<Attribute> internalAxes;
  llvm::SmallDenseSet<uint64_t> partiallyCarriedStructuredDimensions;
  llvm::DenseMap<uint64_t, ParameterCategory> structuredOwnershipCategories;
  kernel.walk([&](RegionFoldOp fold) {
    auto category = static_cast<ParameterCategory>(
        fold.getSegment().getCategory());
    if (category != ParameterCategory::RegionContraction &&
        category != ParameterCategory::RegionReduction)
      return;
    for (Value result : fold.getResults()) {
      llvm::SmallDenseSet<uint64_t> resultDimensions;
      collectPhysicalDimensions(result.getType(), resultDimensions);
      collectPartiallyCarriedDimensions(
          result.getType(), partiallyCarriedStructuredDimensions);
      for (uint64_t dimension : resultDimensions) {
        auto existing = structuredOwnershipCategories.find(dimension);
        if (existing == structuredOwnershipCategories.end() ||
            category == ParameterCategory::RegionContraction)
          structuredOwnershipCategories[dimension] = category;
      }
    }
  });
  kernel.walk([&](HistogramOp histogram) {
    llvm::SmallDenseSet<uint64_t> resultDimensions;
    collectPhysicalDimensions(histogram.getResult().getType(), resultDimensions);
    for (uint64_t dimension : resultDimensions)
      structuredOwnershipCategories.try_emplace(
          dimension, ParameterCategory::Histogram);
  });
  llvm::SmallDenseSet<uint64_t> nonUniqueContractionDimensions;
  kernel.walk([&](Operation *operation) {
    if (!isa<ContractOp, ScaledContractOp, SparseContractOp>(operation))
      return;
    llvm::SmallDenseSet<uint64_t> seen;
    for (const auto &axis : PhysicalProgramAnalysis(kernel).contractFreeAxes(operation).axes) {
      FailureOr<AxisMapAttr> mapping = queryAxisMap(axis.operand.getType(), axis.operandAxis);
      if (failed(mapping) || mapping->getDimensionId() <= 0)
        continue;
      uint64_t dimension = mapping->getDimensionId();
      if (!seen.insert(dimension).second)
        nonUniqueContractionDimensions.insert(dimension);
    }
    if (auto contract = dyn_cast<ContractOp>(operation))
      for (auto [operand, reductionAxes] :
           {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
            std::pair{contract.getRhs(), contract.getRhsReductionAxes()}})
        for (int64_t axis : reductionAxes) {
          auto mapping = queryAxisMap(operand.getType(), axis);
          if (succeeded(mapping) && seen.contains(mapping->getDimensionId()))
            nonUniqueContractionDimensions.insert(mapping->getDimensionId());
        }
  });
  llvm::SmallPtrSet<Operation *, 8> contractionOwnedRanges;
  llvm::SmallPtrSet<Operation *, 8> retainedCartesianRanges;
  llvm::SmallPtrSet<Operation *, 8> independentContractionRanges;
  for (MakeRangeOp range : allRanges) {
    auto dimension = ownershipDimension(kernel, range);
    if (failed(dimension) || !nonUniqueContractionDimensions.contains(*dimension))
      continue;
    auto facts = contractFreeAxisFacts(kernel, range);
    if (facts.sides != ContractFreeAxisLhs && facts.sides != ContractFreeAxisRhs)
      continue;
    bool reduced = false;
    PhysicalProgramAnalysis analysis(kernel);
    kernel.walk([&](ContractOp contract) {
      auto freeAxes = analysis.contractFreeAxes(contract);
      reduced |= llvm::count_if(freeAxes.axes, [&](const auto &axis) {
        return llvm::any_of(axis.ranges.roots, [&](MakeRangeOp root) {
          return sameLogicalRange(root, range);
        });
      }) > 1;
      for (auto [operand, axes] :
           {std::pair{contract.getLhs(), contract.getLhsReductionAxes()},
            std::pair{contract.getRhs(), contract.getRhsReductionAxes()}})
        for (int64_t axis : axes) {
          auto roots = analysis.axisRanges(operand, axis);
          reduced |= !roots.isExact() || !roots.blockers.empty() ||
              llvm::any_of(roots.roots, [&](MakeRangeOp root) {
                return sameLogicalRange(root, range);
              });
        }
    });
    kernel.walk([&](Operation *operation) {
      if (!isa<ScaledContractOp, SparseContractOp>(operation))
        return;
      for (Value operand : operation->getOperands())
        reduced |= !queryFragmentAxes(operand.getType(), sourceAxisIdentity(range))
                        .empty();
    });
    if (!reduced)
      independentContractionRanges.insert(range.getOperation());
  }
  auto hasPointwiseOwnership = [&](MakeRangeOp range) {
    if (contractionOwnedRanges.contains(range.getOperation()) ||
        retainedCartesianRanges.contains(range.getOperation()))
      return false;
    FailureOr<uint64_t> dimension = ownershipDimension(kernel, range);
    // Equal dimensions do not identify Cartesian coordinates. Only a distinct
    // free occurrence can bind source-specific ownership here; contraction
    // blocking owns the remaining operand/result positions.
    if (succeeded(dimension) &&
        nonUniqueContractionDimensions.contains(*dimension) &&
        !independentContractionRanges.contains(range.getOperation()))
      return false;
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    bool effectOwned = ownershipSources.contains(source) ||
                       writeTraversalRanges.contains(range.getOperation());
    return effectOwned &&
           (!range->hasAttr(sourceSubregionAttr) ||
            directOwnershipSources.contains(source) ||
            writeTraversalRanges.contains(range.getOperation())) &&
           !reductionTraversalRanges.contains(range.getOperation()) &&
           !scanSegmentSources.contains(source) &&
           (failed(dimension) ||
            (!scanSegmentDimensions.contains(*dimension) &&
             !partiallyCarriedStructuredDimensions.contains(*dimension)));
  };

  llvm::DenseMap<Operation *, MakeRangeOp> occurrenceRoots;
  llvm::DenseMap<Operation *, ParameterOp> occurrenceParameters;
  llvm::SmallPtrSet<Operation *, 8> positionalOccurrences;
  llvm::SmallDenseSet<uint64_t> independentCartesianDimensions;
  uint64_t nextOccurrenceSource = nextPhysicalAxisIdentities(kernel).first;
  for (MakeRangeOp range : allRanges) {
    if (!independentContractionRanges.contains(range.getOperation()) ||
        !ownershipSources.contains(sourceAxisIdentity(range)))
      continue;
    MakeRangeOp root = range;
    for (const auto &entry : occurrenceRoots)
      if (sameLogicalRange(range, entry.second)) {
        root = entry.second;
        break;
      }
    occurrenceRoots[range.getOperation()] = root;
    positionalOccurrences.insert(range.getOperation());
  }
  WalkResult occurrences = kernel.walk([&](StoreOp store) {
    auto valueType = dyn_cast<FragmentType>(store.getValue().getType());
    SmallVector<Value> coordinates;
    for (Value coordinate : store.getCoordinates()) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      coordinates.push_back(coordinate);
    }
    if (!valueType || coordinates.empty())
      return WalkResult::advance();
    bool cartesian = valueType.getShape().size() == coordinates.size() &&
        llvm::all_of(coordinates, [](Value value) {
          return cast<FragmentType>(value.getType()).getShape().size() == 1;
        });
    if (cartesian) {
      PhysicalProgramAnalysis analysis(kernel);
      llvm::DenseMap<uint64_t, unsigned> positions;
      for (auto [position, coordinate] : llvm::enumerate(coordinates))
        for (MakeRangeOp range : analysis.axisRanges(coordinate, 0).roots) {
          auto dimension = ownershipDimension(kernel, range);
          if (failed(dimension))
            continue;
          auto [previous, inserted] = positions.try_emplace(*dimension, position);
          if (!inserted && previous->second != position)
            independentCartesianDimensions.insert(*dimension);
        }
    }
    auto ordered = store->getParentOfType<scf::ForOp>();
    if (cartesian && ordered && !ordered->hasAttr(independentIterationAttr)) {
      PhysicalProgramAnalysis analysis(kernel);
      DominanceInfo dominance(kernel);
      llvm::DenseMap<Operation *, unsigned> sourceAxes;
      SmallVector<MakeRangeOp> shared;
      for (unsigned axis = 0; axis < valueType.getShape().size(); ++axis) {
        auto ranges = analysis.axisRanges(store.getValue(), axis);
        if (!ranges.isExact() || !ranges.blockers.empty())
          continue;
        for (MakeRangeOp range : ranges.roots) {
          auto dimension = queryRangeDimension(range);
          if (failed(dimension) || nonUniqueContractionDimensions.contains(*dimension) ||
              isProvablySingletonLogicalRange(range) ||
              (!ordered->isProperAncestor(range) &&
               (!dominance.dominates(range.getOperation(), ordered.getOperation()) ||
                !hasExactStaticFullCoverage(kernel, range.getResult(), 0))))
            continue;
          auto [previous, inserted] = sourceAxes.try_emplace(range.getOperation(), axis);
          if (!inserted && previous->second != axis &&
              !llvm::is_contained(shared, range))
            shared.push_back(range);
        }
      }
      // One vector can feed both axes of an ordered Cartesian update. Preserve
      // its full snapshot, including an already complete dominating capture.
      for (MakeRangeOp range : allRanges) {
        bool local = ordered->isProperAncestor(range);
        if (range->hasAttr(worksetCoordinateRangeAttr) ||
            (!local && !llvm::is_contained(shared, range)) ||
            !llvm::any_of(shared, [&](MakeRangeOp root) {
              auto dimension = queryRangeDimension(range);
              auto rootDimension = queryRangeDimension(root);
              return sameLogicalRange(range, root) && succeeded(dimension) &&
                     succeeded(rootDimension) && *dimension == *rootDimension;
            }))
          continue;
        if (local &&
            failed(requireFullDimensionCoverage(kernel, range.getResult(), 0))) {
          store.emitOpError("shared Cartesian producer has no exact local full coverage");
          return WalkResult::interrupt();
        }
        retainedCartesianRanges.insert(range.getOperation());
        internalTraversalRanges.insert(range.getOperation());
        occurrenceRoots.erase(range.getOperation());
      }
      valueType = cast<FragmentType>(store.getValue().getType());
    }
    PhysicalProgramAnalysis analysis(kernel);
    llvm::DenseMap<Operation *, unsigned> resultAxes;
    for (auto [axis, attribute] : llvm::enumerate(valueType.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      int64_t dimension = mapping.getDimensionId();
      if (dimension <= 0 || nonUniqueContractionDimensions.contains(dimension))
        continue;
      bool repeatedDimension =
          llvm::count_if(valueType.getAxisMaps(), [&](Attribute other) {
            return cast<AxisMapAttr>(other).getDimensionId() == dimension;
          }) >= 2;
      // Rank-one Cartesian coordinates concatenate into the result axes.
      // Equal dimension values do not equate those independent occurrences.
      PhysicalRangeFact address;
      if (cartesian) {
        address = analysis.axisRanges(coordinates[axis], 0);
      } else {
        address.state = PhysicalFactState::Exact;
        for (Value coordinate : coordinates) {
          auto source = cast<FragmentType>(coordinate.getType());
          auto projection = queryAxisProjection(source, valueType);
          if (!projection.isExact()) {
            address.state = PhysicalFactState::Unknown;
            break;
          }
          auto sourceAxis = projection.targetToSource[axis];
          if (!sourceAxis)
            continue;
          PhysicalRangeFact ranges = analysis.axisRanges(coordinate, *sourceAxis);
          if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty()) {
            address.state = PhysicalFactState::Unknown;
            break;
          }
          for (MakeRangeOp range : ranges.roots)
            if (!llvm::is_contained(address.roots, range))
              address.roots.push_back(range);
        }
        if (!address.roots.empty() &&
            !analysis.lockstepRanges(address.roots).isExact())
          address.state = PhysicalFactState::Unknown;
        address.unitStep = llvm::all_of(address.roots, isUnitStepRange);
      }
      PhysicalRangeFact payload = analysis.axisRanges(store.getValue(), axis);
      if (llvm::any_of(payload.roots, [&](MakeRangeOp range) {
            return retainedCartesianRanges.contains(range.getOperation());
          }))
        continue;
      if (cartesian && address.isExact() && !address.roots.empty() &&
          address.blockers.empty() && payload.isExact() && payload.roots.empty() &&
          payload.blockers.empty()) {
        // A uniform payload still writes an independent Cartesian address
        // axis. Give that occurrence its own source-bound ownership parameter.
        for (MakeRangeOp range : address.roots)
          occurrenceRoots.try_emplace(range.getOperation(), address.roots.front());
        continue;
      }
      if (address.state == PhysicalFactState::Unknown ||
          payload.state == PhysicalFactState::Unknown ||
          !address.blockers.empty() || !payload.blockers.empty() ||
          address.roots.empty() || payload.roots.empty())
        continue;
      if (llvm::all_of(address.roots, isProvablySingletonLogicalRange))
        continue;
      if (llvm::any_of(payload.roots, [&](MakeRangeOp range) {
            auto dimension = queryRangeDimension(range);
            return succeeded(dimension) &&
                   nonUniqueContractionDimensions.contains(*dimension);
          })) {
        // Positional output rebinding can hide repeated contraction free axes
        // behind different address dimensions. Their operand occurrences must
        // remain separate until contraction blocking binds each side.
        for (MakeRangeOp range : address.roots)
          contractionOwnedRanges.insert(range.getOperation());
        // Epilogue reads can materialize the same output coordinates through
        // separate ranges. Their tile must follow the contraction as well.
        for (MakeRangeOp range : payload.roots)
          contractionOwnedRanges.insert(range.getOperation());
        continue;
      }
      auto sameBound = [](Value lhs, Value rhs) {
        if (samePhysicalScalarExpression(lhs, rhs))
          return true;
        PhysicalExprAttr left = queryLaunchExpression(lhs);
        PhysicalExprAttr right = queryLaunchExpression(rhs);
        return left && right && left == right;
      };
      bool equivalentSources = cartesian &&
          !repeatedDimension && address.isExact() &&
          llvm::any_of(payload.roots, [&](MakeRangeOp range) {
            return llvm::none_of(address.roots, [&](MakeRangeOp coordinate) {
              return sourceAxisIdentity(range) == sourceAxisIdentity(coordinate);
            });
          }) && llvm::all_of(payload.roots, [&](MakeRangeOp range) {
            auto sourceDimension = queryRangeDimension(range);
            return succeeded(sourceDimension) && *sourceDimension == dimension &&
                   llvm::all_of(address.roots, [&](MakeRangeOp coordinate) {
                     auto addressDimension = queryRangeDimension(coordinate);
                     bool sameExtent = succeeded(addressDimension) &&
                                       *addressDimension == *sourceDimension;
                     // A positional store pairs ordinals of the same logical
                     // extent even when the source and destination start at
                     // different coordinates. Keep each range's own origin.
                     return sameBound(range.getStep(), coordinate.getStep()) &&
                            (sameExtent ||
                             (sameBound(range.getLogicalStart(),
                                        coordinate.getLogicalStart()) &&
                              sameBound(range.getLogicalStop(),
                                        coordinate.getLogicalStop())));
                   });
          });
      bool positionalRemap = equivalentSources || llvm::any_of(payload.roots, [&](MakeRangeOp range) {
        FailureOr<int64_t> sourceDimension = queryRangeDimension(range);
        return succeeded(sourceDimension) && *sourceDimension != dimension;
      });
      if (!repeatedDimension && !positionalRemap)
        continue;
      if (positionalRemap && !address.isExact())
        continue;
      if (ownershipOnly && cartesian && positionalRemap)
        for (MakeRangeOp range : payload.roots) {
          if (reductionTraversalRanges.contains(range.getOperation()) ||
              llvm::any_of(writeEffects, [&](const WriteEffectFacts &effect) {
                return coordinatesUseRange(effect.coordinates, range);
              }))
            continue;
          bool independentAddress = false;
          for (auto [otherAxis, coordinate] : llvm::enumerate(coordinates)) {
            if (otherAxis == axis)
              continue;
            auto otherPayload = analysis.axisRanges(store.getValue(), otherAxis);
            auto otherAddress = analysis.axisRanges(coordinate, 0);
            independentAddress |= otherPayload.isExact() &&
                !llvm::is_contained(otherPayload.roots, range) &&
                otherPayload.blockers.empty() && otherAddress.isExact() &&
                llvm::any_of(otherAddress.roots, [&](MakeRangeOp other) {
                  return other != range && sameLogicalRange(other, range);
                });
          }
          if (!independentAddress)
            continue;
          // A read-only occurrence used in another result position does not
          // own the independent Cartesian address, even with the same domain.
          // Rebind its current value-axis flow, leaving logical coordinates and
          // every write-address occurrence unchanged.
          PhysicalSourceAxis previous = sourceAxisIdentity(range);
          uint64_t sourceId = nextOccurrenceSource++;
          SmallVector<std::pair<Value, FragmentType>> replacements;
          auto collect = [&](Value value) {
            auto type = dyn_cast<FragmentType>(value.getType());
            if (!type)
              return;
            SmallVector<Attribute> maps(type.getAxisMaps().begin(), type.getAxisMaps().end());
            bool changed = false;
            for (auto [position, attribute] : llvm::enumerate(type.getAxisMaps())) {
              auto mapping = cast<AxisMapAttr>(attribute);
              if (!(sourceAxisIdentity(mapping) == previous))
                continue;
              auto roots = analysis.axisRanges(value, position);
              if (roots.state == PhysicalFactState::Unknown || !roots.blockers.empty() ||
                  roots.roots.empty() ||
                  !llvm::all_of(roots.roots, [&](MakeRangeOp root) { return root == range; }))
                continue;
              maps[position] = AxisMapAttr::get(kernel.getContext(), sourceId,
                  mapping.getSourceAxis(), mapping.getDimensionId(), position, true);
              changed = true;
            }
            if (changed)
              replacements.emplace_back(value, FragmentType::get(kernel.getContext(),
                  type.getElementType(), type.getShape(), ArrayAttr::get(kernel.getContext(), maps),
                  type.getValidity(), type.getOwner()));
          };
          kernel.walk([&](Operation *operation) {
            for (Value result : operation->getResults())
              collect(result);
            for (Region &region : operation->getRegions())
              for (Block &block : region)
                for (BlockArgument argument : block.getArguments())
                  collect(argument);
          });
          for (auto [value, type] : replacements)
            value.setType(type);
          range.setSourceId(sourceId);
          range.setDerived(true);
        }
      SmallVector<MakeRangeOp> ranges(address.roots.begin(), address.roots.end());
      for (MakeRangeOp range : payload.roots)
        if (!llvm::is_contained(ranges, range))
          ranges.push_back(range);
      if (positionalRemap && !repeatedDimension) {
        // Separate access sites can materialize the same logical range. Keep
        // their positional ownership connected instead of multiplying grids.
        SmallVector<MakeRangeOp> related;
        for (auto &entry : occurrenceRoots) {
          auto previous = cast<MakeRangeOp>(entry.first);
          auto previousAxis = resultAxes.find(previous.getOperation());
          if (previousAxis != resultAxes.end() && previousAxis->second != axis)
            continue;
          if (positionalOccurrences.contains(entry.first) &&
              llvm::any_of(ranges, [&](MakeRangeOp range) {
                return sameLogicalRange(range, previous);
              }))
            related.push_back(previous);
        }
        for (MakeRangeOp range : related)
          if (!llvm::is_contained(ranges, range))
            ranges.push_back(range);
      }
      MakeRangeOp root = ranges.front();
      FailureOr<int64_t> addressExtent = exactStaticTraversalExtent(address);
      if (!llvm::all_of(ranges, [&](MakeRangeOp range) {
            if (positionalRemap) {
              // The store pairs these tensor axes by ordinal, even when the
              // address domains have different starts. Share their tile width,
              // retaining each domain's own coordinates and source identity.
              PhysicalExprAttr length = launchRangeExtent(root);
              FailureOr<int64_t> extent = exactStaticTraversalExtent(
                  analysis.axisRanges(range.getResult(), 0));
              bool sameStaticCardinality = succeeded(addressExtent) &&
                                           succeeded(extent) &&
                                           *addressExtent == *extent;
              auto rootDimension = queryRangeDimension(root);
              auto rangeDimension = queryRangeDimension(range);
              bool sameLogicalExtent = succeeded(rootDimension) &&
                                       succeeded(rangeDimension) &&
                                       *rootDimension == *rangeDimension;
              return isUnitStepRange(root) && isUnitStepRange(range) &&
                     (sameLogicalExtent || sameStaticCardinality ||
                      (length && length == launchRangeExtent(range)));
            }
            return sameBound(root.getLogicalStart(), range.getLogicalStart()) &&
                   sameBound(root.getLogicalStop(), range.getLogicalStop()) &&
                   sameBound(root.getStep(), range.getStep());
          }))
        continue;
      bool retainsReducedAxis = llvm::any_of(payload.roots, [&](MakeRangeOp range) {
        if (!reductionTraversalRanges.contains(range.getOperation()))
          return false;
        // Query the current result relation: a structured result can rename
        // the producer range without removing its reduction dependency.
        auto dependency = analysis.reductionDependency(
            store.getValue(), sourceAxisIdentity(mapping), dimension);
        return dependency.isExact() && dependency.depends;
      });
      for (MakeRangeOp range : ranges) {
        auto [found, inserted] = resultAxes.try_emplace(range.getOperation(), axis);
        if (!inserted && found->second != axis) {
          store.emitOpError("Cartesian pointwise axes require independent producer ranges")
              << "; first_axis=" << found->second << "; second_axis=" << axis
              << "; shared_range=" << range.getResult();
          return WalkResult::interrupt();
        }
      }
      for (MakeRangeOp range : ranges) {
        auto previous = occurrenceRoots.find(range.getOperation());
        if (previous == occurrenceRoots.end())
          continue;
        MakeRangeOp old = previous->second;
        for (auto &entry : occurrenceRoots)
          if (entry.second == old)
            entry.second = root;
      }
      for (MakeRangeOp range : ranges) {
        occurrenceRoots[range.getOperation()] = root;
        // The occurrence class is axis-specific even when extents repeat.
        if (positionalRemap)
          positionalOccurrences.insert(range.getOperation());
        ownershipSources.insert(sourceAxisIdentity(range));
        if (positionalRemap)
          directOwnershipSources.insert(sourceAxisIdentity(range));
        if (retainsReducedAxis) {
          reductionTraversalRanges.insert(range.getOperation());
          internalTraversalRanges.insert(range.getOperation());
        }
      }
    }
    return WalkResult::advance();
  });
  if (occurrences.wasInterrupted())
    return failure();

  // A write's range can own a resource slice only if every access to that
  // resource uses the same traversal on that axis. In particular, an initial
  // copy cannot be distributed while a later recurrence reads whole prefixes.
  llvm::SmallPtrSet<Operation *, 8> dependentResourceRanges;
  PhysicalProgramAnalysis accessAnalysis(kernel);
  auto projectedRange = [](Value value) {
    while (Operation *operation = value.getDefiningOp()) {
      if (!isa<BroadcastOp, ReshapeOp>(operation))
        break;
      value = operation->getOperand(0);
    }
    return value.getDefiningOp<MakeRangeOp>();
  };
  kernel.walk([&](StoreOp store) {
    for (auto [coordinate, axis] : llvm::zip(store.getCoordinates(), store.getSourceAxes())) {
      llvm::SmallPtrSet<Operation *, 8> roots;
      collectCoordinateRanges(coordinate, roots);
      if (roots.empty())
        continue;
      auto range = projectedRange(coordinate);
      for (Operation *user : store.getResource().getUsers()) {
        if (user == store.getOperation() || isa<DimOp, AssumeInBoundsOp>(user))
          continue;
        auto access = accessAnalysis.footprint(user);
        auto position = llvm::find(access.sourceAxes, axis);
        auto other = position == access.sourceAxes.end() ? MakeRangeOp() :
            projectedRange(access.coordinates[position - access.sourceAxes.begin()]);
        auto occurrence = range ? occurrenceRoots.lookup(range.getOperation()) : MakeRangeOp();
        bool sameOwnership = range && other &&
            (sameLogicalRange(range, other) ||
             (occurrence && occurrence == occurrenceRoots.lookup(other.getOperation())));
        if (access.state != PhysicalFactState::Exact || !sameOwnership ||
            !accessAnalysis.lockstepRanges({range, other}).isExact()) {
          dependentResourceRanges.insert(roots.begin(), roots.end());
          break;
        }
      }
    }
  });
  for (MakeRangeOp range : allRanges)
    if (llvm::any_of(dependentResourceRanges, [&](Operation *operation) {
          auto dependent = cast<MakeRangeOp>(operation);
          auto occurrence = occurrenceRoots.lookup(range.getOperation());
          return sameLogicalRange(range, dependent) ||
              (occurrence && occurrence == occurrenceRoots.lookup(operation));
        }))
      internalTraversalRanges.insert(range.getOperation());
  // Ordered writes to a parent and its subregion use one absolute tile anchor.
  // Shift the subregion's positional payload by the same ordinal displacement.
  llvm::DenseMap<Operation *, int64_t> parentAnchorOffsets;
  auto constantBound = [](Value value) -> std::optional<int64_t> {
    auto bound = queryLaunchExpression(value);
    if (!bound || bound.getKind() !=
                      static_cast<uint32_t>(PhysicalExprKind::Constant))
      return std::nullopt;
    return bound.getValue();
  };
  SmallVector<StoreOp> precedingStores;
  WalkResult parentAnchors = kernel.walk([&](StoreOp store) {
    PhysicalProgramAnalysis analysis(kernel);
    for (auto [position, coordinate] : llvm::enumerate(store.getCoordinates())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type || type.getShape().size() != 1)
        continue;
      auto child = queryExactLogicalRange(analysis.axisRanges(coordinate, 0));
      if (failed(child) || !isUnitStepRange(*child))
        continue;
      auto parentDimension = querySubregionParentDimension(*child);
      auto childStart = constantBound(child->getLogicalStart());
      auto childStop = constantBound(child->getLogicalStop());
      if (failed(parentDimension) || !childStart || !childStop || *childStart < 0)
        continue;
      std::optional<int64_t> offset;
      for (StoreOp previous : precedingStores) {
        if (previous.getResource() != store.getResource() ||
            previous->getBlock() != store->getBlock())
          continue;
        auto sourceAxis = store.getSourceAxes()[position];
        for (auto [parentPosition, parentCoordinate] :
             llvm::enumerate(previous.getCoordinates())) {
          if (previous.getSourceAxes()[parentPosition] != sourceAxis)
            continue;
          auto parentType = dyn_cast<FragmentType>(parentCoordinate.getType());
          if (!parentType || parentType.getShape().size() != 1)
            continue;
          auto parent = queryExactLogicalRange(analysis.axisRanges(parentCoordinate, 0));
          if (failed(parent) || !isUnitStepRange(*parent) ||
              !(sourceAxisIdentity(*parent) == sourceAxisIdentity(*child)))
            continue;
          auto dimension = queryRangeDimension(*parent);
          auto start = constantBound(parent->getLogicalStart());
          auto stop = constantBound(parent->getLogicalStop());
          if (failed(dimension) || *dimension != *parentDimension ||
              !start || !stop || *start < 0 || *start > *childStart ||
              *stop < *childStop)
            continue;
          int64_t current = *start - *childStart;
          if (offset && *offset != current) {
            store.emitOpError("subregion writes have incompatible parent tile anchors");
            return WalkResult::interrupt();
          }
          offset = current;
        }
      }
      if (!offset || *offset == 0)
        continue;
      MakeRangeOp root = occurrenceRoots.lookup(child->getOperation());
      if (!root)
        root = *child;
      for (MakeRangeOp range : allRanges) {
        if (range != *child && occurrenceRoots.lookup(range.getOperation()) != root)
          continue;
        auto [entry, inserted] = parentAnchorOffsets.try_emplace(range.getOperation(), *offset);
        if (!inserted && entry->second != *offset) {
          store.emitOpError("shared pointwise producer requires incompatible ordinal anchors");
          return WalkResult::interrupt();
        }
      }
    }
    precedingStores.push_back(store);
    return WalkResult::advance();
  });
  if (parentAnchors.wasInterrupted())
    return failure();
  if (ownershipOnly)
    for (auto &entry : occurrenceRoots) {
      auto range = cast<MakeRangeOp>(entry.first);
      if (!llvm::is_contained(dynamicRanges, range))
        dynamicRanges.push_back(range);
    }
  // A read-only operand may reach a write through broadcast/pointwise values
  // while retaining a different source identity from the write coordinates.
  // Share ownership only through that value path and an exact logical range.
  for (MakeRangeOp range : dynamicRanges) {
    if (range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>() ||
        reductionTraversalRanges.contains(range.getOperation()))
      continue;
    FailureOr<uint64_t> dimension = rangeDimension(range);
    if (failed(dimension))
      continue;
    SmallVector<Value> worklist{range.getResult()};
    llvm::SmallDenseSet<Value> visited;
    bool reachesWrite = false;
    while (!worklist.empty() && !reachesWrite) {
      Value value = worklist.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      reachesWrite = llvm::any_of(writeEffects, [&](const WriteEffectFacts &effect) {
        return llvm::is_contained(effect.payloads, value) &&
               llvm::any_of(allRanges, [&](MakeRangeOp owned) {
                 FailureOr<uint64_t> ownedDimension = rangeDimension(owned);
                 return succeeded(ownedDimension) && *ownedDimension == *dimension &&
                        sharesLogicalTraversal(range, owned) &&
                        coordinatesUseRange(effect.coordinates, owned);
               });
      });
      for (Operation *user : value.getUsers()) {
        if (auto yield = dyn_cast<scf::YieldOp>(user)) {
          if (auto branch = dyn_cast<scf::IfOp>(yield->getParentOp()))
            for (auto [position, yielded] : llvm::enumerate(yield.getOperands()))
              if (yielded == value && position < branch.getNumResults() &&
                  queryFragmentDimensions(branch.getResult(position).getType(),
                                          *dimension).size() == 1)
                worklist.push_back(branch.getResult(position));
          continue;
        }
        if (!isa<LoadOp, UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                 BitcastOp, BroadcastOp, TransposeOp, ReshapeOp>(user))
          continue;
        for (Value result : user->getResults())
          if (queryFragmentDimensions(result.getType(), *dimension).size() == 1)
            worklist.push_back(result);
      }
    }
    if (reachesWrite)
      ownershipSources.insert(sourceAxisIdentity(range));
  }
  auto dependsOnSource = [&](ValueRange values, PhysicalSourceAxis source) {
    PhysicalProgramAnalysis analysis(kernel);
    for (Value value : values) {
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment)
        continue;
      for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (ranges.state != PhysicalFactState::Unknown && ranges.blockers.empty() &&
            llvm::any_of(ranges.roots, [&](MakeRangeOp range) {
              return sourceAxisIdentity(range) == source;
            }))
          return true;
      }
    }
    return false;
  };
  llvm::DenseMap<PhysicalSourceAxis, uint64_t> sourceDimensions;
  llvm::MapVector<uint64_t, SmallVector<PhysicalSourceAxis>> dimensionSources;
  for (MakeRangeOp range : dynamicRanges) {
    FailureOr<uint64_t> dimension = ownershipDimension(kernel, range);
    if (failed(dimension) || !hasPointwiseOwnership(range))
      continue;
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    uint64_t dimensionId = *dimension;
    sourceDimensions[source] = dimensionId;
    if (!llvm::is_contained(dimensionSources[dimensionId], source))
      dimensionSources[dimensionId].push_back(source);
  }
  auto effectDependsOn = [&](const WriteEffectFacts &effect,
                             PhysicalSourceAxis source) {
    return dependsOnSource(effect.coordinates, source) ||
           dependsOnSource(effect.payloads, source);
  };
  llvm::SmallDenseSet<uint64_t> jointOwnershipDimensions;
  for (auto [dimensionId, sources] : dimensionSources) {
    if (sources.size() < 2)
      continue;
    bool onePreservedSourcePerEffect =
        llvm::all_of(writeEffects, [&](const WriteEffectFacts &effect) {
          return llvm::count_if(sources, [&](PhysicalSourceAxis source) {
                   return effectDependsOn(effect, source);
                 }) == 1;
        });
    if (onePreservedSourcePerEffect)
      jointOwnershipDimensions.insert(dimensionId);
  }
  for (MakeRangeOp range : dynamicRanges) {
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    if (!hasPointwiseOwnership(range)) {
      internalTraversalRanges.insert(range.getOperation());
      continue;
    }
    bool requiredByEveryEffect =
        llvm::all_of(writeEffects, [&](const WriteEffectFacts &effect) {
          if (effectDependsOn(effect, source))
            return true;
          auto root = occurrenceRoots.lookup(range.getOperation());
          if (!root || !positionalOccurrences.contains(range.getOperation()))
            return false;
          return llvm::any_of(occurrenceRoots, [&](const auto &entry) {
            return entry.second == root &&
                   positionalOccurrences.contains(entry.first) &&
                   effectDependsOn(effect,
                       sourceAxisIdentity(cast<MakeRangeOp>(entry.first)));
          });
        });
    auto dimension = sourceDimensions.find(source);
    bool jointlyOwned =
        dimension != sourceDimensions.end() &&
        jointOwnershipDimensions.contains(dimension->second);
    if (!requiredByEveryEffect && !jointlyOwned)
      internalTraversalRanges.insert(range.getOperation());
  }
  if (ownershipOnly) {
    // Equivalent coordinate occurrences must preserve the selected local
    // writeback traversal instead of restoring duplicate program ownership.
    for (MakeRangeOp range : allRanges)
      if (llvm::any_of(reductionCaptureWritebackRanges,
                       [&](Operation *operation) {
                         auto captured = cast<MakeRangeOp>(operation);
                         MakeRangeOp occurrence =
                             occurrenceRoots.lookup(range.getOperation());
                         bool sameOccurrence =
                             occurrence &&
                             occurrence == occurrenceRoots.lookup(
                                               captured.getOperation()) &&
                             positionalOccurrences.contains(range.getOperation()) &&
                             positionalOccurrences.contains(captured.getOperation());
                         return (sameLogicalRange(range, captured) ||
                                 sameOccurrence) &&
                                PhysicalProgramAnalysis(kernel)
                                    .lockstepRanges({range, captured})
                                    .isExact();
                       }))
        internalTraversalRanges.insert(range.getOperation());
    SmallVector<MakeRangeOp> ownershipSeeds;
    for (MakeRangeOp range : dynamicRanges)
      if (hasPointwiseOwnership(range) &&
          !internalTraversalRanges.contains(range.getOperation()))
        ownershipSeeds.push_back(range);
    for (MakeRangeOp seed : ownershipSeeds)
      for (MakeRangeOp range : allRanges) {
        if (range->getParentOfType<RegionFoldOp>() ||
            range->getParentOfType<RegionScanOp>() ||
            reductionTraversalRanges.contains(range.getOperation()))
          continue;
        MakeRangeOp occurrence = occurrenceRoots.lookup(seed.getOperation());
        bool sameOccurrence = occurrence &&
            occurrence == occurrenceRoots.lookup(range.getOperation()) &&
            positionalOccurrences.contains(seed.getOperation()) &&
            positionalOccurrences.contains(range.getOperation());
        // A positional store pairs producer and address axes by ordinal.
        // Their different source identities still share the selected tile's
        // ownership, including its start, not just its fragment extent.
        if (!sameLogicalRange(seed, range) && !sameOccurrence)
          continue;
        if (dependentResourceRanges.contains(range.getOperation()))
          continue;
        internalTraversalRanges.erase(range.getOperation());
        if (!llvm::is_contained(dynamicRanges, range))
          dynamicRanges.push_back(range);
      }
  }
  SmallVector<std::pair<Value, unsigned>> retainedGatherAxes;
  kernel.walk([&](GatherOp gather) {
    auto fragment = dyn_cast<FragmentType>(gather.getSource().getType());
    if (!fragment)
      return;
    for (int64_t axis : gather.getSourceAxes()) {
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      auto replay = PhysicalProgramAnalysis(kernel).replayability(
          gather.getSource(), sourceAxisIdentity(mapping),
          PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true);
      if (llvm::any_of(replay.blockers, [](Operation *operation) {
            return isa<scf::ForOp>(operation);
          }) && !llvm::is_contained(retainedGatherAxes,
                                    std::pair<Value, unsigned>{gather.getSource(), axis}))
        retainedGatherAxes.emplace_back(gather.getSource(), axis);
    }
  });
  llvm::SmallPtrSet<Operation *, 16> retainedGatherRanges;
  for (auto [source, axis] : retainedGatherAxes) {
    if (failed(requireStructuredReductionFullCoverage(kernel, source, axis)))
      return emitError(source.getLoc(), "ordered tensor gather has no full source-axis realization");
    auto roots = PhysicalProgramAnalysis(kernel).axisRanges(source, axis);
    for (MakeRangeOp range : allRanges)
      if (llvm::any_of(roots.roots, [&](MakeRangeOp root) {
            return sameLogicalRange(root, range) &&
                   samePhysicalScalarExpression(root.getStart(), range.getStart());
          })) {
        // A surviving gather consumes an immutable ordered carry, including
        // members outside an output tile. Retain that source axis; unrelated
        // free axes still participate in ordinary ownership selection.
        internalTraversalRanges.insert(range.getOperation());
        retainedGatherRanges.insert(range.getOperation());
      }
  }
  SmallVector<MakeRangeOp> fullCoverageRanges;
  // Contraction blocking owns the runtime row loop and tail validity for
  // device-derived subregions. They are not pointwise full-coverage values.
  if (!ownershipOnly)
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return internalTraversalRanges.contains(range.getOperation()) &&
             contractionTraversalRanges.contains(range.getOperation()) &&
             hasAccessDependentSubregionBounds(kernel, range);
    });
  if (!ownershipOnly)
    for (MakeRangeOp range : dynamicRanges) {
      if (retainedGatherRanges.contains(range.getOperation())) {
        fullCoverageRanges.push_back(range);
        continue;
      }
      if (!internalTraversalRanges.contains(range.getOperation()) ||
          (structuredTraversalRanges.contains(range.getOperation()) &&
           !writeTraversalRanges.contains(range.getOperation())) ||
          reductionFreeRanges.contains(range.getOperation()) ||
          reuseTraversalRanges.contains(range.getOperation()))
        continue;
      if (failed(requireFullDimensionCoverage(kernel, range.getResult(), 0)))
        return range.emitOpError(
                   "internal pointwise traversal has no exact full-coverage realization")
               << "; source_id=" << range.getSourceId()
               << ", fragment=" << range.getResult().getType();
      fullCoverageRanges.push_back(range);
    }
  llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
    return llvm::is_contained(fullCoverageRanges, range) ||
           (!ownershipOnly &&
            hasExactStaticFullCoverage(kernel, range.getResult(), 0));
  });
  for (MakeRangeOp range : dynamicRanges) {
    if (ownershipOnly &&
        internalTraversalRanges.contains(range.getOperation())) {
      FailureOr<ParameterOp> internalParameter = queryBlockingParameter(kernel, range);
      FailureOr<Attribute> internalAxis =
          succeeded(internalParameter)
              ? parameterAxis(*internalParameter)
              : FailureOr<Attribute>(failure());
      if (succeeded(internalAxis)) {
        internalAxes.insert(*internalAxis);
      } else if (FailureOr<uint64_t> dimension = rangeDimension(range);
                 succeeded(dimension))
        internalAxes.insert(dimensionAxisKey(module.getContext(), *dimension));
      continue;
    }
    if (!ownershipOnly &&
        internalTraversalRanges.contains(range.getOperation()) &&
        !reuseTraversalRanges.contains(range.getOperation()))
      continue;
    if (ownershipOnly && !hasPointwiseOwnership(range))
      continue;
    if (!ownershipOnly &&
        !hasPointwiseOwnership(range) &&
        !internalTraversalRanges.contains(range.getOperation()) &&
        !reuseTraversalRanges.contains(range.getOperation()))
      continue;
    FailureOr<ParameterOp> parameter =
        queryOwnershipBlockingParameter(kernel, range);
    MakeRangeOp occurrenceRoot = occurrenceRoots.lookup(range.getOperation());
    if (occurrenceRoot) {
      ParameterOp selected = occurrenceParameters.lookup(occurrenceRoot.getOperation());
      if (!selected) {
        FailureOr<ParameterOp> existing =
            queryOwnershipBlockingParameter(kernel, occurrenceRoot);
        if (succeeded(existing)) {
          PhysicalParameterBinding binding = queryParameterBinding(*existing);
          // Repeated Cartesian axes have distinct occurrence classes even
          // when their logical extents share the same dimension identity.
          auto dimension = ownershipDimension(kernel, occurrenceRoot);
          bool sharedDimension = binding.isExact() && !binding.source &&
              binding.dimension && succeeded(dimension) &&
              *binding.dimension == static_cast<int64_t>(*dimension) &&
              !independentCartesianDimensions.contains(*dimension) &&
              positionalOccurrences.contains(range.getOperation()) &&
              positionalOccurrences.contains(occurrenceRoot.getOperation());
          if (sharedDimension ||
              (binding.isExact() && binding.source &&
               *binding.source == sourceAxisIdentity(occurrenceRoot)))
            selected = *existing;
        }
      }
      parameter = selected ? FailureOr<ParameterOp>(selected)
                           : FailureOr<ParameterOp>(failure());
    }
    bool requiresBlockingParameter =
        (ownershipOnly && hasPointwiseOwnership(range) &&
         !internalTraversalRanges.contains(range.getOperation())) ||
        (!ownershipOnly &&
         ((hasPointwiseOwnership(range) &&
           !internalTraversalRanges.contains(range.getOperation())) ||
          reuseTraversalRanges.contains(range.getOperation())));
    if (requiresBlockingParameter && succeeded(parameter) &&
        (parameter->getParameter().getCategory() ==
             static_cast<uint32_t>(ParameterCategory::Coverage) ||
         (*parameter)->hasAttr(coverageDimensionAttr)))
      parameter = failure();
    if (failed(parameter) && requiresBlockingParameter) {
      const bool worksetRange = range->hasAttr(worksetCoordinateRangeAttr);
      Value logicalExtentValue = range.getExtent();
      if (worksetRange) {
        auto coordinate = range.getStart().getDefiningOp<WorksetCoordinateOp>();
        auto position = coordinate
                            ? coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr)
                            : IntegerAttr();
        if (!position || position.getInt() < 0 ||
            static_cast<size_t>(position.getInt()) >= mapping.getExtents().size())
          return range.emitOpError(
              "workset ownership has no logical cardinality for blocking");
        logicalExtentValue = mapping.getExtents()[position.getInt()];
      }
      auto logicalExtent = dyn_cast_or_null<IntegerAttr>(
          UniformValueAnalysis(describeUniformValue).evaluate(logicalExtentValue));
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      const bool dynamicSubregion = range->hasAttr(sourceSubregionAttr);
      FailureOr<uint64_t> sourceDimension =
          ownershipDimension(kernel, range);
      const bool launchVisibleDimension =
          succeeded(sourceDimension) &&
          succeeded(dimensionArgument(kernel, *sourceDimension));
      PhysicalExprAttr derivedExtent = launchRangeExtent(range);
      const bool launchVisibleExtent =
          worksetRange ? !logicalExtent
                       : launchVisibleDimension ||
                             (derivedExtent && !isCompileTimePhysicalExpr(derivedExtent));
      ParameterCategory category = ParameterCategory::Pointwise;
      if (succeeded(sourceDimension)) {
        auto structured = structuredOwnershipCategories.find(*sourceDimension);
        if (structured != structuredOwnershipCategories.end())
          category = structured->second;
      }
      if (category == ParameterCategory::Pointwise &&
          contractFreeAxisFacts(kernel, range).regionContraction)
        category = ParameterCategory::RegionContraction;
      if (dynamicSubregion ||
          launchVisibleExtent ||
          (logicalExtent && physicalExtent.getKind() ==
                                static_cast<uint32_t>(PhysicalExprKind::Constant))) {
        SmallVector<int64_t> candidates;
        if (worksetRange ? logicalExtent && logicalExtent.getInt() == 1
                         : isProvablySingletonLogicalRange(range)) {
          candidates.push_back(1);
        } else if (dynamicSubregion) {
          candidates.assign({1, 2, 4, 8, 16, 32, 64, 128, 256});
        } else if (launchVisibleExtent) {
          candidates.assign({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                             2048, 4096});
        } else {
          // Ownership tail validity also permits a padded pointwise tile.
          // Keep the enclosing power-of-two extent available to the profiles;
          // a logical row width must not cap an otherwise legal physical tile.
          uint64_t paddedExtent = llvm::PowerOf2Ceil(
              static_cast<uint64_t>(std::max<int64_t>(logicalExtent.getInt(), 1)));
          for (int64_t candidate : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                                    2048, 4096, 8192, 16384})
            if (category == ParameterCategory::RegionContraction ||
                static_cast<uint64_t>(candidate) <= paddedExtent)
              candidates.push_back(candidate);
          if (logicalExtent.getInt() > 0 &&
              !llvm::is_contained(candidates, logicalExtent.getInt()))
            candidates.push_back(logicalExtent.getInt());
          llvm::sort(candidates);
        }
        OpBuilder builder(&kernel.getBody().front(),
                          kernel.getBody().front().begin());
        PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
        if (occurrenceRoot)
          source = sourceAxisIdentity(occurrenceRoot);
        auto schema = ParameterAttr::get(
            module.getContext(),
            builder.getStringAttr(launchVisibleDimension && !occurrenceRoot
                                      ? ("FRAGMENT_D" +
                                         Twine(*sourceDimension))
                                            .str()
                                      : ("FRAGMENT_S" + Twine(source.sourceId) +
                                         "_A" + Twine(source.sourceAxis))
                                            .str()),
            static_cast<uint32_t>(ParameterRole::OwnershipN),
            static_cast<uint32_t>(category),
            pointwiseElementBitWidth,
            DenseI64ArrayAttr::get(module.getContext(), candidates));
        parameter = builder.create<ParameterOp>(
            range.getLoc(), builder.getIndexType(), schema);
        if (succeeded(sourceDimension))
          (*parameter)->setAttr(dimensionAttr,
                                builder.getI64IntegerAttr(*sourceDimension));
        if (!launchVisibleDimension || occurrenceRoot)
          (*parameter)->setAttr(
              parameterSourceAttr,
              PhysicalSourceAttr::get(module.getContext(), source.sourceId,
                                      source.sourceAxis, source.derived));
      }
    }
    FailureOr<Attribute> axis =
        failed(parameter) ? FailureOr<Attribute>(failure())
                          : parameterAxis(*parameter);
    if (failed(parameter) || failed(axis)) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no canonical blocking dimension");
      diagnostic << "; source_id=" << range.getSourceId() << ", fragment="
                 << range.getResult().getType();
      if (succeeded(parameter))
        diagnostic << ", parameter="
                   << parameter->getParameter().getName().getValue();
      return failure();
    }
    if (occurrenceRoot)
      occurrenceParameters[occurrenceRoot.getOperation()] = *parameter;
    // Repeated Cartesian axes use their proven occurrence classes. A unique
    // logical dimension can still bind connected pointwise values, while a
    // range-local source retains its own traversal relation.
    if (occurrenceRoot)
      retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                           fragmentExtent(*parameter));
    else if (FailureOr<uint64_t> dimension = rangeDimension(range);
        succeeded(dimension))
      retargetDimensionExtent(range.getResult(), *dimension,
                              fragmentExtent(*parameter));
    else if (FailureOr<PhysicalSourceAxis> source = axisSource(*axis);
             succeeded(source))
      retargetSourceExtent(range.getResult(), *source,
                           fragmentExtent(*parameter));
    else
      return range.emitOpError("blocking parameter has no typed axis binding");
    auto found = parameters.find(*axis);
    if (found != parameters.end() && found->second != *parameter)
      return range.emitOpError("one physical axis has multiple blocking parameters")
             << "; axis=" << *axis << ", previous="
             << found->second.getParameter().getName().getValue()
             << ", current=" << parameter->getParameter().getName().getValue()
             << ", source_id=" << range.getSourceId();
    parameters[*axis] = *parameter;
    axes[*axis].push_back(range);
    if (internalTraversalRanges.contains(range.getOperation()))
      internalAxes.insert(*axis);
    if (hasPointwiseOwnership(range) &&
        !internalTraversalRanges.contains(range.getOperation()))
      ownershipAxes.insert(*axis);
  }
  if (!ownershipOnly)
    ownershipAxes.clear();

  if (ownershipOnly) {
    llvm::SmallDenseSet<Attribute> internalOwnershipAxes;
    for (Attribute axis : ownershipAxes) {
      if (llvm::any_of(axes.lookup(axis), [](MakeRangeOp range) {
            return !range->hasAttr(worksetCoordinateRangeAttr);
          }))
        internalOwnershipAxes.insert(axis);
    }
    if (!internalOwnershipAxes.empty()) {
      SmallVector<std::pair<int64_t, Attribute>> scalarWorksetAxes;
      for (Attribute axis : ownershipAxes) {
        if (isSourceAxisKey(axis) || internalOwnershipAxes.contains(axis))
          continue;
        std::optional<int64_t> worksetAxis;
        bool exact = llvm::all_of(axes.lookup(axis), [&](MakeRangeOp range) {
          if (!range->hasAttr(worksetCoordinateRangeAttr))
            return false;
          auto coordinate =
              range.getStart().getDefiningOp<WorksetCoordinateOp>();
          auto position = coordinate ? coordinate->getAttrOfType<IntegerAttr>(
                                           worksetAxisAttr)
                                     : IntegerAttr();
          if (!position || position.getInt() < 0 ||
              (worksetAxis && *worksetAxis != position.getInt()))
            return false;
          worksetAxis = position.getInt();
          return true;
        });
        if (exact && worksetAxis)
          scalarWorksetAxes.emplace_back(*worksetAxis, axis);
      }
      llvm::stable_sort(scalarWorksetAxes,
                        [](const auto &lhs, const auto &rhs) {
                          return lhs.first < rhs.first;
                        });
      llvm::SmallDenseSet<Attribute> structuredWorksetAxes;
      for (auto [_, axis] : scalarWorksetAxes)
        if (llvm::any_of(axes.lookup(axis), [&](MakeRangeOp range) {
              return contractFreeAxisSides(kernel, range) !=
                     ContractFreeAxisNone;
            }))
          structuredWorksetAxes.insert(axis);

      llvm::DenseMap<Attribute, int64_t> exactLocalExtents;
      bool exactLocalCoverage = llvm::all_of(
          internalOwnershipAxes, [&](Attribute axis) {
            std::optional<int64_t> extent;
            if (llvm::any_of(axes.lookup(axis), [&](MakeRangeOp range) {
                  if (range->hasAttr(worksetCoordinateRangeAttr) ||
                      range->hasAttr(sourceSubregionAttr))
                    return true;
                  std::optional<int64_t> current =
                      constantLogicalRangeCardinality(range);
                  if (!current || (extent && *extent != *current))
                    return true;
                  extent = *current;
                  return false;
                }) ||
                !extent)
              return false;
            ParameterOp parameter = parameters.lookup(axis);
            if (!parameter ||
                !llvm::is_contained(
                    parameter.getParameter().getCandidates().asArrayRef(),
                    *extent))
              return false;
            exactLocalExtents[axis] = *extent;
            return true;
          });

      if ((scalarWorksetAxes.size() >= 2 ||
           !structuredWorksetAxes.empty()) &&
          exactLocalCoverage) {
        // The existing non-workset ranges are exact program-local vectors.  Fix
        // them to full coverage and spend ownership dimensions on either the
        // two innermost Cartesian workset axes or an explicitly proven
        // contraction-free workset axis.  This changes both the fragment schema
        // and the launch mapping; no provider serializer inference is involved.
        for (Attribute axis : internalOwnershipAxes) {
          ParameterOp parameter = parameters.lookup(axis);
          ParameterAttr schema = parameter.getParameter();
          parameter->setAttr(
              "parameter",
              ParameterAttr::get(
                  module.getContext(), schema.getName(), schema.getRole(),
                  schema.getCategory(), schema.getElementBitWidth(),
                  DenseI64ArrayAttr::get(module.getContext(),
                                         {exactLocalExtents.lookup(axis)})));
          parameter->setAttr(pointwiseLocalAttr,
                             UnitAttr::get(module.getContext()));
          ownershipAxes.erase(axis);
          internalAxes.insert(axis);
        }
        for (auto [index, entry] : llvm::enumerate(scalarWorksetAxes))
          if (index + 2 < scalarWorksetAxes.size() &&
              !structuredWorksetAxes.contains(entry.second))
            ownershipAxes.erase(entry.second);
      } else {
        for (auto [_, axis] : scalarWorksetAxes)
          ownershipAxes.erase(axis);
      }
    }
  }

  if (ownershipOnly) {
    auto unitExtent = expression(module.getContext(), PhysicalExprKind::Constant, 1);
    for (auto [axis, ranges] : axes) {
      if (ownershipAxes.contains(axis))
        continue;
      for (MakeRangeOp range : ranges) {
        if (!range->hasAttr(worksetCoordinateRangeAttr) ||
            internalTraversalRanges.contains(range.getOperation()))
          continue;
        // An unselected workset coordinate remains a scalar program-grid
        // coordinate.  KIR construction provisionally gives every dynamic ABI
        // dimension a fragment parameter; once ownership selects the actual
        // lane axes, that provisional parameter must not remain in live value
        // types or the provider tuning surface.
        retargetSourceExtent(
            range.getResult(),
            PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()},
            unitExtent);
      }
    }
  }

  SmallVector<Attribute> pointwiseOwnershipAxes;
  for (auto [axis, ranges] : axes) {
    if (!ownershipAxes.contains(axis))
      continue;
    ParameterOp parameter = parameters.lookup(axis);
    if (!parameter ||
        parameter.getParameter().getRole() ==
            static_cast<uint32_t>(ParameterRole::ScanChunk))
      continue;
    pointwiseOwnershipAxes.push_back(axis);
  }

  bool orderedByStore = false;
  kernel.walk([&](StoreOp store) {
    if (orderedByStore)
      return;
    llvm::DenseMap<Attribute, int64_t> outputAxes;
    llvm::SmallDenseSet<int64_t> usedOutputAxes;
    for (Attribute axis : pointwiseOwnershipAxes) {
      std::optional<int64_t> outputAxis;
      for (MakeRangeOp range : axes.lookup(axis)) {
        std::optional<int64_t> projection = storeAxisForRange(store, range);
        if (!projection)
          continue;
        if (outputAxis && outputAxis != projection)
          return;
        outputAxis = projection;
      }
      if (!outputAxis || !usedOutputAxes.insert(*outputAxis).second)
        return;
      outputAxes[axis] = *outputAxis;
    }
    llvm::stable_sort(pointwiseOwnershipAxes, [&](Attribute lhs, Attribute rhs) {
      return outputAxes.lookup(lhs) < outputAxes.lookup(rhs);
    });
    orderedByStore = true;
    if (pointwiseOwnershipAxes.size() <= 2)
      return;
    Value copied = store.getValue();
    while (true) {
      if (auto transpose = copied.getDefiningOp<TransposeOp>()) {
        copied = transpose.getValue();
        continue;
      }
      if (auto broadcast = copied.getDefiningOp<BroadcastOp>()) {
        auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
        auto target = cast<FragmentType>(broadcast.getResult().getType());
        if (source && source.getShape().size() == target.getShape().size() &&
            queryAxisProjection(source, target).isExact()) {
          copied = broadcast.getValue();
          continue;
        }
      }
      break;
    }
    auto load = copied.getDefiningOp<LoadOp>();
    auto input = load ? dyn_cast<ViewType>(load.getResource().getType()) : ViewType();
    if (!input)
      return;
    Attribute inputAxis;
    for (auto [position, coordinate] : llvm::enumerate(load.getCoordinates())) {
      if (load.getSourceAxes()[position] + 1 != input.getRank())
        continue;
      auto range =
          stripAdditiveProjection(coordinate).getDefiningOp<MakeRangeOp>();
      if (!range || !isUnitStepRange(range) ||
          !queryFragmentAxis(coordinate.getType(), sourceAxisIdentity(range))
               .isExact())
        return;
      for (Attribute axis : pointwiseOwnershipAxes)
        if (llvm::is_contained(axes.lookup(axis), range)) {
          if (inputAxis && inputAxis != axis)
            return;
          inputAxis = axis;
        }
    }
    if (!inputAxis || inputAxis == pointwiseOwnershipAxes.back())
      return;
    ParameterOp inputParameter = parameters.lookup(inputAxis);
    if (!inputParameter || llvm::all_of(
            inputParameter.getParameter().getCandidates().asArrayRef(),
            [](int64_t extent) { return extent == 1; }))
      return;
    bool commonOutputDirection = true;
    kernel.walk([&](StoreOp other) {
      auto output = dyn_cast<ViewType>(other.getResource().getType());
      commonOutputDirection &= output && other.getValue() == store.getValue() &&
          llvm::any_of(axes.lookup(pointwiseOwnershipAxes.back()),
                       [&](MakeRangeOp range) {
                         return storeAxisForRange(other, range) == output.getRank() - 1;
                       });
    });
    if (!commonOutputDirection)
      return;
    // Tile both ends of a permutation for row-major source/destination
    // locality. The access graph still carries the actual view strides.
    pointwiseOwnershipAxes.erase(llvm::find(pointwiseOwnershipAxes, inputAxis));
    pointwiseOwnershipAxes.insert(pointwiseOwnershipAxes.end() - 1, inputAxis);
  });

  bool pointwiseOnlyProgram = true;
  kernel.walk([&](Operation *operation) {
    pointwiseOnlyProgram &=
        !isa<ContractOp, ReduceOp, ScanOp, RegionFoldOp, RegionScanOp,
             ScaledContractOp, SparseContractOp, HistogramOp, ScatterReduceOp>(
            operation);
  });
  if (!pointwiseOnlyProgram && orderedByStore &&
      pointwiseOwnershipAxes.size() > 2 &&
      llvm::all_of(pointwiseOwnershipAxes, [&](Attribute axis) {
        return llvm::all_of(axes.lookup(axis), [&](MakeRangeOp range) {
          return contractFreeAxisSides(kernel, range) == ContractFreeAxisNone;
        });
      })) {
    PhysicalProgramAnalysis analysis(kernel);
    llvm::DenseMap<Attribute, uint64_t> invariantReadVolume;
    bool knownReads = true;
    auto ownerOfRange = [&](MakeRangeOp range) -> Attribute {
      MakeRangeOp occurrence = occurrenceRoots.lookup(range.getOperation());
      for (Attribute axis : pointwiseOwnershipAxes)
        if (llvm::any_of(axes.lookup(axis), [&](MakeRangeOp candidate) {
              return candidate == range ||
                     (occurrence && occurrenceRoots.lookup(
                                        candidate.getOperation()) == occurrence);
            }))
          return axis;
      return {};
    };
    kernel.walk([&](LoadOp load) {
      if (!knownReads)
        return;
      SmallVector<MakeRangeOp> roots;
      SmallVector<Value> dependencies(load.getCoordinates());
      if (load.getValid())
        dependencies.push_back(load.getValid());
      if (load.getFill())
        dependencies.push_back(load.getFill());
      for (Value value : dependencies) {
        auto fact = analysis.sourceRanges(value);
        if (!fact.blockers.empty() || !fact.accesses.empty()) {
          knownReads = false;
          return;
        }
        for (MakeRangeOp root : fact.roots)
          if (!llvm::is_contained(roots, root))
            roots.push_back(root);
      }
      uint64_t volume = physicalElementBitWidth(load.getResult().getType());
      llvm::SmallDenseSet<Attribute> countedOwners;
      llvm::SmallDenseSet<Attribute> varyingAxes;
      for (MakeRangeOp root : roots) {
        Attribute owner = ownerOfRange(root);
        if (owner)
          varyingAxes.insert(owner);
        // Estimate a fragment with the current inner output axis free and the
        // other program-grid axes fixed. Local/reduction ranges remain live.
        if (owner && owner != pointwiseOwnershipAxes.back())
          continue;
        if (owner && !countedOwners.insert(owner).second)
          continue;
        auto extent = constantLogicalRangeCardinality(root);
        if (!extent || *extent <= 0 ||
            volume > std::numeric_limits<uint64_t>::max() / *extent) {
          knownReads = false;
          return;
        }
        volume *= *extent;
      }
      for (Attribute axis : pointwiseOwnershipAxes) {
        if (varyingAxes.contains(axis))
          continue;
        uint64_t &score = invariantReadVolume[axis];
        if (score > std::numeric_limits<uint64_t>::max() - volume) {
          knownReads = false;
          return;
        }
        score += volume;
      }
    });
    Attribute selected = pointwiseOwnershipAxes[pointwiseOwnershipAxes.size() - 2];
    for (Attribute axis : llvm::drop_end(pointwiseOwnershipAxes)) {
      ParameterOp parameter = parameters.lookup(axis);
      if (knownReads &&
          llvm::any_of(parameter.getParameter().getCandidates().asArrayRef(),
                       [](int64_t extent) { return extent > 1; }) &&
          invariantReadVolume.lookup(axis) > invariantReadVolume.lookup(selected))
        selected = axis;
    }
    // Keep the current output direction and tile the independent axis that
    // reuses the largest input fragment. This estimates repeated reads, not
    // pointer contiguity; the access graph retains the actual ABI strides.
    pointwiseOwnershipAxes.erase(llvm::find(pointwiseOwnershipAxes, selected));
    pointwiseOwnershipAxes.insert(pointwiseOwnershipAxes.end() - 1, selected);
  }

  llvm::DenseMap<Attribute, CoordinateRole> contractionCoordinateRoles;
  for (auto [ownershipIndex, axis] :
       llvm::enumerate(pointwiseOwnershipAxes)) {
    ParameterOp parameter = parameters.lookup(axis);
    unsigned contractSides = ContractFreeAxisNone;
    unsigned matrixSides = ContractFreeAxisNone;
    bool batchedContraction = false;
    unsigned contractElementBitWidth = 0;
    for (MakeRangeOp range : axes.lookup(axis)) {
      ContractFreeAxisFacts facts = contractFreeAxisFacts(kernel, range);
      contractSides |= facts.sides;
      matrixSides |= facts.matrixSides;
      batchedContraction |= facts.batchedContraction;
      contractElementBitWidth =
          std::max(contractElementBitWidth, facts.operandElementBitWidth);
    }
    const bool contractionAxis = contractSides != ContractFreeAxisNone;
    const bool scalarGridAxis = !pointwiseOnlyProgram && !contractionAxis &&
        ownershipIndex + 2 < pointwiseOwnershipAxes.size();
    ParameterRole ownershipRole = ParameterRole::OwnershipN;
    auto declaredRole =
        static_cast<ParameterRole>(parameter.getParameter().getRole());
    // Matrix operands constrain M/N orientation. A vector contraction with
    // other output axes uses their existing store order, so independent axes
    // do not accidentally consume the same profile column.
    unsigned orientedSides = pointwiseOwnershipAxes.size() == 1
                                 ? contractSides
                                 : matrixSides;
    if (orientedSides == ContractFreeAxisLhs)
      ownershipRole = ParameterRole::OwnershipM;
    else if (orientedSides == ContractFreeAxisRhs)
      ownershipRole = ParameterRole::OwnershipN;
    else if (parameter.getParameter().getCategory() ==
            static_cast<uint32_t>(ParameterCategory::Contraction) &&
        (declaredRole == ParameterRole::OwnershipM ||
         declaredRole == ParameterRole::OwnershipN))
      ownershipRole = declaredRole;
    else if (!scalarGridAxis &&
             ownershipIndex + 1 < pointwiseOwnershipAxes.size())
      ownershipRole = ParameterRole::OwnershipM;
    parameter->removeAttr(coverageDimensionAttr);
    parameter->removeAttr(coverageBoundAttr);
    DenseI64ArrayAttr candidates;
    if (scalarGridAxis)
      candidates = DenseI64ArrayAttr::get(module.getContext(), {1});
    else if (isSourceAxisKey(axis))
      candidates = parameter.getParameter().getCandidates();
    else
      candidates = DenseI64ArrayAttr::get(
          module.getContext(),
          {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
           8192, 16384, 32768, 65536});
    uint32_t category = parameter.getParameter().getCategory();
    if (!scalarGridAxis && batchedContraction &&
        (contractSides == ContractFreeAxisLhs ||
         contractSides == ContractFreeAxisRhs) &&
        category == static_cast<uint32_t>(ParameterCategory::Pointwise))
      category = static_cast<uint32_t>(ParameterCategory::Contraction);
    auto schema = ParameterAttr::get(
        module.getContext(), parameter.getParameter().getName(),
        static_cast<uint32_t>(ownershipRole), category,
        category == static_cast<uint32_t>(ParameterCategory::Contraction) &&
                contractElementBitWidth != 0
            ? contractElementBitWidth
            : pointwiseElementBitWidth,
        candidates);
    parameter->setAttr("parameter", schema);
    if (category == static_cast<uint32_t>(ParameterCategory::Contraction))
      contractionCoordinateRoles[axis] =
          ownershipRole == ParameterRole::OwnershipM
              ? CoordinateRole::ContractionM
              : CoordinateRole::ContractionN;
  }
  if (llvm::count_if(contractionCoordinateRoles, [](const auto &entry) {
        return entry.second == CoordinateRole::ContractionM;
      }) != 1 ||
      llvm::count_if(contractionCoordinateRoles, [](const auto &entry) {
        return entry.second == CoordinateRole::ContractionN;
      }) != 1)
    contractionCoordinateRoles.clear();

  if (ownershipOnly) {
    llvm::MapVector<Attribute, SmallVector<MakeRangeOp>> selectedAxes;
    SmallVector<MakeRangeOp> selectedRanges;
    for (auto [axis, ranges] : axes) {
      if (!ownershipAxes.contains(axis))
        continue;
      for (MakeRangeOp range : ranges) {
        if (internalTraversalRanges.contains(range.getOperation()))
          continue;
        selectedAxes[axis].push_back(range);
        selectedRanges.push_back(range);
      }
    }
    axes = std::move(selectedAxes);
    dynamicRanges = std::move(selectedRanges);
    if (dynamicRanges.empty()) {
      if (failed(alignHistogramOutputOwnership(kernel)))
        return failure();
      return finalizeValueRelations();
    }
  }

  OpBuilder mappingBuilder(mapping);
  SmallVector<Value> runtimeExtents(mapping.getExtents());
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  SmallVector<Type> coordinateTypes(mapping.getResultTypes());
  llvm::DenseMap<Attribute, Value> tileCoordinates;
  llvm::DenseMap<Attribute, Value> logicalDimensions;
  llvm::DenseMap<Attribute, unsigned> mappedAxes;
  llvm::DenseMap<unsigned, Attribute> coordinateAxes;
  auto mappingAxis = [&](Attribute attribute) -> FailureOr<Attribute> {
    auto expression = dyn_cast<PhysicalExprAttr>(attribute);
    if (expression && expression.getKind() ==
                          static_cast<uint32_t>(PhysicalExprKind::CeilDiv) &&
        expression.getOperands().size() == 2) {
      auto divisor = dyn_cast<PhysicalExprAttr>(expression.getOperands()[1]);
      if (divisor && divisor.getKind() ==
                         static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
        FailureOr<ParameterOp> parameter = queryParameterBySymbol(kernel, divisor.getSymbol());
        if (succeeded(parameter))
          return parameterAxis(*parameter);
      }
    }
    FailureOr<uint64_t> blocked = blockedDimension(attribute);
    if (succeeded(blocked))
      return dimensionAxisKey(module.getContext(), *blocked);
    auto physical = dyn_cast<PhysicalExprAttr>(attribute);
    if (!physical ||
        physical.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Dimension))
      return failure();
    return physical.getValue() > 0
               ? FailureOr<Attribute>(dimensionAxisKey(module.getContext(),
                                                       physical.getValue()))
               : FailureOr<Attribute>(failure());
  };
  std::optional<unsigned> reusableUnitAxis;
  for (auto [axis, extent] : llvm::enumerate(mapping.getLaunchExtents())) {
    FailureOr<Attribute> mapped = mappingAxis(extent);
    if (succeeded(mapped) && axis < mapping.getCoordinates().size()) {
      auto existing = mappedAxes.find(*mapped);
      if (existing != mappedAxes.end() && existing->second != axis)
        return kernel.emitError(
            "one pointwise ownership axis maps to multiple program coordinates");
      tileCoordinates[*mapped] = mapping.getCoordinates()[axis];
      mappedAxes[*mapped] = axis;
      coordinateAxes[axis] = *mapped;
    }
    auto physical = dyn_cast<PhysicalExprAttr>(extent);
    if (!reusableUnitAxis && axis < mapping.getCoordinates().size() &&
        mapping.getCoordinates()[axis].use_empty() && physical &&
        physical.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        physical.getValue() == 1)
      reusableUnitAxis = axis;
  }
  bool mappingChanged = false;
  SmallVector<std::pair<Attribute, unsigned>> reusedCoordinates;
  SmallVector<Attribute> appendedCoordinates;
  for (auto [axisKey, ranges] : axes) {
    MakeRangeOp range = ranges.front();
    const bool worksetRange = range->hasAttr(worksetCoordinateRangeAttr);
    std::optional<unsigned> worksetPosition;
    if (worksetRange) {
      auto coordinate = range.getStart().getDefiningOp<WorksetCoordinateOp>();
      auto worksetAxis =
          coordinate
              ? coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr)
              : IntegerAttr();
      if (!worksetAxis || worksetAxis.getInt() < 0 ||
          static_cast<size_t>(worksetAxis.getInt()) >=
              mapping.getCoordinates().size())
        return range.emitOpError(
            "workset ownership has no corresponding program coordinate");
      unsigned position = static_cast<unsigned>(worksetAxis.getInt());
      worksetPosition = position;
      auto existingCoordinate = coordinateAxes.find(position);
      if (existingCoordinate != coordinateAxes.end() &&
          existingCoordinate->second != axisKey)
        return range.emitOpError(
            "workset ownership conflicts with the existing program coordinate relation");
      auto existingAxis = mappedAxes.find(axisKey);
      if (existingAxis != mappedAxes.end() && existingAxis->second != position)
        return range.emitOpError(
            "workset ownership axis maps to multiple program coordinates");
      mappedAxes[axisKey] = position;
      coordinateAxes[position] = axisKey;
    }
    ParameterOp parameter = parameters.lookup(axisKey);
    Value dimension;
    PhysicalExprAttr derivedLogicalExtent;
    std::optional<int64_t> staticExtent;
    FailureOr<uint64_t> sourceDimension = ownershipDimension(kernel, range);
    if (worksetPosition) {
      // A workset coordinate is expressed in source-coordinate units, while
      // ownership tiles the finite workset instance domain.  Reuse the
      // Delinearize runtime cardinality here; its launch expression below is
      // the exact same relation and may include non-zero starts or non-unit
      // domain steps.
      dimension = mapping.getExtents()[*worksetPosition];
    } else if (isSourceAxisKey(axisKey)) {
      staticExtent = constantLogicalRangeCardinality(range);
      if (staticExtent)
        dimension = mappingBuilder.create<arith::ConstantIndexOp>(
            mapping.getLoc(), *staticExtent);
      else if (succeeded(sourceDimension)) {
        FailureOr<Value> argument =
            dimensionArgument(kernel, *sourceDimension);
        if (succeeded(argument))
          dimension = *argument;
      }
    } else {
      FailureOr<uint64_t> dimensionId = axisDimension(axisKey);
      if (succeeded(dimensionId)) {
        FailureOr<Value> argument = dimensionArgument(kernel, *dimensionId);
        if (succeeded(argument))
          dimension = *argument;
      }
    }
    if (!parameter) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no launch-visible dimension or blocking parameter");
      diagnostic << "; axis=" << axisKey << ", fragment="
                 << range.getResult().getType();
      return failure();
    }
    if (!ownershipAxes.contains(axisKey))
      continue;
    if (!dimension) {
      for (MakeRangeOp other : ranges) {
        auto extent = launchRangeExtent(other);
        if (!extent) {
          derivedLogicalExtent = {};
          break;
        }
        if (!derivedLogicalExtent)
          derivedLogicalExtent = extent;
        else if (derivedLogicalExtent != extent)
          derivedLogicalExtent = binaryExpression(
              module.getContext(), PhysicalExprKind::Maximum,
              derivedLogicalExtent, extent);
      }
      if (derivedLogicalExtent)
        dimension = mappingBuilder.create<PhysicalExprOp>(
            mapping.getLoc(), mappingBuilder.getIndexType(),
            derivedLogicalExtent);
    }
    if (!dimension) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic ownership range has no launch-visible logical dimension");
      diagnostic << "; axis=" << axisKey << ", fragment="
                 << range.getResult().getType() << ", parameter="
                 << parameter.getParameter().getName().getValue();
      return failure();
    }
    logicalDimensions[axisKey] = dimension;
    Value one = mappingBuilder.create<arith::ConstantIndexOp>(mapping.getLoc(), 1);
    Value adjusted = mappingBuilder.create<BinaryOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), dimension,
        mappingBuilder.create<BinaryOp>(mapping.getLoc(),
                                        mappingBuilder.getIndexType(),
                                        parameter.getResult(), one,
                                        BinaryOperator::Subtract),
        BinaryOperator::Add);
    Value tiles = mappingBuilder.create<BinaryOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), adjusted,
        parameter.getResult(), BinaryOperator::FloorDivide);
    PhysicalExprAttr logical;
    if (worksetPosition) {
      logical = cast<PhysicalExprAttr>(
          mapping.getLaunchExtents()[*worksetPosition]);
    } else if (derivedLogicalExtent) {
      logical = derivedLogicalExtent;
    } else if (isSourceAxisKey(axisKey) && staticExtent) {
      logical = expression(module.getContext(), PhysicalExprKind::Constant,
                           *staticExtent);
    } else {
      FailureOr<uint64_t> dimensionId = axisDimension(axisKey);
      uint64_t logicalDimension =
          isSourceAxisKey(axisKey) && succeeded(sourceDimension)
              ? *sourceDimension
              : succeeded(dimensionId) ? *dimensionId : 0;
      if (logicalDimension == 0)
        return range.emitOpError(
            "range-local ownership axis lost its logical dimension extent");
      logical = expression(module.getContext(), PhysicalExprKind::Dimension,
                           logicalDimension,
                           ("D" + Twine(logicalDimension)).str());
    }
    PhysicalExprAttr tile = expression(
        module.getContext(), PhysicalExprKind::Parameter, 0,
        parameter.getParameter().getName().getValue());
    PhysicalExprAttr launch = binaryExpression(
        module.getContext(), PhysicalExprKind::CeilDiv, logical, tile);
    auto mapped = mappedAxes.find(axisKey);
    if (mapped != mappedAxes.end() || reusableUnitAxis) {
      unsigned axis = mapped != mappedAxes.end() ? mapped->second
                                                 : *reusableUnitAxis;
      runtimeExtents[axis] = tiles;
      launchExtents[axis] = launch;
      reusedCoordinates.emplace_back(axisKey, axis);
      if (mapped == mappedAxes.end())
        reusableUnitAxis.reset();
    } else {
      runtimeExtents.push_back(tiles);
      coordinateTypes.push_back(mappingBuilder.getIndexType());
      launchExtents.push_back(launch);
      appendedCoordinates.push_back(axisKey);
    }
    mappingChanged = true;
  }

  if (mappingChanged) {
    auto replacement = mappingBuilder.create<DelinearizeOp>(
        mapping.getLoc(), coordinateTypes, mapping.getLinear(), runtimeExtents,
        mappingBuilder.getArrayAttr(launchExtents));
    SmallVector<int64_t> coordinateRoles(
        replacement.getNumResults(),
        static_cast<int64_t>(CoordinateRole::Unspecified));
    if (auto existing =
            mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr))
      if (existing.size() == mapping.getNumResults())
        llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
    auto ownershipRole = [&](Attribute axis) {
      auto contraction = contractionCoordinateRoles.find(axis);
      if (contraction != contractionCoordinateRoles.end())
        return contraction->second;
      return llvm::any_of(axes.lookup(axis), [](MakeRangeOp range) {
               return range->hasAttr(worksetCoordinateRangeAttr);
             })
                 ? CoordinateRole::TiledWorkset
                 : CoordinateRole::PointwiseOwnership;
    };
    for (auto [axisKey, axis] : reusedCoordinates) {
      if (coordinateRoles[axis] !=
          static_cast<int64_t>(CoordinateRole::IndirectTraversal))
        coordinateRoles[axis] =
            static_cast<int64_t>(ownershipRole(axisKey));
    }
    unsigned appendedAxis = mapping.getNumResults();
    for (Attribute axisKey : appendedCoordinates) {
      coordinateRoles[appendedAxis++] =
          static_cast<int64_t>(ownershipRole(axisKey));
    }
    replacement->setAttr(
        coordinateRolesAttr,
        DenseI64ArrayAttr::get(module.getContext(), coordinateRoles));
    for (StringRef attribute : {executionGroupAttr, segmentOffsetAttr})
      if (Attribute value = mapping->getAttr(attribute))
        replacement->setAttr(attribute, value);
    for (auto [axis, pair] : llvm::enumerate(llvm::zip(
             mapping.getCoordinates(),
             replacement.getCoordinates().take_front(
                 mapping.getCoordinates().size())))) {
      Value oldCoordinate = std::get<0>(pair);
      Value newCoordinate = std::get<1>(pair);
      auto knownAxis = coordinateAxes.find(axis);
      FailureOr<Attribute> axisKey =
          knownAxis != coordinateAxes.end()
              ? FailureOr<Attribute>(knownAxis->second)
              : axis < mapping.getLaunchExtents().size()
                    ? mappingAxis(mapping.getLaunchExtents()[axis])
                    : FailureOr<Attribute>(failure());
      if (succeeded(axisKey) && ownershipAxes.contains(*axisKey)) {
        ParameterOp parameter = parameters.lookup(*axisKey);
        if (!parameter)
          return kernel.emitError(
              "pointwise ownership mapping lost its blocking parameter");
        newCoordinate = mappingBuilder.create<BinaryOp>(
            mapping.getLoc(), mappingBuilder.getIndexType(), newCoordinate,
            parameter.getResult(), BinaryOperator::Multiply);
        oldCoordinate.replaceAllUsesWith(newCoordinate);
        continue;
      }
      oldCoordinate.replaceAllUsesWith(newCoordinate);
    }
    for (auto [axisKey, axis] : reusedCoordinates)
      tileCoordinates[axisKey] = replacement.getCoordinates()[axis];
    unsigned extra = mapping.getCoordinates().size();
    for (Attribute axisKey : appendedCoordinates)
      tileCoordinates[axisKey] = replacement.getCoordinates()[extra++];
    PhysicalExprAttr total = cast<PhysicalExprAttr>(launchExtents.front());
    for (Attribute extent : llvm::drop_begin(launchExtents))
      total = binaryExpression(module.getContext(), PhysicalExprKind::Multiply,
                               total, cast<PhysicalExprAttr>(extent));
    replacement->setAttr(segmentLengthAttr, total);
    mapping.erase();
    mapping = replacement;
    kernel->setAttr(programSpaceAttr,
                    ArrayAttr::get(module.getContext(), {total}));
  }

  llvm::DenseMap<Value, Value> rangePredicates =
      std::move(fixedRangePredicates);
  for (MakeRangeOp range : dynamicRanges) {
    OpBuilder builder(range);
    FailureOr<ParameterOp> parameter = queryBlockingParameter(kernel, range);
    if (MakeRangeOp root = occurrenceRoots.lookup(range.getOperation()))
      if (ParameterOp selected = occurrenceParameters.lookup(root.getOperation()))
        parameter = selected;
    FailureOr<Attribute> axisKey =
        failed(parameter) ? FailureOr<Attribute>(failure())
                          : parameterAxis(*parameter);
    if (failed(parameter) || failed(axisKey)) {
      return range.emitOpError(
                 "dynamic pointwise range lost its canonical blocking dimension")
             << "; source_id=" << range.getSourceId()
             << ", fragment=" << range.getResult().getType();
    }
    Value tileCoordinate = tileCoordinates.lookup(*axisKey);
    // A coverage-bound parameter is the executable statement that this axis is
    // traversed in full by one program.  It therefore has the canonical tile
    // coordinate zero even if an earlier same-logical-range ownership probe
    // classified another occurrence as a candidate program axis.  Read the
    // current IR fact here instead of caching it while parameters are still
    // being refined; ownership mapping removes coverage_dimension from axes it
    // places on the program grid.
    if (!tileCoordinate &&
        (internalAxes.contains(*axisKey) ||
         (*parameter)->hasAttr(coverageDimensionAttr)))
      tileCoordinate = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
    if (!tileCoordinate)
      return range.emitOpError(
                 "dynamic pointwise range has no physical tile coordinate")
             << "; axis=" << *axisKey
             << ", source_id=" << range.getSourceId()
             << ", ownership=" << ownershipAxes.contains(*axisKey)
             << ", internal=" << internalAxes.contains(*axisKey)
             << ", parameter_name="
             << (*parameter).getParameter().getName().getValue()
             << ", parameter_role=" << (*parameter).getParameter().getRole()
             << ", has_coverage="
             << (*parameter)->hasAttr(coverageDimensionAttr)
             << ", launch_extents=" << mapping->getAttr("launch_extents");
    Value tileOffset = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), tileCoordinate,
        parameter->getResult(), BinaryOperator::Multiply);
    Value start;
    Value end;
    if (range->hasAttr(worksetCoordinateRangeAttr)) {
      start = range.getStart();
      Value dimension = logicalDimensions.lookup(*axisKey);
      if (!dimension) {
        FailureOr<uint64_t> logicalDimension = axisDimension(*axisKey);
        if (failed(logicalDimension))
          logicalDimension = rangeDimension(range);
        if (succeeded(logicalDimension)) {
          FailureOr<Value> runtimeDimension =
              dimensionArgument(kernel, *logicalDimension);
          if (succeeded(runtimeDimension))
            dimension = *runtimeDimension;
        }
      }
      if (!dimension)
        return range.emitOpError(
                   "workset coordinate range lost its logical dimension extent")
               << "; axis=" << *axisKey << ", source_id="
               << range.getSourceId() << ", parameter="
               << parameter->getParameter().getName().getValue();
      Value remaining = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), dimension, tileOffset,
          BinaryOperator::Subtract);
      Value remainingDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), remaining, range.getStep(),
          BinaryOperator::Multiply);
      end = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                     start, remainingDistance,
                                     BinaryOperator::Add);
    } else {
      Value scaledOffset = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), tileOffset, range.getStep(),
          BinaryOperator::Multiply);
      Value anchor = range.getLogicalStart();
      auto parentOffset = parentAnchorOffsets.find(range.getOperation());
      if (parentOffset != parentAnchorOffsets.end()) {
        Value offset = builder.create<arith::ConstantIndexOp>(
            range.getLoc(), parentOffset->second);
        anchor = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                          anchor, offset, BinaryOperator::Add);
      }
      start = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), anchor, scaledOffset,
          BinaryOperator::Add);
      end = range.getLogicalStop();
    }
    auto sourceType = cast<FragmentType>(range.getResult().getType());
    PhysicalExprAttr tileExtent = fragmentExtent(*parameter);
    if (sourceType.getShape()[0] != tileExtent) {
      if (FailureOr<uint64_t> dimension = rangeDimension(range);
          succeeded(dimension) && !queryParameterBinding(*parameter).source)
        retargetDimensionExtent(range.getResult(), *dimension, tileExtent);
      else
        retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                             tileExtent);
      sourceType = cast<FragmentType>(range.getResult().getType());
    }
    Value physicalExtent = parameter->getResult();
    if (tileExtent.getKind() ==
        static_cast<uint32_t>(PhysicalExprKind::Constant))
      physicalExtent = builder.create<arith::ConstantIndexOp>(
          range.getLoc(), tileExtent.getValue());
    auto blockedType = FragmentType::get(
        module.getContext(), sourceType.getElementType(),
        builder.getArrayAttr({tileExtent}), sourceType.getAxisMaps(),
        sourceType.getValidity(),
        sourceType.getOwner());
    Value blocked = builder.create<MakeRangeOp>(
        range.getLoc(), blockedType, start, physicalExtent,
        range.getStep(), range.getLogicalStart(),
        range->hasAttr(worksetCoordinateRangeAttr) ? end : range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(blocked.getDefiningOp(), range);
    for (BroadcastOp bound : laneBounds.lookup(range.getResult()))
      bound.getValueMutable().assign(end);
    Value endFragment = builder.create<BroadcastOp>(range.getLoc(), blockedType, end);
    auto validComparison = builder.create<CompareOp>(
        range.getLoc(), predicateType(blockedType), blocked, endFragment,
        ComparePredicate::Lt);
    validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
    Value valid = validComparison.getResult();
    if (parentAnchorOffsets.contains(range.getOperation())) {
      Value lower = builder.create<BroadcastOp>(
          range.getLoc(), blockedType, range.getLogicalStart());
      Value active = builder.create<CompareOp>(
          range.getLoc(), predicateType(blockedType), blocked, lower,
          ComparePredicate::Ge);
      valid = builder.create<BinaryOp>(range.getLoc(), predicateType(blockedType),
                                       active, valid, BinaryOperator::LogicalAnd);
    }
    range.getResult().replaceAllUsesWith(blocked);
    rangePredicates[blocked] = valid;
    range.erase();
  }

  // Dynamic tail predicates are side-table facts until they are attached to
  // accesses.  Materialize those SSA uses before histogram realization, whose
  // dead-value cleanup would otherwise erase the unused comparisons and leave
  // dangling Values in rangePredicates.
  if (failed(alignAccessResultRelations(kernel)) ||
      failed(addTailValidity(kernel, rangePredicates,
                             /*includeStores=*/true)))
    return failure();
  if (ownershipOnly) {
    if (failed(alignHistogramOutputOwnership(kernel)))
      return failure();
  } else if (failed(realizeOwnedHistograms(kernel))) {
    return failure();
  }
  if (failed(alignContractValueRelations(kernel)))
    return failure();
  return finalizeValueRelations();
}

LogicalResult realizePointwiseOwnership(ModuleOp module) {
  return realizePointwiseBlockingImpl(module, /*ownershipOnly=*/true);
}

LogicalResult realizePointwiseBlocking(ModuleOp module) {
  return realizePointwiseBlockingImpl(module, /*ownershipOnly=*/false);
}

} // namespace intent::gpu

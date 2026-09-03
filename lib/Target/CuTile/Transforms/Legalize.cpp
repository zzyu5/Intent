#include "Intent/Target/CuTile/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

#include <optional>

using namespace mlir;

namespace intent::cutile {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_cutile.legalized";

std::optional<int64_t>
constantPhysicalExpression(gpu::PhysicalExprAttr expression,
                           func::FuncOp kernel, unsigned depth = 0) {
  if (!expression || depth >= 32)
    return std::nullopt;
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return expression.getValue();
  if (kind == gpu::PhysicalExprKind::Parameter) {
    FailureOr<gpu::ParameterOp> parameter =
        kernel ? gpu::queryParameterBySymbol(kernel, expression.getSymbol())
               : FailureOr<gpu::ParameterOp>(failure());
    if (failed(parameter))
      return std::nullopt;
    ArrayRef<int64_t> candidates =
        parameter->getParameter().getCandidates().asArrayRef();
    return candidates.size() == 1 ? std::optional<int64_t>(candidates.front())
                                  : std::nullopt;
  }
  if (expression.getOperands().size() != 2)
    return std::nullopt;
  auto lhs = constantPhysicalExpression(
      cast<gpu::PhysicalExprAttr>(expression.getOperands()[0]), kernel,
      depth + 1);
  auto rhs = constantPhysicalExpression(
      cast<gpu::PhysicalExprAttr>(expression.getOperands()[1]), kernel,
      depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (kind) {
  case gpu::PhysicalExprKind::Add:
    return *lhs + *rhs;
  case gpu::PhysicalExprKind::Subtract:
    return *lhs - *rhs;
  case gpu::PhysicalExprKind::Multiply:
    return *lhs * *rhs;
  case gpu::PhysicalExprKind::FloorDiv:
    return *rhs == 0 ? std::nullopt
                     : std::optional<int64_t>(*lhs / *rhs);
  case gpu::PhysicalExprKind::CeilDiv:
    return *rhs <= 0 || *lhs < 0
               ? std::nullopt
               : std::optional<int64_t>((*lhs + *rhs - 1) / *rhs);
  case gpu::PhysicalExprKind::Minimum:
    return std::min(*lhs, *rhs);
  case gpu::PhysicalExprKind::Maximum:
    return std::max(*lhs, *rhs);
  default:
    return std::nullopt;
  }
}

std::optional<int64_t> constantValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>()) {
    ArrayRef<int64_t> candidates =
        parameter.getParameter().getCandidates().asArrayRef();
    if (candidates.size() == 1)
      return candidates.front();
  }
  if (auto physical = value.getDefiningOp<gpu::PhysicalExprOp>())
    return constantPhysicalExpression(
        physical.getExpression(),
        physical->getParentOfType<func::FuncOp>());
  if (auto bound = value.getDefiningOp<gpu::RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<gpu::RangeOp>();
    if (!range)
      return std::nullopt;
    if (bound.getBound() == 0)
      return constantValue(range.getStart());
    if (bound.getBound() == 1)
      return constantValue(range.getStop());
    return constantValue(range.getStep());
  }
  if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    std::optional<int64_t> lhs = constantValue(binary.getLhs());
    std::optional<int64_t> rhs = constantValue(binary.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
      return *lhs + *rhs;
    case BinaryOperator::Subtract:
      return *lhs - *rhs;
    case BinaryOperator::Multiply:
      return *lhs * *rhs;
    case BinaryOperator::FloorDivide:
      return *rhs == 0 ? std::nullopt
                       : std::optional<int64_t>(*lhs / *rhs);
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

bool isProvably(Value value, int64_t expected) {
  std::optional<int64_t> actual = constantValue(value);
  return actual && *actual == expected;
}

FailureOr<Value> tileIndex(OpBuilder &builder, Location location, Value start,
                           Value extent) {
  std::optional<int64_t> startConstant = constantValue(start);
  std::optional<int64_t> extentConstant = constantValue(extent);
  if (startConstant && extentConstant && *startConstant >= 0 &&
      *extentConstant > 0 && *startConstant % *extentConstant == 0)
    return builder
        .create<arith::ConstantIndexOp>(location,
                                        *startConstant / *extentConstant)
        .getResult();
  if (isProvably(start, 0))
    return builder.create<arith::ConstantIndexOp>(location, 0).getResult();
  if (auto coordinate = start.getDefiningOp<gpu::WorksetCoordinateOp>())
    return tileIndex(builder, location, coordinate.getCoordinate(), extent);
  if (auto binary = start.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Multiply) {
      if (binary.getLhs() == extent)
        return binary.getRhs();
      if (binary.getRhs() == extent)
        return binary.getLhs();
      if (isProvably(binary.getLhs(), 1))
        return tileIndex(builder, location, binary.getRhs(), extent);
      if (isProvably(binary.getRhs(), 1))
        return tileIndex(builder, location, binary.getLhs(), extent);
    }
    if (binary.getOperatorKind() == BinaryOperator::Add) {
      if (isProvably(binary.getLhs(), 0))
        return tileIndex(builder, location, binary.getRhs(), extent);
      if (isProvably(binary.getRhs(), 0))
        return tileIndex(builder, location, binary.getLhs(), extent);
    }
  }
  auto argument = dyn_cast<BlockArgument>(start);
  auto loop = argument ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
                       : scf::ForOp();
  if (loop && argument == loop.getInductionVar() &&
      isProvably(loop.getLowerBound(), 0) && loop.getStep() == extent)
    return Value(builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), start, extent,
        BinaryOperator::FloorDivide));
  return failure();
}

FailureOr<SmallVector<Value>> orderedCoordinates(Operation *owner, Value resource,
                                                 ValueRange coordinates,
                                                 ArrayRef<int64_t> sourceAxes) {
  auto view = cast<gpu::ViewType>(resource.getType());
  SmallVector<Value> result(view.getRank());
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes))
    result[sourceAxis] = coordinate;
  if (llvm::any_of(result, [](Value value) { return !value; }))
    return owner->emitOpError("cuTile advanced access source axes are incomplete");
  return result;
}

bool samePhysicalDomain(gpu::FragmentType lhs, gpu::FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

gpu::FragmentType transposeRankTwo(gpu::FragmentType source) {
  SmallVector<Attribute> shape = {source.getShape()[1], source.getShape()[0]};
  SmallVector<Attribute> mappings;
  mappings.reserve(2);
  for (unsigned resultAxis = 0; resultAxis < 2; ++resultAxis) {
    const unsigned sourceAxis = 1 - resultAxis;
    auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    mappings.push_back(gpu::AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resultAxis, mapping.getDerived()));
  }
  return gpu::FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape),
      ArrayAttr::get(source.getContext(), mappings), source.getValidity(),
      source.getOwner());
}

bool canBroadcastTo(gpu::FragmentType source, gpu::FragmentType target) {
  for (auto [sourceIndex, sourceAttribute] :
       llvm::enumerate(source.getAxisMaps())) {
    auto sourceMap = cast<gpu::AxisMapAttr>(sourceAttribute);
    std::optional<unsigned> targetIndex;
    for (auto [index, targetAttribute] :
         llvm::enumerate(target.getAxisMaps())) {
      auto targetMap = cast<gpu::AxisMapAttr>(targetAttribute);
      if (sourceMap.getSourceId() != targetMap.getSourceId() ||
          sourceMap.getSourceAxis() != targetMap.getSourceAxis())
        continue;
      if (targetIndex)
        return false;
      targetIndex = index;
    }
    if (!targetIndex)
      return false;
    Attribute sourceExtent = source.getShape()[sourceIndex];
    Attribute targetExtent = target.getShape()[*targetIndex];
    auto constant = dyn_cast<gpu::PhysicalExprAttr>(sourceExtent);
    bool unit = constant &&
                constant.getKind() == static_cast<uint32_t>(
                                          gpu::PhysicalExprKind::Constant) &&
                constant.getValue() == 1;
    if (!unit && sourceExtent != targetExtent)
      return false;
  }
  return true;
}

FailureOr<SmallVector<Value>> materializeCoordinateDomains(
    OpBuilder &builder, Operation *owner, ValueRange coordinates,
    gpu::FragmentType target) {
  SmallVector<Value> results;
  results.reserve(coordinates.size());
  for (Value coordinate : coordinates) {
    auto source = dyn_cast<gpu::FragmentType>(coordinate.getType());
    if (!source || samePhysicalDomain(source, target)) {
      results.push_back(coordinate);
      continue;
    }
    if (source.getOwner() != target.getOwner() ||
        source.getShape().size() > target.getShape().size() ||
        !canBroadcastTo(source, target))
      return owner->emitOpError(
          "cuTile coordinate cannot adopt the selected physical domain");
    auto resultType = gpu::FragmentType::get(
        owner->getContext(), source.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    auto broadcast = builder.create<gpu::BroadcastOp>(
        owner->getLoc(), resultType, coordinate);
    if (Operation *definition = coordinate.getDefiningOp())
      if (Attribute origin = definition->getAttr(gpu::originAttr))
        broadcast->setAttr(gpu::originAttr, origin);
    results.push_back(broadcast.getResult());
  }
  return results;
}

bool sameScalarFill(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rhsConstant = rhs.getDefiningOp<arith::ConstantOp>();
  return lhsConstant && rhsConstant && lhs.getType() == rhs.getType() &&
         lhsConstant.getValue() == rhsConstant.getValue();
}

Value uniformScalarFill(Value fill) {
  if (!fill)
    return fill;
  while (isa<gpu::FragmentType>(fill.getType())) {
    if (auto splat = fill.getDefiningOp<gpu::SplatOp>()) {
      fill = splat.getValue();
      continue;
    }
    if (auto broadcast = fill.getDefiningOp<gpu::BroadcastOp>()) {
      fill = broadcast.getValue();
      continue;
    }
    if (auto select = fill.getDefiningOp<gpu::SelectOp>()) {
      Value trueFill = uniformScalarFill(select.getTrueValue());
      Value falseFill = uniformScalarFill(select.getFalseValue());
      return trueFill && falseFill && sameScalarFill(trueFill, falseFill)
                 ? trueFill
                 : Value();
    }
    return Value();
  }
  return fill;
}

FailureOr<Value> scalarFill(Operation *owner, Value fill) {
  Value scalar = uniformScalarFill(fill);
  if (!fill || scalar)
    return scalar;
  return owner->emitOpError(
      "cuTile gather padding must be an explicit scalar or splat");
}

struct NativeTileAxisPlan {
  unsigned computationAxis = 0;
  Value scalarIndex;
  gpu::MakeRangeOp range;
  SmallVector<Value> offsets;
  bool originInBounds = false;
};

struct NativeTileAccessPlan {
  gpu::FragmentType resourceType;
  SmallVector<NativeTileAxisPlan> axes;
  SmallVector<int64_t> toComputation;
  SmallVector<int64_t> toResource;
};

bool isUnitExtent(Attribute attribute) {
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  return extent &&
         extent.getKind() ==
             static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) &&
         extent.getValue() == 1;
}

bool collectTileOffsets(Value value, Value range,
                        SmallVectorImpl<Value> &offsets) {
  if (value == range)
    return true;
  if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
    return collectTileOffsets(broadcast.getValue(), range, offsets);
  if (auto reshape = value.getDefiningOp<gpu::ReshapeOp>())
    return collectTileOffsets(reshape.getValue(), range, offsets);
  if (auto transpose = value.getDefiningOp<gpu::TransposeOp>())
    return collectTileOffsets(transpose.getValue(), range, offsets);
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary || binary.getOperatorKind() != BinaryOperator::Add)
    return false;
  if (Value scalar = uniformScalarFill(binary.getRhs())) {
    SmallVector<Value> nested;
    if (collectTileOffsets(binary.getLhs(), range, nested)) {
      offsets.append(nested);
      offsets.push_back(scalar);
      return true;
    }
  }
  if (Value scalar = uniformScalarFill(binary.getLhs())) {
    SmallVector<Value> nested;
    if (collectTileOffsets(binary.getRhs(), range, nested)) {
      offsets.append(nested);
      offsets.push_back(scalar);
      return true;
    }
  }
  return false;
}

std::optional<int64_t> constantTileOrigin(gpu::MakeRangeOp range,
                                          ArrayRef<Value> offsets) {
  std::optional<int64_t> result = constantValue(range.getStart());
  if (!result)
    return std::nullopt;
  for (Value offset : offsets) {
    std::optional<int64_t> constant = constantValue(offset);
    if (!constant)
      return std::nullopt;
    *result += *constant;
  }
  return result;
}

bool constantOriginInView(int64_t origin, gpu::ViewType view,
                          unsigned resourceAxis, func::FuncOp kernel) {
  if (origin < 0 || resourceAxis >= view.getLayout().getExtents().size())
    return false;
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(
      view.getLayout().getExtents()[resourceAxis]);
  std::optional<int64_t> constant =
      constantPhysicalExpression(extent, kernel);
  return constant && origin < *constant;
}

bool rangeOriginInView(gpu::MakeRangeOp range, ArrayRef<Value> offsets,
                       gpu::ViewType view, unsigned resourceAxis,
                       int64_t dimension, func::FuncOp kernel) {
  bool zeroOffset = llvm::all_of(
      offsets, [](Value offset) { return isProvably(offset, 0); });
  if (zeroOffset && range->hasAttr(gpu::worksetCoordinateRangeAttr) &&
      !range->hasAttr(gpu::sourceSubregionAttr)) {
    auto coordinate =
        range.getStart().getDefiningOp<gpu::WorksetCoordinateOp>();
    if (coordinate &&
        coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
      return true;
  }
  std::optional<int64_t> origin = constantTileOrigin(range, offsets);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarOriginInView(Value index, gpu::ViewType view,
                        unsigned resourceAxis, int64_t dimension,
                        func::FuncOp kernel) {
  if (auto coordinate = index.getDefiningOp<gpu::WorksetCoordinateOp>())
    if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
      return true;
  std::optional<int64_t> origin = constantValue(index);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

FailureOr<NativeTileAccessPlan> analyzeNativeTileAccess(
    Operation *owner, func::FuncOp kernel,
    gpu::PhysicalProgramAnalysis &analysis, gpu::ViewType view,
    ValueRange coordinates, ArrayRef<int64_t> sourceAxes,
    gpu::FragmentType computationType) {
  const unsigned rank = view.getRank();
  if (coordinates.size() != rank || sourceAxes.size() != rank ||
      computationType.getShape().size() != rank)
    return failure();

  SmallVector<Value> resourceCoordinates(rank);
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes)) {
    if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(rank) ||
        resourceCoordinates[sourceAxis])
      return failure();
    resourceCoordinates[sourceAxis] = coordinate;
  }

  NativeTileAccessPlan plan;
  plan.axes.resize(rank);
  plan.toComputation.assign(rank, -1);
  plan.toResource.assign(rank, -1);
  SmallVector<bool> usedComputationAxis(rank, false);
  SmallVector<unsigned> unresolvedScalarAxes;
  ArrayRef<int64_t> dimensions =
      view.getLayout().getDimensionIds().asArrayRef();
  if (dimensions.size() != rank)
    return failure();

  auto assignComputationAxis = [&](unsigned resourceAxis,
                                   unsigned computationAxis) {
    if (computationAxis >= rank || usedComputationAxis[computationAxis])
      return false;
    usedComputationAxis[computationAxis] = true;
    plan.axes[resourceAxis].computationAxis = computationAxis;
    plan.toComputation[computationAxis] = resourceAxis;
    plan.toResource[resourceAxis] = computationAxis;
    return true;
  };

  for (unsigned resourceAxis = 0; resourceAxis < rank; ++resourceAxis) {
    NativeTileAxisPlan &axis = plan.axes[resourceAxis];
    Value coordinate = resourceCoordinates[resourceAxis];
    Value scalar = uniformScalarFill(coordinate);
    gpu::PhysicalRangeFact ranges = analysis.sourceRanges(coordinate);
    if (ranges.isExact() && ranges.roots.size() == 1) {
      axis.range = ranges.roots.front();
      SmallVector<gpu::PhysicalAxisProjection, 2> projections =
          gpu::queryRangeProjections(computationType, axis.range);
      auto rangeType =
          dyn_cast<gpu::FragmentType>(axis.range.getResult().getType());
      if (projections.size() != 1 ||
          !assignComputationAxis(resourceAxis,
                                 projections.front().fragmentAxis) ||
          !gpu::isUnitStepRange(axis.range) || !rangeType ||
          rangeType.getShape().size() != 1 ||
          rangeType.getShape()[0] !=
              computationType.getShape()[axis.computationAxis] ||
          !collectTileOffsets(coordinate, axis.range.getResult(), axis.offsets))
        return failure();
      axis.originInBounds = rangeOriginInView(
          axis.range, axis.offsets, view, resourceAxis,
          dimensions[resourceAxis], kernel);
      continue;
    }

    if (!scalar || !scalar.getType().isIndex())
      return failure();
    axis.scalarIndex = scalar;
    if (auto workset = scalar.getDefiningOp<gpu::WorksetCoordinateOp>()) {
      gpu::PhysicalDimensionProjection projection =
          gpu::queryFragmentDimension(computationType,
                                      workset.getDimensionId());
      if (!projection.isExact() ||
          !assignComputationAxis(resourceAxis, projection.fragmentAxis) ||
          !isUnitExtent(computationType.getShape()[axis.computationAxis]))
        return failure();
      axis.scalarIndex = scalar;
      axis.originInBounds = scalarOriginInView(
          scalar, view, resourceAxis, dimensions[resourceAxis], kernel);
    } else {
      unresolvedScalarAxes.push_back(resourceAxis);
    }
  }

  SmallVector<unsigned> remainingUnitAxes;
  for (unsigned computationAxis = 0; computationAxis < rank;
       ++computationAxis)
    if (!usedComputationAxis[computationAxis] &&
        isUnitExtent(computationType.getShape()[computationAxis]))
      remainingUnitAxes.push_back(computationAxis);
  if (unresolvedScalarAxes.size() != remainingUnitAxes.size() ||
      unresolvedScalarAxes.size() > 1)
    return failure();
  for (auto [resourceAxis, computationAxis] :
       llvm::zip(unresolvedScalarAxes, remainingUnitAxes)) {
    if (!assignComputationAxis(resourceAxis, computationAxis))
      return failure();
    NativeTileAxisPlan &axis = plan.axes[resourceAxis];
    axis.originInBounds = scalarOriginInView(
        axis.scalarIndex, view, resourceAxis, dimensions[resourceAxis], kernel);
  }

  if (llvm::any_of(usedComputationAxis, [](bool used) { return !used; }))
    return failure();

  SmallVector<Attribute> resourceShape(rank);
  SmallVector<Attribute> resourceMappings(rank);
  for (unsigned resourceAxis = 0; resourceAxis < rank; ++resourceAxis) {
    unsigned computationAxis = plan.axes[resourceAxis].computationAxis;
    resourceShape[resourceAxis] = computationType.getShape()[computationAxis];
    auto mapping = cast<gpu::AxisMapAttr>(
        computationType.getAxisMaps()[computationAxis]);
    resourceMappings[resourceAxis] = gpu::AxisMapAttr::get(
        owner->getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resourceAxis, mapping.getDerived());
  }

  plan.resourceType = gpu::FragmentType::get(
      owner->getContext(), computationType.getElementType(),
      ArrayAttr::get(owner->getContext(), resourceShape),
      ArrayAttr::get(owner->getContext(), resourceMappings),
      computationType.getValidity(), computationType.getOwner());
  return plan;
}

bool boundaryCompatible(const NativeTileAccessPlan &plan,
                        const gpu::PhysicalAccessBoundaryFact &boundary) {
  if (!boundary.isExact())
    return false;
  return llvm::all_of(boundary.boundaryAxes, [&](int64_t axis) {
    return axis >= 0 && axis < static_cast<int64_t>(plan.axes.size()) &&
           plan.axes[axis].originInBounds;
  });
}

FailureOr<SmallVector<Value>>
materializeTileIndices(OpBuilder &builder, Operation *owner,
                       const NativeTileAccessPlan &plan) {
  SmallVector<Value> result;
  result.reserve(plan.axes.size());
  for (const NativeTileAxisPlan &axis : plan.axes) {
    if (axis.scalarIndex) {
      result.push_back(axis.scalarIndex);
      continue;
    }
    gpu::MakeRangeOp range = axis.range;
    Value start = range.getStart();
    for (Value offset : axis.offsets)
      start = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getIndexType(), start, offset,
          BinaryOperator::Add);
    FailureOr<Value> index = tileIndex(builder, owner->getLoc(), start,
                                       range.getExtent());
    if (failed(index))
      return failure();
    result.push_back(*index);
  }
  return result;
}

bool isIdentityPermutation(ArrayRef<int64_t> permutation) {
  return llvm::all_of(llvm::enumerate(permutation),
                      [](auto item) {
                        return static_cast<int64_t>(item.index()) ==
                               item.value();
                      });
}

SmallVector<int64_t> identityAxes(unsigned rank) {
  SmallVector<int64_t> result;
  result.reserve(rank);
  for (unsigned axis = 0; axis < rank; ++axis)
    result.push_back(axis);
  return result;
}

std::optional<BinaryOperator> nativeCombineKind(Region &region) {
  std::optional<BinaryOperator> kind = gpu::queryBinaryCombineKind(region);
  if (!kind)
    return std::nullopt;
  if (*kind == BinaryOperator::Add || *kind == BinaryOperator::MaximumNum ||
      *kind == BinaryOperator::MinimumNum)
    return kind;
  auto result =
      dyn_cast<gpu::FragmentType>(region.front().getArgument(0).getType());
  if (result && result.getElementType().isInteger(1)) {
    if (*kind == BinaryOperator::LogicalOr)
      return BinaryOperator::LogicalOr;
    if (*kind == BinaryOperator::LogicalAnd)
      return BinaryOperator::LogicalAnd;
  }
  return std::nullopt;
}

Attribute scalarConstant(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<gpu::CastOp>()) {
      value = cast.getValue();
      continue;
    }
    if (auto splat = value.getDefiningOp<gpu::SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto extract = value.getDefiningOp<gpu::ExtractOp>()) {
      auto record = extract.getRecord().getDefiningOp<gpu::MakeRecordOp>();
      if (record && extract.getField() < record.getFields().size()) {
        value = record.getFields()[extract.getField()];
        continue;
      }
    }
    break;
  }
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  return constant ? constant.getValue() : Attribute();
}

bool isZeroFill(Value value) {
  Attribute constant = scalarConstant(value);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant))
    return floating.getValue().isZero();
  return false;
}

LogicalResult formNativeTiles(func::FuncOp kernel) {
  SmallVector<gpu::LoadOp> loads;
  SmallVector<gpu::GatherOp> gathers;
  SmallVector<gpu::ContractOp> contracts;
  SmallVector<gpu::ScaledContractOp> scaledContracts;
  SmallVector<gpu::ReduceOp> reductions;
  SmallVector<gpu::ScanOp> scans;
  SmallVector<gpu::StoreOp> stores;
  SmallVector<gpu::AtomicRMWOp> atomics;
  SmallVector<gpu::AssumeInBoundsOp> assumptions;
  kernel.walk([&](gpu::LoadOp op) { loads.push_back(op); });
  kernel.walk([&](gpu::GatherOp op) { gathers.push_back(op); });
  kernel.walk([&](gpu::ContractOp op) { contracts.push_back(op); });
  kernel.walk(
      [&](gpu::ScaledContractOp op) { scaledContracts.push_back(op); });
  kernel.walk([&](gpu::ReduceOp op) { reductions.push_back(op); });
  kernel.walk([&](gpu::ScanOp op) { scans.push_back(op); });
  kernel.walk([&](gpu::StoreOp op) { stores.push_back(op); });
  kernel.walk([&](gpu::AtomicRMWOp op) { atomics.push_back(op); });
  kernel.walk([&](gpu::AssumeInBoundsOp op) { assumptions.push_back(op); });
  gpu::PhysicalProgramAnalysis analysis(kernel);

  for (gpu::LoadOp load : loads) {
    auto view = dyn_cast<gpu::ViewType>(load.getResource().getType());
    if (!view)
      return load.emitOpError(
          "cuTile native load requires an external view resource");
    OpBuilder builder(load);
    if (!isa<gpu::FragmentType>(load.getResult().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(
          load, load.getResource(), load.getCoordinates(), load.getSourceAxes());
      FailureOr<Value> fill = scalarFill(load, load.getFill());
      if (failed(indices) || failed(fill) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (load.getValid() && isa<gpu::FragmentType>(load.getValid().getType())))
        return load.emitOpError(
            "cuTile scalar load requires scalar indices and validity");
      auto replacement = builder.create<ScalarLoadOp>(
          load.getLoc(), load.getResult().getType(), load.getResource(), *indices,
          load.getValid(), *fill);
      if (Attribute origin = load->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      load.getResult().replaceAllUsesWith(replacement.getResult());
      load.erase();
      continue;
    }
    auto result = cast<gpu::FragmentType>(load.getResult().getType());
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(load);
    FailureOr<NativeTileAccessPlan> plan = analyzeNativeTileAccess(
        load, kernel, analysis, view, load.getCoordinates(),
        load.getSourceAxes(), result);
    FailureOr<SmallVector<Value>> indices = failure();
    if (succeeded(plan) && boundaryCompatible(*plan, boundary) &&
        (!load.getFill() || isZeroFill(load.getFill())))
      indices = materializeTileIndices(builder, load, *plan);
    Value replacementResult;
    Operation *replacementOperation = nullptr;
    Operation *relationOperation = nullptr;
    if (succeeded(indices)) {
      auto replacement = builder.create<TileLoadOp>(
          load.getLoc(), plan->resourceType, load.getResource(), *indices);
      replacementResult = replacement.getResult();
      replacementOperation = replacement;
      if (!isIdentityPermutation(plan->toComputation)) {
        auto transpose = builder.create<gpu::TransposeOp>(
            load.getLoc(), result, replacementResult, plan->toComputation);
        replacementResult = transpose.getResult();
        relationOperation = transpose;
      }
    } else {
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
          load, load.getResource(), load.getCoordinates(), load.getSourceAxes());
      FailureOr<Value> fill = scalarFill(load, load.getFill());
      if (failed(coordinates) || failed(fill))
        return failure();
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, load, *coordinates, result);
      if (failed(materialized))
        return failure();
      auto replacement = builder.create<GatherLoadOp>(
          load.getLoc(), result, load.getResource(), *materialized,
          load.getValid(), *fill, identityAxes(view.getRank()));
      replacementResult = replacement.getResult();
      replacementOperation = replacement;
    }
    if (Attribute origin = load->getAttr(gpu::originAttr))
      replacementOperation->setAttr(gpu::originAttr, origin);
    if (relationOperation)
      if (Attribute origin = load->getAttr(gpu::originAttr))
        relationOperation->setAttr(gpu::originAttr, origin);
    load.getResult().replaceAllUsesWith(replacementResult);
    load.erase();
  }

  for (gpu::GatherOp gather : gathers) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    if (!source || isa<gpu::FragmentType>(gather.getResult().getType()) ||
        llvm::any_of(gather.getCoordinates(), [](Value coordinate) {
          return isa<gpu::FragmentType>(coordinate.getType());
        }))
      return gather.emitOpError(
          "cuTile tile extraction requires scalar coordinates and result");
    if (gather.getCoordinates().size() != gather.getSourceAxes().size())
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    SmallVector<Value> coordinates(source.getShape().size());
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(coordinates.size()) ||
          coordinates[sourceAxis])
        return gather.emitOpError(
            "cuTile tile extraction source axes are not a permutation");
      coordinates[sourceAxis] = coordinate;
    }
    if (llvm::any_of(coordinates, [](Value coordinate) { return !coordinate; }))
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    OpBuilder builder(gather);
    auto replacement = builder.create<ExtractScalarOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        coordinates, gather.getValid(), gather.getFill());
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
  }

  for (gpu::ReduceOp reduce : reductions) {
    std::optional<BinaryOperator> kind = nativeCombineKind(reduce.getCombine());
    bool native = reduce.getSourceCount() == 1 &&
                  reduce.getIdentityCount() == 1 &&
                  reduce.getCaptureCount() == 0 &&
                  reduce.getAxes().size() == 1 &&
                  reduce.getNumResults() == 1 && kind.has_value();
    if (native) {
      auto source =
          dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
      if (!source)
        return reduce.emitOpError("cuTile native reduce source must be a tile");
      OpBuilder builder(reduce);
      auto replacement = builder.create<ReduceOp>(
          reduce.getLoc(), reduce.getResultTypes().front(),
          reduce.getInputs().front(), reduce.getAxes().front(), *kind);
      reduce.getResults().front().replaceAllUsesWith(replacement.getResult());
      reduce.erase();
      continue;
    }

    if (reduce.getSourceCount() == 0 ||
        reduce.getSourceCount() != reduce.getIdentityCount() ||
        reduce.getSourceCount() != reduce.getNumResults() ||
        reduce.getCaptureCount() != 0 || reduce.getAxes().size() != 1)
      return reduce.emitOpError(
          "cuTile custom reduce requires matching non-empty source/identity/result schemas, one axis, and no captures");
    for (Value source : reduce.getInputs().take_front(reduce.getSourceCount()))
      if (!isa<gpu::FragmentType>(source.getType()))
        return reduce.emitOpError("cuTile custom reduce source must be a tile");
    for (Value identity : reduce.getInputs().slice(
             reduce.getSourceCount(), reduce.getIdentityCount()))
      if (!scalarConstant(identity)) {
        InFlightDiagnostic diagnostic = reduce.emitOpError(
            "cuTile custom reduce identity must be an explicit scalar constant");
        diagnostic << "; identity type=" << identity.getType();
        if (Operation *producer = identity.getDefiningOp())
          diagnostic << ", producer=" << producer->getName();
        return failure();
      }
  }

  for (gpu::ScanOp scan : scans) {
    std::optional<BinaryOperator> kind = nativeCombineKind(scan.getCombine());
    if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
        scan.getCaptureCount() != 0 || scan.getNumResults() != 1 || !kind ||
        *kind != BinaryOperator::Add || !scan.getInclusive() ||
        scan.getReverse())
      return scan.emitOpError(
          "cuTile native scan requires one source/identity, inclusive forward additive combine");
    auto source = dyn_cast<gpu::FragmentType>(scan.getInputs().front().getType());
    if (!source)
      return scan.emitOpError("cuTile native scan source must be a tile");
    OpBuilder builder(scan);
    auto replacement = builder.create<ScanOp>(
        scan.getLoc(), cast<gpu::FragmentType>(scan.getResultTypes().front()),
        scan.getInputs().front(), scan.getAxis(), *kind, false);
    scan.getResults().front().replaceAllUsesWith(replacement.getResult());
    scan.erase();
  }

  for (gpu::ContractOp contract : contracts) {
    if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      return contract.emitOpError(
          "cuTile native MMA requires [M,K] x [K,N] physical axes");
    OpBuilder builder(contract);
    auto replacement = builder.create<MMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getRhs(), contract.getAccumulator());
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    contract.getResult().replaceAllUsesWith(replacement.getResult());
    contract.erase();
  }

  for (gpu::ScaledContractOp contract : scaledContracts) {
    if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1, 2} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{0, 1} ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty() ||
        contract.getLhsFormat() != ScaledFormat::E4M3 ||
        contract.getRhsFormat() != ScaledFormat::E4M3 ||
        contract.getLhsGroupSize() != 32 ||
        contract.getRhsGroupSize() != 32)
      return contract.emitOpError(
          "cuTile scaled MMA requires E4M3/E8M0 group-32 adjacent reduction axes");
    OpBuilder builder(contract);
    auto rhsScale = builder.create<gpu::TransposeOp>(
        contract.getLoc(),
        transposeRankTwo(
            cast<gpu::FragmentType>(contract.getRhsScale().getType())),
        contract.getRhsScale(), ArrayRef<int64_t>{1, 0});
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      rhsScale->setAttr(gpu::originAttr, origin);
    auto replacement = builder.create<ScaledMMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getLhsScale(), contract.getRhs(), rhsScale,
        contract.getAccumulator(), contract.getLhsFormat(),
        contract.getRhsFormat(), contract.getLhsGroupSize(),
        contract.getRhsGroupSize());
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    contract.getResult().replaceAllUsesWith(replacement.getResult());
    contract.erase();
  }

  for (gpu::AtomicRMWOp atomic : atomics) {
    auto view = dyn_cast<gpu::ViewType>(atomic.getResource().getType());
    if (!view)
      return atomic.emitOpError(
          "cuTile atomic RMW requires an external view resource");
    if (atomic.getValid())
      return atomic.emitOpError(
          "cuTile atomic RMW has no native arbitrary-validity form");
    FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
        atomic, atomic.getResource(), atomic.getCoordinates(),
        atomic.getSourceAxes());
    if (failed(coordinates))
      return failure();
    OpBuilder builder(atomic);
    if (auto target = dyn_cast<gpu::FragmentType>(atomic.getValue().getType())) {
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, atomic, *coordinates, target);
      if (failed(materialized))
        return failure();
      coordinates = std::move(*materialized);
    }
    auto replacement = builder.create<AtomicRMWOp>(
        atomic.getLoc(), atomic.getResult().getType(), atomic.getResource(),
        *coordinates, atomic.getValue(), atomic.getKind(), atomic.getOrdering(),
        atomic.getSharing());
    if (Attribute origin = atomic->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }

  for (gpu::StoreOp store : stores) {
    auto view = dyn_cast<gpu::ViewType>(store.getResource().getType());
    if (!view)
      return store.emitOpError(
          "cuTile native store requires an external unique-write view");
    OpBuilder builder(store);
    if (!isa<gpu::FragmentType>(store.getValue().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(
          store, store.getResource(), store.getCoordinates(), store.getSourceAxes());
      if (failed(indices) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (store.getValid() && isa<gpu::FragmentType>(store.getValid().getType())))
        return store.emitOpError(
            "cuTile scalar store requires scalar indices and validity");
      auto replacement = builder.create<ScalarStoreOp>(
          store.getLoc(), store.getResource(), *indices, store.getValue(),
          store.getValid());
      if (Attribute origin = store->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      store.erase();
      continue;
    }
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(store);
    auto computationType =
        cast<gpu::FragmentType>(store.getValue().getType());
    FailureOr<NativeTileAccessPlan> plan = analyzeNativeTileAccess(
        store, kernel, analysis, view, store.getCoordinates(),
        store.getSourceAxes(), computationType);
    FailureOr<SmallVector<Value>> indices = failure();
    if (succeeded(plan) && boundaryCompatible(*plan, boundary))
      indices = materializeTileIndices(builder, store, *plan);
    Operation *replacementOperation = nullptr;
    Operation *relationOperation = nullptr;
    if (succeeded(indices)) {
      Value nativeValue = store.getValue();
      if (!isIdentityPermutation(plan->toResource)) {
        auto transpose = builder.create<gpu::TransposeOp>(
            store.getLoc(), plan->resourceType, nativeValue,
            plan->toResource);
        nativeValue = transpose.getResult();
        relationOperation = transpose;
      }
      replacementOperation = builder.create<TileStoreOp>(
          store.getLoc(), store.getResource(), *indices, nativeValue);
    } else {
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
          store, store.getResource(), store.getCoordinates(), store.getSourceAxes());
      if (failed(coordinates))
        return failure();
      auto target = cast<gpu::FragmentType>(store.getValue().getType());
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, store, *coordinates, target);
      if (failed(materialized))
        return failure();
      replacementOperation = builder.create<ScatterStoreOp>(
          store.getLoc(), store.getResource(), *materialized, store.getValue(),
          store.getValid(), identityAxes(view.getRank()));
    }
    if (Attribute origin = store->getAttr(gpu::originAttr))
      replacementOperation->setAttr(gpu::originAttr, origin);
    if (relationOperation)
      if (Attribute origin = store->getAttr(gpu::originAttr))
        relationOperation->setAttr(gpu::originAttr, origin);
    store.erase();
  }
  for (gpu::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  return success();
}

bool isCuTileScalarType(Type type) {
  if (type.isIndex() ||
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 1 || integer.getWidth() == 8 ||
                     integer.getWidth() == 16 || integer.getWidth() == 32 ||
                     integer.getWidth() == 64);
}

LogicalResult verifyKernel(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return kernel.emitError(
        "cuTile provider currently requires one explicit linear program space");
  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<gpu::LoadOp, gpu::StoreOp, gpu::ContractOp>(operation)) {
      operation->emitOpError(
          "was not converted to an explicit cuTile tile/MMA form");
      return WalkResult::interrupt();
    }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isCuTileScalarType(fragment.getElementType())) {
          operation->emitOpError("contains a fragment dtype outside cuTile");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType, gpu::RecordType>(type) &&
                 !isCuTileScalarType(type)) {
        operation->emitOpError("has a result type outside the cuTile surface");
        return WalkResult::interrupt();
      }
    }
    if (isa<TileLoadOp, TileStoreOp, ScalarLoadOp, ScalarStoreOp, GatherLoadOp,
            ScatterStoreOp, AtomicRMWOp, ExtractScalarOp, MMAOp, ScaledMMAOp,
            ReduceOp, ScanOp, gpu::ReduceOp, gpu::ParameterOp,
            gpu::PhysicalExprOp, gpu::ProgramIdOp, gpu::WorksetCoordinateOp,
            gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::YieldOp, arith::ConstantOp,
            scf::ForOp, scf::IfOp, scf::YieldOp, func::FuncOp,
            func::ReturnOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed cuTile provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult verifyCuTileProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) || failed(mlir::verify(module)) ? failure()
                                                       : verifyKernel(*kernel);
}

LogicalResult legalizeGPUProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(formNativeTiles(*kernel)) ||
      failed(verifyCuTileProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::cutile

#include "NativeAccess.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

using namespace mlir;
namespace intent::cutile {
bool isProvably(Value value, int64_t expected) {
  std::optional<int64_t> actual = gpu::IndexRelations().constant(value);
  return actual && *actual == expected;
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
    if (auto transpose = fill.getDefiningOp<gpu::TransposeOp>()) {
      fill = transpose.getValue();
      continue;
    }
    if (auto reshape = fill.getDefiningOp<gpu::ReshapeOp>()) {
      fill = reshape.getValue();
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

bool isUnitExtent(Attribute attribute) {
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  return extent &&
         extent.getKind() ==
             gpu::PhysicalExprKind::Constant &&
         extent.getValue() == 1;
}

bool collectTileCoordinates(Value value, NativeTileAxisPlan &axis,
                            int64_t scale = 1) {
  if (auto range = value.getDefiningOp<gpu::MakeRangeOp>()) {
    if (llvm::any_of(axis.ranges, [&](auto term) { return term.first == range; }))
      return false;
    axis.ranges.emplace_back(range, scale);
    return true;
  }
  if (Value scalar = uniformScalarFill(value)) {
    if (!scalar.getType().isIndex())
      return false;
    axis.offsets.emplace_back(scalar, scale);
    return true;
  }
  if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
    return collectTileCoordinates(broadcast.getValue(), axis, scale);
  if (auto reshape = value.getDefiningOp<gpu::ReshapeOp>())
    return collectTileCoordinates(reshape.getValue(), axis, scale);
  if (auto transpose = value.getDefiningOp<gpu::TransposeOp>())
    return collectTileCoordinates(transpose.getValue(), axis, scale);
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary)
    return false;
  if (binary.getOperatorKind() == BinaryOperator::Add)
    return collectTileCoordinates(binary.getLhs(), axis, scale) &&
           collectTileCoordinates(binary.getRhs(), axis, scale);
  if (binary.getOperatorKind() == BinaryOperator::Multiply) {
    for (unsigned i = 0; i != 2; ++i) {
      Value scalar = uniformScalarFill(binary->getOperand(i));
      auto factor = scalar ? gpu::IndexRelations().constant(scalar) : std::nullopt;
      int64_t product;
      if (factor && *factor > 0 && !llvm::MulOverflow(scale, *factor, product))
        return collectTileCoordinates(binary->getOperand(1 - i), axis, product);
    }
  }
  return false;
}

Value stripIndexIdentities(Value value) {
  while (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
      if (isProvably(binary.getLhs(), 0)) {
        value = binary.getRhs();
        continue;
      }
      if (isProvably(binary.getRhs(), 0)) {
        value = binary.getLhs();
        continue;
      }
      break;
    case BinaryOperator::Subtract:
      if (isProvably(binary.getRhs(), 0)) {
        value = binary.getLhs();
        continue;
      }
      break;
    case BinaryOperator::Multiply:
      if (isProvably(binary.getLhs(), 1)) {
        value = binary.getRhs();
        continue;
      }
      if (isProvably(binary.getRhs(), 1)) {
        value = binary.getLhs();
        continue;
      }
      break;
    case BinaryOperator::FloorDivide:
      if (isProvably(binary.getRhs(), 1)) {
        value = binary.getLhs();
        continue;
      }
      break;
    default:
      break;
    }
    break;
  }
  return value;
}

std::optional<int64_t> dimensionIdentity(BlockArgument argument) {
  auto kernel = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!kernel || argument.getOwner() != &kernel.getBody().front())
    return std::nullopt;
  auto binding = gpu::getArgumentBinding(argument);
  if (!binding || binding.getKind() != gpu::ArgumentKind::Dimension)
    return std::nullopt;
  return binding.getDimension().getInt();
}

FailureOr<SmallVector<Value>> uniformAlignmentFactors(
    Value value, Value divisor, unsigned depth) {
  if (!value || depth >= 32)
    return failure();
  value = stripIndexIdentities(value);
  divisor = stripIndexIdentities(divisor);
  if (gpu::IndexRelations().multipleOf(value, divisor))
    return SmallVector<Value>{};
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    if (!integer)
      return failure();
    return integer.getValue().isZero() ? SmallVector<Value>{}
                                       : SmallVector<Value>{value};
  }
  if (value.getDefiningOp<gpu::ParameterOp>() ||
      value.getDefiningOp<gpu::DimOp>() ||
      value.getDefiningOp<gpu::PhysicalExprOp>())
    return SmallVector<Value>{value};
  auto combine = [&](Value lhs, Value rhs) -> FailureOr<SmallVector<Value>> {
    auto left = uniformAlignmentFactors(lhs, divisor, depth + 1);
    auto right = uniformAlignmentFactors(rhs, divisor, depth + 1);
    if (failed(left) || failed(right))
      return failure();
    for (Value factor : *right)
      if (!llvm::is_contained(*left, factor))
        left->push_back(factor);
    return std::move(*left);
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (dimensionIdentity(argument))
      return SmallVector<Value>{value};
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (loop && argument == loop.getInductionVar() &&
        gpu::IndexRelations().positive(loop.getStep()))
      return combine(loop.getLowerBound(), loop.getStep());
    return failure();
  }
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary)
    return failure();
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::Subtract:
  case BinaryOperator::Minimum:
  case BinaryOperator::Maximum:
    return combine(binary.getLhs(), binary.getRhs());
  case BinaryOperator::Multiply: {
    auto lhs = uniformAlignmentFactors(binary.getLhs(), divisor, depth + 1);
    auto rhs = uniformAlignmentFactors(binary.getRhs(), divisor, depth + 1);
    if (succeeded(lhs) && (failed(rhs) || lhs->size() <= rhs->size()))
      return std::move(*lhs);
    return rhs;
  }
  default:
    return failure();
  }
}

bool isAlignedPeriodicTile(Value start, Value extent, Value period) {
  if (!gpu::IndexRelations().powerOfTwo(extent) || !gpu::IndexRelations().positive(period) ||
      !gpu::IndexRelations().multipleOf(period, extent))
    return false;
  auto factors = uniformAlignmentFactors(start, extent);
  // A power-of-two aligned origin plus extent-1 cannot cross signed index
  // wraparound. Dividing the period into complete tiles keeps one quotient.
  return succeeded(factors) && factors->empty();
}

scf::ForOp completeAlignedTileLoop(Value start, Value extent) {
  auto induction = dyn_cast<BlockArgument>(stripIndexIdentities(start));
  auto loop = induction
                  ? dyn_cast_or_null<scf::ForOp>(
                        induction.getOwner()->getParentOp())
                  : scf::ForOp();
  if (!loop || induction != loop.getInductionVar() ||
      !gpu::IndexRelations().powerOfTwo(extent) ||
      !gpu::samePhysicalScalarExpression(loop.getStep(), extent) ||
      !gpu::queryNonNegativeIndexUpperBound(loop.getLowerBound()) ||
      !gpu::IndexRelations().multipleOf(loop.getLowerBound(), extent) ||
      !gpu::IndexRelations().multipleOf(loop.getUpperBound(), extent))
    return {};
  // Both endpoints are tile aligned. Every executing iteration is nonnegative
  // and complete; the final increment reaches the representable upper bound.
  return loop;
}

bool valueIsDimension(Value value, int64_t dimension, func::FuncOp kernel) {
  value = stripIndexIdentities(value);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner() != &kernel.getBody().front())
      return false;
    std::optional<int64_t> identity = dimensionIdentity(argument);
    return identity && *identity == dimension;
  }
  if (auto dim = value.getDefiningOp<gpu::DimOp>()) {
    ArrayRef<int64_t> dimensions =
        dim.getView().getType().getLayout().getDimensionIds().asArrayRef();
    return dim.getAxis() < dimensions.size() &&
           dimensions[dim.getAxis()] == dimension;
  }
  return false;
}

Value dimensionValue(func::FuncOp kernel, int64_t dimension) {
  Value result;
  for (BlockArgument argument : kernel.getArguments()) {
    std::optional<int64_t> identity = dimensionIdentity(argument);
    if (!identity || *identity != dimension)
      continue;
    if (result && result != argument)
      return {};
    result = argument;
  }
  return result;
}

bool isLogicalSubregionDistance(Value value, gpu::MakeRangeOp range) {
  value = stripIndexIdentities(value);
  auto distance = value.getDefiningOp<gpu::BinaryOp>();
  if (distance && distance.getOperatorKind() == BinaryOperator::Subtract &&
      gpu::samePhysicalScalarExpression(distance.getLhs(),
                                        range.getLogicalStop()) &&
      gpu::samePhysicalScalarExpression(distance.getRhs(),
                                        range.getLogicalStart()))
    return true;
  auto multiply = value.getDefiningOp<gpu::BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  auto flooredDistance = [&](Value quotient, Value divisor) {
    auto division = quotient.getDefiningOp<gpu::BinaryOp>();
    return division &&
           division.getOperatorKind() == BinaryOperator::FloorDivide &&
           gpu::samePhysicalScalarExpression(division.getRhs(), divisor) &&
           gpu::IndexRelations().positive(divisor) &&
           isLogicalSubregionDistance(division.getLhs(), range);
  };
  return flooredDistance(multiply.getLhs(), multiply.getRhs()) ||
         flooredDistance(multiply.getRhs(), multiply.getLhs());
}

bool rangeStartsInsideLogicalSubregion(gpu::MakeRangeOp range) {
  Value start = stripIndexIdentities(range.getStart());
  if (gpu::samePhysicalScalarExpression(start, range.getLogicalStart()))
    return true;
  auto add = start.getDefiningOp<gpu::BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  auto check = [&](Value base, Value offset) {
    if (!gpu::samePhysicalScalarExpression(base, range.getLogicalStart()))
      return false;
    auto argument = dyn_cast<BlockArgument>(stripIndexIdentities(offset));
    auto loop =
        argument
            ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
            : scf::ForOp();
    return loop && argument == loop.getInductionVar() &&
           gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
           gpu::IndexRelations().positive(loop.getStep()) &&
           isLogicalSubregionDistance(loop.getUpperBound(), range);
  };
  return check(add.getLhs(), add.getRhs()) ||
         check(add.getRhs(), add.getLhs());
}

bool isBlockedWorksetOrigin(Value value, Value block, int64_t dimension) {
  auto multiply = value.getDefiningOp<gpu::BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  Value coordinate;
  if (multiply.getLhs() == block)
    coordinate = multiply.getRhs();
  else if (multiply.getRhs() == block)
    coordinate = multiply.getLhs();
  else
    return false;
  auto mapping = gpu::queryDecodedCoordinate(coordinate);
  if (!mapping || mapping->axis >= mapping->extents.size())
    return false;
  auto expression = gpu::queryLaunchExpression(mapping->extents[mapping->axis]);
  auto parameter = block.getDefiningOp<gpu::ParameterOp>();
  if (!expression || !parameter || expression.getOperands().size() != 2)
    return false;
  auto isBlock = [&](gpu::PhysicalExprAttr operand) {
    return operand.getKind() == gpu::PhysicalExprKind::Parameter &&
           operand.getParameterReference() == parameter.getReference();
  };
  auto isDimension = [&](gpu::PhysicalExprAttr operand) {
    return operand.getKind() == gpu::PhysicalExprKind::Dimension &&
           operand.getValue() == dimension;
  };
  auto numerator = cast<gpu::PhysicalExprAttr>(expression.getOperands()[0]);
  auto denominator = cast<gpu::PhysicalExprAttr>(expression.getOperands()[1]);
  if (!isBlock(denominator)) return false;
  if (expression.getKind() == gpu::PhysicalExprKind::CeilDiv)
    return isDimension(numerator);
  // Ownership formation spells the runtime tile count as (D + (P - 1)) // P.
  // Read that current SSA relation after execution-group lowering, rather than
  // attaching the old launch schema to a pure coordinate calculation.
  if (expression.getKind() != gpu::PhysicalExprKind::FloorDiv ||
      numerator.getKind() != gpu::PhysicalExprKind::Add ||
      numerator.getOperands().size() != 2)
    return false;
  auto adjusted = [&](gpu::PhysicalExprAttr logical,
                      gpu::PhysicalExprAttr padding) {
    return isDimension(logical) &&
           padding.getKind() == gpu::PhysicalExprKind::Subtract &&
           padding.getOperands().size() == 2 &&
           isBlock(cast<gpu::PhysicalExprAttr>(padding.getOperands()[0])) &&
           gpu::constantPhysicalExpression(
               cast<gpu::PhysicalExprAttr>(padding.getOperands()[1])) == 1;
  };
  auto lhs = cast<gpu::PhysicalExprAttr>(numerator.getOperands()[0]);
  auto rhs = cast<gpu::PhysicalExprAttr>(numerator.getOperands()[1]);
  return adjusted(lhs, rhs) || adjusted(rhs, lhs);
}

std::optional<int64_t> constantTileOrigin(gpu::MakeRangeOp range,
                                          ArrayRef<Value> offsets) {
  std::optional<int64_t> result = gpu::IndexRelations().constant(range.getStart());
  if (!result)
    return std::nullopt;
  for (Value offset : offsets) {
    std::optional<int64_t> constant = gpu::IndexRelations().constant(offset);
    if (!constant)
      return std::nullopt;
    int64_t sum;
    if (llvm::AddOverflow(*result, *constant, sum))
      return std::nullopt;
    *result = sum;
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
      gpu::IndexRelations().constant(extent, kernel);
  return constant && origin < *constant;
}

bool rangeOriginInView(gpu::MakeRangeOp range, ArrayRef<Value> offsets,
                       gpu::ViewType view, unsigned resourceAxis,
                       int64_t dimension, func::FuncOp kernel) {
  bool zeroOffset = llvm::all_of(
      offsets, [](Value offset) { return isProvably(offset, 0); });
  Value logicalDimension = dimensionValue(kernel, dimension);
  if (zeroOffset && range->hasAttr(gpu::sourceSubregionAttr) &&
      logicalDimension && gpu::IndexRelations().nonnegative(range.getLogicalStart()) &&
      gpu::IndexRelations().atMost(range.getLogicalStop(), logicalDimension) &&
      rangeStartsInsideLogicalSubregion(range))
    return true;
  if (zeroOffset && range->hasAttr(gpu::programBoundedOriginAttr) &&
      !range->hasAttr(gpu::sourceSubregionAttr) &&
      isProvably(range.getLogicalStart(), 0) &&
      valueIsDimension(range.getLogicalStop(), dimension, kernel))
    return true;
  if (zeroOffset &&
      valueIsDimension(range.getLogicalStop(), dimension, kernel)) {
    Value start = stripIndexIdentities(range.getStart());
    if (isProvably(range.getLogicalStart(), 0) &&
        isBlockedWorksetOrigin(start, range.getExtent(), dimension))
      return true;
    if (auto coordinate = start.getDefiningOp<gpu::WorksetCoordinateOp>())
      if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
        return true;
    if (auto argument = dyn_cast<BlockArgument>(start)) {
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop && argument == loop.getInductionVar() &&
          isProvably(range.getLogicalStart(), 0) &&
          gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
          gpu::IndexRelations().atMost(loop.getUpperBound(), range.getLogicalStop()))
        return true;
    }
    auto parameter = range.getExtent().getDefiningOp<gpu::ParameterOp>();
    if (isProvably(start, 0) && isProvably(range.getLogicalStart(), 0) &&
        parameter &&
        parameter.getDeclaration().getRole() ==
            gpu::ParameterRole::FullCoverage) {
      gpu::PhysicalParameterBinding binding =
          gpu::queryParameterBinding(parameter.getDeclaration());
      if (binding.isExact() && binding.dimension &&
          *binding.dimension == dimension)
        return true;
    }
  }
  std::optional<int64_t> origin = constantTileOrigin(range, offsets);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarOriginInView(Value index, gpu::ViewType view,
                        unsigned resourceAxis, int64_t dimension,
                        func::FuncOp kernel) {
  index = stripIndexIdentities(index);
  if (auto coordinate = index.getDefiningOp<gpu::WorksetCoordinateOp>())
    if (coordinate.getDimensionId() == static_cast<uint64_t>(dimension))
      return true;
  if (auto argument = dyn_cast<BlockArgument>(index)) {
    auto loop =
        dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    Value logicalDimension = dimensionValue(kernel, dimension);
    if (loop && argument == loop.getInductionVar() && logicalDimension &&
        gpu::IndexRelations().nonnegative(loop.getLowerBound()) &&
        gpu::IndexRelations().positive(loop.getStep()) &&
        gpu::IndexRelations().atMost(loop.getUpperBound(), logicalDimension))
      return true;
  }
  std::optional<int64_t> origin = gpu::IndexRelations().constant(index);
  return origin &&
         constantOriginInView(*origin, view, resourceAxis, kernel);
}

bool scalarCoordinatesInView(ValueRange coordinates, gpu::ViewType view,
                             func::FuncOp kernel) {
  ArrayRef<int64_t> dimensions =
      view.getLayout().getDimensionIds().asArrayRef();
  if (coordinates.size() != view.getRank() ||
      dimensions.size() != view.getRank())
    return false;
  for (auto [axis, coordinate] : llvm::enumerate(coordinates))
    if (!scalarOriginInView(coordinate, view, axis, dimensions[axis], kernel))
      return false;
  return true;
}

FailureOr<unsigned> nativeAccessRangeAxis(gpu::AccessOpInterface access,
                                         unsigned coordinateIndex,
                                         gpu::MakeRangeOp range) {
  auto projection = gpu::queryAccessCoordinateProjection(access, coordinateIndex);
  auto coordinate = dyn_cast<gpu::FragmentType>(
      access.getAccessCoordinates()[coordinateIndex].getType());
  if (!projection.isExact() || !coordinate)
    return failure();
  auto ranges = gpu::queryRangeProjections(coordinate, range);
  if (ranges.size() != 1)
    return failure();
  std::optional<unsigned> result;
  for (auto [targetAxis, sourceAxis] : llvm::enumerate(projection.targetToSource)) {
    if (!sourceAxis || *sourceAxis != ranges.front().fragmentAxis)
      continue;
    if (result)
      return failure();
    result = targetAxis;
  }
  if (result)
    return *result;
  return failure();
}

FailureOr<NativeTileAccessPlan> analyzeNativeTileAccess(
    gpu::AccessOpInterface access, func::FuncOp kernel,
    const gpu::PhysicalAccessBoundsFact &accessBounds) {
  Operation *owner = access.getOperation();
  auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
  auto computationType = dyn_cast<gpu::FragmentType>(access.getAccessValueType());
  if (!view || !computationType) return failure();
  auto coordinates = access.getAccessCoordinates();
  auto sourceAxes = access.getAccessSourceAxes();
  const unsigned resourceRank = view.getRank();
  const unsigned computationRank = computationType.getShape().size();
  if (coordinates.size() != resourceRank ||
      sourceAxes.size() != resourceRank || computationRank == 0)
    return failure();

  SmallVector<Value> resourceCoordinates(resourceRank);
  SmallVector<unsigned> coordinateSlots(resourceRank);
  for (auto [slot, coordinate] : llvm::enumerate(coordinates)) {
    int64_t sourceAxis = sourceAxes[slot];
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(resourceRank) ||
        resourceCoordinates[sourceAxis])
      return failure();
    resourceCoordinates[sourceAxis] = coordinate;
    coordinateSlots[sourceAxis] = slot;
  }

  NativeTileAccessPlan plan;
  plan.axes.resize(resourceRank);
  plan.toComputation.assign(computationRank, -1);
  plan.toResource.assign(computationRank, -1);
  SmallVector<bool> usedComputationAxis(computationRank, false);
  SmallVector<unsigned> unresolvedScalarAxes;
  ArrayRef<int64_t> dimensions =
      view.getLayout().getDimensionIds().asArrayRef();
  if (dimensions.size() != resourceRank)
    return failure();

  auto assignComputationAxis = [&](unsigned resourceAxis,
                                   unsigned computationAxis) {
    if (computationAxis >= computationRank ||
        usedComputationAxis[computationAxis])
      return false;
    usedComputationAxis[computationAxis] = true;
    plan.axes[resourceAxis].computationAxes.push_back(computationAxis);
    return true;
  };

  for (unsigned resourceAxis = 0; resourceAxis < resourceRank;
       ++resourceAxis) {
    NativeTileAxisPlan &axis = plan.axes[resourceAxis];
    Value coordinate = resourceCoordinates[resourceAxis];
    Value scalar = uniformScalarFill(coordinate);
    if (!scalar) {
      Value expression = coordinate;
      while (isa_and_nonnull<gpu::BroadcastOp, gpu::ReshapeOp, gpu::TransposeOp>(
          expression.getDefiningOp()))
        expression = expression.getDefiningOp()->getOperand(0);
      if (auto binary = expression.getDefiningOp<gpu::BinaryOp>()) {
        Value rhs = uniformScalarFill(binary.getRhs());
        auto constant = rhs ? gpu::IndexRelations().constant(rhs) : std::nullopt;
        auto kind = binary.getOperatorKind();
        int64_t divisor = 0;
        if (kind == BinaryOperator::FloorDivide && constant && *constant > 0)
          divisor = *constant;
        if (kind == BinaryOperator::RightShift && constant &&
            *constant >= 0 && *constant < 63)
          divisor = int64_t{1} << *constant;
        if (divisor) {
          if (!collectTileCoordinates(binary.getLhs(), axis) ||
              axis.ranges.size() != 1 || axis.ranges.front().second != 1 ||
              !axis.offsets.empty() || !gpu::isUnitStepRange(axis.ranges.front().first))
            return failure();
          // The quotient is a unit resource axis only under the per-tile
          // uniformity guard emitted with its native indices.
          axis.scalarIndex = axis.ranges.front().first.getStart();
          axis.divisor = divisor;
          axis.originInBounds =
              llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
          auto loop = completeAlignedTileLoop(
              axis.scalarIndex, axis.ranges.front().first.getExtent());
          auto upper = loop ? gpu::queryLaunchExpression(loop.getUpperBound())
                            : gpu::PhysicalExprAttr();
          if (upper && upper.getKind() ==
                           gpu::PhysicalExprKind::Multiply &&
              upper.getOperands().size() == 2) {
            auto covers = [&](Attribute dimension, Attribute factor) {
              auto dim = cast<gpu::PhysicalExprAttr>(dimension);
              auto scale = cast<gpu::PhysicalExprAttr>(factor);
              return dim.getKind() ==
                         gpu::PhysicalExprKind::Dimension &&
                     dim.getValue() == dimensions[resourceAxis] &&
                     scale.getKind() ==
                         gpu::PhysicalExprKind::Constant &&
                     scale.getValue() == divisor;
            };
            axis.originInBounds |= covers(upper.getOperands()[0], upper.getOperands()[1]) ||
                                   covers(upper.getOperands()[1], upper.getOperands()[0]);
          }
          continue;
        }
        if (kind == BinaryOperator::Remainder) {
          if (!rhs || !rhs.getType().isIndex())
            return failure();
          axis.modulus = rhs;
          coordinate = binary.getLhs();
        } else if (kind == BinaryOperator::BitwiseAnd) {
          if (!constant)
            return failure();
          const int64_t mask = constant.value();
          int64_t period;
          if (mask < 0 || llvm::AddOverflow(mask, int64_t{1}, period) ||
              !llvm::isPowerOf2_64(period))
            return failure();
          axis.modulus = IntegerAttr::get(IndexType::get(owner->getContext()), period);
          coordinate = binary.getLhs();
        }
      }
      if (!collectTileCoordinates(coordinate, axis) || axis.ranges.empty())
        return failure();
      llvm::stable_sort(axis.ranges, [](auto lhs, auto rhs) {
        return lhs.second > rhs.second;
      });
      for (auto [range, stride] : axis.ranges) {
        auto resultAxis = nativeAccessRangeAxis(access, coordinateSlots[resourceAxis], range);
        auto rangeType = dyn_cast<gpu::FragmentType>(range.getResult().getType());
        if (failed(resultAxis) || !assignComputationAxis(resourceAxis, *resultAxis) ||
            !gpu::isUnitStepRange(range) || !rangeType ||
            rangeType.getShape().size() != 1 ||
            rangeType.getShape()[0] !=
                computationType.getShape()[axis.computationAxes.back()])
          return failure();
      }
      axis.originInBounds =
          llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
      if (axis.modulus && axis.ranges.size() == 1 &&
          axis.ranges.front().second == 1 && axis.offsets.empty()) {
        auto range = axis.ranges.front().first;
        std::optional<int64_t> modulus;
        if (auto value = dyn_cast<Value>(axis.modulus))
          modulus = gpu::IndexRelations().constant(value);
        else
          modulus = cast<IntegerAttr>(cast<Attribute>(axis.modulus)).getInt();
        if (modulus && *modulus > 0 &&
            completeAlignedTileLoop(range.getStart(), range.getExtent()))
          axis.originInBounds |= constantOriginInView(
              *modulus - 1, view, resourceAxis, kernel);
      }
      if (!axis.modulus && axis.ranges.size() == 1 &&
          axis.ranges.front().second == 1 &&
          llvm::all_of(axis.offsets, [](auto term) { return term.second == 1; })) {
        SmallVector<Value> offsets;
        for (auto term : axis.offsets)
          offsets.push_back(term.first);
        axis.originInBounds |= rangeOriginInView(
            axis.ranges.front().first, offsets, view, resourceAxis,
            dimensions[resourceAxis], kernel);
      }
      continue;
    }

    if (!scalar || !scalar.getType().isIndex())
      return failure();
    axis.scalarIndex = scalar;
    axis.originInBounds =
        scalarOriginInView(scalar, view, resourceAxis,
                           dimensions[resourceAxis], kernel) ||
        llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
    Value strippedScalar = stripIndexIdentities(scalar);
    if (auto workset =
            strippedScalar.getDefiningOp<gpu::WorksetCoordinateOp>()) {
      gpu::PhysicalDimensionProjection projection =
          gpu::queryFragmentDimension(computationType,
                                      workset.getDimensionId());
      if (projection.state == gpu::PhysicalFactState::Ambiguous)
        return failure();
      if (projection.isExact()) {
        if (!assignComputationAxis(resourceAxis, projection.fragmentAxis) ||
            !isUnitExtent(
                computationType.getShape()[axis.computationAxes.back()]))
          return failure();
      }
    } else {
      unresolvedScalarAxes.push_back(resourceAxis);
    }
  }

  SmallVector<unsigned> remainingUnitAxes;
  for (unsigned computationAxis = 0; computationAxis < computationRank;
       ++computationAxis)
    if (!usedComputationAxis[computationAxis] &&
        isUnitExtent(computationType.getShape()[computationAxis]))
      remainingUnitAxes.push_back(computationAxis);
  if (unresolvedScalarAxes.size() == 1 && remainingUnitAxes.size() == 1) {
    unsigned resourceAxis = unresolvedScalarAxes.front();
    unsigned computationAxis = remainingUnitAxes.front();
    if (!assignComputationAxis(resourceAxis, computationAxis))
      return failure();
    NativeTileAxisPlan &axis = plan.axes[resourceAxis];
    axis.originInBounds =
        scalarOriginInView(axis.scalarIndex, view, resourceAxis,
                           dimensions[resourceAxis], kernel) ||
        llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
  }

  if (llvm::any_of(usedComputationAxis, [](bool used) { return !used; }))
    return failure();

  MLIRContext *context = owner->getContext();
  auto unit = gpu::PhysicalExprAttr::get(
      context, gpu::PhysicalExprKind::Constant, 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  SmallVector<Attribute> resourceShape(resourceRank);
  SmallVector<Attribute> resourceMappings(resourceRank);
  SmallVector<Attribute> packedShape;
  SmallVector<Attribute> packedMappings;
  packedShape.reserve(computationRank);
  packedMappings.reserve(computationRank);
  for (unsigned resourceAxis = 0; resourceAxis < resourceRank;
       ++resourceAxis) {
    ArrayRef<unsigned> computationAxes = plan.axes[resourceAxis].computationAxes;
    if (computationAxes.empty()) {
      resourceShape[resourceAxis] = unit;
      resourceMappings[resourceAxis] = gpu::AxisMapAttr::get(
          context, view.getSourceId(), resourceAxis,
          dimensions[resourceAxis], resourceAxis, false);
      continue;
    }
    Attribute extent = unit;
    auto mapping = cast<gpu::AxisMapAttr>(
        computationType.getAxisMaps()[computationAxes.front()]);
    for (unsigned computationAxis : computationAxes) {
      Attribute component = computationType.getShape()[computationAxis];
      if (isUnitExtent(extent))
        extent = component;
      else if (!isUnitExtent(component))
        extent = gpu::PhysicalExprAttr::get(
            context, gpu::PhysicalExprKind::Multiply, 0,
            StringAttr::get(context), ArrayAttr::get(context, {extent, component}));
      auto componentMap = cast<gpu::AxisMapAttr>(
          computationType.getAxisMaps()[computationAxis]);
      unsigned packedAxis = packedShape.size();
      packedShape.push_back(component);
      packedMappings.push_back(gpu::AxisMapAttr::get(
          context, componentMap.getSourceId(), componentMap.getSourceAxis(),
          componentMap.getDimensionId(), packedAxis, componentMap.getDerived()));
      plan.toComputation[computationAxis] = packedAxis;
      plan.toResource[packedAxis] = computationAxis;
    }
    resourceShape[resourceAxis] = extent;
    resourceMappings[resourceAxis] = gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resourceAxis,
        mapping.getDerived() || computationAxes.size() > 1);
  }

  plan.resourceType = gpu::FragmentType::get(
      context, computationType.getElementType(),
      ArrayAttr::get(context, resourceShape),
      ArrayAttr::get(context, resourceMappings), computationType.getValidity(),
      computationType.getOwner());
  plan.packedType = gpu::FragmentType::get(
      context, computationType.getElementType(),
      ArrayAttr::get(context, packedShape),
      ArrayAttr::get(context, packedMappings),
      computationType.getValidity(), computationType.getOwner());
  if (plan.resourceType != plan.packedType) {
    FailureOr<ArrayAttr> resourceToPacked =
        gpu::inferReshapeReassociation(plan.resourceType, plan.packedType);
    FailureOr<ArrayAttr> packedToResource =
        gpu::inferReshapeReassociation(plan.packedType, plan.resourceType);
    if (failed(resourceToPacked) || failed(packedToResource))
      return failure();
    plan.resourceToPacked = *resourceToPacked;
    plan.packedToResource = *packedToResource;
  }
  return plan;
}


} // namespace intent::cutile

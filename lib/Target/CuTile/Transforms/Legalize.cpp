#include "Intent/Target/CuTile/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;

namespace intent::cutile {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_cutile.legalized";
constexpr llvm::StringLiteral accessFormParameter = "CUTILE_ACCESS_FORM";
constexpr llvm::StringLiteral occupancyParameter = "CUTILE_OCCUPANCY";
constexpr int64_t nativeAccessForm = 1;
constexpr int64_t gatherAccessForm = 2;
constexpr int64_t nativeBlockedNoTMAForm = 3;

bool isLegalAccessForm(int64_t value) {
  return value == nativeAccessForm || value == gatherAccessForm ||
         value == nativeBlockedNoTMAForm;
}

bool isLegalOccupancy(int64_t value) { return value >= 1 && value <= 32; }

bool supportsE8M0ScaledMMA(gpu::CapabilitiesAttr capabilities) {
  return capabilities && capabilities.getComputeCapabilityMajor() >= 10;
}

bool isCuTileProviderRole(gpu::ParameterRole role) {
  return role == gpu::ParameterRole::ProviderAccessForm ||
         role == gpu::ParameterRole::ProviderOccupancy;
}

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

bool valueIsMultipleOf(Value value, Value divisor, unsigned depth = 0);

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
    if (binary.getOperatorKind() == BinaryOperator::Subtract &&
        isProvably(binary.getRhs(), 0))
      return tileIndex(builder, location, binary.getLhs(), extent);
  }
  auto argument = dyn_cast<BlockArgument>(start);
  auto loop =
      argument
          ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
          : scf::ForOp();
  if (loop && argument == loop.getInductionVar() &&
      gpu::samePhysicalScalarExpression(loop.getStep(), extent) &&
      valueIsMultipleOf(loop.getLowerBound(), extent))
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

FailureOr<SmallVector<unsigned>>
coordinateTargetAxes(gpu::FragmentType source, gpu::FragmentType target) {
  if (source.getOwner() != target.getOwner() ||
      source.getShape().size() > target.getShape().size())
    return failure();
  SmallVector<unsigned> result;
  SmallVector<bool> usedTargetAxes(target.getShape().size(), false);
  result.reserve(source.getShape().size());
  for (auto [sourceIndex, sourceAttribute] :
       llvm::enumerate(source.getAxisMaps())) {
    auto sourceMap = cast<gpu::AxisMapAttr>(sourceAttribute);
    std::optional<unsigned> targetIndex;
    for (auto [index, targetAttribute] :
         llvm::enumerate(target.getAxisMaps())) {
      auto targetMap = cast<gpu::AxisMapAttr>(targetAttribute);
      if (usedTargetAxes[index] ||
          sourceMap.getSourceId() != targetMap.getSourceId() ||
          sourceMap.getSourceAxis() != targetMap.getSourceAxis() ||
          sourceMap.getDimensionId() != targetMap.getDimensionId() ||
          sourceMap.getDerived() != targetMap.getDerived())
        continue;
      if (targetIndex)
        return failure();
      targetIndex = index;
    }
    if (!targetIndex)
      return failure();
    Attribute sourceExtent = source.getShape()[sourceIndex];
    Attribute targetExtent = target.getShape()[*targetIndex];
    auto constant = dyn_cast<gpu::PhysicalExprAttr>(sourceExtent);
    bool unit = constant &&
                constant.getKind() == static_cast<uint32_t>(
                                          gpu::PhysicalExprKind::Constant) &&
                constant.getValue() == 1;
    if (!unit && sourceExtent != targetExtent)
      return failure();
    usedTargetAxes[*targetIndex] = true;
    result.push_back(*targetIndex);
  }
  return result;
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
    FailureOr<SmallVector<unsigned>> targetAxes =
        coordinateTargetAxes(source, target);
    if (failed(targetAxes))
      return owner->emitOpError(
          "cuTile coordinate cannot adopt the selected physical domain");
    auto preserveOrigin = [&](Operation *operation) {
      if (Operation *definition = coordinate.getDefiningOp())
        if (Attribute origin = definition->getAttr(gpu::originAttr))
          operation->setAttr(gpu::originAttr, origin);
    };

    SmallVector<int64_t> permutation;
    permutation.reserve(source.getShape().size());
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
      permutation.push_back(axis);
    llvm::sort(permutation, [&](int64_t lhs, int64_t rhs) {
      return (*targetAxes)[lhs] < (*targetAxes)[rhs];
    });
    bool identityPermutation = llvm::all_of(
        llvm::enumerate(permutation), [](auto item) {
          return static_cast<int64_t>(item.index()) == item.value();
        });
    if (!identityPermutation) {
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      SmallVector<unsigned> reorderedTargetAxes;
      shape.reserve(permutation.size());
      mappings.reserve(permutation.size());
      reorderedTargetAxes.reserve(permutation.size());
      for (auto [resultAxis, sourceAxis] : llvm::enumerate(permutation)) {
        shape.push_back(source.getShape()[sourceAxis]);
        auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        mappings.push_back(gpu::AxisMapAttr::get(
            owner->getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), resultAxis, mapping.getDerived()));
        reorderedTargetAxes.push_back((*targetAxes)[sourceAxis]);
      }
      auto transposedType = gpu::FragmentType::get(
          owner->getContext(), source.getElementType(),
          ArrayAttr::get(owner->getContext(), shape),
          ArrayAttr::get(owner->getContext(), mappings), source.getValidity(),
          source.getOwner());
      auto transpose = builder.create<gpu::TransposeOp>(
          owner->getLoc(), transposedType, coordinate, permutation);
      preserveOrigin(transpose);
      coordinate = transpose.getResult();
      source = transposedType;
      *targetAxes = std::move(reorderedTargetAxes);
    }

    auto unit = gpu::PhysicalExprAttr::get(
        owner->getContext(),
        static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
        StringAttr::get(owner->getContext()),
        ArrayAttr::get(owner->getContext(), {}));
    SmallVector<Attribute> expandedShape(target.getShape().size(), unit);
    for (auto [sourceAxis, targetAxis] : llvm::enumerate(*targetAxes))
      expandedShape[targetAxis] = source.getShape()[sourceAxis];
    auto expandedType = gpu::FragmentType::get(
        owner->getContext(), source.getElementType(),
        ArrayAttr::get(owner->getContext(), expandedShape),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    if (!samePhysicalDomain(source, expandedType)) {
      FailureOr<ArrayAttr> reassociation =
          gpu::inferReshapeReassociation(source, expandedType);
      if (failed(reassociation))
        return owner->emitOpError(
            "cuTile coordinate occurrence has no row-major domain expansion");
      auto reshape = builder.create<gpu::ReshapeOp>(
          owner->getLoc(), expandedType, coordinate, *reassociation);
      preserveOrigin(reshape);
      coordinate = reshape.getResult();
      source = expandedType;
    }
    auto resultType = gpu::FragmentType::get(
        owner->getContext(), source.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    if (samePhysicalDomain(source, resultType)) {
      results.push_back(coordinate);
      continue;
    }
    auto broadcast = builder.create<gpu::BroadcastOp>(
        owner->getLoc(), resultType, coordinate);
    preserveOrigin(broadcast);
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
  std::optional<unsigned> computationAxis;
  Value scalarIndex;
  gpu::MakeRangeOp range;
  SmallVector<Value> offsets;
  bool originInBounds = false;
};

struct NativeTileAccessPlan {
  gpu::FragmentType resourceType;
  gpu::FragmentType packedType;
  ArrayAttr resourceToPacked;
  ArrayAttr packedToResource;
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

bool isKnownPositive(Value value);

std::optional<int64_t> dimensionIdentity(BlockArgument argument) {
  auto kernel = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!kernel || argument.getOwner() != &kernel.getBody().front())
    return std::nullopt;
  DictionaryAttr attributes =
      kernel.getArgAttrDict(argument.getArgNumber());
  auto kind = attributes.getAs<StringAttr>(gpu::abiKindAttr);
  auto identity = attributes.getAs<IntegerAttr>(gpu::dimensionAttr);
  if (!kind || kind.getValue() != "dimension" || !identity)
    return std::nullopt;
  return identity.getInt();
}

bool isKnownNonNegative(Value value, unsigned depth = 0) {
  if (!value || depth >= 32)
    return false;
  value = stripIndexIdentities(value);
  if (std::optional<int64_t> constant = constantValue(value))
    return *constant >= 0;
  if (isa_and_nonnull<gpu::ProgramIdOp, gpu::WorksetCoordinateOp>(
          value.getDefiningOp()))
    return true;
  if (auto result = dyn_cast<OpResult>(value))
    if (isa<gpu::DelinearizeOp>(result.getOwner()))
      return true;
  if (auto argument = dyn_cast<BlockArgument>(value))
    if (dimensionIdentity(argument))
      return true;
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary)
    return false;
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
    return isKnownNonNegative(binary.getLhs(), depth + 1) &&
           isKnownNonNegative(binary.getRhs(), depth + 1);
  case BinaryOperator::Subtract: {
    std::optional<int64_t> rhs = constantValue(binary.getRhs());
    auto lhs = binary.getLhs().getDefiningOp<gpu::BinaryOp>();
    if (!rhs || *rhs < 0 || !lhs ||
        lhs.getOperatorKind() != BinaryOperator::Add)
      return false;
    auto covers = [&](Value constant, Value remainder) {
      std::optional<int64_t> amount = constantValue(constant);
      return amount && *amount >= *rhs &&
             isKnownNonNegative(remainder, depth + 1);
    };
    return covers(lhs.getLhs(), lhs.getRhs()) ||
           covers(lhs.getRhs(), lhs.getLhs());
  }
  case BinaryOperator::Maximum:
    return isKnownNonNegative(binary.getLhs(), depth + 1) ||
           isKnownNonNegative(binary.getRhs(), depth + 1);
  case BinaryOperator::Minimum:
  case BinaryOperator::Multiply:
    return isKnownNonNegative(binary.getLhs(), depth + 1) &&
           isKnownNonNegative(binary.getRhs(), depth + 1);
  case BinaryOperator::FloorDivide:
    return isKnownNonNegative(binary.getLhs(), depth + 1) &&
           isKnownPositive(binary.getRhs());
  default:
    return false;
  }
}

bool isKnownPositive(Value value) {
  if (std::optional<int64_t> constant = constantValue(value))
    return *constant > 0;
  auto parameter = value.getDefiningOp<gpu::ParameterOp>();
  return parameter &&
         llvm::all_of(parameter.getParameter().getCandidates().asArrayRef(),
                      [](int64_t candidate) { return candidate > 0; });
}

bool valueIsMultipleOf(Value value, Value divisor, unsigned depth) {
  if (!value || !divisor || depth >= 32 || !isKnownPositive(divisor))
    return false;
  value = stripIndexIdentities(value);
  divisor = stripIndexIdentities(divisor);
  if (gpu::samePhysicalScalarExpression(value, divisor) ||
      isProvably(value, 0))
    return true;
  std::optional<int64_t> constant = constantValue(value);
  std::optional<int64_t> divisorConstant = constantValue(divisor);
  if (constant && divisorConstant)
    return *constant % *divisorConstant == 0;
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary)
    return false;
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::Subtract:
  case BinaryOperator::Minimum:
  case BinaryOperator::Maximum:
    return valueIsMultipleOf(binary.getLhs(), divisor, depth + 1) &&
           valueIsMultipleOf(binary.getRhs(), divisor, depth + 1);
  case BinaryOperator::Multiply:
    return valueIsMultipleOf(binary.getLhs(), divisor, depth + 1) ||
           valueIsMultipleOf(binary.getRhs(), divisor, depth + 1);
  default:
    return false;
  }
}

bool valueUpperBoundedBy(Value value, Value bound, unsigned depth = 0) {
  if (!value || !bound || depth >= 32)
    return false;
  value = stripIndexIdentities(value);
  bound = stripIndexIdentities(bound);
  if (gpu::samePhysicalScalarExpression(value, bound))
    return true;
  auto binary = value.getDefiningOp<gpu::BinaryOp>();
  if (!binary)
    return false;
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Minimum:
    return valueUpperBoundedBy(binary.getLhs(), bound, depth + 1) ||
           valueUpperBoundedBy(binary.getRhs(), bound, depth + 1);
  case BinaryOperator::Maximum:
    return valueUpperBoundedBy(binary.getLhs(), bound, depth + 1) &&
           valueUpperBoundedBy(binary.getRhs(), bound, depth + 1);
  case BinaryOperator::Subtract:
    return isKnownNonNegative(binary.getRhs(), depth + 1) &&
           valueUpperBoundedBy(binary.getLhs(), bound, depth + 1);
  case BinaryOperator::Multiply: {
    auto provesFlooredMultiple = [&](Value quotient, Value divisor) {
      auto division = quotient.getDefiningOp<gpu::BinaryOp>();
      return division &&
             division.getOperatorKind() == BinaryOperator::FloorDivide &&
             division.getRhs() == divisor && isKnownPositive(divisor) &&
             valueUpperBoundedBy(division.getLhs(), bound, depth + 1);
    };
    return provesFlooredMultiple(binary.getLhs(), binary.getRhs()) ||
           provesFlooredMultiple(binary.getRhs(), binary.getLhs());
  }
  default:
    return false;
  }
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
           isKnownPositive(divisor) &&
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
           isKnownNonNegative(loop.getLowerBound()) &&
           isKnownPositive(loop.getStep()) &&
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
  auto result = dyn_cast<OpResult>(coordinate);
  auto mapping =
      result ? dyn_cast<gpu::DelinearizeOp>(result.getOwner())
             : gpu::DelinearizeOp();
  if (!mapping || result.getResultNumber() >= mapping.getLaunchExtents().size())
    return false;
  Attribute launchExtent = mapping.getLaunchExtents()[result.getResultNumber()];
  FailureOr<uint64_t> blocked = gpu::blockedDimension(launchExtent);
  auto expression = dyn_cast<gpu::PhysicalExprAttr>(launchExtent);
  if (failed(blocked) || *blocked != static_cast<uint64_t>(dimension) ||
      !expression || expression.getOperands().size() != 2)
    return false;
  auto parameterExpression =
      dyn_cast<gpu::PhysicalExprAttr>(expression.getOperands()[1]);
  auto parameter = block.getDefiningOp<gpu::ParameterOp>();
  return parameterExpression && parameter &&
         parameterExpression.getSymbol() ==
             parameter.getParameter().getName();
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
  Value logicalDimension = dimensionValue(kernel, dimension);
  if (zeroOffset && range->hasAttr(gpu::sourceSubregionAttr) &&
      logicalDimension && isKnownNonNegative(range.getLogicalStart()) &&
      valueUpperBoundedBy(range.getLogicalStop(), logicalDimension) &&
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
          isKnownNonNegative(loop.getLowerBound()) &&
          valueUpperBoundedBy(loop.getUpperBound(), range.getLogicalStop()))
        return true;
    }
    auto parameter = range.getExtent().getDefiningOp<gpu::ParameterOp>();
    if (isProvably(start, 0) && isProvably(range.getLogicalStart(), 0) &&
        parameter &&
        parameter.getParameter().getRole() ==
            static_cast<uint32_t>(gpu::ParameterRole::FullCoverage)) {
      gpu::PhysicalParameterBinding binding =
          gpu::queryParameterBinding(parameter);
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
        isKnownNonNegative(loop.getLowerBound()) &&
        isKnownPositive(loop.getStep()) &&
        valueUpperBoundedBy(loop.getUpperBound(), logicalDimension))
      return true;
  }
  std::optional<int64_t> origin = constantValue(index);
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

FailureOr<NativeTileAccessPlan> analyzeNativeTileAccess(
    Operation *owner, func::FuncOp kernel,
    gpu::PhysicalProgramAnalysis &analysis, gpu::ViewType view,
    ValueRange coordinates, ArrayRef<int64_t> sourceAxes,
    gpu::FragmentType computationType) {
  const unsigned resourceRank = view.getRank();
  const unsigned computationRank = computationType.getShape().size();
  if (coordinates.size() != resourceRank ||
      sourceAxes.size() != resourceRank || computationRank == 0)
    return failure();

  SmallVector<Value> resourceCoordinates(resourceRank);
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes)) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(resourceRank) ||
        resourceCoordinates[sourceAxis])
      return failure();
    resourceCoordinates[sourceAxis] = coordinate;
  }

  NativeTileAccessPlan plan;
  plan.axes.resize(resourceRank);
  plan.toComputation.assign(computationRank, -1);
  plan.toResource.assign(computationRank, -1);
  SmallVector<bool> usedComputationAxis(computationRank, false);
  SmallVector<unsigned> unresolvedScalarAxes;
  gpu::PhysicalAccessBoundsFact accessBounds = analysis.accessBounds(owner);
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
    plan.axes[resourceAxis].computationAxis = computationAxis;
    return true;
  };

  for (unsigned resourceAxis = 0; resourceAxis < resourceRank;
       ++resourceAxis) {
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
              computationType.getShape()[*axis.computationAxis] ||
          !collectTileOffsets(coordinate, axis.range.getResult(), axis.offsets))
        return failure();
      axis.originInBounds =
          rangeOriginInView(axis.range, axis.offsets, view, resourceAxis,
                            dimensions[resourceAxis], kernel) ||
          llvm::is_contained(accessBounds.assumedAxes, resourceAxis);
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
                computationType.getShape()[*axis.computationAxis]))
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
      context, static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), 1,
      StringAttr::get(context), ArrayAttr::get(context, {}));
  SmallVector<Attribute> resourceShape(resourceRank);
  SmallVector<Attribute> resourceMappings(resourceRank);
  SmallVector<Attribute> packedShape;
  SmallVector<Attribute> packedMappings;
  packedShape.reserve(computationRank);
  packedMappings.reserve(computationRank);
  for (unsigned resourceAxis = 0; resourceAxis < resourceRank;
       ++resourceAxis) {
    std::optional<unsigned> computationAxis =
        plan.axes[resourceAxis].computationAxis;
    if (!computationAxis) {
      resourceShape[resourceAxis] = unit;
      resourceMappings[resourceAxis] = gpu::AxisMapAttr::get(
          context, view.getSourceId(), resourceAxis,
          dimensions[resourceAxis], resourceAxis, false);
      continue;
    }
    Attribute extent = computationType.getShape()[*computationAxis];
    auto mapping = cast<gpu::AxisMapAttr>(
        computationType.getAxisMaps()[*computationAxis]);
    resourceShape[resourceAxis] = extent;
    resourceMappings[resourceAxis] = gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), resourceAxis, mapping.getDerived());
    unsigned packedAxis = packedShape.size();
    packedShape.push_back(extent);
    packedMappings.push_back(gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), packedAxis, mapping.getDerived()));
    plan.toComputation[*computationAxis] = packedAxis;
    plan.toResource[packedAxis] = *computationAxis;
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

FailureOr<Value> materializeTileOriginGuard(
    OpBuilder &builder, Operation *owner, Value resource,
    NativeTileAccessPlan &plan,
    const gpu::PhysicalAccessBoundaryFact &boundary) {
  Value condition;
  Value zero;
  for (int64_t rawAxis : boundary.boundaryAxes) {
    if (rawAxis < 0 || rawAxis >= static_cast<int64_t>(plan.axes.size()))
      return failure();
    unsigned axis = static_cast<unsigned>(rawAxis);
    NativeTileAxisPlan &axisPlan = plan.axes[axis];
    if (axisPlan.originInBounds)
      continue;
    Value origin = axisPlan.scalarIndex;
    if (!origin) {
      if (!axisPlan.range)
        return failure();
      origin = axisPlan.range.getStart();
      for (Value offset : axisPlan.offsets)
        origin = builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getIndexType(), origin, offset,
            BinaryOperator::Add);
    }
    if (!zero)
      zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value extent = builder.create<gpu::DimOp>(
        owner->getLoc(), builder.getIndexType(), resource, axis);
    Value nonNegative = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), origin, zero,
        ComparePredicate::Ge);
    Value belowExtent = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), origin, extent,
        ComparePredicate::Lt);
    Value axisCondition = builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getI1Type(), nonNegative, belowExtent,
        BinaryOperator::LogicalAnd);
    condition = condition
                    ? Value(builder.create<gpu::BinaryOp>(
                          owner->getLoc(), builder.getI1Type(), condition,
                          axisCondition, BinaryOperator::LogicalAnd))
                    : axisCondition;
  }
  return condition;
}

OpBuilder prepareBranch(Region &region) {
  Block &block = region.front();
  if (!block.empty() && isa<scf::YieldOp>(block.back()))
    block.back().erase();
  return OpBuilder(&block, block.end());
}

struct MaterializedTileIndices {
  SmallVector<Value> values;
  Value alignment;
};

FailureOr<MaterializedTileIndices>
materializeTileIndices(OpBuilder &builder, Operation *owner,
                       const NativeTileAccessPlan &plan,
                       bool allowDynamicAlignment) {
  MaterializedTileIndices result;
  result.values.reserve(plan.axes.size());
  for (const NativeTileAxisPlan &axis : plan.axes) {
    if (axis.scalarIndex) {
      result.values.push_back(axis.scalarIndex);
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
    if (succeeded(index)) {
      result.values.push_back(*index);
      continue;
    }
    if (!allowDynamicAlignment || !isKnownPositive(range.getExtent()))
      return failure();
    result.values.push_back(builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, range.getExtent(),
        BinaryOperator::FloorDivide));
    Value remainder = builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, range.getExtent(),
        BinaryOperator::Remainder);
    Value zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value aligned = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), remainder, zero,
        ComparePredicate::Eq);
    result.alignment =
        result.alignment
            ? Value(builder.create<gpu::BinaryOp>(
                  owner->getLoc(), builder.getI1Type(), result.alignment,
                  aligned, BinaryOperator::LogicalAnd))
            : aligned;
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

bool hasNativeMMAAxes(gpu::ContractOp contract) {
  auto lhs = contract.getLhs().getType();
  auto rhs = contract.getRhs().getType();
  auto result = contract.getResult().getType();
  const unsigned rank = lhs.getShape().size();
  if (rank < 2 || rank > 3 || rhs.getShape().size() != rank ||
      result.getShape().size() != rank)
    return false;
  const int64_t matrixAxis = rank - 2;
  const bool canonicalBatch =
      rank == 2 ? contract.getLhsBatchAxes().empty() &&
                      contract.getRhsBatchAxes().empty()
                : contract.getLhsBatchAxes() == ArrayRef<int64_t>{0} &&
                      contract.getRhsBatchAxes() == ArrayRef<int64_t>{0};
  return contract.getLhsReductionAxes() ==
             ArrayRef<int64_t>{matrixAxis + 1} &&
         contract.getRhsReductionAxes() ==
             ArrayRef<int64_t>{matrixAxis} &&
         canonicalBatch;
}

Attribute scalarConstant(Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<gpu::CastOp>()) {
      if (cast.getValue().getType() != cast.getResult().getType())
        return {};
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

bool isNativeReductionIdentity(BinaryOperator kind, Value identity) {
  Attribute constant = scalarConstant(identity);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant)) {
    switch (kind) {
    case BinaryOperator::Add:
    case BinaryOperator::LogicalOr:
      return integer.getValue().isZero();
    case BinaryOperator::LogicalAnd:
      return integer.getValue().isAllOnes();
    default:
      return false;
    }
  }
  auto floating = dyn_cast<FloatAttr>(constant);
  if (!floating)
    return false;
  const llvm::APFloat &value = floating.getValue();
  if (kind == BinaryOperator::Add)
    return value.isZero();
  if (!value.isInfinity())
    return false;
  if (kind == BinaryOperator::MaximumNum || kind == BinaryOperator::Maximum)
    return value.isNegative();
  if (kind == BinaryOperator::MinimumNum || kind == BinaryOperator::Minimum)
    return !value.isNegative();
  return false;
}

Type withElementType(Type type, Type elementType) {
  auto fragment = dyn_cast<gpu::FragmentType>(type);
  if (!fragment)
    return elementType;
  return gpu::FragmentType::get(
      fragment.getContext(), elementType, fragment.getShape(),
      fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
}

bool physicalExpressionUsesRole(gpu::PhysicalExprAttr expression,
                                func::FuncOp kernel,
                                gpu::ParameterRole role,
                                unsigned depth = 0) {
  if (!expression || depth >= 32)
    return false;
  if (expression.getKind() ==
      static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter)) {
    FailureOr<gpu::ParameterOp> parameter =
        gpu::queryParameterBySymbol(kernel, expression.getSymbol());
    return succeeded(parameter) &&
           (*parameter).getParameter().getRole() ==
               static_cast<uint32_t>(role);
  }
  return llvm::any_of(expression.getOperands(), [&](Attribute operand) {
    auto nested = dyn_cast<gpu::PhysicalExprAttr>(operand);
    return nested && physicalExpressionUsesRole(nested, kernel, role,
                                                depth + 1);
  });
}

bool fragmentUsesRole(gpu::FragmentType fragment, func::FuncOp kernel,
                      gpu::ParameterRole role) {
  return llvm::any_of(fragment.getShape(), [&](Attribute extent) {
    auto expression = dyn_cast<gpu::PhysicalExprAttr>(extent);
    return expression && physicalExpressionUsesRole(expression, kernel, role);
  });
}

Value regionContractionTransposeCondition(gpu::ContractOp contract,
                                          func::FuncOp kernel) {
  auto result = dyn_cast<gpu::FragmentType>(contract.getResult().getType());
  if (!result || result.getShape().size() != 2)
    return {};
  auto ownershipExtent =
      dyn_cast<gpu::PhysicalExprAttr>(result.getShape()[0]);
  if (!ownershipExtent ||
      ownershipExtent.getKind() !=
          static_cast<uint32_t>(gpu::PhysicalExprKind::Parameter))
    return {};
  FailureOr<gpu::ParameterOp> ownership = gpu::queryParameterBySymbol(
      kernel, ownershipExtent.getSymbol());
  if (failed(ownership) ||
      (*ownership).getParameter().getRole() !=
          static_cast<uint32_t>(gpu::ParameterRole::OwnershipM) ||
      (*ownership).getParameter().getCategory() !=
          static_cast<uint32_t>(gpu::ParameterCategory::RegionContraction))
    return {};
  auto loop = contract->getParentOfType<scf::ForOp>();
  auto segment = loop ? loop.getStep().getDefiningOp<gpu::ParameterOp>()
                      : gpu::ParameterOp();
  if (!segment ||
      segment.getParameter().getRole() !=
          static_cast<uint32_t>(gpu::ParameterRole::ScanChunk) ||
      segment.getParameter().getCategory() !=
          static_cast<uint32_t>(gpu::ParameterCategory::RegionContraction))
    return {};
  OpBuilder builder(contract);
  auto condition = builder.create<gpu::CompareOp>(
      contract.getLoc(), builder.getI1Type(), (*ownership).getResult(),
      segment.getResult(), ComparePredicate::Lt);
  if (Attribute origin = contract->getAttr(gpu::originAttr))
    condition->setAttr(gpu::originAttr, origin);
  return condition;
}

bool containsFragment(Type type) {
  if (isa<gpu::FragmentType>(type))
    return true;
  auto record = dyn_cast<gpu::RecordType>(type);
  return record && llvm::any_of(record.getFieldTypes(), [](Attribute field) {
           return containsFragment(cast<TypeAttr>(field).getValue());
         });
}

bool hasLoopCarriedFragment(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    found |= llvm::any_of(loop.getInitArgs(), [](Value value) {
      return containsFragment(value.getType());
    });
  });
  kernel.walk([&](scf::WhileOp loop) {
    found |= llvm::any_of(loop.getInits(), [](Value value) {
      return containsFragment(value.getType());
    });
  });
  return found;
}

bool hasOccupancySensitiveTileCompute(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](Operation *operation) {
    found |= isa<gpu::ContractOp, gpu::ScaledContractOp, gpu::ReduceOp,
                 gpu::ScanOp, MMAOp, ScaledMMAOp, ReduceOp, ScanOp>(operation);
  });
  return found;
}

bool hasResidentWorkerTraversal(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    found |= parameter.getParameter().getRole() ==
             static_cast<uint32_t>(gpu::ParameterRole::ResidentWorkers);
  });
  return found;
}

StringRef occupancyFamily(func::FuncOp kernel,
                          gpu::CapabilitiesAttr capabilities) {
  if (hasResidentWorkerTraversal(kernel))
    return "occupancy_persistent";
  if (hasLoopCarriedFragment(kernel))
    return "occupancy_loop";
  return capabilities.getComputeCapabilityMajor() < 9
             ? "occupancy_legacy" : "occupancy_modern";
}

LogicalResult verifyScan(gpu::ScanOp scan) {
  if (scan.getSourceCount() == 0 ||
      scan.getSourceCount() != scan.getIdentityCount() ||
      scan.getSourceCount() != scan.getNumResults() ||
      scan.getCaptureCount() != 0 || !scan.getInclusive())
    return scan.emitOpError(
        "cuTile scan lowering requires matching source/identity/result schemas, no captures, and an inclusive prefix");
  auto source = dyn_cast<gpu::FragmentType>(scan.getInputs().front().getType());
  if (!source)
    return scan.emitOpError("cuTile scan source must be a tile");
  for (auto [input, identity] : llvm::zip(
           scan.getInputs().take_front(scan.getSourceCount()),
           scan.getInputs().slice(scan.getSourceCount(), scan.getIdentityCount()))) {
    auto fragment = dyn_cast<gpu::FragmentType>(input.getType());
    if (!fragment || fragment.getShape() != source.getShape())
      return scan.emitOpError(
          "cuTile scan lowering requires source components with the same physical shape");
    auto constant = dyn_cast_or_null<TypedAttr>(scalarConstant(identity));
    if (!constant || constant.getType() != fragment.getElementType())
      return scan.emitOpError(
          "cuTile scan identity must be an explicit scalar constant of the source dtype");
  }
  for (Operation &operation : scan.getCombine().front())
    if (!isa<arith::ConstantOp, gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp,
             gpu::SelectOp, gpu::CastOp, gpu::BitcastOp, gpu::MakeRecordOp,
             gpu::ExtractOp, gpu::YieldOp>(operation))
      return scan.emitOpError(
          "cuTile scan callback lowering requires scalar elementwise operations");
  return success();
}

LogicalResult formNativeTiles(func::FuncOp kernel,
                              const gpu::TuningProfiles &profiles) {
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
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!scaledContracts.empty() && !supportsE8M0ScaledMMA(capabilities))
    return scaledContracts.front().emitOpError(
        "cuTile E8M0 scaled MMA requires compute capability 10.0 or newer");
  if (hasOccupancySensitiveTileCompute(kernel)) {
    auto rows = profiles.get("cutile", occupancyFamily(kernel, capabilities),
                             kernel.getLoc());
    if (failed(rows))
      return failure();
    SmallVector<int64_t> occupancyCandidates;
    for (const auto &row : *rows)
      if (isLegalOccupancy(row[0]))
        occupancyCandidates.push_back(row[0]);
    if (occupancyCandidates.empty())
      return kernel.emitError("cuTile tuning profile has no legal occupancy hints (1..32)");
    bool nameCollision = false;
    kernel.walk([&](gpu::ParameterOp parameter) {
      nameCollision |= parameter.getParameter().getName().getValue() ==
                       occupancyParameter;
    });
    if (nameCollision)
      return kernel.emitError(
          "cuTile occupancy parameter name is already owned");
    OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
    auto schema = gpu::ParameterAttr::get(
        kernel.getContext(), entry.getStringAttr(occupancyParameter),
        static_cast<uint32_t>(gpu::ParameterRole::ProviderOccupancy),
        static_cast<uint32_t>(gpu::ParameterCategory::Provider),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), occupancyCandidates));
    entry.create<gpu::ParameterOp>(kernel.getLoc(), entry.getIndexType(), schema);
  }
  gpu::PhysicalProgramAnalysis analysis(kernel);
  Value accessForm;
  Value preferTileLoads;
  Value allowBlockedTMA;
  Value allowDefaultTMA;
  auto accessFormValue = [&]() -> FailureOr<Value> {
    if (accessForm)
      return accessForm;
    auto rows = profiles.get("cutile", "access_forms", kernel.getLoc());
    if (failed(rows))
      return failure();
    SmallVector<int64_t> candidates;
    for (const auto &row : *rows)
      if (isLegalAccessForm(row[0]))
        candidates.push_back(row[0]);
    if (candidates.empty())
      return kernel.emitError("cuTile tuning profile has no legal access forms (1, 2, 3)");
    bool nameCollision = false;
    kernel.walk([&](gpu::ParameterOp parameter) {
      nameCollision |= parameter.getParameter().getName().getValue() ==
                       accessFormParameter;
    });
    if (nameCollision)
      return kernel.emitError(
          "cuTile access-form parameter name is already owned");
    OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
    auto schema = gpu::ParameterAttr::get(
        kernel.getContext(), entry.getStringAttr(accessFormParameter),
        static_cast<uint32_t>(gpu::ParameterRole::ProviderAccessForm),
        static_cast<uint32_t>(gpu::ParameterCategory::Provider),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    accessForm = entry
                     .create<gpu::ParameterOp>(kernel.getLoc(),
                                               entry.getIndexType(), schema)
                     .getResult();
    return accessForm;
  };
  auto loadFormCondition = [&]() -> FailureOr<Value> {
    if (preferTileLoads)
      return preferTileLoads;
    FailureOr<Value> form = accessFormValue();
    if (failed(form))
      return failure();
    OpBuilder entry(kernel.getContext());
    entry.setInsertionPointAfter((*form).getDefiningOp());
    Value gather = entry.create<arith::ConstantIndexOp>(kernel.getLoc(),
                                                         gatherAccessForm);
    preferTileLoads = entry.create<gpu::CompareOp>(
        kernel.getLoc(), entry.getI1Type(), *form, gather,
        ComparePredicate::Ne);
    return preferTileLoads;
  };
  auto tmaCondition = [&](gpu::FragmentType tile) -> FailureOr<Value> {
    if (fragmentUsesRole(tile, kernel, gpu::ParameterRole::FullCoverage)) {
      if (!allowDefaultTMA) {
        OpBuilder entry(&kernel.getBody().front(),
                        kernel.getBody().front().begin());
        allowDefaultTMA = entry.create<arith::ConstantIntOp>(
            kernel.getLoc(), /*value=*/1, /*width=*/1);
      }
      return allowDefaultTMA;
    }
    if (allowBlockedTMA)
      return allowBlockedTMA;
    FailureOr<Value> form = accessFormValue();
    if (failed(form))
      return failure();
    OpBuilder entry(kernel.getContext());
    entry.setInsertionPointAfter((*form).getDefiningOp());
    Value disabled = entry.create<arith::ConstantIndexOp>(
        kernel.getLoc(), nativeBlockedNoTMAForm);
    allowBlockedTMA = entry.create<gpu::CompareOp>(
        kernel.getLoc(), entry.getI1Type(), *form, disabled,
        ComparePredicate::Ne);
    return allowBlockedTMA;
  };

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
      bool inBounds = scalarCoordinatesInView(*indices, view, kernel);
      Value validity = load.getValid();
      Value padding = *fill;
      gpu::PhysicalAccessBoundaryFact boundary =
          analysis.boundaryValidity(load);
      if (inBounds && boundary.isExact()) {
        validity = {};
        padding = {};
      }
      auto replacement = builder.create<ScalarLoadOp>(
          load.getLoc(), load.getResult().getType(), load.getResource(), *indices,
          validity, padding,
          inBounds ? builder.getUnitAttr() : UnitAttr());
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
    FailureOr<MaterializedTileIndices> indices = failure();
    FailureOr<Value> originGuard = failure();
    if (succeeded(plan) && boundary.isExact() &&
        (!load.getFill() || isZeroFill(load.getFill())))
      indices = materializeTileIndices(builder, load, *plan,
                                       /*allowDynamicAlignment=*/true);
    if (succeeded(indices))
      originGuard = materializeTileOriginGuard(
          builder, load, load.getResource(), *plan, boundary);
    bool guardedNative = succeeded(originGuard) && *originGuard;
    bool native = succeeded(indices) && succeeded(originGuard) &&
                  (!guardedNative ||
                   (load.getFill() && load.getFill().getType() == result));
    FailureOr<Value> allowTMA = failure();
    if (native) {
      allowTMA = tmaCondition(plan->resourceType);
      if (failed(allowTMA))
        return failure();
    }
    Value replacementResult;
    SmallVector<Operation *> createdOperations;
    auto emitNativeLoad = [&](OpBuilder &nested) {
      auto tile = nested.create<TileLoadOp>(
          load.getLoc(), plan->resourceType, load.getResource(), *allowTMA,
          indices->values);
      Value value = tile.getResult();
      createdOperations.push_back(tile);
      if (plan->resourceToPacked) {
        auto reshape = nested.create<gpu::ReshapeOp>(
            load.getLoc(), plan->packedType, value,
            plan->resourceToPacked);
        value = reshape.getResult();
        createdOperations.push_back(reshape);
      }
      if (!isIdentityPermutation(plan->toComputation)) {
        auto transpose = nested.create<gpu::TransposeOp>(
            load.getLoc(), result, value, plan->toComputation);
        value = transpose.getResult();
        createdOperations.push_back(transpose);
      }
      return value;
    };
    auto emitGatherLoad = [&](OpBuilder &nested) -> FailureOr<Value> {
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
          load, load.getResource(), load.getCoordinates(), load.getSourceAxes());
      FailureOr<Value> fill = scalarFill(load, load.getFill());
      if (failed(coordinates) || failed(fill))
        return failure();
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(nested, load, *coordinates, result);
      if (failed(materialized))
        return failure();
      auto replacement = nested.create<GatherLoadOp>(
          load.getLoc(), result, load.getResource(), *materialized,
          load.getValid(), *fill, identityAxes(view.getRank()));
      createdOperations.push_back(replacement);
      return replacement.getResult();
    };
    auto emitNativeOrFill = [&](OpBuilder &nested) -> Value {
      if (!guardedNative)
        return emitNativeLoad(nested);
      auto conditional = nested.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, *originGuard,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder inBounds = prepareBranch(conditional.getThenRegion());
      Value value = emitNativeLoad(inBounds);
      inBounds.create<scf::YieldOp>(load.getLoc(), value);
      OpBuilder outOfBounds = prepareBranch(conditional.getElseRegion());
      outOfBounds.create<scf::YieldOp>(load.getLoc(), load.getFill());
      return conditional.getResult(0);
    };
    auto emitAlignedNative = [&](OpBuilder &nested) -> FailureOr<Value> {
      if (!indices->alignment)
        return emitNativeOrFill(nested);
      auto conditional = nested.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, indices->alignment,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder aligned = prepareBranch(conditional.getThenRegion());
      Value value = emitNativeOrFill(aligned);
      aligned.create<scf::YieldOp>(load.getLoc(), value);
      OpBuilder unaligned = prepareBranch(conditional.getElseRegion());
      FailureOr<Value> gathered = emitGatherLoad(unaligned);
      if (failed(gathered))
        return failure();
      unaligned.create<scf::YieldOp>(load.getLoc(), *gathered);
      return conditional.getResult(0);
    };
    if (native) {
      FailureOr<Value> condition = loadFormCondition();
      if (failed(condition))
        return failure();
      auto conditional = builder.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, *condition,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder tiled = prepareBranch(conditional.getThenRegion());
      FailureOr<Value> tiledValue = emitAlignedNative(tiled);
      if (failed(tiledValue))
        return failure();
      tiled.create<scf::YieldOp>(load.getLoc(), *tiledValue);
      OpBuilder gathered = prepareBranch(conditional.getElseRegion());
      FailureOr<Value> gatheredValue = emitGatherLoad(gathered);
      if (failed(gatheredValue))
        return failure();
      gathered.create<scf::YieldOp>(load.getLoc(), *gatheredValue);
      replacementResult = conditional.getResult(0);
    } else {
      FailureOr<Value> gathered = emitGatherLoad(builder);
      if (failed(gathered))
        return failure();
      replacementResult = *gathered;
    }
    for (Operation *operation : createdOperations)
      if (Attribute origin = load->getAttr(gpu::originAttr))
        operation->setAttr(gpu::originAttr, origin);
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
    // In-bounds coordinates fit i32 under cuTile's tile-size limit.
    // This does not narrow external view dimensions or address arithmetic.
    for (Value &coordinate : coordinates)
      if (!coordinate.getType().isInteger(32))
        coordinate = builder.create<gpu::CastOp>(
            gather.getLoc(), builder.getI32Type(), coordinate);
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
                  reduce.getNumResults() == 1 && kind.has_value() &&
                  isNativeReductionIdentity(
                      *kind, reduce.getInputs()[reduce.getSourceCount()]);
    if (native) {
      auto source =
          dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
      if (!source)
        return reduce.emitOpError("cuTile native reduce source must be a tile");
      OpBuilder builder(reduce);
      auto replacement = builder.create<ReduceOp>(
          reduce.getLoc(), reduce.getResultTypes().front(),
          reduce.getInputs().front(), reduce.getAxes().front(), *kind);
      if (Attribute origin = reduce->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      reduce.getResults().front().replaceAllUsesWith(replacement.getResult());
      reduce.erase();
      continue;
    }

    std::optional<BinaryOperator> canonicalKind =
        gpu::queryBinaryCombineKind(reduce.getCombine());
    bool propagatingMaximum =
        reduce.getSourceCount() == 1 && reduce.getIdentityCount() == 1 &&
        reduce.getCaptureCount() == 0 && reduce.getAxes().size() == 1 &&
        reduce.getNumResults() == 1 && canonicalKind == BinaryOperator::Maximum &&
        isNativeReductionIdentity(
            BinaryOperator::Maximum,
            reduce.getInputs()[reduce.getSourceCount()]);
    if (propagatingMaximum) {
      auto sourceType =
          dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
      if (!sourceType || !isa<FloatType>(sourceType.getElementType()))
        return reduce.emitOpError(
            "cuTile propagating maximum reduction requires a floating tile");
      OpBuilder builder(reduce);
      Location location = reduce.getLoc();
      Type resultType = reduce.getResultTypes().front();
      Type sourcePredicateType =
          withElementType(sourceType, builder.getI1Type());
      Type resultPredicateType =
          withElementType(resultType, builder.getI1Type());
      auto isNan = builder.create<gpu::CompareOp>(
          location, sourcePredicateType, reduce.getInputs().front(),
          reduce.getInputs().front(), ComparePredicate::Ne);
      auto anyNan = builder.create<ReduceOp>(
          location, resultPredicateType, isNan.getResult(),
          reduce.getAxes().front(), BinaryOperator::LogicalOr);
      auto numericMaximum = builder.create<ReduceOp>(
          location, resultType, reduce.getInputs().front(),
          reduce.getAxes().front(), BinaryOperator::MaximumNum);
      auto floatType = cast<FloatType>(sourceType.getElementType());
      auto nan = builder.create<arith::ConstantOp>(
          location, floatType,
          builder.getFloatAttr(
              floatType,
              llvm::APFloat::getNaN(floatType.getFloatSemantics())));
      Value nanValue = nan.getResult();
      if (isa<gpu::FragmentType>(resultType))
        nanValue = builder.create<gpu::SplatOp>(location, resultType, nanValue);
      auto replacement = builder.create<gpu::SelectOp>(
          location, resultType, anyNan.getResult(), nanValue,
          numericMaximum.getResult());
      if (Attribute origin = reduce->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
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
    if (failed(verifyScan(scan)))
      return failure();
    std::optional<BinaryOperator> kind = nativeCombineKind(scan.getCombine());
    if (scan.getSourceCount() != 1 || !kind || *kind != BinaryOperator::Add ||
        !isNativeReductionIdentity(*kind, scan.getInputs()[1]))
      continue;
    OpBuilder builder(scan);
    auto replacement = builder.create<ScanOp>(
        scan.getLoc(), cast<gpu::FragmentType>(scan.getResultTypes().front()),
        scan.getInputs().front(), scan.getAxis(), *kind, scan.getReverse());
    if (Attribute origin = scan->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    scan.getResults().front().replaceAllUsesWith(replacement.getResult());
    scan.erase();
  }

  for (gpu::ContractOp contract : contracts) {
    if (!hasNativeMMAAxes(contract))
      return contract.emitOpError(
          "cuTile native MMA requires [M,K] x [K,N] or [B,M,K] x [B,K,N] "
          "physical axes");
    OpBuilder builder(contract);
    Attribute origin = contract->getAttr(gpu::originAttr);
    auto inheritOrigin = [&](Operation *operation) {
      if (origin)
        operation->setAttr(gpu::originAttr, origin);
    };
    auto emitMMA = [&](OpBuilder &nested) -> Value {
      auto replacement = nested.create<MMAOp>(
          contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
          contract.getRhs(), contract.getAccumulator());
      inheritOrigin(replacement);
      return replacement.getResult();
    };
    auto emitTransposedMMA = [&](OpBuilder &nested) -> Value {
      auto lhsType = cast<gpu::FragmentType>(contract.getLhs().getType());
      auto rhsType = cast<gpu::FragmentType>(contract.getRhs().getType());
      auto accumulatorType =
          cast<gpu::FragmentType>(contract.getAccumulator().getType());
      auto lhs = nested.create<gpu::TransposeOp>(
          contract.getLoc(), transposeRankTwo(lhsType), contract.getLhs(),
          ArrayRef<int64_t>{1, 0});
      auto rhs = nested.create<gpu::TransposeOp>(
          contract.getLoc(), transposeRankTwo(rhsType), contract.getRhs(),
          ArrayRef<int64_t>{1, 0});
      auto accumulator = nested.create<gpu::TransposeOp>(
          contract.getLoc(), transposeRankTwo(accumulatorType),
          contract.getAccumulator(), ArrayRef<int64_t>{1, 0});
      auto mma = nested.create<MMAOp>(
          contract.getLoc(), accumulator.getResult().getType(), rhs, lhs,
          accumulator);
      auto result = nested.create<gpu::TransposeOp>(
          contract.getLoc(), contract.getResult().getType(), mma,
          ArrayRef<int64_t>{1, 0});
      for (Operation *operation :
           {lhs.getOperation(), rhs.getOperation(), accumulator.getOperation(),
            mma.getOperation(), result.getOperation()})
        inheritOrigin(operation);
      return result;
    };
    Value transpose = regionContractionTransposeCondition(contract, kernel);
    Value result;
    if (transpose) {
      auto conditional = builder.create<scf::IfOp>(
          contract.getLoc(), TypeRange{contract.getResult().getType()}, transpose,
          /*withElseRegion=*/true);
      inheritOrigin(conditional);
      OpBuilder transposed = prepareBranch(conditional.getThenRegion());
      transposed.create<scf::YieldOp>(contract.getLoc(),
                                      emitTransposedMMA(transposed));
      OpBuilder direct = prepareBranch(conditional.getElseRegion());
      direct.create<scf::YieldOp>(contract.getLoc(), emitMMA(direct));
      result = conditional.getResult(0);
    } else {
      result = emitMMA(builder);
    }
    contract.getResult().replaceAllUsesWith(result);
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
      bool inBounds = scalarCoordinatesInView(*indices, view, kernel);
      Value validity = store.getValid();
      gpu::PhysicalAccessBoundaryFact boundary =
          analysis.boundaryValidity(store);
      if (inBounds && boundary.isExact())
        validity = {};
      auto replacement = builder.create<ScalarStoreOp>(
          store.getLoc(), store.getResource(), *indices, store.getValue(),
          validity, inBounds ? builder.getUnitAttr() : UnitAttr());
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
    FailureOr<MaterializedTileIndices> indices = failure();
    FailureOr<Value> originGuard = failure();
    if (succeeded(plan) && boundary.isExact())
      indices = materializeTileIndices(builder, store, *plan,
                                       /*allowDynamicAlignment=*/false);
    if (succeeded(indices))
      originGuard = materializeTileOriginGuard(
          builder, store, store.getResource(), *plan, boundary);
    bool native = succeeded(indices) && succeeded(originGuard);
    bool guardedNative = native && *originGuard;
    FailureOr<Value> allowTMA = failure();
    if (native) {
      allowTMA = tmaCondition(plan->resourceType);
      if (failed(allowTMA))
        return failure();
    }
    SmallVector<Operation *> createdOperations;
    auto emitNativeStore = [&](OpBuilder &nested) {
      Value nativeValue = store.getValue();
      if (!isIdentityPermutation(plan->toResource)) {
        auto transpose = nested.create<gpu::TransposeOp>(
            store.getLoc(), plan->packedType, nativeValue,
            plan->toResource);
        nativeValue = transpose.getResult();
        createdOperations.push_back(transpose);
      }
      if (plan->packedToResource) {
        auto reshape = nested.create<gpu::ReshapeOp>(
            store.getLoc(), plan->resourceType, nativeValue,
            plan->packedToResource);
        nativeValue = reshape.getResult();
        createdOperations.push_back(reshape);
      }
      auto tile = nested.create<TileStoreOp>(
          store.getLoc(), store.getResource(), *allowTMA, indices->values,
          nativeValue);
      createdOperations.push_back(tile);
    };
    if (guardedNative) {
      auto conditional = builder.create<scf::IfOp>(
          store.getLoc(), TypeRange{}, *originGuard,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder inBounds = prepareBranch(conditional.getThenRegion());
      emitNativeStore(inBounds);
      inBounds.create<scf::YieldOp>(store.getLoc());
      OpBuilder outOfBounds = prepareBranch(conditional.getElseRegion());
      outOfBounds.create<scf::YieldOp>(store.getLoc());
    } else if (native) {
      emitNativeStore(builder);
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
      auto replacement = builder.create<ScatterStoreOp>(
          store.getLoc(), store.getResource(), *materialized, store.getValue(),
          store.getValid(), identityAxes(view.getRank()));
      createdOperations.push_back(replacement);
    }
    for (Operation *operation : createdOperations)
      if (Attribute origin = store->getAttr(gpu::originAttr))
        operation->setAttr(gpu::originAttr, origin);
    store.erase();
  }
  for (gpu::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult materializeClosedConfigs(func::FuncOp kernel) {
  struct Domain {
    gpu::ParameterOp parameter;
    bool provider;
    bool coverage;
  };

  llvm::StringMap<gpu::ParameterOp> names;
  SmallVector<Domain> domains;
  WalkResult schema = kernel.walk([&](gpu::ParameterOp parameter) {
    gpu::ParameterAttr definition = parameter.getParameter();
    StringRef name = definition.getName().getValue();
    if (!names.try_emplace(name, parameter).second) {
      parameter.emitOpError("duplicates a cuTile physical parameter");
      return WalkResult::interrupt();
    }
    auto role = static_cast<gpu::ParameterRole>(definition.getRole());
    bool provider = definition.getCategory() ==
                    static_cast<uint32_t>(gpu::ParameterCategory::Provider);
    if (provider != isCuTileProviderRole(role)) {
      parameter.emitOpError(
          "cuTile program contains a foreign provider parameter");
      return WalkResult::interrupt();
    }
    if (definition.getCandidates().empty()) {
      parameter.emitOpError("has an empty cuTile parameter domain");
      return WalkResult::interrupt();
    }
    domains.push_back(
        {parameter, provider,
         static_cast<bool>(
             parameter->getAttr(gpu::coverageDimensionAttr))});
    return WalkResult::advance();
  });
  if (schema.wasInterrupted())
    return failure();

  auto shared =
      kernel->getAttrOfType<ArrayAttr>(gpu::sharedConfigTuplesAttr);
  if (!shared || shared.empty())
    return kernel.emitError(
        "cuTile legalization requires shared config tuples");

  Builder builder(kernel.getContext());
  SmallVector<SmallVector<NamedAttribute>> configurations;
  for (Attribute attribute : shared) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (!tuple)
      return kernel.emitError("shared config tuple is malformed");
    SmallVector<NamedAttribute> bindings;
    unsigned sharedParameters = 0;
    for (Domain &domain : domains) {
      if (domain.provider || domain.coverage)
        continue;
      ++sharedParameters;
      gpu::ParameterAttr definition = domain.parameter.getParameter();
      auto value = tuple.getAs<IntegerAttr>(definition.getName());
      if (!value ||
          !llvm::is_contained(definition.getCandidates().asArrayRef(),
                              value.getInt()))
        return kernel.emitError(
                   "shared config tuple does not bind a cuTile kernel parameter: ")
               << definition.getName().getValue();
      bindings.push_back(builder.getNamedAttr(definition.getName(), value));
    }
    if (tuple.size() != sharedParameters)
      return kernel.emitError(
          "shared config tuple contains a non-kernel binding");
    configurations.push_back(std::move(bindings));
  }

  for (Domain &domain : domains) {
    if (!domain.provider)
      continue;
    SmallVector<SmallVector<NamedAttribute>> expanded;
    gpu::ParameterAttr definition = domain.parameter.getParameter();
    for (const auto &base : configurations)
      for (int64_t candidate : definition.getCandidates().asArrayRef()) {
        SmallVector<NamedAttribute> bindings(base);
        bindings.push_back(builder.getNamedAttr(
            definition.getName(), builder.getI64IntegerAttr(candidate)));
        expanded.push_back(std::move(bindings));
      }
    configurations = std::move(expanded);
  }

  SmallVector<Attribute> encoded;
  for (const auto &bindings : configurations) {
    DictionaryAttr candidate = builder.getDictionaryAttr(bindings);
    if (!llvm::is_contained(encoded, Attribute(candidate)))
      encoded.push_back(candidate);
  }
  if (encoded.empty())
    return kernel.emitError("cuTile legalization produced no provider config");
  kernel->setAttr(gpu::cuTileConfigsAttr, builder.getArrayAttr(encoded));
  return success();
}

LogicalResult verifyClosedConfigs(func::FuncOp kernel) {
  llvm::StringMap<gpu::ParameterOp> parameters;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (!parameter->hasAttr(gpu::coverageDimensionAttr))
      parameters.try_emplace(parameter.getParameter().getName().getValue(),
                             parameter);
  });
  auto encoded = kernel->getAttrOfType<ArrayAttr>(gpu::cuTileConfigsAttr);
  if (!encoded || encoded.empty())
    return kernel.emitError(
        "cuTile legalization did not materialize closed provider configs");
  llvm::SmallDenseSet<Attribute, 8> unique;
  for (Attribute attribute : encoded) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (!tuple || tuple.size() != parameters.size())
      return kernel.emitError("contains a malformed cuTile provider config");
    if (!unique.insert(attribute).second)
      return kernel.emitError("contains a duplicate cuTile provider config");
    for (NamedAttribute binding : tuple) {
      auto found = parameters.find(binding.getName().getValue());
      auto value = dyn_cast<IntegerAttr>(binding.getValue());
      if (found == parameters.end() || !value ||
          !llvm::is_contained(
              found->second.getParameter().getCandidates().asArrayRef(),
              value.getInt()))
        return kernel.emitError(
            "cuTile provider config contains an invalid binding");
    }
  }
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
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (!capabilities)
    return kernel.emitError("cuTile provider requires selected GPU capabilities");
  gpu::ParameterOp accessForm;
  gpu::ParameterOp occupancy;
  LogicalResult parameterSchema = success();
  kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    auto role = static_cast<gpu::ParameterRole>(schema.getRole());
    auto category =
        static_cast<gpu::ParameterCategory>(schema.getCategory());
    bool provider = category == gpu::ParameterCategory::Provider;
    bool cuTileProvider = isCuTileProviderRole(role);
    if (provider != cuTileProvider) {
      parameter.emitOpError(
          "cuTile program contains a foreign provider parameter");
      parameterSchema = failure();
      return;
    }
    if (!provider)
      return;
    ArrayRef<int64_t> candidates = schema.getCandidates().asArrayRef();
    if (role == gpu::ParameterRole::ProviderAccessForm) {
      if (accessForm) {
        parameter.emitOpError("duplicates the cuTile access-form parameter");
        parameterSchema = failure();
        return;
      }
      if (schema.getName().getValue() != accessFormParameter ||
          candidates.empty() || !llvm::all_of(candidates, isLegalAccessForm)) {
        parameter.emitOpError("has an invalid cuTile access-form domain");
        parameterSchema = failure();
        return;
      }
      accessForm = parameter;
      return;
    }
    if (occupancy) {
      parameter.emitOpError("duplicates the cuTile occupancy parameter");
      parameterSchema = failure();
      return;
    }
    if (schema.getName().getValue() != occupancyParameter ||
        candidates.empty() || !llvm::all_of(candidates, isLegalOccupancy) ||
        !parameter.getResult().use_empty()) {
      parameter.emitOpError(
          "has an invalid cuTile occupancy hint schema");
      parameterSchema = failure();
      return;
    }
    occupancy = parameter;
  });
  if (failed(parameterSchema))
    return failure();
  if (failed(verifyClosedConfigs(kernel)))
    return failure();
  bool needsOccupancy = hasOccupancySensitiveTileCompute(kernel);
  if (needsOccupancy != static_cast<bool>(occupancy))
    return kernel.emitError(
        "cuTile occupancy hint does not match occupancy-sensitive tile compute");
  auto isTrue = [](Value value) {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
    return integer && integer.getType().isInteger(1) &&
           integer.getValue().isOne();
  };
  auto isBlockedTMACondition = [&](Value value) {
    auto compare = value.getDefiningOp<gpu::CompareOp>();
    return accessForm && compare &&
           compare.getPredicate() == ComparePredicate::Ne &&
           compare.getLhs() == accessForm.getResult() &&
           constantValue(compare.getRhs()) == nativeBlockedNoTMAForm;
  };
  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<gpu::LoadOp, gpu::StoreOp, gpu::ContractOp>(operation)) {
      operation->emitOpError(
          "was not converted to an explicit cuTile tile/MMA form");
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<TileLoadOp>(operation)) {
      bool fullCoverage = fragmentUsesRole(
          load.getResult().getType(), kernel,
          gpu::ParameterRole::FullCoverage);
      if ((fullCoverage && !isTrue(load.getAllowTma())) ||
          (!fullCoverage && !isBlockedTMACondition(load.getAllowTma()))) {
        load.emitOpError(
            "allow_tma is not the typed cuTile access-form decision");
        return WalkResult::interrupt();
      }
    }
    if (auto store = dyn_cast<TileStoreOp>(operation)) {
      bool fullCoverage = fragmentUsesRole(
          store.getValue().getType(), kernel,
          gpu::ParameterRole::FullCoverage);
      if ((fullCoverage && !isTrue(store.getAllowTma())) ||
          (!fullCoverage && !isBlockedTMACondition(store.getAllowTma()))) {
        store.emitOpError(
            "allow_tma is not the typed cuTile access-form decision");
        return WalkResult::interrupt();
      }
    }
    if (auto scaled = dyn_cast<ScaledMMAOp>(operation)) {
      if (!supportsE8M0ScaledMMA(capabilities)) {
        scaled.emitOpError(
            "requires compute capability 10.0 or newer for E8M0 scaled MMA");
        return WalkResult::interrupt();
      }
    }
    if (auto scan = dyn_cast<gpu::ScanOp>(operation))
      if (failed(verifyScan(scan)))
        return WalkResult::interrupt();
    if (auto loop = dyn_cast<scf::ForOp>(operation))
      if (!loop.getInductionVar().getType().isSignlessInteger(32)) {
        loop.emitOpError("cuTile native for requires an i32 induction variable");
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
            ReduceOp, ScanOp, gpu::ReduceOp, gpu::ScanOp, gpu::ParameterOp,
            gpu::PhysicalExprOp, gpu::ProgramIdOp, gpu::WorksetCoordinateOp,
            gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, gpu::YieldOp, arith::ConstantOp,
            scf::ForOp, scf::WhileOp, scf::ConditionOp, scf::IfOp, scf::YieldOp, func::FuncOp,
            func::ReturnOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed cuTile provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool fitsNativeLoopBound(Value value, unsigned depth = 0) {
  if (depth >= 32)
    return false;
  auto integer = dyn_cast<IntegerType>(value.getType());
  if (integer && (integer.getWidth() < 32 ||
                  (integer.getWidth() == 32 && !integer.isUnsigned())))
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attribute)
      return false;
    return integer && integer.isUnsigned()
               ? attribute.getValue().getActiveBits() < 32
               : attribute.getValue().isSignedIntN(32);
  }
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>())
    return llvm::all_of(parameter.getParameter().getCandidates().asArrayRef(),
                        [](int64_t candidate) { return llvm::isInt<32>(candidate); });
  if (auto bound = value.getDefiningOp<gpu::RangeBoundOp>())
    if (auto range = bound.getRange().getDefiningOp<gpu::RangeOp>())
      return fitsNativeLoopBound(bound.getBound() == 0 ? range.getStart()
                                 : bound.getBound() == 1 ? range.getStop()
                                                        : range.getStep(),
                                 depth + 1);
  if (auto cast = value.getDefiningOp<gpu::CastOp>())
    if (value.getType().isIndex() ||
        (integer && integer.getWidth() == 64 && !integer.isUnsigned()))
      return fitsNativeLoopBound(cast.getValue(), depth + 1);
  return false;
}

bool isSpecializedLoopBound(Value value, unsigned depth = 0) {
  if (depth >= 32)
    return false;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto kernel = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!kernel)
      return false;
    auto kind = kernel.getArgAttrOfType<StringAttr>(argument.getArgNumber(),
                                                   gpu::abiKindAttr);
    return kind && (kind.getValue() == "dimension" ||
                    kind.getValue() == "stride" ||
                    kind.getValue() == "constexpr");
  }
  Operation *producer = value.getDefiningOp();
  if (isa_and_nonnull<arith::ConstantOp, gpu::ParameterOp, gpu::DimOp>(producer))
    return true;
  if (auto bound = dyn_cast_or_null<gpu::RangeBoundOp>(producer)) {
    auto range = bound.getRange().getDefiningOp<gpu::RangeOp>();
    return range && isSpecializedLoopBound(
                        bound.getBound() == 0 ? range.getStart()
                        : bound.getBound() == 1 ? range.getStop()
                                               : range.getStep(),
                        depth + 1);
  }
  return isa_and_nonnull<gpu::CastOp, gpu::BinaryOp>(producer) &&
         llvm::all_of(producer->getOperands(), [&](Value operand) {
           return isSpecializedLoopBound(operand, depth + 1);
         });
}

std::optional<int64_t> maximumNativeLoopStep(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
    if (attribute && attribute.getValue().isSignedIntN(32) &&
        attribute.getInt() > 0)
      return attribute.getInt();
  }
  if (auto parameter = value.getDefiningOp<gpu::ParameterOp>()) {
    ArrayRef<int64_t> candidates =
        parameter.getParameter().getCandidates().asArrayRef();
    if (!candidates.empty() && llvm::all_of(candidates, [](int64_t candidate) {
          return candidate > 0 && llvm::isInt<32>(candidate);
        }))
      return *llvm::max_element(candidates);
  }
  return std::nullopt;
}

void realizeWideLoops(func::FuncOp kernel) {
  SmallVector<scf::ForOp> loops;
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
    Type type = loop.getInductionVar().getType();
    if (type.isIndex() || type.isInteger(64))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    OpBuilder builder(loop);
    Location location = loop.getLoc();
    std::optional<int64_t> maximumStep = maximumNativeLoopStep(loop.getStep());
    auto nativeLoop = [&](OpBuilder &nested) {
      Value lower = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), loop.getLowerBound());
      Value upper = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), loop.getUpperBound());
      Value step = nested.create<gpu::CastOp>(
          location, nested.getI32Type(), loop.getStep());
      auto replacement = nested.create<scf::ForOp>(
          location, lower, upper, step, loop.getInitArgs(),
          [&](OpBuilder &body, Location bodyLoc, Value induction,
              ValueRange carried) {
            IRMapping mapping;
            Value wide = body.create<gpu::CastOp>(
                bodyLoc, loop.getInductionVar().getType(), induction);
            mapping.map(loop.getInductionVar(), wide);
            mapping.map(loop.getRegionIterArgs(), carried);
            for (Operation &operation : loop.getBody()->without_terminator())
              body.clone(operation, mapping);
            SmallVector<Value> yielded;
            for (Value value :
                 cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands())
              yielded.push_back(mapping.lookupOrDefault(value));
            body.create<scf::YieldOp>(bodyLoc, yielded);
          });
      replacement->setAttrs(loop->getAttrs());
      return replacement;
    };
    auto wideLoop = [&](OpBuilder &nested) {
      OpBuilder::InsertionGuard guard(nested);
      SmallVector<Value> initial{loop.getLowerBound()};
      llvm::append_range(initial, loop.getInitArgs());
      SmallVector<Type> types;
      for (Value value : initial)
        types.push_back(value.getType());
      SmallVector<Location> locations(types.size(), location);
      auto replacement = nested.create<scf::WhileOp>(location, types, initial);
      if (Attribute origin = loop->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      Block *before = nested.createBlock(&replacement.getBefore(), {}, types, locations);
      Value condition = nested.create<gpu::CompareOp>(
          location, nested.getI1Type(), before->getArgument(0),
          loop.getUpperBound(), ComparePredicate::Lt);
      nested.create<scf::ConditionOp>(location, condition, before->getArguments());
      Block *after = nested.createBlock(&replacement.getAfter(), {}, types, locations);
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), after->getArgument(0));
      mapping.map(loop.getRegionIterArgs(), after->getArguments().drop_front());
      for (Operation &operation : loop.getBody()->without_terminator())
        nested.clone(operation, mapping);
      Value next = nested.create<gpu::BinaryOp>(
          location, types.front(), after->getArgument(0), loop.getStep(),
          BinaryOperator::Add);
      SmallVector<Value> yielded{next};
      for (Value value : cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands())
        yielded.push_back(mapping.lookupOrDefault(value));
      nested.create<scf::YieldOp>(location, yielded);
      return replacement;
    };
    // Unit steps keep the terminating increment in range as well as the body IV.
    bool unitStep = maximumStep && *maximumStep == 1;
    if (unitStep && fitsNativeLoopBound(loop.getLowerBound()) &&
        fitsNativeLoopBound(loop.getUpperBound())) {
      auto replacement = nativeLoop(builder);
      loop.replaceAllUsesWith(replacement.getResults());
      loop.erase();
      continue;
    }
    auto integer = dyn_cast<IntegerType>(loop.getInductionVar().getType());
    if (maximumStep && (!integer || !integer.isUnsigned()) &&
        isSpecializedLoopBound(loop.getLowerBound()) &&
        isSpecializedLoopBound(loop.getUpperBound())) {
      Value minimum = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), INT32_MIN));
      Value maximum = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(), INT32_MAX));
      // The last body IV is at most upper - 1; its increment must also fit.
      Value maximumUpper = builder.create<arith::ConstantOp>(
          location, loop.getInductionVar().getType(),
          builder.getIntegerAttr(loop.getInductionVar().getType(),
                                 int64_t{INT32_MAX} - *maximumStep + 1));
      Value condition;
      for (Value bound : {loop.getLowerBound(), loop.getUpperBound()}) {
        Value lower = builder.create<gpu::CompareOp>(
            location, builder.getI1Type(), bound, minimum, ComparePredicate::Ge);
        Value upper = builder.create<gpu::CompareOp>(
            location, builder.getI1Type(), bound,
            bound == loop.getUpperBound() ? maximumUpper : maximum,
            ComparePredicate::Le);
        Value fits = builder.create<gpu::BinaryOp>(
            location, builder.getI1Type(), lower, upper, BinaryOperator::LogicalAnd);
        condition = condition ? Value(builder.create<gpu::BinaryOp>(
                                    location, builder.getI1Type(), condition, fits,
                                    BinaryOperator::LogicalAnd))
                              : fits;
      }
      auto replacement = builder.create<scf::IfOp>(
          location, condition,
          [&](OpBuilder &nested, Location nestedLocation) {
            auto native = nativeLoop(nested);
            nested.create<scf::YieldOp>(nestedLocation, native.getResults());
          },
          [&](OpBuilder &nested, Location nestedLocation) {
            auto wide = wideLoop(nested);
            nested.create<scf::YieldOp>(nestedLocation, wide.getResults().drop_front());
          });
      loop.replaceAllUsesWith(replacement.getResults());
      loop.erase();
      continue;
    }
    // cuTile range always has an i32 IV. Unproven bounds retain explicit wide state.
    auto replacement = wideLoop(builder);
    loop.replaceAllUsesWith(replacement.getResults().drop_front());
    loop.erase();
  }
}

} // namespace

LogicalResult verifyCuTileProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) || failed(mlir::verify(module)) ? failure()
                                                       : verifyKernel(*kernel);
}

LogicalResult legalizeGPUProgram(ModuleOp module,
                                const gpu::TuningProfiles &profiles) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(formNativeTiles(*kernel, profiles)))
    return failure();
  realizeWideLoops(*kernel);
  if (failed(materializeClosedConfigs(*kernel)) ||
      failed(verifyCuTileProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::cutile

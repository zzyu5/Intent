#include "Legalization.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::triton::detail {
Value stripShapeOnly(Value value) {
  while (true) {
    if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = value.getDefiningOp<gpu::ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto splat = value.getDefiningOp<gpu::SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    return value;
  }
}

bool isTrueValue(Value value) {
  auto constant = stripShapeOnly(value).getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getType().isInteger(1) && integer.getValue().isOne();
}

bool isTritonAtomicAddType(Type type) {
  type = gpu::uniformElementType(type);
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 32 || integer.getWidth() == 64);
}

bool samePhysicalShape(Type lhs, Type rhs) {
  auto left = dyn_cast<gpu::FragmentType>(lhs);
  auto right = dyn_cast<gpu::FragmentType>(rhs);
  if (static_cast<bool>(left) != static_cast<bool>(right))
    return false;
  return !left ||
         (left.getShape() == right.getShape() &&
          left.getAxisMaps() == right.getAxisMaps() &&
          left.getValidity() == right.getValidity() &&
          left.getOwner() == right.getOwner());
}

std::optional<unsigned>
expandedGatherAxis(gpu::FragmentType source, gpu::FragmentType result,
                   unsigned selectedSourceAxis) {
  if (source.getOwner() != result.getOwner() ||
      source.getShape().size() >= result.getShape().size() ||
      selectedSourceAxis >= source.getShape().size())
    return std::nullopt;

  gpu::BroadcastProjection projection = gpu::queryBroadcastProjection(source, result);
  if (!projection.isExact())
    return std::nullopt;
  std::optional<unsigned> selected;
  for (auto [resultIndex, sourceIndex] :
       llvm::enumerate(projection.targetToSource)) {
    if (sourceIndex) {
      if (*sourceIndex == selectedSourceAxis)
        selected = resultIndex;
      continue;
    }
    auto extent = dyn_cast<gpu::PhysicalExprAttr>(result.getShape()[resultIndex]);
    if (!extent ||
        extent.getKind() !=
            gpu::PhysicalExprKind::Constant ||
        extent.getValue() != 1)
      return std::nullopt;
  }
  return selected;
}

FailureOr<Value> zeroLike(OpBuilder &builder, Location location, Type type) {
  Type scalarType = gpu::uniformElementType(type);
  Value zero;
  if (scalarType.isIndex())
    zero = builder.create<arith::ConstantIndexOp>(location, 0);
  else if (auto integer = dyn_cast<IntegerType>(scalarType))
    zero = builder.create<arith::ConstantOp>(
        location, integer, builder.getIntegerAttr(integer, 0));
  else
    return failure();
  if (auto fragment = dyn_cast<gpu::FragmentType>(type))
    return Value(builder.create<gpu::SplatOp>(location, fragment, zero));
  return zero;
}

std::optional<int64_t> staticGatherCoordinate(gpu::GatherOp gather) {
  if (gather.getCoordinates().size() != 1 ||
      gather.getSourceAxes().size() != 1 || !gather.getValid() ||
      !isTrueValue(gather.getValid()))
    return std::nullopt;
  Value coordinate = stripShapeOnly(gather.getCoordinates().front());
  while (auto cast = coordinate.getDefiningOp<gpu::CastOp>())
    coordinate = cast.getValue();
  auto constant = coordinate.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  if (!integer || (integer.getInt() != 0 && integer.getInt() != 1))
    return std::nullopt;

  auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
  if (!source || !result || source.getShape().size() != result.getShape().size() + 1 ||
      gather.getSourceAxes().front() !=
          static_cast<int64_t>(source.getShape().size() - 1) ||
      source.getElementType() != result.getElementType() ||
      source.getValidity() != result.getValidity() ||
      source.getOwner() != result.getOwner())
    return std::nullopt;
  auto trailing = dyn_cast<gpu::PhysicalExprAttr>(
      source.getShape()[source.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() !=
          gpu::PhysicalExprKind::Constant ||
      trailing.getValue() != 2 ||
      !std::equal(result.getShape().begin(), result.getShape().end(),
                  source.getShape().begin()) ||
      !std::equal(result.getAxisMaps().begin(), result.getAxisMaps().end(),
                  source.getAxisMaps().begin()))
    return std::nullopt;
  return integer.getInt();
}

bool belongsToSplitGatherPair(gpu::GatherOp gather) {
  std::optional<int64_t> coordinate = staticGatherCoordinate(gather);
  if (!coordinate || !gather->getBlock())
    return false;
  for (Operation *user : gather.getSource().getUsers()) {
    auto complement = dyn_cast<gpu::GatherOp>(user);
    std::optional<int64_t> other = complement
                                       ? staticGatherCoordinate(complement)
                                       : std::optional<int64_t>();
    if (complement && complement != gather &&
        complement->getBlock() == gather->getBlock() && other &&
        *other == 1 - *coordinate &&
        complement.getResult().getType() == gather.getResult().getType())
      return true;
  }
  return false;
}

LogicalResult legalizeSplitGatherPairs(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  llvm::SmallPtrSet<Operation *, 8> rewritten;
  bool changed = false;
  for (gpu::GatherOp low : gathers) {
    if (rewritten.contains(low.getOperation()) || !low->getBlock() ||
        staticGatherCoordinate(low) != std::optional<int64_t>(0))
      continue;
    for (gpu::GatherOp high : gathers) {
      if (rewritten.contains(high.getOperation()) || !high->getBlock() ||
          high->getBlock() != low->getBlock() ||
          high.getSource() != low.getSource() ||
          high.getResult().getType() != low.getResult().getType() ||
          staticGatherCoordinate(high) != std::optional<int64_t>(1))
        continue;
      Operation *anchor = low->isBeforeInBlock(high) ? low.getOperation()
                                                    : high.getOperation();
      OpBuilder builder(anchor);
      auto split = builder.create<SplitOp>(
          anchor->getLoc(), low.getResult().getType(), high.getResult().getType(),
          low.getSource());
      low.getResult().replaceAllUsesWith(split.getLow());
      high.getResult().replaceAllUsesWith(split.getHigh());
      rewritten.insert(low.getOperation());
      rewritten.insert(high.getOperation());
      changed = true;
      break;
    }
  }
  if (changed) {
    for (gpu::GatherOp gather : gathers)
      if (rewritten.contains(gather.getOperation()))
        gather.erase();
    gpu::eraseDeadPhysicalValues(kernel);
  }
  return success();
}

LogicalResult legalizeMaskedGather(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (!gather.getValid() || belongsToSplitGatherPair(gather))
      continue;
    auto scalarConstant = [](Value value) -> arith::ConstantOp {
      while (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
        value = broadcast.getValue();
      while (auto splat = value.getDefiningOp<gpu::SplatOp>())
        value = splat.getValue();
      while (auto cast = value.getDefiningOp<gpu::CastOp>())
        value = cast.getValue();
      return value.getDefiningOp<arith::ConstantOp>();
    };
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (source && !source.getShape().empty() &&
        gather.getCoordinates().size() == source.getShape().size() &&
        llvm::all_of(gather.getCoordinates(), [](Value coordinate) {
          Type element = coordinate.getType();
          if (auto fragment = dyn_cast<gpu::FragmentType>(element))
            element = fragment.getElementType();
          return isa<IndexType, IntegerType>(element);
        }) &&
        (result || gather.getValid().getType().isInteger(1))) {
      OpBuilder builder(gather);
      Location location = gather.getLoc();
      Type indexType = builder.getIndexType();
      Type predicateType = builder.getI1Type();
      if (result) {
        indexType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        predicateType = gpu::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto projectIndex = [&](Value value) -> FailureOr<Value> {
        if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType())) {
          if (!result)
            return failure();
          if (!fragment.getElementType().isIndex()) {
            auto converted = gpu::FragmentType::get(
                kernel.getContext(), builder.getIndexType(), fragment.getShape(),
                fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
            value = builder.create<gpu::CastOp>(location, converted, value);
          }
        } else if (!value.getType().isIndex()) {
          value = builder.create<gpu::CastOp>(location, builder.getIndexType(), value);
        }
        if (result)
          return gpu::projectPhysicalValueToSchema(
              builder, location, value, cast<gpu::FragmentType>(indexType));
        return value;
      };
      SmallVector<Value> coordinates(source.getShape().size());
      for (auto [axis, value] :
           llvm::zip(gather.getSourceAxes(), gather.getCoordinates())) {
        FailureOr<Value> projected = projectIndex(value);
        if (failed(projected))
          return gather.emitOpError("gather coordinate has no exact result-axis projection");
        coordinates[axis] = *projected;
      }
      Value sourceValue = gather.getSource();
      Value coordinate = coordinates.front();
      Value valid = gather.getValid();
      if (result && valid.getType() != predicateType) {
        auto projected = gpu::projectPhysicalValueToSchema(
            builder, location, valid, cast<gpu::FragmentType>(predicateType));
        if (failed(projected))
          return gather.emitOpError("gather validity has no exact result-axis projection");
        valid = *projected;
      }
      if (source.getShape().size() > 1) {
        auto elements = cast<gpu::PhysicalExprAttr>(source.getShape()[0]);
        auto first = projectIndex(builder.create<arith::ConstantIndexOp>(location, 0));
        if (failed(first))
          return failure();
        coordinate = *first;
        for (auto [axis, value] : llvm::enumerate(coordinates)) {
          auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
          if (axis)
            elements = gpu::PhysicalExprAttr::get(
                kernel.getContext(),
                gpu::PhysicalExprKind::Multiply, 0,
                builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
          Value size = builder.create<gpu::PhysicalExprOp>(
              location, builder.getIndexType(), extent);
          auto projectedSize = projectIndex(size);
          if (failed(projectedSize))
            return failure();
          coordinate = builder.create<gpu::BinaryOp>(
              location, indexType, coordinate, *projectedSize, BinaryOperator::Multiply);
          coordinate = builder.create<gpu::BinaryOp>(
              location, indexType, coordinate, value, BinaryOperator::Add);
        }
        if (auto count = evaluateCompileTimeExpression(elements))
          elements = gpu::PhysicalExprAttr::get(
              kernel.getContext(),
              gpu::PhysicalExprKind::Constant, *count,
              builder.getStringAttr(""), builder.getArrayAttr({}));
        auto mapping = gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatType = gpu::FragmentType::get(
            kernel.getContext(), source.getElementType(),
            builder.getArrayAttr({elements}), builder.getArrayAttr({mapping}),
            source.getValidity(), source.getOwner());
        auto reassociation = gpu::inferReshapeReassociation(source, flatType);
        if (failed(reassociation))
          return gather.emitOpError("scalar gather has no exact row-major linearization");
        sourceValue = builder.create<gpu::ReshapeOp>(
            location, flatType, sourceValue, *reassociation);
        Value size = builder.create<gpu::PhysicalExprOp>(
            location, builder.getIndexType(), elements);
        auto projectedSize = projectIndex(size);
        if (failed(projectedSize))
          return failure();
        Value lower = builder.create<gpu::CompareOp>(
            location, predicateType, coordinate, *first, ComparePredicate::Ge);
        Value upper = builder.create<gpu::CompareOp>(
            location, predicateType, coordinate, *projectedSize, ComparePredicate::Lt);
        valid = builder.create<gpu::BinaryOp>(
            location, predicateType, valid, lower, BinaryOperator::LogicalAnd);
        valid = builder.create<gpu::BinaryOp>(
            location, predicateType, valid, upper, BinaryOperator::LogicalAnd);
      }
      FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinate.getType());
      if (failed(zero))
        return gather.emitOpError("scalar gather coordinate has no integral zero");
      Value safeIndex = builder.create<gpu::SelectOp>(
          gather.getLoc(), coordinate.getType(), valid, coordinate, *zero);
      Type loadedType = gather.getResult().getType();
      if (result && result.getShape().size() > 1) {
        auto elements = cast<gpu::PhysicalExprAttr>(result.getShape()[0]);
        for (Attribute extent : result.getShape().getValue().drop_front())
          elements = gpu::PhysicalExprAttr::get(
              kernel.getContext(),
              gpu::PhysicalExprKind::Multiply, 0,
              builder.getStringAttr(""), builder.getArrayAttr({elements, extent}));
        auto mapping = gpu::AxisMapAttr::get(
            kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
        auto flatIndex = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({elements}),
            builder.getArrayAttr({mapping}), result.getValidity(), result.getOwner());
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(safeIndex.getType()), flatIndex);
        if (failed(relation))
          return gather.emitOpError("gather result has no exact row-major linearization");
        safeIndex = builder.create<gpu::ReshapeOp>(location, flatIndex, safeIndex, *relation);
        loadedType = gpu::FragmentType::get(
            kernel.getContext(), result.getElementType(), flatIndex.getShape(),
            flatIndex.getAxisMaps(), result.getValidity(), result.getOwner());
      }
      auto loaded = builder.create<gpu::GatherOp>(
          gather.getLoc(), loadedType, sourceValue,
          ValueRange{safeIndex}, Value(), Value(), ArrayRef<int64_t>{0});
      Value loadedValue = loaded.getResult();
      if (loadedType != gather.getResult().getType()) {
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(loadedType), result);
        if (failed(relation))
          return gather.emitOpError("linear gather has no exact result reassociation");
        loadedValue = builder.create<gpu::ReshapeOp>(location, result, loadedValue, *relation);
      }
      auto selected = builder.create<gpu::SelectOp>(
          gather.getLoc(), gather.getResult().getType(), valid,
          loadedValue, gather.getFill());
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        selected->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(selected.getResult());
      gather.erase();
      continue;
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() < result.getShape().size()) {
      std::optional<unsigned> selectedAxis = expandedGatherAxis(
          source, result, gather.getSourceAxes().front());
      if (selectedAxis) {
        OpBuilder builder(gather);
        auto expandedSourceType = gpu::FragmentType::get(
            gather.getContext(), source.getElementType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        Value expandedSource = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), expandedSourceType, gather.getSource());
        Value coordinate = gather.getCoordinates().front();
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), coordinate.getType(), result.getShape(),
            result.getAxisMaps(), result.getValidity(), result.getOwner());
        if (auto coordinateFragment =
                dyn_cast<gpu::FragmentType>(coordinate.getType())) {
          coordinateType = gpu::FragmentType::get(
              gather.getContext(), coordinateFragment.getElementType(),
              result.getShape(), result.getAxisMaps(), result.getValidity(),
              result.getOwner());
        }
        coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, coordinate);
        FailureOr<Value> zero =
            zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinateType, gather.getValid(), coordinate,
              *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), result, expandedSource, ValueRange{safeIndex},
              Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(*selectedAxis)});
          auto selected = builder.create<gpu::SelectOp>(
              gather.getLoc(), result, gather.getValid(), safeGather.getResult(),
              gather.getFill());
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            selected->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(selected.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        gather.getSourceAxes().size() == 1 && source && result &&
        source.getShape().size() == result.getShape().size() + 1) {
      unsigned selectedAxis = gather.getSourceAxes().front();
      auto selectedExtent =
          selectedAxis < source.getShape().size()
              ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[selectedAxis])
              : gpu::PhysicalExprAttr();
      bool compatible = selectedExtent &&
          ((selectedExtent.getKind() == gpu::PhysicalExprKind::Constant &&
            selectedExtent.getValue() > 0) ||
           selectedExtent.getKind() == gpu::PhysicalExprKind::Parameter);
      unsigned resultAxis = 0;
      for (unsigned sourceAxis = 0;
           compatible && sourceAxis < source.getShape().size(); ++sourceAxis) {
        if (sourceAxis == selectedAxis)
          continue;
        compatible = resultAxis < result.getShape().size() &&
                     source.getShape()[sourceAxis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
      if (compatible) {
        OpBuilder builder(gather);
        SmallVector<Attribute> selectedShape(source.getShape().begin(),
                                             source.getShape().end());
        selectedShape[selectedAxis] = gpu::PhysicalExprAttr::get(
            gather.getContext(),
            gpu::PhysicalExprKind::Constant, 1,
            builder.getStringAttr(""), builder.getArrayAttr({}));
        auto coordinateType = gpu::FragmentType::get(
            gather.getContext(), gpu::uniformElementType(gather.getCoordinates().front().getType()),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto predicateType = gpu::FragmentType::get(
            gather.getContext(), builder.getI1Type(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            source.getValidity(), source.getOwner());
        auto selectedResultType = gpu::FragmentType::get(
            gather.getContext(), result.getElementType(),
            builder.getArrayAttr(selectedShape), source.getAxisMaps(),
            result.getValidity(), result.getOwner());
        Value coordinate = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), coordinateType, gather.getCoordinates().front());
        Value valid = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), predicateType, gather.getValid());
        Value fill = builder.create<gpu::BroadcastOp>(
            gather.getLoc(), selectedResultType, gather.getFill());
        FailureOr<Value> zero = zeroLike(builder, gather.getLoc(), coordinateType);
        if (succeeded(zero)) {
          Value safeIndex = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinateType, valid, coordinate, *zero);
          auto safeGather = builder.create<gpu::GatherOp>(
              gather.getLoc(), selectedResultType, gather.getSource(),
              ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
          Value selected = builder.create<gpu::SelectOp>(
              gather.getLoc(), selectedResultType, valid, safeGather.getResult(),
              fill);
          FailureOr<ArrayAttr> reassociation =
              gpu::inferReshapeReassociation(selectedResultType, result);
          if (failed(reassociation))
            return gather.emitOpError(
                "gather fallback has no exact row-major reassociation");
          auto reshaped = builder.create<gpu::ReshapeOp>(
              gather.getLoc(), result, selected, *reassociation);
          if (Attribute origin = gather->getAttr(gpu::originAttr))
            reshaped->setAttr(gpu::originAttr, origin);
          gather.getResult().replaceAllUsesWith(reshaped.getResult());
          gather.erase();
          continue;
        }
      }
    }
    if (gather.getCoordinates().size() == 1 &&
        isa<gpu::FragmentType>(gather.getCoordinates().front().getType()) &&
        gather.getSourceAxes().size() == 1 && source) {
      unsigned axis = gather.getSourceAxes().front();
      auto coordinate = scalarConstant(gather.getCoordinates().front());
      auto predicate = scalarConstant(gather.getValid());
      auto extent = axis < source.getShape().size()
                        ? dyn_cast<gpu::PhysicalExprAttr>(source.getShape()[axis])
                        : gpu::PhysicalExprAttr();
      auto coordinateValue =
          coordinate ? dyn_cast<IntegerAttr>(coordinate.getValue()) : IntegerAttr();
      auto predicateValue = predicate ? dyn_cast<IntegerAttr>(predicate.getValue())
                                      : IntegerAttr();
      if (extent &&
          extent.getKind() == gpu::PhysicalExprKind::Constant &&
          coordinateValue && coordinateValue.getInt() >= 0 &&
          coordinateValue.getInt() < extent.getValue() && predicateValue &&
          predicateValue.getValue().isOne()) {
        OpBuilder builder(gather);
        auto replacement = builder.create<gpu::GatherOp>(
            gather.getLoc(), gather.getResult().getType(), gather.getSource(),
            gather.getCoordinates(), Value(), Value(), gather.getSourceAxes());
        if (Attribute origin = gather->getAttr(gpu::originAttr))
          replacement->setAttr(gpu::originAttr, origin);
        gather.getResult().replaceAllUsesWith(replacement.getResult());
        gather.erase();
        continue;
      }
    }
    if (gather.getCoordinates().size() != 1)
      continue;
    OpBuilder builder(gather);
    Value coordinate = gather.getCoordinates().front();
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType())) {
      auto predicateType = dyn_cast<gpu::FragmentType>(gather.getValid().getType());
      Type indexElement = gpu::uniformElementType(coordinate.getType());
      if (!predicateType || !isa<IntegerType, IndexType>(indexElement))
        continue;
      if (isa<gpu::FragmentType>(coordinate.getType()) &&
          (!source || !result || gather.getSourceAxes().size() != 1 ||
           source.getShape().size() != result.getShape().size()))
        continue;
      auto coordinateType = gpu::FragmentType::get(
          gather.getContext(), indexElement, predicateType.getShape(),
          predicateType.getAxisMaps(), predicateType.getValidity(),
          predicateType.getOwner());
      auto projected = gpu::projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, coordinateType);
      if (failed(projected))
        continue;
      coordinate = *projected;
    }
    if (!samePhysicalShape(gather.getValid().getType(), coordinate.getType()) ||
        !samePhysicalShape(gather.getValid().getType(),
                           gather.getResult().getType()))
      continue;
    FailureOr<Value> zero =
        zeroLike(builder, gather.getLoc(), coordinate.getType());
    if (failed(zero))
      continue;
    Value safeIndex = builder.create<gpu::SelectOp>(
        gather.getLoc(), coordinate.getType(), gather.getValid(), coordinate,
        *zero);
    auto safeGather = builder.create<gpu::GatherOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        ValueRange{safeIndex}, Value(), Value(), gather.getSourceAxes());
    auto selected = builder.create<gpu::SelectOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getValid(),
        safeGather.getResult(), gather.getFill());
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      selected->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(selected.getResult());
    gather.erase();
  }
  return success();
}

LogicalResult legalizeExpandingGathers(func::FuncOp kernel) {
  auto [nextSource, nextDimension] = gpu::nextPhysicalAxisIdentities(kernel);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    if (gather.getValid() || gather.getCoordinates().size() != 1 ||
        gather.getSourceAxes().size() != 1)
      continue;
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    auto indices = dyn_cast<gpu::FragmentType>(
        gather.getCoordinates().front().getType());
    if (!source || !result || !indices || source.getShape().size() < 2 ||
        source.getShape().size() != result.getShape().size() ||
        indices.getShape() != result.getShape() ||
        indices.getAxisMaps() != result.getAxisMaps())
      continue;
    unsigned axis = gather.getSourceAxes().front();
    bool compatible = true;
    for (unsigned position = 0; position < source.getShape().size(); ++position)
      compatible &= position == axis ||
                    (source.getShape()[position] == result.getShape()[position] &&
                     source.getAxisMaps()[position] == result.getAxisMaps()[position]);
    auto sourceExtent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
    auto resultExtent = cast<gpu::PhysicalExprAttr>(result.getShape()[axis]);
    if (!compatible || sourceExtent == resultExtent ||
        !isTritonFragmentExtent(sourceExtent) ||
        !isTritonFragmentExtent(resultExtent))
      continue;
    auto constantExtent = [](gpu::PhysicalExprAttr extent) {
      return extent.getKind() ==
             gpu::PhysicalExprKind::Constant;
    };
    bool constantShape = constantExtent(sourceExtent) &&
                         constantExtent(resultExtent);
    if (constantShape && resultExtent.getValue() <= sourceExtent.getValue())
      continue;

    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto product = [&](ArrayRef<Attribute> shape) {
      auto extent = gpu::PhysicalExprAttr::get(
          kernel.getContext(),
          gpu::PhysicalExprKind::Constant, 1,
          builder.getStringAttr(""), builder.getArrayAttr({}));
      if (shape.empty())
        return extent;
      extent = cast<gpu::PhysicalExprAttr>(shape.front());
      for (Attribute dimension : shape.drop_front())
        extent = gpu::PhysicalExprAttr::get(
            kernel.getContext(),
            gpu::PhysicalExprKind::Multiply, 0,
            builder.getStringAttr(""), builder.getArrayAttr({extent, dimension}));
      return extent;
    };
    auto flatSourceAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto flatResultAxis = gpu::AxisMapAttr::get(
        kernel.getContext(), nextSource++, 0, nextDimension++, 0, true);
    auto sourceElements = product(source.getShape().getValue());
    auto resultElements = product(result.getShape().getValue());
    auto flatType = [&](gpu::FragmentType original, Type element,
                        gpu::PhysicalExprAttr extent, gpu::AxisMapAttr mapping) {
      return gpu::FragmentType::get(
          kernel.getContext(), element, builder.getArrayAttr({extent}),
          builder.getArrayAttr({mapping}), original.getValidity(),
          original.getOwner());
    };
    auto flatSourceType = flatType(source, source.getElementType(),
                                   sourceElements, flatSourceAxis);
    auto flatResultType = flatType(result, result.getElementType(),
                                   resultElements, flatResultAxis);
    auto flatIndexType = flatType(indices, builder.getIndexType(),
                                  resultElements, flatResultAxis);
    auto linearize = [&](OpBuilder &nested) -> FailureOr<Value> {
      auto reshape = [&](Value value,
                         gpu::FragmentType target) -> FailureOr<Value> {
        auto relation = gpu::inferReshapeReassociation(
            cast<gpu::FragmentType>(value.getType()), target);
        if (failed(relation))
          return failure();
        return Value(nested.create<gpu::ReshapeOp>(location, target, value,
                                                  *relation));
      };
      auto flatOriginalIndexType = flatType(
          indices, indices.getElementType(), resultElements, flatResultAxis);
      FailureOr<Value> flatSource = reshape(gather.getSource(), flatSourceType);
      FailureOr<Value> flatIndices =
          reshape(gather.getCoordinates().front(), flatOriginalIndexType);
      if (failed(flatSource) || failed(flatIndices))
        return failure();
      Value index = *flatIndices;
      if (index.getType() != flatIndexType)
        index = nested.create<gpu::CastOp>(location, flatIndexType, index);
      auto extentValue = [&](gpu::PhysicalExprAttr extent) -> Value {
        return nested.create<gpu::PhysicalExprOp>(
            location, nested.getIndexType(), extent);
      };
      auto broadcastExtent = [&](gpu::PhysicalExprAttr extent) -> Value {
        return nested.create<gpu::BroadcastOp>(location, flatIndexType,
                                               extentValue(extent));
      };
      auto binary = [&](Value lhs, Value rhs, BinaryOperator kind) -> Value {
        return nested.create<gpu::BinaryOp>(location, flatIndexType, lhs, rhs,
                                            kind);
      };
      Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
      Value one = nested.create<arith::ConstantIndexOp>(location, 1);
      Value count = extentValue(resultElements);
      Value ordinal = nested.create<gpu::MakeRangeOp>(
          location, flatIndexType, zero, count, one, zero, count,
          flatResultAxis.getSourceId(), flatResultAxis.getSourceAxis(), true);
      Value inner = broadcastExtent(
          product(result.getShape().getValue().drop_front(axis + 1)));
      Value sourceStride = binary(broadcastExtent(sourceExtent), inner,
                                   BinaryOperator::Multiply);
      Value resultStride = binary(broadcastExtent(resultExtent), inner,
                                   BinaryOperator::Multiply);
      Value outer = binary(ordinal, resultStride, BinaryOperator::FloorDivide);
      Value tail = binary(ordinal, inner, BinaryOperator::Remainder);
      Value offset = binary(outer, sourceStride, BinaryOperator::Multiply);
      offset = binary(offset, binary(index, inner, BinaryOperator::Multiply),
                      BinaryOperator::Add);
      offset = binary(offset, tail, BinaryOperator::Add);
      Value selected = nested.create<gpu::GatherOp>(
          location, flatResultType, *flatSource, ValueRange{offset}, Value(),
          Value(), ArrayRef<int64_t>{0});
      return reshape(selected, result);
    };

    // Expanding gathers can require cross-warp exchange.  A linear gather
    // retains that meaning without the rank-dependent warp-local rewrite.
    Value replacement;
    if (constantShape) {
      FailureOr<Value> linear = linearize(builder);
      if (failed(linear))
        return gather.emitOpError("expanding gather has no exact linear reshape");
      replacement = *linear;
    } else {
      Value sourceSize = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), sourceExtent);
      Value resultSize = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), resultExtent);
      Value expanding = builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), resultSize, sourceSize,
          ComparePredicate::Gt);
      auto conditional = builder.create<scf::IfOp>(
          location, TypeRange{result}, expanding, /*withElseRegion=*/true);
      OpBuilder linearBuilder = conditional.getThenBodyBuilder();
      FailureOr<Value> linear = linearize(linearBuilder);
      if (failed(linear))
        return gather.emitOpError("expanding gather has no exact linear reshape");
      linearBuilder.create<scf::YieldOp>(location, *linear);
      OpBuilder originalBuilder = conditional.getElseBodyBuilder();
      Operation *original = originalBuilder.clone(*gather.getOperation());
      originalBuilder.create<scf::YieldOp>(location, original->getResults());
      replacement = conditional.getResult(0);
    }
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement.getDefiningOp()->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement);
    gather.erase();
  }
  return success();
}

bool isAddCombine(gpu::ScatterReduceOp scatter) {
  return gpu::queryBinaryCombineKind(scatter.getCombine()) ==
         BinaryOperator::Add;
}

LogicalResult legalizeScatterAdd(func::FuncOp kernel) {
  SmallVector<gpu::ScatterReduceOp> scatters;
  kernel.walk([&](gpu::ScatterReduceOp scatter) { scatters.push_back(scatter); });
  for (gpu::ScatterReduceOp scatter : scatters) {
    if (!isAddCombine(scatter) || !isTritonAtomicAddType(scatter.getValue().getType()))
      continue;
    OpBuilder builder(scatter);
    auto atomic = builder.create<gpu::AtomicRMWOp>(
        scatter.getLoc(), scatter.getValue().getType(), scatter.getResource(),
        scatter.getCoordinates(), scatter.getValue(), scatter.getValid(),
        AtomicRMWKind::Add, AtomicOrdering::Relaxed, scatter.getSharing(),
        scatter.getSourceAxes());
    if (Attribute origin = scatter->getAttr(gpu::originAttr))
      atomic->setAttr(gpu::originAttr, origin);
    scatter.erase();
  }
  return success();
}

LogicalResult legalizeCollectiveCallbacks(func::FuncOp kernel) {
  SmallVector<Operation *> collectives;
  kernel.walk([&](Operation *operation) {
    if (isa<gpu::ReduceOp, gpu::ScanOp>(operation))
      collectives.push_back(operation);
  });
  for (Operation *operation : collectives) {
    auto reduce = dyn_cast<gpu::ReduceOp>(operation);
    auto scan = dyn_cast<gpu::ScanOp>(operation);
    ValueRange sources = reduce ? reduce.getSources() : scan.getSources();
    ValueRange identities = reduce ? reduce.getIdentities() : scan.getIdentities();
    if ((reduce && (reduce.getAxes().size() != 1 || reduce.getCaptures().size())) ||
        (scan && scan.getCaptures().size()))
      return operation->emitOpError("native collective requires one axis and a capture-free callback");
    // ODS inferred builders treat failure as a construction error. Diagnose
    // unsupported native schemas before invoking that builder, and retain the
    // result contract of the shared operation being replaced.
    SmallVector<Type> resultTypes;
    int64_t axis = reduce ? reduce.getAxes().front() : scan.getAxis();
    if (failed(gpu::inferScalarCollectiveResultTypes(
            operation->getLoc(), sources, axis, bool(scan), resultTypes)))
      return failure();
    if (!llvm::equal(resultTypes, operation->getResultTypes()))
      return operation->emitOpError(
          "native collective cannot preserve the shared result schema");
    Region &region = reduce ? reduce.getCombine() : scan.getCombine();
    OpBuilder builder(operation);
    Operation *native;
    Region *combine;
    if (reduce) {
      auto replacement = builder.create<ReduceOp>(operation->getLoc(),
          sources, identities, axis, false);
      native = replacement;
      combine = &replacement.getCombine();
    } else {
      auto replacement = builder.create<ScanOp>(operation->getLoc(),
          sources, identities, axis, scan.getReverse());
      native = replacement;
      combine = &replacement.getCombine();
    }
    if (Attribute origin = operation->getAttr(gpu::originAttr)) native->setAttr(gpu::originAttr, origin);
    if (failed(gpu::scalarizeElementwiseCallback(region, *combine)))
      return failure();
    SmallVector<Value> results(native->getResults());
    if (scan && !scan.getInclusive()) {
      builder.setInsertionPointAfter(native);
      Location location = scan.getLoc();
      for (unsigned component = 0; component < sources.size(); ++component) {
        auto type = dyn_cast<gpu::FragmentType>(results[component].getType());
        if (!type || scan.getAxis() >= type.getShape().size())
          return scan.emitOpError("exclusive scan requires a ranked physical result");
        auto axis = cast<gpu::AxisMapAttr>(type.getAxisMaps()[scan.getAxis()]);
        auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[scan.getAxis()]);
        auto ordinalAxis = gpu::AxisMapAttr::get(
            kernel.getContext(), axis.getSourceId(), axis.getSourceAxis(),
            axis.getDimensionId(), 0, axis.getDerived());
        auto rangeType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(),
            builder.getArrayAttr({extent}), builder.getArrayAttr({ordinalAxis}),
            type.getValidity(), type.getOwner());
        auto indexType = gpu::FragmentType::get(
            kernel.getContext(), builder.getIndexType(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        auto predicateType = gpu::FragmentType::get(
            kernel.getContext(), builder.getI1Type(), type.getShape(),
            type.getAxisMaps(), type.getValidity(), type.getOwner());
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        Value size = builder.create<gpu::PhysicalExprOp>(
            location, builder.getIndexType(), extent);
        Value ordinal = builder.create<gpu::MakeRangeOp>(
            location, rangeType, zero, size, one, zero, size,
            axis.getSourceId(), axis.getSourceAxis(), axis.getDerived());
        ordinal = builder.create<gpu::BroadcastOp>(location, indexType, ordinal);
        Value step = builder.create<gpu::BroadcastOp>(location, indexType, one);
        Value first = builder.create<gpu::BroadcastOp>(location, indexType, zero);
        Value shifted = builder.create<gpu::BinaryOp>(
            location, indexType, ordinal, step,
            scan.getReverse() ? BinaryOperator::Add : BinaryOperator::Subtract);
        Value bound = scan.getReverse()
                          ? Value(builder.create<gpu::BroadcastOp>(location, indexType, size))
                          : first;
        Value valid = builder.create<gpu::CompareOp>(
            location, predicateType, shifted, bound,
            scan.getReverse() ? ComparePredicate::Lt : ComparePredicate::Ge);
        Value safeIndex = builder.create<gpu::SelectOp>(
            location, indexType, valid, shifted, first);
        Value prefix = builder.create<gpu::GatherOp>(
            location, type, results[component], ValueRange{safeIndex},
            Value(), Value(), ArrayRef<int64_t>{static_cast<int64_t>(scan.getAxis())});
        Value identity = builder.create<gpu::BroadcastOp>(
            location, type, scan.getIdentities()[component]);
        results[component] = builder.create<gpu::SelectOp>(
            location, type, valid, prefix, identity);
      }
    }
    operation->replaceAllUsesWith(results);
    operation->erase();
  }
  return success();
}

void foldIntegerScanTails(func::FuncOp kernel) {
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  bool changed = false;
  for (gpu::GatherOp gather : gathers) {
    auto type = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    if (!type || type.getShape().size() != 1 ||
        (!type.getElementType().isInteger(32) &&
         !type.getElementType().isInteger(64)) ||
        gather.getType() != type.getElementType() ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
        gather.getCoordinates().size() != 1 ||
        (gather.getValid() &&
         (!gather.getValid().getType().isInteger(1) || !gather.getFill())))
      continue;
    auto extent = cast<gpu::PhysicalExprAttr>(type.getShape()[0]);
    bool positive = extent.getKind() ==
                        gpu::PhysicalExprKind::Constant &&
                    extent.getValue() > 0;
    if (extent.getKind() ==
        gpu::PhysicalExprKind::Parameter) {
      auto parameter = gpu::queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      positive = succeeded(parameter) && llvm::all_of(
          parameter->getCandidates().asArrayRef(),
          [](int64_t candidate) { return candidate > 0; });
    }
    if (!positive)
      continue;
    Value index = gather.getCoordinates().front();
    auto coordinate = gpu::queryLaunchExpression(index);
    bool last = coordinate &&
                coordinate.getKind() ==
                    gpu::PhysicalExprKind::Constant &&
                extent.getKind() == coordinate.getKind() &&
                coordinate.getValue() == extent.getValue() - 1;
    if (auto subtract = index.getDefiningOp<gpu::BinaryOp>();
        subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
      auto one = gpu::queryLaunchExpression(subtract.getRhs());
      last |= gpu::queryLaunchExpression(subtract.getLhs()) == extent && one &&
              one.getKind() ==
                  gpu::PhysicalExprKind::Constant &&
              one.getValue() == 1;
    }
    if (!last)
      continue;

    Value source = gather.getSource();
    Value base;
    auto addition = source.getDefiningOp<gpu::BinaryOp>();
    if (addition && addition.getOperatorKind() == BinaryOperator::Add) {
      auto scalar = [&](Value value) -> Value {
        if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
          value = broadcast.getValue();
        else if (auto splat = value.getDefiningOp<gpu::SplatOp>())
          value = splat.getValue();
        return value.getType() == type.getElementType() ? value : Value();
      };
      if ((base = scalar(addition.getLhs())))
        source = addition.getRhs();
      else if ((base = scalar(addition.getRhs())))
        source = addition.getLhs();
    }
    auto scan = source.getDefiningOp<gpu::ScanOp>();
    if (!scan || scan.getSources().size() != 1 || scan.getIdentities().size() != 1 ||
        scan.getCaptures().size() || scan.getAxis() != 0 || !scan.getInclusive() ||
        scan.getReverse() || source.getType() != type ||
        !gpu::isLiteralZeroProjection(scan.getIdentities().front()))
      continue;
    Block &body = scan.getCombine().front();
    auto combine = dyn_cast<gpu::BinaryOp>(body.front());
    auto structured = cast<StructuredOpInterface>(scan.getOperation());
    if (!llvm::hasSingleElement(body.without_terminator()) || !combine ||
        combine.getLhs() != structured.getCombineLhs().front() ||
        gpu::queryBinaryCombineKind(scan.getCombine()) != BinaryOperator::Add)
      continue;

    // A scalar cross-warp gather materializes the whole prefix in shared memory.
    // Integer addition gives the same terminal value through a scalar reduction.
    OpBuilder builder(gather);
    Value zero = builder.create<arith::ConstantOp>(
        gather.getLoc(), builder.getZeroAttr(type.getElementType()));
    auto reduced = builder.create<gpu::ReduceOp>(gather.getLoc(),
        TypeRange{type.getElementType()}, scan.getSources(), ValueRange{zero},
        ValueRange{}, ArrayRef<int64_t>{0});
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      reduced->setAttr(gpu::originAttr, origin);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *scalarBody = new Block();
      reduced.getCombine().push_back(scalarBody);
      Value lhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      Value rhs = scalarBody->addArgument(type.getElementType(), gather.getLoc());
      builder.setInsertionPointToEnd(scalarBody);
      auto sum = builder.create<gpu::BinaryOp>(
          gather.getLoc(), type.getElementType(), lhs, rhs, BinaryOperator::Add);
      sum->setAttrs(combine->getAttrs());
      builder.create<gpu::YieldOp>(gather.getLoc(), sum.getResult());
    }
    Value result = reduced.getResult(0);
    if (base) {
      auto sum = builder.create<gpu::BinaryOp>(
          gather.getLoc(), type.getElementType(), base, result,
          BinaryOperator::Add);
      sum->setAttrs(addition->getAttrs());
      result = sum;
    }
    if (gather.getValid())
      result = builder.create<gpu::SelectOp>(
          gather.getLoc(), type.getElementType(), gather.getValid(), result,
          gather.getFill());
    gather.getResult().replaceAllUsesWith(result);
    gather.erase();
    changed = true;
  }
  if (changed)
    gpu::eraseDeadPhysicalValues(kernel);
}

LogicalResult legalizeLargeScalarGathers(func::FuncOp kernel) {
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  SmallVector<gpu::GatherOp> gathers;
  kernel.walk([&](gpu::GatherOp gather) { gathers.push_back(gather); });
  for (gpu::GatherOp gather : gathers) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || !isa<IntegerType, FloatType>(source.getElementType()) ||
        gather.getCoordinates().empty() ||
        gather.getCoordinates().size() != gather.getSourceAxes().size())
      continue;
    uint64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    uint64_t limit = capabilities.getMaxDynamicSharedMemoryPerBlock();
    bool oversized = bytes > limit;
    bool fixed = true;
    for (Attribute attribute : source.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() != gpu::PhysicalExprKind::Constant ||
          extent.getValue() <= 0) {
        fixed = false;
        break;
      }
      if (!oversized) {
        oversized = bytes > limit / extent.getValue();
        if (!oversized)
          bytes *= extent.getValue();
      }
    }
    if (!fixed || !oversized || (result && result.getShape().empty()))
      continue;
    SmallVector<std::pair<unsigned, Value>> selected;
    SmallVector<std::pair<unsigned, Value>> indexed;
    llvm::SmallDenseSet<unsigned> axes;
    llvm::SmallDenseSet<unsigned> seenAxes;
    bool compatible = true;
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      Value originalCoordinate = coordinate;
      coordinate = stripShapeOnly(coordinate);
      if (axis < 0 || static_cast<unsigned>(axis) >= source.getShape().size() ||
          !seenAxes.insert(axis).second) {
        compatible = false;
        break;
      }
      if (auto fragment = dyn_cast<gpu::FragmentType>(coordinate.getType())) {
        if (!isa<IntegerType, IndexType>(fragment.getElementType())) {
          compatible = false;
          break;
        }
        indexed.emplace_back(axis, originalCoordinate);
        continue;
      }
      if (!isa<IntegerType, IndexType>(coordinate.getType())) {
        compatible = false;
        break;
      }
      axes.insert(axis);
      selected.emplace_back(axis, coordinate);
    }
    if (!compatible || selected.empty())
      continue;
    unsigned resultAxis = 0;
    for (unsigned axis = 0; indexed.empty() && compatible && axis < source.getShape().size(); ++axis)
      if (!axes.contains(axis)) {
        compatible = result && resultAxis < result.getShape().size() &&
                     source.getShape()[axis] == result.getShape()[resultAxis];
        ++resultAxis;
      }
    if (!compatible || (indexed.empty() &&
                        (result ? resultAxis != result.getShape().size() : resultAxis != 0)))
      continue;

    // Exactly one source element contributes to each result lane. Combining
    // its integer representation with zero preserves NaNs and signed zero,
    // while native reductions avoid the whole-source scratch of a gather.
    OpBuilder builder(gather);
    Location location = gather.getLoc();
    auto bits = builder.getIntegerType(source.getElementType().getIntOrFloatBitWidth());
    auto bitsType = gpu::FragmentType::get(
        kernel.getContext(), bits, source.getShape(), source.getAxisMaps(),
        source.getValidity(), source.getOwner());
    Value value = builder.create<gpu::BitcastOp>(location, bitsType, gather.getSource());
    llvm::sort(selected, [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });
    for (auto [axis, coordinate] : selected) {
      // Select one axis at a time so later selections consume the reduced
      // fragment rather than constructing another full-source predicate.
      auto input = cast<gpu::FragmentType>(value.getType());
      auto mapping = cast<gpu::AxisMapAttr>(input.getAxisMaps()[axis]);
      auto ordinalAxis = gpu::AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), 0, mapping.getDerived());
      auto rangeType = gpu::FragmentType::get(
          kernel.getContext(), builder.getIndexType(),
          builder.getArrayAttr({input.getShape()[axis]}),
          builder.getArrayAttr({ordinalAxis}), input.getValidity(), input.getOwner());
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value size = builder.create<gpu::PhysicalExprOp>(
          location, builder.getIndexType(), cast<gpu::PhysicalExprAttr>(input.getShape()[axis]));
      Value ordinal = builder.create<gpu::MakeRangeOp>(
          location, rangeType, zero, size, one, zero, size,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
      if (!coordinate.getType().isIndex())
        coordinate = builder.create<gpu::CastOp>(location, builder.getIndexType(), coordinate);
      Value index = builder.create<gpu::BroadcastOp>(location, rangeType, coordinate);
      auto predicate = gpu::FragmentType::get(
          kernel.getContext(), builder.getI1Type(), rangeType.getShape(),
          rangeType.getAxisMaps(), rangeType.getValidity(), rangeType.getOwner());
      Value equal = builder.create<gpu::CompareOp>(
          location, predicate, ordinal, index, ComparePredicate::Eq);
      auto expanded = gpu::projectPredicateToFragmentAxis(
          builder, location, equal, input, axis);
      if (failed(expanded))
        return gather.emitOpError("scalar selection lost its source-axis projection");
      auto emptyBits = zeroLike(builder, location, input);
      if (failed(emptyBits))
        return failure();
      value = builder.create<gpu::SelectOp>(location, input, *expanded, value, *emptyBits);
      auto inferred = gpu::inferCollectiveResultType(input,
          ArrayRef<int64_t>{static_cast<int64_t>(axis)}, bits);
      if (failed(inferred))
        return gather.emitOpError("scalar selection has an invalid reduction schema");
      Type reducedType = *inferred;
      auto identity = zeroLike(builder, location, reducedType);
      if (failed(identity))
        return failure();
      auto reduction = builder.create<gpu::ReduceOp>(location, TypeRange{reducedType},
          ValueRange{value}, ValueRange{*identity}, ValueRange{},
          ArrayRef<int64_t>{static_cast<int64_t>(axis)});
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *body = builder.createBlock(&reduction.getCombine(), {},
            {reducedType, reducedType}, {location, location});
        Value combined = builder.create<gpu::BinaryOp>(
            location, reducedType, body->getArgument(0), body->getArgument(1),
            BinaryOperator::BitwiseOr);
        builder.create<gpu::YieldOp>(location, combined);
      }
      value = reduction.getResult(0);
    }
    Type decodedType = source.getElementType();
    if (auto fragment = dyn_cast<gpu::FragmentType>(value.getType()))
      decodedType = gpu::FragmentType::get(
          kernel.getContext(), source.getElementType(), fragment.getShape(),
          fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
    value = builder.create<gpu::BitcastOp>(location, decodedType, value);
    if (!indexed.empty()) {
      SmallVector<Value> coordinates;
      SmallVector<int64_t> sourceAxes;
      for (auto [axis, coordinate] : indexed) {
        unsigned removed = llvm::count_if(selected, [&](const auto &item) {
          return item.first < axis;
        });
        coordinates.push_back(coordinate);
        sourceAxes.push_back(axis - removed);
      }
      value = builder.create<gpu::GatherOp>(location, gather.getResult().getType(),
          value, coordinates, gather.getValid(), gather.getFill(), sourceAxes);
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        value.getDefiningOp()->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(value);
      gather.erase();
      continue;
    }
    if (value.getType() != gather.getResult().getType()) {
      auto reassociation = gpu::inferReshapeReassociation(
          cast<gpu::FragmentType>(value.getType()), result);
      if (failed(reassociation))
        return gather.emitOpError("scalar selection lost its result relation");
      value = builder.create<gpu::ReshapeOp>(location, result, value, *reassociation);
    }
    if (gather.getValid())
      value = builder.create<gpu::SelectOp>(
          location, gather.getResult().getType(), gather.getValid(), value, gather.getFill());
    gather.getResult().replaceAllUsesWith(value);
    gather.erase();
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult materializeOversizedGathers(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return success();
  auto configurations = gpu::ParameterSpace::read(kernel);
  if (failed(configurations)) return failure();
  auto tuples = configurations->configurations(gpu::ConfigurationStage::Shared);
  if (failed(tuples)) return failure();
  int64_t maximumPrograms = 0;
  for (DictionaryAttr tuple : *tuples) {
    NamedAttrList bindings;
    Builder attributes(kernel.getContext());
    for (gpu::ParameterAttr schema : configurations->extentDeclarations()) {
      if (schema.getCandidates().size() == 1)
        bindings.set(schema.getName(), attributes.getI64IntegerAttr(schema.getCandidates()[0]));
    }
    for (NamedAttribute entry : tuple)
      bindings.set(entry.getName(), entry.getValue());
    auto count = evaluateCompileTimeExpression(
        cast<gpu::PhysicalExprAttr>(space[0]), bindings.getDictionary(kernel.getContext()));
    if (!count || *count <= 0)
      return success();
    maximumPrograms = std::max(maximumPrograms, *count);
  }
  auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  llvm::MapVector<Value, SmallVector<gpu::GatherOp>> readers;
  kernel.walk([&](gpu::GatherOp gather) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
    if (!source || gather.getCoordinates().empty() ||
        belongsToSplitGatherPair(gather) ||
        !isa<FloatType, IntegerType>(source.getElementType()))
      return;
    int64_t bytes = (source.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    for (Attribute attribute : source.getShape()) {
      auto extent = cast<gpu::PhysicalExprAttr>(attribute);
      if (extent.getKind() != gpu::PhysicalExprKind::Constant ||
          extent.getValue() <= 0 ||
          bytes > std::numeric_limits<int64_t>::max() / extent.getValue())
        return;
      bytes *= extent.getValue();
    }
    if (bytes <= capabilities.getMaxDynamicSharedMemoryPerBlock())
      return;
    SmallVector<Value> selected(source.getShape().size());
    for (auto [coordinate, axis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
      selected[axis] = coordinate;
    for (unsigned axis = 0; axis < source.getShape().size(); ++axis) {
      Type coordinateType;
      if (selected[axis]) {
        coordinateType = selected[axis].getType();
      } else {
        if (!result)
          return;
        auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[axis]);
        auto projection = gpu::queryFragmentAxis(result, gpu::sourceAxisIdentity(mapping));
        if (!projection.isExact() ||
            projection.dimensionId != mapping.getDimensionId() ||
            result.getShape()[projection.fragmentAxis] != source.getShape()[axis])
          return;
        auto ordinal = gpu::AxisMapAttr::get(kernel.getContext(),
            mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
            0, mapping.getDerived());
        coordinateType = gpu::FragmentType::get(kernel.getContext(),
            IndexType::get(kernel.getContext()), ArrayAttr::get(kernel.getContext(),
                {source.getShape()[axis]}), ArrayAttr::get(kernel.getContext(), {ordinal}),
            source.getValidity(), source.getOwner());
      }
      if (auto coordinate = dyn_cast<gpu::FragmentType>(coordinateType)) {
        if (!result)
          return;
        auto projected = gpu::FragmentType::get(kernel.getContext(),
            coordinate.getElementType(), result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        if (!gpu::queryBroadcastProjection(coordinate, projected).isExact())
          return;
      }
    }
    readers[gather.getSource()].push_back(gather);
  });
  if (readers.empty())
    return success();
  OpBuilder entry(&kernel.front(), kernel.front().begin());
  gpu::ProgramIdOp programId;
  kernel.walk([&](gpu::ProgramIdOp operation) {
    if (operation.getAxis() == 0)
      programId = operation;
  });
  if (programId) {
    if (programId->getBlock() != &kernel.front() ||
        programId.getOperation() != &kernel.front().front())
      programId->moveBefore(&kernel.front(), kernel.front().begin());
  } else {
    programId = entry.create<gpu::ProgramIdOp>(kernel.getLoc(), entry.getIndexType(), 0);
  }
  Value program = programId.getResult();
  entry.setInsertionPointAfter(programId);
  auto prefix = gpu::PhysicalExprAttr::get(kernel.getContext(),
      gpu::PhysicalExprKind::Constant, maximumPrograms,
      entry.getStringAttr(""), entry.getArrayAttr({}));
  for (auto &readerGroup : readers) {
    auto &gathers = readerGroup.second;
    Value source = gathers.front().getSource();
    auto payload = cast<gpu::FragmentType>(source.getType());
    SmallVector<Attribute> shape{prefix};
    llvm::append_range(shape, payload.getShape());
    Value workspace = gpu::createInvocationWorkspace(
        kernel, source.getLoc(), payload.getElementType(),
        entry.getArrayAttr(shape), payload.getOwner());
    // The maximum is evaluated over every shared tuple; the private prefix
    // remains valid while Triton chooses its local configuration.
    entry.create<gpu::AssumeInBoundsOp>(source.getLoc(), program, workspace, 0);
    OpBuilder builder(kernel.getContext());
    if (Operation *definition = source.getDefiningOp())
      builder.setInsertionPointAfter(definition);
    else
      builder.setInsertionPointToStart(cast<BlockArgument>(source).getOwner());
    SmallVector<Value> coordinates{program};
    SmallVector<int64_t> sourceAxes{0};
    SmallVector<Value> ordinals;
    for (auto [axis, attribute] : llvm::enumerate(payload.getShape())) {
      auto mapping = cast<gpu::AxisMapAttr>(payload.getAxisMaps()[axis]);
      auto ordinalMap = gpu::AxisMapAttr::get(kernel.getContext(),
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
          0, mapping.getDerived());
      auto type = gpu::FragmentType::get(kernel.getContext(), builder.getIndexType(),
          builder.getArrayAttr({attribute}), builder.getArrayAttr({ordinalMap}),
          payload.getValidity(), payload.getOwner());
      Value zero = builder.create<arith::ConstantIndexOp>(source.getLoc(), 0);
      Value one = builder.create<arith::ConstantIndexOp>(source.getLoc(), 1);
      Value size = builder.create<gpu::PhysicalExprOp>(source.getLoc(),
          builder.getIndexType(), cast<gpu::PhysicalExprAttr>(attribute));
      Value ordinal = builder.create<gpu::MakeRangeOp>(source.getLoc(), type,
          zero, size, one, zero, size, mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDerived());
      ordinals.push_back(ordinal);
      coordinates.push_back(ordinal);
      sourceAxes.push_back(axis + 1);
    }
    builder.create<gpu::StoreOp>(source.getLoc(), workspace, coordinates, source,
                                 Value(), sourceAxes);
    for (gpu::GatherOp gather : gathers) {
      builder.setInsertionPoint(gather);
      auto result = dyn_cast<gpu::FragmentType>(gather.getResult().getType());
      SmallVector<Value> selected(payload.getShape().size());
      for (auto [coordinate, axis] :
           llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
        selected[axis] = coordinate;
      SmallVector<Value> access{program};
      for (unsigned axis = 0; axis < payload.getShape().size(); ++axis) {
        Value coordinate = selected[axis] ? selected[axis] : ordinals[axis];
        if (result) {
          auto indices = gpu::FragmentType::get(kernel.getContext(),
              gpu::uniformElementType(coordinate.getType()), result.getShape(), result.getAxisMaps(),
              result.getValidity(), result.getOwner());
          if (coordinate.getType() != indices)
            coordinate = builder.create<gpu::BroadcastOp>(gather.getLoc(), indices, coordinate);
        }
        access.push_back(coordinate);
      }
      auto load = builder.create<gpu::LoadOp>(gather.getLoc(), gather.getResult().getType(), workspace,
          access, gather.getValid(), gather.getFill(), sourceAxes);
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        load->setAttr(gpu::originAttr, origin);
      gather.getResult().replaceAllUsesWith(load.getResult());
      gather.erase();
    }
  }
  gpu::eraseDeadPhysicalValues(kernel);
  return success();
}


} // namespace intent::triton::detail

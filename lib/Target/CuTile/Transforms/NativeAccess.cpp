#include "NativeAccess.h"
#include "Configurations.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;
namespace intent::cutile {
FailureOr<Value> tileIndex(OpBuilder &builder, Location location, Value start,
                           Value extent) {
  std::optional<int64_t> startConstant = gpu::IndexRelations().constant(start);
  std::optional<int64_t> extentConstant = gpu::IndexRelations().constant(extent);
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
    if (binary.getOperatorKind() == BinaryOperator::Subtract) {
      if (isProvably(binary.getRhs(), 0))
        return tileIndex(builder, location, binary.getLhs(), extent);
      auto induction = dyn_cast<BlockArgument>(binary.getLhs());
      auto loop = induction
                      ? dyn_cast_or_null<scf::ForOp>(
                            induction.getOwner()->getParentOp())
                      : scf::ForOp();
      // An iteration's offset from its lower bound is tile aligned even when
      // the lower bound itself is not (for example a dynamic retained slice).
      if (loop && induction == loop.getInductionVar() &&
          gpu::samePhysicalScalarExpression(loop.getLowerBound(),
                                            binary.getRhs()) &&
          gpu::samePhysicalScalarExpression(loop.getStep(), extent))
        return Value(builder.create<gpu::BinaryOp>(
            location, builder.getIndexType(), start, extent,
            BinaryOperator::FloorDivide));
    }
  }
  auto argument = dyn_cast<BlockArgument>(start);
  auto loop =
      argument
          ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
          : scf::ForOp();
  if (loop && argument == loop.getInductionVar() &&
      gpu::samePhysicalScalarExpression(loop.getStep(), extent) &&
      gpu::IndexRelations().multipleOf(loop.getLowerBound(), extent))
    return Value(builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), start, extent,
        BinaryOperator::FloorDivide));
  return failure();
}

FailureOr<SmallVector<Value>> orderedCoordinates(gpu::AccessOpInterface access) {
  Operation *owner = access.getOperation();
  auto view = cast<gpu::ViewType>(access.getAccessResource().getType());
  auto coordinates = access.getAccessCoordinates();
  auto sourceAxes = access.getAccessSourceAxes();
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


FailureOr<SmallVector<unsigned>>
coordinateTargetAxes(gpu::FragmentType source, gpu::FragmentType target) {
  if (source.getOwner() != target.getOwner() ||
      source.getShape().size() > target.getShape().size())
    return failure();
  SmallVector<unsigned> result(source.getShape().size());
  bool positional = source.getShape().size() == target.getShape().size();
  for (unsigned axis = 0; positional && axis < source.getShape().size(); ++axis) {
    if (isUnitExtent(source.getShape()[axis]))
      continue;
    auto left = cast<gpu::AxisMapAttr>(source.getAxisMaps()[axis]);
    auto right = cast<gpu::AxisMapAttr>(target.getAxisMaps()[axis]);
    positional = gpu::sourceAxisIdentity(left) == gpu::sourceAxisIdentity(right) ||
                 (left.getDimensionId() > 0 &&
                  left.getDimensionId() == right.getDimensionId());
  }
  if (positional) {
    auto projection = gpu::queryBroadcastProjection(source, target);
    if (projection.isExact()) {
      for (auto [axis, origin] : llvm::enumerate(projection.targetToSource))
        if (origin)
          result[*origin] = axis;
      return result;
    }
  }
  SmallVector<unsigned> unitAxes;
  SmallVector<bool> usedTargetAxes(target.getShape().size(), false);
  for (auto [sourceIndex, sourceAttribute] :
       llvm::enumerate(source.getAxisMaps())) {
    auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[sourceIndex]);
    if (extent.getKind() ==
            gpu::PhysicalExprKind::Constant &&
        extent.getValue() == 1) {
      unitAxes.push_back(sourceIndex);
      continue;
    }
    auto sourceMap = cast<gpu::AxisMapAttr>(sourceAttribute);
    std::optional<unsigned> targetIndex;
    for (auto [index, targetAttribute] :
         llvm::enumerate(target.getAxisMaps())) {
      auto targetMap = cast<gpu::AxisMapAttr>(targetAttribute);
      if (usedTargetAxes[index] ||
          source.getShape()[sourceIndex] != target.getShape()[index] ||
          sourceMap.getSourceId() != targetMap.getSourceId() ||
          sourceMap.getSourceAxis() != targetMap.getSourceAxis() ||
          sourceMap.getDimensionId() != targetMap.getDimensionId() ||
          sourceMap.getDerived() != targetMap.getDerived())
        continue;
      if (targetIndex)
        return failure();
      targetIndex = index;
    }
    if (!targetIndex && sourceMap.getDimensionId() > 0) {
      unsigned occurrences = 0;
      for (auto [axis, mapping] : llvm::enumerate(source.getAxisMaps()))
        occurrences += cast<gpu::AxisMapAttr>(mapping).getDimensionId() ==
                           sourceMap.getDimensionId() &&
                       source.getShape()[axis] == source.getShape()[sourceIndex];
      if (occurrences != 1)
        return failure();
      for (auto [axis, mapping] : llvm::enumerate(target.getAxisMaps())) {
        if (usedTargetAxes[axis] ||
            cast<gpu::AxisMapAttr>(mapping).getDimensionId() !=
                sourceMap.getDimensionId() ||
            target.getShape()[axis] != source.getShape()[sourceIndex])
          continue;
        if (targetIndex)
          return failure();
        targetIndex = axis;
      }
    }
    if (!targetIndex)
      return failure();
    Attribute sourceExtent = source.getShape()[sourceIndex];
    Attribute targetExtent = target.getShape()[*targetIndex];
    if (sourceExtent != targetExtent)
      return failure();
    usedTargetAxes[*targetIndex] = true;
    result[sourceIndex] = *targetIndex;
  }
  // Singleton axes carry no coordinate variation.  Match the varying axes
  // first, then embed singleton axes into the remaining broadcast dimensions.
  // A newaxis introduced by author indexing need not share the access axis's
  // provenance identity.
  unsigned nextTargetAxis = 0;
  for (unsigned sourceAxis : unitAxes) {
    while (usedTargetAxes[nextTargetAxis])
      ++nextTargetAxis;
    result[sourceAxis] = nextTargetAxis;
    usedTargetAxes[nextTargetAxis] = true;
  }
  return result;
}

FailureOr<SmallVector<Value>> materializeCoordinateDomains(
    OpBuilder &builder, Operation *owner, ValueRange coordinates,
    gpu::FragmentType target) {
  SmallVector<Value> results;
  results.reserve(coordinates.size());
  bool cartesian = static_cast<size_t>(llvm::count_if(coordinates, [](Value value) {
    return isa<gpu::FragmentType>(value.getType());
  })) == target.getShape().size() &&
      llvm::all_of(coordinates, [](Value value) {
        auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
        return !fragment || fragment.getShape().size() == 1;
      });
  unsigned fragmentSlot = 0;
  for (Value coordinate : coordinates) {
    auto source = dyn_cast<gpu::FragmentType>(coordinate.getType());
    unsigned position = source ? fragmentSlot++ : 0;
    if (!source || samePhysicalDomain(source, target)) {
      results.push_back(coordinate);
      continue;
    }
    FailureOr<SmallVector<unsigned>> targetAxes =
        coordinateTargetAxes(source, target);
    if (failed(targetAxes) && cartesian && source.getOwner() == target.getOwner() &&
        source.getShape()[0] == target.getShape()[position]) {
      auto sourceAxis = cast<gpu::AxisMapAttr>(source.getAxisMaps()[0]);
      auto targetAxis = cast<gpu::AxisMapAttr>(target.getAxisMaps()[position]);
      // Cartesian coordinates are ordered by resource axis. Reusing the same
      // range on two axes retains those distinct positional occurrences.
      if (gpu::sourceAxisIdentity(sourceAxis) == gpu::sourceAxisIdentity(targetAxis) &&
          sourceAxis.getDimensionId() == targetAxis.getDimensionId())
        targetAxes = SmallVector<unsigned>{position};
    }
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
        gpu::PhysicalExprKind::Constant, 1,
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

FailureOr<Value> scalarFill(Operation *owner, Value fill) {
  Value scalar = uniformScalarFill(fill);
  if (!fill || scalar)
    return scalar;
  return owner->emitOpError(
      "cuTile gather padding must be an explicit scalar or splat");
}

FailureOr<std::pair<Value, bool>>
extractionTileIndex(OpBuilder &builder, Location location, Value coordinate,
                    gpu::MakeRangeOp range, Value sourceExtent) {
  if (coordinate == range.getResult()) {
    auto index = tileIndex(builder, location, range.getStart(), range.getExtent());
    if (failed(index))
      return failure();
    return std::pair{*index, true};
  }
  if (auto reshape = coordinate.getDefiningOp<gpu::ReshapeOp>()) {
    auto input = cast<gpu::FragmentType>(reshape.getValue().getType());
    auto output = cast<gpu::FragmentType>(reshape.getResult().getType());
    unsigned sourceRank = 0, resultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<gpu::ReshapeGroupAttr>(attribute);
      sourceRank += group.getSourceAxes().size();
      resultRank += group.getResultAxes().size();
    }
    unsigned sourcePrefix = input.getShape().size() - sourceRank;
    unsigned resultPrefix = output.getShape().size() - resultRank;
    if (sourcePrefix != resultPrefix)
      return failure();
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<gpu::ReshapeGroupAttr>(attribute);
      SmallVector<Attribute> before, after;
      for (int64_t axis : group.getSourceAxes().asArrayRef())
        if (Attribute extent = input.getShape()[sourcePrefix + axis];
            !isUnitExtent(extent))
          before.push_back(extent);
      for (int64_t axis : group.getResultAxes().asArrayRef())
        if (Attribute extent = output.getShape()[resultPrefix + axis];
            !isUnitExtent(extent))
          after.push_back(extent);
      if (before.size() > 1 || before != after)
        return failure();
    }
    return extractionTileIndex(builder, location, reshape.getValue(), range,
                               sourceExtent);
  }
  if (isa_and_nonnull<gpu::BroadcastOp, gpu::TransposeOp>(
          coordinate.getDefiningOp()))
    return extractionTileIndex(builder, location,
                               coordinate.getDefiningOp()->getOperand(0),
                               range, sourceExtent);
  if (Value scalar = uniformScalarFill(coordinate)) {
    if (auto index = tileIndex(builder, location, scalar, range.getExtent());
        succeeded(index))
      return std::pair{*index, false};
    // Native tile extents are powers of two. A source-aligned shift either
    // selects whole sub-tiles, or is periodic in a repeated wider extraction.
    auto index = tileIndex(builder, location, scalar, sourceExtent);
    if (failed(index))
      return failure();
    Value ratio = builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), sourceExtent, range.getExtent(),
        BinaryOperator::FloorDivide);
    return std::pair{Value(builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), *index, ratio,
        BinaryOperator::Multiply)), false};
  }
  auto binary = coordinate.getDefiningOp<gpu::BinaryOp>();
  if (!binary || (binary.getOperatorKind() != BinaryOperator::Add &&
                  binary.getOperatorKind() != BinaryOperator::Subtract))
    return failure();
  if (binary.getOperatorKind() == BinaryOperator::Subtract &&
      binary.getLhs() == range.getResult()) {
    if (Value offset = uniformScalarFill(binary.getRhs())) {
      Value start = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), range.getStart(), offset,
          BinaryOperator::Subtract);
      if (auto index = tileIndex(builder, location, start, range.getExtent());
          succeeded(index))
        return std::pair{*index, true};
    }
  }
  auto lhs = extractionTileIndex(builder, location, binary.getLhs(), range,
                                 sourceExtent);
  auto rhs = extractionTileIndex(builder, location, binary.getRhs(), range,
                                 sourceExtent);
  if (failed(lhs) || failed(rhs))
    return failure();
  if ((lhs->second && rhs->second) ||
      (rhs->second && binary.getOperatorKind() == BinaryOperator::Subtract))
    return failure();
  return std::pair{Value(builder.create<gpu::BinaryOp>(
      location, builder.getIndexType(), lhs->first, rhs->first,
      binary.getOperatorKind())), lhs->second || rhs->second};
}

Value repeatTileForExtraction(OpBuilder &builder, Location location, Value value,
                              unsigned axis, gpu::PhysicalExprAttr requested) {
  auto source = cast<gpu::FragmentType>(value.getType());
  auto extent = cast<gpu::PhysicalExprAttr>(source.getShape()[axis]);
  if (extent == requested)
    return value;
  auto context = builder.getContext();
  auto expression = [&](gpu::PhysicalExprKind kind,
                         ArrayRef<Attribute> operands) {
    return gpu::PhysicalExprAttr::get(context, kind, 0,
                                     builder.getStringAttr(""),
                                     builder.getArrayAttr(operands));
  };
  auto unit = gpu::PhysicalExprAttr::get(
      context, gpu::PhysicalExprKind::Constant, 1,
      builder.getStringAttr(""), builder.getArrayAttr({}));
  auto repeat = expression(gpu::PhysicalExprKind::FloorDiv,
      {expression(gpu::PhysicalExprKind::Maximum, {extent, requested}), extent});
  auto expanded = expression(gpu::PhysicalExprKind::Multiply, {repeat, extent});
  SmallVector<Attribute> splitShape, splitMaps, splitGroups, mergeGroups;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  uint64_t repeatSource = gpu::nextPhysicalAxisIdentities(kernel).first;
  for (unsigned original = 0; original < source.getShape().size(); ++original) {
    SmallVector<int64_t> splitAxes;
    if (original == axis) {
      splitAxes.push_back(splitShape.size());
      splitMaps.push_back(gpu::AxisMapAttr::get(
          context, repeatSource, 0, 0, splitShape.size(), true));
      splitShape.push_back(unit);
    }
    splitAxes.push_back(splitShape.size());
    auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[original]);
    splitMaps.push_back(gpu::AxisMapAttr::get(
        context, mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), splitShape.size(), mapping.getDerived()));
    splitShape.push_back(source.getShape()[original]);
    splitGroups.push_back(gpu::ReshapeGroupAttr::get(
        context, builder.getDenseI64ArrayAttr({original}),
        builder.getDenseI64ArrayAttr(splitAxes)));
    mergeGroups.push_back(gpu::ReshapeGroupAttr::get(
        context, builder.getDenseI64ArrayAttr(splitAxes),
        builder.getDenseI64ArrayAttr({original})));
  }
  auto fragment = [&](ArrayRef<Attribute> shape, ArrayRef<Attribute> mappings) {
    return gpu::FragmentType::get(context, source.getElementType(),
        builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
        source.getValidity(), source.getOwner());
  };
  Value split = builder.create<gpu::ReshapeOp>(
      location, fragment(splitShape, splitMaps), value,
      builder.getArrayAttr(splitGroups));
  splitShape[axis] = repeat;
  Value repeated = builder.create<gpu::BroadcastOp>(
      location, fragment(splitShape, splitMaps), split);
  SmallVector<Attribute> resultShape(source.getShape().getValue());
  resultShape[axis] = expanded;
  return builder.create<gpu::ReshapeOp>(
      location, fragment(resultShape, source.getAxisMaps().getValue()), repeated,
      builder.getArrayAttr(mergeGroups));
}

Value materializeTileOrigin(OpBuilder &builder, Location location,
                            const NativeTileAxisPlan &axis) {
  if (axis.scalarIndex) {
    if (!axis.divisor)
      return axis.scalarIndex;
    Value divisor = builder.create<arith::ConstantIndexOp>(location, axis.divisor);
    return builder.create<gpu::BinaryOp>(location, builder.getIndexType(),
        axis.scalarIndex, divisor, BinaryOperator::FloorDivide);
  }
  Value origin;
  auto add = [&](Value value, int64_t scale) {
    if (scale != 1) {
      Value coefficient = builder.create<arith::ConstantIndexOp>(location, scale);
      value = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), value, coefficient,
          BinaryOperator::Multiply);
    }
    origin = origin ? Value(builder.create<gpu::BinaryOp>(
                          location, builder.getIndexType(), origin, value,
                          BinaryOperator::Add))
                    : value;
  };
  for (auto [range, scale] : axis.ranges)
    add(range.getStart(), scale);
  for (auto [offset, scale] : axis.offsets)
    add(offset, scale);
  return origin;
}

Value materializeTileModulus(OpBuilder &builder, Location location,
                             const NativeTileAxisPlan &axis) {
  if (auto value = dyn_cast<Value>(axis.modulus))
    return value;
  return builder.create<arith::ConstantIndexOp>(
      location, cast<IntegerAttr>(cast<Attribute>(axis.modulus)).getInt());
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
    Value origin = materializeTileOrigin(builder, owner->getLoc(), axisPlan);
    if (!origin)
      return failure();
    if (axisPlan.modulus)
      origin = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getIndexType(), origin,
          materializeTileModulus(builder, owner->getLoc(), axisPlan),
          BinaryOperator::Remainder);
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
    if (gpu::PhysicalExprAttr upper =
            gpu::queryNonNegativeIndexUpperBound(origin)) {
      Value bound = builder.create<gpu::PhysicalExprOp>(
          owner->getLoc(), builder.getIndexType(), upper);
      Value allOriginsInBounds = builder.create<gpu::CompareOp>(
          owner->getLoc(), builder.getI1Type(), bound, extent,
          ComparePredicate::Lt);
      // Specialization can discharge the whole mapped domain. Otherwise the
      // original per-origin predicate still determines exactly the same access.
      axisCondition = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getI1Type(), allOriginsInBounds, axisCondition,
          BinaryOperator::LogicalOr);
    }
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

Value materializeFullRangeGuard(
    OpBuilder &builder, Location location,
    const gpu::PhysicalAccessBoundaryFact &boundary) {
  Value guard;
  for (auto [range, upper] : boundary.rangeBounds) {
    if (auto loop = completeAlignedTileLoop(range.getStart(), range.getExtent()))
      if (gpu::samePhysicalScalarExpression(loop.getUpperBound(), upper))
        continue;
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value origin = range.getStart();
    Value nonnegativeOrigin = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), origin, zero, ComparePredicate::Ge);
    Value nonnegativeUpper = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), upper, zero, ComparePredicate::Ge);
    Value nonnegative = builder.create<gpu::BinaryOp>(
        location, builder.getI1Type(), nonnegativeOrigin, nonnegativeUpper,
        BinaryOperator::LogicalAnd);
    // With nonnegative endpoints, subtraction cannot overflow. A failed guard
    // keeps the original predicated gather, including a partial logical tile.
    Value remaining = builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), upper, origin,
        BinaryOperator::Subtract);
    Value fullRange = builder.create<gpu::CompareOp>(
        location, builder.getI1Type(), remaining, range.getExtent(),
        ComparePredicate::Ge);
    Value condition = builder.create<gpu::BinaryOp>(
        location, builder.getI1Type(), nonnegative, fullRange,
        BinaryOperator::LogicalAnd);
    guard = guard ? Value(builder.create<gpu::BinaryOp>(
                        location, builder.getI1Type(), guard, condition,
                        BinaryOperator::LogicalAnd))
                  : condition;
  }
  return guard;
}

FailureOr<MaterializedTileIndices>
materializeTileIndices(OpBuilder &builder, Operation *owner,
                       const NativeTileAccessPlan &plan,
                       bool allowDynamicAlignment) {
  MaterializedTileIndices result;
  result.values.reserve(plan.axes.size());
  auto require = [&](Value condition) {
    result.alignment = result.alignment
                           ? Value(builder.create<gpu::BinaryOp>(
                                 owner->getLoc(), builder.getI1Type(),
                                 result.alignment, condition,
                                 BinaryOperator::LogicalAnd))
                           : condition;
  };
  for (auto [resourceAxis, axis] : llvm::enumerate(plan.axes)) {
    if (axis.scalarIndex) {
      Value index = materializeTileOrigin(builder, owner->getLoc(), axis);
      if (axis.divisor) {
        auto range = axis.ranges.front().first;
        Value extent = range.getExtent();
        if (!gpu::IndexRelations().positive(extent))
          return failure();
        Value divisor = builder.create<arith::ConstantIndexOp>(owner->getLoc(), axis.divisor);
        if (isAlignedPeriodicTile(axis.scalarIndex, extent, divisor)) {
          result.values.push_back(index);
          continue;
        }
        Value one = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 1);
        Value offset = builder.create<gpu::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), extent, one, BinaryOperator::Subtract);
        Value last = builder.create<gpu::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), axis.scalarIndex, offset, BinaryOperator::Add);
        Value lastIndex = builder.create<gpu::BinaryOp>(owner->getLoc(),
            builder.getIndexType(), last, divisor, BinaryOperator::FloorDivide);
        // Positive extent and divisor make equal endpoint quotients sufficient;
        // signed wraparound would change the quotient's sign and fail the guard.
        require(builder.create<gpu::CompareOp>(owner->getLoc(), builder.getI1Type(),
                                               index, lastIndex, ComparePredicate::Eq));
      }
      result.values.push_back(index);
      continue;
    }
    Value start = materializeTileOrigin(builder, owner->getLoc(), axis);
    Value extent;
    if (axis.ranges.size() == 1 && axis.ranges.front().second == 1) {
      auto range = axis.ranges.front().first;
      extent = range.getExtent();
    } else {
      extent = builder.create<gpu::PhysicalExprOp>(
          owner->getLoc(), builder.getIndexType(),
          cast<gpu::PhysicalExprAttr>(plan.resourceType.getShape()[resourceAxis]));
      Value inner = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 1);
      Value one = inner;
      // A varying axis is contiguous only when its stride equals the product
      // of the inner tile extents. Unit axes contribute an origin, not a gap.
      for (auto [range, stride] : llvm::reverse(axis.ranges)) {
        Value coefficient = builder.create<arith::ConstantIndexOp>(
            owner->getLoc(), stride);
        Value unit = builder.create<gpu::CompareOp>(
            owner->getLoc(), builder.getI1Type(), range.getExtent(), one,
            ComparePredicate::Eq);
        Value contiguous = builder.create<gpu::CompareOp>(
            owner->getLoc(), builder.getI1Type(), coefficient, inner,
            ComparePredicate::Eq);
        require(builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), unit, contiguous,
            BinaryOperator::LogicalOr));
        inner = builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getIndexType(), inner, range.getExtent(),
            BinaryOperator::Multiply);
      }
    }
    if (axis.modulus) {
      if (!gpu::IndexRelations().positive(extent))
        return failure();
      Location location = owner->getLoc();
      Value modulus = materializeTileModulus(builder, location, axis);
      if (isAlignedPeriodicTile(start, extent, modulus) &&
          completeAlignedTileLoop(start, extent)) {
        Value remainder = builder.create<gpu::BinaryOp>(
            location, builder.getIndexType(), start, modulus,
            BinaryOperator::Remainder);
        result.values.push_back(builder.create<gpu::BinaryOp>(
            location, builder.getIndexType(), remainder, extent,
            BinaryOperator::FloorDivide));
        continue;
      }
      Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
      Value one = builder.create<arith::ConstantIndexOp>(location, 1);
      Value maximum = builder.create<arith::ConstantIndexOp>(
          location, std::numeric_limits<int64_t>::max());
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), modulus, zero, ComparePredicate::Gt));
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), start, zero, ComparePredicate::Ge));
      Value available = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), maximum, start,
          BinaryOperator::Subtract);
      Value lastOffset = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), extent, one,
          BinaryOperator::Subtract);
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), available, lastOffset,
          ComparePredicate::Ge));
      start = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), start, modulus,
          BinaryOperator::Remainder);
      Value remaining = builder.create<gpu::BinaryOp>(
          location, builder.getIndexType(), modulus, start,
          BinaryOperator::Subtract);
      // A tile that crosses the modulo boundary keeps its original gather.
      require(builder.create<gpu::CompareOp>(
          location, builder.getI1Type(), remaining, extent, ComparePredicate::Ge));
    }
    FailureOr<Value> index = tileIndex(builder, owner->getLoc(), start,
                                       extent);
    if (succeeded(index)) {
      result.values.push_back(*index);
      continue;
    }
    if (!allowDynamicAlignment || !gpu::IndexRelations().positive(extent))
      return failure();
    result.values.push_back(builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, extent,
        BinaryOperator::FloorDivide));
    Value remainder = builder.create<gpu::BinaryOp>(
        owner->getLoc(), builder.getIndexType(), start, extent,
        BinaryOperator::Remainder);
    Value zero = builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    Value aligned = builder.create<gpu::CompareOp>(
        owner->getLoc(), builder.getI1Type(), remainder, zero,
        ComparePredicate::Eq);
    if (gpu::IndexRelations().powerOfTwo(extent)) {
      auto factors = uniformAlignmentFactors(start, extent);
      if (succeeded(factors)) {
        if (factors->empty())
          continue;
        Value uniform;
        for (Value factor : *factors) {
          Value modulus = builder.create<gpu::BinaryOp>(
              owner->getLoc(), builder.getIndexType(), factor,
              extent, BinaryOperator::Remainder);
          Value condition = builder.create<gpu::CompareOp>(
              owner->getLoc(), builder.getI1Type(), modulus, zero,
              ComparePredicate::Eq);
          uniform = uniform ? Value(builder.create<gpu::BinaryOp>(
                                  owner->getLoc(), builder.getI1Type(), uniform,
                                  condition, BinaryOperator::LogicalAnd))
                            : condition;
        }
        // Specialization can discard the gather branch for the entire loop.
        // Otherwise retain the original per-origin alignment predicate.
        aligned = builder.create<gpu::BinaryOp>(
            owner->getLoc(), builder.getI1Type(), uniform, aligned,
            BinaryOperator::LogicalOr);
      }
    }
    require(aligned);
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

bool isZeroFill(Value value) {
  Value scalar = uniformScalarFill(value);
  if (auto cast = scalar ? scalar.getDefiningOp<gpu::CastOp>() : gpu::CastOp())
    if (cast.getValue().getType().isIntOrIndex() &&
        cast.getResult().getType().isIntOrIndex())
      return isZeroFill(cast.getValue());
  Attribute constant = getCompileTimeScalar(scalar ? scalar : value);
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant))
    return floating.getValue().isZero() && !floating.getValue().isNegative();
  return false;
}

LogicalResult formNativeAccesses(func::FuncOp kernel,
    const gpu::TuningProfiles &profiles, const NativeProgramInputs &inputs,
    bool matrixCompute, NativeFormRewriter &rewriter) {
  Value accessForm;
  Value loadPolicy;
  Value preferTileLoads;
  Value allowNativeTMA;
  auto accessFormValue = [&]() -> FailureOr<Value> {
    if (accessForm)
      return accessForm;
    auto parameter = declareProviderParameter(
        kernel, profiles, "access_form", accessFormParameter,
        gpu::ParameterRole::ProviderAccessForm, isLegalAccessForm);
    if (failed(parameter))
      return failure();
    OpBuilder entry(&kernel.front(), kernel.front().begin());
    accessForm = gpu::materializeParameter(entry, kernel.getLoc(), *parameter);
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
  auto tmaCondition = [&]() -> FailureOr<Value> {
    if (allowNativeTMA)
      return allowNativeTMA;
    FailureOr<Value> form = accessFormValue();
    if (failed(form))
      return failure();
    OpBuilder entry(kernel.getContext());
    entry.setInsertionPointAfter((*form).getDefiningOp());
    Value disabled = entry.create<arith::ConstantIndexOp>(
        kernel.getLoc(), nativeNoTMAForm);
    allowNativeTMA = entry.create<gpu::CompareOp>(
        kernel.getLoc(), entry.getI1Type(), *form, disabled,
        ComparePredicate::Ne);
    return allowNativeTMA;
  };

  for (gpu::LoadOp load : inputs.loads) {
    auto access = cast<gpu::AccessOpInterface>(load.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
    if (!view)
      return load.emitOpError(
          "cuTile native load requires an external view resource");
    OpBuilder builder(load);
    if (!isa<gpu::FragmentType>(access.getAccessResult().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(access);
      FailureOr<Value> fill = scalarFill(load, access.getAccessFill());
      if (failed(indices) || failed(fill) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (access.getAccessValidity() && isa<gpu::FragmentType>(access.getAccessValidity().getType())))
        return load.emitOpError(
            "cuTile scalar load requires scalar indices and validity");
      bool inBounds = scalarCoordinatesInView(*indices, view, kernel);
      Value validity = access.getAccessValidity();
      Value padding = *fill;
      gpu::PhysicalAccessBoundaryFact boundary =
          analysis.boundaryValidity(load);
      if (inBounds && boundary.isExact()) {
        validity = {};
        padding = {};
      }
      auto replacement = builder.create<ScalarLoadOp>(
          load.getLoc(), access.getAccessResult().getType(), access.getAccessResource(), *indices,
          validity, padding,
          inBounds ? builder.getUnitAttr() : UnitAttr());
      if (Attribute origin = load->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.replace(load, ValueRange{replacement.getResult()});
      continue;
    }
    auto result = cast<gpu::FragmentType>(access.getAccessResult().getType());
    const auto accessBounds = analysis.accessBounds(load);
    bool activeInBounds = accessBounds.isExact();
    // Vector inputs stay register loads, not cluster TMA payloads.
    bool vectorInput = matrixCompute && result.getShape().size() == 1;
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(load, /*allowRangeGuards=*/true);
    FailureOr<NativeTileAccessPlan> plan = analyzeNativeTileAccess(access, kernel, accessBounds);
    FailureOr<MaterializedTileIndices> indices = failure();
    FailureOr<Value> originGuard = failure();
    if (!vectorInput && succeeded(plan) && boundary.isExact() &&
        (!access.getAccessFill() || isZeroFill(access.getAccessFill())))
      indices = materializeTileIndices(builder, load, *plan,
                                       /*allowDynamicAlignment=*/true);
    if (succeeded(indices))
      originGuard = materializeTileOriginGuard(
          builder, load, access.getAccessResource(), *plan, boundary);
    Value rangeGuard = succeeded(indices)
                           ? materializeFullRangeGuard(builder, load.getLoc(),
                                                       boundary)
                           : Value();
    bool guardedNative = succeeded(originGuard) && *originGuard;
    bool native = succeeded(indices) && succeeded(originGuard) &&
                  (!guardedNative ||
                   (access.getAccessFill() && access.getAccessFill().getType() == result));
    if (rangeGuard && guardedNative)
      rangeGuard = builder.create<gpu::BinaryOp>(
          load.getLoc(), builder.getI1Type(), rangeGuard, *originGuard,
          BinaryOperator::LogicalAnd);
    FailureOr<Value> allowTMA = failure();
    if (native) {
      allowTMA = tmaCondition();
      if (failed(allowTMA))
        return failure();
    }
    Value replacementResult;
    SmallVector<Operation *> createdOperations;
    Value loopLatency;
    if (matrixCompute && load->getParentOfType<scf::ForOp>()) {
      if (!loadPolicy) {
        auto parameter = declareProviderParameter(
            kernel, profiles, "load_policy_loop", loadPolicyParameter,
            gpu::ParameterRole::ProviderLoadPolicy, isLegalLoadPolicy);
        if (failed(parameter))
          return failure();
        OpBuilder entry(&kernel.front(), kernel.front().begin());
        loadPolicy = gpu::materializeParameter(entry, kernel.getLoc(), *parameter);
      }
      loopLatency = loadPolicy;
    }
    auto emitNativeLoad = [&](OpBuilder &nested) {
      Value fullTiles = nested.create<arith::ConstantIntOp>(load.getLoc(), 1, 1);
      Value zero = nested.create<arith::ConstantIndexOp>(load.getLoc(), 0);
      for (auto [axis, attribute] : llvm::enumerate(plan->resourceType.getShape())) {
        auto extent = cast<gpu::PhysicalExprAttr>(attribute);
        if (extent.getKind() ==
                gpu::PhysicalExprKind::Constant &&
            extent.getValue() == 1)
          continue;
        Value size = nested.create<gpu::DimOp>(
            load.getLoc(), nested.getIndexType(), access.getAccessResource(), axis);
        Value width = nested.create<gpu::PhysicalExprOp>(
            load.getLoc(), nested.getIndexType(), extent);
        Value remainder = nested.create<gpu::BinaryOp>(
            load.getLoc(), nested.getIndexType(), size, width,
            BinaryOperator::Remainder);
        Value divisible = nested.create<gpu::CompareOp>(
            load.getLoc(), nested.getI1Type(), remainder, zero,
            ComparePredicate::Eq);
        fullTiles = nested.create<gpu::BinaryOp>(
            load.getLoc(), nested.getI1Type(), fullTiles, divisible,
            BinaryOperator::LogicalAnd);
      }
      auto tile = nested.create<TileLoadOp>(
          load.getLoc(), plan->resourceType, access.getAccessResource(), *allowTMA,
          indices->values, loopLatency, fullTiles);
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
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(access);
      if (failed(coordinates))
        return failure();
      Value fill = uniformScalarFill(access.getAccessFill());
      bool fragmentFill = access.getAccessFill() && !fill;
      if (fragmentFill) {
        if (!access.getAccessValidity())
          return load.emitOpError(
              "fragment padding requires an explicit access validity");
        FailureOr<Value> zero = gpu::materializeScalarConstant(
            nested, load.getLoc(), nested.getZeroAttr(view.getElementType()),
            view.getElementType());
        if (failed(zero))
          return failure();
        fill = *zero;
      }
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(nested, load, *coordinates, result);
      if (failed(materialized))
        return failure();
      auto replacement = nested.create<GatherLoadOp>(
          load.getLoc(), result, access.getAccessResource(), *materialized,
          access.getAccessValidity(), fill, loopLatency, identityAxes(view.getRank()),
          activeInBounds ? nested.getUnitAttr() : UnitAttr());
      createdOperations.push_back(replacement);
      if (fragmentFill) {
        auto selected = nested.create<gpu::SelectOp>(
            load.getLoc(), result, access.getAccessValidity(), replacement.getResult(),
            access.getAccessFill());
        createdOperations.push_back(selected);
        return selected.getResult();
      }
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
      outOfBounds.create<scf::YieldOp>(load.getLoc(), access.getAccessFill());
      return conditional.getResult(0);
    };
    auto emitGuardedNative = [&](OpBuilder &nested) -> FailureOr<Value> {
      Value condition = indices->alignment;
      if (rangeGuard)
        condition = condition ? Value(nested.create<gpu::BinaryOp>(
                                    load.getLoc(), nested.getI1Type(), condition,
                                    rangeGuard, BinaryOperator::LogicalAnd))
                              : rangeGuard;
      if (!condition)
        return emitNativeOrFill(nested);
      auto conditional = nested.create<scf::IfOp>(
          load.getLoc(), TypeRange{result}, condition,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder aligned = prepareBranch(conditional.getThenRegion());
      Value value = rangeGuard ? emitNativeLoad(aligned)
                              : emitNativeOrFill(aligned);
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
      FailureOr<Value> tiledValue = emitGuardedNative(tiled);
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
    rewriter.replace(load, ValueRange{replacementResult});
  }

  for (gpu::GatherOp gather : inputs.gathers) {
    auto access = cast<gpu::AccessOpInterface>(gather.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    auto source = dyn_cast<gpu::FragmentType>(access.getAccessResource().getType());
    auto result = dyn_cast<gpu::FragmentType>(access.getAccessResult().getType());
    if (!source)
      return gather.emitOpError("cuTile tile extraction requires a source fragment");
    if (access.getAccessCoordinates().size() != access.getAccessSourceAxes().size())
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    SmallVector<Value> coordinates(source.getShape().size());
    OpBuilder builder(gather);
    auto unit = gpu::PhysicalExprAttr::get(
        kernel.getContext(), gpu::PhysicalExprKind::Constant,
        1, builder.getStringAttr(""), builder.getArrayAttr({}));
    SmallVector<Attribute> extractionShape(source.getShape().size(), unit);
    SmallVector<bool> slicedAxes(source.getShape().size(), false);
    for (auto [coordinate, sourceAxis] :
         llvm::zip(access.getAccessCoordinates(), access.getAccessSourceAxes())) {
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(coordinates.size()) ||
          coordinates[sourceAxis])
        return gather.emitOpError(
            "cuTile tile extraction source axes are not a unique subset");
      while (true) {
        if (auto splat = coordinate.getDefiningOp<gpu::SplatOp>())
          coordinate = splat.getValue();
        else if (auto broadcast = coordinate.getDefiningOp<gpu::BroadcastOp>())
          coordinate = broadcast.getValue();
        else
          break;
      }
      if (result) {
        if (isa<gpu::FragmentType>(coordinate.getType()) &&
            !getCompileTimeScalar(coordinate)) {
          auto ranges = analysis.sourceRanges(coordinate);
          // Scalar offsets can make general range analysis unknown. Use its
          // root only as a candidate; extractionTileIndex proves unit slope,
          // alignment and axis-preserving reshapes for the whole expression.
          if (ranges.roots.size() != 1 ||
              !gpu::isUnitStepRange(ranges.roots.front())) {
            InFlightDiagnostic diagnostic = gather.emitOpError(
                "cuTile tile extraction requires a contiguous unit-step coordinate");
            diagnostic << "; coordinate=" << coordinate
                       << "; range_state=" << static_cast<unsigned>(ranges.state)
                       << "; roots=" << ranges.roots.size();
            return failure();
          }
          auto range = ranges.roots.front();
          auto projections = gpu::queryRangeProjections(result, range);
          if (projections.size() != 1 ||
              result.getShape()[projections.front().fragmentAxis] !=
                  range.getResult().getType().getShape()[0])
            return gather.emitOpError(
                "cuTile tile extraction coordinate has no unique result axis");
          extractionShape[sourceAxis] =
              result.getShape()[projections.front().fragmentAxis];
          auto fullExtent =
              cast<gpu::PhysicalExprAttr>(source.getShape()[sourceAxis]);
          Value full;
          if (fullExtent.getKind() ==
              gpu::PhysicalExprKind::Parameter) {
            auto parameter = gpu::queryParameterBySymbol(kernel, fullExtent.getSymbolName());
            if (failed(parameter))
              return gather.emitOpError("tile extraction source extent has no parameter");
            full = gpu::materializeParameter(builder, gather.getLoc(), parameter->getReference());
          } else {
            full = builder.create<gpu::PhysicalExprOp>(
                gather.getLoc(), builder.getIndexType(), fullExtent);
          }
          auto index = extractionTileIndex(builder, gather.getLoc(), coordinate,
                                           range, full);
          if (failed(index) || !index->second)
            return gather.emitOpError(
                "cuTile coordinate is not an aligned rectangular sub-tile");
          Value extractionExtent = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), full, range.getExtent(),
              BinaryOperator::Maximum);
          Value count = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), extractionExtent, range.getExtent(),
              BinaryOperator::FloorDivide);
          Value one = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 1);
          Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
          Value last = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), count, one,
              BinaryOperator::Subtract);
          Value nonnegative = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), index->first, zero,
              BinaryOperator::Maximum);
          // An aligned sub-tile is either entirely inside or outside the
          // physical source. Preserve the gather mask/fill while keeping the
          // native extraction address valid for inactive sub-tiles.
          coordinate = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), nonnegative, last,
              BinaryOperator::Minimum);
          coordinates[sourceAxis] = coordinate;
          slicedAxes[sourceAxis] = true;
          continue;
        }
        auto integer =
            dyn_cast_or_null<IntegerAttr>(getCompileTimeScalar(coordinate));
        if (!integer && !isa<gpu::FragmentType>(coordinate.getType())) {
          // The shared access contract bounds active members. Clamp only the
          // extraction address; the original validity still selects the fill.
          if (!coordinate.getType().isIndex())
            coordinate = builder.create<gpu::CastOp>(
                gather.getLoc(), builder.getIndexType(), coordinate);
          Value extent = builder.create<gpu::PhysicalExprOp>(
              gather.getLoc(), builder.getIndexType(),
              cast<gpu::PhysicalExprAttr>(source.getShape()[sourceAxis]));
          Value zero = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
          Value one = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 1);
          Value last = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), extent, one,
              BinaryOperator::Subtract);
          Value nonnegative = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), coordinate, zero,
              BinaryOperator::Maximum);
          coordinates[sourceAxis] = builder.create<gpu::BinaryOp>(
              gather.getLoc(), builder.getIndexType(), nonnegative, last,
              BinaryOperator::Minimum);
          continue;
        }
        auto extent = gpu::IndexRelations().constant(
            cast<gpu::PhysicalExprAttr>(source.getShape()[sourceAxis]), kernel);
        if (!integer || !extent || integer.getInt() < 0 || integer.getInt() >= *extent)
          return gather.emitOpError(
              "cuTile rectangular tile extraction requires in-bounds constant selected coordinates");
        coordinate = builder.create<arith::ConstantIntOp>(
            gather.getLoc(), integer.getInt(), 32);
      } else {
        if (isa<gpu::FragmentType>(coordinate.getType()))
          return gather.emitOpError("cuTile scalar extraction requires scalar coordinates");
        if (access.getAccessValidity()) {
          Value zero = builder.create<arith::ConstantOp>(
              gather.getLoc(), coordinate.getType(),
              builder.getIntegerAttr(coordinate.getType(), 0));
          coordinate = builder.create<gpu::SelectOp>(
              gather.getLoc(), coordinate.getType(), access.getAccessValidity(), coordinate, zero);
        }
      }
      coordinates[sourceAxis] = coordinate;
    }
    SmallVector<int64_t> retainedAxes;
    for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
      if (coordinates[axis]) {
        if (slicedAxes[axis])
          retainedAxes.push_back(axis);
        continue;
      }
      if (!result)
        return gather.emitOpError("cuTile scalar extraction must select every source axis");
      retainedAxes.push_back(axis);
      extractionShape[axis] = source.getShape()[axis];
      coordinates[axis] = builder.create<arith::ConstantIntOp>(gather.getLoc(), 0, 32);
    }
    // In-bounds coordinates fit i32 under cuTile's tile-size limit.
    // This does not narrow external view dimensions or address arithmetic.
    for (Value &coordinate : coordinates)
      if (!coordinate.getType().isInteger(32))
        coordinate = builder.create<gpu::CastOp>(
            gather.getLoc(), builder.getI32Type(), coordinate);
    Value tile = access.getAccessResource();
    for (unsigned axis = 0; axis < slicedAxes.size(); ++axis)
      if (slicedAxes[axis])
        tile = repeatTileForExtraction(
            builder, gather.getLoc(), tile, axis,
            cast<gpu::PhysicalExprAttr>(extractionShape[axis]));
    Type extractedType = access.getAccessResult().getType();
    if (result) {
      SmallVector<Attribute> shape, maps;
      auto tileType = cast<gpu::FragmentType>(tile.getType());
      for (auto [resultAxis, sourceAxis] : llvm::enumerate(retainedAxes)) {
        shape.push_back(extractionShape[sourceAxis]);
        auto map = cast<gpu::AxisMapAttr>(tileType.getAxisMaps()[sourceAxis]);
        maps.push_back(gpu::AxisMapAttr::get(
            kernel.getContext(), map.getSourceId(), map.getSourceAxis(),
            map.getDimensionId(), resultAxis, map.getDerived()));
      }
      extractedType = gpu::FragmentType::get(
          kernel.getContext(), result.getElementType(), builder.getArrayAttr(shape),
          builder.getArrayAttr(maps), result.getValidity(), result.getOwner());
      if (builder.getArrayAttr(shape) != result.getShape() &&
          !gpu::queryBroadcastProjection(
               cast<gpu::FragmentType>(extractedType), result).isExact())
        return gather.emitOpError(
            "cuTile extraction requires an exact retained-axis broadcast");
    }
    auto replacement = builder.create<ExtractOp>(
        gather.getLoc(), extractedType, tile,
        coordinates, builder.getArrayAttr(extractionShape),
        builder.getDenseI64ArrayAttr(retainedAxes));
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    Value value = replacement.getResult();
    if (extractedType != access.getAccessResult().getType()) {
      auto extracted = cast<gpu::FragmentType>(extractedType);
      if (gpu::queryBroadcastProjection(extracted, result).isExact()) {
        value = builder.create<gpu::BroadcastOp>(gather.getLoc(), result, value);
      } else {
        auto reassociation = gpu::inferReshapeReassociation(extracted, result);
        if (failed(reassociation))
          return gather.emitOpError(
              "cuTile extraction has no row-major result domain projection");
        value = builder.create<gpu::ReshapeOp>(
            gather.getLoc(), result, value, *reassociation);
      }
    }
    if (access.getAccessValidity()) {
      auto selected = builder.create<gpu::SelectOp>(
          gather.getLoc(), access.getAccessResult().getType(), access.getAccessValidity(),
          value, access.getAccessFill());
      if (Attribute origin = gather->getAttr(gpu::originAttr))
        selected->setAttr(gpu::originAttr, origin);
      value = selected.getResult();
    }
    rewriter.replace(gather, ValueRange{value});
  }

  for (gpu::AtomicRMWOp atomic : inputs.atomics) {
    auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    const auto accessBounds = analysis.accessBounds(atomic);
    bool activeInBounds = accessBounds.isExact();
    auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
    if (!view)
      return atomic.emitOpError(
          "cuTile atomic RMW requires an external view resource");
    if (access.getAccessValidity() &&
        (!access.getAccessResult().use_empty() || view.getRank() == 0))
      return atomic.emitOpError(
          "masked cuTile array atomic requires a ranked view and an unused old value");
    auto tileType = dyn_cast<gpu::FragmentType>(access.getAccessPayloads().front().getType());
    Type element = view.getElementType();
    if (tileType && (element.isF16() || element.isBF16()) &&
        atomic.getKind() == AtomicRMWKind::Add &&
        atomic.getOrdering() == AtomicOrdering::Relaxed &&
        atomic.getSharing() == gpu::AtomicSharingDomain::KernelInvocation &&
        access.getAccessResult().use_empty()) {
      auto boundary = analysis.boundaryValidity(atomic);
      auto plan = analyzeNativeTileAccess(access, kernel, accessBounds);
      OpBuilder builder(atomic);
      FailureOr<MaterializedTileIndices> indices = failure();
      FailureOr<Value> originGuard = failure();
      if (succeeded(plan) && boundary.isExact())
        indices = materializeTileIndices(builder, atomic, *plan,
                                         /*allowDynamicAlignment=*/false);
      if (succeeded(indices) && !indices->alignment)
        originGuard = materializeTileOriginGuard(
            builder, atomic, access.getAccessResource(), *plan, boundary);
      if (succeeded(originGuard)) {
        auto emit = [&](OpBuilder &nested) {
          Value value = access.getAccessPayloads().front();
          if (!isIdentityPermutation(plan->toResource))
            value = nested.create<gpu::TransposeOp>(
                atomic.getLoc(), plan->packedType, value, plan->toResource);
          if (plan->packedToResource)
            value = nested.create<gpu::ReshapeOp>(
                atomic.getLoc(), plan->resourceType, value,
                plan->packedToResource);
          auto replacement = nested.create<TileAtomicAddOp>(
              atomic.getLoc(), access.getAccessResource(), indices->values, value);
          if (Attribute origin = atomic->getAttr(gpu::originAttr))
            replacement->setAttr(gpu::originAttr, origin);
        };
        if (*originGuard) {
          auto conditional = builder.create<scf::IfOp>(
              atomic.getLoc(), TypeRange{}, *originGuard,
              /*withElseRegion=*/false);
          OpBuilder body = prepareBranch(conditional.getThenRegion());
          emit(body);
          body.create<scf::YieldOp>(atomic.getLoc());
        } else {
          emit(builder);
        }
        rewriter.erase(atomic);
        continue;
      }
    }
    FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(access);
    if (failed(coordinates))
      return failure();
    OpBuilder builder(atomic);
    if (auto target = dyn_cast<gpu::FragmentType>(access.getAccessPayloads().front().getType())) {
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, atomic, *coordinates, target);
      if (failed(materialized))
        return failure();
      coordinates = std::move(*materialized);
    }
    if (access.getAccessValidity()) {
      // Native array atomics suppress out-of-bounds lanes but leave their
      // returned old values unspecified. The old value is unused here, so
      // encode the validity in the existing native bounds predicate.
      Type coordinateType = withElementType(access.getAccessPayloads().front().getType(),
                                            builder.getIndexType());
      Value zero = builder.create<arith::ConstantIndexOp>(atomic.getLoc(), 0);
      Value end = builder.create<gpu::DimOp>(
          atomic.getLoc(), builder.getIndexType(), access.getAccessResource(), 0);
      for (auto [axis, coordinate] : llvm::enumerate(*coordinates)) {
        Type indexed = withElementType(coordinate.getType(),
                                       builder.getIndexType());
        if (coordinate.getType() != indexed)
          coordinate = builder.create<gpu::CastOp>(atomic.getLoc(), indexed,
                                                   coordinate);
        auto active = gpu::projectPhysicalValueToSchema(
            builder, atomic.getLoc(), coordinate, coordinateType);
        auto inactive = gpu::projectPhysicalValueToSchema(
            builder, atomic.getLoc(), axis == 0 ? end : zero, coordinateType);
        if (failed(active) || failed(inactive))
          return atomic.emitOpError(
              "masked atomic coordinate cannot adopt its value domain");
        (*coordinates)[axis] = builder.create<gpu::SelectOp>(
            atomic.getLoc(), coordinateType, access.getAccessValidity(), *active,
            *inactive);
      }
    }
    auto replacement = builder.create<AtomicRMWOp>(
        atomic.getLoc(), access.getAccessResult().getType(), access.getAccessResource(),
        *coordinates, access.getAccessPayloads().front(), atomic.getKind(), atomic.getOrdering(),
        atomic.getSharing(), activeInBounds ? builder.getUnitAttr() : UnitAttr());
    if (Attribute origin = atomic->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    rewriter.replace(atomic, ValueRange{replacement.getResult()});
  }

  for (gpu::StoreOp store : inputs.stores) {
    auto access = cast<gpu::AccessOpInterface>(store.getOperation());
    gpu::PhysicalProgramAnalysis analysis(kernel);
    auto view = dyn_cast<gpu::ViewType>(access.getAccessResource().getType());
    if (!view)
      return store.emitOpError(
          "cuTile native store requires an external unique-write view");
    OpBuilder builder(store);
    if (!isa<gpu::FragmentType>(access.getAccessPayloads().front().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(access);
      if (failed(indices) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (access.getAccessValidity() && isa<gpu::FragmentType>(access.getAccessValidity().getType())))
        return store.emitOpError(
            "cuTile scalar store requires scalar indices and validity");
      bool inBounds = scalarCoordinatesInView(*indices, view, kernel);
      Value validity = access.getAccessValidity();
      gpu::PhysicalAccessBoundaryFact boundary =
          analysis.boundaryValidity(store);
      if (inBounds && boundary.isExact())
        validity = {};
      auto replacement = builder.create<ScalarStoreOp>(
          store.getLoc(), access.getAccessResource(), *indices, access.getAccessPayloads().front(),
          validity, inBounds ? builder.getUnitAttr() : UnitAttr());
      if (Attribute origin = store->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      rewriter.erase(store);
      continue;
    }
    gpu::PhysicalAccessBoundaryFact boundary =
        analysis.boundaryValidity(store, /*allowRangeGuards=*/true);
    const auto accessBounds = analysis.accessBounds(store);
    bool activeInBounds = accessBounds.isExact();
    auto computationType =
        cast<gpu::FragmentType>(access.getAccessPayloads().front().getType());
    FailureOr<NativeTileAccessPlan> plan = analyzeNativeTileAccess(access, kernel, accessBounds);
    FailureOr<MaterializedTileIndices> indices = failure();
    FailureOr<Value> originGuard = failure();
    if (succeeded(plan) && boundary.isExact())
      indices = materializeTileIndices(builder, store, *plan,
                                       /*allowDynamicAlignment=*/true);
    if (succeeded(indices))
      originGuard = materializeTileOriginGuard(
          builder, store, access.getAccessResource(), *plan, boundary);
    bool native = succeeded(indices) && succeeded(originGuard);
    bool guardedNative = native && *originGuard;
    Value nativeCondition;
    if (native) {
      nativeCondition = indices->alignment;
      Value rangeGuard = materializeFullRangeGuard(builder, store.getLoc(),
                                                  boundary);
      if (rangeGuard)
        nativeCondition = nativeCondition
                              ? Value(builder.create<gpu::BinaryOp>(
                                    store.getLoc(), builder.getI1Type(),
                                    nativeCondition, rangeGuard,
                                    BinaryOperator::LogicalAnd))
                              : rangeGuard;
    }
    FailureOr<Value> allowTMA = failure();
    if (native) {
      allowTMA = tmaCondition();
      if (failed(allowTMA))
        return failure();
    }
    SmallVector<Operation *> createdOperations;
    auto emitNativeStore = [&](OpBuilder &nested) {
      Value nativeValue = access.getAccessPayloads().front();
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
          store.getLoc(), access.getAccessResource(), *allowTMA, indices->values,
          nativeValue);
      createdOperations.push_back(tile);
    };
    auto emitBoundedNativeStore = [&](OpBuilder &nested) {
      if (!guardedNative) {
        emitNativeStore(nested);
        return;
      }
      auto conditional = nested.create<scf::IfOp>(
          store.getLoc(), TypeRange{}, *originGuard,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder inBounds = prepareBranch(conditional.getThenRegion());
      emitNativeStore(inBounds);
      inBounds.create<scf::YieldOp>(store.getLoc());
      OpBuilder outOfBounds = prepareBranch(conditional.getElseRegion());
      outOfBounds.create<scf::YieldOp>(store.getLoc());
    };
    auto emitScatterStore = [&](OpBuilder &nested) -> LogicalResult {
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(access);
      if (failed(coordinates))
        return failure();
      auto target = cast<gpu::FragmentType>(access.getAccessPayloads().front().getType());
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(nested, store, *coordinates, target);
      if (failed(materialized))
        return failure();
      auto replacement = nested.create<ScatterStoreOp>(
          store.getLoc(), access.getAccessResource(), *materialized, access.getAccessPayloads().front(),
          access.getAccessValidity(), identityAxes(view.getRank()),
          activeInBounds ? nested.getUnitAttr() : UnitAttr());
      createdOperations.push_back(replacement);
      return success();
    };
    if (native && nativeCondition) {
      auto conditional = builder.create<scf::IfOp>(
          store.getLoc(), TypeRange{}, nativeCondition,
          /*withElseRegion=*/true);
      createdOperations.push_back(conditional);
      OpBuilder tiled = prepareBranch(conditional.getThenRegion());
      emitBoundedNativeStore(tiled);
      tiled.create<scf::YieldOp>(store.getLoc());
      OpBuilder scattered = prepareBranch(conditional.getElseRegion());
      if (failed(emitScatterStore(scattered)))
        return failure();
      scattered.create<scf::YieldOp>(store.getLoc());
    } else if (native) {
      emitBoundedNativeStore(builder);
    } else if (failed(emitScatterStore(builder))) {
      return failure();
    }
    for (Operation *operation : createdOperations)
      if (Attribute origin = store->getAttr(gpu::originAttr))
        operation->setAttr(gpu::originAttr, origin);
    rewriter.erase(store);
  }
  return success();
}
} // namespace intent::cutile

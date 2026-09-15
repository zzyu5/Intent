#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

bool isZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getValue().isZero();
}

FailureOr<Value> replayFragmentValue(OpBuilder &builder, Value value,
                                     FragmentType target, IRMapping &mapping,
                                     PhysicalProgramAnalysis &analysis) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    Value replacement;
    for (const auto &entry : mapping.getValueMap()) {
      auto known = entry.first.getDefiningOp<MakeRangeOp>();
      if (!known || !sameLogicalRange(range, known) ||
          !samePhysicalScalarExpression(range.getStart(), known.getStart()) ||
          !samePhysicalScalarExpression(range.getExtent(), known.getExtent()))
        continue;
      if (replacement && replacement != entry.second)
        return failure();
      replacement = entry.second;
    }
    if (!replacement)
      return failure();
    mapping.map(value, replacement);
    return replacement;
  }
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record)
      return failure();
    FailureOr<Value> field = replayFragmentValue(
        builder, record.getFields()[extract.getField()], target, mapping, analysis);
    if (failed(field))
      return failure();
    mapping.map(value, *field);
    return *field;
  }
  PhysicalReplayFact replay = analysis.replayability(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false);
  Operation *producer = value.getDefiningOp();
  if (!producer || !replay.isReplayable())
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayFragmentValue(builder, operand, target, mapping, analysis);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  if (isa<BroadcastOp, ReshapeOp, TransposeOp>(producer)) {
    auto resultType = FragmentType::get(
        target.getContext(), fragment.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    // Replayed coordinates already use the destination axes.  The old shape
    // relation must not introduce its singleton axes a second time.
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, producer->getLoc(),
        mapping.lookupOrDefault(producer->getOperand(0)), resultType);
    if (failed(projected))
      return failure();
    mapping.map(value, *projected);
    return *projected;
  }
  Operation *clone = builder.clone(*producer, mapping);
  for (Value result : clone->getResults()) {
    auto original = dyn_cast<FragmentType>(result.getType());
    if (!original)
      continue;
    result.setType(FragmentType::get(
        target.getContext(), original.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner()));
  }
  Value result = clone->getResult(0);
  mapping.map(value, result);
  return result;
}

FailureOr<Value> replayScalarValue(OpBuilder &builder, Value value,
                                   IRMapping &mapping,
                                   PhysicalProgramAnalysis &analysis) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    FailureOr<Value> scalar =
        replayScalarValue(builder, broadcast.getValue(), mapping, analysis);
    if (succeeded(scalar))
      mapping.map(value, *scalar);
    return scalar;
  }
  if (auto splat = value.getDefiningOp<SplatOp>()) {
    FailureOr<Value> scalar =
        replayScalarValue(builder, splat.getValue(), mapping, analysis);
    if (succeeded(scalar))
      mapping.map(value, *scalar);
    return scalar;
  }
  PhysicalReplayFact replay = analysis.replayability(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false);
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<MakeRangeOp>(producer) || !replay.isReplayable() ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer))
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayScalarValue(builder, operand, mapping, analysis);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  Operation *clone = builder.clone(*producer, mapping);
  for (Value result : clone->getResults())
    if (auto resultType = dyn_cast<FragmentType>(result.getType()))
      result.setType(resultType.getElementType());
  Value result = clone->getResult(0);
  mapping.map(value, result);
  return result;
}

FailureOr<Value> combinePredicates(OpBuilder &builder, Location location,
                                   FragmentType valueType, Value lhs,
                                   Value rhs) {
  auto predicate = FragmentType::get(
      valueType.getContext(), builder.getI1Type(), valueType.getShape(),
      valueType.getAxisMaps(), valueType.getValidity(), valueType.getOwner());
  for (Value *value : {&lhs, &rhs}) {
    if (!*value)
      continue;
    if ((*value).getType() != predicate) {
      FailureOr<Value> projected =
          materializeBroadcastToFragment(builder, location, *value, predicate);
      if (failed(projected))
        return failure();
      *value = *projected;
    }
  }
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return Value(builder.create<BinaryOp>(location, predicate, lhs, rhs,
                                        BinaryOperator::LogicalAnd));
}

bool sameUniformValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto scalar = [](Value value) {
    while (value) {
      UniformExpression expression = describeUniformValue(value);
      if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
        break;
      value = expression.operands.front();
    }
    return value;
  };
  lhs = scalar(lhs);
  rhs = scalar(rhs);
  if (!lhs || !rhs || isa<FragmentType>(lhs.getType()) ||
      isa<FragmentType>(rhs.getType()))
    return false;
  if (lhs == rhs)
    return true;
  UniformValueAnalysis constants(describeUniformValue);
  return equalUniformConstants(constants.evaluate(lhs), constants.evaluate(rhs));
}

FailureOr<bool> composeSelectLoad(SelectOp select) {
  auto resultType = dyn_cast<FragmentType>(select.getResult().getType());
  Value loaded = select.getTrueValue();
  SmallVector<BroadcastOp> projections;
  while (auto broadcast = loaded.getDefiningOp<BroadcastOp>()) {
    auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
    auto result = dyn_cast<FragmentType>(broadcast.getResult().getType());
    if (!source || !result || source.getShape() != result.getShape() ||
        !broadcast.getResult().hasOneUse())
      return false;
    auto projection = queryAxisProjection(source, result);
    if (!projection.isExact() || llvm::any_of(
            llvm::enumerate(projection.targetToSource), [](auto item) {
              return !item.value() || *item.value() != item.index();
            }))
      return false;
    projections.push_back(broadcast);
    loaded = broadcast.getValue();
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  if (!resultType || !load || !load.getResult().hasOneUse() ||
      !canReplayReadAt(load, select))
    return false;

  Value fill = select.getFalseValue();
  if (load.getValid()) {
    if (!load.getFill() || !sameUniformValue(load.getFill(), fill))
      return false;
  } else if (load.getFill()) {
    return false;
  }

  OpBuilder builder(select);
  auto loadedType = projections.empty()
                        ? resultType
                        : cast<FragmentType>(load.getResult().getType());
  FailureOr<Value> valid = combinePredicates(
      builder, select.getLoc(), loadedType, load.getValid(),
      select.getCondition());
  if (failed(valid)) {
    select.emitOpError(
        "masked load predicate cannot adopt the loaded value relation");
    return failure();
  }
  if (fill.getType() != loadedType) {
    FailureOr<Value> projected = materializeBroadcastToFragment(
        builder, select.getLoc(), fill, loadedType);
    if (failed(projected)) {
      select.emitOpError(
          "masked load fill cannot adopt the loaded value relation");
      return failure();
    }
    fill = *projected;
  }
  auto replacement = builder.create<LoadOp>(
      select.getLoc(), loadedType, load.getResource(), load.getCoordinates(),
      *valid, fill, load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  // Positional broadcasts may rebind unit-axis provenance. Preserve that
  // relation instead of changing the load schema without its coordinates.
  IRMapping mapping;
  mapping.map(load.getResult(), replacement.getResult());
  for (BroadcastOp broadcast : llvm::reverse(projections))
    builder.clone(*broadcast, mapping);
  select.getResult().replaceAllUsesWith(
      mapping.lookupOrDefault(select.getTrueValue()));
  select.erase();
  for (BroadcastOp broadcast : projections)
    broadcast.erase();
  load.erase();
  return true;
}

FailureOr<bool> composeReshapedGather(GatherOp gather) {
  auto reshape = gather.getSource().getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!reshape || !result)
    return false;
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto shaped = cast<FragmentType>(reshape.getResult().getType());
  if (source.getShape() == shaped.getShape() &&
      llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        return group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1 &&
               group.getSourceAxes()[0] == group.getResultAxes()[0];
      }))
    return false;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (analysis.axisRealization(reshape.getValue(), axis).constructionScalarSeed)
      return false;
  for (unsigned axis = 0; axis < shaped.getShape().size(); ++axis)
    if (analysis.axisRealization(reshape.getResult(), axis).constructionScalarSeed)
      return false;
  unsigned sourceRank = 0, resultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    sourceRank += group.getSourceAxes().size();
    resultRank += group.getResultAxes().size();
  }
  unsigned sourcePrefix = source.getShape().size() - sourceRank;
  unsigned resultPrefix = shaped.getShape().size() - resultRank;
  if (sourcePrefix != resultPrefix)
    return false;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1)
      continue;
    for (auto [type, axes] :
         {std::pair{source, group.getSourceAxes().asArrayRef()},
          std::pair{shaped, group.getResultAxes().asArrayRef()}})
      for (int64_t axis : axes) {
        auto extent = cast<PhysicalExprAttr>(type.getShape()[sourcePrefix + axis]);
        if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() <= 0)
          return false;
      }
  }
  SmallVector<Value> coordinates(shaped.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
    coordinates[axis] = coordinate;
  SmallVector<unsigned> retainedAxes(shaped.getShape().size());
  for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
    if (coordinates[axis])
      continue;
    auto mapping = cast<AxisMapAttr>(shaped.getAxisMaps()[axis]);
    auto projection = queryFragmentAxis(result, sourceAxisIdentity(mapping));
    if (!projection.isExact() || projection.dimensionId != mapping.getDimensionId() ||
        result.getShape()[projection.fragmentAxis] != shaped.getShape()[axis])
      return false;
    retainedAxes[axis] = projection.fragmentAxis;
  }
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto indexType = FragmentType::get(
      result.getContext(), builder.getIndexType(), result.getShape(),
      result.getAxisMaps(), result.getValidity(), result.getOwner());
  auto extentValue = [&](Attribute attribute) -> Value {
    auto extent = cast<PhysicalExprAttr>(attribute);
    if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant))
      return builder.create<arith::ConstantIndexOp>(location, extent.getValue());
    return builder.create<PhysicalExprOp>(location, builder.getIndexType(), extent);
  };
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
    Value coordinate = coordinates[axis];
    if (!coordinate) {
      auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[retainedAxes[axis]]);
      auto type = FragmentType::get(
          result.getContext(), builder.getIndexType(),
          builder.getArrayAttr({shaped.getShape()[axis]}),
          builder.getArrayAttr({AxisMapAttr::get(
              result.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), 0, mapping.getDerived())}),
          result.getValidity(), result.getOwner());
      Value extent = extentValue(shaped.getShape()[axis]);
      coordinate = builder.create<MakeRangeOp>(
          location, type, zero, extent, one, zero, extent,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
    }
    Type element = coordinate.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (!element.isIndex()) {
      Type target = builder.getIndexType();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        target = FragmentType::get(result.getContext(), builder.getIndexType(),
            fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(),
            fragment.getOwner());
      coordinate = builder.create<CastOp>(location, target, coordinate);
    }
    auto projected = projectPhysicalValueToSchema(builder, location, coordinate, indexType);
    if (failed(projected))
      return gather.emitOpError("reshape gather lost an exact result coordinate projection");
    coordinates[axis] = *projected;
  }
  SmallVector<Value> selected(source.getShape().size());
  for (unsigned axis = 0; axis < sourcePrefix; ++axis)
    selected[axis] = coordinates[axis];
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty())
      continue;
    if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1) {
      selected[sourcePrefix + group.getSourceAxes()[0]] =
          coordinates[resultPrefix + group.getResultAxes()[0]];
      continue;
    }
    Value ordinal = builder.create<SplatOp>(location, indexType, zero);
    for (int64_t axis : group.getResultAxes().asArrayRef()) {
      Value extent = builder.create<SplatOp>(location, indexType,
          extentValue(shaped.getShape()[resultPrefix + axis]));
      ordinal = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                        BinaryOperator::Multiply);
      ordinal = builder.create<BinaryOp>(location, indexType, ordinal,
          coordinates[resultPrefix + axis], BinaryOperator::Add);
    }
    auto axes = group.getSourceAxes().asArrayRef();
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned axis = sourcePrefix + axes[position];
      Value coordinate = ordinal;
      if (position != 0) {
        Value extent = builder.create<SplatOp>(location, indexType,
                                              extentValue(source.getShape()[axis]));
        coordinate = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                              BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                           BinaryOperator::FloorDivide);
      }
      selected[axis] = coordinate;
    }
  }
  SmallVector<int64_t> axes;
  auto predicate = FragmentType::get(result.getContext(), builder.getI1Type(),
      result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
  Value valid = gather.getValid();
  Value lower = builder.create<SplatOp>(location, indexType, zero);
  for (unsigned axis = 0; axis < selected.size(); ++axis)
  {
    axes.push_back(axis);
    Value upper = builder.create<SplatOp>(location, indexType,
                                         extentValue(source.getShape()[axis]));
    Value nonnegative = builder.create<CompareOp>(location, predicate,
        selected[axis], lower, ComparePredicate::Ge);
    Value below = builder.create<CompareOp>(location, predicate,
        selected[axis], upper, ComparePredicate::Lt);
    Value bounded = builder.create<BinaryOp>(location, predicate, nonnegative,
                                            below, BinaryOperator::LogicalAnd);
    auto combined = combinePredicates(builder, location, result, valid, bounded);
    if (failed(combined))
      return failure();
    valid = *combined;
  }
  Value fill = gather.getFill();
  if (!fill) {
    auto zeroFill = materializeZeroFragment(builder, location, result);
    if (failed(zeroFill))
      return failure();
    fill = *zeroFill;
  }
  auto replacement = builder.create<GatherOp>(location, result, reshape.getValue(),
      selected, valid, fill, axes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

FailureOr<bool> composePointwiseGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  Operation *producer = gather.getSource().getDefiningOp();
  if (!source || !result || !producer ||
      !isa<UnaryOp, BinaryOp, BroadcastOp, SplatOp, ReshapeOp>(producer) ||
      !isa<FloatType>(source.getElementType()) ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;
  bool positionalRebind = false;
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    auto input = cast<FragmentType>(reshape.getValue().getType());
    positionalRebind = input.getShape() == source.getShape() &&
        llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getSourceAxes().size() == 1 &&
                 group.getResultAxes().size() == 1 &&
                 group.getSourceAxes()[0] == group.getResultAxes()[0];
        });
    if (!positionalRebind)
      return false;
  }
  SmallVector<Value> coordinates(source.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (axis < 0 || axis >= static_cast<int64_t>(coordinates.size()) ||
        coordinates[axis])
      return false;
    coordinates[axis] = coordinate;
  }
  if (llvm::any_of(coordinates, [](Value value) { return !value; }))
    return false;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  auto broadcastUnit = [&](Value operand, unsigned axis) {
    auto input = cast<FragmentType>(operand.getType());
    auto extent = cast<PhysicalExprAttr>(input.getShape()[axis]);
    if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        extent.getValue() != 1)
      return false;
    PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
    return ranges.isExact() && llvm::all_of(ranges.roots, [](MakeRangeOp range) {
      return isProvablySingletonLogicalRange(range);
    });
  };
  SmallVector<SmallVector<unsigned>> operandAxes;
  for (Value operand : producer->getOperands()) {
    SmallVector<unsigned> axes;
    if (auto input = dyn_cast<FragmentType>(operand.getType())) {
      if (positionalRebind) {
        for (unsigned axis = 0; axis < input.getShape().size(); ++axis)
          axes.push_back(axis);
        operandAxes.push_back(std::move(axes));
        continue;
      }
      BroadcastProjection projection = queryAxisProjection(input, source);
      if (!projection.isExact())
        return false;
      for (unsigned inputAxis = 0; inputAxis < input.getShape().size(); ++inputAxis) {
        std::optional<unsigned> selected;
        for (auto [axis, mapped] : llvm::enumerate(projection.targetToSource))
          if (mapped && *mapped == inputAxis) {
            if (selected)
              return false;
            selected = axis;
          }
        if (!selected)
          return false;
        if (!broadcastUnit(operand, inputAxis) &&
            input.getShape()[inputAxis] != source.getShape()[*selected])
          return false;
        axes.push_back(*selected);
      }
    }
    operandAxes.push_back(std::move(axes));
  }
  OpBuilder builder(gather);
  IRMapping mapping;
  for (auto [operand, axes] : llvm::zip(producer->getOperands(), operandAxes)) {
    auto input = dyn_cast<FragmentType>(operand.getType());
    if (!input || mapping.contains(operand))
      continue;
    auto selectedType = FragmentType::get(
        result.getContext(), input.getElementType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    SmallVector<Value> selectedCoordinates;
    SmallVector<int64_t> selectedAxes;
    for (auto [axis, sourceAxis] : llvm::enumerate(axes)) {
      Value coordinate = coordinates[sourceAxis];
      if (broadcastUnit(operand, axis))
        coordinate = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
      Type element = coordinate.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto indexType = FragmentType::get(
          result.getContext(), element, result.getShape(), result.getAxisMaps(),
          result.getValidity(), result.getOwner());
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, indexType);
      if (failed(projected))
        return gather.emitOpError("pointwise gather index lost its result projection");
      selectedCoordinates.push_back(*projected);
      selectedAxes.push_back(axis);
    }
    Value fill;
    if (gather.getValid()) {
      FailureOr<Value> zero = materializeZeroFragment(builder, gather.getLoc(), selectedType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    mapping.map(operand, builder.create<GatherOp>(
        gather.getLoc(), selectedType, operand, selectedCoordinates,
        gather.getValid(), fill, selectedAxes).getResult());
  }
  // Keep reductions and immutable reads as captured SSA producers. Only the
  // floating pointwise suffix moves to the selected coordinates. Inactive
  // operands use zero; restore the original gather fill after IEEE arithmetic.
  Value value;
  if (positionalRebind) {
    value = mapping.lookup(producer->getOperand(0));
  } else {
    Operation *replacement = builder.clone(*producer, mapping);
    replacement->getResult(0).setType(result);
    value = replacement->getResult(0);
  }
  if (gather.getValid())
    value = builder.create<SelectOp>(gather.getLoc(), result, gather.getValid(),
                                     value, gather.getFill());
  gather.getResult().replaceAllUsesWith(value);
  gather.erase();
  return true;
}

FailureOr<bool> composeLoadGather(GatherOp gather) {
  auto sourceType = dyn_cast<FragmentType>(gather.getSource().getType());
  auto sourceLoad = gather.getSource().getDefiningOp<LoadOp>();
  if (!sourceType || !sourceLoad ||
      !isa<ViewType>(sourceLoad.getResource().getType()) ||
      !canReplayReadAt(sourceLoad, gather))
    return false;
  if (gather.getCoordinates().size() != gather.getSourceAxes().size()) {
    gather.emitOpError("gather coordinate/source-axis schema is incomplete");
    return failure();
  }

  OpBuilder builder(gather);
  auto resultType = dyn_cast<FragmentType>(gather.getResult().getType());
  SmallVector<Value> originalCoordinates;
  for (Value coordinate : sourceLoad.getCoordinates()) {
    while (auto broadcast = coordinate.getDefiningOp<BroadcastOp>())
      coordinate = broadcast.getValue();
    originalCoordinates.push_back(coordinate);
  }
  SmallVector<Value> coordinates(originalCoordinates);
  IRMapping replay;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(sourceType.getShape().size())) {
      gather.emitOpError("gather source axis is outside its loaded value");
      return failure();
    }
    FailureOr<AxisMapAttr> mapping =
        queryAxisMap(sourceType, static_cast<unsigned>(sourceAxis));
    if (failed(mapping)) {
      gather.emitOpError("gather source axis lost coordinate provenance");
      return failure();
    }
    PhysicalAxisProjection target = queryCoordinateIndex(
        originalCoordinates, sourceAxisIdentity(*mapping));
    if (!target.isExact() || target.dimensionId != mapping->getDimensionId()) {
      gather.emitOpError(
          "loaded source coordinate cannot be composed with gather indexing");
      return failure();
    }
    Value original = originalCoordinates[target.fragmentAxis];
    auto range = original.getDefiningOp<MakeRangeOp>();
    if (!range) {
      auto coordinateType = dyn_cast<FragmentType>(original.getType());
      if (!coordinateType || coordinateType.getShape().size() != 1)
        return false;
      PhysicalRangeFact roots = analysis.axisRanges(original, 0);
      if (!roots.isUnique() ||
          !analysis.replayability(original, std::nullopt,
                                  PhysicalReplayScope::Coordinate,
                                  /*allowAccesses=*/false).isReplayable())
        return false;
      range = roots.roots.front();
      auto dimension = queryRangeDimension(range);
      auto axis = queryAxisMap(coordinateType, 0);
      if (coordinateType.getShape()[0] !=
              range.getResult().getType().getShape()[0] ||
          failed(dimension) || failed(axis) ||
          *dimension != axis->getDimensionId())
        return false;
    }
    if (resultType) {
      Type element = coordinate.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto coordinateType = FragmentType::get(
          resultType.getContext(), element, resultType.getShape(),
          resultType.getAxisMaps(), resultType.getValidity(), resultType.getOwner());
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, coordinateType);
      if (failed(projected))
        return gather.emitOpError(
            "composed gather index cannot adopt its result coordinate relation");
      coordinate = *projected;
    }
    if (!isZero(range.getStart()) || !isUnitStepRange(range)) {
      Type indexType = range.getResult().getType().getElementType();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        indexType = FragmentType::get(
            fragment.getContext(), indexType, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      if (coordinate.getType() != indexType)
        coordinate = builder.create<CastOp>(gather.getLoc(), indexType, coordinate);
      auto projectedBound = [&](Value bound) -> Value {
        if (auto fragment = dyn_cast<FragmentType>(indexType))
          return builder.create<BroadcastOp>(gather.getLoc(), fragment, bound);
        return bound;
      };
      // Gather indexes positions in the loaded tensor. Its ordinal must be
      // composed with the load range before becoming a resource coordinate.
      if (!isUnitStepRange(range))
        coordinate = builder.create<BinaryOp>(
            gather.getLoc(), indexType, coordinate, projectedBound(range.getStep()),
            BinaryOperator::Multiply);
      if (!isZero(range.getStart()))
        coordinate = builder.create<BinaryOp>(
            gather.getLoc(), indexType, projectedBound(range.getStart()), coordinate,
            BinaryOperator::Add);
    }
    replay.map(range.getResult(), coordinate);
    if (original != range.getResult()) {
      FailureOr<Value> selected = resultType
          ? replayFragmentValue(builder, original, resultType, replay, analysis)
          : replayScalarValue(builder, original, replay, analysis);
      if (failed(selected))
        return gather.emitOpError("indexed load coordinate cannot follow its source range");
      coordinate = *selected;
    }
    replay.map(original, coordinate);
    coordinates[target.fragmentAxis] = coordinate;
  }

  if (resultType) {
    for (auto [slot, original] : llvm::enumerate(originalCoordinates)) {
      if (replay.lookupOrNull(original) || !isa<FragmentType>(original.getType()))
        continue;
      auto coordinateType = cast<FragmentType>(original.getType());
      auto indexType = FragmentType::get(
          resultType.getContext(), coordinateType.getElementType(),
          resultType.getShape(), resultType.getAxisMaps(),
          resultType.getValidity(), resultType.getOwner());
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), original, indexType);
      if (failed(projected))
        return gather.emitOpError(
            "retained load coordinate cannot adopt the gather result relation");
      coordinates[slot] = *projected;
      replay.map(original, *projected);
    }
    for (Value predicateOrFill : {sourceLoad.getValid(), sourceLoad.getFill()}) {
      if (!predicateOrFill)
        continue;
      PhysicalRangeFact roots = analysis.sourceRanges(predicateOrFill);
      // Bounds may use the raw range while the address uses a guarded index.
      // Preserve each untouched range's values, not the address expression.
      for (MakeRangeOp range : roots.roots) {
        if (replay.lookupOrNull(range.getResult()))
          continue;
        auto sourceAxis = queryFragmentAxis(sourceType, sourceAxisIdentity(range));
        if (!sourceAxis.isExact() ||
            llvm::is_contained(gather.getSourceAxes(),
                               static_cast<int64_t>(sourceAxis.fragmentAxis)))
          continue;
        auto axis = queryFragmentAxis(resultType, sourceAxisIdentity(range));
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        auto rangeType = range.getResult().getType();
        if (!axis.isExact() || failed(dimension) ||
            sourceAxis.dimensionId != *dimension ||
            axis.dimensionId != *dimension ||
            resultType.getShape()[axis.fragmentAxis] != rangeType.getShape()[0])
          continue;
        auto rangeTarget = FragmentType::get(
            resultType.getContext(), rangeType.getElementType(),
            resultType.getShape(), resultType.getAxisMaps(),
            resultType.getValidity(), resultType.getOwner());
        FailureOr<Value> projectedRange = projectPhysicalValueToSchema(
            builder, gather.getLoc(), range.getResult(), rangeTarget);
        if (failed(projectedRange))
          return gather.emitOpError(
              "retained coordinate range cannot adopt the gather result relation");
        replay.map(range.getResult(), *projectedRange);
      }
    }
  }

  if (!resultType) {
    FailureOr<Value> sourceValid = replayScalarValue(
        builder, sourceLoad.getValid(), replay, analysis);
    FailureOr<Value> sourceFill = replayScalarValue(
        builder, sourceLoad.getFill(), replay, analysis);
    if (failed(sourceValid) || failed(sourceFill)) {
      gather.emitOpError(
          "source access validity/fill cannot follow scalar composed coordinates");
      return failure();
    }
    Value valid = *sourceValid;
    if (gather.getValid()) {
      if (!gather.getValid().getType().isInteger(1)) {
        gather.emitOpError("scalar gather validity is not scalar i1");
        return failure();
      }
      valid = valid ? Value(builder.create<BinaryOp>(
                            gather.getLoc(), builder.getI1Type(), valid,
                            gather.getValid(), BinaryOperator::LogicalAnd))
                    : gather.getValid();
    }
    Value fill = *sourceFill;
    if (sourceLoad.getValid() && gather.getValid()) {
      if (!fill || !gather.getFill() ||
          fill.getType() != gather.getResult().getType() ||
          gather.getFill().getType() != gather.getResult().getType()) {
        gather.emitOpError(
            "scalar composed conditional access has incompatible fills");
        return failure();
      }
      fill = builder.create<SelectOp>(
          gather.getLoc(), gather.getResult().getType(), gather.getValid(), fill,
          gather.getFill());
    } else if (gather.getValid()) {
      fill = gather.getFill();
    }
    if (static_cast<bool>(valid) != static_cast<bool>(fill)) {
      gather.emitOpError(
          "scalar composed access requires paired validity and fill");
      return failure();
    }
    auto replacement = builder.create<LoadOp>(
        gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
        coordinates, valid, fill, sourceLoad.getSourceAxes());
    if (Attribute origin = sourceLoad->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
    if (sourceLoad->getBlock() && sourceLoad.getResult().use_empty())
      sourceLoad.erase();
    return true;
  }
  FailureOr<Value> sourceValid = replayFragmentValue(
      builder, sourceLoad.getValid(), resultType, replay, analysis);
  FailureOr<Value> sourceFill = replayFragmentValue(
      builder, sourceLoad.getFill(), resultType, replay, analysis);
  if (failed(sourceValid) || failed(sourceFill)) {
    gather.emitOpError(
        "source access validity/fill cannot follow composed coordinates");
    return failure();
  }
  FailureOr<Value> valid = combinePredicates(
      builder, gather.getLoc(), resultType, *sourceValid, gather.getValid());
  if (failed(valid)) {
    gather.emitOpError(
        "source and gather validity cannot share the composed result relation");
    return failure();
  }
  Value fill = *sourceFill;
  if (sourceLoad.getValid() && gather.getValid()) {
    Value gatherFill = gather.getFill();
    if (!fill || !gatherFill) {
      gather.emitOpError(
          "composed conditional access requires both source and gather fill");
      return failure();
    }
    auto result = resultType;
    if (fill.getType() != result)
      fill = builder.create<BroadcastOp>(gather.getLoc(), result, fill);
    if (gatherFill.getType() != result)
      gatherFill =
          builder.create<BroadcastOp>(gather.getLoc(), result, gatherFill);
    Value condition = gather.getValid();
    auto predicate = FragmentType::get(
        result.getContext(), builder.getI1Type(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    if (condition.getType() != predicate)
      condition =
          builder.create<BroadcastOp>(gather.getLoc(), predicate, condition);
    fill = builder.create<SelectOp>(gather.getLoc(), result, condition, fill,
                                    gatherFill);
  } else if (gather.getValid()) {
    fill = gather.getFill();
  }
  auto replacement = builder.create<LoadOp>(
      gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
      coordinates, *valid, fill,
      sourceLoad.getSourceAxes());
  if (Attribute origin = sourceLoad->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  if (sourceLoad->getBlock() && sourceLoad.getResult().use_empty())
    sourceLoad.erase();
  return true;
}

FailureOr<bool> composeIdentityFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result || source.getOwner() != result.getOwner() ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;
  SmallVector<bool> represented(result.getShape().size(), false);
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return false;
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    PhysicalSourceAxis physicalSource = sourceAxisIdentity(expected);
    PhysicalAxisProjection resultAxis =
        queryFragmentAxis(result, physicalSource);
    PhysicalAxisProjection coordinateAxis =
        queryCoordinateIndex(ValueRange{coordinate}, physicalSource);
    auto resultMapping =
        resultAxis.isExact()
            ? dyn_cast<AxisMapAttr>(result.getAxisMaps()[resultAxis.fragmentAxis])
            : AxisMapAttr();
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
    if (!resultAxis.isExact() || !coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType || !coordinateRange ||
        !isZero(coordinateRange.getStart()) ||
        !isUnitStepRange(coordinateRange) ||
        source.getShape()[sourceAxis] !=
            result.getShape()[resultAxis.fragmentAxis] ||
        coordinateType.getShape().size() != 1 ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        coordinateAxis.fragmentAxis != 0)
      return false;
    represented[resultAxis.fragmentAxis] = true;
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (represented[axis])
      continue;
    auto expression = dyn_cast<PhysicalExprAttr>(extent);
    if (!expression ||
        expression.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        expression.getValue() != 1)
      return false;
  }
  OpBuilder builder(gather);
  Value replacement = builder.create<BroadcastOp>(gather.getLoc(), result,
                                                   gather.getSource());
  if (gather.getValid())
    replacement = builder.create<SelectOp>(gather.getLoc(), result,
                                           gather.getValid(), replacement,
                                           gather.getFill());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement.getDefiningOp()->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement);
  gather.erase();
  return true;
}

FailureOr<bool> projectFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;

  SmallVector<Value> selectedCoordinates;
  SmallVector<int64_t> selectedAxes;
  bool projected = false;
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return false;
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    PhysicalSourceAxis physicalSource = sourceAxisIdentity(expected);
    PhysicalAxisProjection resultAxis =
        queryFragmentAxis(result, physicalSource);
    if (resultAxis.state == PhysicalFactState::Ambiguous)
      return false;
    if (!resultAxis.isExact()) {
      selectedCoordinates.push_back(coordinate);
      selectedAxes.push_back(sourceAxis);
      continue;
    }
    PhysicalAxisProjection coordinateAxis =
        queryCoordinateIndex(ValueRange{coordinate}, physicalSource);
    auto resultMapping =
        dyn_cast<AxisMapAttr>(result.getAxisMaps()[resultAxis.fragmentAxis]);
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    if (!coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType ||
        coordinateType.getShape().size() != 1 ||
        source.getShape()[sourceAxis] !=
            result.getShape()[resultAxis.fragmentAxis] ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        coordinateAxis.fragmentAxis != 0)
      return false;
    projected = true;
  }
  if (!projected || selectedCoordinates.empty())
    return false;

  OpBuilder builder(gather);
  auto replacement = builder.create<GatherOp>(
      gather.getLoc(), result, gather.getSource(), selectedCoordinates,
      gather.getValid(), gather.getFill(), selectedAxes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

bool hasNonUnitAxisSplit(ReshapeOp reshape) {
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto result = cast<FragmentType>(reshape.getResult().getType());
  if (source.getShape().size() >= result.getShape().size())
    return false;
  bool split = false;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty())
      return false;
    if (group.getSourceAxes().empty())
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1)
          return false;
      }
    if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() > 1) {
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() <= 1)
          return false;
      }
      split = true;
    }
  }
  return split;
}

bool composeReshapedPointwise(ReshapeOp reshape) {
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto result = cast<FragmentType>(reshape.getResult().getType());
  if (!hasNonUnitAxisSplit(reshape) &&
      source.getShape().size() <= result.getShape().size())
    return false;
  Operation *producer = reshape.getValue().getDefiningOp();
  if (!producer ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
           SplatOp, BroadcastOp>(producer))
    return false;
  for (Value operand : producer->getOperands()) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (fragment &&
        (fragment.getShape() != source.getShape() ||
         fragment.getAxisMaps() != source.getAxisMaps() ||
         fragment.getOwner() != source.getOwner()))
      return false;
  }
  OpBuilder builder(reshape);
  IRMapping mapping;
  for (Value operand : producer->getOperands()) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment || mapping.contains(operand))
      continue;
    auto target = FragmentType::get(
        result.getContext(), fragment.getElementType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    mapping.map(operand, builder.create<ReshapeOp>(
                             reshape.getLoc(), target, operand,
                             reshape.getReassociation()).getResult());
  }
  Operation *replacement = builder.clone(*producer, mapping);
  replacement->getResult(0).setType(result);
  reshape.getResult().replaceAllUsesWith(replacement->getResult(0));
  reshape.erase();
  return true;
}

FailureOr<bool> composeReshapedLoad(ReshapeOp reshape) {
  Value sourceValue = reshape.getValue();
  Value loaded = sourceValue;
  bool transposed = false;
  while (true) {
    if (auto broadcast = loaded.getDefiningOp<BroadcastOp>()) {
      auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
      auto output = dyn_cast<FragmentType>(broadcast.getResult().getType());
      if (!input || !output || input.getShape() != output.getShape())
        break;
      BroadcastProjection projection = queryAxisProjection(input, output);
      if (!projection.isExact() || llvm::any_of(
              llvm::enumerate(projection.targetToSource), [](auto item) {
                return !item.value() || *item.value() != item.index();
              }))
        break;
      loaded = broadcast.getValue();
      continue;
    }
    auto transpose = loaded.getDefiningOp<TransposeOp>();
    if (!transpose)
      break;
    auto source = cast<FragmentType>(transpose.getValue().getType());
    auto result = cast<FragmentType>(transpose.getResult().getType());
    for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
      auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
      auto transposed = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
      if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
          original.getDimensionId() != transposed.getDimensionId())
        return false;
    }
    transposed = true;
    loaded = transpose.getValue();
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
  if (!load || !result || !isa<ViewType>(load.getResource().getType()) ||
      !canReplayReadAt(load, reshape))
    return false;
  auto source = cast<FragmentType>(sourceValue.getType());
  if (source.getShape().size() <= result.getShape().size() &&
      !hasNonUnitAxisSplit(reshape) && !transposed)
    return false;
  auto kernel = reshape->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  unsigned sourceRank = 0;
  unsigned resultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty())
      for (int64_t axis : group.getSourceAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
        PhysicalRangeFact ranges = analysis.axisRanges(sourceValue, axis);
        if (extent.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1 || !ranges.isExact() ||
            !llvm::all_of(ranges.roots, isProvablySingletonLogicalRange))
          return false;
      }
    if (group.getSourceAxes().empty())
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            extent.getValue() != 1)
          return false;
      }
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      sourceRank = std::max(sourceRank, static_cast<unsigned>(axis + 1));
    for (int64_t axis : group.getResultAxes().asArrayRef())
      resultRank = std::max(resultRank, static_cast<unsigned>(axis + 1));
  }
  if (sourceRank != source.getShape().size() || resultRank != result.getShape().size())
    return false;
  SmallVector<bool> preservedSource(sourceRank, false);
  SmallVector<bool> preservedResult(resultRank, false);
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty()) {
      for (int64_t axis : group.getResultAxes().asArrayRef())
        preservedResult[axis] = true;
      continue;
    }
    if (group.getResultAxes().empty()) {
      for (int64_t axis : group.getSourceAxes().asArrayRef())
        preservedSource[axis] = true;
      continue;
    }
    if (group.getSourceAxes().size() != 1 ||
        group.getResultAxes().size() != 1)
      continue;
    unsigned sourceAxis = group.getSourceAxes()[0];
    unsigned resultAxis = group.getResultAxes()[0];
    auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    auto target = cast<AxisMapAttr>(result.getAxisMaps()[resultAxis]);
    if (sourceAxisIdentity(original) == sourceAxisIdentity(target) &&
        original.getDimensionId() == target.getDimensionId() &&
        source.getShape()[sourceAxis] == result.getShape()[resultAxis]) {
      preservedSource[sourceAxis] = true;
      preservedResult[resultAxis] = true;
    }
  }
  SmallVector<MakeRangeOp> sourceRanges(sourceRank);
  SmallVector<SmallVector<MakeRangeOp>> sourceRoots(sourceRank);
  SmallVector<Attribute> sourceExtents(sourceRank);
  llvm::DenseMap<Operation *, unsigned> rootAxes;
  for (unsigned axis = 0; axis < sourceRank; ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(sourceValue, axis);
    for (MakeRangeOp root : fact.roots) {
      auto [found, inserted] = rootAxes.try_emplace(root.getOperation(), axis);
      if (!inserted && found->second != axis)
        return false;
    }
    sourceRoots[axis].append(fact.roots.begin(), fact.roots.end());
    if (preservedSource[axis]) {
      auto realization = analysis.axisRealization(sourceValue, axis);
      bool introducedUnit = fact.roots.empty() && realization.isExact() &&
                            !realization.constructionScalarSeed &&
                            cast<PhysicalExprAttr>(source.getShape()[axis]).getKind() ==
                                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                            cast<PhysicalExprAttr>(source.getShape()[axis]).getValue() == 1;
      if (fact.state != PhysicalFactState::Exact && !introducedUnit)
        return false;
      continue;
    }
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(fact);
    if (failed(authority))
      return false;
    for (MakeRangeOp root : fact.roots) {
      if (!isZero(root.getStart()) || !isZero(root.getLogicalStart()) ||
          !isUnitStepRange(root))
        return false;
      auto realization = analysis.axisRealization(root.getResult(), 0);
      if (!realization.constructionScalarSeed &&
          !samePhysicalScalarExpression(root.getExtent(), root.getLogicalStop()))
        return false;
    }
    MakeRangeOp range = *authority;
    PhysicalExprAttr extent = queryLaunchExpression(range.getLogicalStop());
    if (!extent)
      return false;
    if (!queryNonNegativeIndexUpperBound(range.getLogicalStop())) {
      auto zero = PhysicalExprAttr::get(
          reshape.getContext(), static_cast<uint32_t>(PhysicalExprKind::Constant),
          0, StringAttr::get(reshape.getContext(), ""), ArrayAttr::get(reshape.getContext(), {}));
      extent = PhysicalExprAttr::get(
          reshape.getContext(), static_cast<uint32_t>(PhysicalExprKind::Maximum),
          0, StringAttr::get(reshape.getContext(), ""),
          ArrayAttr::get(reshape.getContext(), {extent, zero}));
    }
    sourceRanges[axis] = range;
    sourceExtents[axis] = extent;
  }
  SmallVector<PhysicalExprAttr> resultExtents(resultRank);
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty())
      continue;
    unsigned resultAxis = group.getResultAxes()[0];
    if (preservedResult[resultAxis])
      continue;
    auto axes = group.getSourceAxes().asArrayRef();
    auto extent = cast<PhysicalExprAttr>(sourceExtents[axes.front()]);
    if (group.getResultAxes().size() > 1) {
      // The declared row-major group may split a complete source axis.
      // Static result extents give each constituent its own exact range;
      // physical padding or a partial source tile cannot satisfy this proof.
      int64_t product = 1;
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto part = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (part.getKind() !=
                static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            part.getValue() <= 0 ||
            product > std::numeric_limits<int64_t>::max() / part.getValue())
          return false;
        product *= part.getValue();
        resultExtents[axis] = part;
      }
      int64_t sourceProduct = 1;
      for (int64_t axis : axes) {
        auto part = cast<PhysicalExprAttr>(sourceExtents[axis]);
        if (part.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
            part.getValue() <= 0 ||
            sourceProduct > std::numeric_limits<int64_t>::max() / part.getValue())
          return false;
        sourceProduct *= part.getValue();
      }
      if (sourceProduct != product)
        return false;
      continue;
    }
    for (int64_t axis : axes.drop_front())
      extent = PhysicalExprAttr::get(reshape.getContext(),
          static_cast<uint32_t>(PhysicalExprKind::Multiply), 0,
          StringAttr::get(reshape.getContext(), ""),
          ArrayAttr::get(reshape.getContext(), {extent, sourceExtents[axis]}));
    resultExtents[resultAxis] = extent;
  }
  if (llvm::all_of(preservedResult, [](bool preserved) { return preserved; }))
    return false;
  OpBuilder builder(reshape);
  auto materializeExtent = [&](PhysicalExprAttr extent) -> Value {
    if (extent.getKind() ==
        static_cast<uint32_t>(PhysicalExprKind::Constant))
      return builder.create<arith::ConstantIndexOp>(reshape.getLoc(),
                                                    extent.getValue());
    return builder.create<PhysicalExprOp>(reshape.getLoc(), builder.getIndexType(),
                                         extent);
  };
  SmallVector<Value> resultStops(resultRank);
  for (unsigned axis = 0; axis < resultRank; ++axis)
    if (!preservedResult[axis])
      resultStops[axis] = materializeExtent(resultExtents[axis]);

  Value zero = builder.create<arith::ConstantIndexOp>(reshape.getLoc(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(reshape.getLoc(), 1);
  SmallVector<Value> flatCoordinates(resultRank);
  auto indexType = FragmentType::get(
      result.getContext(), builder.getIndexType(), result.getShape(),
      result.getAxisMaps(), result.getValidity(), result.getOwner());
  for (unsigned axis = 0; axis < resultRank; ++axis) {
    if (preservedResult[axis])
      continue;
    auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
    Value stop = resultStops[axis];
    Value extent = materializeExtent(
        cast<PhysicalExprAttr>(result.getShape()[axis]));
    auto rangeType = FragmentType::get(
        result.getContext(), builder.getIndexType(),
        builder.getArrayAttr({result.getShape()[axis]}),
        builder.getArrayAttr({AxisMapAttr::get(
            result.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), 0, mapping.getDerived())}),
        result.getValidity(), result.getOwner());
    Value range = builder.create<MakeRangeOp>(
        reshape.getLoc(), rangeType, zero, extent, one, zero, stop,
        mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, reshape.getLoc(), range, indexType);
    if (failed(projected))
      return reshape.emitOpError("collapsed load has no flat coordinate projection");
    flatCoordinates[axis] = *projected;
  }
  SmallVector<Value> sourceCoordinates(sourceRank);
  IRMapping mapping;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty())
      continue;
    if (group.getResultAxes().empty()) {
      // A removed logical singleton still contributes its original address.
      // It becomes uniform in the destination shape, not an extra flat axis.
      for (int64_t axis : group.getSourceAxes().asArrayRef())
        for (MakeRangeOp root : sourceRoots[axis])
          mapping.map(root.getResult(), builder.create<SplatOp>(
              reshape.getLoc(), indexType, root.getStart()).getResult());
      continue;
    }
    if (preservedResult[group.getResultAxes()[0]]) {
      // Unmerged axes retain their current tile and coordinates, including
      // program-local batch coordinates and already blocked free dimensions.
      for (MakeRangeOp root : sourceRoots[group.getSourceAxes()[0]]) {
        unsigned axis = group.getResultAxes()[0];
        SmallVector<Attribute> shape(resultRank,
            PhysicalExprAttr::get(result.getContext(),
                static_cast<uint32_t>(PhysicalExprKind::Constant), 1,
                builder.getStringAttr(""), builder.getArrayAttr({})));
        shape[axis] = root.getResult().getType().getShape()[0];
        SmallVector<Attribute> groups;
        for (unsigned position = 0; position < resultRank; ++position) {
          SmallVector<int64_t> inputAxes;
          if (position == axis)
            inputAxes.push_back(0);
          groups.push_back(ReshapeGroupAttr::get(result.getContext(),
              builder.getDenseI64ArrayAttr(inputAxes),
              builder.getDenseI64ArrayAttr({position})));
        }
        auto shaped = FragmentType::get(result.getContext(), builder.getIndexType(),
            builder.getArrayAttr(shape), indexType.getAxisMaps(),
            indexType.getValidity(), indexType.getOwner());
        Value positioned = builder.create<ReshapeOp>(reshape.getLoc(), shaped,
            root.getResult(), builder.getArrayAttr(groups));
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, reshape.getLoc(), positioned, indexType);
        if (failed(projected))
          return reshape.emitOpError("collapsed load lost an unmerged axis projection");
        mapping.map(root.getResult(), *projected);
      }
      continue;
    }
    Value ordinal = flatCoordinates[group.getResultAxes()[0]];
    for (int64_t axis : group.getResultAxes().asArrayRef().drop_front()) {
      Value extent = builder.create<SplatOp>(reshape.getLoc(), indexType,
                                            resultStops[axis]);
      ordinal = builder.create<BinaryOp>(reshape.getLoc(), indexType, ordinal,
                                        extent, BinaryOperator::Multiply);
      ordinal = builder.create<BinaryOp>(reshape.getLoc(), indexType, ordinal,
                                        flatCoordinates[axis],
                                        BinaryOperator::Add);
    }
    auto axes = group.getSourceAxes().asArrayRef();
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned axis = axes[position];
      MakeRangeOp range = sourceRanges[axis];
      Value coordinate = ordinal;
      if (position != 0) {
        Value divisor = builder.create<BinaryOp>(
            reshape.getLoc(), builder.getIndexType(), range.getLogicalStop(), one,
            BinaryOperator::Maximum);
        Value extent = builder.create<SplatOp>(reshape.getLoc(), indexType, divisor);
        coordinate = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, ordinal, extent, BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, ordinal, extent, BinaryOperator::FloorDivide);
      }
      sourceCoordinates[axis] = coordinate;
      for (MakeRangeOp root : sourceRoots[axis])
        mapping.map(root.getResult(), coordinate);
    }
  }
  SmallVector<Value> coordinates;
  for (Value coordinate : load.getCoordinates()) {
    FailureOr<Value> replayed = replayFragmentValue(
        builder, coordinate, result, mapping, analysis);
    if (failed(replayed))
      return reshape.emitOpError("collapsed load could not preserve its coordinate graph");
    coordinates.push_back(*replayed);
  }
  FailureOr<Value> valid = replayFragmentValue(
      builder, load.getValid(), result, mapping, analysis);
  FailureOr<Value> fill = replayFragmentValue(
      builder, load.getFill(), result, mapping, analysis);
  if (failed(valid) || failed(fill))
    return reshape.emitOpError("collapsed load could not preserve validity and fill");
  auto predicate = FragmentType::get(
      result.getContext(), builder.getI1Type(), result.getShape(), result.getAxisMaps(),
      result.getValidity(), result.getOwner());
  Value lower = builder.create<SplatOp>(reshape.getLoc(), indexType, zero);
  Value active = *valid;
  for (unsigned axis = 0; axis < sourceRank; ++axis) {
    if (preservedSource[axis])
      continue;
    Value coordinate = sourceCoordinates[axis];
    Value end = builder.create<SplatOp>(
        reshape.getLoc(), indexType, sourceRanges[axis].getLogicalStop());
    Value nonNegative = builder.create<CompareOp>(
        reshape.getLoc(), predicate, coordinate, lower, ComparePredicate::Ge);
    Value belowEnd = builder.create<CompareOp>(
        reshape.getLoc(), predicate, coordinate, end, ComparePredicate::Lt);
    Value within = builder.create<BinaryOp>(
        reshape.getLoc(), predicate, nonNegative, belowEnd, BinaryOperator::LogicalAnd);
    active = active ? Value(builder.create<BinaryOp>(
        reshape.getLoc(), predicate, active, within, BinaryOperator::LogicalAnd)) : within;
  }
  if (!*fill) {
    fill = materializeZeroFragment(builder, reshape.getLoc(), result);
    if (failed(fill))
      return failure();
  }
  auto replacement = builder.create<LoadOp>(
      reshape.getLoc(), result, load.getResource(), coordinates, active, *fill,
      load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  reshape.getResult().replaceAllUsesWith(replacement.getResult());
  reshape.erase();
  if (load.getResult().use_empty())
    load.erase();
  return true;
}

FailureOr<bool> composeReshapedStore(StoreOp store) {
  auto isPositionalRebinding = [](Operation *operation) {
      if (!isa_and_nonnull<BroadcastOp, ReshapeOp>(operation))
        return false;
      auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
      auto result = dyn_cast<FragmentType>(operation->getResult(0).getType());
      if (!source || !result || source.getShape() != result.getShape() ||
          source.getElementType() != result.getElementType() ||
          source.getOwner() != result.getOwner() || source.getValidity() != result.getValidity())
        return false;
      if (auto reshape = dyn_cast<ReshapeOp>(operation))
        return llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getSourceAxes().size() == 1 &&
                 group.getResultAxes().size() == 1 &&
                 group.getSourceAxes()[0] == group.getResultAxes()[0];
        });
      BroadcastProjection relation = queryAxisProjection(source, result);
      if (!relation.isExact())
        return false;
      for (auto [axis, inputAxis] : llvm::enumerate(relation.targetToSource)) {
        auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
        bool unit = extent.getKind() ==
                        static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                    extent.getValue() == 1;
        if (inputAxis ? *inputAxis != axis : !unit)
          return false;
      }
      return true;
  };
  auto stripPositionalRebindings = [&](Value value) {
    while (isPositionalRebinding(value.getDefiningOp()))
      value = value.getDefiningOp()->getOperand(0);
    return value;
  };
  auto output = dyn_cast<FragmentType>(store.getValue().getType());
  if (!output || output.getShape().empty())
    return false;
  auto kernel = store->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  unsigned outputRank = output.getShape().size();
  SmallVector<MakeRangeOp> outputRanges(outputRank);
  SmallVector<unsigned> coordinateSlots(outputRank);
  for (auto [slot, coordinate] : llvm::enumerate(store.getCoordinates())) {
    if (!isa<FragmentType>(coordinate.getType()))
      continue;
    auto range = coordinate.getDefiningOp<MakeRangeOp>();
    FailureOr<int64_t> dimension = range ? queryRangeDimension(range)
                                         : FailureOr<int64_t>(failure());
    if (!range || failed(dimension) || !isZero(range.getStart()) ||
        !isZero(range.getLogicalStart()) || !isUnitStepRange(range))
      return false;
    auto realization = analysis.axisRealization(range.getResult(), 0);
    if (!realization.constructionScalarSeed &&
        !samePhysicalScalarExpression(range.getExtent(), range.getLogicalStop()))
      return false;
    auto projection = queryFragmentDimension(output, *dimension);
    if (!projection.isExact() || outputRanges[projection.fragmentAxis])
      return false;
    outputRanges[projection.fragmentAxis] = range;
    coordinateSlots[projection.fragmentAxis] = slot;
  }
  if (llvm::any_of(outputRanges, [](MakeRangeOp range) { return !range; }))
    return false;

  SmallVector<unsigned> outerAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outerAxes.push_back(axis);
  auto isPointwise = [](Operation *operation) {
    return operation && isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                            BitcastOp, SplatOp, BroadcastOp>(operation);
  };
  ReshapeOp reshape;
  llvm::SmallPtrSet<Operation *, 8> transposes;
  std::function<bool(Value, SmallVector<unsigned>)> findReshape =
      [&](Value value, SmallVector<unsigned> axes) {
        value = stripPositionalRebindings(value);
        if (auto transpose = value.getDefiningOp<TransposeOp>()) {
          if (transpose.getPermutation().size() != axes.size())
            return false;
          auto source = cast<FragmentType>(transpose.getValue().getType());
          auto result = cast<FragmentType>(transpose.getResult().getType());
          SmallVector<unsigned> inputAxes(axes.size());
          for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
            auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
            auto resultMap = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
            if (!(sourceAxisIdentity(sourceMap) == sourceAxisIdentity(resultMap)) ||
                sourceMap.getDimensionId() != resultMap.getDimensionId())
              return false;
            inputAxes[sourceAxis] = axes[axis];
          }
          if (!findReshape(transpose.getValue(), std::move(inputAxes)))
            return false;
          transposes.insert(transpose);
          return true;
        }
        if (auto candidate = value.getDefiningOp<ReshapeOp>()) {
          reshape = candidate;
          outerAxes = std::move(axes);
          return true;
        }
        Operation *producer = value.getDefiningOp();
        return isPointwise(producer) && !isa<BroadcastOp>(producer) &&
               llvm::any_of(producer->getOperands(), [&](Value operand) {
                 return findReshape(operand, axes);
               });
      };
  if (!findReshape(store.getValue(), outerAxes))
    return false;
  SmallVector<LoadOp> companions;
  SmallVector<ReshapeOp> sourceReshapes{reshape};
  llvm::DenseMap<Value, unsigned> companionAxes;
  using OutputAxes = SmallVector<std::optional<unsigned>>;
  llvm::DenseMap<Value, OutputAxes> checked;
  auto isSourceFrame = [&](ArrayRef<std::optional<unsigned>> axes) {
    return axes.size() == outerAxes.size() &&
           llvm::all_of(llvm::zip(axes, outerAxes), [](const auto &pair) {
             return std::get<0>(pair) == std::get<1>(pair);
           });
  };
  std::function<bool(Value, OutputAxes)> canReplayEpilogue =
      [&](Value value, OutputAxes axes) {
    auto result = dyn_cast<FragmentType>(value.getType());
    if (value == reshape.getResult())
      return isSourceFrame(axes);
    if (!result)
      return true;
    if (axes.size() != result.getShape().size())
      return false;
    auto [entry, inserted] = checked.try_emplace(value, axes);
    if (!inserted) {
      bool changed = false;
      for (auto [known, current] : llvm::zip(entry->second, axes)) {
        if (known && current && known != current)
          return false;
        if (!known && current) {
          known = current;
          changed = true;
        }
      }
      if (!changed)
        return true;
      axes = entry->second;
    }
    Operation *producer = value.getDefiningOp();
    if (auto transpose = dyn_cast_or_null<TransposeOp>(producer)) {
      if (!transposes.contains(producer))
        return false;
      OutputAxes inputAxes(axes.size());
      for (auto [axis, inputAxis] : llvm::enumerate(transpose.getPermutation()))
        inputAxes[inputAxis] = axes[axis];
      return canReplayEpilogue(transpose.getValue(), std::move(inputAxes));
    }
    if (auto load = dyn_cast_or_null<LoadOp>(producer)) {
      if (!canReplayReadAt(load, store))
        return false;
      for (auto [axis, outputAxis] : llvm::enumerate(axes)) {
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
          return false;
        if (ranges.roots.empty())
          continue;
        if (!outputAxis || *outputAxis >= outputRanges.size())
          return false;
        for (MakeRangeOp range : ranges.roots) {
          if (!analysis.lockstepRanges({range, outputRanges[*outputAxis]}).isExact())
            return false;
          auto [found, inserted] = companionAxes.try_emplace(range.getResult(), *outputAxis);
          if (!inserted && found->second != *outputAxis)
            return false;
        }
      }
      companions.push_back(load);
      return true;
    }
    if (auto companion = dyn_cast_or_null<ReshapeOp>(producer)) {
      auto source = cast<FragmentType>(companion.getValue().getType());
      auto primarySource = cast<FragmentType>(reshape.getValue().getType());
      auto primaryResult = cast<FragmentType>(reshape.getResult().getType());
      if (companion.getReassociation() == reshape.getReassociation() &&
          source.getShape() == primarySource.getShape() &&
          source.getAxisMaps() == primarySource.getAxisMaps() &&
          source.getOwner() == primarySource.getOwner() &&
          source.getValidity() == primarySource.getValidity() &&
          result.getShape() == primaryResult.getShape() &&
          result.getAxisMaps() == primaryResult.getAxisMaps() &&
          result.getOwner() == primaryResult.getOwner() &&
          result.getValidity() == primaryResult.getValidity()) {
        if (!isSourceFrame(axes))
          return false;
        if (!llvm::is_contained(sourceReshapes, companion))
          sourceReshapes.push_back(companion);
        return true;
      }
      auto unitAxes = [](FragmentType type, ArrayRef<int64_t> axes, unsigned prefix) {
        return llvm::all_of(axes, [&](int64_t axis) {
          auto extent = cast<PhysicalExprAttr>(type.getShape()[prefix + axis]);
          return extent.getKind() ==
                     static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                 extent.getValue() == 1;
        });
      };
      unsigned sourceRank = 0, resultRank = 0;
      for (Attribute attribute : companion.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        sourceRank += group.getSourceAxes().size();
        resultRank += group.getResultAxes().size();
      }
      unsigned sourcePrefix = source.getShape().size() - sourceRank;
      unsigned resultPrefix = result.getShape().size() - resultRank;
      if (sourcePrefix != resultPrefix)
        return false;
      OutputAxes inputAxes(source.getShape().size());
      for (unsigned axis = 0; axis < sourcePrefix; ++axis)
        inputAxes[axis] = axes[axis];
      for (Attribute attribute : companion.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        if (group.getSourceAxes().empty()) {
          if (!unitAxes(result, group.getResultAxes().asArrayRef(), resultPrefix))
            return false;
        } else if (group.getResultAxes().empty()) {
          if (!unitAxes(source, group.getSourceAxes().asArrayRef(), sourcePrefix))
            return false;
        } else if (group.getSourceAxes().size() != 1 ||
                   group.getResultAxes().size() != 1) {
          return false;
        } else {
          inputAxes[sourcePrefix + group.getSourceAxes()[0]] =
              axes[resultPrefix + group.getResultAxes()[0]];
        }
      }
      return canReplayEpilogue(companion.getValue(), std::move(inputAxes));
    }
    if (!isPointwise(producer))
      return false;
    return llvm::all_of(producer->getOperands(), [&](Value operand) {
      auto input = dyn_cast<FragmentType>(operand.getType());
      if (!input)
        return true;
      auto projection = queryBroadcastProjection(input, result);
      if (!projection.isExact())
        return false;
      OutputAxes inputAxes(input.getShape().size());
      for (auto [axis, inputAxis] : llvm::enumerate(projection.targetToSource)) {
        if (!inputAxis || !axes[axis])
          continue;
        if (inputAxes[*inputAxis] && inputAxes[*inputAxis] != axes[axis])
          return false;
        inputAxes[*inputAxis] = axes[axis];
      }
      return canReplayEpilogue(operand, std::move(inputAxes));
    });
  };
  OutputAxes outputAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outputAxes.push_back(axis);
  if (!canReplayEpilogue(store.getValue(), std::move(outputAxes)))
    return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  if (!input)
    return false;
  unsigned logicalSourceRank = 0;
  unsigned logicalResultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      logicalSourceRank = std::max(logicalSourceRank, static_cast<unsigned>(axis + 1));
    for (int64_t axis : group.getResultAxes().asArrayRef())
      logicalResultRank = std::max(logicalResultRank, static_cast<unsigned>(axis + 1));
  }
  if (logicalSourceRank != input.getShape().size() || logicalResultRank != outputRank)
    return false;
  SmallVector<MakeRangeOp> inputRanges;
  SmallVector<Attribute> inputExtents;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = analysis.axisRanges(reshape.getValue(), axis);
    FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
    if (failed(range) && ranges.blockers.empty()) {
      auto traversal = analysis.lockstepRanges(ranges.roots);
      if (traversal.isExact())
        range = traversal.authority;
    }
    if (failed(range) || !isZero((*range).getStart()) ||
        !isZero((*range).getLogicalStart()) || !isUnitStepRange(*range))
      return false;
    auto realization = analysis.axisRealization(reshape.getValue(), axis);
    if (!realization.constructionScalarSeed &&
        !samePhysicalScalarExpression((*range).getExtent(), (*range).getLogicalStop()))
      return false;
    PhysicalExprAttr extent = queryLaunchExpression((*range).getLogicalStop());
    if (!extent)
      return false;
    inputRanges.push_back(*range);
    inputExtents.push_back(extent);
  }
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty() || group.getResultAxes().empty())
      return false;
    SmallVector<Attribute> sourceExtents;
    for (int64_t axis : group.getSourceAxes().asArrayRef()) {
      if (axis < 0 || axis >= static_cast<int64_t>(inputRanges.size()))
        return false;
      sourceExtents.push_back(inputExtents[axis]);
    }
    SmallVector<Attribute> resultExtents;
    for (int64_t axis : group.getResultAxes().asArrayRef()) {
      if (axis < 0 || axis >= static_cast<int64_t>(outerAxes.size()))
        return false;
      PhysicalExprAttr extent = queryLaunchExpression(
          outputRanges[outerAxes[axis]].getLogicalStop());
      if (!extent)
        return false;
      resultExtents.push_back(extent);
    }
    // Check logical range extents, never provisional physical singletons.
    // The existing reassociation supplies the row-major axis order.
    if (failed(inferReshapeReassociation(
            store.getContext(), sourceExtents, resultExtents)))
      return false;
  }

  OpBuilder builder(store);
  SmallVector<Attribute> inputMappings;
  for (auto [axis, range] : llvm::enumerate(inputRanges)) {
    auto mapping = cast<AxisMapAttr>(range.getResult().getType().getAxisMaps()[0]);
    inputMappings.push_back(AxisMapAttr::get(
        store.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), axis, mapping.getDerived()));
  }
  input = FragmentType::get(
      store.getContext(), input.getElementType(), input.getShape(),
      builder.getArrayAttr(inputMappings), input.getValidity(), input.getOwner());
  auto indexType = FragmentType::get(
      input.getContext(), builder.getIndexType(), input.getShape(),
      input.getAxisMaps(), input.getValidity(), input.getOwner());
  SmallVector<Value> coordinates(store.getCoordinates());
  IRMapping mapping;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    Value ordinal;
    for (int64_t sourceAxis : group.getSourceAxes().asArrayRef()) {
      auto rangeType = inputRanges[sourceAxis].getResult().getType();
      SmallVector<Attribute> axisShape(input.getShape().size(),
          PhysicalExprAttr::get(store.getContext(),
              static_cast<uint32_t>(PhysicalExprKind::Constant), 1,
              builder.getStringAttr(""), builder.getArrayAttr({})));
      axisShape[sourceAxis] = rangeType.getShape()[0];
      auto axisType = FragmentType::get(
          store.getContext(), builder.getIndexType(), builder.getArrayAttr(axisShape),
          input.getAxisMaps(), input.getValidity(), input.getOwner());
      auto relation = inferReshapeReassociation(rangeType, axisType);
      if (failed(relation))
        return store.emitOpError("flattened store has no source-coordinate projection")
               << "; source_axis=" << sourceAxis
               << "; source=" << inputRanges[sourceAxis].getResult().getType()
               << "; target=" << indexType;
      Value projected = builder.create<ReshapeOp>(
          store.getLoc(), axisType, inputRanges[sourceAxis].getResult(), *relation);
      if (axisType != indexType)
        projected = builder.create<BroadcastOp>(store.getLoc(), indexType, projected);
      if (ordinal) {
        Value extent = builder.create<SplatOp>(
            store.getLoc(), indexType, inputRanges[sourceAxis].getLogicalStop());
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::Multiply);
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, projected, BinaryOperator::Add);
      } else {
        ordinal = projected;
      }
    }
    auto axes = group.getResultAxes().asArrayRef();
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned outerAxis = outerAxes[axes[position]];
      MakeRangeOp range = outputRanges[outerAxis];
      Value coordinate = ordinal;
      if (position != 0) {
        Value one = builder.create<arith::ConstantIndexOp>(store.getLoc(), 1);
        // Empty logical domains have no active store.  A positive divisor
        // keeps their inactive physical lanes well-defined as well.
        Value divisor = builder.create<BinaryOp>(
            store.getLoc(), builder.getIndexType(), range.getLogicalStop(), one,
            BinaryOperator::Maximum);
        Value extent = builder.create<SplatOp>(store.getLoc(), indexType, divisor);
        coordinate = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::FloorDivide);
      }
      coordinates[coordinateSlots[outerAxis]] = coordinate;
      mapping.map(range.getResult(), coordinate);
      for (auto [root, axis] : companionAxes)
        if (axis == outerAxis)
          mapping.map(root, coordinate);
      kernel.walk([&](MakeRangeOp occurrence) {
        FailureOr<int64_t> occurrenceDimension = queryRangeDimension(occurrence);
        FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
        if (succeeded(occurrenceDimension) && succeeded(rangeDimension) &&
            *occurrenceDimension == *rangeDimension &&
            analysis.lockstepRanges({occurrence, range}).isExact())
          mapping.map(occurrence.getResult(), coordinate);
      });
    }
  }
  FailureOr<Value> valid = replayFragmentValue(
      builder, store.getValid(), input, mapping, analysis);
  if (failed(valid))
    return store.emitOpError("flattened store could not preserve its access validity");
  auto predicate = FragmentType::get(
      input.getContext(), builder.getI1Type(), input.getShape(), input.getAxisMaps(),
      input.getValidity(), input.getOwner());
  Value zero = builder.create<arith::ConstantIndexOp>(store.getLoc(), 0);
  Value lower = builder.create<SplatOp>(store.getLoc(), indexType, zero);
  Value active = *valid;
  for (unsigned axis = 0; axis < outputRank; ++axis) {
    Value coordinate = coordinates[coordinateSlots[axis]];
    Value end = builder.create<SplatOp>(
        store.getLoc(), indexType, outputRanges[axis].getLogicalStop());
    Value nonNegative = builder.create<CompareOp>(
        store.getLoc(), predicate, coordinate, lower, ComparePredicate::Ge);
    Value belowEnd = builder.create<CompareOp>(
        store.getLoc(), predicate, coordinate, end, ComparePredicate::Lt);
    Value within = builder.create<BinaryOp>(
        store.getLoc(), predicate, nonNegative, belowEnd, BinaryOperator::LogicalAnd);
    active = active ? Value(builder.create<BinaryOp>(
        store.getLoc(), predicate, active, within, BinaryOperator::LogicalAnd)) : within;
  }
  Value source = reshape.getValue();
  if (source.getType() != input)
    source = builder.create<BroadcastOp>(store.getLoc(), input, source);
  mapping.map(reshape.getResult(), source);
  for (ReshapeOp companion : llvm::drop_begin(sourceReshapes)) {
    auto original = cast<FragmentType>(companion.getValue().getType());
    auto target = FragmentType::get(
        input.getContext(), original.getElementType(), input.getShape(),
        input.getAxisMaps(), input.getValidity(), input.getOwner());
    FailureOr<Value> value = projectPhysicalValueToSchema(
        builder, store.getLoc(), companion.getValue(), target);
    if (failed(value))
      return store.emitOpError("aligned reshapes have no common input relation");
    mapping.map(companion.getResult(), *value);
  }
  for (LoadOp companion : companions) {
    auto companionType = FragmentType::get(
        input.getContext(), cast<FragmentType>(companion.getResult().getType()).getElementType(),
        input.getShape(), input.getAxisMaps(), input.getValidity(), input.getOwner());
    SmallVector<Value> companionCoordinates;
    for (Value coordinate : companion.getCoordinates()) {
      if (!isa<FragmentType>(coordinate.getType())) {
        companionCoordinates.push_back(coordinate);
        continue;
      }
      FailureOr<Value> projected = replayFragmentValue(
          builder, coordinate, indexType, mapping, analysis);
      if (failed(projected))
        return store.emitOpError(
            "reshaped store companion has no exact coordinate mapping")
               << "; coordinate=" << coordinate;
      companionCoordinates.push_back(*projected);
    }
    FailureOr<Value> companionValid = replayFragmentValue(
        builder, companion.getValid(), input, mapping, analysis);
    FailureOr<Value> companionFill = replayFragmentValue(
        builder, companion.getFill(), input, mapping, analysis);
    if (failed(companionValid) || failed(companionFill))
      return store.emitOpError(
          "reshaped store companion could not preserve validity and fill");
    companionValid = combinePredicates(
        builder, store.getLoc(), input, *companionValid, active);
    if (!*companionFill)
      companionFill = materializeZeroFragment(builder, store.getLoc(), companionType);
    if (failed(companionValid) || failed(companionFill))
      return failure();
    auto loaded = builder.create<LoadOp>(
        companion.getLoc(), companionType, companion.getResource(), companionCoordinates,
        *companionValid, *companionFill, companion.getSourceAxes());
    if (Attribute origin = companion->getAttr(originAttr))
      loaded->setAttr(originAttr, origin);
    mapping.map(companion.getResult(), loaded.getResult());
  }
  std::function<FailureOr<Value>(Value)> replayEpilogue = [&](Value value) -> FailureOr<Value> {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return value;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> mapped = replayEpilogue(operand);
      if (failed(mapped))
        return failure();
      mapping.map(operand, *mapped);
    }
    auto target = FragmentType::get(
        input.getContext(), fragment.getElementType(), input.getShape(),
        input.getAxisMaps(), input.getValidity(), input.getOwner());
    if (isa<BroadcastOp, TransposeOp, ReshapeOp>(producer)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, producer->getLoc(), mapping.lookup(producer->getOperand(0)), target);
      if (succeeded(projected))
        mapping.map(value, *projected);
      return projected;
    }
    Operation *clone = builder.clone(*producer, mapping);
    clone->getResult(0).setType(target);
    mapping.map(value, clone->getResult(0));
    return clone->getResult(0);
  };
  FailureOr<Value> payload = replayEpilogue(store.getValue());
  if (failed(payload))
    return store.emitOpError("flattened store could not preserve its pointwise epilogue");
  auto replacement = builder.create<StoreOp>(
      store.getLoc(), store.getResource(), coordinates, *payload,
      active, store.getSourceAxes());
  if (Attribute origin = store->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  store.erase();
  return true;
}

bool sameImmutableLoad(LoadOp available, LoadOp current) {
  auto view = dyn_cast<ViewType>(current.getResource().getType());
  if (!view || view.getAccess() != 0 ||
      available.getResource() != current.getResource() ||
      available.getResult().getType() != current.getResult().getType() ||
      available.getValid() != current.getValid() ||
      available.getFill() != current.getFill() ||
      available.getSourceAxes() != current.getSourceAxes() ||
      available.getCoordinates().size() != current.getCoordinates().size())
    return false;
  return llvm::equal(available.getCoordinates(), current.getCoordinates());
}

bool deduplicateImmutableLoads(func::FuncOp kernel) {
  SmallVector<LoadOp> loads;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  bool changed = false;
  for (LoadOp current : loads) {
    if (!current->getBlock())
      continue;
    Block *block = current->getBlock();
    auto cursor = current->getIterator();
    while (cursor != block->begin()) {
      --cursor;
      Operation *candidate = &*cursor;
      if (candidate->getNumRegions() != 0)
        break;
      if (auto available = dyn_cast<LoadOp>(candidate)) {
        if (!sameImmutableLoad(available, current))
          continue;
        current.getResult().replaceAllUsesWith(available.getResult());
        current.erase();
        changed = true;
        break;
      }
      // Different ABI views may alias by default.  A write or another
      // side-effecting operation therefore ends the interval in which an
      // immutable-view load is known to retain its value.
      if (!isMemoryEffectFree(candidate))
        break;
    }
  }
  return changed;
}

void sinkImmutableLoadChains(func::FuncOp kernel) {
  llvm::SmallPtrSet<Operation *, 32> selected;
  SmallVector<Operation *> pending;
  kernel.walk([&](LoadOp load) {
    auto view = dyn_cast<ViewType>(load.getResource().getType());
    if (view && view.getAccess() == 0 && selected.insert(load).second)
      pending.push_back(load);
  });
  while (!pending.empty()) {
    Operation *producer = pending.pop_back_val();
    for (Operation *user : producer->getUsers()) {
      if (user->getBlock() != producer->getBlock() ||
          !isa<CastOp, BitcastOp, ReshapeOp, TransposeOp, BroadcastOp,
               SelectOp, BinaryOp, UnaryOp>(user) ||
          !isMemoryEffectFree(user) || !selected.insert(user).second)
        continue;
      pending.push_back(user);
    }
  }
  SmallVector<Operation *> ordered;
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (selected.contains(operation))
      ordered.push_back(operation);
  });
  for (Operation *operation : llvm::reverse(ordered)) {
    Operation *firstUse = nullptr;
    for (Operation *user : operation->getUsers()) {
      Operation *ancestor = operation->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor) {
        firstUse = nullptr;
        break;
      }
      if (!firstUse || ancestor->isBeforeInBlock(firstUse))
        firstUse = ancestor;
    }
    if (!firstUse || operation->getNextNode() == firstUse)
      continue;
    bool crossesEffect = false;
    for (Operation *next = operation->getNextNode(); next != firstUse;
         next = next->getNextNode()) {
      // ABI views can alias. Do not cross writes, atomics, synchronization,
      // or an unknown effect while shortening an immutable load's live range.
      if (!isMemoryEffectFree(next) && !isa<LoadOp, GatherOp>(next)) {
        crossesEffect = true;
        break;
      }
    }
    if (!crossesEffect)
      operation->moveBefore(firstUse);
  }
}

LogicalResult materializeIndexedFragments(func::FuncOp kernel) {
  SmallVector<GatherOp> gathers;
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  for (GatherOp gather : gathers) {
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      PhysicalProgramAnalysis analysis(kernel);
      if (!analysis.axisRealization(gather.getSource(), sourceAxis)
               .constructionScalarSeed)
        continue;
      auto source = cast<FragmentType>(gather.getSource().getType());
      auto extent = cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(coordinate);
      if (bound &&
          bound.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          bound.getValue() < extent.getValue())
        continue;
      if (failed(realizeFullCoverageDimension(kernel, gather.getSource(),
                                               sourceAxis)))
        return gather.emitOpError(
            "indexed tensor source has no complete physical extent");
    }
  }
  return success();
}

} // namespace

LogicalResult realizeAccessComposition(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  eraseDeadPhysicalValues(*physicalKernel);
  bool changed;
  do {
    changed = false;
    SmallVector<SelectOp> selects;
    physicalKernel->walk([&](SelectOp select) { selects.push_back(select); });
    for (SelectOp select : selects) {
      if (!select->getBlock())
        continue;
      FailureOr<bool> load = composeSelectLoad(select);
      if (failed(load))
        return failure();
      changed |= *load;
    }
    SmallVector<GatherOp> gathers;
    physicalKernel->walk([&](GatherOp gather) { gathers.push_back(gather); });
    for (GatherOp gather : gathers) {
      if (!gather->getBlock())
        continue;
      FailureOr<bool> reshaped = composeReshapedGather(gather);
      if (failed(reshaped))
        return failure();
      if (*reshaped) {
        changed = true;
        continue;
      }
      FailureOr<bool> pointwise = composePointwiseGather(gather);
      if (failed(pointwise))
        return failure();
      if (*pointwise) {
        changed = true;
        continue;
      }
      FailureOr<bool> identity = composeIdentityFragmentGather(gather);
      if (failed(identity))
        return failure();
      if (*identity) {
        changed = true;
        continue;
      }
      FailureOr<bool> projection = projectFragmentGather(gather);
      if (failed(projection))
        return failure();
      if (*projection) {
        changed = true;
        continue;
      }
      FailureOr<bool> load = composeLoadGather(gather);
      if (failed(load))
        return failure();
      changed |= *load;
    }
    SmallVector<ReshapeOp> reshapes;
    physicalKernel->walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
    for (ReshapeOp reshape : reshapes) {
      if (composeReshapedPointwise(reshape)) {
        changed = true;
        continue;
      }
      FailureOr<bool> composed = composeReshapedLoad(reshape);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    SmallVector<StoreOp> stores;
    physicalKernel->walk([&](StoreOp store) { stores.push_back(store); });
    for (StoreOp store : stores) {
      FailureOr<bool> composed = composeReshapedStore(store);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    changed |= deduplicateImmutableLoads(*physicalKernel);
    eraseDeadPhysicalValues(*physicalKernel);
  } while (changed);
  if (failed(materializeIndexedFragments(*physicalKernel)))
    return failure();
  sinkImmutableLoadChains(*physicalKernel);
  return success();
}

LogicalResult simplifyMaskedAccessCoordinates(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  kernel->walk([&](LoadOp load) {
    Value valid = load.getValid();
    if (!valid)
      return;
    llvm::SmallDenseSet<Value, 16> conjuncts;
    SmallVector<Value> pending{valid};
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (value.getType() != valid.getType() ||
          !conjuncts.insert(value).second)
        continue;
      auto binary = value.getDefiningOp<BinaryOp>();
      if (!binary ||
          (binary.getOperatorKind() != BinaryOperator::LogicalAnd &&
           binary.getOperatorKind() != BinaryOperator::BitwiseAnd))
        continue;
      pending.push_back(binary.getLhs());
      pending.push_back(binary.getRhs());
    }
    SmallVector<Value> coordinates(load.getCoordinates());
    bool changed = false;
    for (Value &coordinate : coordinates) {
      auto select = coordinate.getDefiningOp<SelectOp>();
      if (!select || !conjuncts.contains(select.getCondition()))
        continue;
      // Exact SSA predicates with the same schema describe the same lanes.
      // Change only this masked read; other uses retain their selected value.
      coordinate = select.getTrueValue();
      changed = true;
    }
    if (changed)
      load.getCoordinatesMutable().assign(coordinates);
  });
  eraseDeadPhysicalValues(*kernel);
  return success();
}

} // namespace intent::gpu

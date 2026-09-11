#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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
  PhysicalReplayFact replay = analysis.replayability(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false);
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<MakeRangeOp>(producer) || !replay.isReplayable())
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayFragmentValue(builder, operand, target, mapping, analysis);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
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

FailureOr<bool> composeSelectLoad(SelectOp select) {
  auto resultType = dyn_cast<FragmentType>(select.getResult().getType());
  auto load = select.getTrueValue().getDefiningOp<LoadOp>();
  if (!resultType || !load || !load.getResult().hasOneUse())
    return false;

  Value fill = select.getFalseValue();
  if (load.getValid()) {
    if (!load.getFill() || load.getFill() != fill)
      return false;
  } else if (load.getFill()) {
    return false;
  }

  OpBuilder builder(select);
  FailureOr<Value> valid = combinePredicates(
      builder, select.getLoc(), resultType, load.getValid(),
      select.getCondition());
  if (failed(valid)) {
    select.emitOpError(
        "masked load predicate cannot adopt the loaded value relation");
    return failure();
  }
  if (fill.getType() != resultType) {
    FailureOr<Value> projected = materializeBroadcastToFragment(
        builder, select.getLoc(), fill, resultType);
    if (failed(projected)) {
      select.emitOpError(
          "masked load fill cannot adopt the loaded value relation");
      return failure();
    }
    fill = *projected;
  }
  auto replacement = builder.create<LoadOp>(
      select.getLoc(), resultType, load.getResource(), load.getCoordinates(),
      *valid, fill, load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  select.getResult().replaceAllUsesWith(replacement.getResult());
  select.erase();
  load.erase();
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
  SmallVector<Value> coordinates(sourceLoad.getCoordinates());
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
        sourceLoad.getCoordinates(), sourceAxisIdentity(*mapping));
    if (!target.isExact() || target.dimensionId != mapping->getDimensionId()) {
      gather.emitOpError(
          "loaded source coordinate cannot be composed with gather indexing");
      return failure();
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
    Value original = sourceLoad.getCoordinates()[target.fragmentAxis];
    replay.map(original, coordinate);
    PhysicalRangeFact roots = analysis.sourceRanges(original);
    if (roots.isUnique() && !replay.lookupOrNull(roots.roots.front().getResult()))
      replay.map(roots.roots.front().getResult(), coordinate);
    coordinates[target.fragmentAxis] = coordinate;
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

FailureOr<bool> composeReshapedStore(StoreOp store) {
  auto output = dyn_cast<FragmentType>(store.getValue().getType());
  if (!output || output.getShape().empty())
    return false;
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
    auto projection = queryFragmentDimension(output, *dimension);
    if (!projection.isExact() || outputRanges[projection.fragmentAxis])
      return false;
    outputRanges[projection.fragmentAxis] = range;
    coordinateSlots[projection.fragmentAxis] = slot;
  }
  if (llvm::any_of(outputRanges, [](MakeRangeOp range) { return !range; }))
    return false;

  Value value = store.getValue();
  SmallVector<unsigned> outerAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outerAxes.push_back(axis);
  while (auto transpose = value.getDefiningOp<TransposeOp>()) {
    if (transpose.getPermutation().size() != outerAxes.size())
      return false;
    SmallVector<unsigned> inputAxes(outerAxes.size());
    for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation()))
      inputAxes[sourceAxis] = outerAxes[axis];
    outerAxes = std::move(inputAxes);
    value = transpose.getValue();
  }
  auto reshape = value.getDefiningOp<ReshapeOp>();
  if (!reshape)
    return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  if (!input || input.getShape().size() >= outputRank)
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
  auto kernel = store->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<MakeRangeOp> inputRanges;
  SmallVector<Attribute> inputExtents;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = analysis.axisRanges(reshape.getValue(), axis);
    FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
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
    if (group.getSourceAxes().size() != 1 || group.getResultAxes().empty())
      return false;
    unsigned sourceAxis = group.getSourceAxes()[0];
    if (sourceAxis >= inputRanges.size())
      return false;
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
            store.getContext(), ArrayRef<Attribute>{inputExtents[sourceAxis]},
            resultExtents)))
      return false;
  }

  OpBuilder builder(store);
  auto indexType = FragmentType::get(
      input.getContext(), builder.getIndexType(), input.getShape(),
      input.getAxisMaps(), input.getValidity(), input.getOwner());
  SmallVector<Value> coordinates(store.getCoordinates());
  IRMapping mapping;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    unsigned sourceAxis = group.getSourceAxes()[0];
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, store.getLoc(), inputRanges[sourceAxis].getResult(), indexType);
    if (failed(projected))
      return store.emitOpError("flattened store has no source-coordinate projection");
    Value ordinal = *projected;
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
      kernel.walk([&](MakeRangeOp occurrence) {
        FailureOr<int64_t> occurrenceDimension = queryRangeDimension(occurrence);
        FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
        if (sameLogicalRange(occurrence, range) &&
            succeeded(occurrenceDimension) && succeeded(rangeDimension) &&
            *occurrenceDimension == *rangeDimension)
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
  auto replacement = builder.create<StoreOp>(
      store.getLoc(), store.getResource(), coordinates, reshape.getValue(),
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

} // namespace

LogicalResult realizeAccessComposition(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
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
  sinkImmutableLoadChains(*physicalKernel);
  return success();
}

} // namespace intent::gpu

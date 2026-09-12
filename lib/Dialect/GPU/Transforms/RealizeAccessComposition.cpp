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
      if (auto splat = value.getDefiningOp<SplatOp>())
        value = splat.getValue();
      else if (auto broadcast = value.getDefiningOp<BroadcastOp>())
        value = broadcast.getValue();
      else
        break;
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
  auto left = lhs.getDefiningOp<arith::ConstantOp>();
  auto right = rhs.getDefiningOp<arith::ConstantOp>();
  return left && right && left.getValue() == right.getValue();
}

FailureOr<bool> composeSelectLoad(SelectOp select) {
  auto resultType = dyn_cast<FragmentType>(select.getResult().getType());
  auto load = select.getTrueValue().getDefiningOp<LoadOp>();
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

  if (resultType) {
    for (auto [slot, original] : llvm::enumerate(sourceLoad.getCoordinates())) {
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

FailureOr<bool> composeReshapedLoad(ReshapeOp reshape) {
  Value sourceValue = reshape.getValue();
  Value loaded = sourceValue;
  while (auto transpose = loaded.getDefiningOp<TransposeOp>()) {
    auto source = cast<FragmentType>(transpose.getValue().getType());
    auto result = cast<FragmentType>(transpose.getResult().getType());
    for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
      auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
      auto transposed = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
      if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
          original.getDimensionId() != transposed.getDimensionId())
        return false;
    }
    loaded = transpose.getValue();
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
  if (!load || !result || !isa<ViewType>(load.getResource().getType()) ||
      !canReplayReadAt(load, reshape))
    return false;
  auto source = cast<FragmentType>(sourceValue.getType());
  if (source.getShape().size() <= result.getShape().size())
    return false;
  unsigned sourceRank = 0;
  unsigned resultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty() || group.getResultAxes().size() != 1)
      return false;
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      sourceRank = std::max(sourceRank, static_cast<unsigned>(axis + 1));
    resultRank = std::max(resultRank,
                         static_cast<unsigned>(group.getResultAxes()[0] + 1));
  }
  if (sourceRank != source.getShape().size() || resultRank != result.getShape().size())
    return false;
  auto kernel = reshape->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<bool> preservedSource(sourceRank, false);
  SmallVector<bool> preservedResult(resultRank, false);
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().size() != 1)
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
  OpBuilder builder(reshape);
  SmallVector<Value> resultStops(resultRank);
  // Each group merges complete zero-based source-value axes. Its logical
  // extent is their product, including when no author domain names that axis.
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    unsigned resultAxis = group.getResultAxes()[0];
    if (preservedResult[resultAxis])
      continue;
    auto axes = group.getSourceAxes().asArrayRef();
    auto extent = cast<PhysicalExprAttr>(sourceExtents[axes.front()]);
    for (int64_t axis : axes.drop_front())
      extent = PhysicalExprAttr::get(reshape.getContext(),
          static_cast<uint32_t>(PhysicalExprKind::Multiply), 0,
          builder.getStringAttr(""), builder.getArrayAttr({extent, sourceExtents[axis]}));
    resultStops[resultAxis] = builder.create<PhysicalExprOp>(
        reshape.getLoc(), builder.getIndexType(), extent);
  }

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
    Value extent = builder.create<PhysicalExprOp>(
        reshape.getLoc(), builder.getIndexType(),
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
    if (preservedResult[group.getResultAxes()[0]]) {
      // Unmerged axes retain their current tile and coordinates, including
      // program-local batch coordinates and already blocked free dimensions.
      for (MakeRangeOp root : sourceRoots[group.getSourceAxes()[0]]) {
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, reshape.getLoc(), root.getResult(), indexType);
        if (failed(projected))
          return reshape.emitOpError("collapsed load lost an unmerged axis projection");
        mapping.map(root.getResult(), *projected);
      }
      continue;
    }
    Value ordinal = flatCoordinates[group.getResultAxes()[0]];
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
  auto stripIdentityBroadcast = [](Value value) {
    while (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
      auto result = dyn_cast<FragmentType>(value.getType());
      if (!source || !result || source.getShape() != result.getShape() ||
          source.getElementType() != result.getElementType() ||
          source.getOwner() != result.getOwner() || source.getValidity() != result.getValidity())
        break;
      bool sameDimensions = true;
      for (auto [lhs, rhs] : llvm::zip(source.getAxisMaps(), result.getAxisMaps()))
        sameDimensions &= cast<AxisMapAttr>(lhs).getDimensionId() ==
                          cast<AxisMapAttr>(rhs).getDimensionId();
      if (!sameDimensions)
        break;
      value = broadcast.getValue();
    }
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

  Value value = stripIdentityBroadcast(store.getValue());
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
    value = stripIdentityBroadcast(transpose.getValue());
  }
  auto reshape = value.getDefiningOp<ReshapeOp>();
  BinaryOp pointwise;
  LoadOp companion;
  bool reshapedLhs = false;
  if (!reshape) {
    pointwise = value.getDefiningOp<BinaryOp>();
    if (!pointwise)
      return false;
    Value companionValue;
    reshape = stripIdentityBroadcast(pointwise.getLhs()).getDefiningOp<ReshapeOp>();
    if (reshape) {
      reshapedLhs = true;
      companionValue = pointwise.getRhs();
    }
    else {
      reshape = stripIdentityBroadcast(pointwise.getRhs()).getDefiningOp<ReshapeOp>();
      companionValue = pointwise.getLhs();
    }
    if (!reshape)
      return false;
    auto companionType = dyn_cast<FragmentType>(companionValue.getType());
    auto reshaped = cast<FragmentType>(reshape.getResult().getType());
    if (!companionType || companionType.getShape() != reshaped.getShape() ||
        companionType.getElementType() != reshaped.getElementType())
      return false;
    for (auto [lhs, rhs] : llvm::zip(companionType.getAxisMaps(), reshaped.getAxisMaps()))
      if (cast<AxisMapAttr>(lhs).getDimensionId() != cast<AxisMapAttr>(rhs).getDimensionId())
        return false;
    while (auto broadcast = companionValue.getDefiningOp<BroadcastOp>()) {
      auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
      auto target = cast<FragmentType>(broadcast.getResult().getType());
      if (!source)
        return false;
      BroadcastProjection projection = queryBroadcastProjection(source, target);
      if (!projection.isExact())
        return false;
      for (auto [targetAxis, sourceAxis] : llvm::enumerate(projection.targetToSource)) {
        if (!sourceAxis)
          continue;
        auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[*sourceAxis]);
        auto targetMap = cast<AxisMapAttr>(target.getAxisMaps()[targetAxis]);
        if (sourceMap.getDimensionId() != targetMap.getDimensionId())
          return false;
      }
      companionValue = broadcast.getValue();
    }
    companion = companionValue.getDefiningOp<LoadOp>();
    if (!companion || !canReplayReadAt(companion, store))
      return false;
  }
  if (!reshape)
    return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  if (!input || input.getShape().size() > outputRank)
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
  Value payload = reshape.getValue();
  if (pointwise) {
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
      companionFill = materializeZeroFragment(builder, store.getLoc(), input);
    if (failed(companionValid) || failed(companionFill))
      return failure();
    auto loaded = builder.create<LoadOp>(
        companion.getLoc(), input, companion.getResource(), companionCoordinates,
        *companionValid, *companionFill, companion.getSourceAxes());
    if (Attribute origin = companion->getAttr(originAttr))
      loaded->setAttr(originAttr, origin);
    IRMapping operands;
    operands.map(reshapedLhs ? pointwise.getLhs() : pointwise.getRhs(), payload);
    operands.map(reshapedLhs ? pointwise.getRhs() : pointwise.getLhs(), loaded.getResult());
    Operation *combined = builder.clone(*pointwise, operands);
    combined->getResult(0).setType(input);
    payload = combined->getResult(0);
  }
  auto replacement = builder.create<StoreOp>(
      store.getLoc(), store.getResource(), coordinates, payload,
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
    SmallVector<ReshapeOp> reshapes;
    physicalKernel->walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
    for (ReshapeOp reshape : reshapes) {
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
  sinkImmutableLoadChains(*physicalKernel);
  return success();
}

} // namespace intent::gpu

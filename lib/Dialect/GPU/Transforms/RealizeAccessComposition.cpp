#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isTrue(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  while (auto splat = value.getDefiningOp<SplatOp>())
    value = splat.getValue();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getType().isInteger(1) && integer.getInt() != 0;
}

FailureOr<AxisMapAttr> fragmentAxisMap(FragmentType type, unsigned axis) {
  for (Attribute attribute : type.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == axis)
      return mapping;
  }
  return failure();
}

FailureOr<unsigned> coordinateForMap(ValueRange coordinates,
                                     AxisMapAttr expected) {
  std::optional<unsigned> result;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    auto fragment = dyn_cast<FragmentType>(coordinate.getType());
    if (!fragment)
      continue;
    bool matches = llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      auto mapping = cast<AxisMapAttr>(attribute);
      return mapping.getSourceId() == expected.getSourceId() &&
             mapping.getSourceAxis() == expected.getSourceAxis();
    });
    if (!matches)
      continue;
    if (result)
      return failure();
    result = index;
  }
  return result ? FailureOr<unsigned>(*result)
                : FailureOr<unsigned>(failure());
}

FailureOr<Value> replayFragmentValue(OpBuilder &builder, Value value,
                                     FragmentType target, IRMapping &mapping) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer ||
      !isPhysicalReplayNode(producer, PhysicalReplayScope::Coordinate,
                            /*allowAccesses=*/false))
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayFragmentValue(builder, operand, target, mapping);
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

LogicalResult composeLoadGather(GatherOp gather) {
  auto sourceType = dyn_cast<FragmentType>(gather.getSource().getType());
  auto sourceLoad = gather.getSource().getDefiningOp<LoadOp>();
  if (!sourceType || !sourceLoad ||
      !isa<ViewType>(sourceLoad.getResource().getType()))
    return success();
  if (gather.getCoordinates().size() != gather.getSourceAxes().size())
    return gather.emitOpError("gather coordinate/source-axis schema is incomplete");

  SmallVector<Value> coordinates(sourceLoad.getCoordinates());
  IRMapping replay;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(sourceType.getShape().size()))
      return gather.emitOpError("gather source axis is outside its loaded value");
    FailureOr<AxisMapAttr> mapping =
        fragmentAxisMap(sourceType, static_cast<unsigned>(sourceAxis));
    if (failed(mapping))
      return gather.emitOpError("gather source axis lost coordinate provenance");
    FailureOr<unsigned> target =
        coordinateForMap(sourceLoad.getCoordinates(), *mapping);
    if (failed(target))
      return gather.emitOpError(
          "loaded source coordinate cannot be composed with gather indexing");
    Value original = sourceLoad.getCoordinates()[*target];
    replay.map(original, coordinate);
    PhysicalRangeFact roots = analysis.sourceRanges(original);
    if (roots.isUnique() && !replay.lookupOrNull(roots.roots.front().getResult()))
      replay.map(roots.roots.front().getResult(), coordinate);
    coordinates[*target] = coordinate;
  }

  OpBuilder builder(gather);
  FailureOr<Value> sourceValid = replayFragmentValue(
      builder, sourceLoad.getValid(), cast<FragmentType>(gather.getResult().getType()),
      replay);
  FailureOr<Value> sourceFill = replayFragmentValue(
      builder, sourceLoad.getFill(), cast<FragmentType>(gather.getResult().getType()),
      replay);
  if (failed(sourceValid) || failed(sourceFill))
    return gather.emitOpError(
        "source access validity/fill cannot follow composed coordinates");
  FailureOr<Value> valid = combinePredicates(
      builder, gather.getLoc(), cast<FragmentType>(gather.getResult().getType()),
      *sourceValid, gather.getValid());
  if (failed(valid))
    return gather.emitOpError(
        "source and gather validity cannot share the composed result relation");
  Value fill = *sourceFill;
  if (sourceLoad.getValid() && gather.getValid()) {
    Value gatherFill = gather.getFill();
    if (!fill || !gatherFill)
      return gather.emitOpError(
          "composed conditional access requires both source and gather fill");
    auto result = cast<FragmentType>(gather.getResult().getType());
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
  return success();
}

LogicalResult composeIdentityFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result || gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size() ||
      (gather.getValid() && !isTrue(gather.getValid())))
    return success();
  SmallVector<bool> represented(result.getShape().size(), false);
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return success();
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    std::optional<unsigned> resultAxis;
    for (auto [axis, attribute] : llvm::enumerate(result.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getSourceId() == expected.getSourceId() &&
          mapping.getSourceAxis() == expected.getSourceAxis()) {
        if (resultAxis)
          return success();
        resultAxis = axis;
      }
    }
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    if (!resultAxis || !coordinateType ||
        source.getShape()[sourceAxis] != result.getShape()[*resultAxis] ||
        coordinateType.getShape().size() != 1 ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        failed(coordinateForMap(ValueRange{coordinate}, expected)))
      return success();
    represented[*resultAxis] = true;
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (represented[axis])
      continue;
    auto expression = dyn_cast<PhysicalExprAttr>(extent);
    if (!expression ||
        expression.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Constant) ||
        expression.getValue() != 1)
      return success();
  }
  OpBuilder builder(gather);
  auto replacement = builder.create<BroadcastOp>(
      gather.getLoc(), result, gather.getSource());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return success();
}

LogicalResult projectFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return success();

  SmallVector<Value> selectedCoordinates;
  SmallVector<int64_t> selectedAxes;
  bool projected = false;
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return success();
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    std::optional<unsigned> resultAxis;
    for (auto [axis, attribute] : llvm::enumerate(result.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getSourceId() == expected.getSourceId() &&
          mapping.getSourceAxis() == expected.getSourceAxis()) {
        if (resultAxis)
          return success();
        resultAxis = axis;
      }
    }
    if (!resultAxis) {
      selectedCoordinates.push_back(coordinate);
      selectedAxes.push_back(sourceAxis);
      continue;
    }
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    if (!coordinateType || coordinateType.getShape().size() != 1 ||
        source.getShape()[sourceAxis] != result.getShape()[*resultAxis] ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        failed(coordinateForMap(ValueRange{coordinate}, expected)))
      return success();
    projected = true;
  }
  if (!projected || selectedCoordinates.empty())
    return success();

  OpBuilder builder(gather);
  auto replacement = builder.create<GatherOp>(
      gather.getLoc(), result, gather.getSource(), selectedCoordinates,
      gather.getValid(), gather.getFill(), selectedAxes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return success();
}

} // namespace

LogicalResult realizeAccessComposition(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  SmallVector<GatherOp> gathers;
  physicalKernel->walk([&](GatherOp gather) { gathers.push_back(gather); });
  for (GatherOp gather : gathers)
    if (gather->getBlock()) {
      if (failed(composeIdentityFragmentGather(gather)))
        return failure();
      if (gather->getBlock() && failed(projectFragmentGather(gather)))
        return failure();
      if (gather->getBlock() && failed(composeLoadGather(gather)))
        return failure();
    }
  eraseDeadPhysicalValues(*physicalKernel);
  return success();
}

} // namespace intent::gpu

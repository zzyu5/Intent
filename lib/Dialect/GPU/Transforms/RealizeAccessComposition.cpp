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
      !isa<ViewType>(sourceLoad.getResource().getType()))
    return false;
  if (gather.getCoordinates().size() != gather.getSourceAxes().size()) {
    gather.emitOpError("gather coordinate/source-axis schema is incomplete");
    return failure();
  }

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
    Value original = sourceLoad.getCoordinates()[target.fragmentAxis];
    replay.map(original, coordinate);
    PhysicalRangeFact roots = analysis.sourceRanges(original);
    if (roots.isUnique() && !replay.lookupOrNull(roots.roots.front().getResult()))
      replay.map(roots.roots.front().getResult(), coordinate);
    coordinates[target.fragmentAxis] = coordinate;
  }

  OpBuilder builder(gather);
  auto resultType = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!resultType) {
    if (sourceLoad.getValid() || sourceLoad.getFill() || gather.getValid() ||
        gather.getFill())
      return false;
    auto replacement = builder.create<LoadOp>(
        gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
        coordinates, Value(), Value(), sourceLoad.getSourceAxes());
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
  if (!source || !result || gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size() ||
      (gather.getValid() && !isTrue(gather.getValid())))
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
    if (!resultAxis.isExact() || !coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType ||
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
  auto replacement = builder.create<BroadcastOp>(
      gather.getLoc(), result, gather.getSource());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
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
    changed |= deduplicateImmutableLoads(*physicalKernel);
    eraseDeadPhysicalValues(*physicalKernel);
  } while (changed);
  return success();
}

} // namespace intent::gpu

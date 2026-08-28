#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

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

LogicalResult composeLoadGather(GatherOp gather) {
  auto sourceType = dyn_cast<FragmentType>(gather.getSource().getType());
  auto sourceLoad = gather.getSource().getDefiningOp<LoadOp>();
  if (!sourceType || !sourceLoad ||
      !isa<ViewType>(sourceLoad.getResource().getType()))
    return success();
  if (sourceLoad.getValid() || sourceLoad.getFill())
    return gather.emitOpError(
        "cannot compose a source load with residual validity/fill");
  if (gather.getCoordinates().size() != gather.getSourceAxes().size())
    return gather.emitOpError("gather coordinate/source-axis schema is incomplete");

  SmallVector<Value> coordinates(sourceLoad.getCoordinates());
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
    coordinates[*target] = coordinate;
  }

  OpBuilder builder(gather);
  auto replacement = builder.create<LoadOp>(
      gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
      coordinates, gather.getValid(), gather.getFill(),
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

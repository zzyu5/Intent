#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

using namespace mlir;

namespace intent::gpu {
namespace {

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

} // namespace

LogicalResult realizeAccessComposition(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  SmallVector<GatherOp> gathers;
  physicalKernel->walk([&](GatherOp gather) { gathers.push_back(gather); });
  for (GatherOp gather : gathers)
    if (gather->getBlock() && failed(composeLoadGather(gather)))
      return failure();
  eraseDeadPhysicalValues(*physicalKernel);
  return success();
}

} // namespace intent::gpu

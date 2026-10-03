#include "ReductionRealization.h"
#include "Intent/Dialect/GPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/GPU/Transforms/Control/Traversal.h"

using namespace mlir;

namespace intent::gpu::reduction {

LogicalResult prepareReductionReads(ArrayRef<RootAccess> accesses,
                                    ReduceOp reduce, func::FuncOp kernel) {
  // Full coverage can retarget the original coordinate domain. Do it before
  // creating chunk coordinates so they retain their selected physical extent.
  for (RootAccess access : accesses)
    if (!canReplayReadAt(access.load, reduce) &&
        !PhysicalProgramAnalysis(kernel)
             .axisRealization(access.load.getResult(), access.fragmentAxis)
             .physicalized &&
        failed(realizeFullCoverageDimension(
            kernel, access.load.getResult(), access.fragmentAxis)))
      return reduce.emitOpError(
          "reduction could not materialize its retained source read");
  return success();
}

FailureOr<Value> materializeReductionRead(
    OpBuilder &builder, Location location, RootAccess access,
    FragmentType resultType, ValueRange coordinates, Value valid, Value fill,
    Value traversalCoordinate, ReduceOp reduce) {
  if (canReplayReadAt(access.load, reduce))
    return Value(builder.create<LoadOp>(
        location, resultType, access.load.getResource(), coordinates, valid,
        fill, access.load.getSourceAxes()));
  auto extent = cast<PhysicalExprAttr>(resultType.getShape()[access.fragmentAxis]);
  if (!traversalCoordinate)
    return failure();
  // The saved SSA tensor is indexed by its traversal ordinal, not the resource
  // address (which may include offsets, permutation or other index arithmetic).
  return materializeRetainedSlice(
      builder, location, access.load.getResult(), access.fragmentAxis, extent,
      traversalCoordinate, reduce);
}

} // namespace intent::gpu::reduction

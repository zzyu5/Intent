#include "Intent/Transforms/RealizeStableSoftmax.h"

#include "Intent/Analysis/StableSoftmax.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace intent {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

void createPrimitive(OpBuilder &builder, Location location, Operation *source,
                     StringRef kind, StringRef operatorName, int64_t axis,
                     StringRef identity) {
  builder.create<plan::PrimitiveOp>(
      location, i64(builder, getIntentNodeID(source)), string(builder, kind),
      string(builder, operatorName), i64(builder, axis),
      string(builder, identity));
}

} // namespace

LogicalResult realizeStableSoftmax(ModuleOp module,
                                   const triton::TargetOptions &target) {
  if (target.architecture.empty() || target.device < 0 || target.warpSize <= 0)
    return module.emitError("stable softmax realization requires a complete target");
  if (!module.getOps<plan::PlanOp>().empty())
    return module.emitError("module already contains a Physical Plan");
  if (failed(verifyKernelModule(module)))
    return failure();
  FailureOr<StableSoftmaxMatch> matched = matchStableSoftmax(module);
  if (failed(matched))
    return failure();
  StableSoftmaxMatch &softmax = *matched;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto physicalPlan = builder.create<plan::PlanOp>(
      softmax.entry.getLoc(),
      FlatSymbolRefAttr::get(module.getContext(), softmax.entry.getName()),
      string(builder, "triton"), string(builder, target.architecture),
      i64(builder, target.device), i64(builder, target.warpSize));
  Block &body = physicalPlan.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);

  builder.create<plan::ExtentOp>(
      softmax.rowLoop->getLoc(), i64(builder, getIntentNodeID(softmax.rowLoop)),
      i64(builder, 0), string(builder, softmax.rowExtent),
      string(builder, "one"));
  builder.create<plan::ExtentOp>(
      softmax.columnDomain->getLoc(),
      i64(builder, getIntentNodeID(softmax.columnDomain)), i64(builder, 1),
      string(builder, softmax.columnExtent),
      string(builder, "next_power_of_two"));
  builder.create<plan::OwnershipOp>(
      softmax.rowLoop->getLoc(), i64(builder, getIntentNodeID(softmax.rowLoop)),
      string(builder, "program"), i64(builder, 0),
      string(builder, "persistent"), string(builder, "grid_stride"));

  builder.create<plan::StorageOp>(softmax.entry.getLoc(),
                                  i64(builder, softmax.inputValueID),
                                  string(builder, "global"),
                                  string(builder, "read"));
  builder.create<plan::StorageOp>(softmax.entry.getLoc(),
                                  i64(builder, softmax.outputValueID),
                                  string(builder, "global"),
                                  string(builder, "write"));
  auto rowMajor = builder.getDenseI64ArrayAttr({0, 1});
  builder.create<plan::LayoutOp>(softmax.entry.getLoc(),
                                 i64(builder, softmax.inputValueID),
                                 string(builder, "row_major"), rowMajor);
  builder.create<plan::LayoutOp>(softmax.entry.getLoc(),
                                 i64(builder, softmax.outputValueID),
                                 string(builder, "row_major"), rowMajor);

  createPrimitive(builder, softmax.reduceMax->getLoc(), softmax.reduceMax,
                  "reduction", "maximum", 0, "negative_infinity");
  createPrimitive(builder, softmax.maxBroadcast->getLoc(),
                  softmax.maxBroadcast, "pointwise", "broadcast", -1, "none");
  createPrimitive(builder, softmax.subtract->getLoc(), softmax.subtract,
                  "pointwise", "subtract", -1, "none");
  createPrimitive(builder, softmax.exponential->getLoc(), softmax.exponential,
                  "pointwise", "exp", -1, "none");
  createPrimitive(builder, softmax.reduceSum->getLoc(), softmax.reduceSum,
                  "reduction", "add", 0, "zero");
  createPrimitive(builder, softmax.sumBroadcast->getLoc(), softmax.sumBroadcast,
                  "pointwise", "broadcast", -1, "none");
  createPrimitive(builder, softmax.divide->getLoc(), softmax.divide,
                  "pointwise", "true_divide", -1, "none");

  builder.create<plan::BoundaryOp>(
      softmax.columnDomain->getLoc(),
      i64(builder, getIntentNodeID(softmax.columnDomain)),
      string(builder, softmax.columnExtent), string(builder, "masked"),
      string(builder, "index_lt_extent"),
      string(builder, "negative_infinity"));
  builder.create<plan::PipelineOp>(
      softmax.rowLoop->getLoc(), i64(builder, 2), i64(builder, 4),
      i64(builder, 200000), builder.getBoolAttr(false),
      builder.getBoolAttr(false));
  builder.create<plan::LaunchOp>(
      softmax.rowLoop->getLoc(), i64(builder, getIntentNodeID(softmax.rowLoop)),
      string(builder, "persistent_occupancy"), i64(builder, 8));
  builder.create<plan::YieldOp>(softmax.entry.getLoc());
  return success();
}

} // namespace intent

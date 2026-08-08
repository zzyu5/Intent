#include "Intent/Dialect/Plan/IR/PlanOps.h"

#include "Intent/Analysis/StableSoftmax.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace intent::plan;

namespace {

struct ExpectedPrimitive {
  Operation *source;
  StringRef kind;
  StringRef operatorName;
  int64_t axis;
  StringRef identity;
};

bool matches(PrimitiveOp primitive, const ExpectedPrimitive &expected) {
  return primitive.getNodeAttr().getInt() ==
             intent::getIntentNodeID(expected.source) &&
         primitive.getKind() == expected.kind &&
         primitive.getOperatorName() == expected.operatorName &&
         primitive.getAxisAttr().getInt() == expected.axis &&
         primitive.getIdentity() == expected.identity;
}

} // namespace

LogicalResult PlanOp::verify() {
  if (getBackend() != "triton")
    return emitOpError("initial Physical Plan supports only Triton");
  if (getArchitecture().empty() || getDeviceAttr().getInt() < 0 ||
      getWarpSizeAttr().getInt() <= 0)
    return emitOpError("requires complete target architecture/device/warp data");

  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("must be nested directly in an MLIR module");
  FailureOr<intent::StableSoftmaxMatch> matched =
      intent::matchStableSoftmax(module);
  if (failed(matched))
    return failure();
  intent::StableSoftmaxMatch &softmax = *matched;
  if (getEntry() != softmax.entry.getName())
    return emitOpError("entry does not reference the matched stable softmax kernel");

  DenseMap<int64_t, ExtentOp> extents;
  SmallVector<OwnershipOp> ownerships;
  DenseMap<int64_t, StorageOp> storage;
  DenseMap<int64_t, LayoutOp> layouts;
  DenseMap<int64_t, PrimitiveOp> primitives;
  SmallVector<BoundaryOp> boundaries;
  SmallVector<PipelineOp> pipelines;
  SmallVector<LaunchOp> launches;

  for (Operation &operation : getBody().front()) {
    if (isa<YieldOp>(operation))
      continue;
    if (auto extent = dyn_cast<ExtentOp>(operation)) {
      int64_t axis = extent.getAxisAttr().getInt();
      if (!extents.try_emplace(axis, extent).second)
        return extent.emitOpError("duplicates a physical axis");
      continue;
    }
    if (auto ownership = dyn_cast<OwnershipOp>(operation)) {
      ownerships.push_back(ownership);
      continue;
    }
    if (auto binding = dyn_cast<StorageOp>(operation)) {
      if (!storage.try_emplace(binding.getValueAttr().getInt(), binding).second)
        return binding.emitOpError("duplicates a storage value");
      continue;
    }
    if (auto layout = dyn_cast<LayoutOp>(operation)) {
      if (!layouts.try_emplace(layout.getValueAttr().getInt(), layout).second)
        return layout.emitOpError("duplicates a layout value");
      continue;
    }
    if (auto primitive = dyn_cast<PrimitiveOp>(operation)) {
      if (!primitives.try_emplace(primitive.getNodeAttr().getInt(), primitive)
               .second)
        return primitive.emitOpError("duplicates a Kernel IR operation binding");
      continue;
    }
    if (auto boundary = dyn_cast<BoundaryOp>(operation)) {
      boundaries.push_back(boundary);
      continue;
    }
    if (auto pipeline = dyn_cast<PipelineOp>(operation)) {
      pipelines.push_back(pipeline);
      continue;
    }
    if (auto launch = dyn_cast<LaunchOp>(operation)) {
      launches.push_back(launch);
      continue;
    }
    return operation.emitOpError("is not legal inside intent_plan.plan");
  }

  if (extents.size() != 2 || !extents.count(0) || !extents.count(1))
    return emitOpError("requires one row and one column extent");
  ExtentOp rowExtent = extents.lookup(0);
  ExtentOp columnExtent = extents.lookup(1);
  if (rowExtent.getNodeAttr().getInt() !=
          intent::getIntentNodeID(softmax.rowLoop) ||
      rowExtent.getLogical() != softmax.rowExtent ||
      rowExtent.getTile() != "one")
    return rowExtent.emitOpError("does not realize the source row extent");
  if (columnExtent.getNodeAttr().getInt() !=
          intent::getIntentNodeID(softmax.columnDomain) ||
      columnExtent.getLogical() != softmax.columnExtent ||
      columnExtent.getTile() != "next_power_of_two")
    return columnExtent.emitOpError("does not realize the source column extent");

  if (ownerships.size() != 1)
    return emitOpError("requires exactly one ownership binding");
  OwnershipOp ownership = ownerships.front();
  if (ownership.getLoopNodeAttr().getInt() !=
          intent::getIntentNodeID(softmax.rowLoop) ||
      ownership.getWorker() != "program" ||
      ownership.getWorkerAxisAttr().getInt() != 0 ||
      ownership.getTraversal() != "persistent" ||
      ownership.getMapping() != "grid_stride")
    return ownership.emitOpError(
        "must use persistent program-axis-zero grid-stride row ownership");

  if (storage.size() != 2 || !storage.count(softmax.inputValueID) ||
      !storage.count(softmax.outputValueID))
    return emitOpError("storage must exactly cover the two ABI views");
  StorageOp inputStorage = storage.lookup(softmax.inputValueID);
  StorageOp outputStorage = storage.lookup(softmax.outputValueID);
  if (inputStorage.getSpace() != "global" ||
      inputStorage.getAccess() != "read" ||
      outputStorage.getSpace() != "global" ||
      outputStorage.getAccess() != "write")
    return emitOpError("storage changes the source ABI access contract");

  if (layouts.size() != 2 || !layouts.count(softmax.inputValueID) ||
      !layouts.count(softmax.outputValueID))
    return emitOpError("layout must exactly cover the two ABI views");
  for (LayoutOp layout : {layouts.lookup(softmax.inputValueID),
                          layouts.lookup(softmax.outputValueID)})
    if (layout.getKind() != "row_major" || layout.getOrder().size() != 2 ||
        layout.getOrder()[0] != 0 || layout.getOrder()[1] != 1)
      return layout.emitOpError("must preserve row-major axes [0, 1]");

  SmallVector<ExpectedPrimitive> expected = {
      {softmax.reduceMax, "reduction", "maximum", 0,
       "negative_infinity"},
      {softmax.maxBroadcast, "pointwise", "broadcast", -1, "none"},
      {softmax.subtract, "pointwise", "subtract", -1, "none"},
      {softmax.exponential, "pointwise", "exp", -1, "none"},
      {softmax.reduceSum, "reduction", "add", 0, "zero"},
      {softmax.sumBroadcast, "pointwise", "broadcast", -1, "none"},
      {softmax.divide, "pointwise", "true_divide", -1, "none"},
  };
  if (primitives.size() != expected.size())
    return emitOpError("primitive bindings must exactly cover stable softmax");
  for (const ExpectedPrimitive &binding : expected) {
    int64_t node = intent::getIntentNodeID(binding.source);
    PrimitiveOp primitive = primitives.lookup(node);
    if (!primitive || !matches(primitive, binding))
      return emitOpError()
             << "primitive binding for Kernel IR node " << node
             << " changes its operator/axis/identity";
  }

  if (boundaries.size() != 1)
    return emitOpError("requires exactly one column boundary");
  BoundaryOp boundary = boundaries.front();
  if (boundary.getNodeAttr().getInt() !=
          intent::getIntentNodeID(softmax.columnDomain) ||
      boundary.getLogical() != softmax.columnExtent ||
      boundary.getTail() != "masked" ||
      boundary.getPredicate() != "index_lt_extent" ||
      boundary.getLoadFill() != "negative_infinity")
    return boundary.emitOpError("does not preserve the source column boundary");

  if (pipelines.size() != 1)
    return emitOpError("requires exactly one pipeline policy");
  PipelineOp pipeline = pipelines.front();
  if (pipeline.getLowStagesAttr().getInt() <= 0 ||
      pipeline.getHighStagesAttr().getInt() <= 0 ||
      pipeline.getSmemThresholdAttr().getInt() <= 0 ||
      pipeline.getPrefetch() || pipeline.getAsyncCopy())
    return pipeline.emitOpError("contains an unsupported pipeline policy");

  if (launches.size() != 1)
    return emitOpError("requires exactly one launch policy");
  LaunchOp launch = launches.front();
  if (launch.getLoopNodeAttr().getInt() !=
          intent::getIntentNodeID(softmax.rowLoop) ||
      launch.getGridPolicy() != "persistent_occupancy" ||
      launch.getNumWarpsAttr().getInt() <= 0)
    return launch.emitOpError("does not launch the realized row ownership");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Dialect/Plan/IR/PlanOps.cpp.inc"

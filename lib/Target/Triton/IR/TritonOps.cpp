#include "Intent/Target/Triton/IR/TritonOps.h"

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace intent::triton::plan;

namespace {

LogicalResult requireNonNegative(Operation *operation, int64_t value,
                                 StringRef field) {
  if (value < 0)
    return operation->emitOpError() << field << " must be non-negative";
  return success();
}

LogicalResult requireNode(Operation *operation, int64_t value) {
  return requireNonNegative(operation, value, "Kernel IR node ID");
}

} // namespace

LogicalResult TargetOp::verify() {
  if (getArchitecture().empty() || getDeviceAttr().getInt() < 0 ||
      getWarpSizeAttr().getInt() <= 0)
    return emitOpError("requires architecture, device and warp size");
  return success();
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())))
    return failure();
  if (getSourceAxisAttr().getInt() < 0)
    return emitOpError("source axis must be non-negative");
  if (getRole() != "row" && getRole() != "column")
    return emitOpError("role must be row or column");
  if (getTile() != "one" && getTile() != "next_power_of_two")
    return emitOpError("contains an unsupported Triton tile choice");
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(requireNode(*this, getLoopNodeAttr().getInt())))
    return failure();
  if (getWorkerAxisAttr().getInt() < 0)
    return emitOpError("worker axis must be non-negative");
  if (getTraversal() != "persistent" || getMapping() != "grid_stride")
    return emitOpError("contains an unsupported program traversal");
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (getSpace() != "global")
    return emitOpError("initial Triton realization supports global storage");
  return success();
}

LogicalResult LayoutOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (getKind() != "row_major" || getOrder().size() != 2 ||
      getOrder()[0] != 0 || getOrder()[1] != 1)
    return emitOpError("initial Triton realization supports row-major [0, 1]");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())))
    return failure();
  if (getAxisAttr().getInt() < 0)
    return emitOpError("reduction axis must be non-negative");
  if (getLowering() != "tl.max" && getLowering() != "tl.sum")
    return emitOpError("contains an unsupported Triton reduction lowering");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering().empty())
    return emitOpError("requires a target lowering name");
  return success();
}

LogicalResult BoundaryOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getDomainNode())))
    return failure();
  if (getPredicate() != "index_lt_extent" ||
      getStoreMask() != "predicate")
    return emitOpError("contains an unsupported boundary mechanism");
  if (getLoadFill() != "negative_infinity")
    return emitOpError("contains an unsupported masked-load fill");
  return success();
}

LogicalResult PipelineOp::verify() {
  if (failed(requireNode(*this, getLoopNode())) || getLowStages() <= 0 ||
      getHighStages() <= 0 || getSmemThreshold() <= 0)
    return emitOpError("requires a loop and positive pipeline parameters");
  if (getPrefetch() || getAsyncCopy())
    return emitOpError("contains an unsupported pipeline mechanism");
  return success();
}

LogicalResult LaunchOp::verify() {
  if (failed(requireNode(*this, getLoopNode())) || getNumWarps() <= 0)
    return failure();
  if (getGridPolicy() != "persistent_occupancy")
    return emitOpError("contains an unsupported grid policy");
  return success();
}

LogicalResult intent::triton::plan::verifyTritonRealization(
    intent::plan::RealizationOp realization) {
  if (realization.getTarget() != "triton")
    return realization.emitOpError("is not a Triton realization");
  unsigned targets = 0;
  unsigned programs = 0;
  unsigned pipelines = 0;
  unsigned launches = 0;
  llvm::DenseSet<int64_t> axes;
  llvm::DenseSet<int64_t> storage;
  llvm::DenseSet<int64_t> layouts;
  llvm::DenseSet<int64_t> primitives;
  llvm::DenseSet<int64_t> boundaries;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() != "intent_triton")
      return operation.emitOpError("is not legal in a Triton realization");
    if (isa<TargetOp>(operation))
      ++targets;
    else if (auto axis = dyn_cast<AxisOp>(operation)) {
      if (!axes.insert(axis.getNode()).second)
        return axis.emitOpError("duplicates an axis node binding");
    } else if (isa<ProgramOp>(operation))
      ++programs;
    else if (auto binding = dyn_cast<StorageOp>(operation)) {
      if (!storage.insert(binding.getValue()).second)
        return binding.emitOpError("duplicates a storage value binding");
    } else if (auto binding = dyn_cast<LayoutOp>(operation)) {
      if (!layouts.insert(binding.getValue()).second)
        return binding.emitOpError("duplicates a layout value binding");
    } else if (auto binding = dyn_cast<ReductionOp>(operation)) {
      if (!primitives.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation lowering");
    } else if (auto binding = dyn_cast<PointwiseOp>(operation)) {
      if (!primitives.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation lowering");
    } else if (auto binding = dyn_cast<BoundaryOp>(operation)) {
      if (!boundaries.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a boundary binding");
    } else if (isa<PipelineOp>(operation))
      ++pipelines;
    else if (isa<LaunchOp>(operation))
      ++launches;
  }
  if (targets != 1 || programs != 1 || pipelines != 1 || launches != 1)
    return realization.emitOpError(
        "requires one target, program, pipeline and launch choice");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"

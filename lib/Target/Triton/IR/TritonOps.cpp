#include "Intent/Target/Triton/IR/TritonOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
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
  if (getRole().empty() || getTile().empty())
    return emitOpError("requires a physical role and tile choice");
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(requireNode(*this, getLoopNodeAttr().getInt())))
    return failure();
  if (getWorkerAxes().empty() || getWorkerAxes().size() > 3)
    return emitOpError("requires between one and three Triton program axes");
  for (int64_t axis : getWorkerAxes())
    if (axis < 0 || axis > 2)
      return emitOpError("program axes must be in [0, 2]");
  if (getTraversal().empty() || getMapping().empty())
    return emitOpError("requires traversal and program mapping choices");
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (getSpace() != "global" && getSpace() != "shared" &&
      getSpace() != "register")
    return emitOpError("contains an unsupported Triton storage space");
  return success();
}

LogicalResult LayoutOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (getKind().empty() || getOrder().empty())
    return emitOpError("requires a layout kind and dimension order");
  llvm::DenseSet<int64_t> dimensions;
  for (int64_t dimension : getOrder())
    if (dimension < 0 || !dimensions.insert(dimension).second)
      return emitOpError("layout order must be a non-negative permutation");
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

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering() != "tl.dot")
    return emitOpError("contains an unsupported Triton contraction primitive");
  if (getAccumulatorType() != "f32")
    return emitOpError("Triton contraction currently requires f32 accumulation");
  return success();
}

LogicalResult StreamOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getAxisNode())))
    return failure();
  if (getTile().empty() || getOrder() != "forward" ||
      getCarrySpace() != "register")
    return emitOpError(
        "requires a forward tile stream with register-carried state");
  return success();
}

LogicalResult BoundaryOp::verify() {
  if (failed(requireNode(*this, getNode())) || getDomainNodes().empty())
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if (getPredicate() != "index_lt_extent" ||
      getStoreMask() != "predicate")
    return emitOpError("contains an unsupported boundary mechanism");
  if (getLoadFill() != "negative_infinity" && getLoadFill() != "zero" &&
      getLoadFill() != "none")
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
  if (getGridPolicy().empty())
    return emitOpError("requires a grid policy");
  return success();
}

LogicalResult AutotuneOp::verify() {
  if (getKey().empty())
    return emitOpError("requires at least one specialization key");
  llvm::StringSet<> keys;
  for (Attribute attribute : getKey()) {
    auto key = dyn_cast<StringAttr>(attribute);
    if (!key || key.getValue().empty() || !keys.insert(key.getValue()).second)
      return emitOpError("autotune keys must be unique non-empty strings");
  }
  return success();
}

LogicalResult ConfigOp::verify() {
  if (getParameters().empty() || getNumStages() <= 0 || getNumWarps() <= 0)
    return emitOpError("requires positive Triton autotune parameters");
  for (NamedAttribute parameter : getParameters()) {
    auto value = dyn_cast<IntegerAttr>(parameter.getValue());
    if (!value || value.getInt() <= 0)
      return emitOpError(
          "Triton config parameters must be positive integers");
  }
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
  ProgramOp programChoice;
  PipelineOp pipelineChoice;
  LaunchOp launchChoice;
  StreamOp streamChoice;
  llvm::DenseSet<int64_t> axes;
  llvm::StringSet<> axisRoles;
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
      if (!axisRoles.insert(axis.getRole()).second)
        return axis.emitOpError("duplicates a physical axis role");
    } else if (auto program = dyn_cast<ProgramOp>(operation)) {
      ++programs;
      programChoice = program;
    } else if (auto binding = dyn_cast<StorageOp>(operation)) {
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
    } else if (auto binding = dyn_cast<ContractOp>(operation)) {
      if (!primitives.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation lowering");
    } else if (auto stream = dyn_cast<StreamOp>(operation)) {
      if (streamChoice)
        return stream.emitOpError("duplicates an ordered stream binding");
      streamChoice = stream;
    } else if (auto binding = dyn_cast<BoundaryOp>(operation)) {
      if (!boundaries.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a boundary binding");
    } else if (auto pipeline = dyn_cast<PipelineOp>(operation)) {
      ++pipelines;
      pipelineChoice = pipeline;
    } else if (auto launch = dyn_cast<LaunchOp>(operation)) {
      ++launches;
      launchChoice = launch;
    } else
      return operation.emitOpError(
          "is not a recognized resolved Triton choice");
  }
  if (targets != 1 || programs != 1 || pipelines > 1 || launches > 1)
    return realization.emitOpError(
        "requires one target/program and at most one fixed pipeline/launch");
  if (pipelineChoice && pipelineChoice.getLoopNode() != programChoice.getLoopNode())
    return pipelineChoice.emitOpError("does not bind the resolved program root");
  if (launchChoice && launchChoice.getLoopNode() != programChoice.getLoopNode())
    return launchChoice.emitOpError("does not bind the resolved program root");
  if (programChoice.getMapping() == "grid_stride" &&
      (!pipelineChoice || !launchChoice))
    return realization.emitOpError(
        "grid-stride programs require fixed pipeline and launch choices");
  if (programChoice.getMapping() == "grouped_2d_tiles" &&
      (pipelineChoice || launchChoice))
    return realization.emitOpError(
        "autotuned grouped programs cannot carry fixed pipeline/launch choices");
  if (programChoice.getMapping() == "multi_axis_stream" &&
      (pipelineChoice || launchChoice || !streamChoice))
    return realization.emitOpError(
        "multi-axis streams require one stream and no fixed pipeline/launch");
  if (streamChoice && !axes.contains(streamChoice.getAxisNode()))
    return streamChoice.emitOpError("references an unbound stream axis");
  if (programChoice.getMapping() != "grid_stride" &&
      programChoice.getMapping() != "grouped_2d_tiles" &&
      programChoice.getMapping() != "multi_axis_stream")
    return programChoice.emitOpError("contains an unsupported Triton mapping");
  return success();
}

LogicalResult intent::triton::plan::verifyTritonSearchSpace(
    intent::plan::SearchSpaceOp searchSpace) {
  if (searchSpace.getTarget() != "triton")
    return searchSpace.emitOpError("is not a Triton search space");
  unsigned autotune = 0;
  unsigned configs = 0;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() != "intent_triton")
      return operation.emitOpError("is not legal in a Triton search space");
    if (isa<AutotuneOp>(operation))
      ++autotune;
    else if (isa<ConfigOp>(operation))
      ++configs;
    else
      return operation.emitOpError(
          "is a resolved choice, not a search candidate");
  }
  if (autotune != 1 || configs == 0)
    return searchSpace.emitOpError(
        "requires one autotune key declaration and at least one config");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"

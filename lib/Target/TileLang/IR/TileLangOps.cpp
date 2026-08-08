#include "Intent/Target/TileLang/IR/TileLangOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace intent::tilelang::plan;

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

bool isBufferSpace(StringRef space) {
  return space == "global" || space == "shared" || space == "local" ||
         space == "fragment";
}

} // namespace

LogicalResult TargetOp::verify() {
  if (getArchitecture().empty() || getDeviceAttr().getInt() < 0)
    return emitOpError("requires architecture and device");
  return success();
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      getSourceAxisAttr().getInt() < 0)
    return failure();
  if (getRole().empty() || getTile().empty())
    return emitOpError("requires a physical role and tile choice");
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(requireNode(*this, getLoopNode())))
    return failure();
  if (getWorkerAxes().empty() || getWorkerAxes().size() > 3)
    return emitOpError("requires between one and three TileLang grid axes");
  for (int64_t axis : getWorkerAxes())
    if (axis < 0 || axis > 2)
      return emitOpError("grid axes must be in [0, 2]");
  if (getTraversal().empty() || getMapping().empty() || getGroupSize() <= 0)
    return emitOpError("requires traversal, mapping and positive group size");
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (!isBufferSpace(getSpace()))
    return emitOpError("contains an unsupported TileLang buffer space");
  return success();
}

LogicalResult LayoutOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")) ||
      getKind().empty() || getOrder().empty())
    return failure();
  if (getKind() != "row_major" && getKind() != "shared" &&
      getKind() != "fragment")
    return emitOpError("contains an unsupported TileLang layout kind");
  llvm::DenseSet<int64_t> dimensions;
  for (int64_t dimension : getOrder())
    if (dimension < 0 || !dimensions.insert(dimension).second)
      return emitOpError("layout order must be a non-negative permutation");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNode())) || getAxisAttr().getInt() < 0)
    return failure();
  if (getLowering() != "T.reduce_max" && getLowering() != "T.reduce_sum")
    return emitOpError("contains an unsupported TileLang reduction lowering");
  if (getInputSpace() != "fragment" || getResultSpace() != "fragment")
    return emitOpError("requires fragment-resident reduction values");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering().empty() || !isBufferSpace(getSpace()) ||
      getReuseOperandAttr().getInt() < -1)
    return emitOpError("requires a lowering and explicit TileLang value space");
  return success();
}

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering() != "T.gemm" || getAccumulatorType() != "f32")
    return emitOpError("requires T.gemm with f32 accumulation");
  if (getWarpPolicy() != "square" && getWarpPolicy() != "full_row")
    return emitOpError("contains an unsupported TileLang GEMM warp policy");
  auto isOperandSpace = [](StringRef space) {
    return space == "shared" || space == "fragment";
  };
  if (!isOperandSpace(getLhsSpace()) || !isOperandSpace(getRhsSpace()) ||
      getAccumulatorSpace() != "fragment")
    return emitOpError(
        "requires shared/fragment operands and a fragment accumulator");
  return success();
}

LogicalResult StreamOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getAxisNode())))
    return failure();
  if (getTile().empty() || getOrder() != "forward" ||
      getCarrySpace() != "fragment" || !getReuseInitial())
    return emitOpError(
        "requires a forward tile stream that reuses fragment initial state");
  return success();
}

LogicalResult RaggedOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getOuterNode())) ||
      failed(requireNode(*this, getMemberNode())))
    return failure();
  if (getTraversal() != "expert_offset_ranges")
    return emitOpError("contains an unsupported TileLang ragged traversal");
  return success();
}

LogicalResult StageOp::verify() {
  if (getOrdinalAttr().getInt() < 0 || failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t value : getInputs())
    if (failed(requireNonNegative(*this, value, "stage input value ID")))
      return failure();
  for (int64_t value : getOutputs())
    if (failed(requireNonNegative(*this, value, "stage output value ID")))
      return failure();
  return success();
}

LogicalResult AtomicOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering() != "T.atomic_add")
    return emitOpError("requires a T.atomic_add lowering");
  return success();
}

LogicalResult BoundaryOp::verify() {
  if (failed(requireNode(*this, getNode())) || getDomainNodes().empty())
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if (getAccess() != "gather" && getAccess() != "scatter" &&
      getAccess() != "load" && getAccess() != "store")
    return emitOpError("contains an unsupported TileLang memory access");
  if (getTransfer() != "bulk_copy" &&
      getTransfer() != "parallel_elements")
    return emitOpError("contains an unsupported TileLang transfer mechanism");
  if (getPadding() != "negative_infinity" && getPadding() != "zero" &&
      getPadding() != "none")
    return emitOpError("contains an unsupported TileLang padding mode");
  if ((getAccess() == "gather" || getAccess() == "load") &&
      getPadding() == "none")
    return emitOpError("load-like access requires a padding choice");
  if ((getAccess() == "scatter" || getAccess() == "store") &&
      getPadding() != "none")
    return emitOpError("store-like access cannot carry load padding");
  return success();
}

LogicalResult PipelineOp::verify() {
  if (failed(requireNode(*this, getLoopNode())) || getNumStages() <= 0)
    return failure();
  return success();
}

LogicalResult LaunchOp::verify() {
  if (failed(requireNode(*this, getLoopNode())) || getThreads() <= 0)
    return failure();
  if (getGridPolicy().empty())
    return emitOpError("requires a TileLang grid policy");
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
  if (getParameters().empty() || getNumStages() <= 0 || getThreads() <= 0)
    return emitOpError("requires positive TileLang candidate choices");
  for (NamedAttribute parameter : getParameters()) {
    auto value = dyn_cast<IntegerAttr>(parameter.getValue());
    if (!value || value.getInt() <= 0)
      return emitOpError("TileLang config parameters must be positive integers");
  }
  return success();
}

LogicalResult intent::tilelang::plan::verifyTileLangRealization(
    intent::plan::RealizationOp realization) {
  if (realization.getTarget() != "tilelang")
    return realization.emitOpError("is not a TileLang realization");
  unsigned targets = 0;
  unsigned programs = 0;
  unsigned launches = 0;
  unsigned pipelines = 0;
  ProgramOp programChoice;
  LaunchOp launchChoice;
  PipelineOp pipelineChoice;
  StreamOp streamChoice;
  RaggedOp raggedChoice;
  llvm::DenseSet<int64_t> axes;
  llvm::StringSet<> axisRoles;
  llvm::DenseSet<int64_t> storage;
  llvm::DenseSet<int64_t> layouts;
  llvm::DenseSet<int64_t> primitives;
  llvm::DenseSet<int64_t> boundaries;
  llvm::DenseSet<int64_t> stageOrdinals;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() != "intent_tilelang")
      return operation.emitOpError("is not legal in a TileLang realization");
    if (isa<TargetOp>(operation))
      ++targets;
    else if (auto axis = dyn_cast<AxisOp>(operation)) {
      if (!axes.insert(axis.getNode()).second ||
          !axisRoles.insert(axis.getRole()).second)
        return axis.emitOpError("duplicates an axis node or role binding");
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
    } else if (auto ragged = dyn_cast<RaggedOp>(operation)) {
      if (raggedChoice)
        return ragged.emitOpError("duplicates a ragged traversal binding");
      raggedChoice = ragged;
    } else if (auto stage = dyn_cast<StageOp>(operation)) {
      if (!stageOrdinals.insert(stage.getOrdinal()).second)
        return stage.emitOpError("duplicates a physical stage ordinal");
    } else if (auto atomic = dyn_cast<AtomicOp>(operation)) {
      if (!primitives.insert(atomic.getNode()).second)
        return atomic.emitOpError("duplicates an operation lowering");
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
          "is not a recognized resolved TileLang choice");
  }
  if (targets != 1 || programs != 1 || launches > 1 || pipelines > 1)
    return realization.emitOpError(
        "requires one target/program and at most one fixed launch/pipeline");
  if (launchChoice && launchChoice.getLoopNode() != programChoice.getLoopNode())
    return launchChoice.emitOpError("does not bind the resolved program root");
  if (pipelineChoice &&
      pipelineChoice.getLoopNode() != programChoice.getLoopNode())
    return pipelineChoice.emitOpError("does not bind the resolved program root");
  if (programChoice.getMapping() == "persistent_rows" &&
      (!launchChoice || !pipelineChoice))
    return realization.emitOpError(
        "persistent rows require fixed launch and pipeline choices");
  if (programChoice.getMapping() != "persistent_rows" &&
      (launchChoice || pipelineChoice))
    return realization.emitOpError(
        "autotuned mappings cannot carry fixed launch or pipeline choices");
  if (programChoice.getMapping() == "multi_axis_stream" && !streamChoice)
    return realization.emitOpError("multi-axis streams require one stream");
  if (programChoice.getMapping() == "ragged_stages" &&
      (!raggedChoice || stageOrdinals.empty()))
    return realization.emitOpError(
        "ragged stages require a ragged traversal and explicit stages");
  if (streamChoice && !axes.contains(streamChoice.getAxisNode()))
    return streamChoice.emitOpError("references an unbound stream axis");
  if (programChoice.getMapping() != "persistent_rows" &&
      programChoice.getMapping() != "grouped_2d_tiles" &&
      programChoice.getMapping() != "multi_axis_stream" &&
      programChoice.getMapping() != "ragged_stages")
    return programChoice.emitOpError("contains an unsupported TileLang mapping");
  return success();
}

LogicalResult intent::tilelang::plan::verifyTileLangSearchSpace(
    intent::plan::SearchSpaceOp searchSpace) {
  if (searchSpace.getTarget() != "tilelang")
    return searchSpace.emitOpError("is not a TileLang search space");
  unsigned autotune = 0;
  unsigned configs = 0;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() != "intent_tilelang")
      return operation.emitOpError("is not legal in a TileLang search space");
    if (isa<AutotuneOp>(operation))
      ++autotune;
    else if (isa<ConfigOp>(operation))
      ++configs;
    else
      return operation.emitOpError("is a resolved choice, not a candidate");
  }
  if (autotune != 1 || configs == 0)
    return searchSpace.emitOpError(
        "requires one autotune declaration and at least one config");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Target/TileLang/IR/TileLangOps.cpp.inc"

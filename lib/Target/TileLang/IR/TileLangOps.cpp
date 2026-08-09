#include "Intent/Target/TileLang/IR/TileLangOps.h"

#include "Intent/Target/Common/Projection/Capabilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

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
         space == "fragment" || space == "none";
}

LogicalResult verifyStringKeys(Operation *operation, ArrayAttr keys) {
  if (keys.empty())
    return operation->emitOpError("requires at least one specialization key");
  llvm::StringSet<> unique;
  for (Attribute attribute : keys) {
    auto key = dyn_cast<StringAttr>(attribute);
    if (!key || key.getValue().empty() ||
        !unique.insert(key.getValue()).second)
      return operation->emitOpError(
          "specialization keys must be unique non-empty strings");
  }
  return success();
}

StringRef projectedOwnership(StringRef ownership) {
  if (ownership == "row")
    return "block_rows";
  if (ownership == "tiled")
    return "block_tiles";
  if (ownership == "ragged")
    return "block_ragged";
  return {};
}

} // namespace

LogicalResult TargetOp::verify() {
  return requireNonNegative(*this, getDevice(), "device");
}

const intent::target::CapabilityProfile &
intent::tilelang::plan::getCapabilityProfile() {
  static const StringRef intentDecided[] = {
      "tile_shape", "ownership", "traversal", "boundary",
      "value_residency", "matrix_primitive",
      "explicit_onchip_buffer_allocation"};
  static const StringRef delegated[] = {
      "layout", "pipeline", "launch_resources", "register_allocation",
      "instruction_selection", "autotune_candidates"};
  static const intent::target::CapabilityProfile profile{intentDecided,
                                                          delegated, {}};
  return profile;
}

LogicalResult CapabilitiesOp::verify() {
  return intent::target::verifyCapabilityProfile(
      *this, getIntentDecided(), getDelegated(), getAbsent(),
      getCapabilityProfile());
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      getSourceAxisAttr().getInt() < 0)
    return failure();
  if (getRole().empty() || getTile().empty())
    return emitOpError("requires a physical role and tile spelling");
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(requireNode(*this, getLoopNode())) || getWorkerAxes().empty() ||
      getWorkerAxes().size() > 3)
    return failure();
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getWorkerAxes())
    if (axis < 0 || axis > 2 || !axes.insert(axis).second)
      return emitOpError("grid axes must be unique values in [0, 2]");
  if (getOwnership() != "block_rows" && getOwnership() != "block_tiles" &&
      getOwnership() != "block_ragged")
    return emitOpError("contains an unsupported TileLang ownership spelling");
  if (getTraversals().empty())
    return emitOpError("requires grid traversal spellings");
  llvm::StringSet<> traversals;
  for (Attribute attribute : getTraversals()) {
    auto traversal = dyn_cast<StringAttr>(attribute);
    if (!traversal || traversal.getValue().empty() ||
        !traversals.insert(traversal.getValue()).second)
      return emitOpError("grid traversal spellings must be unique strings");
  }
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")) ||
      !isBufferSpace(getSpace()) || getSpace() == "none")
    return failure();
  return success();
}

LogicalResult PaddingOp::verify() {
  return intent::plan::verifyPaddingFields(
      *this, getValue(), getTensorAxes(), getDomainNodes(), getFill(),
      getMaterialization());
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNode())) || getAxisAttr().getInt() < 0)
    return failure();
  if (getLowering() != "T.reduce_max" && getLowering() != "T.reduce_sum")
    return emitOpError("contains an unsupported TileLang reduction spelling");
  if (getInputSpace() != "fragment" || getResultSpace() != "fragment")
    return emitOpError("requires fragment-resident reduction values");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())) || getLowering().empty() ||
      !isBufferSpace(getSpace()) || getSpace() == "none" ||
      getReuseOperandAttr().getInt() < -1)
    return failure();
  return success();
}

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering() != "T.gemm" || getAccumulatorType() != "f32")
    return emitOpError("requires T.gemm with f32 accumulation");
  if (getWarpPolicy() != "square" && getWarpPolicy() != "full_row")
    return emitOpError("contains an unsupported TileLang GEMM policy spelling");
  auto validOperand = [](StringRef space) {
    return space == "shared" || space == "fragment";
  };
  if (!validOperand(getLhsSpace()) || !validOperand(getRhsSpace()) ||
      getAccumulatorSpace() != "fragment")
    return emitOpError("contains an unsupported TileLang matrix residency");
  return success();
}

LogicalResult StreamOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getAxisNode())))
    return failure();
  if (getStopNodeAttr() && failed(requireNode(*this, getStopNodeAttr().getInt())))
    return failure();
  if (getTile().empty() || getOrder() != "forward" ||
      getCarrySpace() != "fragment" || !getReuseInitial())
    return emitOpError("requires a forward reused fragment stream spelling");
  return success();
}

LogicalResult RaggedOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getOuterNode())) || getMemberNodes().empty())
    return failure();
  for (int64_t member : getMemberNodes())
    if (failed(requireNode(*this, member)))
      return failure();
  if (getTraversal() != "expert_offset_ranges" &&
      getTraversal() != "compact_offset_tiles")
    return emitOpError("contains an unsupported TileLang ragged spelling");
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
  if (failed(requireNode(*this, getNode())) || getLowering() != "T.atomic_add")
    return emitOpError("requires a T.atomic_add spelling");
  return success();
}

LogicalResult BoundaryOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if (getAccess() != "gather" && getAccess() != "scatter" &&
      getAccess() != "load" && getAccess() != "store")
    return emitOpError("contains an unsupported TileLang access spelling");
  if (getTransfer() != "bulk_copy" &&
      getTransfer() != "parallel_elements")
    return emitOpError("contains an unsupported TileLang transfer spelling");
  if (getPadding() != "negative_infinity" && getPadding() != "zero" &&
      getPadding() != "none")
    return emitOpError("contains an unsupported TileLang padding spelling");
  if (!isBufferSpace(getResultSpace()))
    return emitOpError("contains an unsupported TileLang result space");
  return success();
}

LogicalResult AutotuneOp::verify() {
  if (failed(verifyStringKeys(*this, getKey())))
    return failure();
  return intent::target::verifyParameterMap(*this, getParameterMap());
}

LogicalResult intent::tilelang::plan::verifyTileLangRealization(
    intent::plan::RealizationOp realization) {
  if (failed(intent::plan::verifyGpuRealization(realization)))
    return failure();
  unsigned targets = 0;
  unsigned capabilities = 0;
  unsigned programs = 0;
  ProgramOp surfaceProgram;
  intent::plan::ProgramOp machineProgram;
  llvm::DenseSet<int64_t> machineAxes;
  llvm::DenseSet<int64_t> surfaceAxes;
  llvm::DenseMap<int64_t, intent::plan::TransferOp> transfers;
  llvm::DenseMap<int64_t, intent::plan::PaddingOp> machinePaddings;
  llvm::DenseMap<int64_t, PaddingOp> surfacePaddings;
  for (Operation &operation : realization.getBody().front()) {
    if (auto value = dyn_cast<intent::plan::ProgramOp>(operation))
      machineProgram = value;
    else if (auto value = dyn_cast<intent::plan::AxisOp>(operation))
      machineAxes.insert(value.getNode());
    else if (auto value = dyn_cast<intent::plan::TransferOp>(operation))
      transfers[value.getNode()] = value;
    else if (auto value = dyn_cast<intent::plan::PaddingOp>(operation))
      machinePaddings[value.getValue()] = value;
    if (operation.getName().getDialectNamespace() != "intent_tilelang")
      continue;
    if (isa<TargetOp>(operation))
      ++targets;
    else if (isa<CapabilitiesOp>(operation))
      ++capabilities;
    else if (auto value = dyn_cast<AxisOp>(operation))
      surfaceAxes.insert(value.getNode());
    else if (auto value = dyn_cast<ProgramOp>(operation)) {
      ++programs;
      surfaceProgram = value;
    } else if (auto value = dyn_cast<PaddingOp>(operation)) {
      if (surfacePaddings.count(value.getValue()))
        return value.emitOpError("duplicates a projected padding decision");
      surfacePaddings[value.getValue()] = value;
    } else if (isa<StorageOp, ReductionOp, PointwiseOp, ContractOp, StreamOp,
                   RaggedOp, StageOp, AtomicOp>(operation)) {
    } else if (auto value = dyn_cast<BoundaryOp>(operation)) {
      auto transfer = transfers.find(value.getNode());
      if (transfer == transfers.end() ||
          transfer->second.getDefer() != value.getDefer())
        return value.emitOpError("does not project its machine transfer decision");
    } else
      return operation.emitOpError("is not legal in a TileLang projection");
  }
  if (targets != 1 || capabilities != 1 || programs != 1 || !machineProgram)
    return realization.emitOpError(
        "requires one TileLang surface, capability cut, and program projection");
  if (machineAxes.size() != surfaceAxes.size() ||
      llvm::any_of(machineAxes, [&](int64_t node) {
        return !surfaceAxes.contains(node);
      }))
    return realization.emitOpError("TileLang axes do not project the GPU axes");
  if (machinePaddings.size() != surfacePaddings.size())
    return realization.emitOpError(
        "TileLang padding does not project every GPU padding decision");
  for (auto &entry : machinePaddings) {
    auto projected = surfacePaddings.find(entry.first);
    if (projected == surfacePaddings.end() ||
        projected->second.getTensorAxes() != entry.second.getTensorAxes() ||
        projected->second.getDomainNodes() != entry.second.getDomainNodes() ||
        projected->second.getFill() != entry.second.getFill() ||
        projected->second.getMaterialization() !=
            entry.second.getMaterialization())
      return realization.emitOpError(
          "TileLang padding changes a GPU padding decision");
  }
  if (surfaceProgram.getLoopNode() != machineProgram.getLoopNode() ||
      surfaceProgram.getWorkerAxes() != machineProgram.getWorkerAxes() ||
      surfaceProgram.getTraversals() != machineProgram.getTraversals() ||
      surfaceProgram.getOwnership() !=
          projectedOwnership(machineProgram.getOwnership()))
    return surfaceProgram.emitOpError("does not project the GPU program decision");
  return success();
}

LogicalResult intent::tilelang::plan::verifyTileLangSearchSpace(
    intent::plan::SearchSpaceOp searchSpace) {
  if (failed(intent::plan::verifyGpuSearchSpace(searchSpace)))
    return failure();
  auto machine =
      *searchSpace.getBody().front().getOps<intent::plan::AutotuneOp>().begin();
  AutotuneOp surface;
  unsigned count = 0;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (operation.getName().getDialectNamespace() != "intent_tilelang")
      continue;
    if (auto value = dyn_cast<AutotuneOp>(operation)) {
      surface = value;
      ++count;
    } else
      return operation.emitOpError("is not legal in a TileLang search projection");
  }
  if (count != 1 || surface.getKey() != machine.getKey())
    return searchSpace.emitOpError(
        "requires one TileLang tuner projection with the GPU specialization keys");
  llvm::StringSet<> expected;
  for (Attribute parameter : machine.getParameters())
    expected.insert(cast<StringAttr>(parameter).getValue());
  llvm::StringSet<> projected;
  for (NamedAttribute mapping : surface.getParameterMap())
    projected.insert(cast<StringAttr>(mapping.getValue()).getValue());
  if (expected.size() != projected.size())
    return surface.emitOpError("does not map every GPU tunable parameter");
  for (const auto &role : expected)
    if (!projected.contains(role.getKey()))
      return surface.emitOpError("does not map every GPU tunable parameter");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Target/TileLang/IR/TileLangOps.cpp.inc"

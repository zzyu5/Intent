#include "Intent/Target/Triton/IR/TritonOps.h"

#include "Intent/Target/Common/Projection/Capabilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

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
    return "program_rows";
  if (ownership == "tiled")
    return "program_tiles";
  if (ownership == "ragged")
    return "program_ragged";
  return {};
}

} // namespace

LogicalResult TargetOp::verify() {
  return requireNonNegative(*this, getDevice(), "device");
}

const intent::target::CapabilityProfile &
intent::triton::plan::getCapabilityProfile() {
  static const StringRef intentDecided[] = {
      "tile_shape", "ownership", "traversal", "boundary",
      "value_residency", "matrix_primitive"};
  static const StringRef delegated[] = {
      "layout", "pipeline", "launch_resources", "register_allocation",
      "instruction_selection", "autotune_candidates"};
  static const StringRef absent[] = {"explicit_onchip_buffer_allocation"};
  static const intent::target::CapabilityProfile profile{intentDecided,
                                                          delegated, absent};
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
      return emitOpError("program axes must be unique values in [0, 2]");
  if (getOwnership() != "program_rows" && getOwnership() != "program_tiles" &&
      getOwnership() != "program_ragged")
    return emitOpError("contains an unsupported Triton ownership spelling");
  if (getTraversals().empty())
    return emitOpError("requires program traversal spellings");
  llvm::StringSet<> traversals;
  for (Attribute attribute : getTraversals()) {
    auto traversal = dyn_cast<StringAttr>(attribute);
    if (!traversal || traversal.getValue().empty() ||
        !traversals.insert(traversal.getValue()).second)
      return emitOpError("program traversal spellings must be unique strings");
  }
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (getSpace() != "global" && getSpace() != "shared" &&
      getSpace() != "register")
    return emitOpError("contains an unsupported Triton storage spelling");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNode())) || getAxisAttr().getInt() < 0)
    return failure();
  if (getLowering() != "tl.max" && getLowering() != "tl.sum")
    return emitOpError("contains an unsupported Triton reduction spelling");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())) || getLowering().empty())
    return failure();
  return success();
}

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getLowering() != "tl.dot" || getAccumulatorType() != "f32")
    return emitOpError("requires tl.dot with f32 accumulation");
  return success();
}

LogicalResult StreamOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getAxisNode())))
    return failure();
  if (getStopNodeAttr() && failed(requireNode(*this, getStopNodeAttr().getInt())))
    return failure();
  if (getTile().empty() || getOrder() != "forward" ||
      getCarrySpace() != "register")
    return emitOpError("requires a forward register-carried stream spelling");
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
    return emitOpError("contains an unsupported Triton ragged spelling");
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
  if (getLowering() != "tl.atomic_add" || getScope() != "gpu")
    return emitOpError("requires a GPU-scoped tl.atomic_add spelling");
  return success();
}

LogicalResult BoundaryOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if ((!getDomainNodes().empty() && getPredicate() != "index_lt_extent") ||
      (getDomainNodes().empty() && getPredicate() != "none") ||
      getStoreMask() != "predicate")
    return emitOpError("contains an unsupported Triton boundary spelling");
  if (getLoadFill() != "negative_infinity" && getLoadFill() != "zero" &&
      getLoadFill() != "none")
    return emitOpError("contains an unsupported Triton load fill");
  return success();
}

LogicalResult AutotuneOp::verify() {
  if (failed(verifyStringKeys(*this, getKey())))
    return failure();
  return intent::target::verifyParameterMap(*this, getParameterMap());
}

LogicalResult intent::triton::plan::verifyTritonRealization(
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
  llvm::DenseSet<int64_t> surfaceOperations;
  for (Operation &operation : realization.getBody().front()) {
    if (auto value = dyn_cast<intent::plan::ProgramOp>(operation))
      machineProgram = value;
    else if (auto value = dyn_cast<intent::plan::AxisOp>(operation))
      machineAxes.insert(value.getNode());
    else if (auto value = dyn_cast<intent::plan::TransferOp>(operation))
      transfers[value.getNode()] = value;
    if (operation.getName().getDialectNamespace() != "intent_triton")
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
    } else if (isa<StorageOp, StreamOp, RaggedOp, StageOp>(operation)) {
    } else if (auto value = dyn_cast<ReductionOp>(operation))
      surfaceOperations.insert(value.getNode());
    else if (auto value = dyn_cast<PointwiseOp>(operation))
      surfaceOperations.insert(value.getNode());
    else if (auto value = dyn_cast<ContractOp>(operation))
      surfaceOperations.insert(value.getNode());
    else if (auto value = dyn_cast<AtomicOp>(operation))
      surfaceOperations.insert(value.getNode());
    else if (auto value = dyn_cast<BoundaryOp>(operation)) {
      auto transfer = transfers.find(value.getNode());
      if (transfer == transfers.end() ||
          transfer->second.getDefer() != value.getDefer())
        return value.emitOpError("does not project its machine transfer decision");
    } else
      return operation.emitOpError("is not legal in a Triton projection");
  }
  if (targets != 1 || capabilities != 1 || programs != 1 || !machineProgram)
    return realization.emitOpError(
        "requires one Triton surface, capability cut, and program projection");
  if (machineAxes.size() != surfaceAxes.size() ||
      llvm::any_of(machineAxes, [&](int64_t node) {
        return !surfaceAxes.contains(node);
      }))
    return realization.emitOpError("Triton axes do not project the GPU axes");
  if (surfaceProgram.getLoopNode() != machineProgram.getLoopNode() ||
      surfaceProgram.getWorkerAxes() != machineProgram.getWorkerAxes() ||
      surfaceProgram.getTraversals() != machineProgram.getTraversals() ||
      surfaceProgram.getOwnership() !=
          projectedOwnership(machineProgram.getOwnership()))
    return surfaceProgram.emitOpError("does not project the GPU program decision");
  return success();
}

LogicalResult intent::triton::plan::verifyTritonSearchSpace(
    intent::plan::SearchSpaceOp searchSpace) {
  if (failed(intent::plan::verifyGpuSearchSpace(searchSpace)))
    return failure();
  auto machine =
      *searchSpace.getBody().front().getOps<intent::plan::AutotuneOp>().begin();
  AutotuneOp surface;
  unsigned count = 0;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (operation.getName().getDialectNamespace() != "intent_triton")
      continue;
    if (auto value = dyn_cast<AutotuneOp>(operation)) {
      surface = value;
      ++count;
    } else
      return operation.emitOpError("is not legal in a Triton search projection");
  }
  if (count != 1 || surface.getKey() != machine.getKey())
    return searchSpace.emitOpError(
        "requires one Triton tuner projection with the GPU specialization keys");
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
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"

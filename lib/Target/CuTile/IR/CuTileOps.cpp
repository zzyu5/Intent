#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace intent::cutile::plan;

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
  if (getArchitecture().empty() || getDeviceAttr().getInt() < 0)
    return emitOpError("requires architecture and device");
  return success();
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())) ||
      getSourceAxisAttr().getInt() < 0)
    return failure();
  if (getRole().empty() || getTile().empty())
    return emitOpError("requires a physical role and tile choice");
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(requireNode(*this, getLoopNodeAttr().getInt())))
    return failure();
  if (getWorkerAxes().empty() || getWorkerAxes().size() > 3)
    return emitOpError("requires between one and three cuTile block axes");
  for (int64_t axis : getWorkerAxes())
    if (axis < 0 || axis > 2)
      return emitOpError("block axes must be in [0, 2]");
  if (getTraversal().empty() || getMapping().empty() ||
      getGroupSizeAttr().getInt() <= 0)
    return emitOpError("requires traversal, mapping and positive group size");
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValueAttr().getInt(),
                                "Kernel IR value ID")))
    return failure();
  if (getSpace() != "global" && getSpace() != "register")
    return emitOpError("contains an unsupported cuTile storage space");
  return success();
}

LogicalResult LayoutOp::verify() {
  if (failed(requireNonNegative(*this, getValueAttr().getInt(),
                                "Kernel IR value ID")) ||
      getOrder().empty())
    return failure();
  llvm::DenseSet<int64_t> dimensions;
  for (int64_t dimension : getOrder())
    if (dimension < 0 || !dimensions.insert(dimension).second)
      return emitOpError("layout order must be a non-negative permutation");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())) ||
      getAxisAttr().getInt() < 0)
    return failure();
  if (getLowering() != "ct.max" && getLowering() != "ct.sum")
    return emitOpError("contains an unsupported cuTile reduction lowering");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())))
    return failure();
  if (getLowering().empty())
    return emitOpError("requires a target lowering name");
  return success();
}

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())))
    return failure();
  if (getLowering() != "ct.mma" || getAccumulatorType() != "f32")
    return emitOpError("requires ct.mma with f32 accumulation");
  return success();
}

LogicalResult StreamOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())) ||
      failed(requireNode(*this, getAxisNodeAttr().getInt())))
    return failure();
  if (getTile().empty() || getOrder() != "forward" ||
      getCarrySpace() != "register")
    return emitOpError(
        "requires a forward tile stream with register-carried state");
  return success();
}

LogicalResult BoundaryOp::verify() {
  if (failed(requireNode(*this, getNodeAttr().getInt())) ||
      getDomainNodes().empty())
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if (getAccess() != "gather" && getAccess() != "scatter" &&
      getAccess() != "load" && getAccess() != "store")
    return emitOpError("contains an unsupported cuTile memory access");
  if (getPadding() != "negative_infinity" && getPadding() != "zero" &&
      getPadding() != "none")
    return emitOpError("contains an unsupported cuTile padding mode");
  if ((getAccess() == "gather" || getAccess() == "load") &&
      getPadding() == "none")
    return emitOpError("load-like access requires a padding choice");
  if ((getAccess() == "scatter" || getAccess() == "store") &&
      getPadding() != "none")
    return emitOpError("store-like access cannot carry load padding");
  return success();
}

LogicalResult LaunchOp::verify() {
  if (failed(requireNode(*this, getLoopNodeAttr().getInt())) ||
      getOccupancyAttr().getInt() <= 0)
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
  if (getParameters().empty() || getNumCtas() <= 0 || getOccupancy() <= 0)
    return emitOpError("requires positive cuTile candidate parameters and hints");
  for (NamedAttribute parameter : getParameters()) {
    auto value = dyn_cast<IntegerAttr>(parameter.getValue());
    if (!value || value.getInt() <= 0)
      return emitOpError("cuTile config parameters must be positive integers");
  }
  return success();
}

LogicalResult intent::cutile::plan::verifyCuTileRealization(
    intent::plan::RealizationOp realization) {
  if (realization.getTarget() != "cutile")
    return realization.emitOpError("is not a cuTile realization");
  unsigned targets = 0;
  unsigned programs = 0;
  unsigned launches = 0;
  ProgramOp programChoice;
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
    if (operation.getName().getDialectNamespace() != "intent_cutile")
      return operation.emitOpError("is not legal in a cuTile realization");
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
    } else if (auto binding = dyn_cast<BoundaryOp>(operation)) {
      if (!boundaries.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a boundary binding");
    } else if (auto launch = dyn_cast<LaunchOp>(operation)) {
      ++launches;
      launchChoice = launch;
    } else
      return operation.emitOpError(
          "is not a recognized resolved cuTile choice");
  }
  if (targets != 1 || programs != 1 || launches > 1)
    return realization.emitOpError(
        "requires one target/program and at most one fixed launch");
  if (launchChoice && launchChoice.getLoopNode() != programChoice.getLoopNode())
    return launchChoice.emitOpError("does not bind the resolved program root");
  if (programChoice.getMapping() == "persistent_rows" && !launchChoice)
    return realization.emitOpError("persistent rows require a fixed launch choice");
  if (programChoice.getMapping() == "grouped_2d_tiles" && launchChoice)
    return realization.emitOpError(
        "autotuned grouped tiles cannot carry a fixed launch choice");
  if (programChoice.getMapping() == "multi_axis_stream" &&
      (launchChoice || !streamChoice))
    return realization.emitOpError(
        "multi-axis streams require one stream and no fixed launch");
  if (streamChoice && !axes.contains(streamChoice.getAxisNode()))
    return streamChoice.emitOpError("references an unbound stream axis");
  if (programChoice.getMapping() != "persistent_rows" &&
      programChoice.getMapping() != "grouped_2d_tiles" &&
      programChoice.getMapping() != "multi_axis_stream")
    return programChoice.emitOpError("contains an unsupported cuTile mapping");
  return success();
}

LogicalResult intent::cutile::plan::verifyCuTileSearchSpace(
    intent::plan::SearchSpaceOp searchSpace) {
  if (searchSpace.getTarget() != "cutile")
    return searchSpace.emitOpError("is not a cuTile search space");
  unsigned autotune = 0;
  unsigned configs = 0;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() != "intent_cutile")
      return operation.emitOpError("is not legal in a cuTile search space");
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
#include "Intent/Target/CuTile/IR/CuTileOps.cpp.inc"

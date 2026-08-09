#include "Intent/Dialect/Plan/IR/PlanOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace intent::plan;

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

LogicalResult verifyEnvelope(Operation *operation, FlatSymbolRefAttr entry,
                             StringRef target, Region &body) {
  if (target.empty())
    return operation->emitOpError("requires a non-empty target name");
  auto module = operation->getParentOfType<ModuleOp>();
  if (!module || operation->getParentOp() != module)
    return operation->emitOpError("must be nested directly in an MLIR module");
  auto function = module.lookupSymbol<func::FuncOp>(entry.getValue());
  auto kind = function ? function->getAttrOfType<StringAttr>("intent.kind")
                       : StringAttr();
  if (!function || !kind || kind.getValue() != "kernel")
    return operation->emitOpError("entry must reference an Intent kernel function");
  if (!llvm::hasSingleElement(body))
    return operation->emitOpError("requires exactly one body block");
  return success();
}

bool isPrivateSpace(StringRef space) {
  return space == "private_scalar" || space == "private_fragment";
}

LogicalResult verifyStringArray(Operation *operation, ArrayAttr values,
                                StringRef label) {
  if (values.empty())
    return operation->emitOpError() << "requires at least one " << label;
  llvm::StringSet<> unique;
  for (Attribute attribute : values) {
    auto value = dyn_cast<StringAttr>(attribute);
    if (!value || value.getValue().empty() ||
        !unique.insert(value.getValue()).second)
      return operation->emitOpError()
             << label << " values must be unique non-empty strings";
  }
  return success();
}

bool hasString(ArrayAttr values, StringRef expected) {
  return llvm::any_of(values, [&](Attribute attribute) {
    auto value = dyn_cast<StringAttr>(attribute);
    return value && value.getValue() == expected;
  });
}

} // namespace

LogicalResult RealizationOp::verify() {
  return verifyEnvelope(*this, getEntryAttr(), getTarget(), getBody());
}

LogicalResult SearchSpaceOp::verify() {
  return verifyEnvelope(*this, getEntryAttr(), getTarget(), getBody());
}

LogicalResult DeviceOp::verify() {
  if (getDeviceAttr().getInt() < 0 || getComputeUnits() <= 0 ||
      getSharedMemoryPerUnit() <= 0 || getRegistersPerUnit() <= 0)
    return emitOpError("requires a device and positive algorithm-visible capacities");
  return success();
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      getSourceAxisAttr().getInt() < 0)
    return failure();
  if (getRole().empty() || getTile().empty())
    return emitOpError("requires a physical role and canonical tile role");
  return success();
}

LogicalResult ProgramOp::verify() {
  if (failed(requireNode(*this, getLoopNode())) || getWorkerAxes().empty() ||
      getWorkerAxes().size() > 3)
    return failure();
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : getWorkerAxes())
    if (axis < 0 || axis > 2 || !axes.insert(axis).second)
      return emitOpError("worker axes must be unique values in [0, 2]");
  if (getOwnership().empty() ||
      failed(verifyStringArray(*this, getTraversals(),
                               "program traversal component")))
    return failure();
  return success();
}

LogicalResult StorageOp::verify() {
  if (failed(requireNonNegative(*this, getValue(), "Kernel IR value ID")))
    return failure();
  if (getSpace() != "external" && getSpace() != "workspace")
    return emitOpError("contains an unsupported machine storage class");
  return success();
}

LogicalResult TransferOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if (getAccess() != "load" && getAccess() != "store")
    return emitOpError("requires a logical load or store access");
  if (getFill() != "negative_infinity" && getFill() != "zero" &&
      getFill() != "none")
    return emitOpError("contains an unsupported boundary fill");
  if (getMaterialization() != "direct" &&
      getMaterialization() != "contract_operand")
    return emitOpError("contains an unsupported materialization role");
  if (getResultSpace() != "none" && getResultSpace() != "shared" &&
      !isPrivateSpace(getResultSpace()))
    return emitOpError("contains an unsupported result residency");
  if (getAccess() == "load" && getFill() == "none" &&
      !getDomainNodes().empty())
    return emitOpError("bounded loads require an explicit fill");
  if (getAccess() == "store" && getFill() != "none")
    return emitOpError("stores cannot carry a load fill");
  if (getDefer() && (getAccess() != "load" ||
                     getMaterialization() != "contract_operand"))
    return emitOpError("only contraction-operand loads may be deferred");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNode())) || getRole().empty() ||
      getAxisAttr().getInt() < 0)
    return failure();
  if (!isPrivateSpace(getInputSpace()) || !isPrivateSpace(getResultSpace()))
    return emitOpError("requires private reduction input and result residency");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())) || getRole().empty() ||
      getReuseOperandAttr().getInt() < -1)
    return failure();
  if (!isPrivateSpace(getResultSpace()))
    return emitOpError("requires private pointwise result residency");
  if (getMaterialization() != "elementwise" &&
      getMaterialization() != "contract_operand")
    return emitOpError("contains an unsupported pointwise materialization");
  if (getDefer() && getMaterialization() != "contract_operand")
    return emitOpError("only contraction operands may be deferred");
  return success();
}

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getPrimitive() != "matrix_multiply" || getAccumulatorType() != "f32")
    return emitOpError("requires matrix multiplication with f32 accumulation");
  if (getWarpPolicy() != "square" && getWarpPolicy() != "full_row")
    return emitOpError("contains an unsupported matrix-unit warp policy");
  auto validOperand = [](StringRef space) {
    return space == "shared" || space == "private_fragment";
  };
  if (!validOperand(getLhsSpace()) || !validOperand(getRhsSpace()) ||
      getAccumulatorSpace() != "private_fragment")
    return emitOpError("contains an invalid matrix operand residency");
  return success();
}

LogicalResult StreamOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getAxisNode())) || getTile().empty())
    return failure();
  if (getOrder() != "forward" || getCarrySpace() != "private_fragment" ||
      !getReuseInitial())
    return emitOpError("requires forward traversal with reused private state");
  return success();
}

LogicalResult RaggedOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getOuterNode())) || getMemberNodes().empty())
    return failure();
  llvm::DenseSet<int64_t> members;
  for (int64_t member : getMemberNodes())
    if (failed(requireNode(*this, member)) || !members.insert(member).second)
      return emitOpError("member nodes must be unique Kernel IR domains");
  if (getTraversal() != "expert_offset_ranges" &&
      getTraversal() != "compact_offset_tiles")
    return emitOpError("contains an unsupported ragged traversal");
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
  if (getCombine() != "add" || getMemoryOrder() != "relaxed" ||
      getMemoryScope() != "device")
    return emitOpError("requires a relaxed device-scoped additive merge");
  return success();
}

LogicalResult AutotuneOp::verify() {
  if (failed(verifyStringArray(*this, getKey(), "specialization key")) ||
      failed(verifyStringArray(*this, getParameters(), "tunable parameter")))
    return failure();
  return success();
}

LogicalResult intent::plan::verifyGpuRealization(RealizationOp realization) {
  if (realization.getTarget() != "gpu")
    return realization.emitOpError("is not a GPU machine realization");
  unsigned devices = 0;
  unsigned programs = 0;
  ProgramOp program;
  llvm::DenseSet<int64_t> streams;
  llvm::DenseSet<int64_t> ragged;
  llvm::DenseSet<int64_t> axes;
  llvm::StringSet<> roles;
  llvm::DenseSet<int64_t> storage;
  llvm::DenseSet<int64_t> operations;
  llvm::DenseSet<int64_t> stageOrdinals;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<YieldOp>(operation) ||
        operation.getName().getDialectNamespace() != "intent_plan")
      continue;
    if (isa<DeviceOp>(operation))
      ++devices;
    else if (auto axis = dyn_cast<AxisOp>(operation)) {
      if (!axes.insert(axis.getNode()).second ||
          !roles.insert(axis.getRole()).second)
        return axis.emitOpError("duplicates a machine axis node or role");
    } else if (auto choice = dyn_cast<ProgramOp>(operation)) {
      ++programs;
      program = choice;
    } else if (auto binding = dyn_cast<StorageOp>(operation)) {
      if (!storage.insert(binding.getValue()).second)
        return binding.emitOpError("duplicates a storage binding");
    } else if (auto binding = dyn_cast<TransferOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<ReductionOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<PointwiseOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<ContractOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<AtomicOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<StreamOp>(operation)) {
      if (!streams.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an ordered stream decision");
    } else if (auto binding = dyn_cast<RaggedOp>(operation)) {
      if (!ragged.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a ragged traversal decision");
    } else if (auto binding = dyn_cast<StageOp>(operation)) {
      if (!stageOrdinals.insert(binding.getOrdinal()).second)
        return binding.emitOpError("duplicates a physical stage ordinal");
    } else
      return operation.emitOpError("is not legal inside a GPU realization");
  }
  if (devices != 1 || programs != 1)
    return realization.emitOpError("requires one GPU device and one program mapping");
  if (program.getOwnership() != "row" && program.getOwnership() != "tiled" &&
      program.getOwnership() != "ragged")
    return program.emitOpError("contains an unsupported GPU ownership family");
  for (Attribute attribute : program.getTraversals()) {
    StringRef traversal = cast<StringAttr>(attribute).getValue();
    if (traversal != "persistent" && traversal != "grouped" &&
        traversal != "ordered_stream" && traversal != "staged")
      return program.emitOpError("contains an unsupported traversal component");
  }
  bool ordered = hasString(program.getTraversals(), "ordered_stream");
  bool staged = hasString(program.getTraversals(), "staged");
  if (ordered != !streams.empty())
    return realization.emitOpError(
        "ordered-stream traversal and stream decisions must appear together");
  if (staged != !stageOrdinals.empty())
    return realization.emitOpError(
        "staged traversal and physical stages must appear together");
  if ((program.getOwnership() == "ragged") != !ragged.empty())
    return realization.emitOpError(
        "ragged ownership and ragged relation decisions must appear together");
  if (hasString(program.getTraversals(), "persistent") &&
      program.getOwnership() != "row")
    return program.emitOpError("persistent traversal requires row ownership");
  if (hasString(program.getTraversals(), "grouped") &&
      program.getOwnership() != "tiled")
    return program.emitOpError("grouped traversal requires tiled ownership");
  if (staged && program.getOwnership() != "ragged")
    return program.emitOpError("staged traversal requires ragged ownership");
  for (Operation &operation : realization.getBody().front())
    if (auto stream = dyn_cast<StreamOp>(operation);
        stream && !axes.contains(stream.getAxisNode()))
      return stream.emitOpError("references an unbound stream axis");
  for (Operation &operation : realization.getBody().front())
    if (auto relation = dyn_cast<RaggedOp>(operation))
      for (int64_t member : relation.getMemberNodes())
        if (!axes.contains(member))
          return relation.emitOpError("references an unbound ragged member axis");
  return success();
}

LogicalResult intent::plan::verifyGpuSearchSpace(SearchSpaceOp searchSpace) {
  if (searchSpace.getTarget() != "gpu")
    return searchSpace.emitOpError("is not a GPU machine search space");
  unsigned declarations = 0;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (isa<YieldOp>(operation) ||
        operation.getName().getDialectNamespace() != "intent_plan")
      continue;
    if (isa<AutotuneOp>(operation))
      ++declarations;
    else
      return operation.emitOpError("is not legal inside a GPU search space");
  }
  if (declarations != 1)
    return searchSpace.emitOpError("requires one unresolved autotune declaration");
  return success();
}

#define GET_OP_CLASSES
#include "Intent/Dialect/Plan/IR/PlanOps.cpp.inc"

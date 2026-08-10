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

LogicalResult verifyValidityBinding(Operation *operation, ArrayRef<int64_t> axes,
                                    ArrayRef<int64_t> nodes,
                                    StringRef operand) {
  if (axes.size() != nodes.size())
    return operation->emitOpError()
           << operand << " validity axes and nodes must have equal length";
  llvm::DenseSet<int64_t> uniqueAxes;
  for (auto [axis, node] : llvm::zip(axes, nodes)) {
    if (failed(requireNonNegative(operation, axis, "validity tensor axis")) ||
        failed(requireNode(operation, node)))
      return failure();
    if (!uniqueAxes.insert(axis).second)
      return operation->emitOpError()
             << operand << " validity tensor axes must be unique";
  }
  return success();
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

bool axisHasRole(AxisOp axis, StringRef expected) {
  return llvm::any_of(axis.getRoles(), [&](Attribute attribute) {
    auto role = dyn_cast<StringAttr>(attribute);
    return role && role.getValue() == expected;
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
  return requireNonNegative(*this, getDevice(), "device index");
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (failed(verifyStringArray(*this, getRoles(), "axis role")) ||
      getTile().empty())
    return failure();
  for (Attribute attribute : getRoles()) {
    StringRef role = cast<StringAttr>(attribute).getValue();
    if (role != "parallel" && role != "ordered" && role != "reduction" &&
        role != "ragged_member" && role != "lane")
      return emitOpError() << "contains unsupported axis role " << role;
  }
  bool parallel = axisHasRole(*this, "parallel");
  if (parallel) {
    if (!getProgramOrderAttr() || !getWorkerAxisAttr() || !getFoldOrderAttr())
      return emitOpError(
          "parallel axes require program-order, worker-axis, and fold-order decisions");
    if (getProgramOrderAttr().getInt() < 0 || getWorkerAxisAttr().getInt() < 0 ||
        getWorkerAxisAttr().getInt() > 2 ||
        getFoldOrderAttr().getInt() < 0)
      return emitOpError("contains an invalid program-space assignment");
    if (getGroupAttr() && getGroup()->empty())
      return emitOpError("contains an empty program grouping role");
  } else if (getProgramOrderAttr() || getWorkerAxisAttr() ||
             getFoldOrderAttr() || getReuseWorker() || getGroupAttr()) {
    return emitOpError(
        "non-parallel axes cannot own program-space assignment fields");
  }
  return success();
}

LogicalResult ProgramOp::verify() {
  return requireNode(*this, getLoopNode());
}

LogicalResult intent::plan::verifyPaddingFields(
    Operation *operation, int64_t value, ArrayRef<int64_t> tensorAxes,
    ArrayRef<int64_t> domainNodes, StringRef fill) {
  if (failed(requireNonNegative(operation, value, "Kernel IR value ID")) ||
      tensorAxes.empty() ||
      failed(verifyValidityBinding(operation, tensorAxes, domainNodes, "value")))
    return failure();
  if (fill != "negative_infinity" && fill != "zero")
    return operation->emitOpError(
        "contains an unsupported physical padding value");
  return success();
}

LogicalResult BufferOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getSpace() != "private_scalar_array")
    return emitOpError("requires private scalar-array residency");
  return success();
}

LogicalResult PaddingOp::verify() {
  return verifyPaddingFields(*this, getValue(), getTensorAxes(),
                             getDomainNodes(), getFill());
}

LogicalResult TransferOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t domain : getDomainNodes())
    if (failed(requireNode(*this, domain)))
      return failure();
  if (getFill() != "negative_infinity" && getFill() != "zero" &&
      getFill() != "none")
    return emitOpError("contains an unsupported boundary fill");
  if (getConsumerNeutralized() &&
      (getDomainNodes().empty() || getFill() == "none" ||
       getResultSpace() == "none"))
    return emitOpError(
        "consumer-neutralized transfer requires a bounded load with a fill");
  if (getResultSpace() != "none" && getResultSpace() != "shared" &&
      !isPrivateSpace(getResultSpace()))
    return emitOpError("contains an unsupported result residency");
  return success();
}

LogicalResult ReductionOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (!isPrivateSpace(getResultSpace()))
    return emitOpError("requires private reduction result residency");
  return success();
}

LogicalResult ScanOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (!isPrivateSpace(getResultSpace()))
    return emitOpError("requires private scan result residency");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      getReuseOperandAttr().getInt() < -1)
    return failure();
  if (!isPrivateSpace(getResultSpace()))
    return emitOpError("requires private pointwise result residency");
  return success();
}

LogicalResult ContractOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  auto validOperand = [](StringRef space) {
    return space == "shared" || space == "private_fragment";
  };
  if (!validOperand(getLhsSpace()) || !validOperand(getRhsSpace()) ||
      getAccumulatorSpace() != "private_fragment")
    return emitOpError("contains an invalid matrix operand residency");
  return success();
}

LogicalResult StageOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t value : getInputs())
    if (failed(requireNonNegative(*this, value, "stage input value ID")))
      return failure();
  for (int64_t value : getOutputs())
    if (failed(requireNonNegative(*this, value, "stage output value ID")))
      return failure();
  if (getOperations().empty())
    return emitOpError("requires an explicit physical operation slice");
  llvm::DenseSet<int64_t> operations;
  for (int64_t operation : getOperations())
    if (failed(requireNode(*this, operation)) ||
        !operations.insert(operation).second)
      return emitOpError("stage operation nodes must be unique");
  llvm::DenseSet<int64_t> terminals;
  for (int64_t terminal : getTerminals())
    if (failed(requireNode(*this, terminal)) ||
        !operations.contains(terminal) || !terminals.insert(terminal).second)
      return emitOpError(
          "stage terminals must be unique members of its operation slice");
  return success();
}

LogicalResult StageAxisOp::verify() {
  if (failed(requireNode(*this, getStageNode())) || getRole().empty() ||
      getExtent().empty() || getTile().empty())
    return failure();
  if (getAxisNodeAttr() && failed(requireNode(*this, getAxisNodeAttr().getInt())))
    return failure();
  if (getWorkerAxisAttr() &&
      (getWorkerAxisAttr().getInt() < 0 || getWorkerAxisAttr().getInt() > 2))
    return emitOpError("contains an invalid stage worker axis");
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
  llvm::DenseMap<int64_t, AxisOp> axes;
  llvm::DenseSet<int64_t> programOrders;
  llvm::DenseSet<int64_t> paddedValues;
  llvm::DenseSet<int64_t> buffers;
  SmallVector<PaddingOp> paddings;
  llvm::DenseSet<int64_t> operations;
  llvm::DenseSet<int64_t> stageNodes;
  llvm::StringSet<> stageAxisRoles;
  SmallVector<StageAxisOp> stageAxes;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<YieldOp>(operation) ||
        operation.getName().getDialectNamespace() != "intent_plan")
      continue;
    if (isa<DeviceOp>(operation))
      ++devices;
    else if (auto axis = dyn_cast<AxisOp>(operation)) {
      if (!axes.try_emplace(axis.getNode(), axis).second)
        return axis.emitOpError("duplicates a logical-axis physical decision");
      if (axisHasRole(axis, "parallel") &&
          !programOrders.insert(axis.getProgramOrderAttr().getInt()).second)
        return axis.emitOpError("duplicates a program-axis order");
    } else if (auto choice = dyn_cast<ProgramOp>(operation)) {
      ++programs;
      program = choice;
    } else if (auto binding = dyn_cast<BufferOp>(operation)) {
      if (!buffers.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a logical-buffer decision");
    } else if (auto binding = dyn_cast<PaddingOp>(operation)) {
      if (!paddedValues.insert(binding.getValue()).second)
        return binding.emitOpError("duplicates a value padding decision");
      paddings.push_back(binding);
    } else if (auto binding = dyn_cast<TransferOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<ReductionOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<ScanOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<PointwiseOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<ContractOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<StageOp>(operation)) {
      if (!stageNodes.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a physical stage decision");
    } else if (auto binding = dyn_cast<StageAxisOp>(operation)) {
      std::string key = std::to_string(binding.getStageNode()) + ":" +
                        binding.getRole().str();
      if (!stageAxisRoles.insert(key).second)
        return binding.emitOpError("duplicates a stage-axis role");
      stageAxes.push_back(binding);
    } else
      return operation.emitOpError("is not legal inside a GPU realization");
  }
  if (devices != 1 || programs != 1)
    return realization.emitOpError("requires one GPU device and one program mapping");
  if (programOrders.empty())
    return program.emitOpError("has no per-axis program-space assignment");
  if (program.getPersistent())
    for (const auto &entry : axes) {
      AxisOp axis = entry.second;
      if (!axisHasRole(axis, "parallel"))
        continue;
      if (axis.getWorkerAxis() != 0 || axis.getReuseWorker())
        return axis.emitOpError(
            "persistent program axes must share worker axis zero without nested reuse");
    }
  for (PaddingOp padding : paddings)
    for (int64_t domain : padding.getDomainNodes())
      if (!axes.count(domain))
        return padding.emitOpError("references an unbound validity domain");
  for (StageAxisOp axis : stageAxes) {
    if (!stageNodes.contains(axis.getStageNode()))
      return axis.emitOpError("references an unknown physical stage");
    if (axis.getAxisNodeAttr() && !axes.count(axis.getAxisNodeAttr().getInt()))
      return axis.emitOpError("references an unbound logical axis");
  }
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

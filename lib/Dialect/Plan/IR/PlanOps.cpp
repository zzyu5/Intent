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

LogicalResult verifyTopLevelEnvelope(Operation *operation, StringRef target,
                                     Region &body) {
  if (target.empty())
    return operation->emitOpError("requires a non-empty target name");
  auto module = operation->getParentOfType<ModuleOp>();
  if (!module || operation->getParentOp() != module)
    return operation->emitOpError("must be nested directly in an MLIR module");
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

LogicalResult ProgramOp::verify() {
  if (failed(verifyTopLevelEnvelope(*this, getTarget(), getBody())))
    return failure();
  for (Operation &operation : getBody().front())
    if (!isa<YieldOp, func::FuncOp>(operation) &&
        operation.getName().getDialectNamespace() != "intent_plan")
      return operation.emitOpError("is not legal inside a physical program");
  FailureOr<func::FuncOp> entry = getPhysicalEntry(*this);
  if (failed(entry))
    return failure();
  if ((*entry).getName() != getEntry())
    return emitOpError("physical entry name does not match the program entry");
  auto kind = (*entry)->getAttrOfType<StringAttr>("intent.kind");
  if (!kind || kind.getValue() != "physical")
    return emitOpError("must own one physical Intent function");
  return success();
}

LogicalResult SearchSpaceOp::verify() {
  if (failed(verifyTopLevelEnvelope(*this, getTarget(), getBody())))
    return failure();
  auto module = getOperation()->getParentOfType<ModuleOp>();
  unsigned matches = 0;
  for (ProgramOp program : module.getOps<ProgramOp>())
    matches += program.getEntry() == getEntry() &&
               program.getTarget() == getTarget();
  if (matches != 1)
    return emitOpError("must reference exactly one physical program");
  return success();
}

LogicalResult TargetProgramOp::verify() {
  if (getProvider() != "triton" && getProvider() != "cutile" &&
      getProvider() != "tilelang")
    return emitOpError("contains an unknown provider");
  if (getSource().empty())
    return emitOpError("requires a non-empty materialized target program");
  if (!getOperation()->getParentOfType<ProgramOp>())
    return emitOpError("must be nested in an Intent physical program");
  return success();
}

LogicalResult DeviceOp::verify() {
  return requireNonNegative(*this, getDevice(), "device index");
}

LogicalResult AxisOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (failed(verifyStringArray(*this, getRoles(), "axis role")))
    return failure();
  for (Attribute attribute : getRoles()) {
    StringRef role = cast<StringAttr>(attribute).getValue();
    if (role != "parallel" && role != "ordered" && role != "reduction" &&
        role != "ragged_member" && role != "lane" &&
        role != "packed_lane" && role != "contraction_m" &&
        role != "contraction_n")
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

LogicalResult RangeOp::verify() {
  if (failed(requireNode(*this, getAxisNode())) ||
      failed(requireNonNegative(*this, getLevel(), "range level")) ||
      getPurpose().empty() || getTile().empty() || getExtent().empty())
    return failure();
  if (!llvm::is_contained(
          {StringRef("ownership"), StringRef("traversal"),
           StringRef("reduction"), StringRef("lane"),
           StringRef("access")},
          getPurpose()))
    return emitOpError() << "contains unsupported range purpose " << getPurpose();
  if (getPurpose() != "traversal" && getLevel() != 0)
    return emitOpError("only ordered traversal may contain nested range levels");
  if (getPurpose() == "access") {
    if (!getTransferNodeAttr() || !getSourceAxisAttr() ||
        failed(requireNode(*this, getTransferNodeAttr().getInt())) ||
        failed(requireNonNegative(*this, getSourceAxisAttr().getInt(),
                                  "source view axis")))
      return emitOpError(
          "access ranges require a transfer node and source view axis");
    if (getDivisorAttr() && getDivisorAttr().getInt() <= 1)
      return emitOpError("compact access divisor must be greater than one");
    if (getOffsetAttr() && getOffsetAttr().getInt() < 0)
      return emitOpError("access offset must be non-negative");
    if (getDivisorAttr() && !getOffsetAttr())
      return emitOpError(
          "compact access divisor requires an offset");
  } else if (getTransferNodeAttr() || getSourceAxisAttr() ||
             getDivisorAttr() || getOffsetAttr()) {
    return emitOpError(
        "non-access ranges cannot carry transfer-relative bindings");
  }
  return success();
}

LogicalResult RegionBindingOp::verify() {
  if (failed(requireNonNegative(*this, getArgument(),
                                "region argument value ID")) ||
      failed(requireNode(*this, getAxisNode())) ||
      failed(requireNonNegative(*this, getLevel(), "range level")))
    return failure();
  if (getPurpose() != "ownership" && getPurpose() != "traversal")
    return emitOpError(
        "region arguments may only bind ownership or traversal ranges");
  return success();
}

LogicalResult LaunchOp::verify() {
  return requireNode(*this, getLoopNode());
}

LogicalResult BlockExtentOp::verify() {
  if (getLogicalExtent().empty())
    return emitOpError("requires a logical extent");
  if (getRounding() != "power_of_two")
    return emitOpError("contains an unsupported block-extent rounding rule");
  if (getFill() != "zero")
    return emitOpError("contains an unsupported block-extent fill rule");
  return success();
}

LogicalResult intent::plan::verifyPaddingFields(
    Operation *operation, int64_t value, ArrayRef<int64_t> tensorAxes,
    ArrayRef<int64_t> domainNodes, StringRef fill) {
  if (failed(requireNonNegative(operation, value, "Kernel IR value ID")) ||
      tensorAxes.empty() ||
      failed(verifyValidityBinding(operation, tensorAxes, domainNodes, "value")))
    return failure();
  bool literal = fill.starts_with("literal_integer:") ||
                 fill.starts_with("literal_float:");
  if (fill != "negative_infinity" && fill != "positive_infinity" &&
      fill != "nan" && fill != "zero" && fill != "false" &&
      fill != "true" && !literal)
    return operation->emitOpError(
        "contains an unsupported physical padding value");
  if (literal && fill.split(':').second.empty())
    return operation->emitOpError("contains an empty literal padding value");
  return success();
}

LogicalResult BufferOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getSpace() != "private_scalar_array" &&
      getSpace() != "private_vector" && getSpace() != "private_workspace")
    return emitOpError("has an unsupported logical-buffer residency");
  if (getSpace() == "private_workspace" && getOwnerNodes().empty())
    return emitOpError("private workspace requires explicit program owners");
  if (getSpace() != "private_workspace" && !getOwnerNodes().empty())
    return emitOpError("kernel-local buffer cannot carry workspace owners");
  for (int64_t owner : getOwnerNodes())
    if (failed(requireNode(*this, owner)))
      return failure();
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
  if (failed(verifyValidityBinding(*this, getValidityTensorAxes(),
                                   getValidityDomainNodes(), "transfer")))
    return failure();
  for (int64_t domain : getValidityDomainNodes())
    if (!llvm::is_contained(getDomainNodes(), domain))
      return emitOpError(
          "validity domain is not part of the transfer boundary");
  if (getFill() != "negative_infinity" && getFill() != "zero" &&
      getFill() != "none")
    return emitOpError("contains an unsupported boundary fill");
  if (getConsumerNeutralized() &&
      (getDomainNodes().empty() || getFill() == "none" ||
       getResultSpace() == "none"))
    return emitOpError(
        "consumer-neutralized transfer requires a bounded load with a fill");
  if (getMaterialization() != "direct" &&
      getMaterialization() != "deferred_to_contract")
    return emitOpError("contains an unsupported transfer materialization");
  if (getTensorIndexing() != "none" && getTensorIndexing() != "structured" &&
      getTensorIndexing() != "compact" &&
      getTensorIndexing() != "data_dependent")
    return emitOpError("contains an unsupported tensor-indexing class");
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
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getAxisNode())))
    return failure();
  if (((getMaterialization() == "scalar_access" &&
        getResultSpace() != "private_workspace") ||
       (getMaterialization() == "fragment_access" &&
        getResultSpace() != "private_fragment")) ||
      getCarrySpace() != "private_scalar" ||
      (getMaterialization() != "scalar_access" &&
       getMaterialization() != "fragment_access"))
    return emitOpError(
        "requires owner-private materialization and a scalar carry");
  if (getOwnerNodes().empty())
    return emitOpError("requires physical owner axes");
  if (getMaterialization() == "scalar_access" && getProducers().empty())
    return emitOpError(
        "workspace-backed scan requires a non-empty producer slice");
  if (getMaterialization() == "fragment_access" &&
      (!getProducers().empty() || !getMaterializedValues().empty()))
    return emitOpError(
        "fragment scan cannot carry a replay slice or materialized values");
  for (int64_t owner : getOwnerNodes())
    if (failed(requireNode(*this, owner)))
      return failure();
  llvm::DenseSet<int64_t> producers;
  for (int64_t producer : getProducers())
    if (failed(requireNode(*this, producer)) ||
        !producers.insert(producer).second)
      return emitOpError("scan producer nodes must be unique");
  llvm::DenseSet<int64_t> materialized;
  for (int64_t value : getMaterializedValues())
    if (failed(requireNonNegative(*this, value, "Kernel IR value ID")) ||
        !materialized.insert(value).second)
      return emitOpError("scan materialized values must be unique");
  return success();
}

LogicalResult PointwiseOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  for (int64_t axis : getAxisNodes())
    if (axis < -1)
      return emitOpError("contains an invalid pointwise result-axis binding");
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

LogicalResult SparseContractOp::verify() {
  if (failed(requireNode(*this, getNode())) ||
      failed(requireNode(*this, getRowAxisNode())) ||
      failed(requireNode(*this, getColumnAxisNode())) ||
      failed(requireNode(*this, getReductionAxisNode())) ||
      getFormat() != "two_of_four")
    return failure();
  if (getCompressedSpace() != "shared" || getMetadataSpace() != "shared" ||
      getRhsSpace() != "shared" || getAccumulatorSpace() != "private_fragment")
    return emitOpError("contains an invalid 2:4 sparse matrix residency");
  return success();
}

LogicalResult StreamAxisOp::verify() {
  if (failed(requireNode(*this, getStreamNode())) ||
      failed(requireNode(*this, getAxisNode())))
    return failure();
  return success();
}

LogicalResult StreamBindingOp::verify() {
  if (failed(requireNode(*this, getStreamNode())) ||
      failed(requireNode(*this, getAxisNode())) ||
      failed(requireNonNegative(*this, getLevel(), "stream range level")))
    return failure();
  if (getPurpose() != "traversal")
    return emitOpError("state streams require a traversal range binding");
  if (getRelationNodeAttr() &&
      failed(requireNode(*this, getRelationNodeAttr().getInt())))
    return failure();
  return success();
}

LogicalResult StageOp::verify() {
  if (failed(requireNode(*this, getNode())))
    return failure();
  if (getOperations().empty())
    return emitOpError("requires an explicit physical operation slice");
  llvm::DenseSet<int64_t> operations;
  for (int64_t operation : getOperations())
    if (failed(requireNode(*this, operation)) ||
        !operations.insert(operation).second)
      return emitOpError("stage operation nodes must be unique");
  if (getSynchronization() != "same_stream")
    return emitOpError(
        "contains an unsupported stage synchronization contract");
  return success();
}

LogicalResult StageAxisOp::verify() {
  if (failed(requireNode(*this, getStageNode())) || getRole().empty() ||
      getTile().empty())
    return failure();
  if (getAxisNodeAttr() && failed(requireNode(*this, getAxisNodeAttr().getInt())))
    return failure();
  if (getSourceValueAttr() &&
      failed(requireNonNegative(*this, getSourceValueAttr().getInt(),
                                "stage-axis source value ID")))
    return failure();
  if (getTensorAxisAttr() &&
      failed(requireNonNegative(*this, getTensorAxisAttr().getInt(),
                                "stage-axis tensor dimension")))
    return failure();
  bool domainAxis = static_cast<bool>(getAxisNodeAttr());
  bool tensorAxis = static_cast<bool>(getSourceValueAttr()) &&
                    static_cast<bool>(getTensorAxisAttr());
  if (domainAxis == tensorAxis ||
      static_cast<bool>(getSourceValueAttr()) !=
          static_cast<bool>(getTensorAxisAttr()))
    return emitOpError(
        "must bind exactly one logical domain axis or value tensor dimension");
  if ((getRole() == "member" && !domainAxis) ||
      ((getRole() == "feature" || getRole() == "reduction") && !tensorAxis) ||
      (getRole() != "member" && getRole() != "feature" &&
       getRole() != "reduction"))
    return emitOpError("contains an unsupported stage-axis role binding");
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

FailureOr<func::FuncOp> intent::plan::getPhysicalEntry(ProgramOp program) {
  func::FuncOp entry;
  for (Operation &operation : program.getBody().front()) {
    auto function = dyn_cast<func::FuncOp>(operation);
    if (!function)
      continue;
    if (entry)
      return program.emitOpError("owns more than one physical function");
    entry = function;
  }
  if (!entry)
    return program.emitOpError("does not own a physical function");
  return entry;
}

FailureOr<TargetProgramOp>
intent::plan::getTargetProgram(ProgramOp program, StringRef provider) {
  TargetProgramOp result;
  for (TargetProgramOp candidate : program.getBody().getOps<TargetProgramOp>()) {
    if (candidate.getProvider() != provider)
      continue;
    if (result)
      return candidate.emitOpError(
          "duplicates the materialized target program for this provider");
    result = candidate;
  }
  if (!result)
    return program.emitOpError()
           << "has no materialized target program for " << provider;
  return result;
}

LogicalResult intent::plan::verifyGpuProgram(ProgramOp program) {
  if (program.getTarget() != "gpu")
    return program.emitOpError("is not a GPU physical program");
  unsigned devices = 0;
  unsigned launches = 0;
  unsigned targetPrograms = 0;
  LaunchOp launch;
  llvm::StringSet<> blockExtents;
  llvm::DenseMap<int64_t, AxisOp> axes;
  llvm::StringSet<> rangeKeys;
  SmallVector<RangeOp> ranges;
  llvm::DenseSet<int64_t> regionArguments;
  SmallVector<RegionBindingOp> regionBindings;
  SmallVector<ScanOp> scans;
  SmallVector<PointwiseOp> pointwise;
  llvm::DenseSet<int64_t> programOrders;
  llvm::DenseSet<int64_t> paddedValues;
  llvm::DenseSet<int64_t> buffers;
  SmallVector<BufferOp> bufferBindings;
  SmallVector<PaddingOp> paddings;
  llvm::DenseSet<int64_t> operations;
  llvm::DenseSet<int64_t> transfers;
  llvm::DenseSet<int64_t> stageNodes;
  llvm::StringSet<> streamAxisRoles;
  SmallVector<StreamAxisOp> streamAxes;
  llvm::DenseSet<int64_t> streamNodes;
  SmallVector<StreamBindingOp> streamBindings;
  llvm::StringSet<> stageAxisRoles;
  SmallVector<StageAxisOp> stageAxes;
  for (Operation &operation : program.getBody().front()) {
    if (isa<YieldOp, func::FuncOp>(operation))
      continue;
    if (operation.getName().getDialectNamespace() != "intent_plan")
      return operation.emitOpError("is not legal inside a GPU physical program");
    if (isa<DeviceOp>(operation))
      ++devices;
    else if (isa<TargetProgramOp>(operation))
      ++targetPrograms;
    else if (auto axis = dyn_cast<AxisOp>(operation)) {
      if (!axes.try_emplace(axis.getNode(), axis).second)
        return axis.emitOpError("duplicates a logical-axis physical decision");
      if (axisHasRole(axis, "parallel") &&
          !programOrders.insert(axis.getProgramOrderAttr().getInt()).second)
        return axis.emitOpError("duplicates a program-axis order");
    } else if (auto range = dyn_cast<RangeOp>(operation)) {
      std::string key = std::to_string(range.getAxisNode()) + ":" +
                        range.getPurpose().str() + ":" +
                        std::to_string(range.getLevel());
      if (range.getPurpose() == "access")
        key += ":" + std::to_string(range.getTransferNodeAttr().getInt()) +
               ":" + std::to_string(range.getSourceAxisAttr().getInt());
      if (!rangeKeys.insert(key).second)
        return range.emitOpError("duplicates an axis physical range");
      ranges.push_back(range);
    } else if (auto binding = dyn_cast<RegionBindingOp>(operation)) {
      if (!regionArguments.insert(binding.getArgument()).second)
        return binding.emitOpError(
            "duplicates a region-argument physical binding");
      regionBindings.push_back(binding);
    } else if (auto choice = dyn_cast<LaunchOp>(operation)) {
      ++launches;
      launch = choice;
    } else if (auto extent = dyn_cast<BlockExtentOp>(operation)) {
      if (!blockExtents.insert(extent.getLogicalExtent()).second)
        return extent.emitOpError("duplicates a logical block-extent decision");
    } else if (auto binding = dyn_cast<BufferOp>(operation)) {
      if (!buffers.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates a logical-buffer decision");
      bufferBindings.push_back(binding);
      for (int64_t owner : binding.getOwnerNodes()) {
        auto axis = axes.find(owner);
        if (axis == axes.end() || !axisHasRole(axis->second, "parallel"))
          return binding.emitOpError(
              "workspace owner is not a program ownership axis");
      }
    } else if (auto binding = dyn_cast<PaddingOp>(operation)) {
      if (!paddedValues.insert(binding.getValue()).second)
        return binding.emitOpError("duplicates a value padding decision");
      paddings.push_back(binding);
    } else if (auto binding = dyn_cast<TransferOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
      transfers.insert(binding.getNode());
    } else if (auto binding = dyn_cast<ReductionOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<ScanOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
      scans.push_back(binding);
    } else if (auto binding = dyn_cast<PointwiseOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
      pointwise.push_back(binding);
    } else if (auto binding = dyn_cast<ContractOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
    } else if (auto binding = dyn_cast<SparseContractOp>(operation)) {
      if (!operations.insert(binding.getNode()).second)
        return binding.emitOpError("duplicates an operation decision");
      if (!axes.count(binding.getRowAxisNode()) ||
          !axes.count(binding.getColumnAxisNode()) ||
          !axes.count(binding.getReductionAxisNode()))
        return binding.emitOpError(
            "references an unbound sparse-contraction axis");
    } else if (auto binding = dyn_cast<StreamAxisOp>(operation)) {
      std::string key = std::to_string(binding.getStreamNode()) + ":" +
                        std::to_string(binding.getAxisNode());
      if (!streamAxisRoles.insert(key).second)
        return binding.emitOpError("duplicates a stream-axis relation");
      streamAxes.push_back(binding);
    } else if (auto binding = dyn_cast<StreamBindingOp>(operation)) {
      if (!streamNodes.insert(binding.getStreamNode()).second)
        return binding.emitOpError("duplicates a state-stream physical binding");
      streamBindings.push_back(binding);
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
      return operation.emitOpError("is not legal inside a GPU physical program");
  }
  if (devices != 1 || launches != 1)
    return program.emitOpError("requires one GPU device and one launch mapping");
  if (programOrders.empty())
    return launch.emitOpError("has no per-axis program-space assignment");
  if (targetPrograms > 1)
    return program.emitOpError(
        "cannot contain more than one materialized provider program");
  for (RangeOp range : ranges) {
    auto axis = axes.find(range.getAxisNode());
    if (axis == axes.end())
      return range.emitOpError("references an unbound logical axis");
    StringRef purpose = range.getPurpose();
    if (purpose == "access") {
      bool compactStream = range.getDivisorAttr() &&
                           (axisHasRole(axis->second, "ordered") ||
                            axisHasRole(axis->second, "reduction"));
      bool translatedLane = range.getOffsetAttr() &&
                            !range.getDivisorAttr() &&
                            axisHasRole(axis->second, "lane");
      if (!axisHasRole(axis->second, "parallel") && !compactStream &&
          !translatedLane)
        return range.emitOpError(
            "access footprint has no compatible physical source axis");
      if (!transfers.contains(range.getTransferNodeAttr().getInt()))
        return range.emitOpError("references an unbound transfer operation");
      continue;
    }
    StringRef requiredRole = purpose == "ownership" ? "parallel"
                             : purpose == "traversal" ? "ordered"
                             : purpose == "reduction" ? "reduction"
                                                        : "lane";
    if (!axisHasRole(axis->second, requiredRole))
      return range.emitOpError()
             << purpose << " range is not backed by axis role " << requiredRole;
  }
  auto hasRange = [&](int64_t node, StringRef purpose, int64_t level = 0) {
    std::string key = std::to_string(node) + ":" + purpose.str() + ":" +
                      std::to_string(level);
    return rangeKeys.contains(key);
  };
  for (RegionBindingOp binding : regionBindings) {
    if (!axes.count(binding.getAxisNode()) ||
        !hasRange(binding.getAxisNode(), binding.getPurpose(),
                  binding.getLevel()))
      return binding.emitOpError(
          "references an unbound selected physical range");
  }
  for (StreamBindingOp binding : streamBindings) {
    auto axis = axes.find(binding.getAxisNode());
    if (axis == axes.end() || !axisHasRole(axis->second, "ordered") ||
        !hasRange(binding.getAxisNode(), binding.getPurpose(),
                  binding.getLevel()))
      return binding.emitOpError(
          "references an unbound ordered traversal range");
  }
  auto isScalarUnreusedProgramOwner = [&](int64_t node) {
    auto axis = axes.find(node);
    if (axis == axes.end() || !axisHasRole(axis->second, "parallel") ||
        axis->second.getReuseWorker() || axis->second.getGroupAttr())
      return false;
    return llvm::any_of(ranges, [&](RangeOp range) {
      return static_cast<int64_t>(range.getAxisNode()) == node &&
             range.getPurpose() == "ownership" &&
             range.getLevel() == 0 && range.getTile() == "one";
    });
  };
  for (BufferOp buffer : bufferBindings) {
    if (buffer.getSpace() != "private_workspace")
      continue;
    if (launch.getPersistent())
      return buffer.emitOpError(
          "owner-private workspace cannot outlive a persistent program mapping");
    if (!stageNodes.empty())
      return buffer.emitOpError(
          "owner-private workspace cannot cross physical stage boundaries");
    for (int64_t owner : buffer.getOwnerNodes())
      if (!isScalarUnreusedProgramOwner(owner))
        return buffer.emitOpError(
            "private workspace requires scalar unreused program owners");
  }
  for (ScanOp scan : scans) {
    auto axis = axes.find(scan.getAxisNode());
    bool workspaceScan = scan.getResultSpace() == "private_workspace";
    if (workspaceScan && launch.getPersistent())
      return scan.emitOpError(
          "scan workspace cannot outlive a persistent program mapping");
    if (workspaceScan && !stageNodes.empty())
      return scan.emitOpError(
          "scan workspace cannot cross physical stage boundaries");
    if (axis == axes.end() ||
        (workspaceScan && (!axisHasRole(axis->second, "ordered") ||
                           !hasRange(scan.getAxisNode(), "traversal"))))
      return scan.emitOpError("requires an ordered traversal range");
    for (int64_t owner : scan.getOwnerNodes()) {
      auto ownerAxis = axes.find(owner);
      if (ownerAxis == axes.end() ||
          !axisHasRole(ownerAxis->second, "parallel"))
        return scan.emitOpError("references a non-program scan owner");
      if (workspaceScan && !isScalarUnreusedProgramOwner(owner))
        return scan.emitOpError(
            "scan workspace requires scalar unreused program owners");
    }
  }
  for (PointwiseOp binding : pointwise)
    for (int64_t axis : binding.getAxisNodes())
      if (axis >= 0 && !axes.count(axis))
        return binding.emitOpError(
            "references an unbound pointwise result axis");
  for (const auto &entry : axes) {
    AxisOp axis = entry.second;
    for (auto [role, purpose] :
         {std::pair<StringRef, StringRef>("parallel", "ownership"),
          std::pair<StringRef, StringRef>("ordered", "traversal"),
          std::pair<StringRef, StringRef>("reduction", "reduction"),
          std::pair<StringRef, StringRef>("lane", "lane")})
      if (axisHasRole(axis, role) && !hasRange(axis.getNode(), purpose))
        return axis.emitOpError() << "has no " << purpose
                                  << " physical range for role " << role;
  }
  for (RangeOp range : ranges) {
    if (range.getPurpose() != "traversal" || range.getLevel() == 0)
      continue;
    if (range.getLevel() != 1 ||
        !hasRange(range.getAxisNode(), "traversal", 0) ||
        range.getTile() != "one")
      return range.emitOpError(
          "nested ordered traversal requires one scalar level after level zero");
  }
  if (launch.getPersistent())
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
  for (StreamAxisOp relation : streamAxes) {
    auto axis = axes.find(relation.getAxisNode());
    if (axis == axes.end() || !axisHasRole(axis->second, "reduction") ||
        !hasRange(relation.getAxisNode(), "reduction"))
      return relation.emitOpError(
          "requires a planned inner reduction axis and range");
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

#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Emission/Combiner.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace mlir;

namespace intent::tilelang::emission {
namespace {

bool workerReuse(const RealizationIndex &index) {
  return llvm::any_of(index.axesByRole, [](const auto &binding) {
    return binding.getValue().getReuseWorker();
  });
}

bool feedsAtomicValue(Operation &operation) {
  if (operation.getNumResults() != 1 ||
      !llvm::hasSingleElement(operation.getResult(0).getUsers()))
    return false;
  Operation *user = *operation.getResult(0).user_begin();
  auto valueIndex =
      user->getAttrOfType<IntegerAttr>("intent.value_operand_index");
  return user->getName().getStringRef() == "intent.atomic_add" && valueIndex &&
         valueIndex.getInt() >= 0 &&
         static_cast<unsigned>(valueIndex.getInt()) < user->getNumOperands() &&
         user->getOperand(valueIndex.getInt()) == operation.getResult(0);
}

bool isRankReducingReductionChain(Operation &operation) {
  if (operation.getNumOperands() == 0 || operation.getNumResults() != 1)
    return false;
  Operation *producer = operation.getOperand(0).getDefiningOp();
  if (!producer || producer->getName().getStringRef() != "intent.reduce" ||
      producer->getNumOperands() == 0)
    return false;
  auto source = dyn_cast<RankedTensorType>(producer->getOperand(0).getType());
  auto intermediate =
      dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  return source && intermediate && result && source.getRank() >= 4 &&
         intermediate.getRank() + 1 == source.getRank() &&
         result.getRank() + 1 == intermediate.getRank();
}

} // namespace

FailureOr<RealizationIndex>
indexRealization(intent::plan::ProgramOp realization,
                 const target::KernelModel &kernel) {
  RealizationIndex index;
  SmallVector<intent::plan::ReductionOp> reductions;
  SmallVector<intent::plan::ScanOp> scans;
  SmallVector<intent::plan::PointwiseOp> pointwise;
  SmallVector<intent::plan::TransferOp> transfers;
  SmallVector<intent::plan::RangeOp> ranges;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (auto value = dyn_cast<intent::plan::DeviceOp>(operation)) {
      index.target.operation = value;
    } else if (auto value = dyn_cast<intent::plan::AxisOp>(operation)) {
      plan::AxisOp binding;
      binding.operation = value;
      if (std::optional<StringRef> group = value.getGroup()) {
        FailureOr<std::string> spelling = syntax::parameter(value, *group);
        if (failed(spelling))
          return failure();
        binding.group = *spelling;
      }
      index.axes.try_emplace(value.getNode(), binding);
    } else if (auto value = dyn_cast<intent::plan::RangeOp>(operation)) {
      ranges.push_back(value);
    } else if (auto value =
                   dyn_cast<intent::plan::RegionBindingOp>(operation)) {
      index.regionBindings[value.getArgument()] = value;
    } else if (auto value = dyn_cast<intent::plan::LaunchOp>(operation)) {
      index.program.operation = value;
    } else if (auto value = dyn_cast<intent::plan::BlockExtentOp>(operation)) {
      plan::BlockExtentOp binding;
      binding.operation = value;
      index.blockExtents[value.getLogicalExtent()] = binding;
    } else if (auto value = dyn_cast<intent::plan::BufferOp>(operation)) {
      Operation *buffer = kernel.nodes.lookup(value.getNode());
      if (!buffer || buffer->getName().getStringRef() != "intent.buffer" ||
          (value.getSpace() != "private_scalar_array" &&
           value.getSpace() != "private_vector" &&
           value.getSpace() != "private_workspace"))
        return value.emitOpError("does not bind a logical buffer residency");
      plan::BufferOp binding;
      binding.operation = value;
      index.buffers[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::PaddingOp>(operation)) {
      plan::PaddingOp binding;
      binding.operation = value;
      index.paddings[value.getValue()] = binding;
    } else if (auto value = dyn_cast<intent::plan::ReductionOp>(operation)) {
      reductions.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::ScanOp>(operation)) {
      scans.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::PointwiseOp>(operation)) {
      pointwise.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::ContractOp>(operation)) {
      plan::ContractOp binding;
      binding.operation = value;
      Operation *contract = kernel.nodes.lookup(value.getNode());
      binding.lowering =
          contract && contract->getName().getStringRef() ==
                          "intent.scaled_contract"
              ? syntax::scaledContraction().str()
              : syntax::contraction().str();
      binding.lhsSpace = syntax::bufferSpace(value.getLhsSpace()).str();
      binding.rhsSpace = syntax::bufferSpace(value.getRhsSpace()).str();
      binding.accumulatorSpace =
          syntax::bufferSpace(value.getAccumulatorSpace()).str();
      if (binding.lhsSpace.empty() || binding.rhsSpace.empty() ||
          binding.accumulatorSpace.empty()) {
        value.emitOpError("has no TileLang contraction residency spelling");
        return failure();
      }
      index.contracts[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::SparseContractOp>(operation)) {
      if (value.getFormat() != "two_of_four")
        return value.emitOpError("has no TileLang sparse contraction spelling");
      index.sparseContracts[value.getNode()] = value;
    } else if (auto value = dyn_cast<intent::plan::TransferOp>(operation)) {
      transfers.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::StageOp>(operation)) {
      plan::StageOp binding;
      binding.operation = value;
      binding.position = index.stages.size();
      index.stages.push_back(binding);
    } else if (auto value = dyn_cast<intent::plan::StageAxisOp>(operation)) {
      plan::StageAxisOp binding;
      binding.operation = value;
      index.stageAxes[value.getStageNode()][value.getRole()] = binding;
    } else if (auto value = dyn_cast<intent::plan::StreamAxisOp>(operation)) {
      index.streamAxes.push_back(value);
    } else if (auto value =
                   dyn_cast<intent::plan::StreamBindingOp>(operation)) {
      index.streamBindings[value.getStreamNode()] = value;
    }
  }
  if (failed(target::emission::indexAxisRanges(index, ranges, syntax::tile)) ||
      failed(target::emission::indexCanonicalStructure(index, kernel)))
    return failure();
  target::emission::indexAxisRoles(index);
  for (auto &entry : index.streams) {
    plan::StreamOp &binding = entry.second;
    plan::AxisOp axis = index.axes.lookup(binding.getAxisNode());
    const target::emission::RangeBinding *traversal =
        axis ? axis.getRange(binding.getRangePurpose(), binding.getRangeLevel())
             : nullptr;
    FailureOr<std::string> tile =
        traversal ? syntax::tile(binding.operation, traversal->getTileRole())
                  : FailureOr<std::string>(failure());
    if (failed(tile))
      return binding.emitOpError("does not bind an ordered physical axis");
    binding.tile = *tile;
  }
  index.components = target::emission::indexPhysicalComponents(index);
  bool rowStrided = workerReuse(index);
  for (intent::plan::ReductionOp value : reductions) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::reductionRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<int64_t> axis =
        operation ? target::emission::reductionAxis(*operation)
                  : FailureOr<int64_t>(failure());
    if (failed(role) || failed(axis))
      return value.emitOpError("does not bind a canonical reduction");
    index.requiresSymmetricProgramTiles =
        index.requiresSymmetricProgramTiles ||
        isRankReducingReductionChain(*operation);
    if (*role == "reduce_generic")
      return operation->emitOpError(
          "TileLang 0.1.13 CUDA codegen cannot lower the tirx.Reduce produced "
          "by comm_reducer; generic reduction combiners are unsupported");
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = syntax::reduction(*role).str();
    binding.resultSpace = syntax::bufferSpace(value.getResultSpace()).str();
    binding.axis = *axis;
    if (binding.resultSpace.empty())
      return value.emitOpError("has no TileLang reduction residency spelling");
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::scanRole(*operation)
                  : FailureOr<std::string>(failure());
    if (succeeded(role) && *role == "scan_generic_inclusive")
      return value.emitOpError(
          "TileLang 0.1.13 has no mechanically lowerable generic scan "
          "combiner path; generic scan combiners are unsupported");
    if (failed(role) || *role != "scan_inclusive_add" ||
        !index.axes.count(value.getAxisNode()))
      return value.emitOpError("does not bind a canonical scan");
    plan::ScanOp binding;
    binding.operation = value;
    binding.lowering = syntax::scan().str();
    binding.resultSpace = value.getResultSpace() == "private_workspace"
                              ? "global"
                              : syntax::bufferSpace(value.getResultSpace()).str();
    binding.axis = value.getTensorAxis();
    binding.axisNode = value.getAxisNode();
    if (binding.resultSpace.empty())
      return value.emitOpError("has no TileLang scan residency spelling");
    index.scans[value.getNode()] = binding;
  }
  for (intent::plan::PointwiseOp value : pointwise) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    std::string materialization =
        operation && target::emission::feedsContraction(*operation)
            ? "contract_operand"
            : "elementwise";
    FailureOr<StringRef> lowering =
        succeeded(role)
            ? syntax::pointwise(value, *role, materialization)
            : FailureOr<StringRef>(failure());
    plan::PointwiseOp binding;
    binding.operation = value;
    binding.resultSpace = syntax::bufferSpace(value.getResultSpace()).str();
    if (failed(lowering) || binding.resultSpace.empty())
      return value.emitOpError("does not bind canonical pointwise semantics");
    binding.lowering = lowering->str();
    binding.defer = target::emission::feedsStagedContraction(index, *operation);
    index.pointwise[value.getNode()] = binding;
  }
  for (intent::plan::TransferOp value : transfers) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    bool load = operation &&
                operation->getName().getStringRef() == "intent.view_load";
    bool store = operation &&
                 (operation->getName().getStringRef() == "intent.view_store" ||
                  operation->getName().getStringRef() == "intent.scatter_unique" ||
                  operation->getName().getStringRef() == "intent.atomic_add" ||
                  operation->getName().getStringRef() == "intent.atomic_cas");
    if (!load && !store)
      return value.emitOpError("does not bind a canonical transfer");
    FailureOr<bool> derivedScalar =
        target::hasDerivedScalarIndex(*operation);
    bool tensorIndirect = value.getTensorIndexing() == "data_dependent";
    if (failed(derivedScalar))
      return failure();
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = rowStrided ? (load ? "gather" : "scatter")
                                : (load ? "load" : "store");
    bool raggedBound = llvm::any_of(value.getDomainNodes(), [&](int64_t axis) {
      return target::emission::isRaggedBoundAxis(index.components, axis);
    });
    bool plannedValidity = !value.getValidityDomainNodes().empty() &&
                           (store || value.getFill() != "none");
    bool materializeLogicalBounds =
        (raggedBound || plannedValidity) && !value.getConsumerNeutralized();
    bool packedScalar =
        target::emission::hasPackedScalarDomain(index, binding);
    binding.transfer = *derivedScalar || tensorIndirect ||
                               materializeLogicalBounds
                           ? "parallel_elements"
                           : "bulk_copy";
    binding.resultSpace = syntax::bufferSpace(value.getResultSpace()).str();
    binding.defer = load && index.stages.empty() && feedsAtomicValue(*operation);
    binding.explicitBounds =
        rowStrided || materializeLogicalBounds ||
        (!index.stages.empty() && store) || *derivedScalar || tensorIndirect ||
        packedScalar;
    if (binding.resultSpace.empty()) {
      value.emitOpError("has no TileLang transfer residency spelling");
      return failure();
    }
    index.boundaries[value.getNode()] = binding;
  }
  if (!index.target || !index.program) {
    realization.emitOpError("lacks TileLang target or program choices");
    return failure();
  }
  return index;
}

FailureOr<SearchIndex>
indexSearchSpace(intent::plan::SearchSpaceOp searchSpace) {
  SearchIndex index;
  if (!searchSpace)
    return index;
  for (Operation &operation : searchSpace.getBody().front()) {
    if (auto autotune = dyn_cast<intent::plan::AutotuneOp>(operation)) {
      SmallVector<NamedAttribute> mappings;
      OpBuilder builder(searchSpace.getContext());
      for (Attribute attribute : autotune.getParameters()) {
        StringRef role = cast<StringAttr>(attribute).getValue();
        FailureOr<std::string> spelling = syntax::parameter(autotune, role);
        if (failed(spelling))
          return failure();
        mappings.push_back(builder.getNamedAttr(*spelling,
                                               builder.getStringAttr(role)));
      }
      index.autotune.operation = autotune;
      index.autotune.parameterMap = builder.getDictionaryAttr(mappings);
    }
  }
  return index;
}

SourceEmitter::SourceEmitter(target::KernelModel kernel,
                             intent::plan::ProgramOp realization,
                             intent::plan::SearchSpaceOp searchSpace,
                             RealizationIndex planIndex,
                             SearchIndex searchIndex, raw_ostream &output)
    : kernel(std::move(kernel)), realization(realization),
      searchSpace(searchSpace), planIndex(std::move(planIndex)),
      searchIndex(std::move(searchIndex)), output(output) {}

LogicalResult SourceEmitter::emit() {
  return target::emitSource(*this);
}

LogicalResult SourceEmitter::prepare() {
  if (failed(indexABI()))
    return failure();
  if (!planIndex.ragged.empty() && failed(prepareRaggedMetadata()))
    return failure();
  if (failed(resolvePhysicalBindings()))
    return failure();
  if (!searchSpace && planIndex.stages.empty() &&
      planIndex.components.reusedAxes.empty()) {
    auto lane = planIndex.axesByRole.find("lane_0");
    tuneRowLaunch = lane != planIndex.axesByRole.end() &&
                    lane->second.getTileRole().starts_with("row_vector") &&
                    !target::emission::hasNonReplayableEffect(
                        kernel.entry.getOperation());
  }
  if (failed(target::emission::indexScanProducerOperations(
          kernel, planIndex, scanProducerOwners)))
    return failure();
  if (failed(target::emission::indexDeferredContractReplays(
          kernel, planIndex, deferredContractReplays,
          deferredContractProducerOwners)))
    return failure();
  for (const auto &entry : planIndex.scans)
    if (failed(target::emission::verifyScanMaterializedValues(kernel,
                                                              entry.second)))
      return failure();
  if (failed(preparePrivateWorkspaces()))
    return failure();
  if (!planIndex.stages.empty())
    return prepareRaggedStages();
  return success();
}

LogicalResult SourceEmitter::preparePrivateWorkspaces() {
  SmallVector<int64_t> nodes;
  for (const auto &entry : planIndex.buffers)
    if (entry.second.getSpace() == "private_workspace")
      nodes.push_back(entry.first);
  llvm::sort(nodes);
  for (int64_t node : nodes) {
    plan::BufferOp binding = planIndex.buffers.lookup(node);
    Operation *buffer = kernel.nodes.lookup(node);
    FailureOr<target::LogicalBufferInfo> info =
        buffer ? target::getLogicalBufferInfo(*buffer)
               : FailureOr<target::LogicalBufferInfo>(failure());
    if (!buffer || failed(info) || info->shape.empty() ||
        buffer->getNumResults() != 1)
      return binding.emitOpError(
          "does not bind a logical private workspace");
    workspaceNames[buffer->getResult(0)] =
        "workspace_" + std::to_string(node);
    privateWorkspaceBuffers.push_back(buffer);
  }
  for (const auto &entry : planIndex.scans) {
    plan::ScanOp binding = entry.second;
    if (binding.getResultSpace() != "global")
      continue;
    Operation *scan = kernel.nodes.lookup(entry.first);
    std::string extent = axisDimensions.lookup(binding.getAxisNode());
    if (!scan || scan->getName().getStringRef() != "intent.scan" ||
        scan->getNumResults() == 0 || extent.empty())
      return binding.emitOpError(
          "does not bind a workspace-backed TileLang scan tensor");
    for (auto [component, result] : llvm::enumerate(scan->getResults())) {
      workspaceNames[result] = "scan_workspace_" +
                               std::to_string(entry.first) + "_" +
                               std::to_string(component);
      scanResults[result] = binding;
    }
    scanExtents[entry.first] = extent;
    for (int64_t valueID : binding.getMaterializedValues()) {
      FailureOr<Value> value = target::emission::lookupScanMaterializedValue(
          kernel, binding, valueID);
      if (failed(value))
        return failure();
      workspaceNames[*value] =
          "scan_materialized_" + std::to_string(valueID);
      scanMaterializedValues[*value] = binding;
    }
  }
  return success();
}

LogicalResult SourceEmitter::registerOperationHandlers(
    target::OperationHandlerRegistry &registry) {
  return registerEmissionHandlers(registry, *this);
}

LogicalResult SourceEmitter::indexABI() {
  views.reserve(kernel.abi.arguments.size());
  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view) {
      auto kind = argument.metadata.getAs<StringAttr>("kind");
      if (!kind)
        return kernel.entry.emitOpError()
               << "ABI value " << argument.valueID << " has no parameter kind";
      if (kind.getValue() == "constexpr")
        continue;
      if (kind.getValue() != "runtime_scalar" ||
          !isa<FloatType, IntegerType, IndexType>(argument.type))
        return kernel.entry.emitOpError()
               << "ABI value " << argument.valueID
               << " has no TileLang runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("TileLang emitter requires ranked views");
    ABIView emitted{&argument, view, tensor, {}};
    auto shape = argument.metadata.getAs<ArrayAttr>("shape");
    if (!shape || shape.size() != static_cast<size_t>(tensor.getRank()))
      return kernel.entry.emitOpError()
             << "view " << argument.name << " has incomplete shape metadata";
    for (auto [axis, extent] : llvm::enumerate(shape)) {
      if (auto symbol = dyn_cast<StringAttr>(extent)) {
        uint64_t staticExtent = 0;
        if (!symbol.getValue().getAsInteger(10, staticExtent)) {
          emitted.shape.push_back(symbol.getValue().str());
          continue;
        }
        std::string spelling = syntax::dimension(symbol.getValue());
        emitted.shape.push_back(spelling);
        if (!dimensionOwners.count(spelling)) {
          dimensionOwners[spelling] =
              argument.name + ".shape[" + std::to_string(axis) + "]";
          dimensionOrder.push_back(std::move(spelling));
        }
      } else if (auto integer = dyn_cast<IntegerAttr>(extent)) {
        emitted.shape.push_back(std::to_string(integer.getInt()));
      } else {
        return kernel.entry.emitOpError("contains an unsupported shape extent");
      }
    }
    unsigned position = views.size();
    views.push_back(std::move(emitted));
    viewPositions[argument.value] = position;
    valueNames[argument.value] = argument.name;
  }
  if (views.empty())
    return kernel.entry.emitOpError("TileLang emitter requires external views");
  return success();
}

LogicalResult SourceEmitter::resolvePhysicalBindings() {
  programRoot = kernel.nodes.lookup(planIndex.program.getLoopNode());
  if (!programRoot || programRoot->getName().getStringRef() != "intent.parallel")
    return planIndex.program.emitOpError("does not bind an intent.parallel op");
  for (auto &entry : planIndex.axesByRole) {
    Operation *domain = kernel.nodes.lookup(entry.getValue().getNode());
    if (!domain ||
        (domain->getName().getStringRef() != "intent.domain" &&
         domain->getName().getStringRef() != "intent.ragged_outer" &&
         domain->getName().getStringRef() != "intent.ragged_member"))
      return entry.getValue().emitOpError("does not bind a logical domain op");
    FailureOr<std::string> dimension = dimensionName(*domain);
    if (failed(dimension))
      return entry.getValue().emitOpError("cannot resolve its source dimension");
    roleDimensions[entry.getValue().getRole()] = *dimension;
    axisDimensions[entry.getValue().getNode()] = *dimension;
  }
  for (auto &entry : planIndex.axes) {
    Operation *domain = kernel.nodes.lookup(entry.first);
    if (!domain)
      return entry.second.emitOpError("does not bind a logical domain op");
    if (domain->getNumResults() != 1)
      return entry.second.emitOpError("binds a domain without one canonical value");
    FailureOr<int64_t> valueID = target::getValueID(
        domain->getResult(0), kernel, *domain, "TileLang domain tile binding");
    if (failed(valueID))
      return failure();
    std::string tile = entry.second.getTile().str();
    if (!entry.second.getReuseWorker() &&
        entry.second.getTileRole().starts_with("row_vector")) {
      const target::emission::RangeBinding *range = entry.second.roleRange();
      if (!range || !planIndex.blockExtents.count(range->getExtent()))
        return entry.second.emitOpError("cannot resolve its row-vector extent");
      tile = physicalExtent(range->getExtent());
    }
    regionTiles["?region_" + std::to_string(*valueID) + "_0"] =
        tile;
  }
  for (auto &entry : planIndex.paddings) {
    Value value = kernel.values.lookup(entry.first);
    if (!value || !isa<RankedTensorType>(value.getType()))
      return entry.second.emitOpError("does not bind a ranked tensor value");
  }
  for (const auto &entry : planIndex.regionBindings) {
    Value value = kernel.values.lookup(entry.first);
    auto argument = dyn_cast<BlockArgument>(value);
    plan::RegionBindingOp binding = entry.second;
    plan::AxisOp axis = planIndex.axes.lookup(binding.getAxisNode());
    const target::emission::RangeBinding *range =
        axis ? axis.getRange(binding.getPurpose(), binding.getLevel()) : nullptr;
    if (!argument || !axis || !range)
      return binding.emitOpError(
          "does not bind a canonical region argument and selected range");
    bool roundedRow = !axis.getReuseWorker() &&
                      range->getTileRole().starts_with("row_vector");
    if (roundedRow && !planIndex.blockExtents.count(range->getExtent()))
      return binding.emitOpError(
          "row-vector region range lacks its selected physical extent");
    std::string tile = roundedRow ? physicalExtent(range->getExtent())
                                  : range->getTile().str();
    regionTiles["?region_" + std::to_string(entry.first) + "_0"] = tile;
  }

  if (!planIndex.components.reusedAxes.empty()) {
    if (searchSpace)
      return realization.emitOpError(
          "fixed TileLang rows cannot consume a search space");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (!planIndex.axesByRole.count("program_0") ||
        lane == planIndex.axesByRole.end())
      return realization.emitOpError(
          "fixed TileLang rows require program_0 and lane_0 axes");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if ((view.view.getAccess() == "out" ||
           view.view.getAccess() == "inout") &&
          !fixedOutput)
        fixedOutput = &view;
    }
    if (!fixedOutput || fixedOutput->tensor.getRank() < 1)
      return kernel.entry.emitOpError(
          "fixed-row TileLang program requires one ranked output view");
  }
  if (target::emission::requiresDelegatedTuning(planIndex) &&
      (!searchSpace || !searchIndex.autotune))
    return realization.emitOpError(
        "tiled physical components require a delegated TileLang tuner");
  SmallVector<StringRef> logicalBlockExtents;
  logicalBlockExtents.reserve(planIndex.blockExtents.size());
  for (const auto &entry : planIndex.blockExtents)
    logicalBlockExtents.push_back(entry.getKey());
  llvm::sort(logicalBlockExtents);
  for (StringRef logicalExtent : logicalBlockExtents)
    blockExtentConstants.emplace_back(physicalExtent(logicalExtent),
                                      syntax::dimension(logicalExtent));
  return success();
}

LogicalResult SourceEmitter::prepareRaggedMetadata() {
  raggedRuntimes.reserve(planIndex.ragged.size());
  for (plan::RaggedOp ragged : planIndex.ragged) {
    RaggedRuntime runtime;
    runtime.binding = ragged;
    runtime.relation = kernel.nodes.lookup(ragged.getNode());
    runtime.outer = kernel.nodes.lookup(ragged.getOuterNode());
    StringRef outerName = runtime.outer
                              ? runtime.outer->getName().getStringRef()
                              : StringRef();
    if (!runtime.relation ||
        runtime.relation->getName().getStringRef() != "intent.ragged" ||
        (outerName != "intent.ragged_outer" &&
         outerName != "intent.ragged_member") ||
        ragged.getMemberNodes().empty())
      return ragged.emitOpError(
          "does not resolve to canonical ragged ownership operations");
    for (int64_t memberNode : ragged.getMemberNodes()) {
      Operation *member = kernel.nodes.lookup(memberNode);
      plan::AxisOp axis = planIndex.axes.lookup(memberNode);
      if (!member ||
          member->getName().getStringRef() != "intent.ragged_member" || !axis)
        return ragged.emitOpError(
            "references a non-canonical ragged member domain");
      runtime.members.push_back(member);
      if (axis.hasRole("parallel"))
        runtime.ownedMembers.push_back(member);
    }
    Operation *offsetsLoad = runtime.relation->getNumOperands() >= 3
                                 ? runtime.relation->getOperand(2).getDefiningOp()
                                 : nullptr;
    FailureOr<ABIView *> offsets =
        offsetsLoad &&
                offsetsLoad->getName().getStringRef() == "intent.view_load" &&
                offsetsLoad->getNumOperands() == 1
            ? lookupView(offsetsLoad->getOperand(0), *runtime.relation)
            : FailureOr<ABIView *>(failure());
    if (failed(offsets) || (*offsets)->tensor.getRank() != 1)
      return runtime.relation->emitOpError(
          "requires a canonical rank-one offsets view");
    runtime.offsets = *offsets;
    if (runtime.relation->getNumOperands() == 4) {
      Operation *indicesLoad = runtime.relation->getOperand(3).getDefiningOp();
      FailureOr<ABIView *> indices =
          indicesLoad &&
                  indicesLoad->getName().getStringRef() == "intent.view_load" &&
                  indicesLoad->getNumOperands() == 1
              ? lookupView(indicesLoad->getOperand(0), *runtime.relation)
              : FailureOr<ABIView *>(failure());
      if (failed(indices) || (*indices)->tensor.getRank() != 1)
        return runtime.relation->emitOpError(
            "requires a canonical rank-one member-index view");
      runtime.indices = *indices;
    }
    unsigned position = raggedRuntimes.size();
    raggedRuntimeByRelation[ragged.getNode()] = position;
    raggedRuntimesByAxis[ragged.getOuterNode()].push_back(position);
    for (int64_t member : ragged.getMemberNodes())
      raggedRuntimesByAxis[member].push_back(position);
    raggedRuntimes.push_back(std::move(runtime));
  }
  return success();
}

LogicalResult SourceEmitter::prepareRaggedStages() {
  if (failed(target::emission::indexStageOperations(
          kernel, planIndex, operationStages)))
    return failure();
  stageBodies.resize(planIndex.stages.size());
  for (auto [position, stage] : llvm::enumerate(planIndex.stages)) {
    Operation *contract = kernel.nodes.lookup(stage.getNode());
    if (!contract || contract->getName().getStringRef() != "intent.contract" ||
        contract->getNumResults() != 1)
      return stage.emitOpError("does not bind one canonical contraction");
    auto axes = planIndex.stageAxes.find(stage.getNode());
    plan::StageAxisOp feature =
        axes == planIndex.stageAxes.end() ? plan::StageAxisOp()
                                         : axes->second.lookup("feature");
    plan::StageAxisOp reduction =
        axes == planIndex.stageAxes.end() ? plan::StageAxisOp()
                                         : axes->second.lookup("reduction");
    plan::StageAxisOp member =
        axes == planIndex.stageAxes.end() ? plan::StageAxisOp()
                                         : axes->second.lookup("member");
    if (!feature || !reduction || !member || !feature.getWorkerAxisAttr() ||
        !member.getWorkerAxisAttr() || !member.getAxisNodeAttr())
      return stage.emitOpError("has incomplete physical stage-axis decisions");
    auto runtimes =
        raggedRuntimesByAxis.find(member.getAxisNodeAttr().getInt());
    if (runtimes == raggedRuntimesByAxis.end() ||
        runtimes->second.size() != 1)
      return stage.emitOpError(
          "does not resolve one ragged relation for its member axis");
    unsigned runtimeIndex = runtimes->second.front();
    RaggedRuntime &runtime = raggedRuntimes[runtimeIndex];
    bool owned = llvm::any_of(runtime.ownedMembers, [&](Operation *candidate) {
      auto node = candidate->getAttrOfType<IntegerAttr>("intent.node");
      return node && node.getInt() == member.getAxisNodeAttr().getInt();
    });
    if (!owned)
      return stage.emitOpError("uses a ragged member without program ownership");
    if (runtime.memberDimension.empty())
      runtime.memberDimension = member.getExtent().str();
    else if (runtime.memberDimension != member.getExtent())
      return stage.emitOpError("changes its staged ragged member extent");
    Operation *members = nullptr;
    for (int64_t operationNode : stage.getOperations()) {
      Operation *candidate = kernel.nodes.lookup(operationNode);
      if (!candidate ||
          candidate->getName().getStringRef() != "intent.members")
        continue;
      if (members)
        return stage.emitOpError("contains multiple member enumeration ops");
      members = candidate;
    }
    if (!members)
      return stage.emitOpError("has no member enumeration operation");
    FailureOr<std::string> featureTile =
        syntax::tile(feature.operation, feature.getTile());
    FailureOr<std::string> memberTile =
        syntax::tile(member.operation, member.getTile());
    FailureOr<std::string> reductionTile =
        syntax::tile(reduction.operation, reduction.getTile());
    if (failed(featureTile) || failed(memberTile) || failed(reductionTile))
      return failure();
    stageRaggedRuntime[position] = runtimeIndex;
    stageFeatureDimensions[position] = feature.getExtent().str();
    stageMemberDimensions[position] = member.getExtent().str();
    stageReductionDimensions[position] = reduction.getExtent().str();
    stageFeatureTiles[position] = *featureTile;
    stageMemberTiles[position] = *memberTile;
    stageReductionTiles[position] = *reductionTile;
    stageFeatureWorkers[position] = feature.getWorkerAxisAttr().getInt();
    stageMemberWorkers[position] = member.getWorkerAxisAttr().getInt();
    for (int64_t valueID : stage.getInputs())
      if (!kernel.values.count(valueID))
        return stage.emitOpError("references an unknown stage input value");
    for (int64_t valueID : stage.getOutputs()) {
      auto found = kernel.values.find(valueID);
      if (found == kernel.values.end() ||
          !stageOutputOwners.try_emplace(found->second, position).second)
        return stage.emitOpError("has an invalid stage output value");
      workspaceNames[found->second] =
          "workspace_" + std::to_string(valueID);
    }
    if (!llvm::is_contained(operationStages.lookup(contract), position))
      return stage.emitOpError("stage roots do not depend on its contraction");
  }
  return success();
}

bool SourceEmitter::selectOperation(Operation &operation) {
  auto contractProducer = deferredContractProducerOwners.find(&operation);
  if (contractProducer != deferredContractProducerOwners.end())
    return activeDeferredContract &&
           llvm::is_contained(contractProducer->second,
                              activeDeferredContract);
  auto scanProducer = scanProducerOwners.find(&operation);
  if (scanProducer != scanProducerOwners.end())
    return activeScanReplay == scanProducer->second;
  if (planIndex.stages.empty())
    return true;
  activeStages = operationStages.lookup(&operation);
  return !activeStages.empty() &&
         !target::emission::isAbsorbedStagedAccessMetadata(planIndex,
                                                           operation);
}

void SourceEmitter::stageLine(unsigned stage, StringRef text, unsigned indent) {
  stageBodies[stage].append(indent * 4, ' ');
  stageBodies[stage] += text.str();
  stageBodies[stage] += "\n";
}

void SourceEmitter::emitImports() {
  bool tuneRow = !planIndex.components.reusedAxes.empty() || tuneRowLaunch;
  bool tuneGemmWarpPolicy = searchSpace && usesMatrixContraction();
  output << "import torch\n";
  output << "import tilelang\n";
  output << "import tilelang.language as T\n";
  output << "from intent.runtime.tuning.tilelang import DEFAULT_NUM_STAGES, DEFAULT_THREADS";
  if (tuneRow)
    output << ", row_autotune_configurations";
  output << "\n";
  if (planIndex.program.getPersistent())
    output << "_NUM_SMS = torch.cuda.get_device_properties("
           << planIndex.target.getDevice() << ").multi_processor_count\n";
  if (searchSpace || tuneRow) {
    output << "from tilelang.autotuner import autotune\n";
  }
  if (searchSpace) {
    output << "from intent.runtime.tuning.tilelang import autotune_configurations\n";
    output << "\n_PARAMETER_MAP = {";
    bool programM = false;
    bool programN = false;
    for (auto [index, mapping] :
         llvm::enumerate(searchIndex.autotune.getParameterMap())) {
      if (index)
        output << ", ";
      output << "'" << mapping.getName().getValue() << "': '"
             << cast<StringAttr>(mapping.getValue()).getValue() << "'";
      StringRef role = cast<StringAttr>(mapping.getValue()).getValue();
      programM = programM || role == "program_m";
      programN = programN || role == "program_n";
    }
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP";
    if (programM && programN && planIndex.requiresSymmetricProgramTiles)
      output << ", equal_role_groups=(('program_m', 'program_n'),)";
    llvm::StringMap<int64_t> roleDivisors;
    for (const auto &entry : planIndex.axes) {
      for (const target::emission::RangeBinding &range : entry.second.ranges) {
        if (!range.isCompact())
          continue;
        int64_t &divisor = roleDivisors[range.getTileRole()];
        divisor = divisor == 0 ? range.getDivisor()
                               : std::lcm(divisor, range.getDivisor());
      }
    }
    if (!roleDivisors.empty()) {
      output << ", role_divisors={";
      bool first = true;
      for (const auto &entry : roleDivisors) {
        if (!first)
          output << ", ";
        output << "'" << entry.getKey() << "': " << entry.getValue();
        first = false;
      }
      output << "}";
    }
    if (tuneGemmWarpPolicy)
      output << ", extra_parameters={'gemm_warp_policy': (0, 1, 2)}";
    output << ")\n";
  }
  if (tuneRow)
    output << "_ROW_CONFIGS = row_autotune_configurations()\n";
  if (searchSpace || tuneRow) {
    output << "_AUTOTUNE_INPUTS = None\n\n";
    output << "def _fresh_autotune_inputs(_):\n";
    output << "    if _AUTOTUNE_INPUTS is None:\n";
    output << "        raise RuntimeError('TileLang autotuning has no captured launch inputs')\n";
    output << "    return [value.clone() if isinstance(value, torch.Tensor) else value for value in _AUTOTUNE_INPUTS]\n";
  }
  output << "\n\n";
}

LogicalResult SourceEmitter::emitHelpers() {
  FailureOr<SmallVector<func::FuncOp>> combiners =
      target::emission::collectCombiners(kernel.entry);
  if (failed(combiners))
    return failure();
  if (!combiners->empty())
    return combiners->front().emitOpError(
        "TileLang 0.1.13 CUDA codegen cannot lower the tirx.Reduce produced by "
        "comm_reducer and exposes no generic scan equivalent; generic "
        "combiners are unsupported");
  return success();
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  auto emitBuilderParameters = [&](raw_ostream &stream, int64_t stage) {
    bool first = true;
    auto parameter = [&](StringRef value) {
      if (!first)
        stream << ", ";
      stream << value;
      first = false;
    };
    for (const std::string &dimension : dimensionOrder)
      parameter(dimension);
    if (stage >= 0) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      parameter(ragged.indices ? "MAX_ROUTES" : "ROUTE_LENGTHS");
    }
    for (int64_t axis : planIndex.components.orderedRaggedProgramAxes)
      parameter("MAX_SEQUENCE_LENGTH_" + std::to_string(axis));
    if (usesMatrixContraction())
      parameter("gemm_warp_policy=0");
    if (!planIndex.components.reusedAxes.empty()) {
      parameter("TILE_SIZE");
      parameter("num_stages=DEFAULT_NUM_STAGES");
      parameter("threads=DEFAULT_THREADS");
    } else {
      if (searchIndex.autotune)
        for (NamedAttribute config : searchIndex.autotune.getParameterMap())
          parameter(config.getName().getValue().str() + "=1");
      parameter("num_stages=DEFAULT_NUM_STAGES");
      parameter("threads=DEFAULT_THREADS");
    }
  };
  auto emitMainParameters = [&](raw_ostream &stream) -> LogicalResult {
    bool first = true;
    auto parameter = [&](StringRef value) {
      if (!first)
        stream << ", ";
      stream << value;
      first = false;
    };
    for (ABIView &view : views) {
      std::string dtype =
          dtypeName(view.tensor.getElementType(), *kernel.entry.getOperation());
      if (dtype.empty())
        return failure();
      std::string shape = "(";
      for (auto [index, extent] : llvm::enumerate(view.shape)) {
        if (index)
          shape += ", ";
        shape += extent;
      }
      if (view.shape.size() == 1)
        shape += ",";
      shape += ")";
      parameter(view.argument->name + ": T.Tensor(" + shape + ", " + dtype +
                ")");
    }
    for (ABIScalar &scalar : scalars) {
      std::string dtype = dtypeName(scalar.type, *kernel.entry.getOperation());
      if (dtype.empty())
        return failure();
      parameter(scalar.name + ": " + dtype);
    }
    for (Operation *buffer : privateWorkspaceBuffers) {
      FailureOr<std::string> size =
          target::emission::privateWorkspaceElementCount(
              *buffer, planIndex, axisDimensions);
      FailureOr<target::LogicalBufferInfo> info =
          target::getLogicalBufferInfo(*buffer);
      std::string dtype = succeeded(info)
                              ? dtypeName(info->elementType, *buffer)
                              : std::string();
      if (failed(size) || failed(info) || dtype.empty())
        return buffer->emitOpError(
            "has no supported TileLang private-workspace parameter");
      parameter(workspaceNames.lookup(buffer->getResult(0)) +
                ": T.Tensor((" + *size + ",), " + dtype + ")");
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      FailureOr<std::string> size = target::emission::scanWorkspaceElementCount(
          entry.second, scanExtents.lookup(entry.first), planIndex,
          axisDimensions);
      if (!scan || scan->getNumResults() == 0 || failed(size))
        return entry.second.emitOpError(
            "has no supported TileLang scan-workspace parameter");
      for (Value result : scan->getResults()) {
        auto tensor = dyn_cast<RankedTensorType>(result.getType());
        std::string dtype = tensor ? dtypeName(tensor.getElementType(), *scan)
                                   : std::string();
        if (dtype.empty())
          return entry.second.emitOpError(
              "has an unsupported TileLang scan-workspace component dtype");
        parameter(workspaceNames.lookup(result) +
                  ": T.Tensor((" + *size + ",), " + dtype + ")");
      }
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        Value value = kernel.values.lookup(valueID);
        auto materialized = dyn_cast<RankedTensorType>(value.getType());
        std::string valueDtype = materialized
                                     ? dtypeName(materialized.getElementType(),
                                                 *scan)
                                     : std::string();
        if (!materialized || materialized.getRank() != 1 || valueDtype.empty())
          return entry.second.emitOpError(
              "requires rank-one materialized TileLang scan producer values");
        parameter(workspaceNames.lookup(value) + ": T.Tensor((" + *size +
                  ",), " + valueDtype + ")");
      }
    }
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        Value value = kernel.values.lookup(valueID);
        auto tensor = dyn_cast<RankedTensorType>(value.getType());
        std::string dtype = tensor
                                ? dtypeName(tensor.getElementType(),
                                            *kernel.entry.getOperation())
                                   : std::string();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported TileLang workspace");
        parameter(workspaceNames.lookup(value) + ": T.Tensor((" +
                  stageMemberDimensions.lookup(stage.getOrdinal()) + ", " +
                  stageFeatureDimensions.lookup(stage.getOrdinal()) + "), " +
                  dtype + ")");
      }
    return success();
  };

  auto emitDecorator = [&](raw_ostream &stream) {
    if (searchSpace || !planIndex.components.reusedAxes.empty() || tuneRowLaunch)
      stream << "@autotune(configs="
             << (searchSpace ? "_CONFIGS" : "_ROW_CONFIGS")
             << ", warmup=3, rep=10, "
                "timeout=100, skip_check=True, "
                "supply_prog=_fresh_autotune_inputs)\n";
    stream << "@tilelang.jit\n";
  };
  auto emitBlockExtentConstants = [&](raw_ostream &stream) {
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      stream << "    " << physicalExtent << " = 1 << (" << logicalExtent
             << " - 1).bit_length()\n";
    if (usesMatrixContraction())
      stream << "    GEMM_WARP_POLICY = (T.GemmWarpPolicy.FullRow, "
                "T.GemmWarpPolicy.Square, T.GemmWarpPolicy.FullCol)"
                "[gemm_warp_policy]\n";
  };

  if (!planIndex.stages.empty()) {
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      bool compact = !ragged.indices;
      llvm::raw_string_ostream source(stageBodies[stage]);
      emitDecorator(source);
      source << "def " << kernelName << "_stage_" << stage << "(";
      emitBuilderParameters(source, stage);
      source << "):\n";
      emitBlockExtentConstants(source);
      std::string featureTile = stageFeatureTiles.lookup(stage);
      std::string memberTile = stageMemberTiles.lookup(stage);
      if (compact)
        source << "    total_route_tiles = sum((length + " << memberTile
               << " - 1) // " << memberTile << " for length in ROUTE_LENGTHS)\n";
      source << "    @T.prim_func\n    def main(";
      if (failed(emitMainParameters(source)))
        return failure();
      source << "):\n";
      std::string feature = stageFeatureDimensions.lookup(stage);
      int64_t featureWorker = stageFeatureWorkers.lookup(stage);
      int64_t memberWorker = stageMemberWorkers.lookup(stage);
      if (featureWorker < 0 || featureWorker > 1 || memberWorker < 0 ||
          memberWorker > 1 || featureWorker == memberWorker)
        return planIndex.stages[stage].emitOpError(
            "has invalid TileLang stage worker-axis decisions");
      SmallVector<std::string> grids(2);
      SmallVector<std::string> names(2);
      grids[featureWorker] =
          "T.ceildiv(" + feature + ", " + featureTile + ")";
      plan::AxisOp outerAxis =
          planIndex.axes.lookup(ragged.binding.getOuterNode());
      std::string experts =
          outerAxis ? roleDimensions.lookup(
                          "program_" + std::to_string(outerAxis.getProgramOrder()))
                    : std::string();
      if (experts.empty())
        return ragged.binding.emitOpError(
            "has no program-owned outer-axis dimension");
      grids[memberWorker] = compact
                                ? "total_route_tiles"
                                : experts +
                                      " * T.ceildiv(MAX_ROUTES, " + memberTile +
                                      ")";
      names[featureWorker] = "bid_feature";
      names[memberWorker] = "bid_expert_route";
      source << "        with T.Kernel(" << grids[0] << ", " << grids[1]
             << ", threads=threads) as (" << names[0] << ", " << names[1]
             << "):\n";
      source.flush();
      if (compact) {
        stageLine(stage, "expert = T.alloc_var(dtype=T.int32)");
        stageLine(stage, "route_tile = T.alloc_var(dtype=T.int32)");
        stageLine(stage, "tile_cursor = T.alloc_var(dtype=T.int32)");
        stageLine(stage, "expert = 0");
        stageLine(stage, "route_tile = 0");
        stageLine(stage, "tile_cursor = 0");
        stageLine(stage,
                  "for candidate in range(" + experts + "):");
        stageLine(stage, "candidate_begin = " +
                             ragged.offsets->argument->name + "[" +
                             addressIndex("candidate") + "]",
                  4);
        stageLine(stage, "candidate_end = " +
                             ragged.offsets->argument->name +
                             "[" + addressIndex("candidate + 1") + "]",
                  4);
        stageLine(stage,
                  "candidate_tiles = T.ceildiv(candidate_end - "
                  "candidate_begin, " + memberTile + ")",
                  4);
        stageLine(stage,
                  "owns_tile = bid_expert_route >= tile_cursor and "
                  "bid_expert_route < tile_cursor + candidate_tiles",
                  4);
        stageLine(stage,
                  "expert = T.if_then_else(owns_tile, candidate, expert)", 4);
        stageLine(stage,
                  "route_tile = T.if_then_else(owns_tile, bid_expert_route - "
                  "tile_cursor, route_tile)",
                  4);
        stageLine(stage, "tile_cursor = tile_cursor + candidate_tiles", 4);
      } else {
        stageLine(stage,
                  "num_route_tiles = T.ceildiv(MAX_ROUTES, " + memberTile +
                      ")");
        stageLine(stage, "expert = bid_expert_route // num_route_tiles");
        stageLine(stage, "route_tile = bid_expert_route % num_route_tiles");
      }
      stageLine(stage, "route_begin = " + ragged.offsets->argument->name +
                           "[" + addressIndex("expert") + "]");
      stageLine(stage, "route_end = " + ragged.offsets->argument->name +
                           "[" + addressIndex("expert + 1") + "]");
      stageLine(stage,
                "member_start = " + addressIndex("route_begin") + " + " +
                    addressIndex("route_tile") + " * " + memberTile);
    }
    return success();
  }

  emitDecorator(output);
  output << "def " << kernelName << "(";
  emitBuilderParameters(output, -1);
  output << "):\n";
  emitBlockExtentConstants(output);
  output << "    @T.prim_func\n    def main(";
  if (failed(emitMainParameters(output)))
    return failure();
  output << "):\n";
  if (!planIndex.components.reusedAxes.empty()) {
    output << "        with T.Kernel("
           << roleDimensions.lookup("program_0")
           << ", threads=threads) as program_index:\n";
    if (planIndex.components.reusedAxes.size() != 1)
      return realization.emitOpError(
          "TileLang worker reuse requires one explicit program axis");
    plan::AxisOp reused = planIndex.components.reusedAxes.front();
    programBlocks[reused.getNode()] = addressIndex("program_index");
    axisIndices[reused.getNode()] = addressIndex("program_index");
    plan::AxisOp lane = planIndex.axesByRole.lookup("lane_0");
    if (lane)
      axisIndices[lane.getNode()] = "0";
    return success();
  }

  SmallVector<plan::AxisOp> programAxes =
      target::emission::orderedProgramAxes(planIndex);
  auto axisExtent = [&](plan::AxisOp axis) {
    std::string role = "program_" + std::to_string(axis.getProgramOrder());
    std::string extent =
        planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode())
            ? "MAX_SEQUENCE_LENGTH_" + std::to_string(axis.getNode())
            : roleDimensions.lookup(role);
    return axis.isScalar()
               ? extent
               : "T.ceildiv(" + extent + ", " + axis.getTile().str() + ")";
  };
  std::array<std::string, 3> grid =
      target::emission::projectProgramGrid(planIndex, axisExtent);
  bool persistent = planIndex.program.getPersistent();
  std::string linear = "persistent_program";
  unsigned workers = 0;
  for (plan::AxisOp axis : programAxes)
    workers = std::max(workers,
                       static_cast<unsigned>(axis.getWorkerAxis() + 1));
  if (persistent) {
    output << "        total_program_tiles = "
           << target::emission::projectProgramVolume(planIndex, axisExtent)
           << "\n";
    output << "        persistent_programs = T.min(_NUM_SMS, total_program_tiles)\n";
    workers = 1;
  }
  output << "        with T.Kernel(";
  if (persistent) {
    output << "persistent_programs";
  } else {
    for (unsigned worker = 0; worker < workers; ++worker) {
      if (worker)
        output << ", ";
      output << grid[worker];
    }
  }
  bool addressableLocalBuffer = llvm::any_of(
      planIndex.buffers, [](const auto &entry) {
        return entry.second.getSpace() == "private_vector" ||
               entry.second.getSpace() == "private_workspace";
      });
  output << ", threads=" << (addressableLocalBuffer ? "1" : "threads") << ") as ";
  if (workers == 1)
    output << "pid_worker_0:\n";
  else {
    output << "(";
    for (unsigned worker = 0; worker < workers; ++worker) {
      if (worker)
        output << ", ";
      output << "pid_worker_" << worker;
    }
    output << "):\n";
  }
  std::string programIndent = "            ";
  if (persistent) {
    output << "            for persistent_wave in T.serial(T.ceildiv(total_program_tiles, persistent_programs)):\n";
    output << "                " << linear
           << " = " << addressIndex("persistent_wave") << " * "
           << addressIndex("persistent_programs") << " + "
           << addressIndex("pid_worker_0") << "\n";
    output << "                if " << linear << " < total_program_tiles:\n";
    programIndent = "                    ";
    indentation = 5;
  }

  for (const auto &entry : planIndex.components.groups) {
    SmallVector<plan::AxisOp> axes(entry.getValue().begin(), entry.getValue().end());
    llvm::sort(axes, [](plan::AxisOp lhs, plan::AxisOp rhs) {
      return lhs.getProgramOrder() < rhs.getProgramOrder();
    });
    if (axes.size() != 2 ||
        axes.front().getWorkerAxis() != axes.back().getWorkerAxis())
      return axes.front().emitOpError(
          "TileLang supports two-axis program grouping on one worker axis");
    StringRef group = axes.front().getGroupSpelling();
    if (group.empty())
      return axes.front().emitOpError("has no TileLang group spelling");
    plan::AxisOp lhs = axes[0];
    plan::AxisOp rhs = axes[1];
    std::string lhsRole =
        "program_" + std::to_string(lhs.getProgramOrder());
    std::string rhsRole =
        "program_" + std::to_string(rhs.getProgramOrder());
    std::string pid = "group_bid_" + std::to_string(lhs.getNode());
    std::string lhsCount = "group_count_" + std::to_string(lhs.getNode());
    std::string rhsCount = "group_count_" + std::to_string(rhs.getNode());
    std::string span = "group_span_" + std::to_string(lhs.getNode());
    std::string id = "group_id_" + std::to_string(lhs.getNode());
    std::string first = "group_first_" + std::to_string(lhs.getNode());
    std::string size = "group_size_" + std::to_string(lhs.getNode());
    std::string lhsBlock = "block_axis_" + std::to_string(lhs.getNode());
    std::string rhsBlock = "block_axis_" + std::to_string(rhs.getNode());
    if (persistent) {
      FailureOr<std::string> groupIndex =
          target::emission::projectLinearGroupIndex(
              planIndex, axes, axisExtent, linear,
              *lhs.operation.getOperation());
      if (failed(groupIndex))
        return failure();
      output << programIndent << pid << " = " << *groupIndex << "\n";
    } else {
      output << programIndent << pid << " = "
             << addressIndex("pid_worker_" +
                             std::to_string(lhs.getWorkerAxis()))
             << "\n";
    }
    output << programIndent << lhsCount << " = "
           << addressIndex("T.ceildiv(" + roleDimensions.lookup(lhsRole) +
                           ", " + lhs.getTile().str() + ")")
           << "\n";
    output << programIndent << rhsCount << " = "
           << addressIndex("T.ceildiv(" + roleDimensions.lookup(rhsRole) +
                           ", " + rhs.getTile().str() + ")")
           << "\n";
    output << programIndent << span << " = " << group << " * " << rhsCount
           << "\n";
    output << programIndent << id << " = " << pid << " // " << span << "\n";
    output << programIndent << first << " = " << id << " * " << group << "\n";
    output << programIndent << size << " = T.min(" << lhsCount << " - "
           << first << ", " << group << ")\n";
    output << programIndent << lhsBlock << " = " << first << " + (" << pid
           << " % " << size << ")\n";
    output << programIndent << rhsBlock << " = (" << pid << " % " << span
           << ") // " << size << "\n";
    programBlocks[lhs.getNode()] = lhsBlock;
    programBlocks[rhs.getNode()] = rhsBlock;
    axisIndices[lhs.getNode()] = lhsBlock + " * " + lhs.getTile().str();
    axisIndices[rhs.getNode()] = rhsBlock + " * " + rhs.getTile().str();
  }
  SmallVector<target::emission::ProgramIndexProjection> projections =
      persistent
          ? target::emission::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::emission::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("pid_worker_" + std::to_string(worker));
                });
  for (const target::emission::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    output << programIndent << block << " = "
           << addressIndex(projection.expression) << "\n";
    programBlocks[axis.getNode()] = block;
    axisIndices[axis.getNode()] =
        axis.isScalar() ? block : block + " * " + axis.getTile().str();
  }
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
  auto torchDtype = [&](Type type) -> StringRef {
    if (type.isInteger(1))
      return "torch.bool";
    if (type.isF16())
      return "torch.float16";
    if (type.isF32())
      return "torch.float32";
    if (type.isBF16())
      return "torch.bfloat16";
    if (isa<Float8E4M3FNType>(type))
      return "torch.float8_e4m3fn";
    if (isa<Float8E5M2Type>(type))
      return "torch.float8_e5m2";
    if (isa<Float8E8M0FNUType>(type))
      return "torch.float8_e8m0fnu";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 8)
      return integer.isUnsigned() ? "torch.uint8" : "torch.int8";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 16)
      return integer.isUnsigned() ? "torch.uint16" : "torch.int16";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 32)
      return "torch.int32";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 64)
      return "torch.int64";
    return {};
  };
  auto emitValidation = [&]() -> LogicalResult {
    for (ABIView &view : views) {
      StringRef dtype = torchDtype(view.tensor.getElementType());
      if (dtype.empty())
        return kernel.entry.emitOpError("has an unsupported TileLang ABI dtype");
      output << "    if " << view.argument->name << ".device != _DEVICE:\n";
      output << "        raise ValueError('" << view.argument->name
             << " must reside on the realized CUDA device')\n";
      output << "    if " << view.argument->name << ".ndim != "
             << view.tensor.getRank() << ":\n";
      output << "        raise ValueError('" << view.argument->name
             << " has the wrong rank')\n";
      output << "    if " << view.argument->name << ".dtype != " << dtype
             << ":\n";
      output << "        raise ValueError('" << view.argument->name
             << " has the wrong dtype')\n";
      output << "    if any(extent == 0 for extent in "
             << view.argument->name << ".shape):\n";
      output << "        raise NotImplementedError('TileLang does not support "
                "zero-extent external views in this compiler')\n";
      output << "    if sum(max(0, extent - 1) * abs(stride) for extent, stride "
                "in zip("
             << view.argument->name << ".shape, " << view.argument->name
             << ".stride())) > 2147483647:\n";
      output << "        raise NotImplementedError('TileLang cannot project this "
                "64-bit external-buffer address through its current bulk-copy "
                "lowering')\n";
    }
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    llvm::SmallVector<StringRef> exactExtents;
    exactExtents.reserve(exactBulkExtents.size());
    for (const auto &extent : exactBulkExtents)
      exactExtents.push_back(extent.getKey());
    llvm::sort(exactExtents);
    for (StringRef extent : exactExtents) {
      std::string logical = syntax::dimension(extent);
      output << "    if " << logical << " != 1 << (" << logical
             << " - 1).bit_length():\n";
      output << "        raise NotImplementedError('TileLang guarded float16 "
                "bulk transfer requires an exact power-of-two extent')\n";
    }
    for (ABIView &view : views) {
      output << "    if tuple(" << view.argument->name << ".shape) != (";
      for (auto [axis, extent] : llvm::enumerate(view.shape)) {
        if (axis)
          output << ", ";
        output << extent;
      }
      if (view.shape.size() == 1)
        output << ",";
      output << "):\n";
      output << "        raise ValueError('" << view.argument->name
             << " shape violates the kernel symbols')\n";
    }
    return success();
  };
  auto emitPrivateWorkspaceAllocations = [&]() -> LogicalResult {
    for (Operation *buffer : privateWorkspaceBuffers) {
      FailureOr<std::string> size =
          target::emission::privateWorkspaceElementCount(
              *buffer, planIndex, axisDimensions);
      FailureOr<std::string> initializer =
          target::emission::logicalBufferPythonInitializer(*buffer);
      FailureOr<target::LogicalBufferInfo> info =
          target::getLogicalBufferInfo(*buffer);
      StringRef dtype =
          succeeded(info) ? torchDtype(info->elementType) : StringRef();
      if (failed(size) || failed(initializer) || failed(info) || dtype.empty())
        return buffer->emitOpError(
            "has no supported TileLang private-workspace allocation");
      output << "    " << workspaceNames.lookup(buffer->getResult(0))
             << " = torch.full((" << *size << ",), " << *initializer
             << ", device=_DEVICE, dtype=" << dtype << ")\n";
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      FailureOr<std::string> size = target::emission::scanWorkspaceElementCount(
          entry.second, scanExtents.lookup(entry.first), planIndex,
          axisDimensions);
      if (!scan || scan->getNumResults() == 0 || failed(size))
        return entry.second.emitOpError(
            "has no supported TileLang scan-workspace allocation");
      for (Value result : scan->getResults()) {
        auto tensor = dyn_cast<RankedTensorType>(result.getType());
        StringRef dtype = tensor ? torchDtype(tensor.getElementType())
                                 : StringRef();
        if (dtype.empty())
          return entry.second.emitOpError(
              "has an unsupported TileLang scan-workspace component dtype");
        output << "    " << workspaceNames.lookup(result)
               << " = torch.empty((" << *size
               << ",), device=_DEVICE, dtype=" << dtype << ")\n";
      }
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        Value value = kernel.values.lookup(valueID);
        auto materialized = dyn_cast<RankedTensorType>(value.getType());
        StringRef valueDtype =
            materialized ? torchDtype(materialized.getElementType()) : StringRef();
        if (!materialized || materialized.getRank() != 1 || valueDtype.empty())
          return entry.second.emitOpError(
              "requires rank-one materialized TileLang scan producer values");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty(("
               << *size << ",), device=_DEVICE, dtype=" << valueDtype << ")\n";
      }
    }
    return success();
  };
  auto emitLaunchSignature = [&]() {
    bool first = true;
    for (ABIView &view : views) {
      if (!first)
        output << ", ";
      output << view.argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!first)
        output << ", ";
      output << scalar.name;
      first = false;
    }
  };
  auto emitKernelArguments = [&](ABIView *substitute = nullptr) {
    bool first = true;
    for (ABIView &view : views) {
      if (!first)
        output << ", ";
      if (&view == substitute)
        output << "torch.zeros_like(" << view.argument->name << ")";
      else
        output << view.argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!first)
        output << ", ";
      output << scalar.name;
      first = false;
    }
    for (Operation *buffer : privateWorkspaceBuffers) {
      if (!first)
        output << ", ";
      output << workspaceNames.lookup(buffer->getResult(0));
      first = false;
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      for (Value result : scan->getResults()) {
        if (!first)
          output << ", ";
        output << workspaceNames.lookup(result);
        first = false;
      }
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        output << ", " << workspaceNames.lookup(kernel.values.lookup(valueID));
      }
    }
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        if (!first)
          output << ", ";
        output << workspaceNames.lookup(kernel.values.lookup(valueID));
        first = false;
      }
  };
  auto emitBuilderArguments = [&](int64_t stage) {
    bool first = true;
    for (const std::string &dimension : dimensionOrder) {
      if (!first)
        output << ", ";
      output << dimension;
      first = false;
    }
    if (stage >= 0) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      if (!first)
        output << ", ";
      std::string suffix = std::to_string(ragged.binding.getNode());
      output << (ragged.indices ? "max_routes_" : "route_lengths_")
             << suffix;
    }
    for (int64_t axis : planIndex.components.orderedRaggedProgramAxes) {
      if (!first)
        output << ", ";
      output << "max_sequence_length_" << axis;
      first = false;
    }
  };

  if (!planIndex.stages.empty()) {
    for (const std::string &body : stageBodies)
      output << body << "    return main\n\n\n";
  } else
    output << "    return main\n\n\n";
  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n_KERNEL_CACHE = {}\n\n\n";
  output << "def launch(";
  emitLaunchSignature();
  output << "):\n";
  if (searchSpace || !planIndex.components.reusedAxes.empty() || tuneRowLaunch)
    output << "    global _AUTOTUNE_INPUTS\n";
  if (failed(emitValidation()))
    return failure();
  if (failed(emitPrivateWorkspaceAllocations()))
    return failure();

  if (!planIndex.stages.empty()) {
    SmallVector<ABIView *> inputs;
    ABIView *merge = nullptr;
    for (ABIView &view : views) {
      if (view.view.getAccess() == "in")
        inputs.push_back(&view);
      else if ((view.view.getAccess() == "out" ||
                view.view.getAccess() == "inout") &&
               !merge)
        merge = &view;
      else
        return kernel.entry.emitOpError(
            "ragged TileLang stages require inputs and one inout merge view");
    }
    if (!merge)
      return kernel.entry.emitOpError("ragged TileLang stages have no merge view");
    llvm::DenseSet<unsigned> preparedRagged;
    for (const auto &entry : stageRaggedRuntime) {
      if (!preparedRagged.insert(entry.second).second)
        continue;
      RaggedRuntime &ragged = raggedRuntimes[entry.second];
      std::string suffix = std::to_string(ragged.binding.getNode());
      if (!ragged.indices)
        output << "    route_lengths_" << suffix
               << " = tuple(int(length) for length in ("
               << ragged.offsets->argument->name << "[1:] - "
               << ragged.offsets->argument->name << "[:-1]).tolist())\n";
      else
        output << "    max_routes_" << suffix << " = int(("
               << ragged.offsets->argument->name << "[1:] - "
               << ragged.offsets->argument->name
               << "[:-1]).max().item())\n";
    }
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        Value value = kernel.values.lookup(valueID);
        auto tensor = dyn_cast<RankedTensorType>(value.getType());
        StringRef dtype =
            tensor ? torchDtype(tensor.getElementType()) : StringRef();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported TileLang workspace");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty(("
               << stageMemberDimensions.lookup(stage.getOrdinal()) << ", "
               << stageFeatureDimensions.lookup(stage.getOrdinal())
               << "), device=_DEVICE, dtype=" << dtype << ")\n";
      }
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      bool compact = !ragged.indices;
      std::string suffix = std::to_string(ragged.binding.getNode());
      output << "    cache_key = (" << stage;
      for (const std::string &dimension : dimensionOrder)
        output << ", " << dimension;
      output << ", " << (compact ? "route_lengths_" : "max_routes_")
             << suffix;
      for (ABIView *input : inputs)
        output << ", " << input->argument->name << ".dtype";
      output << ", str(_DEVICE))\n";
      output << "    if cache_key not in _KERNEL_CACHE:\n";
      output << "        _AUTOTUNE_INPUTS = [";
      bool first = true;
      for (ABIView &view : views) {
        if (!first)
          output << ", ";
        if (planIndex.stages[stage].getOutputs().empty() && &view == merge)
          output << (target::emission::stageUsesScatterReduction(
                         planIndex.stages[stage], kernel)
                         ? "torch.zeros_like("
                         : "torch.empty_like(")
                 << view.argument->name << ")";
        else
          output << view.argument->name;
        first = false;
      }
      for (ABIScalar &scalar : scalars) {
        if (!first)
          output << ", ";
        output << scalar.name;
        first = false;
      }
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs()) {
          if (!first)
            output << ", ";
          output << workspaceNames.lookup(kernel.values.lookup(valueID));
          first = false;
        }
      output << "]\n        try:\n            compiled = " << kernelName << "_stage_" << stage
             << "(";
      emitBuilderArguments(stage);
      output << ")\n        finally:\n            _AUTOTUNE_INPUTS = None\n";
      output << "        _KERNEL_CACHE[cache_key] = compiled\n";
      output << "    compiled = _KERNEL_CACHE[cache_key]\n    compiled(";
      emitKernelArguments();
      output << ")\n";
    }
    output << "    return " << merge->argument->name << "\n\n\n";
    output << "def run(";
    bool first = true;
    for (ABIView *view : inputs) {
      if (!first)
        output << ", ";
      output << view->argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!first)
        output << ", ";
      output << scalar.name;
      first = false;
    }
    output << "):\n";
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    output << "    " << merge->argument->name << " = torch."
           << (target::emission::planUsesScatterReduction(planIndex, kernel)
                   ? "zeros"
                   : "empty")
           << "((";
    for (auto [axis, extent] : llvm::enumerate(merge->shape)) {
      if (axis)
        output << ", ";
      output << extent;
    }
    output << "), device=_DEVICE, dtype="
           << torchDtype(merge->tensor.getElementType()) << ")\n    launch(";
    first = true;
    for (ABIView &view : views) {
      if (!first)
        output << ", ";
      output << view.argument->name;
      first = false;
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    output << ")\n    return " << merge->argument->name << "\n";
    return success();
  }

  SmallVector<ABIView *> inputs;
  SmallVector<ABIView *> outputs;
  for (ABIView &view : views) {
    if (view.view.getAccess() == "in" || view.view.getAccess() == "inout")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out")
      outputs.push_back(&view);
    else
      return kernel.entry.emitOpError(
          "TileLang wrapper supports input and output views");
  }
  bool hasInOut = llvm::any_of(views, [](const ABIView &view) {
    return view.view.getAccess() == "inout";
  });
  if (outputs.empty() && !hasInOut)
    return kernel.entry.emitOpError("TileLang wrapper has no writable views");
  for (int64_t axis : planIndex.components.orderedRaggedProgramAxes) {
    FailureOr<plan::RaggedOp> relation =
        target::emission::uniqueRaggedRelation(
            planIndex, axis, *kernel.entry.getOperation());
    auto runtime = succeeded(relation)
                       ? raggedRuntimeByRelation.find(relation->getNode())
                       : raggedRuntimeByRelation.end();
    if (failed(relation) || runtime == raggedRuntimeByRelation.end())
      return kernel.entry.emitOpError(
          "has no ordered ragged runtime metadata");
    ABIView *offsets = raggedRuntimes[runtime->second].offsets;
    output << "    max_sequence_length_" << axis << " = int(("
           << offsets->argument->name << "[1:] - "
           << offsets->argument->name << "[:-1]).max().item())\n";
  }
  output << "    cache_key = (";
  bool firstCacheKey = true;
  auto emitCacheKey = [&](const std::string &value) {
    if (!firstCacheKey)
      output << ", ";
    output << value;
    firstCacheKey = false;
  };
  for (const std::string &dimension : dimensionOrder)
    emitCacheKey(dimension);
  for (int64_t axis : planIndex.components.orderedRaggedProgramAxes)
    emitCacheKey("max_sequence_length_" + std::to_string(axis));
  for (ABIView *input : inputs)
    emitCacheKey(input->argument->name + ".dtype");
  emitCacheKey("str(_DEVICE)");
  output << ")\n";
  output << "    if cache_key not in _KERNEL_CACHE:\n";
  if (searchSpace || !planIndex.components.reusedAxes.empty() || tuneRowLaunch) {
    output << "        _AUTOTUNE_INPUTS = [";
    emitKernelArguments();
    output << "]\n        try:\n            compiled = " << kernelName << "(";
    emitBuilderArguments(-1);
    if (!planIndex.components.reusedAxes.empty()) {
      plan::AxisOp laneAxis = planIndex.axesByRole.lookup("lane_0");
      std::string rowTile = laneAxis ? laneAxis.getTile().str() : std::string();
      if (rowTile.empty())
        return kernel.entry.emitOpError(
            "TileLang row configuration has no selected lane range");
      if (!dimensionOrder.empty())
        output << ", ";
      output << rowTile;
    }
    output << ")\n        finally:\n            _AUTOTUNE_INPUTS = None\n";
  } else {
    output << "        compiled = " << kernelName << "(";
    emitBuilderArguments(-1);
    output << ")\n";
  }
  output << "        _KERNEL_CACHE[cache_key] = compiled\n";
  output << "    compiled = _KERNEL_CACHE[cache_key]\n    compiled(";
  emitKernelArguments();
  output << ")\n";
  if (privateWorkspaceBuffers.empty() && scanResults.empty()) {
    output << "    return compiled\n\n\n";
  } else {
    output << "    return lambda: compiled(";
    emitKernelArguments();
    output << ")\n\n\n";
  }
  output << "def run(";
  bool first = true;
  for (ABIView *view : inputs) {
    if (!first)
      output << ", ";
    output << view->argument->name;
    first = false;
  }
  for (ABIScalar &scalar : scalars) {
    if (!first)
      output << ", ";
    output << scalar.name;
    first = false;
  }
  output << "):\n";
  for (const std::string &dimension : dimensionOrder)
    output << "    " << dimension << " = "
           << dimensionOwners.lookup(dimension) << "\n";
  for (ABIView *result : outputs) {
    StringRef outputDtype = torchDtype(result->tensor.getElementType());
    if (outputDtype.empty())
      return kernel.entry.emitOpError("has an unsupported TileLang output dtype");
    output << "    " << result->argument->name << " = torch.empty((";
    for (auto [axis, extent] : llvm::enumerate(result->shape)) {
      if (axis)
        output << ", ";
      output << extent;
    }
    if (result->shape.size() == 1)
      output << ",";
    output << "), device=_DEVICE, dtype=" << outputDtype << ")\n";
  }
  output << "    launch(";
  first = true;
  for (ABIView &view : views) {
    if (!first)
      output << ", ";
    output << view.argument->name;
    first = false;
  }
  for (ABIScalar &scalar : scalars)
    output << ", " << scalar.name;
  output << ")\n    return ";
  if (outputs.size() == 1) {
    output << outputs.front()->argument->name;
  } else {
    output << "(";
    for (auto [index, result] : llvm::enumerate(outputs)) {
      if (index)
        output << ", ";
      output << result->argument->name;
    }
    output << ")";
  }
  output << "\n";
  return success();
}

FailureOr<StringRef> SourceEmitter::lookupValue(Operation &consumer,
                                                 unsigned operandIndex) {
  if (operandIndex >= consumer.getNumOperands()) {
    consumer.emitOpError("references a missing operand during TileLang emission");
    return failure();
  }
  auto found = valueNames.find(consumer.getOperand(operandIndex));
  if (found == valueNames.end()) {
    consumer.emitOpError() << "operand " << operandIndex
                           << " has no emitted TileLang value";
    return failure();
  }
  return StringRef(found->second);
}

FailureOr<ABIView *> SourceEmitter::lookupView(Value value,
                                               Operation &consumer) {
  auto found = viewPositions.find(value);
  if (found == viewPositions.end()) {
    consumer.emitOpError("references a non-ABI external view");
    return failure();
  }
  return &views[found->second];
}

FailureOr<Operation *> SourceEmitter::resolveDomain(Value indexedValue,
                                                    Operation &consumer) {
  FailureOr<target::ScalarIndexSource> scalarSource =
      target::traceScalarIndexSource(indexedValue, consumer);
  if (failed(scalarSource))
    return failure();
  if (scalarSource->domain)
    return scalarSource->domain;
  if (scalarSource->hasDomain()) {
    consumer.emitOpError(
        "cannot resolve multi-axis scalar ownership during TileLang emission");
    return failure();
  }
  if (scalarSource->opaque) {
    consumer.emitOpError(
        "cannot resolve an opaque index source during TileLang emission");
    return failure();
  }
  if (Operation *definition = indexedValue.getDefiningOp())
    if (definition->getName().getStringRef() == "intent.domain" ||
        definition->getName().getStringRef() == "intent.ragged_outer" ||
        definition->getName().getStringRef() == "intent.ragged_member")
      return definition;
  consumer.emitOpError("cannot resolve index ownership during TileLang emission");
  return failure();
}

FailureOr<plan::AxisOp> SourceEmitter::resolveAxis(Value indexedValue,
                                                  Operation &consumer) {
  FailureOr<Operation *> domain = resolveDomain(indexedValue, consumer);
  if (failed(domain))
    return failure();
  FailureOr<int64_t> node = target::getNodeID(**domain, "axis lookup");
  plan::AxisOp axis =
      succeeded(node) ? planIndex.axes.lookup(*node) : plan::AxisOp();
  if (failed(node) || !axis) {
    consumer.emitOpError("indexes a domain without a TileLang axis binding");
    return failure();
  }
  return axis;
}

FailureOr<std::string> SourceEmitter::dimensionName(Operation &domain) {
  return target::emission::logicalDomainExtent(
      domain, [&](Value value,
                  Operation &consumer) -> FailureOr<ArrayRef<std::string>> {
        FailureOr<ABIView *> view = lookupView(value, consumer);
        if (failed(view))
          return failure();
        return ArrayRef<std::string>((*view)->shape);
      });
}

bool SourceEmitter::usesMatrixContraction() const {
  return !planIndex.contracts.empty() || !planIndex.sparseContracts.empty();
}

std::string SourceEmitter::addressIndex(StringRef expression) const {
  return "(" + expression.str() + ")";
}

std::string SourceEmitter::physicalExtent(StringRef logicalExtent) const {
  auto extent = planIndex.blockExtents.find(logicalExtent);
  if (extent == planIndex.blockExtents.end())
    return syntax::dimension(logicalExtent);
  return "PHYSICAL_" + syntax::dimension(logicalExtent);
}

std::string SourceEmitter::logicalExtent(StringRef extent) const {
  return syntax::dimension(extent);
}

FailureOr<std::string>
SourceEmitter::transferPhysicalExtentFill(Operation &operation) {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "physical transfer extent");
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || failed(relation) || failed(view) || !boundary)
    return failure();
  ArrayRef<int64_t> neutralized = boundary.getConsumerNeutralized()
                                      ? boundary.getValidityTensorAxes()
                                      : ArrayRef<int64_t>();
  return target::emission::transferPhysicalExtentFill(
      planIndex, *relation, (*view)->shape, operation, neutralized);
}

FailureOr<std::string> SourceEmitter::structuredIndexExpression(
    Value value, Operation &consumer) {
  llvm::DenseSet<Value> active;
  auto project = [&](auto &self, Value current) -> FailureOr<std::string> {
    if (!active.insert(current).second)
      return consumer.emitOpError("contains a cyclic structured index");
    auto finish = [&](FailureOr<std::string> result) {
      active.erase(current);
      return result;
    };
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      Operation *owner = argument.getOwner()->getParentOp();
      if (owner && argument.getArgNumber() == 0 &&
          (owner->getName().getStringRef() == "intent.parallel" ||
           owner->getName().getStringRef() == "intent.ordered" ||
           owner->getName().getStringRef() == "intent.for" ||
           owner->getName().getStringRef() == "intent.state_stream")) {
        FailureOr<plan::AxisOp> axis = resolveAxis(current, consumer);
        std::string spelling =
            succeeded(axis) ? axisIndices.lookup(axis->getNode()) : std::string();
        if (!spelling.empty())
          return finish(spelling);
      }
      auto emitted = valueNames.find(current);
      return finish(emitted == valueNames.end()
                        ? FailureOr<std::string>(failure())
                        : FailureOr<std::string>(emitted->second));
    }
    Operation *definition = current.getDefiningOp();
    if (!definition)
      return finish(failure());
    StringRef name = definition->getName().getStringRef();
    if (name == "intent.constant") {
      if (auto integer =
              definition->getAttrOfType<IntegerAttr>("intent.value"))
        return finish(std::to_string(integer.getInt()));
      return finish(failure());
    }
    if (!isa<RankedTensorType>(current.getType())) {
      auto emitted = valueNames.find(current);
      if (emitted != valueNames.end())
        return finish(emitted->second);
    }
    if (name == "intent.dim" || name == "intent.region_end") {
      auto emitted = valueNames.find(current);
      return finish(emitted == valueNames.end()
                        ? FailureOr<std::string>(failure())
                        : FailureOr<std::string>(emitted->second));
    }
    if (name == "intent.indices" && definition->getNumOperands() == 1) {
      FailureOr<plan::AxisOp> axis =
          resolveAxis(definition->getOperand(0), consumer);
      std::string base =
          succeeded(axis) ? axisIndices.lookup(axis->getNode()) : std::string();
      if (base.empty())
        return finish(failure());
      return finish(base);
    }
    if (name == "intent.gather" && definition->getNumOperands() > 0)
      return finish(self(self, definition->getOperand(0)));
    if ((name == "intent.broadcast" || name == "intent.reshape" ||
         name == "intent.cast") &&
        definition->getNumOperands() == 1)
      return finish(self(self, definition->getOperand(0)));
    if (name == "intent.unary" && definition->getNumOperands() == 1) {
      auto logical =
          definition->getAttrOfType<StringAttr>("intent.operator");
      FailureOr<std::string> operand = self(self, definition->getOperand(0));
      if (!logical || logical.getValue() != "negate" || failed(operand))
        return finish(failure());
      return finish("-(" + *operand + ")");
    }
    if (name != "intent.binary" || definition->getNumOperands() != 2)
      return finish(failure());
    auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
    FailureOr<std::string> lhs = self(self, definition->getOperand(0));
    FailureOr<std::string> rhs = self(self, definition->getOperand(1));
    StringRef symbol = !logical   ? StringRef()
                       : logical.getValue() == "add"      ? "+"
                       : logical.getValue() == "subtract" ? "-"
                       : logical.getValue() == "multiply" ? "*"
                                                           : StringRef();
    if (failed(lhs) || failed(rhs) || symbol.empty())
      return finish(failure());
    return finish("(" + *lhs + ") " + symbol.str() + " (" + *rhs + ")");
  };
  return project(project, value);
}

FailureOr<std::string> SourceEmitter::accessIndices(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<std::string> indices;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice") {
      indices.push_back(":");
      continue;
    }
    if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError(
            "static TileLang index has no canonical value");
      indices.push_back(std::to_string(*term.staticValues.front()));
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "TileLang access has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      auto result = dyn_cast<OpResult>(indexed);
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      FailureOr<SmallVector<std::string>> extents =
          result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                 : FailureOr<SmallVector<std::string>>(failure());
      FailureOr<int64_t> transferNode =
          target::getNodeID(operation, "TileLang tensor-index transfer");
      plan::BoundaryOp boundary =
          succeeded(transferNode) ? planIndex.boundaries.lookup(*transferNode)
                                  : plan::BoundaryOp();
      if (failed(exact) || failed(extents) || failed(transferNode) || !boundary ||
          extents->size() != static_cast<size_t>(tensor.getRank()))
        return failure();
      if (!boundary.hasDataDependentTensorIndex()) {
        std::optional<unsigned> varying;
        FailureOr<std::string> projected =
            structuredIndexExpression(indexed, operation);
        std::string base = succeeded(projected) ? *projected : std::string();
        bool materialized = failed(projected);
        if (materialized)
          base = exact->str() + "[";
        for (auto [axis, extent] : llvm::enumerate(*extents)) {
          if (materialized) {
            if (axis)
              base += ", ";
            base += "0";
          }
          if (extent != "1") {
            if (varying)
              return operation.emitOpError(
                  "regular TileLang index tile varies along more than one axis");
            varying = axis;
          }
        }
        if (materialized)
          base += "]";
        indices.push_back(varying ? addressIndex(base) + " : " +
                                        addressIndex(base) + " + " +
                                        (*extents)[*varying]
                                  : addressIndex(base));
        continue;
      }
      if (tensor.getRank() != 1 || extents->size() != 1 ||
          extents->front() != "1")
        return operation.emitOpError(
            "TileLang bulk indirect access requires one singleton index tile");
      auto assumed = assumedIndexNames.find(indexed);
      indices.push_back(assumed == assumedIndexNames.end()
                            ? addressIndex(exact->str() + "[0]")
                            : addressIndex(assumed->second));
    }
    FailureOr<bool> traversalRegion =
        term.kind == "region_index"
            ? isTraversalRegionArgument(indexed, operation)
            : FailureOr<bool>(false);
    if (failed(traversalRegion))
      return failure();
    if (term.kind == "value_index" ||
        target::emission::isSequentialIterator(indexed) || *traversalRegion) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      indices.push_back(addressIndex(*exact));
    } else {
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      std::string base = axisIndices.lookup(axis->getNode());
      if (base.empty()) {
        operation.emitOpError()
            << "has no active TileLang index for logical axis "
            << axis->getNode();
        return failure();
      }
      std::string wideBase = addressIndex(base);
      if (target::emission::isPackedScalarAxis(*axis)) {
        indices.push_back(wideBase);
        continue;
      }
      std::string extent = axis->getTile().str();
      if (!axis->getReuseWorker() &&
          axis->getTileRole().starts_with("row_vector")) {
        const target::emission::RangeBinding *range = axis->roleRange();
        if (!range)
          return axis->emitOpError(
              "cannot resolve its row-vector physical extent");
        extent = physicalExtent(range->getExtent());
      }
      indices.push_back(axis->isScalar()
                            ? wideBase
                            : wideBase + " : " + wideBase + " + " +
                                  extent);
    }
  }
  std::string result;
  for (auto [index, value] : llvm::enumerate(indices)) {
    if (index)
      result += ", ";
    result += value;
  }
  return result;
}

FailureOr<bool>
SourceEmitter::isTraversalRegionArgument(Value value, Operation &consumer) {
  if (!isa<BlockArgument>(value))
    return false;
  FailureOr<int64_t> valueID = target::getValueID(
      value, kernel, consumer, "TileLang region argument range lookup");
  if (failed(valueID))
    return failure();
  auto binding = planIndex.regionBindings.find(*valueID);
  if (binding == planIndex.regionBindings.end()) {
    consumer.emitOpError("indexes a region argument without a physical binding");
    return failure();
  }
  return binding->second.getPurpose() == "traversal";
}

FailureOr<std::string>
SourceEmitter::elementAccessIndices(Operation &operation,
                                    ArrayRef<std::string> tileIndices) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned tensorIndexCount = llvm::count_if(
      *relation, [&](const target::IndexTerm &term) {
        return term.kind == "value_index" && term.operands.size() == 1 &&
               term.operands.front() &&
               isa<RankedTensorType>(
                   operation.getOperand(*term.operands.front()).getType());
      });
  SmallVector<std::string> indices;
  unsigned tileAxis = 0;
  bool advancedTensorAxesCovered = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "parallel TileLang transfer has too few new-axis indices");
      ++tileAxis;
      continue;
    }
    if (term.kind == "full_slice") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "parallel TileLang transfer has too few tile indices");
      indices.push_back(tileIndices[tileAxis++]);
      continue;
    }
    if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError(
            "parallel TileLang transfer has an invalid static index");
      indices.push_back(std::to_string(*term.staticValues.front()));
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "parallel TileLang transfer has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<std::string> projected =
          structuredIndexExpression(indexed, operation);
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact) && failed(projected))
        return failure();
      if (tensorIndexCount > 1 || tensor.getRank() > 1) {
        if (tensor.getRank() > static_cast<int64_t>(tileIndices.size()) ||
            (tileAxis != 0 && !advancedTensorAxesCovered))
          return operation.emitOpError(
              "TileLang broadcasted tensor index exceeds the transfer rank");
        auto result = dyn_cast<OpResult>(indexed);
        FailureOr<SmallVector<std::string>> extents =
            result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                   : FailureOr<SmallVector<std::string>>(failure());
        if (failed(extents) || extents->size() != static_cast<size_t>(tensor.getRank()))
          return operation.emitOpError(
              "TileLang broadcasted tensor index has no canonical extents");
        std::string element;
        if (succeeded(projected) && tensor.getRank() == 1) {
          element = *projected + " + " + tileIndices.front();
        } else {
          element = exact->str() + "[";
          for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
            if (axis)
              element += ", ";
            element += (*extents)[axis] == "1" ? "0" : tileIndices[axis];
          }
          element += "]";
        }
        indices.push_back(addressIndex(element));
        if (!advancedTensorAxesCovered) {
          tileAxis = tensor.getRank();
          advancedTensorAxesCovered = true;
        }
      } else {
        if (tensor.getRank() != 1 || tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "TileLang indirect element access requires one index-tile axis");
        auto assumed = assumedIndexNames.find(indexed);
        if (assumed != assumedIndexNames.end())
          indices.push_back(addressIndex(assumed->second));
        else if (succeeded(projected))
          indices.push_back(
              addressIndex(*projected + " + " + tileIndices[tileAxis]));
        else
          indices.push_back(addressIndex(exact->str() + "[" +
                                         tileIndices[tileAxis] + "]"));
        ++tileAxis;
      }
      continue;
    }
    FailureOr<bool> traversalRegion =
        term.kind == "region_index"
            ? isTraversalRegionArgument(indexed, operation)
            : FailureOr<bool>(false);
    if (failed(traversalRegion))
      return failure();
    if (term.kind == "value_index" ||
        target::emission::isSequentialIterator(indexed) || *traversalRegion) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      indices.push_back(addressIndex(*exact));
    } else {
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      std::string base = axisIndices.lookup(axis->getNode());
      if (base.empty())
        return operation.emitOpError(
            "has no active TileLang element index for its logical axis");
      if (target::emission::isPackedScalarAxis(*axis)) {
        indices.push_back(addressIndex(base));
        continue;
      }
      if (!axis->isScalar()) {
        if (tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "parallel TileLang transfer has too few tile indices");
        indices.push_back(addressIndex(base) + " + " +
                          addressIndex(tileIndices[tileAxis++]));
      } else {
        indices.push_back(addressIndex(base));
      }
    }
  }
  if (tileAxis != tileIndices.size())
    return operation.emitOpError(
        "parallel TileLang transfer has unused tile indices");
  std::string result;
  for (auto [index, value] : llvm::enumerate(indices)) {
    if (index)
      result += ", ";
    result += value;
  }
  return result;
}

FailureOr<std::string>
SourceEmitter::elementBoundsPredicate(Operation &operation,
                                      ArrayRef<std::string> tileIndices,
                                      bool physicalOnly,
                                      bool includePhysical) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view) ||
      static_cast<size_t>(llvm::count_if(
          *relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != (*view)->shape.size())
    return failure();
  unsigned tensorIndexCount = llvm::count_if(
      *relation, [&](const target::IndexTerm &term) {
        return term.kind == "value_index" && term.operands.size() == 1 &&
               term.operands.front() &&
               isa<RankedTensorType>(
                   operation.getOperand(*term.operands.front()).getType());
      });
  SmallVector<std::string> predicates;
  unsigned tileAxis = 0;
  unsigned sourceAxis = 0;
  bool advancedTensorAxesCovered = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "bounded TileLang transfer has too few new-axis indices");
      ++tileAxis;
      continue;
    }
    unsigned axisNumber = sourceAxis++;
    if (term.kind == "full_slice") {
      if (tileAxis >= tileIndices.size())
        return operation.emitOpError(
            "bounded TileLang transfer has too few tile indices");
      if (includePhysical &&
          planIndex.blockExtents.count((*view)->shape[axisNumber]))
        predicates.push_back(tileIndices[tileAxis] + " < " +
                             (*view)->shape[axisNumber]);
      ++tileAxis;
      continue;
    }
    if (term.kind == "static_index")
      continue;
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "bounded TileLang transfer has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<std::string> projected =
          structuredIndexExpression(indexed, operation);
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact) && failed(projected))
        return failure();
      std::string index;
      if (tensorIndexCount > 1 || tensor.getRank() > 1) {
        if (tensor.getRank() > static_cast<int64_t>(tileIndices.size()) ||
            (tileAxis != 0 && !advancedTensorAxesCovered))
          return operation.emitOpError(
              "TileLang broadcasted tensor bounds exceed the transfer rank");
        auto result = dyn_cast<OpResult>(indexed);
        FailureOr<SmallVector<std::string>> extents =
            result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                   : FailureOr<SmallVector<std::string>>(failure());
        if (failed(extents) || extents->size() != static_cast<size_t>(tensor.getRank()))
          return operation.emitOpError(
              "TileLang broadcasted tensor bounds have no canonical extents");
        if (succeeded(projected) && tensor.getRank() == 1) {
          index = *projected + " + " + tileIndices.front();
        } else {
          index = exact->str() + "[";
          for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
            if (axis)
              index += ", ";
            index += (*extents)[axis] == "1" ? "0" : tileIndices[axis];
          }
          index += "]";
        }
        if (!advancedTensorAxesCovered) {
          tileAxis = tensor.getRank();
          advancedTensorAxesCovered = true;
        }
      } else {
        if (tensor.getRank() != 1 || tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "TileLang indirect bounds require one index-tile axis");
        index = succeeded(projected)
                    ? *projected + " + " + tileIndices[tileAxis++]
                    : exact->str() + "[" + tileIndices[tileAxis++] + "]";
      }
      if (!physicalOnly) {
        predicates.push_back("0 <= " + index);
        predicates.push_back(index + " < " + (*view)->shape[axisNumber]);
      }
      continue;
    }
    if (term.kind == "value_index" &&
        !isa<RankedTensorType>(indexed.getType())) {
      FailureOr<target::ScalarIndexSource> source =
          target::traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      bool packedScalar = false;
      if (source->domain) {
        FailureOr<int64_t> node =
            target::getNodeID(*source->domain, "packed scalar bounds");
        auto axis = succeeded(node) ? planIndex.axes.find(*node)
                                    : planIndex.axes.end();
        if (failed(node))
          return failure();
        packedScalar =
            axis != planIndex.axes.end() &&
            target::emission::isPackedScalarAxis(axis->second);
      }
      if (!physicalOnly && source->hasDomain() &&
          (source->transformed || packedScalar)) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        std::string extent = (*view)->shape[axisNumber];
        predicates.push_back("0 <= (" + exact->str() + ")");
        predicates.push_back("(" + exact->str() + ") < " + extent);
      }
      continue;
    }
    FailureOr<plan::AxisOp> axis =
        resolveAxis(indexed, operation);
    if (failed(axis))
      return failure();
    if (target::emission::isPackedScalarAxis(*axis)) {
      if (!physicalOnly) {
        std::string exact = axisIndices.lookup(axis->getNode());
        Operation *domain = kernel.nodes.lookup(axis->getNode());
        FailureOr<std::string> extent =
            domain ? dimensionName(*domain)
                   : FailureOr<std::string>(failure());
        if (exact.empty() || failed(extent))
          return operation.emitOpError(
              "bounded TileLang packed scalar has no exact interval");
        predicates.push_back("0 <= " + exact);
        predicates.push_back(exact + " < " + *extent);
      }
      continue;
    }
    if (axis->isScalar())
      continue;
    if (tileAxis >= tileIndices.size())
      return operation.emitOpError(
          "bounded TileLang transfer has too few tile indices");
    std::string base = axisIndices.lookup(axis->getNode());
    Operation *domain = kernel.nodes.lookup(axis->getNode());
    FailureOr<std::string> extent = failure();
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> orderedAxis =
          target::emission::representativeOrderedAxis(planIndex,
                                                      axis->getNode(), operation);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else if (domain) {
      extent = dimensionName(*domain);
    }
    if (base.empty() || failed(extent))
      return operation.emitOpError(
          "bounded TileLang transfer has no active axis interval");
    if (physicalOnly && !planIndex.blockExtents.count(*extent)) {
      ++tileAxis;
      continue;
    }
    predicates.push_back(base + " + " + tileIndices[tileAxis++] + " < " +
                         *extent);
  }
  if (tileAxis != tileIndices.size())
    return operation.emitOpError(
        "bounded TileLang transfer has unused tile indices");
  if (predicates.empty())
    return std::string("True");
  std::string result;
  for (auto [index, predicate] : llvm::enumerate(predicates)) {
    if (index)
      result += " and ";
    result += predicate;
  }
  return result;
}

FailureOr<std::string> SourceEmitter::scalarTransferPredicate(
    Operation &operation, const plan::BoundaryOp &boundary) {
  SmallVector<std::string> predicates;
  FailureOr<bool> derivedScalar = target::hasDerivedScalarIndex(operation);
  if (failed(derivedScalar))
    return failure();
  if (*derivedScalar) {
    FailureOr<std::string> address = elementBoundsPredicate(operation, {});
    if (failed(address))
      return failure();
    if (!address->empty())
      predicates.push_back(*address);
  }
  for (int64_t node : boundary.getDomainNodes()) {
    auto found = planIndex.axes.find(node);
    if (found == planIndex.axes.end() ||
        !target::emission::isPackedScalarAxis(found->second))
      continue;
    std::string exact = axisIndices.lookup(node);
    Operation *domain = kernel.nodes.lookup(node);
    FailureOr<std::string> extent =
        domain ? dimensionName(*domain)
               : FailureOr<std::string>(failure());
    if (exact.empty() || failed(extent))
      return operation.emitOpError(
          "packed scalar transfer has no exact TileLang interval");
    predicates.push_back("0 <= " + exact);
    predicates.push_back(exact + " < " + *extent);
  }
  std::string result;
  for (StringRef predicate : predicates) {
    if (!result.empty())
      result += " and ";
    result += predicate;
  }
  return result;
}

FailureOr<std::string> SourceEmitter::tileBoundsPredicate(
    Operation &operation, ArrayRef<std::string> tileExtents,
    bool includeBase) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(relation) || failed(view) ||
      static_cast<size_t>(llvm::count_if(
          *relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != (*view)->shape.size())
    return failure();
  SmallVector<std::string> predicates;
  unsigned tileAxis = 0;
  unsigned sourceAxis = 0;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      if (tileAxis >= tileExtents.size())
        return operation.emitOpError(
            "whole-tile TileLang transfer has too few new-axis extents");
      ++tileAxis;
      continue;
    }
    unsigned axisNumber = sourceAxis++;
    if (term.kind == "full_slice") {
      if (tileAxis >= tileExtents.size())
        return operation.emitOpError(
            "whole-tile TileLang transfer has too few tile extents");
      if (!includeBase)
        predicates.push_back(tileExtents[tileAxis] + " <= " +
                             (*view)->shape[axisNumber]);
      ++tileAxis;
      continue;
    }
    if (term.kind == "static_index")
      continue;
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "whole-tile TileLang transfer has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      if (!includeBase)
        return std::string();
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (tensor.getRank() != 1 || failed(exact) ||
          tileAxis >= tileExtents.size() || tileExtents[tileAxis] != "1")
        return operation.emitOpError(
            "TileLang bulk bounds require one singleton indirect index tile");
      ++tileAxis;
      std::string index = exact->str() + "[0]";
      predicates.push_back("0 <= " + index);
      predicates.push_back(index + " < " + (*view)->shape[axisNumber]);
      continue;
    }
    if (term.kind == "value_index" &&
        !isa<RankedTensorType>(indexed.getType())) {
      FailureOr<target::ScalarIndexSource> source =
          target::traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      if (source->hasDomain() && source->transformed) {
        if (!includeBase)
          return std::string();
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        predicates.push_back("0 <= (" + exact->str() + ")");
        predicates.push_back("(" + exact->str() + ") < " +
                             (*view)->shape[axisNumber]);
      }
      continue;
    }
    FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
    if (failed(axis))
      return failure();
    if (axis->isScalar())
      continue;
    if (tileAxis >= tileExtents.size())
      return operation.emitOpError(
          "whole-tile TileLang transfer has too few tile extents");
    std::string base = axisIndices.lookup(axis->getNode());
    FailureOr<std::string> extent = failure();
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> orderedAxis =
          target::emission::representativeOrderedAxis(planIndex,
                                                      axis->getNode(), operation);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else {
      Operation *domain = kernel.nodes.lookup(axis->getNode());
      if (domain)
        extent = dimensionName(*domain);
    }
    if (base.empty() || failed(extent))
      return operation.emitOpError(
          "whole-tile TileLang transfer has no active axis interval");
    std::string tile = tileExtents[tileAxis++];
    predicates.push_back(includeBase ? base + " + " + tile + " <= " + *extent
                                     : tile + " <= " + *extent);
  }
  if (tileAxis != tileExtents.size())
    return operation.emitOpError(
        "whole-tile TileLang transfer has unused tile extents");
  if (predicates.empty())
    return std::string();
  std::string result = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    result += " and " + predicate.str();
  return result;
}

FailureOr<std::string> SourceEmitter::wholeTileBoundsPredicate(
    Operation &operation, ArrayRef<std::string> tileExtents) {
  return tileBoundsPredicate(operation, tileExtents, true);
}

FailureOr<std::string> SourceEmitter::tileFitsViewPredicate(
    Operation &operation, ArrayRef<std::string> tileExtents) {
  return tileBoundsPredicate(operation, tileExtents, false);
}

FailureOr<std::string> SourceEmitter::elementValidityPredicate(
    ArrayRef<int64_t> tensorAxes, ArrayRef<int64_t> domainNodes,
    ArrayRef<std::string> elementIndices, Operation &consumer) {
  if (tensorAxes.size() != domainNodes.size())
    return consumer.emitOpError(
        "has no TileLang value-validity binding");
  SmallVector<std::string> predicates;
  for (auto [tensorAxis, domainNode] : llvm::zip(tensorAxes, domainNodes)) {
    auto domain = kernel.nodes.find(domainNode);
    auto physical = planIndex.axes.find(domainNode);
    if (tensorAxis < 0 ||
        static_cast<size_t>(tensorAxis) >= elementIndices.size() ||
        domain == kernel.nodes.end() || physical == planIndex.axes.end())
      return consumer.emitOpError(
          "references an unresolved TileLang validity axis");
    plan::AxisOp axis = physical->second;
    if (axis.isScalar())
      continue;
    std::string base = axisIndices.lookup(axis.getNode());
    if (base.empty())
      return consumer.emitOpError(
          "has no active TileLang index for its planned validity axis");
    FailureOr<std::string> extent = failure();
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> orderedAxis =
          target::emission::representativeOrderedAxis(planIndex, axis.getNode(),
                                                      consumer);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else {
      extent = dimensionName(*domain->second);
    }
    if (failed(extent))
      return failure();
    predicates.push_back(base + " + " + elementIndices[tensorAxis] + " < " +
                         *extent);
  }
  if (predicates.empty())
    return std::string("True");
  std::string result = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    result += " and " + predicate.str();
  return result;
}

FailureOr<std::string> SourceEmitter::padElementExpression(
    Value value, StringRef expression, ArrayRef<std::string> elementIndices,
    Operation &consumer) {
  FailureOr<int64_t> valueID =
      target::getValueID(value, kernel, consumer, "TileLang padding lookup");
  if (failed(valueID))
    return failure();
  plan::PaddingOp padding = planIndex.paddings.lookup(*valueID);
  if (!padding)
    return expression.str();
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor)
    return consumer.emitOpError(
        "cannot apply TileLang padding to a non-tensor value");
  std::string dtype = dtypeName(tensor.getElementType(), consumer);
  if (dtype.empty())
    return failure();
  FailureOr<std::string> predicate = elementValidityPredicate(
      padding.getTensorAxes(), padding.getDomainNodes(), elementIndices,
      consumer);
  if (failed(predicate))
    return failure();
  StringRef planned = padding.getFill();
  std::string fill = planned == "negative_infinity"
                         ? "-T.infinity(" + dtype + ")"
                     : planned == "positive_infinity"
                         ? "T.infinity(" + dtype + ")"
                     : planned == "nan" ? "float('nan')"
                     : planned == "true" ? "True"
                     : planned == "false" ? "False"
                     : planned.starts_with("literal_integer:")
                         ? planned.drop_front(16).str()
                     : planned.starts_with("literal_float:")
                         ? planned.drop_front(14).str()
                     : isa<IntegerType, IndexType>(tensor.getElementType())
                         ? "0"
                         : "0.0";
  return "T.if_then_else(" + *predicate + ", " + expression.str() + ", " +
         fill + ")";
}

FailureOr<SmallVector<std::string>>
SourceEmitter::tensorExtents(Operation &operation, unsigned resultIndex,
                             bool physical) {
  auto shapes = operation.getAttrOfType<ArrayAttr>("intent.result_shapes");
  auto shape = shapes && resultIndex < shapes.size()
                   ? dyn_cast<ArrayAttr>(shapes[resultIndex])
                   : ArrayAttr();
  if (!shape)
    return operation.emitOpError("has no canonical tensor shape metadata");
  SmallVector<std::string> extents;
  for (Attribute attribute : shape) {
    auto label = dyn_cast<StringAttr>(attribute);
    if (!label)
      return operation.emitOpError(
          "tensor shape contains a non-symbolic extent");
    auto tile = regionTiles.find(label.getValue());
    if (tile != regionTiles.end())
      extents.push_back(tile->getValue());
    else if (!planIndex.stages.empty() && activeStages.size() == 1 &&
             label.getValue() ==
                 stageFeatureDimensions.lookup(activeStages.front()))
      extents.push_back(stageFeatureTiles.lookup(activeStages.front()));
    else if (label.getValue().starts_with("?region_"))
      return operation.emitOpError(
          "tensor shape region has no TileLang tile binding");
    else
      extents.push_back(physical ? physicalExtent(label.getValue())
                                 : syntax::dimension(label.getValue()));
  }
  return extents;
}

FailureOr<std::string> SourceEmitter::tensorShape(Operation &operation,
                                                  unsigned resultIndex) {
  FailureOr<SmallVector<std::string>> extents =
      tensorExtents(operation, resultIndex);
  if (failed(extents))
    return failure();
  std::string shape = "(";
  for (auto [index, extent] : llvm::enumerate(*extents)) {
    if (index)
      shape += ", ";
    shape += extent;
  }
  if (extents->size() == 1)
    shape += ",";
  return shape + ")";
}

FailureOr<std::string>
SourceEmitter::tensorElement(Value value, ArrayRef<std::string> indices,
                             Operation &consumer) {
  auto found = valueNames.find(value);
  if (found == valueNames.end())
    return consumer.emitOpError("references an unemitted TileLang value");
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor)
    return found->second;
  if (tensor.getRank() > static_cast<int64_t>(indices.size()))
    return consumer.emitOpError("cannot broadcast a higher-rank TileLang value");
  unsigned offset = indices.size() - tensor.getRank();
  std::string result = found->second + "[";
  for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
    if (axis)
      result += ", ";
    int64_t extent = tensor.getDimSize(axis);
    result += extent == 1 ? "0" : indices[offset + axis];
  }
  return result + "]";
}

FailureOr<std::string> SourceEmitter::allocateResult(Operation &operation,
                                                     unsigned resultIndex,
                                                     StringRef space) {
  if (resultIndex >= operation.getNumResults())
    return operation.emitOpError("references a missing result allocation");
  auto tensor = dyn_cast<RankedTensorType>(operation.getResult(resultIndex).getType());
  FailureOr<std::string> shape = tensorShape(operation, resultIndex);
  if (!tensor || failed(shape))
    return operation.emitOpError("requires a ranked TileLang result buffer");
  std::string dtype = dtypeName(tensor.getElementType(), operation);
  if (dtype.empty())
    return failure();
  std::string result = makeResultName(operation, resultIndex);
  StringRef allocator = space == "shared"     ? "T.alloc_shared"
                        : space == "local"    ? "T.alloc_local"
                        : space == "fragment" ? "T.alloc_fragment"
                                               : StringRef();
  if (allocator.empty())
    return operation.emitOpError("has an unsupported TileLang allocation space");
  line(result + " = " + allocator.str() + "(" + *shape + ", " + dtype + ")");
  return result;
}

std::string SourceEmitter::dtypeName(Type type, Operation &consumer) {
  if (type.isF16())
    return "T.float16";
  if (type.isF32())
    return "T.float32";
  if (type.isF64())
    return "T.float64";
  if (type.isBF16())
    return "T.bfloat16";
  if (isa<Float8E4M3FNType>(type))
    return "T.float8_e4m3fn";
  if (isa<Float8E5M2Type>(type))
    return "T.float8_e5m2";
  if (isa<Float8E8M0FNUType>(type))
    return "T.float8_e8m0fnu";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 8)
    return integer.isUnsigned() ? "T.uint8" : "T.int8";
  if (type.isInteger(1))
    return "T.bool";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 16)
    return integer.isUnsigned() ? "T.uint16" : "T.int16";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 32)
    return "T.int32";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 64 && !integer.isUnsigned())
    return "T.int64";
  if (isa<IndexType>(type))
    return "T.int32";
  consumer.emitOpError("uses an unsupported TileLang dtype");
  return {};
}

void SourceEmitter::bindResult(Operation &operation, unsigned index,
                               StringRef name) {
  Value value = operation.getResult(index);
  valueNames[value] = name.str();
  auto owner = stageOutputOwners.find(value);
  if (planIndex.stages.empty() || owner == stageOutputOwners.end())
    return;
  std::string memberTile = stageMemberTiles.lookup(owner->second);
  std::string featureTile = stageFeatureTiles.lookup(owner->second);
  std::string feature = stageFeatureDimensions.lookup(owner->second);
  line("for store_i, store_j in T.Parallel(" + memberTile + ", " +
       featureTile + "):");
  ++indentation;
  line("if member_start + store_i < route_end and "
       "bid_feature * " + featureTile + " + store_j < " + feature + ":");
  ++indentation;
  line(workspaceNames.lookup(value) +
       "[" + addressIndex("member_start + store_i") + ", " +
       addressIndex("bid_feature * " + featureTile + " + store_j") + "] = " +
       name.str() + "[store_i, store_j]");
  --indentation;
  --indentation;
}

std::string SourceEmitter::uniqueName(StringRef candidate, int64_t node) {
  // TileLang lowers these names into generated C++, so even a valid Python
  // identifier such as `signed` may be a target-language keyword.  A stable
  // target prefix keeps author-provided SSA names out of that namespace
  // without changing the canonical Kernel IR name.
  std::string result = "intent_" + candidate.str();
  if (!usedNames.insert(result).second) {
    result += "_" + std::to_string(node);
    usedNames.insert(result);
  }
  return result;
}

std::string SourceEmitter::makeResultName(Operation &operation, unsigned index) {
  auto names = operation.getAttrOfType<ArrayAttr>("intent.result_names");
  auto name = names && index < names.size() ? dyn_cast<StringAttr>(names[index])
                                           : StringAttr();
  FailureOr<int64_t> node = target::getNodeID(operation, "result naming");
  return uniqueName(name ? name.getValue() : "value", *node);
}

std::string SourceEmitter::makeRegionArgumentName(Operation &operation,
                                                  unsigned argumentIndex) {
  auto regions =
      operation.getAttrOfType<ArrayAttr>("intent.region_argument_names");
  auto blocks = regions && !regions.empty() ? dyn_cast<ArrayAttr>(regions[0])
                                            : ArrayAttr();
  auto arguments = blocks && !blocks.empty() ? dyn_cast<ArrayAttr>(blocks[0])
                                             : ArrayAttr();
  auto name = arguments && argumentIndex < arguments.size()
                  ? dyn_cast<StringAttr>(arguments[argumentIndex])
                  : StringAttr();
  FailureOr<int64_t> node = target::getNodeID(operation, "region naming");
  return uniqueName(name ? name.getValue() : "region", *node);
}

void SourceEmitter::line(StringRef text) {
  if (!planIndex.stages.empty()) {
    for (unsigned stage : activeStages)
      stageLine(stage, text, indentation);
    return;
  }
  output.indent(indentation * 4) << text << "\n";
}

LogicalResult emitRealizedKernelSource(
    target::KernelModel kernel, intent::plan::ProgramOp realization,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization, kernel);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return SourceEmitter(std::move(kernel), realization, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .emit();
}

} // namespace intent::tilelang::emission

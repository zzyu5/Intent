#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Lowering/Combiner.h"
#include "Intent/Target/CuTile/Lowering/Passes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cutile::lowering {
namespace {

std::string projectTensorIndex(StringRef value, unsigned valueRank,
                               unsigned groupRank, unsigned groupAxis,
                               unsigned resultRank) {
  if (valueRank == resultRank && groupAxis == 0)
    return value.str();
  unsigned valueAxis = groupAxis + groupRank - valueRank;
  std::string expression = value.str() + "[";
  for (unsigned axis = 0; axis < resultRank; ++axis) {
    if (axis)
      expression += ", ";
    expression += axis >= valueAxis && axis < groupAxis + groupRank ? ":"
                                                                      : "None";
  }
  return expression + "]";
}

} // namespace

FailureOr<PhysicalProgramIndex>
indexPhysicalProgram(intent::plan::ProgramOp physicalProgram,
                 const target::KernelModel &kernel) {
  PhysicalProgramIndex index;
  SmallVector<intent::plan::ReductionOp> reductions;
  SmallVector<intent::plan::ScanOp> scans;
  SmallVector<intent::plan::PointwiseOp> pointwise;
  SmallVector<intent::plan::TransferOp> transfers;
  SmallVector<intent::plan::RangeOp> ranges;
  for (Operation &operation : physicalProgram.getBody().front()) {
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
      if (!buffer || ::intent::target::semanticOperationName(*buffer) != "intent.buffer" ||
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
      auto lowering = value->getAttrOfType<StringAttr>(contractLoweringAttr);
      auto orientation =
          value->getAttrOfType<StringAttr>(contractOrientationAttr);
      auto batched = value->getAttrOfType<BoolAttr>(contractBatchedAttr);
      auto layout = value->getAttrOfType<StringAttr>(scaledContractLayoutAttr);
      if (!lowering || !orientation || !batched ||
          (value.getForm() == "scaled_direct" && !layout))
        return value.emitOpError(
            "has no realized cuTile contraction provider form");
      binding.lowering = lowering.getValue().str();
      binding.lhsSpace = value.getLhsSpace().str();
      binding.rhsSpace = value.getRhsSpace().str();
      binding.accumulatorSpace = value.getAccumulatorSpace().str();
      binding.orientation = orientation.getValue().str();
      binding.batched = batched.getValue();
      if (layout)
        binding.scaledLayout = layout.getValue().str();
      index.contracts[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::SparseContractOp>(operation)) {
      Operation *sparse = kernel.nodes.lookup(value.getNode());
      if (!sparse || ::intent::target::semanticOperationName(*sparse) != "intent.sparse_contract")
        return value.emitOpError("does not bind sparse contraction semantics");
      return sparse->emitOpError(
          "cuTile has no native 2:4 sparse contraction projection");
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
  if (failed(target::lowering::indexAxisRanges(index, ranges, syntax::tile)) ||
      failed(target::lowering::indexCanonicalStructure(index, kernel)))
    return failure();
  for (plan::RaggedOp &ragged : index.ragged) {
    auto route = ragged.operation
                     ? ragged.operation->getAttrOfType<StringAttr>(raggedRouteAttr)
                     : StringAttr();
    if (!route)
      return physicalProgram.emitOpError(
          "has no realized cuTile ragged-route form");
    ragged.route = route.getValue().str();
  }
  target::lowering::indexAxisRoles(index);
  for (auto &entry : index.streams) {
    plan::StreamOp &binding = entry.second;
    auto tile = binding.operation->getAttrOfType<StringAttr>(streamTileAttr);
    if (!tile)
      return binding.emitOpError("has no realized cuTile stream-tile spelling");
    binding.tile = tile.getValue().str();
  }
  index.components = target::lowering::indexPhysicalComponents(index);
  for (intent::plan::ReductionOp value : reductions) {
    auto lowering = value->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = value->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    if (!lowering || !axis)
      return value.emitOpError("has no realized cuTile reduction spelling");
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = axis.getInt();
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    auto lowering = value->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || !index.axes.count(value.getAxisNode()))
      return value.emitOpError("has no realized cuTile scan spelling");
    plan::ScanOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = value.getTensorAxis();
    binding.axisNode = value.getAxisNode();
    index.scans[value.getNode()] = binding;
  }
  for (intent::plan::PointwiseOp value : pointwise) {
    auto lowering = value->getAttrOfType<StringAttr>(pointwiseLoweringAttr);
    auto deferred = value->getAttrOfType<BoolAttr>(pointwiseDeferredAttr);
    if (!lowering || !deferred)
      return value.emitOpError("has no realized cuTile pointwise spelling");
    plan::PointwiseOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    binding.defer = deferred.getValue();
    index.pointwise[value.getNode()] = binding;
  }
  for (intent::plan::TransferOp value : transfers) {
    auto access = value->getAttrOfType<StringAttr>(accessAttr);
    auto bounds = value->getAttrOfType<BoolAttr>(boundsAttr);
    if (!access || !bounds)
      return value.emitOpError("has no realized cuTile transfer form");
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = access.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    binding.explicitBounds = bounds.getValue();
    index.boundaries[value.getNode()] = binding;
  }
  for (const auto &entry : index.axes) {
    const plan::AxisOp &axis = entry.second;
    if (axis.hasRole("contraction_m") &&
        axis.getTileRole().starts_with("row_vector") &&
        !axis.getReuseWorker())
      return axis.emitOpError(
          "cuTile does not support a runtime-sized lane as the matrix-M axis");
  }
  if (!index.target || !index.program) {
    physicalProgram.emitOpError("lacks device or launch decisions for cuTile");
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

ProgramMaterializer::ProgramMaterializer(target::KernelModel kernel,
                             intent::plan::ProgramOp physicalProgram,
                             intent::plan::SearchSpaceOp searchSpace,
                             PhysicalProgramIndex planIndex,
                             SearchIndex searchIndex, raw_ostream &output)
    : kernel(std::move(kernel)), physicalProgram(physicalProgram),
      searchSpace(searchSpace), planIndex(std::move(planIndex)),
      searchIndex(std::move(searchIndex)), output(output) {}

LogicalResult ProgramMaterializer::materialize() {
  return target::materializeSource(*this);
}

LogicalResult ProgramMaterializer::prepare() {
  if (searchIndex.autotune)
    for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
      tuningParameters.emplace_back(
          parameter.getName().getValue().str(),
          cast<StringAttr>(parameter.getValue()).getValue().str());
  if (failed(indexABI()))
    return failure();
  if (!planIndex.ragged.empty() && failed(prepareRaggedMetadata()))
    return failure();
  if (failed(resolvePhysicalBindings()))
    return failure();
  tuneGatherSpelling = searchIndex.autotune &&
                       static_cast<bool>(searchIndex.autotune.getParameterMap().get(
                           "GATHER_SPELLING"));
  auto rowForm = planIndex.program.operation->getAttrOfType<StringAttr>(
      rowOccupancyAttr);
  if (!rowForm)
    return planIndex.program.emitOpError(
        "has no realized cuTile row-occupancy form");
  tuneRowOccupancy = rowForm.getValue() == "delegated";
  if (failed(target::lowering::indexScanProducerOperations(
          kernel, planIndex, scanProducerOwners)))
    return failure();
  if (failed(target::lowering::indexDeferredContractReplays(
          kernel, planIndex, deferredContractReplays,
          deferredContractProducerOwners)))
    return failure();
  for (const auto &entry : planIndex.scans)
    if (failed(target::lowering::verifyScanMaterializedValues(kernel,
                                                              entry.second)))
      return failure();
  if (failed(preparePrivateWorkspaces()))
    return failure();
  if (!planIndex.stages.empty())
    return prepareRaggedStages();
  return success();
}

LogicalResult ProgramMaterializer::preparePrivateWorkspaces() {
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
    if (binding.getResultSpace() != "private_workspace")
      continue;
    Operation *scan = kernel.nodes.lookup(entry.first);
    std::string extent = axisDimensions.lookup(binding.getAxisNode());
    if (!scan || ::intent::target::semanticOperationName(*scan) != "intent.scan" ||
        scan->getNumResults() == 0 || extent.empty())
      return binding.emitOpError("does not bind a workspace-backed scan tensor");
    for (auto [component, result] : llvm::enumerate(scan->getResults())) {
      workspaceNames[result] = "scan_workspace_" +
                               std::to_string(entry.first) + "_" +
                               std::to_string(component);
      scanResults[result] = binding;
    }
    scanExtents[entry.first] = extent;
    for (int64_t valueID : binding.getMaterializedValues()) {
      FailureOr<Value> value = target::lowering::lookupScanMaterializedValue(
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

LogicalResult ProgramMaterializer::registerOperationHandlers(
    target::OperationHandlerRegistry &registry) {
  return registerEmissionHandlers(registry, *this);
}

LogicalResult ProgramMaterializer::indexABI() {
  SmallVector<StringRef> requiredDimensions;
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
               << " has no cuTile runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("cuTile emitter requires ranked views");
    ABIView emitted{&argument, view, tensor, {}};
    auto shape = argument.metadata.getAs<ArrayAttr>("shape");
    if (!shape || shape.size() != static_cast<size_t>(tensor.getRank()))
      return kernel.entry.emitOpError()
             << "view " << argument.name << " has incomplete shape metadata";
    for (auto [axis, extent] : llvm::enumerate(shape)) {
      if (auto symbol = dyn_cast<StringAttr>(extent)) {
        emitted.shape.push_back(symbol.getValue().str());
        uint64_t staticExtent = 0;
        if (!symbol.getValue().getAsInteger(10, staticExtent))
          continue;
        if (!llvm::is_contained(requiredDimensions, symbol.getValue()))
          requiredDimensions.push_back(symbol.getValue());
        if (view.getAccess() != "out" &&
            !dimensionOwners.count(symbol.getValue())) {
          dimensionOwners[symbol.getValue()] =
              argument.name + ".shape[" + std::to_string(axis) + "]";
          dimensionOrder.push_back(symbol.getValue().str());
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
    return kernel.entry.emitOpError("cuTile emitter requires external views");
  for (StringRef dimension : requiredDimensions) {
    if (!dimensionOwners.count(dimension))
      return kernel.entry.emitOpError()
             << "dynamic output dimension " << dimension
             << " has no input or inout ABI owner";
  }
  return success();
}

LogicalResult ProgramMaterializer::resolvePhysicalBindings() {
  std::optional<int64_t> rootNode = planIndex.program.getLoopNode();
  programRoot = rootNode ? kernel.nodes.lookup(*rootNode) : nullptr;
  if (programRoot &&
      ::intent::target::semanticOperationName(*programRoot) !=
          "intent.parallel")
    return planIndex.program.emitOpError(
        "binds a non-parallel explicit program root");
  for (auto &entry : planIndex.axesByRole) {
    Operation *domain = kernel.nodes.lookup(entry.getValue().getNode());
    if (!domain ||
        (::intent::target::semanticOperationName(*domain) != "intent.domain" &&
         ::intent::target::semanticOperationName(*domain) != "intent.ragged_outer" &&
         ::intent::target::semanticOperationName(*domain) != "intent.ragged_member"))
      return entry.getValue().emitOpError("does not bind a logical domain op");
    FailureOr<std::string> dimension = dimensionName(*domain);
    if (failed(dimension))
      return entry.getValue().emitOpError("cannot resolve its source dimension");
    roleDimensions[entry.getValue().getRole()] = *dimension;
    axisDimensions[entry.getValue().getNode()] = *dimension;
  }
  for (const auto &entry : planIndex.axes) {
    plan::AxisOp axis = entry.second;
    Operation *domain = kernel.nodes.lookup(axis.getNode());
    auto resultNodes =
        domain ? domain->getAttrOfType<ArrayAttr>("intent.result_nodes")
               : ArrayAttr();
    auto resultNode = resultNodes && resultNodes.size() == 1
                          ? dyn_cast<IntegerAttr>(resultNodes[0])
                          : IntegerAttr();
    if (!domain || domain->getNumResults() != 1 || !resultNode)
      return axis.emitOpError(
          "cannot index its domain result shape against the cuTile plan");
    FailureOr<std::string> tile = physicalAxisTile(axis);
    if (failed(tile))
      return failure();
    regionTiles["?region_" + std::to_string(resultNode.getInt()) + "_0"] =
        *tile;
    StringRef dimension = axisDimensions.lookup(axis.getNode());
    auto existing = regionTiles.find(dimension);
    if (!dimension.empty() && existing != regionTiles.end() &&
        existing->getValue() != *tile)
      return axis.emitOpError(
          "selects conflicting physical tiles for one logical extent");
    if (!dimension.empty())
      regionTiles[dimension] = *tile;
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
    const target::lowering::RangeBinding *range =
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
    auto program = planIndex.axesByRole.find("program_0");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (program == planIndex.axesByRole.end() ||
        lane == planIndex.axesByRole.end())
      return physicalProgram.emitOpError(
          "persistent rows require program_0 and lane_0 choices");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if ((view.view.getAccess() == "out" ||
           view.view.getAccess() == "inout") &&
          !fixedOutput)
        fixedOutput = &view;
    }
    if (!fixedOutput || fixedOutput->tensor.getRank() < 1 || searchSpace)
      return physicalProgram.emitOpError(
          "fixed persistent rows require one ranked output and no search space");
  }
  if (target::lowering::requiresDelegatedTuning(planIndex) &&
      (!searchSpace || !searchIndex.autotune))
    return physicalProgram.emitOpError(
        "tiled physical components require a delegated cuTile tuner");
  kernelConstants.append(dimensionOrder.begin(), dimensionOrder.end());
  SmallVector<StringRef> logicalBlockExtents;
  logicalBlockExtents.reserve(planIndex.blockExtents.size());
  for (const auto &entry : planIndex.blockExtents)
    logicalBlockExtents.push_back(entry.getKey());
  llvm::sort(logicalBlockExtents);
  for (StringRef logicalExtent : logicalBlockExtents)
    blockExtentConstants.emplace_back(physicalExtent(logicalExtent),
                                      logicalExtent.str());
  return success();
}

LogicalResult ProgramMaterializer::prepareRaggedMetadata() {
  raggedRuntimes.reserve(planIndex.ragged.size());
  for (plan::RaggedOp ragged : planIndex.ragged) {
    RaggedRuntime runtime;
    runtime.binding = ragged;
    runtime.relation = kernel.nodes.lookup(ragged.getNode());
    runtime.outer = kernel.nodes.lookup(ragged.getOuterNode());
    StringRef outerName =
        runtime.outer ? ::intent::target::semanticOperationName(*runtime.outer)
                      : StringRef();
    if (!runtime.relation ||
        ::intent::target::semanticOperationName(*runtime.relation) !=
            "intent.ragged" ||
        (outerName != "intent.ragged_outer" &&
         outerName != "intent.ragged_member") ||
        ragged.getMemberNodes().empty())
      return ragged.emitOpError(
          "does not resolve to canonical ragged ownership operations");
    for (int64_t memberNode : ragged.getMemberNodes()) {
      Operation *member = kernel.nodes.lookup(memberNode);
      plan::AxisOp axis = planIndex.axes.lookup(memberNode);
      if (!member ||
          ::intent::target::semanticOperationName(*member) != "intent.ragged_member" || !axis)
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
                ::intent::target::semanticOperationName(*offsetsLoad) == "intent.view_load" &&
                offsetsLoad->getNumOperands() == 1
            ? lookupView(offsetsLoad->getOperand(0), *runtime.relation)
            : FailureOr<ABIView *>(failure());
    Operation *memberSource = runtime.relation->getOperand(1).getDefiningOp();
    Operation *memberDim = memberSource && memberSource->getNumOperands() >= 2
                               ? memberSource->getOperand(1).getDefiningOp()
                               : nullptr;
    FailureOr<ABIView *> membersView =
        memberDim && ::intent::target::semanticOperationName(*memberDim) == "intent.dim" &&
                memberDim->getNumOperands() == 1
            ? lookupView(memberDim->getOperand(0), *runtime.relation)
            : FailureOr<ABIView *>(failure());
    if (failed(offsets) || (*offsets)->tensor.getRank() != 1 ||
        failed(membersView))
      return runtime.relation->emitOpError(
          "requires canonical offsets and member-source views");
    runtime.offsets = *offsets;
    runtime.membersView = *membersView;
    if (ragged.getRoute() == "indexed") {
      if (runtime.relation->getNumOperands() != 4)
        return runtime.relation->emitOpError(
            "does not match its selected indexed cuTile ragged route");
      Operation *indicesLoad = runtime.relation->getOperand(3).getDefiningOp();
      FailureOr<ABIView *> indices =
          indicesLoad && indicesLoad->getNumOperands() == 1
              ? lookupView(indicesLoad->getOperand(0), *runtime.relation)
              : FailureOr<ABIView *>(failure());
      if (failed(indices) || (*indices)->tensor.getRank() != 1)
        return runtime.relation->emitOpError(
            "requires a canonical rank-one member-index view");
      runtime.indices = *indices;
    } else if (ragged.getRoute() != "compact" ||
               runtime.relation->getNumOperands() != 3)
      return runtime.relation->emitOpError(
          "does not match its selected compact cuTile ragged route");
    unsigned position = raggedRuntimes.size();
    raggedRuntimeByRelation[ragged.getNode()] = position;
    raggedRuntimesByAxis[ragged.getOuterNode()].push_back(position);
    for (int64_t member : ragged.getMemberNodes())
      raggedRuntimesByAxis[member].push_back(position);
    raggedRuntimes.push_back(std::move(runtime));
  }
  return success();
}

LogicalResult ProgramMaterializer::prepareRaggedStages() {
  if (failed(target::lowering::indexStageOperations(
          kernel, planIndex, operationStages)))
    return failure();
  stageBodies.resize(planIndex.stages.size());
  for (auto [position, stage] : llvm::enumerate(planIndex.stages)) {
    Operation *contract = kernel.nodes.lookup(stage.getNode());
    if (!contract || ::intent::target::semanticOperationName(*contract) != "intent.contract" ||
        contract->getNumResults() != 1)
      return stage.emitOpError("does not bind one canonical contraction");
    auto resultType = dyn_cast<RankedTensorType>(contract->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2)
      return stage.emitOpError("requires a rank-two staged contraction result");
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
    Operation *members = nullptr;
    for (int64_t operationNode : stage.getOperations()) {
      Operation *candidate = kernel.nodes.lookup(operationNode);
      if (!candidate ||
          ::intent::target::semanticOperationName(*candidate) != "intent.members")
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
          !isa<RankedTensorType>(found->second.getType()) ||
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

bool ProgramMaterializer::selectOperation(Operation &operation) {
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
         !target::lowering::isAbsorbedStagedAccessMetadata(planIndex,
                                                           operation);
}

void ProgramMaterializer::stageLine(unsigned stage, StringRef text, unsigned indent) {
  stageBodies[stage].append(indent * 4, ' ');
  stageBodies[stage] += text.str();
  stageBodies[stage] += "\n";
}

void ProgramMaterializer::stageGuardedGather(
    unsigned stage, StringRef result, StringRef array, StringRef indices,
    StringRef padding, StringRef valid, unsigned indent) {
  auto gather = [&](StringRef mask) {
    return result.str() + " = " +
           syntax::gather(array, indices, padding, mask);
  };
  if (tuneGatherSpelling) {
    stageLine(stage, "if GATHER_SPELLING:", indent);
    stageLine(stage, gather(valid), indent + 1);
    stageLine(stage, "else:", indent);
    stageLine(stage, gather(""), indent + 1);
    stageLine(stage,
              result.str() + " = ct.where(" + valid.str() + ", " +
                  result.str() + ", " + padding.str() + ")",
              indent + 1);
    return;
  }
  stageLine(stage, gather(""), indent);
  stageLine(stage,
            result.str() + " = ct.where(" + valid.str() + ", " +
                result.str() + ", " + padding.str() + ")",
            indent);
}

void ProgramMaterializer::bindResult(Operation &operation, unsigned index,
                               StringRef name) {
  Value value = operation.getResult(index);
  valueNames[value] = name.str();
  auto outputStage = stageOutputOwners.find(value);
  if (planIndex.stages.empty() || outputStage == stageOutputOwners.end())
    return;
  line("ct.scatter(" + workspaceNames.lookup(value) +
       ", (" + addressIndex("safe_member_offsets[:, None]") + ", " +
       addressIndex("offs_feature[None, :]") + "), " + name.str() +
       ", check_bounds=True)");
}

void ProgramMaterializer::emitImports() {
  output << "import math\n";
  output << "import cuda.tile as ct\n";
  output << "import torch\n";
  if (!planIndex.components.reusedAxes.empty())
    output << "from intent.runtime.tuning.cutile import tune_persistent_row\n";
  if (searchSpace) {
    output << "from math import ceil\n";
    output << "from cuda.tile.tune import exhaustive_search\n";
    output << "from intent.runtime.tuning.cutile import autotune_configurations, autotune_timeout\n";
    output << "\n_PARAMETER_MAP = {";
    for (auto [index, mapping] : llvm::enumerate(tuningParameters)) {
      if (index)
        output << ", ";
      output << "'" << mapping.first << "': '" << mapping.second << "'";
    }
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP)\n";
    output << "_TUNE_TIMEOUT = autotune_timeout(_PARAMETER_MAP)\n";
  }
  if (tuneRowOccupancy)
    output << "from intent.runtime.tuning.cutile import tune_row_occupancy\n";
  output << "\nConstInt = ct.Constant[int]\n";
  output << "\n\n";
}

LogicalResult ProgramMaterializer::emitHelpers() {
  FailureOr<SmallVector<func::FuncOp>> combiners =
      target::lowering::collectCombiners(kernel.entry);
  if (failed(combiners))
    return failure();
  for (func::FuncOp function : *combiners) {
    auto expression = [&](Operation &operation,
                          ArrayRef<std::string> operands)
        -> FailureOr<std::string> {
      FailureOr<std::string> role =
          target::lowering::pointwiseRole(operation);
      FailureOr<StringRef> spelling =
          succeeded(role)
              ? syntax::pointwise(&operation, *role, "private_fragment")
              : FailureOr<StringRef>(failure());
      if (failed(spelling))
        return failure();
      return target::lowering::renderPythonPointwiseExpression(
          operation, operands, *spelling,
          [&](Type type) -> FailureOr<std::string> {
            std::string dtype = dtypeName(type, operation);
            return dtype.empty() ? FailureOr<std::string>(failure())
                                 : FailureOr<std::string>(std::move(dtype));
          });
    };
    FailureOr<std::string> source = target::lowering::renderPythonCombiner(
        function, function.getName(), "", expression);
    if (failed(source))
      return failure();
    output << *source;
  }
  WalkResult projected = kernel.entry.walk([&](Operation *operation) {
    if (!target::lowering::hasGenericCombiner(*operation) ||
        operation->getAttrOfType<StringAttr>("intent.combine_builtin"))
      return WalkResult::advance();
    FailureOr<target::lowering::CombinerUse> combiner =
        target::lowering::resolveCombiner(*operation);
    if (failed(combiner))
      return WalkResult::interrupt();
    if (combiner->captureCount == 0)
      return WalkResult::advance();
    FailureOr<std::string> source =
        target::lowering::renderPythonCombinerProjection(
            *operation, "", "ct.where");
    if (failed(source))
      return WalkResult::interrupt();
    output << *source;
    return WalkResult::advance();
  });
  if (projected.wasInterrupted())
    return failure();
  return success();
}

LogicalResult ProgramMaterializer::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  if (!planIndex.stages.empty()) {
    if (!searchIndex.autotune)
      return physicalProgram.emitOpError(
          "cannot resolve the delegated staged tuner");
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      auto runtime = stageRaggedRuntime.find(stage);
      if (runtime == stageRaggedRuntime.end())
        return planIndex.stages[stage].emitOpError(
            "has no staged ragged runtime binding");
      RaggedRuntime &ragged = raggedRuntimes[runtime->second];
      bool compact = !ragged.indices;
      ABIView *offsets = ragged.offsets;
      ABIView *indices = ragged.indices;
      llvm::raw_string_ostream source(stageBodies[stage]);
      source << "@ct.kernel\ndef " << kernelName << "_stage_" << stage << "(";
      bool first = true;
      auto parameter = [&](StringRef value) {
        if (!first)
          source << ", ";
        source << value;
        first = false;
      };
      for (ABIView &view : views)
        parameter(view.argument->name);
      for (ABIScalar &scalar : scalars) {
        StringRef annotation = isa<FloatType>(scalar.type) ? "float" : "int";
        parameter(scalar.name + ": " + annotation.str());
      }
      for (const std::string &dimension : dimensionOrder)
        parameter(dimension + ": ConstInt");
      for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
        parameter(physicalExtent + ": ConstInt");
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          parameter(workspaceNames.lookup(kernel.values.lookup(valueID)));
      if (!compact)
        parameter("MAX_ROUTES: ConstInt");
      for (const auto &configParameter : tuningParameters)
        parameter(configParameter.first + ": ConstInt");
      source << "):\n";
      source.flush();

      std::string feature = stageFeatureDimensions.lookup(stage);
      std::string featureTile = stageFeatureTiles.lookup(stage);
      std::string memberTile = stageMemberTiles.lookup(stage);
      stageLine(stage, "bid_feature = " +
                           addressIndex("ct.bid(" +
                                        std::to_string(
                                            stageFeatureWorkers.lookup(stage)) +
                                        ")"));
      stageLine(stage, "bid_expert_route = " +
                           addressIndex("ct.bid(" +
                                        std::to_string(
                                            stageMemberWorkers.lookup(stage)) +
                                        ")"));
      if (compact) {
        plan::AxisOp outerAxis =
            planIndex.axes.lookup(ragged.binding.getOuterNode());
        std::string experts =
            outerAxis ? roleDimensions.lookup(
                            "program_" +
                            std::to_string(outerAxis.getProgramOrder()))
                      : std::string();
        if (experts.empty())
          return ragged.binding.emitOpError(
              "has no program-owned outer-axis dimension");
        stageLine(stage, "expert = " + addressIndex("0"));
        stageLine(stage, "route_tile = " + addressIndex("0"));
        stageLine(stage, "tile_cursor = " + addressIndex("0"));
        stageLine(stage, "for candidate in range(" + experts + "):");
        stageLine(stage, "candidate_begin = ct.load(" +
                             offsets->argument->name +
                             ", index=" + addressIndex("candidate") +
                             ", shape=())",
                  2);
        stageLine(stage, "candidate_end = ct.load(" +
                             offsets->argument->name +
                             ", index=" + addressIndex("candidate + 1") +
                             ", shape=())",
                  2);
        stageLine(stage,
                  "candidate_tiles = " +
                      addressIndex("ct.cdiv(candidate_end - candidate_begin, " +
                                   memberTile + ")"),
                  2);
        stageLine(stage,
                  "owns_tile = (bid_expert_route >= tile_cursor) & "
                  "(bid_expert_route < tile_cursor + candidate_tiles)",
                  2);
        stageLine(stage,
                  "expert = ct.where(owns_tile, " + addressIndex("candidate") +
                      ", expert)",
                  2);
        stageLine(stage,
                  "route_tile = ct.where(owns_tile, bid_expert_route - "
                  "tile_cursor, route_tile)",
                  2);
        stageLine(stage, "tile_cursor += candidate_tiles", 2);
      } else {
        stageLine(stage,
                  "num_route_tiles = ct.cdiv(MAX_ROUTES, " + memberTile + ")");
        stageLine(stage, "expert = bid_expert_route // num_route_tiles");
        stageLine(stage, "route_tile = bid_expert_route % num_route_tiles");
      }
      stageLine(stage, "route_begin = ct.load(" +
                           offsets->argument->name +
                           ", index=" + addressIndex("expert") +
                           ", shape=())");
      stageLine(stage, "route_end = ct.load(" +
                           offsets->argument->name +
                           ", index=" + addressIndex("expert + 1") +
                           ", shape=())");
      stageLine(stage,
                "member_offsets = " + addressIndex("route_begin") + " + " +
                    addressIndex("route_tile") + " * " + memberTile + " + " +
                    addressIndex("ct.arange(" + memberTile +
                                 ", dtype=ct.int32)"));
      stageLine(stage, "member_mask = member_offsets < route_end");
      stageLine(stage,
                "safe_member_offsets = ct.where(member_mask, member_offsets, " +
                    stageMemberDimensions.lookup(stage) + ")");
      if (indices) {
        stageGuardedGather(stage, "routes", indices->argument->name,
                           addressIndex("member_offsets"), "0", "member_mask");
      } else {
        stageLine(stage, "routes = member_offsets");
      }
      stageLine(stage,
                "offs_feature = " + addressIndex("bid_feature") +
                    " * " + featureTile + " + " +
                    addressIndex("ct.arange(" + featureTile +
                                 ", dtype=ct.int32)"));
      stageLine(stage, "feature_mask = offs_feature < " + feature);
    }
    return success();
  }
  if (!planIndex.components.reusedAxes.empty()) {
    output << "@ct.kernel\n";
    programIndex = makeRegionArgumentName(*programRoot, 0);
    vectorIndex = makeResultName(*vectorDomain, 0);
    valueNames[programRoot->getRegion(0).front().getArgument(0)] = programIndex;
    valueNames[vectorDomain->getResult(0)] = vectorIndex;
    plan::AxisOp lane = planIndex.axesByRole.lookup("lane_0");
    if (lane)
      axisIndices[lane.getNode()] = vectorIndex;
  } else {
    output << "@ct.kernel\n";
  }
  output << "def " << kernelName << "(";
  bool first = true;
  auto emitParameter = [&](StringRef parameter) {
    if (!first)
      output << ", ";
    output << parameter;
    first = false;
  };
  for (ABIView &view : views)
    emitParameter(view.argument->name);
  for (ABIScalar &scalar : scalars) {
    StringRef annotation = isa<FloatType>(scalar.type) ? "float" : "int";
    emitParameter(scalar.name + ": " + annotation.str());
  }
  for (Operation *buffer : privateWorkspaceBuffers)
    emitParameter(workspaceNames.lookup(buffer->getResult(0)));
  for (const auto &entry : planIndex.scans) {
    if (entry.second.getResultSpace() != "private_workspace")
      continue;
    Operation *scan = kernel.nodes.lookup(entry.first);
    for (Value result : scan->getResults())
      emitParameter(workspaceNames.lookup(result));
    for (int64_t valueID : entry.second.getMaterializedValues())
      emitParameter(workspaceNames.lookup(kernel.values.lookup(valueID)));
  }
  if (!planIndex.components.reusedAxes.empty()) {
    for (const std::string &dimension : kernelConstants)
      emitParameter(dimension + ": ConstInt");
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      emitParameter(physicalExtent + ": ConstInt");
    for (StringRef parameter : {"N_ROWS: ConstInt", "TILE_SIZE: ConstInt",
                                "DIM_COLS: ConstInt"})
      emitParameter(parameter);
  } else {
    for (const std::string &dimension : kernelConstants)
      emitParameter(dimension + ": ConstInt");
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      emitParameter(physicalExtent + ": ConstInt");
    if (searchIndex.autotune)
      for (const auto &configParameter : tuningParameters)
        emitParameter(configParameter.first + ": ConstInt");
  }
  output << "):\n";
  if (!programRoot && failed(emitProgramBindings()))
    return failure();
  return success();
}

LogicalResult ProgramMaterializer::emitWrapper() {
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
        integer && integer.getWidth() == 32)
      return "torch.int32";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 64)
      return "torch.int64";
    return {};
  };
  auto emitViewCapabilityCheck = [&](const ABIView &view) {
    output << "    if any(extent == 0 for extent in "
           << view.argument->name << ".shape):\n";
    output << "        raise NotImplementedError('cuTile does not support "
              "zero-extent external views in this compiler')\n";
    output << "    if sum(max(0, extent - 1) * abs(stride) for extent, stride "
              "in zip("
           << view.argument->name << ".shape, " << view.argument->name
           << ".stride())) > 2147483647:\n";
    output << "        raise NotImplementedError('cuTile cannot encode this "
              "64-bit external-buffer address in its current tensor "
              "descriptor')\n";
  };
  auto emitBlockExtentConstants = [&]() {
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << "    " << physicalExtent << " = 1 << (" << logicalExtent
             << " - 1).bit_length()\n";
  };
  auto emitPrivateWorkspaceAllocations = [&]() -> LogicalResult {
    for (Operation *buffer : privateWorkspaceBuffers) {
      FailureOr<std::string> size =
          target::lowering::privateWorkspaceElementCount(
              *buffer, planIndex, axisDimensions);
      FailureOr<std::string> initializer =
          target::lowering::logicalBufferPythonInitializer(*buffer);
      FailureOr<target::LogicalBufferInfo> info =
          target::getLogicalBufferInfo(*buffer);
      StringRef dtype =
          succeeded(info) ? torchDtype(info->elementType) : StringRef();
      if (failed(size) || failed(initializer) || failed(info) || dtype.empty())
        return buffer->emitOpError(
            "has no supported cuTile private-workspace allocation");
      output << "    if " << *size << " > 2147483647:\n";
      output << "        raise NotImplementedError('cuTile private workspace "
                "exceeds its current 32-bit address projection')\n";
      output << "    " << workspaceNames.lookup(buffer->getResult(0))
             << " = torch.full((" << *size << ",), " << *initializer
             << ", device=_DEVICE, dtype=" << dtype << ")\n";
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "private_workspace")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      FailureOr<std::string> size = target::lowering::scanWorkspaceElementCount(
          entry.second, scanExtents.lookup(entry.first), planIndex,
          axisDimensions);
      if (!scan || scan->getNumResults() == 0 || failed(size))
        return entry.second.emitOpError(
            "has no supported cuTile scan-workspace allocation");
      output << "    if " << *size << " > 2147483647:\n";
      output << "        raise NotImplementedError('cuTile scan workspace exceeds "
                "its current 32-bit address projection')\n";
      for (Value result : scan->getResults()) {
        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        StringRef dtype = resultType ? torchDtype(resultType.getElementType())
                                     : StringRef();
        if (dtype.empty())
          return entry.second.emitOpError(
              "has an unsupported cuTile scan-workspace component dtype");
        output << "    " << workspaceNames.lookup(result)
               << " = torch.empty((" << *size
               << ",), device=_DEVICE, dtype=" << dtype << ")\n";
      }
      for (int64_t valueID : entry.second.getMaterializedValues()) {
        Value value = kernel.values.lookup(valueID);
        auto tensor = dyn_cast<RankedTensorType>(value.getType());
        StringRef valueDtype =
            tensor ? torchDtype(tensor.getElementType()) : StringRef();
        if (!tensor || tensor.getRank() != 1 || valueDtype.empty())
          return entry.second.emitOpError(
              "requires rank-one materialized scan producer values");
        output << "    " << workspaceNames.lookup(value)
               << " = torch.empty((" << *size
               << ",), device=_DEVICE, dtype=" << valueDtype << ")\n";
      }
    }
    return success();
  };
  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n";

  if (!planIndex.stages.empty()) {
    SmallVector<ABIView *> inputs;
    ABIView *merge = nullptr;
    for (ABIView &view : views) {
      StringRef access = view.view.getAccess();
      if (access == "in") {
        inputs.push_back(&view);
        continue;
      }
      if (access == "out" || access == "inout") {
        if (merge)
          return kernel.entry.emitOpError(
              "ragged stages require input views and one inout merge view");
        merge = &view;
        continue;
      }
      return kernel.entry.emitOpError(
          "ragged stages require input views and one inout merge view");
    }
    if (!merge)
      return kernel.entry.emitOpError("ragged stages have no merge destination");
    for (const std::string &body : stageBodies)
      output << "\n" << body;

    output << "\n_TUNE_CACHE = {}\n\n\n";

    output << "def launch(";
    bool firstParameter = true;
    for (ABIView &view : views) {
      if (!firstParameter)
        output << ", ";
      output << view.argument->name;
      firstParameter = false;
    }
    for (ABIScalar &scalar : scalars) {
      if (!firstParameter)
        output << ", ";
      output << scalar.name;
      firstParameter = false;
    }
    output << "):\n";
    for (ABIView &view : views) {
      StringRef dtype = torchDtype(view.tensor.getElementType());
      if (dtype.empty())
        return kernel.entry.emitOpError("has an unsupported staged ABI dtype");
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
      emitViewCapabilityCheck(view);
    }
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    emitBlockExtentConstants();
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
          return stage.emitOpError("has an unsupported workspace tensor");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty(("
               << stageMemberDimensions.lookup(stage.getOrdinal()) << ", "
               << stageFeatureDimensions.lookup(stage.getOrdinal())
               << "), device=_DEVICE, dtype=" << dtype << ")\n";
      }
    output << "    stream = torch.cuda.current_stream()\n";
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      RaggedRuntime &ragged =
          raggedRuntimes[stageRaggedRuntime.lookup(stage)];
      bool compact = !ragged.indices;
      std::string suffix = std::to_string(ragged.binding.getNode());
      plan::AxisOp outerAxis = planIndex.axes.lookup(ragged.binding.getOuterNode());
      std::string experts =
          outerAxis ? roleDimensions.lookup(
                          "program_" + std::to_string(outerAxis.getProgramOrder()))
                    : std::string();
      if (experts.empty())
        return ragged.binding.emitOpError(
            "has no program-owned outer-axis dimension");
      std::string feature = stageFeatureDimensions.lookup(stage);
      std::string featureTile = stageFeatureTiles.lookup(stage);
      std::string memberTile = stageMemberTiles.lookup(stage);
      output << "    cache_key_" << stage << " = (" << stage;
      for (const std::string &dimension : dimensionOrder)
        output << ", " << dimension;
      if (compact)
        output << ", route_lengths_" << suffix;
      for (ABIView *input : inputs)
        output << ", " << input->argument->name << ".dtype";
      output << ", str(_DEVICE))\n";
      output << "    if cache_key_" << stage << " not in _TUNE_CACHE:\n";
      output << "        with ct.compiler_timeout(_TUNE_TIMEOUT):\n";
      output << "            result = exhaustive_search(\n";
      output << "                _CONFIGS,\n                stream,\n";
      output << "                lambda cfg: (ceil(" << feature
             << " / cfg." << featureTile << "), ";
      if (compact)
        output << "sum(ceil(length / cfg." << memberTile << ") for length in "
                  "route_lengths_"
               << suffix << ")";
      else
        output << experts << " * ceil(max_routes_" << suffix
               << " / cfg." << memberTile << ")";
      output << ", 1),\n";
      output << "                " << kernelName << "_stage_" << stage
             << ",\n";
      output << "                lambda cfg: (";
      for (ABIView &view : views) {
        if (planIndex.stages[stage].getOutputs().empty() && &view == merge)
          output << (target::lowering::stageUsesScatterReduction(
                         planIndex.stages[stage], kernel)
                         ? "torch.zeros_like("
                         : "torch.empty_like(")
                 << view.argument->name << "), ";
        else if (view.view.getAccess() == "inout")
          output << view.argument->name << ".clone(), ";
        else
          output << view.argument->name << ", ";
      }
      for (ABIScalar &scalar : scalars)
        output << scalar.name << ", ";
      for (const std::string &dimension : dimensionOrder)
        output << dimension << ", ";
      for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
        output << physicalExtent << ", ";
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
      if (!compact)
        output << "max_routes_" << suffix << ", ";
      for (const auto &parameter : tuningParameters)
        output << "cfg." << parameter.first << ", ";
      output << "),\n";
      output << "                lambda cfg: {'num_ctas': cfg.num_ctas, 'occupancy': cfg.occupancy},\n";
      output << "            )\n";
      output << "        best = result.best.config\n";
      output << "        _TUNE_CACHE[cache_key_" << stage << "] = (best, "
             << kernelName << "_stage_" << stage
             << ".replace_hints(num_ctas=best.num_ctas, occupancy=best.occupancy))\n";
      output << "    best, tuned_kernel = _TUNE_CACHE[cache_key_" << stage
             << "]\n";
      output << "    grid = (ceil(" << feature
             << " / best." << featureTile << "), ";
      if (compact)
        output << "sum(ceil(length / best." << memberTile << ") for length in "
                  "route_lengths_"
               << suffix << ")";
      else
        output << experts << " * ceil(max_routes_" << suffix
               << " / best." << memberTile << ")";
      output << ", 1)\n";
      output << "    ct.launch(stream, grid, tuned_kernel, (";
      for (ABIView &view : views)
        output << view.argument->name << ", ";
      for (ABIScalar &scalar : scalars)
        output << scalar.name << ", ";
      for (const std::string &dimension : dimensionOrder)
        output << dimension << ", ";
      for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
        output << physicalExtent << ", ";
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
      if (!compact)
        output << "max_routes_" << suffix << ", ";
      for (const auto &parameter : tuningParameters)
        output << "best." << parameter.first << ", ";
      output << "))\n";
    }
    output << "    return " << merge->argument->name << "\n\n\n";
    output << "def run(";
    for (auto [index, view] : llvm::enumerate(inputs)) {
      if (index)
        output << ", ";
      output << view->argument->name;
    }
    output << "):\n";
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    output << "    " << merge->argument->name << " = torch."
           << (target::lowering::planUsesScatterReduction(planIndex, kernel)
                   ? "zeros"
                   : "empty")
           << "((";
    for (auto [axis, extent] : llvm::enumerate(merge->shape)) {
      if (axis)
        output << ", ";
      output << extent;
    }
    output << "), device=_DEVICE, dtype="
           << torchDtype(merge->tensor.getElementType()) << ")\n";
    output << "    launch(";
    bool first = true;
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

  if (!planIndex.components.reusedAxes.empty()) {
    SmallVector<ABIView *> inputs;
    SmallVector<ABIView *> outputs;
    for (ABIView &view : views) {
      if (view.view.getAccess() == "in" || view.view.getAccess() == "inout")
        inputs.push_back(&view);
      else if (view.view.getAccess() == "out")
        outputs.push_back(&view);
      else
        return kernel.entry.emitOpError(
            "persistent-row wrapper supports input and output views");
    }
    if (inputs.empty())
      return kernel.entry.emitOpError(
          "persistent-row wrapper requires an input view");
    bool hasInOut = llvm::any_of(views, [](const ABIView &view) {
      return view.view.getAccess() == "inout";
    });
    if (outputs.empty() && !hasInOut)
      return kernel.entry.emitOpError(
          "persistent-row wrapper requires a writable view");
    plan::AxisOp laneAxis = planIndex.axesByRole.lookup("lane_0");
    if (!laneAxis)
      return physicalProgram.emitOpError(
          "persistent-row wrapper has no selected lane range");
    FailureOr<std::string> rowTile = physicalAxisTile(laneAxis);
    if (failed(rowTile))
      return physicalProgram.emitOpError(
          "persistent-row wrapper has no selected lane range");
    output << "_ROW_TUNE_CACHE = {}\n\n\n";
    output << "def launch(";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    output << "):\n";
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    emitBlockExtentConstants();
    for (ABIView &view : views) {
      StringRef dtype = torchDtype(view.tensor.getElementType());
      if (dtype.empty())
        return kernel.entry.emitOpError("has an unsupported cuTile ABI dtype");
      output << "    if " << view.argument->name << ".device != _DEVICE:\n";
      output << "        raise ValueError('" << view.argument->name
             << " must reside on the realized CUDA device')\n";
      output << "    if " << view.argument->name << ".dtype != " << dtype
             << ":\n";
      output << "        raise ValueError('" << view.argument->name
             << " has the wrong dtype')\n";
      output << "    if " << view.argument->name << ".ndim != "
             << view.tensor.getRank() << ":\n";
      output << "        raise ValueError('" << view.argument->name
             << " has the wrong rank')\n";
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
      emitViewCapabilityCheck(view);
      output << "    if not " << view.argument->name << ".is_contiguous():\n";
      output << "        raise ValueError('persistent-row views must be contiguous')\n";
    }
    std::string rowDimension = roleDimensions.lookup("program_0");
    std::string columnDimension = roleDimensions.lookup("lane_0");
    std::string rowOwner = dimensionOwners.lookup(rowDimension);
    std::string columnOwner = dimensionOwners.lookup(columnDimension);
    output << "    n_rows = " << (rowOwner.empty() ? rowDimension : rowOwner)
           << "\n";
    output << "    n_cols = "
           << (columnOwner.empty() ? columnDimension : columnOwner) << "\n";
    output << "    tile_size = " << *rowTile << "\n";
    output << "    cache_key = (n_rows, n_cols";
    for (ABIView *input : inputs)
      output << ", " << input->argument->name << ".dtype";
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    for (const std::string &dimension : kernelConstants)
      output << ", " << dimension;
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << ", " << physicalExtent;
    output << ", str(_DEVICE))\n";
    output << "    stream = torch.cuda.current_stream()\n";
    output << "    if cache_key not in _ROW_TUNE_CACHE:\n";
    output << "        _ROW_TUNE_CACHE[cache_key] = tune_persistent_row(\n";
    output << "            stream, n_rows, _DEVICE, " << kernelName
           << ", lambda _: (";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
      if (view.view.getAccess() == "inout")
        output << ".clone()";
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    for (const std::string &dimension : kernelConstants)
      output << ", " << dimension;
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << ", " << physicalExtent;
    output << ", n_rows, tile_size, n_cols))\n";
    output << "    selection = _ROW_TUNE_CACHE[cache_key]\n";
    output << "    return ct.launch(stream, selection.grid, selection.kernel, (";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    for (const std::string &dimension : kernelConstants)
      output << ", " << dimension;
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << ", " << physicalExtent;
    output << ", n_rows, tile_size, n_cols))\n\n\n";
    output << "def run(";
    for (auto [index, view] : llvm::enumerate(inputs)) {
      if (index)
        output << ", ";
      output << view->argument->name;
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    output << "):\n";
    for (const std::string &dimension : dimensionOrder)
      output << "    " << dimension << " = "
             << dimensionOwners.lookup(dimension) << "\n";
    for (ABIView *result : outputs) {
      StringRef outputDtype = torchDtype(result->tensor.getElementType());
      if (outputDtype.empty())
        return kernel.entry.emitOpError("has an unsupported cuTile output dtype");
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
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
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

  SmallVector<ABIView *> inputs;
  SmallVector<ABIView *> outputs;
  for (ABIView &view : views) {
    if (view.view.getAccess() == "in" || view.view.getAccess() == "inout")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out")
      outputs.push_back(&view);
    else
      return kernel.entry.emitOpError(
          "autotuned cuTile wrapper supports input and output views");
  }
  bool hasInOut = llvm::any_of(views, [](const ABIView &view) {
    return view.view.getAccess() == "inout";
  });
  if (outputs.empty() && !hasInOut)
    return kernel.entry.emitOpError(
        "autotuned cuTile wrapper has no writable views");
  bool requiresE8M0 = llvm::any_of(planIndex.contracts, [](const auto &entry) {
    return entry.second.getLowering() == "ct.mma_scaled";
  });

  output << "_TUNE_CACHE = {}\n\n\n";
  output << "def launch(";
  bool firstParameter = true;
  for (ABIView &view : views) {
    if (!firstParameter)
      output << ", ";
    output << view.argument->name;
    firstParameter = false;
  }
  for (ABIScalar &scalar : scalars) {
    if (!firstParameter)
      output << ", ";
    output << scalar.name;
    firstParameter = false;
  }
  output << "):\n";
  if (requiresE8M0) {
    output << "    if torch.cuda.get_device_capability(_DEVICE)[0] < 10:\n";
    output << "        raise NotImplementedError('cuTile E8M0 scaled MMA is "
              "not supported by this CUDA device')\n";
  }
  for (ABIView &view : views) {
    StringRef dtype = torchDtype(view.tensor.getElementType());
    if (dtype.empty())
      return kernel.entry.emitOpError("has an unsupported cuTile ABI dtype");
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
    emitViewCapabilityCheck(view);
  }
  for (const std::string &dimension : dimensionOrder)
    output << "    " << dimension << " = " << dimensionOwners.lookup(dimension)
           << "\n";
  emitBlockExtentConstants();
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
  if (failed(emitPrivateWorkspaceAllocations()))
    return failure();
  output << "    stream = torch.cuda.current_stream()\n";
  if (!searchIndex.autotune) {
    SmallVector<plan::AxisOp> axes =
        target::lowering::orderedProgramAxes(planIndex);
    if (planIndex.program.getPersistent() ||
        llvm::any_of(axes, [](plan::AxisOp axis) { return !axis.isScalar(); }))
      return physicalProgram.emitOpError(
          "untuned cuTile launch requires scalar non-persistent program axes");
    std::array<std::string, 3> grid =
        target::lowering::projectProgramGrid(
            planIndex, [&](plan::AxisOp axis) {
              std::string role =
                  "program_" + std::to_string(axis.getProgramOrder());
              return roleDimensions.lookup(role);
            });
    output << "    grid = (" << grid[0] << ", " << grid[1] << ", "
           << grid[2] << ")\n";
    std::string launchKernel = kernelName;
    if (tuneRowOccupancy) {
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
      for (ABIView *input : inputs)
        emitCacheKey(input->argument->name + ".dtype");
      emitCacheKey("str(_DEVICE)");
      output << ")\n";
      output << "    if cache_key not in _TUNE_CACHE:\n";
      output << "        _TUNE_CACHE[cache_key] = tune_row_occupancy(\n";
      output << "            stream, grid, " << kernelName
             << ", lambda _: (";
      for (ABIView &view : views) {
        output << view.argument->name;
        if (view.view.getAccess() == "inout")
          output << ".clone()";
        output << ", ";
      }
      for (ABIScalar &scalar : scalars)
        output << scalar.name << ", ";
      for (Operation *buffer : privateWorkspaceBuffers)
        output << workspaceNames.lookup(buffer->getResult(0)) << ", ";
      for (const auto &entry : planIndex.scans) {
        if (entry.second.getResultSpace() != "private_workspace")
          continue;
        Operation *scan = kernel.nodes.lookup(entry.first);
        for (Value result : scan->getResults())
          output << workspaceNames.lookup(result) << ", ";
        for (int64_t valueID : entry.second.getMaterializedValues())
          output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
      }
      for (const std::string &dimension : kernelConstants)
        output << dimension << ", ";
      for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
        output << physicalExtent << ", ";
      output << "))\n";
      launchKernel = "_TUNE_CACHE[cache_key]";
    }
    output << "    return ct.launch(stream, grid, " << launchKernel << ", (";
    for (ABIView &view : views) {
      output << view.argument->name;
      output << ", ";
    }
    for (ABIScalar &scalar : scalars)
      output << scalar.name << ", ";
    for (Operation *buffer : privateWorkspaceBuffers)
      output << workspaceNames.lookup(buffer->getResult(0)) << ", ";
  for (const auto &entry : planIndex.scans) {
    if (entry.second.getResultSpace() != "private_workspace")
      continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      for (Value result : scan->getResults())
        output << workspaceNames.lookup(result) << ", ";
      for (int64_t valueID : entry.second.getMaterializedValues())
        output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
    }
    for (const std::string &dimension : kernelConstants)
      output << dimension << ", ";
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << physicalExtent << ", ";
    output << "))\n\n\n";
  } else {
    SmallVector<plan::AxisOp> programAxes =
        target::lowering::orderedProgramAxes(planIndex);
    SmallVector<plan::AxisOp> dynamicRaggedAxes;
    for (plan::AxisOp axis : programAxes) {
      if (!planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode()))
        continue;
      FailureOr<plan::RaggedOp> relation =
          target::lowering::uniqueRaggedRelation(
              planIndex, axis.getNode(), *axis.operation.getOperation());
      auto runtime = succeeded(relation)
                         ? raggedRuntimeByRelation.find(relation->getNode())
                         : raggedRuntimeByRelation.end();
      if (failed(relation) || runtime == raggedRuntimeByRelation.end())
        return axis.emitOpError("has no ordered ragged runtime metadata");
      ABIView *offsets = raggedRuntimes[runtime->second].offsets;
      output << "    max_member_length_" << axis.getNode() << " = int(("
             << offsets->argument->name << "[1:] - "
             << offsets->argument->name << "[:-1]).max().item())\n";
      dynamicRaggedAxes.push_back(axis);
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
    for (plan::AxisOp axis : dynamicRaggedAxes)
      emitCacheKey("max_member_length_" + std::to_string(axis.getNode()));
    for (ABIView *input : inputs)
      emitCacheKey(input->argument->name + ".dtype");
    emitCacheKey("str(_DEVICE)");
    output << ")\n";
    output << "    if cache_key not in _TUNE_CACHE:\n";
    output << "        with ct.compiler_timeout(_TUNE_TIMEOUT):\n";
    output << "            result = exhaustive_search(\n";
    output << "                _CONFIGS,\n                stream,\n";
    std::array<std::string, 3> candidateGrid =
        target::lowering::projectProgramGrid(
            planIndex, [&](plan::AxisOp axis) {
              std::string role =
                  "program_" + std::to_string(axis.getProgramOrder());
              std::string extent =
                  planIndex.components.orderedRaggedProgramAxes.contains(
                      axis.getNode())
                      ? "max_member_length_" + std::to_string(axis.getNode())
                      : roleDimensions.lookup(role);
              return axis.isScalar()
                         ? extent
                         : "ceil(" + extent + " / cfg." +
                               axis.getTile().str() + ")";
            });
    if (planIndex.program.getPersistent()) {
      std::string total = target::lowering::projectProgramVolume(
          planIndex, [&](plan::AxisOp axis) {
            std::string role =
                "program_" + std::to_string(axis.getProgramOrder());
            std::string extent = roleDimensions.lookup(role);
            return axis.isScalar()
                       ? extent
                       : "ceil(" + extent + " / cfg." +
                             axis.getTile().str() + ")";
          });
      output << "                lambda cfg: (min(torch.cuda.get_device_properties(_DEVICE).multi_processor_count // cfg.num_ctas, "
             << total << ") * cfg.occupancy, 1, 1),\n";
    } else {
      output << "                lambda cfg: (" << candidateGrid[0] << ", "
             << candidateGrid[1] << ", " << candidateGrid[2] << "),\n";
    }
    output << "                " << kernelName << ",\n";
    output << "                lambda cfg: (";
    for (ABIView &view : views) {
      output << view.argument->name;
      if (view.view.getAccess() == "inout")
        output << ".clone()";
      output << ", ";
    }
    for (ABIScalar &scalar : scalars)
      output << scalar.name << ", ";
    for (Operation *buffer : privateWorkspaceBuffers)
      output << workspaceNames.lookup(buffer->getResult(0)) << ", ";
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "private_workspace")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      for (Value result : scan->getResults())
        output << workspaceNames.lookup(result) << ", ";
      for (int64_t valueID : entry.second.getMaterializedValues())
        output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
    }
    for (const std::string &dimension : kernelConstants)
      output << dimension << ", ";
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << physicalExtent << ", ";
    for (const auto &parameter : tuningParameters)
      output << "cfg." << parameter.first << ", ";
    output << "),\n";
    output << "                lambda cfg: {'num_ctas': cfg.num_ctas, 'occupancy': cfg.occupancy},\n";
    output << "                single_run_timeout_sec=_TUNE_TIMEOUT,\n";
    output << "            )\n";
    output << "        best = result.best.config\n";
    output << "        _TUNE_CACHE[cache_key] = (best, " << kernelName
           << ".replace_hints(num_ctas=best.num_ctas, occupancy=best.occupancy))\n";
    output << "    best, tuned_kernel = _TUNE_CACHE[cache_key]\n";
    std::array<std::string, 3> selectedGrid =
        target::lowering::projectProgramGrid(
            planIndex, [&](plan::AxisOp axis) {
              std::string role =
                  "program_" + std::to_string(axis.getProgramOrder());
              std::string extent =
                  planIndex.components.orderedRaggedProgramAxes.contains(
                      axis.getNode())
                      ? "max_member_length_" + std::to_string(axis.getNode())
                      : roleDimensions.lookup(role);
              return axis.isScalar()
                         ? extent
                         : "ceil(" + extent + " / best." +
                               axis.getTile().str() + ")";
            });
    if (planIndex.program.getPersistent()) {
      std::string total = target::lowering::projectProgramVolume(
          planIndex, [&](plan::AxisOp axis) {
            std::string role =
                "program_" + std::to_string(axis.getProgramOrder());
            std::string extent = roleDimensions.lookup(role);
            return axis.isScalar()
                       ? extent
                       : "ceil(" + extent + " / best." +
                             axis.getTile().str() + ")";
          });
      output << "    grid = (min(torch.cuda.get_device_properties(_DEVICE).multi_processor_count // best.num_ctas, "
             << total << ") * best.occupancy, 1, 1)\n";
    } else {
      output << "    grid = (" << selectedGrid[0] << ", " << selectedGrid[1]
             << ", " << selectedGrid[2] << ")\n";
    }
    output << "    return ct.launch(stream, grid, tuned_kernel, (";
    for (ABIView &view : views)
      output << view.argument->name << ", ";
    for (ABIScalar &scalar : scalars)
      output << scalar.name << ", ";
    for (Operation *buffer : privateWorkspaceBuffers)
      output << workspaceNames.lookup(buffer->getResult(0)) << ", ";
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "private_workspace")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      for (Value result : scan->getResults())
        output << workspaceNames.lookup(result) << ", ";
      for (int64_t valueID : entry.second.getMaterializedValues())
        output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
    }
    for (const std::string &dimension : kernelConstants)
      output << dimension << ", ";
    for (const auto &[physicalExtent, logicalExtent] : blockExtentConstants)
      output << physicalExtent << ", ";
    for (const auto &parameter : tuningParameters)
      output << "best." << parameter.first << ", ";
    output << "))\n\n\n";
  }
  output << "def run(";
  firstParameter = true;
  for (ABIView *view : inputs) {
    if (!firstParameter)
      output << ", ";
    output << view->argument->name;
    firstParameter = false;
  }
  for (ABIScalar &scalar : scalars) {
    if (!firstParameter)
      output << ", ";
    output << scalar.name;
    firstParameter = false;
  }
  output << "):\n";
  for (const std::string &dimension : dimensionOrder)
    output << "    " << dimension << " = " << dimensionOwners.lookup(dimension)
           << "\n";
  for (ABIView *result : outputs) {
    StringRef outputDtype = torchDtype(result->tensor.getElementType());
    if (outputDtype.empty())
      return kernel.entry.emitOpError("has an unsupported cuTile output dtype");
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
  for (auto [index, view] : llvm::enumerate(views)) {
    if (index)
      output << ", ";
    output << view.argument->name;
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

FailureOr<StringRef> ProgramMaterializer::lookupValue(Operation &consumer,
                                                 unsigned operandIndex) {
  if (operandIndex >= consumer.getNumOperands()) {
    consumer.emitOpError("references a missing operand during cuTile emission");
    return failure();
  }
  auto found = valueNames.find(consumer.getOperand(operandIndex));
  if (found == valueNames.end()) {
    consumer.emitOpError() << "operand " << operandIndex
                           << " has no emitted cuTile value";
    return failure();
  }
  return StringRef(found->second);
}

FailureOr<ABIView *> ProgramMaterializer::lookupView(Value value,
                                               Operation &consumer) {
  auto found = viewPositions.find(value);
  if (found == viewPositions.end()) {
    consumer.emitOpError("references a non-ABI external view");
    return failure();
  }
  return &views[found->second];
}

FailureOr<Operation *> ProgramMaterializer::resolveDomain(Value indexedValue,
                                                    Operation &consumer) {
  FailureOr<target::ScalarIndexSource> scalarSource =
      target::traceScalarIndexSource(indexedValue, consumer);
  if (failed(scalarSource))
    return failure();
  if (scalarSource->domain)
    return scalarSource->domain;
  if (scalarSource->hasDomain()) {
    consumer.emitOpError(
        "cannot resolve multi-axis scalar ownership during cuTile emission");
    return failure();
  }
  if (scalarSource->opaque) {
    consumer.emitOpError(
        "cannot resolve an opaque index source during cuTile emission");
    return failure();
  }
  if (Operation *definition = indexedValue.getDefiningOp())
    if (::intent::target::semanticOperationName(*definition) == "intent.domain" ||
        ::intent::target::semanticOperationName(*definition) == "intent.ragged_outer" ||
        ::intent::target::semanticOperationName(*definition) == "intent.ragged_member")
      return definition;
  consumer.emitOpError("cannot resolve index ownership during cuTile emission");
  return failure();
}

FailureOr<plan::AxisOp> ProgramMaterializer::resolveAxis(Value indexedValue,
                                                  Operation &consumer) {
  FailureOr<Operation *> domain = resolveDomain(indexedValue, consumer);
  if (failed(domain))
    return failure();
  FailureOr<int64_t> node = target::getNodeID(**domain, "axis lookup");
  plan::AxisOp axis =
      succeeded(node) ? planIndex.axes.lookup(*node) : plan::AxisOp();
  if (failed(node) || !axis) {
    consumer.emitOpError("indexes a domain without a cuTile axis binding");
    return failure();
  }
  return axis;
}

FailureOr<std::string> ProgramMaterializer::dimensionName(Operation &domain) {
  return target::lowering::logicalDomainExtent(
      domain, [&](Value value,
                  Operation &consumer) -> FailureOr<ArrayRef<std::string>> {
        FailureOr<ABIView *> view = lookupView(value, consumer);
        if (failed(view))
          return failure();
        return ArrayRef<std::string>((*view)->shape);
      });
}

std::string ProgramMaterializer::addressIndex(StringRef expression) const {
  return "ct.astype((" + expression.str() + "), ct.int32)";
}

std::string ProgramMaterializer::physicalExtent(StringRef logicalExtent) const {
  auto extent = planIndex.blockExtents.find(logicalExtent);
  if (extent == planIndex.blockExtents.end())
    return logicalExtent.str();
  return "PHYSICAL_" + logicalExtent.str();
}

FailureOr<std::string> ProgramMaterializer::physicalAxisTile(plan::AxisOp axis) {
  auto scanTile = scanAxisTiles.find(axis.getNode());
  if (scanTile != scanAxisTiles.end())
    return scanTile->second;
  if (axis.getReuseWorker() || !axis.getTileRole().starts_with("row_vector"))
    return axis.getTile().str();
  const target::lowering::RangeBinding *range = axis.roleRange();
  if (!range || !planIndex.blockExtents.count(range->getExtent()))
    return axis.emitOpError("cannot resolve its row-vector physical extent");
  return physicalExtent(range->getExtent());
}

FailureOr<std::string>
ProgramMaterializer::transferPhysicalExtentFill(Operation &operation) {
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
  return target::lowering::transferPhysicalExtentFill(
      planIndex, *relation, (*view)->shape, operation, neutralized);
}

FailureOr<std::string> ProgramMaterializer::indexTuple(Operation &operation,
                                                 bool elementwiseAccess) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  FailureOr<int64_t> transferNode =
      target::getNodeID(operation, "cuTile translated access projection");
  SmallVector<target::lowering::RangeBinding> accessRanges =
      succeeded(transferNode)
          ? target::lowering::accessRangesForTransfer(planIndex, *transferNode)
          : SmallVector<target::lowering::RangeBinding>();
  RankedTensorType projectedTensor;
  if (operation.getNumResults() == 1)
    projectedTensor = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  if (!projectedTensor) {
    auto valueIndex =
        operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
    if (valueIndex && valueIndex.getInt() >= 0 &&
        static_cast<unsigned>(valueIndex.getInt()) < operation.getNumOperands())
      projectedTensor = dyn_cast<RankedTensorType>(
          operation.getOperand(valueIndex.getInt()).getType());
  }
  if (elementwiseAccess && projectedTensor) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view))
      return operation.emitOpError(
          "indirect cuTile gather has no external-view ABI binding");
    if (static_cast<size_t>(llvm::count_if(
            *relation, [](const target::IndexTerm &term) {
              return term.kind != "new_axis";
            })) != (*view)->shape.size())
      return operation.emitOpError("indirect cuTile gather has ")
             << relation->size() << " index terms for a "
             << (*view)->shape.size() << "-rank external view";
    target::TensorIndexGroup tensorIndices =
        target::tensorIndexGroup(operation, *relation);
    unsigned resultAxis = 0;
    std::optional<unsigned> tensorGroupAxis;
    auto broadcast = [&](StringRef value, unsigned axis) {
      if (projectedTensor.getRank() == 1)
        return value.str();
      std::string expression = value.str() + "[";
      for (unsigned position = 0; position <
                                  static_cast<unsigned>(projectedTensor.getRank());
           ++position) {
        if (position)
          expression += ", ";
        expression += position == axis ? ":" : "None";
      }
      return expression + "]";
    };
    SmallVector<std::string> indices;
    unsigned sourceAxis = 0;
    for (const target::IndexTerm &term : *relation) {
      if (term.kind == "new_axis") {
        ++resultAxis;
        continue;
      }
      unsigned axisNumber = sourceAxis++;
      if (term.kind == "full_slice") {
        if (resultAxis >= static_cast<unsigned>(projectedTensor.getRank()))
          return operation.emitOpError(
              "indirect cuTile gather has too many vector axes");
        indices.push_back(broadcast(addressIndex(
                                        "ct.arange(" +
                                        physicalExtent(
                                            (*view)->shape[axisNumber]) +
                                        ", dtype=ct.int32)"),
                                    resultAxis++));
        continue;
      }
      if (term.kind == "static_index") {
        if (term.staticValues.size() != 1 || !term.staticValues.front())
          return operation.emitOpError(
              "indirect cuTile gather has an invalid static index");
        indices.push_back(std::to_string(*term.staticValues.front()));
        continue;
      }
      if ((term.kind != "region_index" && term.kind != "value_index") ||
          term.operands.size() != 1 || !term.operands.front())
        return operation.emitOpError(
            "indirect cuTile gather has no mechanical index relation");
      Value indexed = operation.getOperand(*term.operands.front());
      if (auto tensor = dyn_cast<RankedTensorType>(indexed.getType())) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (term.kind != "value_index" || failed(exact))
          return operation.emitOpError(
              "cuTile indirect tensor index has no canonical value");
        if (tensorIndices.requiresBroadcastProjection()) {
          if (!tensorGroupAxis) {
            if (resultAxis + tensorIndices.rank >
                static_cast<unsigned>(projectedTensor.getRank()))
              return operation.emitOpError(
                  "cuTile broadcasted tensor index exceeds the result rank");
            tensorGroupAxis = resultAxis;
            resultAxis += tensorIndices.rank;
          }
          if (tensor.getRank() > static_cast<int64_t>(tensorIndices.rank))
            return operation.emitOpError(
                "cuTile broadcasted tensor index exceeds the result rank");
          std::string projected = projectTensorIndex(
              *exact, tensor.getRank(), tensorIndices.rank, *tensorGroupAxis,
              projectedTensor.getRank());
          indices.push_back(addressIndex(projected));
        } else {
          if (tensor.getRank() != 1 ||
              resultAxis >= static_cast<unsigned>(projectedTensor.getRank()))
            return operation.emitOpError(
                "cuTile indirect tensor indices require one logical axis");
          indices.push_back(broadcast(addressIndex(*exact), resultAxis++));
        }
        continue;
      }
      if (isa<IntegerType, IndexType, intent::LogicalIndexType>(
              indexed.getType())) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        indices.push_back(addressIndex(*exact));
        continue;
      }
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis) ||
          resultAxis >= static_cast<unsigned>(projectedTensor.getRank()))
        return failure();
      std::string base = axisIndices.lookup(axis->getNode());
      if (base.empty())
        return operation.emitOpError(
            "indirect cuTile gather has no active vector axis");
      bool directVector =
          target::lowering::isRaggedBoundAxis(planIndex.components,
                                              axis->getNode()) ||
          (axis->hasRole("lane") &&
           kernel.nodes.lookup(axis->getNode()) == vectorDomain);
      FailureOr<std::string> tile = physicalAxisTile(*axis);
      if (failed(tile))
        return failure();
      std::string exact =
          directVector
              ? addressIndex(base)
              : activeScanReplay >= 0
                    ? addressIndex(base)
                    : addressIndex(base) + " * " + *tile + " + " +
                    addressIndex("ct.arange(" + *tile +
                                 ", dtype=ct.int32)");
      indices.push_back(broadcast(exact, resultAxis++));
    }
    if (resultAxis != static_cast<unsigned>(projectedTensor.getRank()))
      return operation.emitOpError(
          "indirect cuTile gather does not cover every result axis");
    std::string tuple = "(";
    for (auto [index, value] : llvm::enumerate(indices)) {
      if (index)
        tuple += ", ";
      tuple += value;
    }
    if (indices.size() == 1)
      tuple += ",";
    return tuple + ")";
  }
  bool raggedMatrix = false;
  if (relation->size() == 2 &&
      ((*relation)[0].kind == "region_index" ||
       (*relation)[0].kind == "value_index") &&
      (*relation)[0].operands.size() == 1 &&
      (*relation)[0].operands.front()) {
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*(*relation)[0].operands.front()),
                    operation);
    raggedMatrix = succeeded(axis) && target::lowering::isRaggedBoundAxis(
                                          planIndex.components, axis->getNode());
  }
  if (raggedMatrix) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view) || (*view)->tensor.getRank() != 2 ||
        relation->size() != 2 || (*relation)[1].kind != "full_slice" ||
        ((*relation)[0].kind != "region_index" &&
         (*relation)[0].kind != "value_index") ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front())
      return operation.emitOpError(
          "ragged cuTile matrix access requires member rows and one full feature axis");
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*(*relation)[0].operands.front()),
                    operation);
    if (failed(axis))
      return failure();
    std::string rows = axisIndices.lookup(axis->getNode());
    if (rows.empty())
      return operation.emitOpError(
          "ragged cuTile matrix access has no member-axis role");
    return "(" + addressIndex(rows + "[:, None]") + ", " +
           addressIndex("ct.arange(" + physicalExtent((*view)->shape[1]) +
                        ", dtype=ct.int32)[None, :]") +
           ")";
  }
  SmallVector<std::string> indices;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice") {
      indices.push_back("0");
      continue;
    }
    if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError(
            "static cuTile index has no canonical value");
      indices.push_back(std::to_string(*term.staticValues.front()));
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "cuTile access has no mechanical index relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto translated = llvm::find_if(accessRanges, [&](const auto &range) {
        return range.getSourceAxis() == static_cast<int64_t>(indices.size()) &&
               !range.isCompact() && range.getOffset() > 0;
      });
      if (translated != accessRanges.end()) {
        auto axis = planIndex.axes.find(translated->getAxisNode());
        if (axis == planIndex.axes.end())
          return translated->emitOpError(
              "references an unbound cuTile translated-access axis");
        std::string base = axisIndices.lookup(axis->second.getNode());
        if (base.empty() && axis->second.hasRole("lane"))
          base = "0";
        if (base.empty())
          return translated->emitOpError(
              "has no active cuTile translated-access base");
        StringRef role = translated->getTileRole();
        int64_t tile = 0;
        if (!role.starts_with("fixed_") ||
            role.drop_front(6).getAsInteger(10, tile) || tile <= 0 ||
            translated->getOffset() % tile != 0)
          return translated->emitOpError(
              "has no tile-aligned cuTile translated-access projection");
        indices.push_back(addressIndex(
            base + " + " + std::to_string(translated->getOffset() / tile)));
        continue;
      }
      if (!elementwiseAccess)
        return operation.emitOpError(
            "cuTile tensor-index transfer requires element-coordinate projection");
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      indices.push_back(addressIndex(*exact));
      continue;
    }
    if ((term.kind == "value_index" &&
         !isa<RankedTensorType>(indexed.getType())) ||
        (term.kind == "region_index" &&
         (target::lowering::isSequentialIterator(indexed) ||
          isa<BlockArgument>(indexed)))) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      indices.push_back(addressIndex(*exact));
    } else {
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      std::string index = axisIndices.lookup(axis->getNode());
      if (index.empty()) {
        operation.emitOpError()
            << "has no active cuTile index for logical axis " << axis->getNode()
            << " (parallel=" << axis->hasRole("parallel")
            << ", ordered=" << axis->hasRole("ordered")
            << ", reduction=" << axis->hasRole("reduction")
            << ", lane=" << axis->hasRole("lane") << ")";
        return failure();
      }
      indices.push_back(addressIndex(index));
    }
  }
  std::string tuple = "(";
  for (auto [index, value] : llvm::enumerate(indices)) {
    if (index)
      tuple += ", ";
    tuple += value;
  }
  if (indices.size() == 1)
    tuple += ",";
  tuple += ")";
  return tuple;
}

FailureOr<std::string> ProgramMaterializer::tileShape(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  FailureOr<int64_t> transferNode =
      target::getNodeID(operation, "cuTile translated tile-shape projection");
  SmallVector<target::lowering::RangeBinding> accessRanges =
      succeeded(transferNode)
          ? target::lowering::accessRangesForTransfer(planIndex, *transferNode)
          : SmallVector<target::lowering::RangeBinding>();
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view))
    return failure();
  if (static_cast<size_t>(llvm::count_if(
          *relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != (*view)->shape.size())
    return operation.emitOpError(
        "cuTile tile shape does not cover every source view axis");
  SmallVector<std::string> extents;
  unsigned sourceAxis = 0;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    unsigned axisNumber = sourceAxis++;
    if (term.kind == "full_slice") {
      extents.push_back(physicalExtent((*view)->shape[axisNumber]));
      continue;
    }
    if (term.kind == "static_index") {
      extents.push_back("1");
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "cuTile tile has no mechanical shape relation");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto translated = llvm::find_if(accessRanges, [&](const auto &range) {
        return range.getSourceAxis() == static_cast<int64_t>(axisNumber) &&
               !range.isCompact() && range.getOffset() > 0;
      });
      if (translated != accessRanges.end()) {
        auto axis = planIndex.axes.find(translated->getAxisNode());
        if (axis == planIndex.axes.end())
          return translated->emitOpError(
              "references an unbound cuTile translated-access tile");
        FailureOr<std::string> tile = physicalAxisTile(axis->second);
        if (failed(tile))
          return failure();
        extents.push_back(*tile);
        continue;
      }
      auto result = dyn_cast<OpResult>(indexed);
      auto tensor = dyn_cast<RankedTensorType>(indexed.getType());
      auto shapes = result ? result.getOwner()->getAttrOfType<ArrayAttr>(
                                 "intent.result_shapes")
                           : ArrayAttr();
      auto labels = shapes && result.getResultNumber() < shapes.size()
                        ? dyn_cast<ArrayAttr>(shapes[result.getResultNumber()])
                        : ArrayAttr();
      if (!tensor || !labels ||
          labels.size() != static_cast<size_t>(tensor.getRank()))
        return failure();
      std::optional<std::string> varying;
      for (Attribute attribute : labels) {
        auto label = dyn_cast<StringAttr>(attribute);
        if (!label)
          return failure();
        std::string extent = label.getValue() == "1"
                                 ? "1"
                                 : regionTiles.lookup(label.getValue());
        if (extent.empty())
          extent = physicalExtent(label.getValue());
        if (extent == "1")
          continue;
        if (varying)
          return operation.emitOpError(
              "regular cuTile index tile varies along more than one axis");
        varying = extent;
      }
      extents.push_back(varying.value_or("1"));
      continue;
    }
    if ((term.kind == "value_index" &&
         !isa<RankedTensorType>(indexed.getType())) ||
        (term.kind == "region_index" &&
         target::lowering::isSequentialIterator(indexed))) {
      extents.push_back("1");
    } else {
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      FailureOr<std::string> extent = physicalAxisTile(*axis);
      if (failed(extent))
        return failure();
      extents.push_back(*extent);
    }
  }
  std::string tuple = "(";
  for (auto [index, extent] : llvm::enumerate(extents)) {
    if (index)
      tuple += ", ";
    tuple += extent;
  }
  if (extents.size() == 1)
    tuple += ",";
  tuple += ")";
  return tuple;
}

FailureOr<std::string> ProgramMaterializer::emitValidityExpression(
    ArrayRef<int64_t> tensorAxes, ArrayRef<int64_t> domainNodes, Value value,
    Operation &consumer) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || tensorAxes.size() != domainNodes.size())
    return consumer.emitOpError(
        "has no ranked cuTile value-validity binding");
  SmallVector<std::string> predicates;
  for (auto [tensorAxis, domainNode] : llvm::zip(tensorAxes, domainNodes)) {
    auto domain = kernel.nodes.find(domainNode);
    auto physical = planIndex.axes.find(domainNode);
    if (tensorAxis < 0 || tensorAxis >= tensor.getRank() ||
        domain == kernel.nodes.end() || physical == planIndex.axes.end())
      return consumer.emitOpError(
          "references an unresolved cuTile validity axis");
    plan::AxisOp axis = physical->second;
    if (axis.isScalar())
      continue;
    bool stagedMember = llvm::any_of(activeStages, [&](unsigned ordinal) {
      auto stage = llvm::find_if(planIndex.stages, [&](plan::StageOp candidate) {
        return candidate.getOrdinal() == ordinal;
      });
      if (stage == planIndex.stages.end())
        return false;
      auto axes = planIndex.stageAxes.find(stage->getNode());
      plan::StageAxisOp member =
          axes == planIndex.stageAxes.end()
              ? plan::StageAxisOp()
              : axes->second.lookup("member");
      return member && member.getAxisNodeAttr() &&
             member.getAxisNodeAttr().getInt() == domainNode;
    });
    if (stagedMember) {
      std::string predicate = "member_mask";
      if (tensor.getRank() > 1) {
        predicate += "[";
        for (int64_t axisNumber = 0; axisNumber < tensor.getRank();
             ++axisNumber) {
          if (axisNumber)
            predicate += ", ";
          predicate += axisNumber == tensorAxis ? ":" : "None";
        }
        predicate += "]";
      }
      predicates.push_back(std::move(predicate));
      continue;
    }
    std::string base = axisIndices.lookup(axis.getNode());
    if (base.empty())
      return consumer.emitOpError(
          "has no active cuTile index for its planned validity axis");
    bool direct = target::lowering::isRaggedBoundAxis(
                      planIndex.components, axis.getNode()) ||
                  (!planIndex.components.reusedAxes.empty() &&
                   kernel.nodes.lookup(axis.getNode()) == vectorDomain);
    FailureOr<std::string> tile = physicalAxisTile(axis);
    if (failed(tile))
      return failure();
    std::string index =
        direct ? addressIndex(base)
               : activeScanReplay >= 0
                     ? addressIndex(base)
                     : addressIndex(base) + " * " + *tile + " + " +
                     addressIndex("ct.arange(" + *tile +
                                  ", dtype=ct.int32)");
    if (tensor.getRank() > 1) {
      std::string broadcast = index + "[";
      for (int64_t axisNumber = 0; axisNumber < tensor.getRank(); ++axisNumber) {
        if (axisNumber)
          broadcast += ", ";
        broadcast += axisNumber == tensorAxis ? ":" : "None";
      }
      index = broadcast + "]";
    }
    FailureOr<std::string> extent = dimensionName(*domain->second);
    if (target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> ordered = target::lowering::representativeOrderedAxis(
          planIndex, axis.getNode(), consumer);
      if (failed(ordered))
        return failure();
      extent = "sequence_end_" + std::to_string(*ordered);
    }
    if (failed(extent))
      return failure();
    if (!target::lowering::isRaggedBoundAxis(planIndex.components,
                                             axis.getNode())) {
      auto owner = dimensionOwners.find(*extent);
      int64_t staticExtent;
      if (owner != dimensionOwners.end())
        *extent = owner->second;
      else if (StringRef(*extent).getAsInteger(10, staticExtent))
        return consumer.emitOpError(
            "has no cuTile runtime binding for a symbolic validity extent");
    }
    predicates.push_back("(" + index + " < " + *extent + ")");
  }
  if (predicates.empty())
    return std::string("True");
  std::string result = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    result += " & " + predicate.str();
  return result;
}

FailureOr<std::string> ProgramMaterializer::padExpression(
    Value value, StringRef expression, Operation &consumer) {
  FailureOr<int64_t> valueID =
      target::getValueID(value, kernel, consumer, "cuTile padding lookup");
  if (failed(valueID))
    return failure();
  plan::PaddingOp padding = planIndex.paddings.lookup(*valueID);
  if (!padding)
    return expression.str();
  FailureOr<std::string> predicate = emitValidityExpression(
      padding.getTensorAxes(), padding.getDomainNodes(), value, consumer);
  if (failed(predicate))
    return failure();
  Type valueType = value.getType();
  if (auto tensor = dyn_cast<RankedTensorType>(valueType))
    valueType = tensor.getElementType();
  StringRef planned = padding.getFill();
  std::string fill = planned == "negative_infinity"
                         ? "-math.inf"
                     : planned == "positive_infinity" ? "math.inf"
                     : planned == "nan" ? "math.nan"
                     : planned == "true" ? "True"
                     : planned == "false" ? "False"
                     : planned.starts_with("literal_integer:")
                         ? planned.drop_front(16).str()
                     : planned.starts_with("literal_float:")
                         ? planned.drop_front(14).str()
                     : isa<IntegerType, IndexType>(valueType) ? "0"
                                                              : "0.0";
  return "ct.where(" + *predicate + ", " + expression.str() + ", " +
         fill + ")";
}

FailureOr<std::string>
ProgramMaterializer::emitTensorShape(Operation &operation, unsigned resultIndex,
                               bool transposeLastTwo) {
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
          "tensor shape region has no cuTile tile binding");
    else
      extents.push_back(physicalExtent(label.getValue()));
  }
  if (transposeLastTwo) {
    if (extents.size() < 2)
      return operation.emitOpError(
          "cannot transpose a tensor result with fewer than two dimensions");
    std::swap(extents[extents.size() - 2], extents.back());
  }
  std::string result = "(";
  for (auto [index, extent] : llvm::enumerate(extents)) {
    if (index)
      result += ", ";
    result += extent;
  }
  if (extents.size() == 1)
    result += ",";
  return result + ")";
}

FailureOr<unsigned> ProgramMaterializer::emittedTensorRank(Operation &operation,
                                                     bool store) {
  Type type;
  if (store) {
    auto valueIndex =
        operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
    if (!valueIndex || valueIndex.getInt() < 0 ||
        static_cast<unsigned>(valueIndex.getInt()) >= operation.getNumOperands())
      return operation.emitOpError("store has no canonical value operand");
    type = operation.getOperand(valueIndex.getInt()).getType();
  } else if (operation.getNumResults() == 1) {
    type = operation.getResult(0).getType();
  }
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return static_cast<unsigned>(tensor.getRank());
  if (type.isIntOrIndexOrFloat())
    return 0;
  return operation.emitOpError("boundary value is neither a tensor nor a scalar");
}

std::string ProgramMaterializer::dtypeName(Type type, Operation &consumer) {
  if (type.isInteger(1))
    return "ct.bool_";
  if (type.isF16())
    return "ct.float16";
  if (type.isF32())
    return "ct.float32";
  if (type.isBF16())
    return "ct.bfloat16";
  if (isa<Float8E4M3FNType>(type))
    return "ct.float8_e4m3fn";
  if (isa<Float8E5M2Type>(type))
    return "ct.float8_e5m2";
  if (isa<Float8E8M0FNUType>(type))
    return "ct.float8_e8m0fnu";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 8)
    return integer.isUnsigned() ? "ct.uint8" : "ct.int8";
  if (type.isInteger(32))
    return "ct.int32";
  if (type.isInteger(64))
    return "ct.int64";
  if (isa<IndexType>(type))
    return "ct.int64";
  consumer.emitOpError("uses an unsupported cuTile dtype");
  return {};
}

std::string ProgramMaterializer::uniqueName(StringRef candidate, int64_t node) {
  std::string result = candidate.str();
  if (!usedNames.insert(result).second) {
    result += "_" + std::to_string(node);
    usedNames.insert(result);
  }
  return result;
}

std::string ProgramMaterializer::makeResultName(Operation &operation, unsigned index) {
  auto names = operation.getAttrOfType<ArrayAttr>("intent.result_names");
  auto name = names && index < names.size() ? dyn_cast<StringAttr>(names[index])
                                           : StringAttr();
  FailureOr<int64_t> node = target::getNodeID(operation, "result naming");
  return uniqueName(name ? name.getValue() : "value", *node);
}

std::string ProgramMaterializer::makeRegionArgumentName(Operation &operation,
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

void ProgramMaterializer::line(StringRef text) {
  if (!planIndex.stages.empty()) {
    for (unsigned stage : activeStages)
      stageLine(stage, text, indentation);
    return;
  }
  output.indent(indentation * 4) << text << "\n";
}

LogicalResult materializeProgramSource(
    target::KernelModel kernel, intent::plan::ProgramOp physicalProgram,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<PhysicalProgramIndex> indexed = indexPhysicalProgram(physicalProgram, kernel);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return ProgramMaterializer(std::move(kernel), physicalProgram, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .materialize();
}

} // namespace intent::cutile::lowering

#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Lowering/Combiner.h"
#include "Intent/Target/TileLang/Lowering/Passes.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace mlir;

namespace intent::tilelang::lowering {

static bool isPlainRaggedMemberAxis(const PhysicalProgramIndex &index,
                                    int64_t axis) {
  auto relations = index.components.raggedByAxis.find(axis);
  if (relations == index.components.raggedByAxis.end())
    return false;
  return llvm::any_of(relations->second, [&](const plan::RaggedOp &relation) {
    return llvm::is_contained(relation.getMemberNodes(), axis);
  });
}

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
      index.regionBindings[value.getValue()] = value;
    } else if (auto value =
                   dyn_cast<intent::plan::PartitionBindingOp>(operation)) {
      index.partitionBindings.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::LaunchOp>(operation)) {
      index.program.operation = value;
      auto equal = value->getAttrOfType<BoolAttr>(equalProgramTilesAttr);
      if (!equal)
        return value.emitOpError(
            "has no realized TileLang program-tile constraint");
      index.requiresSymmetricProgramTiles = equal.getValue();
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
            "has no realized TileLang contraction provider form");
      binding.lowering = lowering.getValue().str();
      binding.lhsSpace = syntax::bufferSpace(value.getLhsSpace()).str();
      binding.rhsSpace = syntax::bufferSpace(value.getRhsSpace()).str();
      binding.accumulatorSpace =
          syntax::bufferSpace(value.getAccumulatorSpace()).str();
      if (binding.lhsSpace.empty() || binding.rhsSpace.empty() ||
          binding.accumulatorSpace.empty()) {
        value.emitOpError("has no TileLang contraction residency spelling");
        return failure();
      }
      binding.orientation = orientation.getValue().str();
      binding.batched = batched.getValue();
      if (layout)
        binding.scaledLayout = layout.getValue().str();
      index.contracts[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::SparseContractOp>(operation)) {
      if (value.getFormat() != "two_of_four")
        return value.emitOpError("has no TileLang sparse contraction spelling");
      index.sparseContracts[value.getNode()] = value;
    } else if (auto value = dyn_cast<intent::plan::TransferOp>(operation)) {
      transfers.push_back(value);
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
  FailureOr<func::FuncOp> physicalEntry =
      intent::plan::getPhysicalEntry(physicalProgram);
  if (failed(physicalEntry))
    return failure();
  WalkResult loopForms = physicalEntry->walk([&](intent::plan::ExecForOp loop) {
    auto node = loop->getAttrOfType<IntegerAttr>("intent.node");
    auto spaces = loop->getAttrOfType<ArrayAttr>(loopCarrierSpacesAttr);
    if (!node || !spaces || spaces.size() != loop.getNumResults()) {
      loop.emitOpError("has no indexed TileLang loop-carrier form");
      return WalkResult::interrupt();
    }
    SmallVector<std::string> selected;
    selected.reserve(spaces.size());
    for (Attribute attribute : spaces) {
      auto space = dyn_cast<StringAttr>(attribute);
      if (!space) {
        loop.emitOpError("has a non-string TileLang loop-carrier form");
        return WalkResult::interrupt();
      }
      selected.push_back(space.getValue().str());
    }
    if (!index.loopCarrierSpaces.try_emplace(node.getInt(), std::move(selected))
             .second) {
      loop.emitOpError("duplicates one TileLang loop-carrier form");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (loopForms.wasInterrupted())
    return failure();
  for (plan::RaggedOp &ragged : index.ragged) {
    auto route = ragged.operation
                     ? ragged.operation->getAttrOfType<StringAttr>(raggedRouteAttr)
                     : StringAttr();
    if (!route)
      return physicalProgram.emitOpError(
          "has no realized TileLang ragged-route form");
    ragged.route = route.getValue().str();
  }
  target::lowering::indexAxisRoles(index);
  for (auto &entry : index.streams) {
    plan::StreamOp &binding = entry.second;
    auto tile = binding.physical->getAttrOfType<StringAttr>(streamTileAttr);
    if (!tile)
      return binding.emitOpError("has no realized TileLang stream-tile spelling");
    binding.tile = tile.getValue().str();
  }
  index.components = target::lowering::indexPhysicalComponents(index);
  for (intent::plan::ReductionOp value : reductions) {
    auto lowering = value->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = value->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    if (!lowering || !axis)
      return value.emitOpError("has no realized TileLang reduction spelling");
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
    binding.resultSpace = syntax::bufferSpace(value.getResultSpace()).str();
    binding.axis = axis.getInt();
    if (binding.resultSpace.empty())
      return value.emitOpError("has no TileLang reduction residency spelling");
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    auto lowering = value->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || !index.axes.count(value.getAxisNode()))
      return value.emitOpError("has no realized TileLang scan spelling");
    plan::ScanOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
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
    auto lowering = value->getAttrOfType<StringAttr>(pointwiseLoweringAttr);
    auto storage = value->getAttrOfType<StringAttr>(pointwiseStorageAttr);
    plan::PointwiseOp binding;
    binding.operation = value;
    binding.resultSpace = storage && storage.getValue() == "shared"
                              ? "shared"
                              : syntax::bufferSpace(value.getResultSpace()).str();
    if (!lowering || !storage || binding.resultSpace.empty())
      return value.emitOpError("has no realized TileLang pointwise spelling");
    binding.lowering = lowering.getValue().str();
    index.pointwise[value.getNode()] = binding;
  }
  for (intent::plan::TransferOp value : transfers) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    bool load = operation &&
                ::intent::target::semanticOperationName(*operation) == "intent.view_load";
    bool store = operation &&
                 (::intent::target::semanticOperationName(*operation) == "intent.view_store" ||
                  ::intent::target::semanticOperationName(*operation) == "intent.scatter_unique" ||
                  ::intent::target::semanticOperationName(*operation) == "intent.atomic_add" ||
                  ::intent::target::semanticOperationName(*operation) == "intent.atomic_cas");
    if (!load && !store)
      return value.emitOpError("does not bind a canonical transfer");
    auto access = value->getAttrOfType<StringAttr>(accessAttr);
    auto transfer = value->getAttrOfType<StringAttr>(transferAttr);
    auto bounds = value->getAttrOfType<BoolAttr>(boundsAttr);
    auto deferred = value->getAttrOfType<BoolAttr>(deferredAttr);
    auto nativeReshape =
        value->getAttrOfType<IntegerAttr>(nativeContractReshapeNodeAttr);
    if (!access || !transfer || !bounds || !deferred)
      return value.emitOpError("has no realized TileLang transfer form");
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = access.getValue().str();
    binding.transfer = transfer.getValue().str();
    binding.resultSpace = syntax::bufferSpace(value.getResultSpace()).str();
    binding.defer = deferred.getValue();
    binding.explicitBounds = bounds.getValue();
    if (binding.resultSpace.empty()) {
      value.emitOpError("has no TileLang transfer residency spelling");
      return failure();
    }
    index.boundaries[value.getNode()] = binding;
    if (nativeReshape)
      index.nativeContractOperandReshapes[value.getNode()] =
          nativeReshape.getInt();
  }
  if (!index.target || !index.program) {
    physicalProgram.emitOpError("lacks device or launch decisions for TileLang");
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
  if (failed(indexABI()))
    return failure();
  if (!planIndex.ragged.empty() && failed(prepareRaggedMetadata()))
    return failure();
  if (failed(resolvePhysicalBindings()))
    return failure();
  auto rowForm =
      planIndex.program.operation->getAttrOfType<StringAttr>(rowLaunchAttr);
  if (!rowForm)
    return planIndex.program.emitOpError(
        "has no realized TileLang row-launch form");
  tuneRowLaunch = rowForm.getValue() == "delegated";
  if (failed(target::lowering::indexScanProducerOperations(
          kernel, planIndex, scanProducerOwners)))
    return failure();
  if (failed(target::lowering::indexDeferredContractReplays(
          kernel, planIndex, deferredContractReplays,
          deferredContractProducerOwners)))
    return failure();
  for (const auto &entry : planIndex.contracts) {
    auto form = entry.second.operation->getAttrOfType<StringAttr>(
        contractMmaFormAttr);
    if (!form || form.getValue() != "packed_int2_i8_mma")
      continue;
    auto producers = entry.second.operation->getAttrOfType<DenseI64ArrayAttr>(
        packedDecodeProducerNodesAttr);
    if (!producers)
      return entry.second.emitOpError(
          "has no packed INT2 producer ownership list");
    needsPackedInt2Decode = true;
    for (int64_t node : producers.asArrayRef()) {
      Operation *producer = kernel.nodes.lookup(node);
      if (!producer)
        return entry.second.emitOpError(
            "references an unknown packed INT2 producer node");
      packedDecodeProducers.insert(producer);
    }
  }
  for (const auto &entry : planIndex.scans)
    if (failed(target::lowering::verifyScanMaterializedValues(kernel,
                                                              entry.second)))
      return failure();
  if (failed(preparePrivateWorkspaces()))
    return failure();
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
    if (binding.getResultSpace() != "global")
      continue;
    Operation *scan = kernel.nodes.lookup(entry.first);
    std::string extent = axisDimensions.lookup(binding.getAxisNode());
    if (!scan || ::intent::target::semanticOperationName(*scan) != "intent.scan" ||
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
          requiresPreallocatedOutputs |= view.getAccess() == "out";
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
  if (failed(target::lowering::indexPartitionExtents(
          planIndex, kernel, axisDimensions, dimensionOwners, dimensionOrder,
          syntax::tile)))
    return failure();
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
    if (!entry.second.getReuseWorker())
      for (target::lowering::RangeBinding &range : entry.second.ranges) {
        if (!range.getTileRole().starts_with("row_vector"))
          continue;
        if (!planIndex.blockExtents.count(range.getExtent()))
          return entry.second.emitOpError(
              "cannot resolve its row-vector extent");
        range.tile = physicalExtent(range.getExtent());
      }
    const target::lowering::RangeBinding *canonical =
        target::lowering::canonicalDomainRange(entry.second);
    if (!canonical)
      return entry.second.emitOpError("has no canonical TileLang domain range");
    std::string tile = canonical->getTile().str();
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
    plan::RegionBindingOp binding = entry.second;
    plan::AxisOp axis = planIndex.axes.lookup(binding.getAxisNode());
    const target::lowering::RangeBinding *range =
        axis ? axis.getRange(binding.getPurpose(), binding.getLevel()) : nullptr;
    if (!value || !axis || !range)
      return binding.emitOpError(
          "does not bind a canonical region value and selected range");
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
      return physicalProgram.emitOpError(
          "fixed TileLang rows cannot consume a search space");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (!planIndex.axesByRole.count("program_0") ||
        lane == planIndex.axesByRole.end())
      return physicalProgram.emitOpError(
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
  if (target::lowering::requiresDelegatedTuning(planIndex) &&
      (!searchSpace || !searchIndex.autotune))
    return physicalProgram.emitOpError(
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
    if (failed(offsets) || (*offsets)->tensor.getRank() != 1)
      return runtime.relation->emitOpError(
          "requires a canonical rank-one offsets view");
    runtime.offsets = *offsets;
    if (ragged.getRoute() == "indexed") {
      if (runtime.relation->getNumOperands() != 4)
        return runtime.relation->emitOpError(
            "does not match its selected indexed TileLang ragged route");
      Operation *indicesLoad = runtime.relation->getOperand(3).getDefiningOp();
      FailureOr<ABIView *> indices =
          indicesLoad &&
                  ::intent::target::semanticOperationName(*indicesLoad) == "intent.view_load" &&
                  indicesLoad->getNumOperands() == 1
              ? lookupView(indicesLoad->getOperand(0), *runtime.relation)
              : FailureOr<ABIView *>(failure());
      if (failed(indices) || (*indices)->tensor.getRank() != 1)
        return runtime.relation->emitOpError(
            "requires a canonical rank-one member-index view");
      runtime.indices = *indices;
    } else if (ragged.getRoute() != "compact" ||
               runtime.relation->getNumOperands() != 3)
      return runtime.relation->emitOpError(
          "does not match its selected compact TileLang ragged route");
    unsigned position = raggedRuntimes.size();
    raggedRuntimeByRelation[ragged.getNode()] = position;
    raggedRuntimesByAxis[ragged.getOuterNode()].push_back(position);
    for (int64_t member : ragged.getMemberNodes())
      raggedRuntimesByAxis[member].push_back(position);
    raggedRuntimes.push_back(std::move(runtime));
  }
  return success();
}

bool ProgramMaterializer::selectOperation(Operation &operation) {
  if (packedDecodeProducers.contains(&operation))
    return false;
  if (target::lowering::isAbsorbedRaggedDescriptorLoad(operation))
    return false;
  auto contractProducer = deferredContractProducerOwners.find(&operation);
  if (contractProducer != deferredContractProducerOwners.end())
    return activeDeferredContract &&
           llvm::is_contained(contractProducer->second,
                              activeDeferredContract);
  auto scanProducer = scanProducerOwners.find(&operation);
  if (scanProducer != scanProducerOwners.end())
    return activeScanReplay == scanProducer->second;
  return true;
}

void ProgramMaterializer::emitImports() {
  bool tuneRow = !planIndex.components.reusedAxes.empty() || tuneRowLaunch;
  auto tuneGemm = searchIndex.autotune
                      ? searchIndex.autotune.operation
                            ->getAttrOfType<BoolAttr>(gemmWarpPolicyAttr)
                      : BoolAttr();
  bool tuneGemmWarpPolicy = tuneGemm && tuneGemm.getValue();
  output << "import torch\n";
  output << "import tilelang\n";
  output << "import tilelang.language as T\n";
  if (!planIndex.partitionBindings.empty())
    output << "\n\ndef _intent_partition_extent(logical_extent, count):\n"
              "    if count < 1:\n"
              "        raise ValueError('partition count must be positive')\n"
              "    return (logical_extent + count - 1) // count\n";
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
      for (const target::lowering::RangeBinding &range : entry.second.ranges) {
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

LogicalResult ProgramMaterializer::emitHelpers() {
  FailureOr<SmallVector<func::FuncOp>> combiners =
      target::lowering::collectCombiners(kernel.entry);
  if (failed(combiners))
    return failure();
  if (!combiners->empty())
    return combiners->front().emitOpError(
        "TileLang 0.1.13 CUDA codegen cannot lower the tirx.Reduce produced by "
        "comm_reducer and exposes no generic scan equivalent; generic "
        "combiners are unsupported");
  if (needsPackedInt2Decode)
    output << R"INTENT(_INTENT_U2X16_TO_I8_SOURCE = r"""
template <typename T1, typename T2>
__device__ void intent_u2x16_to_i8(T1 *packed, T2 *decoded) {
  unsigned int *out = reinterpret_cast<unsigned int *>(decoded);
  const unsigned int word = *reinterpret_cast<unsigned int *>(packed);
  constexpr unsigned int lut = (0xf0 & 0xcc) | 0xaa;
  constexpr unsigned int mask = 0x03030303;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(out[lane])
                 : "r"(word >> (2 * lane)), "n"(mask), "n"(0), "n"(lut));
  }
}
"""

)INTENT";
  return success();
}

LogicalResult ProgramMaterializer::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  auto emitBuilderParameters = [&](raw_ostream &stream) {
    bool first = true;
    auto parameter = [&](StringRef value) {
      if (!first)
        stream << ", ";
      stream << value;
      first = false;
    };
    for (const std::string &dimension : dimensionOrder)
      parameter(dimension);
    for (int64_t axis : planIndex.components.raggedProgramAxes)
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
          target::lowering::privateWorkspaceElementCount(
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
      FailureOr<std::string> size = target::lowering::scanWorkspaceElementCount(
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

  emitDecorator(output);
  output << "def " << kernelName << "(";
  emitBuilderParameters(output);
  output << "):\n";
  emitBlockExtentConstants(output);
  llvm::StringSet<> selectedContiguousConstraints;
  for (const auto &entry : planIndex.boundaries) {
    const plan::BoundaryOp &boundary = entry.second;
    if (boundary.getTransfer() != "selected_contiguous")
      continue;
    Operation *operation = kernel.nodes.lookup(boundary.getNode());
    FailureOr<SmallVector<std::string>> physical =
        operation ? tensorExtents(*operation, 0)
                  : FailureOr<SmallVector<std::string>>(failure());
    FailureOr<SmallVector<std::string>> logical =
        operation ? tensorExtents(*operation, 0, false)
                  : FailureOr<SmallVector<std::string>>(failure());
    if (failed(physical) || failed(logical) ||
        physical->size() != logical->size())
      return boundary.emitOpError(
          "has no exact selected-contiguous TileLang result shape");
    for (auto [physicalExtent, logicalExtent] :
         llvm::zip(*physical, *logical))
      if (physicalExtent != logicalExtent)
        selectedContiguousConstraints.insert(physicalExtent + " != " +
                                             logicalExtent);
    for (int64_t node : boundary.getDomainNodes()) {
      auto axis = planIndex.axes.find(node);
      if (axis == planIndex.axes.end() || axis->second.isScalar())
        continue;
      std::string extent = axisDimensions.lookup(node);
      std::string tile = axis->second.getTile().str();
      if (extent.empty() || tile.empty())
        return boundary.emitOpError(
            "has no exact selected-contiguous TileLang axis interval");
      selectedContiguousConstraints.insert(extent + " % " + tile + " != 0");
    }
  }
  for (StringRef constraint : selectedContiguousConstraints.keys()) {
    output << "    if " << constraint << ":\n";
    output << "        raise NotImplementedError('TileLang selected-contiguous "
              "transfer requires exact physical tiles')\n";
  }
  output << "    @T.prim_func\n    def main(";
  if (failed(emitMainParameters(output)))
    return failure();
  output << "):\n";
  if (!planIndex.components.reusedAxes.empty()) {
    output << "        with T.Kernel("
           << roleDimensions.lookup("program_0")
           << ", threads=threads) as program_index:\n";
    if (planIndex.components.reusedAxes.size() != 1)
      return physicalProgram.emitOpError(
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
      target::lowering::orderedProgramAxes(planIndex);
  auto axisExtent = [&](plan::AxisOp axis) {
    std::string role = "program_" + std::to_string(axis.getProgramOrder());
    std::string extent =
        planIndex.components.raggedProgramAxes.contains(axis.getNode())
            ? "MAX_SEQUENCE_LENGTH_" + std::to_string(axis.getNode())
            : roleDimensions.lookup(role);
    return axis.isScalar()
               ? extent
               : "T.ceildiv(" + extent + ", " + axis.getTile().str() + ")";
  };
  std::array<std::string, 3> grid =
      target::lowering::projectProgramGrid(planIndex, axisExtent);
  bool persistent = planIndex.program.getPersistent();
  std::string linear = "persistent_program";
  unsigned workers = 0;
  for (plan::AxisOp axis : programAxes)
    workers = std::max(workers,
                       static_cast<unsigned>(axis.getWorkerAxis() + 1));
  if (persistent) {
    output << "        total_program_tiles = "
           << target::lowering::projectProgramVolume(planIndex, axisExtent)
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
          target::lowering::projectLinearGroupIndex(
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
  SmallVector<target::lowering::ProgramIndexProjection> projections =
      persistent
          ? target::lowering::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::lowering::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("pid_worker_" + std::to_string(worker));
                });
  for (const target::lowering::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    output << programIndent << block << " = "
           << addressIndex(projection.expression) << "\n";
    programBlocks[axis.getNode()] = block;
    axisIndices[axis.getNode()] =
        axis.isScalar() ? block : block + " * " + axis.getTile().str();
  }
  for (const auto &entry : planIndex.axes) {
    plan::AxisOp axis = entry.second;
    if (!axisIndices.lookup(axis.getNode()).empty() ||
        !axis.getRange("reduction", 0))
      continue;
    axisIndices[axis.getNode()] = "0";
  }
  if (needsPackedInt2Decode)
    output << programIndent << "T.import_source(_INTENT_U2X16_TO_I8_SOURCE)\n";
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
    SmallVector<std::string> dynamicContractionRowExtents;
    for (const auto &entry : planIndex.contracts) {
      std::optional<int64_t> axisNode = entry.second.getLhsResultAxisNode();
      auto axis = axisNode ? planIndex.axes.find(*axisNode) : planIndex.axes.end();
      const target::lowering::RangeBinding *range =
          axis == planIndex.axes.end()
              ? nullptr
              : target::lowering::canonicalDomainRange(axis->second);
      if (!range)
        continue;
      int64_t staticExtent = 0;
      if (!range->getExtent().getAsInteger(10, staticExtent))
        continue;
      std::string extent = syntax::dimension(range->getExtent());
      if (!llvm::is_contained(dynamicContractionRowExtents, extent))
        dynamicContractionRowExtents.push_back(std::move(extent));
    }
    for (const std::string &extent : dynamicContractionRowExtents) {
      output << "    if 0 < " << extent << " < 16:\n";
      output << "        raise NotImplementedError('TileLang cannot pad a "
                "logical contraction row extent smaller than 16 to a native "
                "MMA fragment')\n";
    }
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
            "has no supported TileLang private-workspace allocation");
      output << "    " << workspaceNames.lookup(buffer->getResult(0))
             << " = torch.full((" << *size << ",), " << *initializer
             << ", device=_DEVICE, dtype=" << dtype << ")\n";
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "global")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      FailureOr<std::string> size = target::lowering::scanWorkspaceElementCount(
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
  };
  auto emitBuilderArguments = [&]() {
    bool first = true;
    for (const std::string &dimension : dimensionOrder) {
      if (!first)
        output << ", ";
      output << dimension;
      first = false;
    }
    for (int64_t axis : planIndex.components.raggedProgramAxes) {
      if (!first)
        output << ", ";
      output << "max_sequence_length_" << axis;
      first = false;
    }
  };

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
  for (int64_t axis : planIndex.components.raggedProgramAxes) {
    FailureOr<plan::RaggedOp> relation =
        target::lowering::uniqueRaggedRelation(
            planIndex, axis, *kernel.entry.getOperation());
    auto runtime = succeeded(relation)
                       ? raggedRuntimeByRelation.find(relation->getNode())
                       : raggedRuntimeByRelation.end();
    if (failed(relation) || runtime == raggedRuntimeByRelation.end())
      return kernel.entry.emitOpError(
          "has no ragged program runtime metadata");
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
  for (int64_t axis : planIndex.components.raggedProgramAxes)
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
    emitBuilderArguments();
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
    emitBuilderArguments();
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
  if (requiresPreallocatedOutputs) {
    output << "    raise NotImplementedError('this callable requires preallocated output views; use launch')\n\n\n";
    return success();
  }
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

FailureOr<StringRef> ProgramMaterializer::lookupValue(Operation &consumer,
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
        "cannot resolve multi-axis scalar ownership during TileLang emission");
    return failure();
  }
  if (scalarSource->opaque) {
    consumer.emitOpError(
        "cannot resolve an opaque index source during TileLang emission");
    return failure();
  }
  if (Operation *definition = indexedValue.getDefiningOp())
    if (::intent::target::semanticOperationName(*definition) == "intent.domain" ||
        ::intent::target::semanticOperationName(*definition) == "intent.ragged_outer" ||
        ::intent::target::semanticOperationName(*definition) == "intent.ragged_member")
      return definition;
  consumer.emitOpError("cannot resolve index ownership during TileLang emission");
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
    consumer.emitOpError("indexes a domain without a TileLang axis binding");
    return failure();
  }
  return axis;
}

FailureOr<std::string> ProgramMaterializer::dimensionName(Operation &domain) {
  return target::lowering::plannedDomainExtent(domain, planIndex);
}

bool ProgramMaterializer::usesMatrixContraction() const {
  return !planIndex.contracts.empty() || !planIndex.sparseContracts.empty();
}

std::string ProgramMaterializer::addressIndex(StringRef expression) const {
  return "(" + expression.str() + ")";
}

std::string ProgramMaterializer::physicalExtent(StringRef logicalExtent) const {
  auto extent = planIndex.blockExtents.find(logicalExtent);
  if (extent == planIndex.blockExtents.end())
    return syntax::dimension(logicalExtent);
  return "PHYSICAL_" + syntax::dimension(logicalExtent);
}

std::string ProgramMaterializer::logicalExtent(StringRef extent) const {
  return syntax::dimension(extent);
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

FailureOr<std::string> ProgramMaterializer::structuredIndexExpression(
    Value value, Operation &consumer) {
  FailureOr<int64_t> consumerNode =
      target::getNodeID(consumer, "TileLang structured-index projection");
  plan::BoundaryOp boundary =
      succeeded(consumerNode) ? planIndex.boundaries.lookup(*consumerNode)
                              : plan::BoundaryOp();
  auto selectedScalars =
      boundary ? boundary.operation->getAttrOfType<DenseI64ArrayAttr>(
                     selectedScalarIndexNodesAttr)
               : DenseI64ArrayAttr();
  llvm::DenseSet<Value> active;
  auto project = [&](auto &self, Value current) -> FailureOr<std::string> {
    if (!active.insert(current).second)
      return consumer.emitOpError("contains a cyclic structured index");
    auto finish = [&](FailureOr<std::string> result) {
      active.erase(current);
      return result;
    };
    if (isa<BlockArgument>(current)) {
      auto emitted = regionIndices.find(current);
      return finish(emitted == regionIndices.end()
                        ? FailureOr<std::string>(failure())
                        : FailureOr<std::string>(emitted->second));
    }
    Operation *definition = current.getDefiningOp();
    if (!definition)
      return finish(failure());
    StringRef name = ::intent::target::semanticOperationName(*definition);
    auto node = definition->getAttrOfType<IntegerAttr>("intent.node");
    if (node && selectedScalars &&
        llvm::is_contained(selectedScalars.asArrayRef(), node.getInt())) {
      auto assumed = assumedIndexNames.find(current);
      if (assumed != assumedIndexNames.end())
        return finish(assumed->second);
      auto emitted = valueNames.find(current);
      auto tensor = dyn_cast<RankedTensorType>(current.getType());
      if (emitted == valueNames.end() || !tensor || tensor.getRank() <= 0)
        return finish(failure());
      std::string scalar = emitted->second + "[";
      for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
        if (axis)
          scalar += ", ";
        scalar += "0";
      }
      scalar += "]";
      return finish(std::move(scalar));
    }
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
      if (auto argument =
              dyn_cast<BlockArgument>(definition->getOperand(0))) {
        auto emitted = regionIndices.find(argument);
        return finish(emitted == regionIndices.end()
                          ? FailureOr<std::string>(failure())
                          : FailureOr<std::string>(emitted->second));
      }
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
      auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
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

FailureOr<std::string> ProgramMaterializer::accessIndices(Operation &operation) {
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
      if (!boundary.hasDataDependentTensorIndex() ||
          boundary.getTransfer() == "selected_contiguous") {
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
    if (term.kind == "region_index" && isa<BlockArgument>(indexed)) {
      FailureOr<target::lowering::RegionRangeBinding> selected =
          target::lowering::selectedRegionArgumentRange(planIndex, kernel,
                                                        indexed, operation);
      if (failed(selected))
        return failure();
      auto exact = regionIndices.find(indexed);
      if (exact == regionIndices.end())
        return operation.emitOpError(
            "has no active TileLang index for its selected region range");
      std::string base = exact->second;
      base = addressIndex(base);
      indices.push_back(selected->range.getTileRole() == "one"
                            ? base
                            : base + " : " + base + " + " +
                                  selected->range.getTile().str());
      continue;
    }
    if (term.kind == "value_index" ||
        target::lowering::isSequentialIterator(indexed)) {
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
      if (target::lowering::isPackedScalarAxis(*axis)) {
        indices.push_back(wideBase);
        continue;
      }
      std::string extent = axis->getTile().str();
      if (!axis->getReuseWorker() &&
          axis->getTileRole().starts_with("row_vector")) {
        const target::lowering::RangeBinding *range = axis->roleRange();
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

FailureOr<std::string>
ProgramMaterializer::elementAccessIndices(Operation &operation,
                                    ArrayRef<std::string> tileIndices) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  target::TensorIndexGroup tensorIndices =
      target::tensorIndexGroup(operation, *relation);
  SmallVector<std::string> indices;
  unsigned tileAxis = 0;
  std::optional<unsigned> tensorGroupAxis;
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
      if (tensorIndices.requiresBroadcastProjection()) {
        if (!tensorGroupAxis) {
          if (tileAxis + tensorIndices.rank > tileIndices.size())
            return operation.emitOpError(
                "TileLang broadcasted tensor index exceeds the transfer rank");
          tensorGroupAxis = tileAxis;
          tileAxis += tensorIndices.rank;
        }
        if (tensor.getRank() > static_cast<int64_t>(tensorIndices.rank))
          return operation.emitOpError(
              "TileLang broadcasted tensor index exceeds the transfer rank");
        auto result = dyn_cast<OpResult>(indexed);
        FailureOr<SmallVector<std::string>> extents =
            result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                   : FailureOr<SmallVector<std::string>>(failure());
        if (failed(extents) || extents->size() != static_cast<size_t>(tensor.getRank()))
          return operation.emitOpError(
              "TileLang broadcasted tensor index has no canonical extents");
        unsigned valueAxis =
            *tensorGroupAxis + tensorIndices.rank - tensor.getRank();
        std::string element;
        std::optional<unsigned> varyingAxis;
        for (auto [axis, extent] : llvm::enumerate(*extents)) {
          if (extent == "1")
            continue;
          if (varyingAxis) {
            varyingAxis.reset();
            break;
          }
          varyingAxis = axis;
        }
        bool structured = succeeded(projected) &&
                          llvm::count_if(*extents, [](const std::string &extent) {
                            return extent != "1";
                          }) <= 1;
        if (structured) {
          element = *projected;
          if (varyingAxis)
            element += " + " + tileIndices[valueAxis + *varyingAxis];
        } else {
          element = exact->str() + "[";
          for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
            if (axis)
              element += ", ";
            element += (*extents)[axis] == "1"
                           ? "0"
                           : tileIndices[valueAxis + axis];
          }
          element += "]";
        }
        indices.push_back(addressIndex(element));
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
    if (term.kind == "region_index" && isa<BlockArgument>(indexed)) {
      FailureOr<target::lowering::RegionRangeBinding> selected =
          target::lowering::selectedRegionArgumentRange(planIndex, kernel,
                                                        indexed, operation);
      if (failed(selected))
        return failure();
      auto exact = regionIndices.find(indexed);
      if (exact == regionIndices.end())
        return operation.emitOpError(
            "has no active TileLang index for its selected region range");
      std::string base = exact->second;
      if (selected->range.getTileRole() == "one") {
        indices.push_back(addressIndex(base));
      } else {
        if (tileAxis >= tileIndices.size())
          return operation.emitOpError(
              "parallel TileLang transfer has too few region indices");
        indices.push_back(addressIndex(base) + " + " +
                          addressIndex(tileIndices[tileAxis++]));
      }
      continue;
    }
    if (term.kind == "value_index" ||
        target::lowering::isSequentialIterator(indexed)) {
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
      if (target::lowering::isPackedScalarAxis(*axis)) {
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
ProgramMaterializer::elementBoundsPredicate(Operation &operation,
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
  target::TensorIndexGroup tensorIndices =
      target::tensorIndexGroup(operation, *relation);
  SmallVector<std::string> predicates;
  unsigned tileAxis = 0;
  unsigned sourceAxis = 0;
  std::optional<unsigned> tensorGroupAxis;
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
      if (tensorIndices.requiresBroadcastProjection()) {
        if (!tensorGroupAxis) {
          if (tileAxis + tensorIndices.rank > tileIndices.size())
            return operation.emitOpError(
                "TileLang broadcasted tensor bounds exceed the transfer rank");
          tensorGroupAxis = tileAxis;
          tileAxis += tensorIndices.rank;
        }
        if (tensor.getRank() > static_cast<int64_t>(tensorIndices.rank))
          return operation.emitOpError(
              "TileLang broadcasted tensor bounds exceed the transfer rank");
        auto result = dyn_cast<OpResult>(indexed);
        FailureOr<SmallVector<std::string>> extents =
            result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                   : FailureOr<SmallVector<std::string>>(failure());
        if (failed(extents) || extents->size() != static_cast<size_t>(tensor.getRank()))
          return operation.emitOpError(
              "TileLang broadcasted tensor bounds have no canonical extents");
        unsigned valueAxis =
            *tensorGroupAxis + tensorIndices.rank - tensor.getRank();
        std::optional<unsigned> varyingAxis;
        for (auto [axis, extent] : llvm::enumerate(*extents)) {
          if (extent == "1")
            continue;
          if (varyingAxis) {
            varyingAxis.reset();
            break;
          }
          varyingAxis = axis;
        }
        bool structured = succeeded(projected) &&
                          llvm::count_if(*extents, [](const std::string &extent) {
                            return extent != "1";
                          }) <= 1;
        if (structured) {
          index = *projected;
          if (varyingAxis)
            index += " + " + tileIndices[valueAxis + *varyingAxis];
        } else {
          index = exact->str() + "[";
          for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
            if (axis)
              index += ", ";
            index += (*extents)[axis] == "1"
                         ? "0"
                         : tileIndices[valueAxis + axis];
          }
          index += "]";
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
            target::lowering::isPackedScalarAxis(axis->second);
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
    if (target::lowering::isPackedScalarAxis(*axis)) {
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
    if (term.kind == "region_index" && isa<BlockArgument>(indexed)) {
      FailureOr<target::lowering::RegionRangeBinding> region =
          target::lowering::selectedRegionArgumentRange(planIndex, kernel,
                                                        indexed, operation);
      if (failed(region) || region->axis.getNode() != axis->getNode())
        return operation.emitOpError(
            "has no exact TileLang region interval for bounded transfer");
      base = regionIndices.lookup(indexed);
    }
    Operation *domain = kernel.nodes.lookup(axis->getNode());
    FailureOr<std::string> extent = failure();
    bool plainRaggedMember =
        isPlainRaggedMemberAxis(planIndex, axis->getNode());
    if (plainRaggedMember ||
        target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> orderedAxis =
          plainRaggedMember
              ? FailureOr<int64_t>(axis->getNode())
              : target::lowering::representativeOrderedAxis(
                    planIndex, axis->getNode(), operation);
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

FailureOr<std::string> ProgramMaterializer::scalarTransferPredicate(
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
        !target::lowering::isPackedScalarAxis(found->second))
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

FailureOr<std::string> ProgramMaterializer::tileBoundsPredicate(
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
  target::TensorIndexGroup tensorIndices =
      target::tensorIndexGroup(operation, *relation);
  SmallVector<std::string> predicates;
  unsigned tileAxis = 0;
  unsigned sourceAxis = 0;
  std::optional<unsigned> tensorGroupAxis;
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
      if (!tensorIndices.requiresBroadcastProjection()) {
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
      if (!tensorGroupAxis) {
        if (tileAxis + tensorIndices.rank > tileExtents.size())
          return operation.emitOpError(
              "TileLang structured bulk bounds exceed the transfer rank");
        tensorGroupAxis = tileAxis;
        tileAxis += tensorIndices.rank;
      }
      if (tensor.getRank() > static_cast<int64_t>(tensorIndices.rank))
        return operation.emitOpError(
            "TileLang structured bulk bounds exceed the transfer rank");
      auto result = dyn_cast<OpResult>(indexed);
      FailureOr<SmallVector<std::string>> extents =
          result ? tensorExtents(*result.getOwner(), result.getResultNumber())
                 : FailureOr<SmallVector<std::string>>(failure());
      FailureOr<std::string> base =
          structuredIndexExpression(indexed, operation);
      if (failed(extents) || failed(base) ||
          extents->size() != static_cast<size_t>(tensor.getRank()))
        return operation.emitOpError(
            "TileLang structured bulk bounds have no canonical index span");
      std::optional<unsigned> varyingAxis;
      for (auto [axis, extent] : llvm::enumerate(*extents)) {
        if (extent == "1")
          continue;
        if (varyingAxis)
          return operation.emitOpError(
              "TileLang structured bulk index varies along multiple axes");
        varyingAxis = axis;
      }
      predicates.push_back("0 <= " + *base);
      if (varyingAxis) {
        unsigned valueAxis = *tensorGroupAxis + tensorIndices.rank -
                             tensor.getRank() + *varyingAxis;
        predicates.push_back(*base + " + " + tileExtents[valueAxis] +
                             " <= " + (*view)->shape[axisNumber]);
      } else {
        predicates.push_back(*base + " < " + (*view)->shape[axisNumber]);
      }
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
    bool plainRaggedMember =
        isPlainRaggedMemberAxis(planIndex, axis->getNode());
    if (plainRaggedMember ||
        target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> orderedAxis =
          plainRaggedMember
              ? FailureOr<int64_t>(axis->getNode())
              : target::lowering::representativeOrderedAxis(
                    planIndex, axis->getNode(), operation);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else {
      Operation *domain = kernel.nodes.lookup(axis->getNode());
      if (domain)
        extent = dimensionName(*domain);
    }
    if (auto active = activeTraversalEnds.find(axis->getNode());
        active != activeTraversalEnds.end())
      extent = active->second;
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

FailureOr<std::string> ProgramMaterializer::wholeTileBoundsPredicate(
    Operation &operation, ArrayRef<std::string> tileExtents) {
  return tileBoundsPredicate(operation, tileExtents, true);
}

FailureOr<std::string> ProgramMaterializer::tileFitsViewPredicate(
    Operation &operation, ArrayRef<std::string> tileExtents) {
  return tileBoundsPredicate(operation, tileExtents, false);
}

FailureOr<std::string> ProgramMaterializer::elementValidityPredicate(
    ArrayRef<int64_t> tensorAxes, ArrayRef<int64_t> domainNodes,
    ArrayRef<std::string> elementIndices, Value value,
    Operation &consumer) {
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
    FailureOr<std::optional<target::lowering::ResultAxisRegionRangeBinding>>
        region = target::lowering::selectedResultAxisRegionRange(
            planIndex, kernel, value, tensorAxis, consumer);
    if (failed(region))
      return failure();
    if (*region && (*region)->selected.axis.getNode() != domainNode)
      return consumer.emitOpError(
          "binds one result axis to conflicting logical and physical axes");
    std::string base = axisIndices.lookup(axis.getNode());
    if (*region)
      base = regionIndices.lookup((*region)->argument);
    if (base.empty())
      return consumer.emitOpError(
          "has no active TileLang index for its planned validity axis");
    FailureOr<std::string> extent = failure();
    bool plainRaggedMember =
        isPlainRaggedMemberAxis(planIndex, axis.getNode());
    if (plainRaggedMember ||
        target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> orderedAxis =
          plainRaggedMember
              ? FailureOr<int64_t>(axis.getNode())
              : target::lowering::representativeOrderedAxis(
                    planIndex, axis.getNode(), consumer);
      if (failed(orderedAxis))
        return failure();
      extent = "sequence_end_" + std::to_string(*orderedAxis);
    } else {
      extent = dimensionName(*domain->second);
    }
    if (auto active = activeTraversalEnds.find(axis.getNode());
        active != activeTraversalEnds.end())
      extent = active->second;
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

FailureOr<std::string>
ProgramMaterializer::paddingFillExpression(Value value, Operation &consumer) {
  FailureOr<int64_t> valueID =
      target::getValueID(value, kernel, consumer, "TileLang padding lookup");
  if (failed(valueID))
    return failure();
  plan::PaddingOp padding = planIndex.paddings.lookup(*valueID);
  if (!padding)
    return std::string();
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor)
    return consumer.emitOpError(
        "cannot apply TileLang padding to a non-tensor value");
  std::string dtype = dtypeName(tensor.getElementType(), consumer);
  if (dtype.empty())
    return failure();
  StringRef planned = padding.getFill();
  return planned == "negative_infinity"
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
         : isa<IntegerType, IndexType>(tensor.getElementType()) ? "0" : "0.0";
}

FailureOr<std::string> ProgramMaterializer::padElementExpression(
    Value value, StringRef expression, ArrayRef<std::string> elementIndices,
    Operation &consumer) {
  FailureOr<int64_t> valueID =
      target::getValueID(value, kernel, consumer, "TileLang padding lookup");
  if (failed(valueID))
    return failure();
  plan::PaddingOp padding = planIndex.paddings.lookup(*valueID);
  if (!padding)
    return expression.str();
  FailureOr<std::string> predicate = elementValidityPredicate(
      padding.getTensorAxes(), padding.getDomainNodes(), elementIndices,
      value, consumer);
  FailureOr<std::string> fill = paddingFillExpression(value, consumer);
  if (failed(predicate) || failed(fill) || fill->empty())
    return failure();
  return "T.if_then_else(" + *predicate + ", " + expression.str() + ", " +
         *fill + ")";
}

FailureOr<SmallVector<std::string>>
ProgramMaterializer::tensorExtents(Operation &operation, unsigned resultIndex,
                             bool physical) {
  auto shapes = operation.getAttrOfType<ArrayAttr>("intent.result_shapes");
  auto shape = shapes && resultIndex < shapes.size()
                   ? dyn_cast<ArrayAttr>(shapes[resultIndex])
                   : ArrayAttr();
  if (!shape)
    return operation.emitOpError("has no canonical tensor shape metadata");
  SmallVector<std::string> extents;
  Value resultValue = operation.getResult(resultIndex);
  for (auto [tensorAxis, attribute] : llvm::enumerate(shape)) {
    auto label = dyn_cast<StringAttr>(attribute);
    if (!label)
      return operation.emitOpError(
          "tensor shape contains a non-symbolic extent");
    FailureOr<std::optional<target::lowering::ResultAxisRegionRangeBinding>>
        selected = target::lowering::selectedResultAxisRegionRange(
            planIndex, kernel, resultValue, tensorAxis, operation);
    if (failed(selected))
      return failure();
    if (*selected) {
      const target::lowering::RegionRangeBinding &binding = (*selected)->selected;
      const target::lowering::RangeBinding &range = binding.range;
      bool rounded = physical && !binding.axis.getReuseWorker() &&
                     range.getTileRole().starts_with("row_vector");
      extents.push_back(rounded ? physicalExtent(range.getExtent())
                                : range.getTile().str());
      continue;
    }
    auto tile = regionTiles.find(label.getValue());
    if (tile != regionTiles.end())
      extents.push_back(tile->getValue());
    else if (label.getValue().starts_with("?region_"))
      return operation.emitOpError(
          "tensor shape region has no TileLang tile binding");
    else
      extents.push_back(physical ? physicalExtent(label.getValue())
                                 : syntax::dimension(label.getValue()));
  }
  return extents;
}

FailureOr<std::string> ProgramMaterializer::tensorShape(Operation &operation,
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
ProgramMaterializer::tensorElement(Value value, ArrayRef<std::string> indices,
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

FailureOr<std::string> ProgramMaterializer::allocateResult(Operation &operation,
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

std::string ProgramMaterializer::dtypeName(Type type, Operation &consumer) {
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
    return integer.isUnsigned() ? "T.uint32" : "T.int32";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 64)
    return integer.isUnsigned() ? "T.uint64" : "T.int64";
  if (isa<IndexType>(type))
    return "T.int32";
  consumer.emitOpError("uses an unsupported TileLang dtype");
  return {};
}

void ProgramMaterializer::bindResult(Operation &operation, unsigned index,
                               StringRef name) {
  Value value = operation.getResult(index);
  valueNames[value] = name.str();
}

std::string ProgramMaterializer::uniqueName(StringRef candidate, int64_t node) {
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

} // namespace intent::tilelang::lowering

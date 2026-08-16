#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Emission/Combiner.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::triton::emission {
namespace {

StringRef torchDtype(Type type) {
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
    return "torch.uint8";
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
}

StringRef combinerDtype(Type type) {
  if (type.isInteger(1))
    return "tl.int1";
  if (type.isInteger(8))
    return cast<IntegerType>(type).isUnsigned() ? "tl.uint8" : "tl.int8";
  if (type.isInteger(32))
    return "tl.int32";
  if (type.isInteger(64) || isa<IndexType>(type))
    return "tl.int64";
  if (type.isF16())
    return "tl.float16";
  if (type.isBF16())
    return "tl.bfloat16";
  if (type.isF32())
    return "tl.float32";
  return {};
}

FailureOr<std::string>
combinerExpression(Operation &operation, ArrayRef<std::string> operands) {
  FailureOr<std::string> role = target::emission::pointwiseRole(operation);
  FailureOr<StringRef> spelling =
      succeeded(role) ? syntax::pointwise(&operation, *role)
                      : FailureOr<StringRef>(failure());
  if (failed(spelling))
    return failure();
  return target::emission::renderPythonPointwiseExpression(
      operation, operands, *spelling, [&](Type type) -> FailureOr<std::string> {
        StringRef dtype = combinerDtype(type);
        return dtype.empty() ? FailureOr<std::string>(failure())
                             : FailureOr<std::string>(dtype.str());
      });
}

} // namespace

FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization,
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
    } else if (auto value = dyn_cast<intent::plan::ProgramOp>(operation)) {
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
      binding.lhsSpace = value.getLhsSpace().str();
      binding.rhsSpace = value.getRhsSpace().str();
      binding.accumulatorSpace = value.getAccumulatorSpace().str();
      index.contracts[value.getNode()] = binding;
    } else if (auto value = dyn_cast<intent::plan::SparseContractOp>(operation)) {
      Operation *sparse = kernel.nodes.lookup(value.getNode());
      if (!sparse || sparse->getName().getStringRef() != "intent.sparse_contract")
        return value.emitOpError("does not bind sparse contraction semantics");
      return sparse->emitOpError(
          "Triton has no native 2:4 sparse contraction projection");
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
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = syntax::reduction(*role).str();
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = *axis;
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::scanRole(*operation)
                  : FailureOr<std::string>(failure());
    if (failed(role) || !index.axes.count(value.getAxisNode()))
      return value.emitOpError("does not bind a canonical scan");
    plan::ScanOp binding;
    binding.operation = value;
    binding.lowering = syntax::scan(*role).str();
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = value.getTensorAxis();
    binding.axisNode = value.getAxisNode();
    index.scans[value.getNode()] = binding;
  }
  for (intent::plan::PointwiseOp value : pointwise) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<StringRef> lowering =
        succeeded(role) ? syntax::pointwise(value, *role)
                        : FailureOr<StringRef>(failure());
    if (failed(lowering))
      return value.emitOpError("does not bind canonical pointwise semantics");
    plan::PointwiseOp binding;
    binding.operation = value;
    binding.lowering = lowering->str();
    binding.resultSpace = value.getResultSpace().str();
    binding.defer = target::emission::feedsStagedContraction(index, *operation);
    index.pointwise[value.getNode()] = binding;
  }
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
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = load ? "load" : "store";
    binding.resultSpace = value.getResultSpace().str();
    index.boundaries[value.getNode()] = binding;
  }
  for (auto &entry : index.boundaries) {
    Operation *operation = kernel.nodes.lookup(entry.first);
    entry.second.defer =
        operation && target::emission::deferSharedContractionTransfer(
                         index, *operation, entry.second.getResultSpace());
  }
  if (!index.target || !index.program) {
    realization.emitOpError("lacks target or program realization choices");
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
                             intent::plan::RealizationOp realization,
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
    std::string laneDimension = roleDimensions.lookup("lane_0");
    bool hasExternalRead = false;
    bool hasAggregation = false;
    bool hasScan = false;
    kernel.entry.walk([&](Operation *operation) {
      StringRef name = operation->getName().getStringRef();
      hasExternalRead |= name == "intent.view_load";
      hasScan |= name == "intent.scan";
      hasAggregation |= name == "intent.reduce" ||
                        name == "intent.arg_reduce" ||
                        name == "intent.scan" ||
                        name == "intent.contract" ||
                        name == "intent.sparse_contract" ||
                        name == "intent.state_stream";
    });
    configureRowVector = lane != planIndex.axesByRole.end() &&
                         lane->second.getTileRole().starts_with("row_vector") &&
                         llvm::is_contained(dimensionOrder, laneDimension) &&
                         (hasExternalRead || !hasAggregation) &&
                         !hasScan &&
                         !target::emission::hasNonReplayableEffect(
                             kernel.entry.getOperation());
  }
  if (failed(target::emission::indexScanProducerOperations(
          kernel, planIndex, scanProducerOwners)))
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
        "workspace_" + std::to_string(node) + "_ptr";
    privateWorkspaceBuffers.push_back(buffer);
  }
  for (const auto &entry : planIndex.scans) {
    plan::ScanOp binding = entry.second;
    if (binding.getResultSpace() != "private_workspace")
      continue;
    Operation *scan = kernel.nodes.lookup(entry.first);
    std::string extent = axisDimensions.lookup(binding.getAxisNode());
    if (!scan || scan->getName().getStringRef() != "intent.scan" ||
        scan->getNumResults() == 0 || extent.empty())
      return binding.emitOpError("does not bind a workspace-backed scan tensor");
    for (auto [component, result] : llvm::enumerate(scan->getResults())) {
      workspaceNames[result] = "scan_workspace_" +
                               std::to_string(entry.first) + "_" +
                               std::to_string(component) + "_ptr";
      scanResults[result] = binding;
    }
    scanExtents[entry.first] = extent;
    for (int64_t valueID : binding.getMaterializedValues()) {
      FailureOr<Value> value = target::emission::lookupScanMaterializedValue(
          kernel, binding, valueID);
      if (failed(value))
        return failure();
      workspaceNames[*value] =
          "scan_materialized_" + std::to_string(valueID) + "_ptr";
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
               << " has no Triton runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("Triton emitter requires ranked views");
    ABIView emitted{&argument, view, tensor, argument.name + "_ptr", {}, {}};
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis)
      emitted.strides.push_back(argument.name + "_stride_" +
                                std::to_string(axis));
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
        if (!dimensionOwners.count(symbol.getValue())) {
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
  }
  if (views.empty())
    return kernel.entry.emitOpError("Triton emitter requires external views");
  for (ABIView &view : views)
    valueNames[view.argument->value] = view.pointer;
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
  for (const auto &entry : planIndex.axes) {
    plan::AxisOp axis = entry.second;
    Operation *domain = kernel.nodes.lookup(axis.getNode());
    if (!domain || domain->getNumResults() != 1)
      return axis.emitOpError(
          "cannot index its domain result shape against the Triton plan");
    FailureOr<int64_t> valueID = target::getValueID(
        domain->getResult(0), kernel, *domain, "Triton domain tile binding");
    FailureOr<std::string> tile = physicalAxisTile(axis);
    if (failed(valueID) || failed(tile))
      return failure();
    regionTiles["?region_" + std::to_string(*valueID) + "_0"] = *tile;
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
    auto program = planIndex.axesByRole.find("program_0");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (program == planIndex.axesByRole.end() ||
        lane == planIndex.axesByRole.end())
      return realization.emitOpError(
          "grid-stride plan requires program_0 and lane_0 axes");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if ((view.view.getAccess() == "out" ||
           view.view.getAccess() == "inout") &&
          !fixedOutput)
        fixedOutput = &view;
    }
    if (!fixedOutput || fixedOutput->tensor.getRank() < 1)
      return kernel.entry.emitOpError(
          "grid-stride Triton program requires one ranked output view");
    if (searchSpace)
      return searchSpace.emitOpError(
          "fixed grid-stride scheduling cannot consume an autotune space");
  }
  if (target::emission::requiresDelegatedTuning(planIndex) &&
      (!searchSpace || !searchIndex.autotune))
    return realization.emitOpError(
        "tiled physical components require a delegated Triton tuner");
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
          !isa<RankedTensorType>(found->second.getType()) ||
          !stageOutputOwners.try_emplace(found->second, position).second)
        return stage.emitOpError("has an invalid stage output value");
      workspaceNames[found->second] =
          "workspace_" + std::to_string(valueID) + "_ptr";
    }
    if (!llvm::is_contained(operationStages.lookup(contract), position))
      return stage.emitOpError("stage roots do not depend on its contraction");
  }
  return success();
}

bool SourceEmitter::selectOperation(Operation &operation) {
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

void SourceEmitter::bindResult(Operation &operation, unsigned index,
                               StringRef name) {
  Value value = operation.getResult(index);
  valueNames[value] = name.str();
  auto outputStage = stageOutputOwners.find(value);
  if (planIndex.stages.empty() || outputStage == stageOutputOwners.end())
    return;
  unsigned stage = outputStage->second;
  std::string feature = stageFeatureDimensions.lookup(stage);
  line("tl.store(" + workspaceNames.lookup(value) +
       " + " + addressIndex("member_offsets[:, None]") + " * " +
       addressIndex(feature) + " + " + addressIndex("offs_feature[None, :]") +
       ", " + name.str() +
       ", mask=member_mask[:, None] & feature_mask[None, :])");
}

void SourceEmitter::emitImports() {
  output << "import torch\n";
  output << "import triton\n";
  output << "import triton.language as tl\n";
  if (llvm::any_of(planIndex.pointwise, [](const auto &entry) {
        return entry.second.getLowering() == "libdevice.pow";
      }))
    output << "from triton.language.extra import libdevice\n";
  if (!planIndex.components.reusedAxes.empty() || configureRowVector) {
    output << "from intent.runtime.tuning.triton import row_autotune_configurations\n";
    output << "\n_ROW_CONFIGS = row_autotune_configurations(persistent="
           << (!planIndex.components.reusedAxes.empty() ? "True" : "False")
           << ")\n";
    output << "_ROW_TUNED_KEYS = set()\n";
  }
  if (searchSpace) {
    output << "from intent.runtime.tuning.triton import autotune_configurations\n";
    output << "\n_PARAMETER_MAP = {";
    for (auto [index, mapping] :
         llvm::enumerate(searchIndex.autotune.getParameterMap())) {
      if (index)
        output << ", ";
      output << "'" << mapping.getName().getValue() << "': '"
             << cast<StringAttr>(mapping.getValue()).getValue() << "'";
    }
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP";
    if (usesScaledContraction())
      output << ", {'USE_NATIVE_SCALED': (0, 1)}";
    output << ")\n";
  }
  output << "\n\n";
}

LogicalResult SourceEmitter::emitHelpers() {
  FailureOr<SmallVector<func::FuncOp>> combiners =
      target::emission::collectCombiners(kernel.entry);
  if (failed(combiners))
    return failure();
  for (func::FuncOp function : *combiners) {
    FailureOr<std::string> source = target::emission::renderPythonCombiner(
        function, function.getName(), "@triton.jit", combinerExpression);
    if (failed(source))
      return failure();
    output << *source;
  }
  WalkResult projected = kernel.entry.walk([&](Operation *operation) {
    if (!target::emission::hasGenericCombiner(*operation) ||
        operation->getAttrOfType<StringAttr>("intent.combine_builtin"))
      return WalkResult::advance();
    FailureOr<target::emission::CombinerUse> combiner =
        target::emission::resolveCombiner(*operation);
    if (failed(combiner))
      return WalkResult::interrupt();
    if (combiner->captureCount == 0)
      return WalkResult::advance();
    FailureOr<std::string> source =
        target::emission::renderPythonCombinerProjection(
            *operation, "@triton.jit", "tl.where");
    if (failed(source))
      return WalkResult::interrupt();
    output << *source;
    return WalkResult::advance();
  });
  if (projected.wasInterrupted())
    return failure();
  return success();
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  if (!planIndex.stages.empty()) {
    if (!searchIndex.autotune)
      return realization.emitOpError(
          "cannot resolve the delegated staged tuner");
    ABIView *terminalDestination = nullptr;
    bool restoreDestination = false;
    auto bindTerminalDestination = [&](Operation *terminal,
                                       bool restore) -> LogicalResult {
      FailureOr<ABIView *> destination =
          terminal && terminal->getNumOperands() > 0
              ? lookupView(terminal->getOperand(0), *terminal)
              : FailureOr<ABIView *>(failure());
      if (failed(destination) || terminalDestination)
        return realization.emitOpError(
            "requires exactly one staged terminal destination");
      terminalDestination = *destination;
      restoreDestination = restore;
      return success();
    };
    llvm::DenseSet<int64_t> terminalNodes;
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t node : stage.getTerminals())
        terminalNodes.insert(node);
    for (int64_t node : terminalNodes) {
      Operation *terminal = kernel.nodes.lookup(node);
      StringRef name = terminal ? terminal->getName().getStringRef() : StringRef();
      if ((name == "intent.scatter_reduce" || name == "intent.scatter_unique") &&
          failed(bindTerminalDestination(terminal,
                                         name == "intent.scatter_reduce")))
        return failure();
    }

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
      source << "@triton.autotune(\n    configs=_CONFIGS,\n    key=[";
      for (auto [index, attribute] :
           llvm::enumerate(searchIndex.autotune.getKey())) {
        if (index)
          source << ", ";
        source << "'" << cast<StringAttr>(attribute).getValue() << "'";
      }
      source << "]";
      if (planIndex.stages[stage].getOutputs().empty() && restoreDestination)
        source << ",\n    restore_value=['" << terminalDestination->pointer
               << "']";
      source << ",\n)\n@triton.jit\ndef " << kernelName << "_stage_"
             << stage << "(";
      bool first = true;
      auto parameter = [&](StringRef value) {
        if (!first)
          source << ", ";
        source << value;
        first = false;
      };
      for (ABIView &view : views)
        parameter(view.pointer);
      for (ABIScalar &scalar : scalars)
        parameter(scalar.name);
      plan::AxisOp outerAxis = planIndex.axes.lookup(ragged.binding.getOuterNode());
      std::string experts =
          outerAxis ? roleDimensions.lookup(
                          "program_" + std::to_string(outerAxis.getProgramOrder()))
                    : std::string();
      if (experts.empty())
        return ragged.binding.emitOpError(
            "has no program-owned outer-axis dimension");
      for (const std::string &dimension : dimensionOrder)
        parameter(dimension +
                  ((compact && dimension == experts) ||
                           planIndex.blockExtents.count(dimension)
                       ? ": tl.constexpr"
                       : ""));
      for (ABIView &view : views)
        for (const std::string &stride : view.strides)
          parameter(stride);
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          parameter(workspaceNames.lookup(kernel.values.lookup(valueID)));
      if (!compact)
        parameter("MAX_ROUTES");
      for (NamedAttribute config : searchIndex.autotune.getParameterMap())
        parameter(config.getName().getValue().str() + ": tl.constexpr");
      source << "):\n";
      source.flush();

      std::string feature = stageFeatureDimensions.lookup(stage);
      std::string featureTile = stageFeatureTiles.lookup(stage);
      std::string memberTile = stageMemberTiles.lookup(stage);
      stageLine(stage, "pid_feature = " +
                           addressIndex("tl.program_id(axis=" +
                                        std::to_string(
                                            stageFeatureWorkers.lookup(stage)) +
                                        ")"));
      stageLine(stage, "pid_expert_route = " +
                           addressIndex("tl.program_id(axis=" +
                                        std::to_string(
                                            stageMemberWorkers.lookup(stage)) +
                                        ")"));
      if (compact) {
        stageLine(stage, "expert = " + addressIndex("0"));
        stageLine(stage, "route_tile = " + addressIndex("0"));
        stageLine(stage, "tile_cursor = " + addressIndex("0"));
        stageLine(stage, "for candidate in range(0, " + experts + "):");
        stageLine(stage, "candidate_begin = tl.load(" + offsets->pointer +
                             " + " + addressIndex("candidate") + " * " +
                             addressIndex(offsets->strides[0]) + ")",
                  2);
        stageLine(stage, "candidate_end = tl.load(" + offsets->pointer +
                             " + " + addressIndex("candidate + 1") + " * " +
                             addressIndex(offsets->strides[0]) + ")",
                  2);
        stageLine(stage,
                  "candidate_tiles = " +
                      addressIndex("tl.cdiv(candidate_end - candidate_begin, " +
                                   memberTile + ")"),
                  2);
        stageLine(stage,
                  "owns_tile = (pid_expert_route >= tile_cursor) & "
                  "(pid_expert_route < tile_cursor + candidate_tiles)",
                  2);
        stageLine(stage,
                  "expert = tl.where(owns_tile, " + addressIndex("candidate") +
                      ", expert)",
                  2);
        stageLine(stage,
                  "route_tile = tl.where(owns_tile, pid_expert_route - "
                  "tile_cursor, route_tile)",
                  2);
        stageLine(stage, "tile_cursor += candidate_tiles", 2);
      } else {
        stageLine(stage,
                  "num_route_tiles = tl.cdiv(MAX_ROUTES, " + memberTile + ")");
        stageLine(stage, "expert = pid_expert_route // num_route_tiles");
        stageLine(stage, "route_tile = pid_expert_route % num_route_tiles");
      }
      stageLine(stage, "route_begin = tl.load(" + offsets->pointer + " + " +
                           addressIndex("expert") + " * " +
                           addressIndex(offsets->strides[0]) + ")");
      stageLine(stage, "route_end = tl.load(" + offsets->pointer +
                           " + " + addressIndex("expert + 1") + " * " +
                           addressIndex(offsets->strides[0]) + ")");
      stageLine(stage,
                "member_offsets = " + addressIndex("route_begin") + " + " +
                    addressIndex("route_tile") + " * " + memberTile + " + " +
                    addressIndex("tl.arange(0, " + memberTile + ")"));
      stageLine(stage, "member_mask = member_offsets < route_end");
      if (indices)
        stageLine(stage, "routes = tl.load(" + indices->pointer + " + " +
                             addressIndex("member_offsets") + " * " +
                             addressIndex(indices->strides[0]) +
                             ", mask=member_mask, other=0)");
      else
        stageLine(stage, "routes = member_offsets");
      stageLine(stage,
                "offs_feature = " + addressIndex("pid_feature") +
                    " * " + featureTile + " + " +
                    addressIndex("tl.arange(0, " + featureTile + ")"));
      stageLine(stage,
                "feature_mask = offs_feature < " + feature);
    }
    return success();
  }
  if (searchSpace) {
    output << "@triton.autotune(\n    configs=_CONFIGS,\n    key=[";
    for (auto [index, attribute] : llvm::enumerate(searchIndex.autotune.getKey())) {
      if (index)
        output << ", ";
      output << "'" << cast<StringAttr>(attribute).getValue() << "'";
    }
    output << "]";
    bool firstRestoredView = true;
    for (ABIView &view : views) {
      if (view.view.getAccess() != "inout")
        continue;
      output << (firstRestoredView ? ",\n    restore_value=[" : ", ")
             << "'" << view.pointer << "'";
      firstRestoredView = false;
    }
    if (!firstRestoredView)
      output << "]";
    output << ",\n)\n";
  } else if (!planIndex.components.reusedAxes.empty() || configureRowVector) {
    output << "@triton.autotune(\n    configs=_ROW_CONFIGS,\n    key=[";
    if (!planIndex.components.reusedAxes.empty()) {
      output << "'n_rows', 'n_cols'";
    } else {
      for (auto [index, dimension] : llvm::enumerate(dimensionOrder)) {
        if (index)
          output << ", ";
        output << "'" << dimension << "'";
      }
    }
    output << "],\n)\n";
  }
  output << "@triton.jit\n";
  if (!planIndex.components.reusedAxes.empty()) {
    programIndex = makeRegionArgumentName(*programRoot, 0);
    vectorIndex = makeResultName(*vectorDomain, 0);
    valueNames[programRoot->getRegion(0).front().getArgument(0)] = programIndex;
    valueNames[vectorDomain->getResult(0)] = vectorIndex;
    plan::AxisOp lane = planIndex.axesByRole.lookup("lane_0");
    if (lane)
      axisIndices[lane.getNode()] = vectorIndex;
    output << "def " << kernelName << "(";
    bool first = true;
    auto emitParameter = [&](StringRef parameter) {
      if (!first)
        output << ", ";
      output << parameter;
      first = false;
    };
    for (ABIView &view : views)
      emitParameter(view.pointer);
    for (ABIScalar &scalar : scalars)
      emitParameter(scalar.name);
    for (const std::string &dimension : dimensionOrder)
      if (dimension != roleDimensions.lookup("program_0") &&
          dimension != roleDimensions.lookup("lane_0"))
        emitParameter(dimension + ": tl.constexpr");
    for (ABIView &view : views)
      emitParameter(view.strides[0]);
    for (StringRef parameter :
         {"n_rows", "n_cols", "BLOCK_SIZE: tl.constexpr",
          "ROW_OCCUPANCY: tl.constexpr", "PIPELINE_STAGES: tl.constexpr"})
      emitParameter(parameter);
    output << "):\n";
    return success();
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
    emitParameter(view.pointer);
  for (ABIScalar &scalar : scalars)
    emitParameter(scalar.name);
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
  for (const std::string &dimension : dimensionOrder) {
    bool physicalDimension = llvm::any_of(
        roleDimensions,
        [&](const auto &binding) { return binding.getValue() == dimension; });
    bool roundedDimension = planIndex.blockExtents.count(dimension) != 0;
    emitParameter(dimension + (physicalDimension && !roundedDimension
                                   ? ""
                                   : ": tl.constexpr"));
  }
  for (ABIView &view : views)
    for (const std::string &stride : view.strides)
      emitParameter(stride);
  if (searchIndex.autotune)
    for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
      emitParameter(parameter.getName().getValue().str() + ": tl.constexpr");
  if (usesScaledContraction())
    emitParameter("USE_NATIVE_SCALED: tl.constexpr");
  output << "):\n";
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
  auto emitViewCapabilityCheck = [&](const ABIView &view) {
    output << "    if any(extent == 0 for extent in "
           << view.argument->name << ".shape):\n";
    output << "        raise NotImplementedError('Triton does not support "
              "zero-extent external views in this compiler')\n";
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
            "has no supported Triton private-workspace allocation");
      output << "    " << workspaceNames.lookup(buffer->getResult(0))
             << " = torch.full((" << *size
             << ",), " << *initializer << ", device=_DEVICE, dtype=" << dtype
             << ")\n";
    }
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "private_workspace")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      FailureOr<std::string> size = target::emission::scanWorkspaceElementCount(
          entry.second, scanExtents.lookup(entry.first), planIndex,
          axisDimensions);
      if (!scan || scan->getNumResults() == 0 || failed(size))
        return entry.second.emitOpError(
            "has no supported Triton scan-workspace allocation");
      for (Value result : scan->getResults()) {
        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        StringRef dtype = resultType ? torchDtype(resultType.getElementType())
                                     : StringRef();
        if (dtype.empty())
          return entry.second.emitOpError(
              "has an unsupported Triton scan-workspace component dtype");
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
  if (!planIndex.stages.empty()) {
    for (const std::string &body : stageBodies)
      output << body << "\n";
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

    output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
           << ")\n\n\n";
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
        StringRef dtype = tensor ? torchDtype(tensor.getElementType()) : StringRef();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported workspace tensor");
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
      output << "    grid_stage_" << stage
             << " = lambda META: (triton.cdiv(" << feature
             << ", META['" << featureTile << "']), ";
      if (compact)
        output << "sum(triton.cdiv(length, META['" << memberTile
               << "']) for length "
                  "in route_lengths_"
               << suffix << ")";
      else
        output << experts
               << " * triton.cdiv(max_routes_" << suffix
               << ", META['" << memberTile << "'])";
      output << ", 1)\n";
      output << "    compiled_stage_" << stage << " = " << kernelName
             << "_stage_" << stage << "[grid_stage_" << stage << "](";
      bool first = true;
      auto argument = [&](StringRef value) {
        if (!first)
          output << ", ";
        output << value;
        first = false;
      };
      for (ABIView &view : views)
        argument(view.argument->name);
      for (ABIScalar &scalar : scalars)
        argument(scalar.name);
      for (const std::string &dimension : dimensionOrder)
        argument(dimension);
      for (ABIView &view : views)
        for (int64_t axis = 0; axis < view.tensor.getRank(); ++axis)
          argument(view.argument->name + ".stride(" + std::to_string(axis) + ")");
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          argument(workspaceNames.lookup(kernel.values.lookup(valueID)));
      if (!compact)
        argument("max_routes_" + suffix);
      output << ")\n";
    }
    output << "    return compiled_stage_" << planIndex.stages.size() - 1
           << "\n\n\n";
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
    StringRef mergeDtype = torchDtype(merge->tensor.getElementType());
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
    output << "), device=_DEVICE, dtype=" << mergeDtype << ")\n";
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
            "fixed grid-stride wrapper supports input and output views");
    }
    if (inputs.empty())
      return kernel.entry.emitOpError(
          "fixed grid-stride wrapper requires an input view");
    bool hasInOut = llvm::any_of(views, [](const ABIView &view) {
      return view.view.getAccess() == "inout";
    });
    if (outputs.empty() && !hasInOut)
      return kernel.entry.emitOpError(
          "fixed grid-stride wrapper requires a writable view");
    plan::AxisOp laneAxis = planIndex.axesByRole.lookup("lane_0");
    if (!laneAxis)
      return realization.emitOpError(
          "fixed grid-stride wrapper has no selected lane range");
    FailureOr<std::string> rowTile = physicalAxisTile(laneAxis);
    if (failed(rowTile))
      return realization.emitOpError(
          "fixed grid-stride wrapper has no selected lane range");
    output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
           << ")\n";
    output << "_NUM_SMS = torch.cuda.get_device_properties(_DEVICE).multi_processor_count\n\n\n";
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
    for (ABIView &view : views) {
      StringRef dtype = torchDtype(view.tensor.getElementType());
      if (dtype.empty())
        return kernel.entry.emitOpError("has an unsupported fixed ABI dtype");
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
      emitViewCapabilityCheck(view);
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
      output << "    if not " << view.argument->name << ".is_contiguous():\n";
      output << "        raise ValueError('grid-stride views must be contiguous')\n";
    }
    SmallVector<ABIView *> noaliasViews;
    for (ABIView &view : views) {
      auto constraints =
          view.argument->metadata.getAs<DictionaryAttr>("constraints");
      auto noalias = constraints ? constraints.getAs<BoolAttr>("noalias")
                                 : BoolAttr();
      if (!noalias || !noalias.getValue())
        continue;
      noaliasViews.push_back(&view);
      output << "    " << view.argument->name << "_start = "
             << view.argument->name << ".data_ptr()\n";
      output << "    " << view.argument->name << "_end = "
             << view.argument->name << "_start + " << view.argument->name
             << ".numel() * " << view.argument->name << ".element_size()\n";
    }
    for (size_t lhs = 0; lhs < noaliasViews.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < noaliasViews.size(); ++rhs) {
        output << "    if max(" << noaliasViews[lhs]->argument->name
               << "_start, " << noaliasViews[rhs]->argument->name
               << "_start) < min(" << noaliasViews[lhs]->argument->name
               << "_end, " << noaliasViews[rhs]->argument->name << "_end):\n";
        output << "        raise ValueError('realized views violate noalias')\n";
      }
    std::string rowDimension = roleDimensions.lookup("program_0");
    std::string columnDimension = roleDimensions.lookup("lane_0");
    std::string rowOwner = dimensionOwners.lookup(rowDimension);
    std::string columnOwner = dimensionOwners.lookup(columnDimension);
    output << "    n_rows = " << (rowOwner.empty() ? rowDimension : rowOwner)
           << "\n";
    output << "    n_cols = "
           << (columnOwner.empty() ? columnDimension : columnOwner) << "\n";
    output << "    grid = lambda META: (min(_NUM_SMS * META['ROW_OCCUPANCY'], n_rows), 1, 1)\n";
    auto emitRowLaunchArguments = [&](bool cloneInOut) {
      bool first = true;
      auto emitArgument = [&](StringRef argument) {
        if (!first)
          output << ", ";
        output << argument;
        first = false;
      };
      for (ABIView &view : views) {
        std::string argument = view.argument->name;
        if (cloneInOut && view.view.getAccess() == "inout")
          argument += ".clone()";
        emitArgument(argument);
      }
      for (ABIScalar &scalar : scalars)
        emitArgument(scalar.name);
      for (const std::string &dimension : dimensionOrder)
        if (dimension != roleDimensions.lookup("program_0") &&
            dimension != roleDimensions.lookup("lane_0"))
          emitArgument(dimension);
      for (ABIView &view : views)
        emitArgument(view.argument->name + ".stride(0)");
      emitArgument("n_rows");
      emitArgument("n_cols");
      output << ", BLOCK_SIZE=" << *rowTile;
    };
    if (hasInOut) {
      output << "    row_tuning_key = (n_rows, n_cols";
      for (ABIView &view : views)
        output << ", " << view.argument->name << ".dtype";
      output << ", str(_DEVICE))\n";
      output << "    if row_tuning_key not in _ROW_TUNED_KEYS:\n";
      output << "        " << kernelName << "[grid](";
      emitRowLaunchArguments(true);
      output << ")\n";
      output << "        _ROW_TUNED_KEYS.add(row_tuning_key)\n";
    }
    output << "    return " << kernelName << "[grid](";
    emitRowLaunchArguments(false);
    output << ")\n\n\n";
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
        return kernel.entry.emitOpError("has an unsupported fixed output dtype");
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
    bool first = true;
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

  SmallVector<ABIView *> inputs;
  SmallVector<ABIView *> outputs;
  for (ABIView &view : views) {
    if (view.view.getAccess() == "in" || view.view.getAccess() == "inout")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out")
      outputs.push_back(&view);
    else
      return kernel.entry.emitOpError(
          "autotuned wrapper supports input and output views");
  }
  bool hasInOut = llvm::any_of(views, [](const ABIView &view) {
    return view.view.getAccess() == "inout";
  });
  if (outputs.empty() && !hasInOut)
    return kernel.entry.emitOpError("autotuned wrapper has no writable views");

  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n\n\n";
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
    output << "    if " << view.argument->name << ".device != _DEVICE:\n";
    output << "        raise ValueError('" << view.argument->name
           << " must reside on the realized CUDA device')\n";
    output << "    if " << view.argument->name << ".ndim != "
           << view.tensor.getRank() << ":\n";
    output << "        raise ValueError('" << view.argument->name
           << " has the wrong rank')\n";
    StringRef dtype = torchDtype(view.tensor.getElementType());
    if (dtype.empty())
      return kernel.entry.emitOpError("has an unsupported Triton ABI dtype");
    output << "    if " << view.argument->name << ".dtype != " << dtype
           << ":\n";
    output << "        raise ValueError('" << view.argument->name
           << " has the wrong dtype')\n";
    emitViewCapabilityCheck(view);
  }
  for (const std::string &dimension : dimensionOrder)
    output << "    " << dimension << " = " << dimensionOwners.lookup(dimension)
           << "\n";
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
  SmallVector<plan::AxisOp> programAxes =
      target::emission::orderedProgramAxes(planIndex);
  for (plan::AxisOp axis : programAxes) {
    if (!planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode()))
      continue;
    FailureOr<plan::RaggedOp> relation =
        target::emission::uniqueRaggedRelation(
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
  }
  std::array<std::string, 3> grid = target::emission::projectProgramGrid(
      planIndex, [&](plan::AxisOp axis) {
        std::string role = "program_" + std::to_string(axis.getProgramOrder());
        std::string extent =
            planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode())
                ? "max_member_length_" + std::to_string(axis.getNode())
                : roleDimensions.lookup(role);
        return axis.isScalar()
                   ? extent
                   : "triton.cdiv(" + extent + ", META['" +
                         axis.getTile().str() + "'])";
      });
  if (planIndex.program.getPersistent()) {
    std::string total = target::emission::projectProgramVolume(
        planIndex, [&](plan::AxisOp axis) {
          std::string role =
              "program_" + std::to_string(axis.getProgramOrder());
          std::string extent = roleDimensions.lookup(role);
          return axis.isScalar()
                     ? extent
                     : "triton.cdiv(" + extent + ", META['" +
                           axis.getTile().str() + "'])";
        });
    output << "    grid = lambda META: (min(torch.cuda.get_device_properties(_DEVICE).multi_processor_count, "
           << total << "), 1, 1)\n";
  } else {
    output << "    grid = lambda META: (" << grid[0] << ", " << grid[1]
           << ", " << grid[2] << ")\n";
  }
  auto emitKernelLaunchArguments = [&](bool cloneInOut) {
    bool first = true;
    auto emitArgument = [&](StringRef argument) {
      if (!first)
        output << ", ";
      output << argument;
      first = false;
    };
    for (ABIView &view : views) {
      std::string argument = view.argument->name;
      if (cloneInOut && view.view.getAccess() == "inout")
        argument += ".clone()";
      emitArgument(argument);
    }
    for (ABIScalar &scalar : scalars)
      emitArgument(scalar.name);
    for (Operation *buffer : privateWorkspaceBuffers)
      emitArgument(workspaceNames.lookup(buffer->getResult(0)));
    for (const auto &entry : planIndex.scans) {
      if (entry.second.getResultSpace() != "private_workspace")
        continue;
      Operation *scan = kernel.nodes.lookup(entry.first);
      for (Value result : scan->getResults())
        emitArgument(workspaceNames.lookup(result));
      for (int64_t valueID : entry.second.getMaterializedValues())
        emitArgument(workspaceNames.lookup(kernel.values.lookup(valueID)));
    }
    for (const std::string &dimension : dimensionOrder)
      emitArgument(dimension);
    for (ABIView &view : views)
      for (int64_t axis = 0; axis < view.tensor.getRank(); ++axis)
        emitArgument(view.argument->name + ".stride(" + std::to_string(axis) + ")");
  };
  if (configureRowVector && hasInOut) {
    output << "    row_tuning_key = (";
    bool firstRowKey = true;
    for (const std::string &dimension : dimensionOrder) {
      if (!firstRowKey)
        output << ", ";
      output << dimension;
      firstRowKey = false;
    }
    for (ABIView &view : views) {
      output << (firstRowKey ? "" : ", ") << view.argument->name << ".dtype";
      firstRowKey = false;
    }
    output << (firstRowKey ? "" : ", ") << "str(_DEVICE))\n";
    output << "    if row_tuning_key not in _ROW_TUNED_KEYS:\n";
    output << "        " << kernelName << "[grid](";
    emitKernelLaunchArguments(true);
    output << ")\n";
    output << "        _ROW_TUNED_KEYS.add(row_tuning_key)\n";
  }
  output << "    return " << kernelName << "[grid](";
  emitKernelLaunchArguments(false);
  output << ")\n\n\n";
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
      return kernel.entry.emitOpError("has an unsupported Triton output dtype");
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
    consumer.emitOpError("references a missing operand during Triton emission");
    return failure();
  }
  auto found = valueNames.find(consumer.getOperand(operandIndex));
  if (found == valueNames.end()) {
    consumer.emitOpError() << "operand " << operandIndex
                           << " has no emitted target value";
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
        "cannot resolve multi-axis scalar ownership during Triton emission");
    return failure();
  }
  if (scalarSource->opaque) {
    consumer.emitOpError(
        "cannot resolve an opaque index source during Triton emission");
    return failure();
  }
  if (Operation *definition = indexedValue.getDefiningOp()) {
    if (definition->getName().getStringRef() == "intent.domain" ||
        definition->getName().getStringRef() == "intent.ragged_outer" ||
        definition->getName().getStringRef() == "intent.ragged_member")
      return definition;
  }
  consumer.emitOpError("cannot resolve index ownership during Triton emission");
  return failure();
}

FailureOr<plan::AxisOp> SourceEmitter::resolveAxis(Value indexedValue,
                                                  Operation &consumer) {
  FailureOr<Operation *> domain = resolveDomain(indexedValue, consumer);
  if (failed(domain))
    return failure();
  FailureOr<int64_t> node = target::getNodeID(**domain, "axis lookup");
  plan::AxisOp axis = succeeded(node) ? planIndex.axes.lookup(*node) : plan::AxisOp();
  if (failed(node) || !axis) {
    consumer.emitOpError("indexes a domain without a physical axis binding");
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

FailureOr<std::string>
SourceEmitter::indexExpression(plan::AxisOp axis, bool store,
                               unsigned tensorAxis, unsigned tensorRank,
                               Operation &consumer) {
  std::string base = axisIndices.lookup(axis.getNode());
  if (base.empty()) {
    consumer.emitOpError()
        << "has no active physical index for logical axis " << axis.getNode();
    return failure();
  }
  if (axis.isScalar())
    return base;
  if (tensorAxis >= tensorRank) {
    consumer.emitOpError("physical vector axis exceeds the emitted tensor rank");
    return failure();
  }
  return broadcastIndex(base, tensorAxis, tensorRank);
}

std::string SourceEmitter::addressIndex(StringRef expression) const {
  return "tl.cast((" + expression.str() + "), tl.int64)";
}

std::string SourceEmitter::physicalExtent(StringRef logicalExtent) const {
  auto extent = planIndex.blockExtents.find(logicalExtent);
  if (extent == planIndex.blockExtents.end())
    return logicalExtent.str();
  return "triton.next_power_of_2(" + logicalExtent.str() + ")";
}

bool SourceEmitter::usesScaledContraction() const {
  return llvm::any_of(planIndex.contracts, [](const auto &entry) {
    return entry.second.getLowering() == "tl.dot_scaled";
  });
}

FailureOr<std::string> SourceEmitter::physicalAxisTile(plan::AxisOp axis) {
  if (axis.getReuseWorker() || !axis.getTileRole().starts_with("row_vector"))
    return axis.getTile().str();
  const target::emission::RangeBinding *range = axis.roleRange();
  if (!range || !planIndex.blockExtents.count(range->getExtent()))
    return axis.emitOpError("cannot resolve its row-vector physical extent");
  return physicalExtent(range->getExtent());
}

FailureOr<std::string>
SourceEmitter::transferPhysicalExtentFill(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(relation) || failed(view))
    return failure();
  return target::emission::transferPhysicalExtentFill(
      planIndex, *relation, (*view)->shape, operation);
}

FailureOr<std::string>
SourceEmitter::emitPointerExpression(Operation &operation, ABIView &view,
                                     bool store) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation) ||
      static_cast<size_t>(llvm::count_if(
          *relation, [](const target::IndexTerm &term) {
            return term.kind != "new_axis";
          })) != view.strides.size())
    return failure();
  FailureOr<unsigned> tensorRank = emittedTensorRank(operation, store);
  if (failed(tensorRank))
    return failure();
  unsigned tensorIndexCount = llvm::count_if(*relation, [&](const target::IndexTerm &term) {
    return term.kind == "value_index" && term.operands.size() == 1 &&
           term.operands.front() &&
           isa<RankedTensorType>(
               operation.getOperand(*term.operands.front()).getType());
  });
  std::string expression = view.pointer;
  unsigned vectorAxis = 0;
  unsigned sourceAxis = 0;
  bool advancedTensorAxesCovered = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      ++vectorAxis;
      continue;
    }
    unsigned axisNumber = sourceAxis++;
    std::string index;
    if (term.kind == "full_slice") {
      index = broadcastIndex(
          "tl.arange(0, " + physicalExtent(view.shape[axisNumber]) + ")",
          vectorAxis++, *tensorRank);
    } else if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError(
            "static pointer index has no canonical value");
      index = std::to_string(*term.staticValues.front());
    } else {
      if (term.kind != "region_index" && term.kind != "value_index")
        return operation.emitOpError(
            "has no mechanical Triton pointer relation");
      if (term.operands.size() != 1 || !term.operands.front())
        return operation.emitOpError(
            "dynamic pointer index has no canonical operand");
      Value indexed = operation.getOperand(*term.operands.front());
      if (term.kind == "value_index") {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        auto tensor = dyn_cast<RankedTensorType>(indexed.getType());
        if (!tensor) {
          index = "(" + exact->str() + ")";
        } else {
          if (tensorIndexCount > 1 || tensor.getRank() > 1) {
            if (tensor.getRank() > static_cast<int64_t>(*tensorRank) ||
                (vectorAxis != 0 && !advancedTensorAxesCovered))
              return operation.emitOpError(
                  "Triton broadcasted tensor index exceeds the emitted tensor rank");
            index = "(" + exact->str() + ")";
            if (tensor.getRank() < static_cast<int64_t>(*tensorRank)) {
              index += "[";
              for (unsigned axis = 0; axis < *tensorRank; ++axis) {
                if (axis)
                  index += ", ";
                index += axis < static_cast<unsigned>(tensor.getRank()) ? ":"
                                                                        : "None";
              }
              index += "]";
            }
            if (!advancedTensorAxesCovered) {
              vectorAxis = tensor.getRank();
              advancedTensorAxesCovered = true;
            }
          } else {
            if (tensor.getRank() != 1 || vectorAxis >= *tensorRank)
              return operation.emitOpError(
                  "Triton indirect tensor indices currently require one logical axis");
            index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
          }
        }
      } else if (target::emission::isSequentialIterator(indexed)) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        index = "(" + exact->str() + ")";
      } else {
        FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
        if (failed(axis))
          return failure();
        bool vector = !axis->isScalar();
        FailureOr<std::string> resolved =
            indexExpression(*axis, store, vectorAxis, *tensorRank, operation);
        if (failed(resolved))
          return failure();
        index = *resolved;
        if (vector)
          ++vectorAxis;
      }
    }
    std::string stride = view.strides[axisNumber];
    if (!planIndex.components.reusedAxes.empty() && axisNumber > 0) {
      stride = addressIndex("1");
      for (unsigned trailing = axisNumber + 1; trailing < view.shape.size();
           ++trailing)
        stride += " * " + addressIndex(view.shape[trailing]);
    }
    expression += " + " + addressIndex(index) + " * " + addressIndex(stride);
  }
  if (vectorAxis != *tensorRank)
    return operation.emitOpError(
        "pointer relation does not cover every emitted tensor axis");
  return expression;
}

FailureOr<std::string>
SourceEmitter::emitMaskExpression(Operation &operation, bool store) {
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
  FailureOr<unsigned> tensorRank = emittedTensorRank(operation, store);
  if (failed(tensorRank))
    return failure();
  unsigned tensorIndexCount = llvm::count_if(*relation, [&](const target::IndexTerm &term) {
    return term.kind == "value_index" && term.operands.size() == 1 &&
           term.operands.front() &&
           isa<RankedTensorType>(
               operation.getOperand(*term.operands.front()).getType());
  });
  SmallVector<std::string> predicates;
  unsigned vectorAxis = 0;
  unsigned sourceAxis = 0;
  bool advancedTensorAxesCovered = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      ++vectorAxis;
      continue;
    }
    unsigned axisNumber = sourceAxis++;
    if (term.kind == "full_slice") {
      if (planIndex.blockExtents.count((*view)->shape[axisNumber])) {
        std::string index = broadcastIndex(
            "tl.arange(0, " + physicalExtent((*view)->shape[axisNumber]) + ")",
            vectorAxis, *tensorRank);
        predicates.push_back("(" + index + " < " +
                             (*view)->shape[axisNumber] + ")");
      }
      ++vectorAxis;
      continue;
    }
    if (term.kind == "static_index")
      continue;
    if (term.operands.size() != 1 || !term.operands.front()) {
      operation.emitOpError(
          "Triton mask emission requires value-bound index terms");
      return failure();
    }
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      std::string index;
      if (tensorIndexCount > 1 || tensor.getRank() > 1) {
        if (tensor.getRank() > static_cast<int64_t>(*tensorRank) ||
            (vectorAxis != 0 && !advancedTensorAxesCovered))
          return operation.emitOpError(
              "Triton broadcasted tensor bounds exceed the emitted tensor rank");
        index = "(" + exact->str() + ")";
        if (tensor.getRank() < static_cast<int64_t>(*tensorRank)) {
          index += "[";
          for (unsigned axis = 0; axis < *tensorRank; ++axis) {
            if (axis)
              index += ", ";
            index += axis < static_cast<unsigned>(tensor.getRank()) ? ":"
                                                                    : "None";
          }
          index += "]";
        }
        if (!advancedTensorAxesCovered) {
          vectorAxis = tensor.getRank();
          advancedTensorAxesCovered = true;
        }
      } else {
        if (tensor.getRank() != 1 || vectorAxis >= *tensorRank)
          return operation.emitOpError(
              "Triton indirect tensor bounds require one logical axis");
        index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
      }
      std::string extent = (*view)->shape[axisNumber];
      if (!planIndex.components.reusedAxes.empty()) {
        if (roleDimensions.lookup("program_0") == extent)
          extent = "n_rows";
        else if (roleDimensions.lookup("lane_0") == extent)
          extent = "n_cols";
      }
      predicates.push_back("(" + index + " >= 0)");
      predicates.push_back("(" + index + " < " + extent + ")");
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
      if (source->hasDomain() && (source->transformed || packedScalar)) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        std::string index = "(" + exact->str() + ")";
        std::string extent = (*view)->shape[axisNumber];
        if (!planIndex.components.reusedAxes.empty()) {
          if (roleDimensions.lookup("program_0") == extent)
            extent = "n_rows";
          else if (roleDimensions.lookup("lane_0") == extent)
            extent = "n_cols";
        }
        predicates.push_back("(" + index + " >= 0)");
        predicates.push_back("(" + index + " < " + extent + ")");
      }
      continue;
    }
    if (term.kind == "region_index" &&
        target::emission::isSequentialIterator(indexed))
      continue;
    FailureOr<Operation *> domain =
        resolveDomain(indexed, operation);
    FailureOr<plan::AxisOp> axis =
        resolveAxis(indexed, operation);
    if (failed(domain) || failed(axis))
      return failure();
    bool vector = !axis->isScalar();
    if (!vector)
      continue;
    bool gridStride = !planIndex.components.reusedAxes.empty();
    FailureOr<std::string> extent =
        gridStride && axis->getReuseWorker()
            ? FailureOr<std::string>(std::string("n_rows"))
        : gridStride && *domain == vectorDomain
            ? FailureOr<std::string>(std::string("n_cols"))
            : dimensionName(**domain);
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> ordered = target::emission::representativeOrderedAxis(
          planIndex, axis->getNode(), operation);
      if (failed(ordered))
        return failure();
      extent = "sequence_end_" + std::to_string(*ordered);
    }
    if (failed(extent))
      return failure();
    FailureOr<std::string> index =
        indexExpression(*axis, store, vectorAxis, *tensorRank, operation);
    if (failed(index))
      return failure();
    ++vectorAxis;
    predicates.push_back("(" + *index + " < " + *extent + ")");
  }
  if (predicates.empty())
    return std::string("True");
  std::string combined = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    combined += " & " + predicate.str();
  return combined;
}

FailureOr<std::string> SourceEmitter::emitValidityExpression(
    ArrayRef<int64_t> tensorAxes, ArrayRef<int64_t> domainNodes, Value value,
    Operation &consumer) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || tensorAxes.size() != domainNodes.size())
    return consumer.emitOpError(
        "has no ranked Triton value-validity binding");
  SmallVector<std::string> predicates;
  for (auto [tensorAxis, domainNode] : llvm::zip(tensorAxes, domainNodes)) {
    auto domain = kernel.nodes.find(domainNode);
    auto physical = planIndex.axes.find(domainNode);
    if (tensorAxis < 0 || tensorAxis >= tensor.getRank() ||
        domain == kernel.nodes.end() || physical == planIndex.axes.end())
      return consumer.emitOpError(
          "references an unresolved Triton validity axis");
    plan::AxisOp axis = physical->second;
    if (axis.isScalar())
      continue;
    FailureOr<std::string> extent =
        !planIndex.components.reusedAxes.empty() && axis.getReuseWorker()
            ? FailureOr<std::string>(std::string("n_rows"))
        : !planIndex.components.reusedAxes.empty() && domain->second == vectorDomain
            ? FailureOr<std::string>(std::string("n_cols"))
            : dimensionName(*domain->second);
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> ordered = target::emission::representativeOrderedAxis(
          planIndex, axis.getNode(), consumer);
      if (failed(ordered))
        return failure();
      extent = "sequence_end_" + std::to_string(*ordered);
    }
    FailureOr<std::string> index = indexExpression(
        axis, false, tensorAxis, tensor.getRank(), consumer);
    if (failed(extent) || failed(index))
      return failure();
    predicates.push_back("(" + *index + " < " + *extent + ")");
  }
  if (predicates.empty())
    return std::string("True");
  std::string result = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    result += " & " + predicate.str();
  return result;
}

FailureOr<std::string> SourceEmitter::padExpression(
    Value value, StringRef expression, Operation &consumer) {
  FailureOr<int64_t> valueID =
      target::getValueID(value, kernel, consumer, "Triton padding lookup");
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
                         ? "-float('inf')"
                     : planned == "positive_infinity" ? "float('inf')"
                     : planned == "nan" ? "float('nan')"
                     : planned == "true" ? "True"
                     : planned == "false" ? "False"
                     : planned.starts_with("literal_integer:")
                         ? planned.drop_front(16).str()
                     : planned.starts_with("literal_float:")
                         ? planned.drop_front(14).str()
                     : isa<IntegerType, IndexType>(valueType) ? "0"
                                                              : "0.0";
  return "tl.where(" + *predicate + ", " + expression.str() + ", " +
         fill + ")";
}

FailureOr<unsigned> SourceEmitter::emittedTensorRank(Operation &operation,
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

FailureOr<std::string>
SourceEmitter::emitTensorShape(Operation &operation, unsigned resultIndex) {
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
          "tensor shape region has no physical tile binding");
    else
      extents.push_back(physicalExtent(label.getValue()));
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

std::string SourceEmitter::broadcastIndex(StringRef base, unsigned axis,
                                          unsigned rank) {
  if (rank <= 1)
    return base.str();
  std::string result = base.str() + "[";
  for (unsigned index = 0; index < rank; ++index) {
    if (index)
      result += ", ";
    result += index == axis ? ":" : "None";
  }
  return result + "]";
}

std::string SourceEmitter::uniqueName(StringRef candidate, int64_t node) {
  std::string result = candidate.str();
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

std::string SourceEmitter::programGrid(StringRef extent) {
  int64_t axis = planIndex.axesByRole.lookup("program_0").getWorkerAxis();
  if (axis == 0)
    return "(" + extent.str() + ", 1, 1)";
  if (axis == 1)
    return "(1, " + extent.str() + ", 1)";
  return "(1, 1, " + extent.str() + ")";
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
    target::KernelModel kernel,
    intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization, kernel);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return SourceEmitter(std::move(kernel), realization, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .emit();
}

} // namespace intent::triton::emission

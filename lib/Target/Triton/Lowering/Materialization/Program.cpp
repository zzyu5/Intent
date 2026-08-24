#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Lowering/Combiner.h"
#include "Intent/Target/Triton/Lowering/Passes.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::triton::lowering {
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
  FailureOr<std::string> role = target::lowering::pointwiseRole(operation);
  FailureOr<StringRef> spelling =
      succeeded(role) ? syntax::pointwise(&operation, *role)
                      : FailureOr<StringRef>(failure());
  if (failed(spelling))
    return failure();
  return target::lowering::renderPythonPointwiseExpression(
      operation, operands, *spelling, [&](Type type) -> FailureOr<std::string> {
        StringRef dtype = combinerDtype(type);
        return dtype.empty() ? FailureOr<std::string>(failure())
                             : FailureOr<std::string>(dtype.str());
      });
}

std::string launchTile(StringRef tile) {
  int64_t fixed = 0;
  if (!tile.getAsInteger(10, fixed))
    return tile.str();
  return "META['" + tile.str() + "']";
}

std::string projectTensorIndex(StringRef value, unsigned valueRank,
                               unsigned groupRank, unsigned groupAxis,
                               unsigned resultRank) {
  std::string expression = "(" + value.str() + ")";
  if (valueRank == resultRank && groupAxis == 0)
    return expression;
  unsigned valueAxis = groupAxis + groupRank - valueRank;
  expression += "[";
  for (unsigned axis = 0; axis < resultRank; ++axis) {
    if (axis)
      expression += ", ";
    expression += axis >= valueAxis && axis < groupAxis + groupRank ? ":"
                                                                      : "None";
  }
  return expression + "]";
}

bool isDescriptorShapeProjection(Operation &operation) {
  if (target::semanticOperationName(operation) != "intent.gather" ||
      operation.getNumOperands() < 2 || operation.getNumResults() != 1)
    return false;
  auto input = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  auto validIndex =
      operation.getAttrOfType<IntegerAttr>("intent.valid_operand_index");
  Operation *valid = validIndex && validIndex.getInt() >= 0 &&
                             static_cast<unsigned>(validIndex.getInt()) <
                                 operation.getNumOperands()
                         ? operation.getOperand(validIndex.getInt()).getDefiningOp()
                         : nullptr;
  auto validValue = valid && target::semanticOperationName(*valid) ==
                                 "intent.constant"
                        ? valid->getAttrOfType<BoolAttr>("intent.value")
                        : BoolAttr();
  if (!input || !result || failed(relation) || !validValue ||
      !validValue.getValue())
    return false;
  unsigned fullSlices = 0;
  unsigned newAxes = 0;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "full_slice")
      ++fullSlices;
    else if (term.kind == "new_axis")
      ++newAxes;
    else
      return false;
  }
  return fullSlices == static_cast<unsigned>(input.getRank()) &&
         result.getRank() == input.getRank() + static_cast<int64_t>(newAxes);
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
      index.regionBindings[value.getValue()] = value;
    } else if (auto value =
                   dyn_cast<intent::plan::DomainExtentBindingOp>(operation)) {
      index.domainExtentBindings[value.getAxisNode()] = value;
    } else if (auto value =
                   dyn_cast<intent::plan::PartitionBindingOp>(operation)) {
      index.partitionBindings.push_back(value);
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
            "has no realized Triton contraction provider form");
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
          "Triton has no native 2:4 sparse contraction projection");
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
  for (plan::RaggedOp &ragged : index.ragged) {
    auto route = ragged.operation
                     ? ragged.operation->getAttrOfType<StringAttr>(raggedRouteAttr)
                     : StringAttr();
    if (!route)
      return physicalProgram.emitOpError(
          "has no realized Triton ragged-route form");
    ragged.route = route.getValue().str();
  }
  target::lowering::indexAxisRoles(index);
  for (intent::plan::ReductionOp value : reductions) {
    auto lowering = value->getAttrOfType<StringAttr>(reductionLoweringAttr);
    auto axis = value->getAttrOfType<IntegerAttr>(reductionAxisAttr);
    bool allAxes = value->hasAttr(reductionAllAxesAttr);
    if (!lowering || (static_cast<bool>(axis) == allAxes))
      return value.emitOpError("has no realized Triton reduction spelling");
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = axis ? axis.getInt() : -1;
    binding.allAxes = allAxes;
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    auto lowering = value->getAttrOfType<StringAttr>(scanLoweringAttr);
    if (!lowering || !index.axes.count(value.getAxisNode()))
      return value.emitOpError("has no realized Triton scan spelling");
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
    if (!lowering)
      return value.emitOpError("has no realized Triton pointwise spelling");
    plan::PointwiseOp binding;
    binding.operation = value;
    binding.lowering = lowering.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    index.pointwise[value.getNode()] = binding;
  }
  for (auto &entry : index.streams) {
    plan::StreamOp &binding = entry.second;
    auto tile = binding.physical->getAttrOfType<StringAttr>(streamTileAttr);
    if (!tile)
      return binding.emitOpError("has no realized Triton stream-tile spelling");
    binding.tile = tile.getValue().str();
  }
  index.components = target::lowering::indexPhysicalComponents(index);
  for (intent::plan::TransferOp value : transfers) {
    auto access = value->getAttrOfType<StringAttr>(transferAccessAttr);
    auto form = value->getAttrOfType<StringAttr>(transferFormAttr);
    if (!access || !form)
      return value.emitOpError("has no realized Triton transfer spelling");
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = access.getValue().str();
    binding.resultSpace = value.getResultSpace().str();
    index.boundaries[value.getNode()] = binding;
    index.transferForms[value.getNode()] = form.getValue().str();
    if (auto axes =
            value->getAttrOfType<DenseI64ArrayAttr>(descriptorBlockAxesAttr))
      index.descriptorBlockAxes[value.getNode()] =
          SmallVector<int64_t>(axes.asArrayRef());
    if (auto layout = value->getAttrOfType<StringAttr>(descriptorLayoutAttr))
      index.descriptorLayouts[value.getNode()] = layout.getValue().str();
  }
  if (!index.target || !index.program) {
    physicalProgram.emitOpError("lacks device or launch decisions");
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
        "has no realized Triton row-launch form");
  configureRowVector = rowForm.getValue() == "configured";
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
        "workspace_" + std::to_string(node) + "_ptr";
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
                               std::to_string(component) + "_ptr";
      scanResults[result] = binding;
    }
    scanExtents[entry.first] = extent;
    for (int64_t valueID : binding.getMaterializedValues()) {
      FailureOr<Value> value = target::lowering::lookupScanMaterializedValue(
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
               << " has no Triton runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("Triton emitter requires ranked views");
    ABIView emitted{&argument, view, tensor, argument.name + "_ptr", {}, {}, {}};
    for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
      std::optional<int64_t> stride =
          target::lowering::staticViewStride(argument, axis);
      emitted.strides.push_back(
          stride ? std::to_string(*stride)
                 : argument.name + "_stride_" + std::to_string(axis));
      emitted.dynamicStrides.push_back(!stride.has_value());
    }
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
  }
  if (views.empty())
    return kernel.entry.emitOpError("Triton emitter requires external views");
  for (ABIView &view : views)
    valueNames[view.argument->value] = view.pointer;
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
  if (failed(target::lowering::indexRuntimeDomainExtents(
          planIndex, kernel, dimensionOwners, dimensionOrder)))
    return failure();
  if (failed(target::lowering::indexPartitionExtents(
          planIndex, kernel, axisDimensions, dimensionOwners, dimensionOrder,
          syntax::tile)))
    return failure();
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
    auto program = planIndex.axesByRole.find("program_0");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (program == planIndex.axesByRole.end() ||
        lane == planIndex.axesByRole.end())
      return physicalProgram.emitOpError(
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
  if (target::lowering::requiresDelegatedTuning(planIndex) &&
      (!searchSpace || !searchIndex.autotune))
    return physicalProgram.emitOpError(
        "tiled physical components require a delegated Triton tuner");
  for (const auto &entry : planIndex.transferForms) {
    if (entry.second != "pointer_or_descriptor")
      continue;
    Operation *operation = kernel.nodes.lookup(entry.first);
    auto position = operation && operation->getNumOperands() > 0
                        ? viewPositions.find(operation->getOperand(0))
                        : viewPositions.end();
    FailureOr<SmallVector<std::string>> blockShape =
        position != viewPositions.end()
            ? descriptorBlockShape(*operation, views[position->second])
            : FailureOr<SmallVector<std::string>>(failure());
    if (!operation || position == viewPositions.end() || failed(blockShape) ||
        blockShape->empty())
      return physicalProgram.emitOpError(
          "cannot resolve a descriptor block legality constraint");
    descriptorBlocks.emplace_back(views[position->second].pointer,
                                  blockShape->back());
  }
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
            "does not match its selected indexed Triton ragged route");
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
          "does not match its selected compact Triton ragged route");
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

void ProgramMaterializer::bindResult(Operation &operation, unsigned index,
                               StringRef name) {
  Value value = operation.getResult(index);
  valueNames[value] = name.str();
}

void ProgramMaterializer::emitImports() {
  output << "import torch\n";
  output << "import triton\n";
  output << "import triton.language as tl\n";
  if (!planIndex.partitionBindings.empty())
    output << "\n\ndef _intent_partition_extent(logical_extent, count):\n"
              "    if count < 1:\n"
              "        raise ValueError('partition count must be positive')\n"
              "    return (logical_extent + count - 1) // count\n";
  if (llvm::any_of(planIndex.pointwise, [](const auto &entry) {
        return entry.second.getLowering().starts_with("libdevice.");
      }))
    output << "from triton.language.extra import libdevice\n";
  if (!planIndex.components.reusedAxes.empty() || configureRowVector) {
    output << "from intent.runtime.tuning.triton import row_autotune_configurations\n";
    output << "\n_ROW_CONFIGS = row_autotune_configurations(persistent="
           << (!planIndex.components.reusedAxes.empty() ? "True" : "False")
           << ")\n";
    output << "_ROW_TUNED_KEYS = set()\n";
  }
  bool providerAutotune = searchSpace || usesStreamPipelineCandidates();
  if (providerAutotune) {
    llvm::StringMap<uint64_t> staticRoleExtents;
    llvm::StringMap<SmallVector<std::string>> runtimeRoleExtents;
    auto recordExtent = [&](StringRef role, StringRef extent) {
      uint64_t value = 0;
      if (role.empty())
        return;
      if (!extent.getAsInteger(10, value)) {
        if (value == 0)
          return;
        auto found = staticRoleExtents.find(role);
        if (found == staticRoleExtents.end() || value < found->second)
          staticRoleExtents[role] = value;
        return;
      }
      if (!dimensionOwners.count(extent))
        return;
      SmallVector<std::string> &extents = runtimeRoleExtents[role];
      if (!llvm::is_contained(extents, extent))
        extents.push_back(extent.str());
    };
    for (const auto &entry : planIndex.axes)
      for (const target::lowering::RangeBinding &range : entry.second.ranges)
        if (range.getPurpose() != "access")
          recordExtent(range.getTileRole(), range.getExtent());

    output << "from intent.runtime.tuning.triton import autotune_configurations, runtime_extent_pruning\n";
    output << "\n_PARAMETER_MAP = {";
    if (searchSpace)
      for (auto [index, mapping] :
           llvm::enumerate(searchIndex.autotune.getParameterMap())) {
        if (index)
          output << ", ";
        output << "'" << mapping.getName().getValue() << "': '"
               << cast<StringAttr>(mapping.getValue()).getValue() << "'";
      }
    output << "}\n_PARAMETER_EXTENTS = {";
    bool firstExtent = true;
    if (searchSpace)
      for (NamedAttribute mapping : searchIndex.autotune.getParameterMap()) {
        StringRef role = cast<StringAttr>(mapping.getValue()).getValue();
        auto extent = staticRoleExtents.find(role);
        if (extent == staticRoleExtents.end())
          continue;
        if (!firstExtent)
          output << ", ";
        output << "'" << mapping.getName().getValue() << "': "
               << extent->second;
        firstExtent = false;
      }
    output << "}\n_PARAMETER_RUNTIME_EXTENTS = {";
    bool firstRuntimeExtent = true;
    if (searchSpace)
      for (NamedAttribute mapping : searchIndex.autotune.getParameterMap()) {
        StringRef role = cast<StringAttr>(mapping.getValue()).getValue();
        auto extents = runtimeRoleExtents.find(role);
        if (extents == runtimeRoleExtents.end())
          continue;
        if (!firstRuntimeExtent)
          output << ", ";
        output << "'" << mapping.getName().getValue() << "': (";
        for (auto [index, extent] : llvm::enumerate(extents->second)) {
          if (index)
            output << ", ";
          output << "'" << extent << "'";
        }
        if (extents->second.size() == 1)
          output << ",";
        output << ")";
        firstRuntimeExtent = false;
      }
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP";
    if (usesScaledContraction() || usesDescriptorCandidates()) {
      output << ", {";
      bool firstCandidate = true;
      if (usesScaledContraction()) {
        output << "'USE_NATIVE_SCALED': (0, 1)";
        firstCandidate = false;
      }
      if (usesDescriptorCandidates()) {
        output << (firstCandidate ? "" : ", ") << "'USE_TMA': (0, 1)";
        firstCandidate = false;
      }
      output << "}";
    }
    output << ", parameter_extents=_PARAMETER_EXTENTS)\n";
    if (usesDescriptorCandidates()) {
      output << "_DESCRIPTOR_VIEWS = (";
      llvm::StringSet<> emitted;
      bool firstView = true;
      for (const auto &entry : planIndex.transferForms) {
        if (entry.second != "pointer_or_descriptor")
          continue;
        Operation *operation = kernel.nodes.lookup(entry.first);
        auto position = operation && operation->getNumOperands() > 0
                            ? viewPositions.find(operation->getOperand(0))
                            : viewPositions.end();
        if (position == viewPositions.end())
          continue;
        StringRef pointer = views[position->second].pointer;
        if (!emitted.insert(pointer).second)
          continue;
        output << (firstView ? "" : ", ") << "'" << pointer << "'";
        firstView = false;
      }
      if (emitted.size() == 1)
        output << ",";
      output << ")\n";
      output << "_DESCRIPTOR_BLOCKS = (";
      bool firstBlock = true;
      for (const auto &[pointer, lastSpelling] : descriptorBlocks) {
        StringRef last = lastSpelling;
        StringRef parameter;
        if (searchSpace)
          for (NamedAttribute mapping : searchIndex.autotune.getParameterMap())
            if (mapping.getName().getValue() == last) {
              parameter = mapping.getName().getValue();
              break;
            }
        uint64_t fixed = 0;
        bool hasFixed = !last.getAsInteger(10, fixed) && fixed > 0;
        output << (firstBlock ? "" : ", ") << "('" << pointer << "', ";
        if (!parameter.empty())
          output << "'" << parameter << "', None)";
        else if (hasFixed)
          output << "None, " << fixed << ")";
        else
          output << "None, None)";
        firstBlock = false;
      }
      if (!firstBlock)
        output << ",";
      output << ")\n";
    }
  }
  if (usesDescriptorCandidates())
    output << "\ndef _intent_descriptor_allocator(size, alignment, stream):\n"
              "    del alignment, stream\n"
              "    return torch.empty(size, dtype=torch.int8, device=_DEVICE)\n\n"
              "triton.set_allocator(_intent_descriptor_allocator)\n";
  output << "\n\n";
}

LogicalResult ProgramMaterializer::emitHelpers() {
  FailureOr<SmallVector<func::FuncOp>> combiners =
      target::lowering::collectCombiners(kernel.entry);
  if (failed(combiners))
    return failure();
  for (func::FuncOp function : *combiners) {
    FailureOr<std::string> source = target::lowering::renderPythonCombiner(
        function, function.getName(), "@triton.jit", combinerExpression);
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

LogicalResult ProgramMaterializer::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  if (searchSpace || usesStreamPipelineCandidates()) {
    output << "@triton.autotune(\n    configs=_CONFIGS,\n    key=[";
    if (searchSpace)
      for (auto [index, attribute] :
           llvm::enumerate(searchIndex.autotune.getKey())) {
        if (index)
          output << ", ";
        output << "'" << cast<StringAttr>(attribute).getValue() << "'";
      }
    if (usesDescriptorCandidates()) {
      if (searchSpace && !searchIndex.autotune.getKey().empty())
        output << ", ";
      output << "'TMA_LEGAL'";
    }
    output << "]";
    output << ",\n    prune_configs_by=runtime_extent_pruning("
              "_PARAMETER_RUNTIME_EXTENTS";
    if (usesDescriptorCandidates())
      output << ", _DESCRIPTOR_VIEWS, _DESCRIPTOR_BLOCKS, "
             << (usesLinearDescriptorCandidates() ? "True" : "False");
    output << ")";
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
      if (view.dynamicStrides[0])
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
    for (auto [axis, stride] : llvm::enumerate(view.strides))
      if (view.dynamicStrides[axis])
        emitParameter(stride);
  if (searchIndex.autotune)
    for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
      emitParameter(parameter.getName().getValue().str() + ": tl.constexpr");
  if (usesScaledContraction())
    emitParameter("USE_NATIVE_SCALED: tl.constexpr");
  if (usesDescriptorCandidates())
    emitParameter("TMA_LEGAL: tl.constexpr"),
        emitParameter("USE_TMA: tl.constexpr");
  output << "):\n";
  if (failed(emitProgramBindings()))
    return failure();
  if (failed(emitDescriptorDefinitions()))
    return failure();
  return success();
}

LogicalResult ProgramMaterializer::emitWrapper() {
  auto emitViewCapabilityCheck = [&](const ABIView &view) {
    output << "    if any(extent == 0 for extent in "
           << view.argument->name << ".shape):\n";
    output << "        raise NotImplementedError('Triton does not support "
              "zero-extent external views in this compiler')\n";
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
      FailureOr<std::string> size = target::lowering::scanWorkspaceElementCount(
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
      return physicalProgram.emitOpError(
          "fixed grid-stride wrapper has no selected lane range");
    FailureOr<std::string> rowTile = physicalAxisTile(laneAxis);
    if (failed(rowTile))
      return physicalProgram.emitOpError(
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
        if (view.dynamicStrides[0])
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
  if (usesDescriptorCandidates()) {
    output << "    tma_legal = all(view.data_ptr() % 16 == 0 and "
              "view.ndim >= 2 and view.stride(-1) == 1 and ";
    if (usesLinearDescriptorCandidates())
      output << "view.is_contiguous() and ";
    output <<
              "all(stride * view.element_size() % 16 == 0 for stride in "
              "view.stride()[:-1]) and view.shape[-1] * "
              "view.element_size() % 16 == 0 for view in (";
    llvm::StringSet<> emitted;
    bool firstView = true;
    for (const auto &entry : planIndex.transferForms) {
      if (entry.second != "pointer_or_descriptor")
        continue;
      Operation *operation = kernel.nodes.lookup(entry.first);
      auto position = operation && operation->getNumOperands() > 0
                          ? viewPositions.find(operation->getOperand(0))
                          : viewPositions.end();
      if (position == viewPositions.end())
        continue;
      StringRef argument = views[position->second].argument->name;
      if (!emitted.insert(argument).second)
        continue;
      output << (firstView ? "" : ", ") << argument;
      firstView = false;
    }
    if (emitted.size() == 1)
      output << ",";
    output << "))\n";
  }
  if (failed(emitPrivateWorkspaceAllocations()))
    return failure();
  SmallVector<plan::AxisOp> programAxes =
      target::lowering::orderedProgramAxes(planIndex);
  for (plan::AxisOp axis : programAxes) {
    if (!planIndex.components.raggedProgramAxes.contains(axis.getNode()))
      continue;
    FailureOr<plan::RaggedOp> relation =
        target::lowering::uniqueRaggedRelation(
            planIndex, axis.getNode(), *axis.operation.getOperation());
    auto runtime = succeeded(relation)
                       ? raggedRuntimeByRelation.find(relation->getNode())
                       : raggedRuntimeByRelation.end();
    if (failed(relation) || runtime == raggedRuntimeByRelation.end())
      return axis.emitOpError("has no ragged program runtime metadata");
    ABIView *offsets = raggedRuntimes[runtime->second].offsets;
    output << "    max_member_length_" << axis.getNode() << " = int(("
           << offsets->argument->name << "[1:] - "
           << offsets->argument->name << "[:-1]).max().item())\n";
  }
  std::array<std::string, 3> grid = target::lowering::projectProgramGrid(
      planIndex, [&](plan::AxisOp axis) {
        std::string role = "program_" + std::to_string(axis.getProgramOrder());
        std::string extent =
            planIndex.components.raggedProgramAxes.contains(axis.getNode())
                ? "max_member_length_" + std::to_string(axis.getNode())
                : roleDimensions.lookup(role);
        return axis.isScalar()
                   ? extent
                   : "triton.cdiv(" + extent + ", " +
                         launchTile(axis.getTile()) + ")";
      });
  if (planIndex.program.getPersistent()) {
    std::string total = target::lowering::projectProgramVolume(
        planIndex, [&](plan::AxisOp axis) {
          std::string role =
              "program_" + std::to_string(axis.getProgramOrder());
          std::string extent = roleDimensions.lookup(role);
          return axis.isScalar()
                     ? extent
                     : "triton.cdiv(" + extent + ", " +
                           launchTile(axis.getTile()) + ")";
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
        if (view.dynamicStrides[axis])
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
  bool directPipelineLaunch = usesStreamPipelineCandidates() && !hasInOut;
  if (directPipelineLaunch) {
    output << "    " << kernelName << "[grid](";
    emitKernelLaunchArguments(false);
    if (usesDescriptorCandidates())
      output << ", TMA_LEGAL=tma_legal";
    output << ")\n";
    output << "    selected_config = " << kernelName << ".best_config\n";
    output << "    def selected_launch():\n";
    output << "        return " << kernelName << ".fn[grid](";
    emitKernelLaunchArguments(false);
    if (usesDescriptorCandidates())
      output << ", TMA_LEGAL=tma_legal";
    output << ", **selected_config.kwargs, num_warps=selected_config.num_warps, "
              "num_stages=selected_config.num_stages, "
              "num_ctas=selected_config.num_ctas, maxnreg=selected_config.maxnreg)\n";
    output << "    return selected_launch\n\n\n";
  } else {
    output << "    return " << kernelName << "[grid](";
    emitKernelLaunchArguments(false);
    if (usesDescriptorCandidates())
      output << ", TMA_LEGAL=tma_legal";
    output << ")\n\n\n";
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
  if (requiresPreallocatedOutputs) {
    output << "    raise NotImplementedError('this callable requires preallocated output views; use launch')\n\n\n";
    return success();
  }
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

FailureOr<StringRef> ProgramMaterializer::lookupValue(Operation &consumer,
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
        "cannot resolve multi-axis scalar ownership during Triton emission");
    return failure();
  }
  if (scalarSource->opaque) {
    consumer.emitOpError(
        "cannot resolve an opaque index source during Triton emission");
    return failure();
  }
  if (Operation *definition = indexedValue.getDefiningOp()) {
    if (::intent::target::semanticOperationName(*definition) == "intent.domain" ||
        ::intent::target::semanticOperationName(*definition) == "intent.ragged_outer" ||
        ::intent::target::semanticOperationName(*definition) == "intent.ragged_member")
      return definition;
  }
  consumer.emitOpError("cannot resolve index ownership during Triton emission");
  return failure();
}

FailureOr<plan::AxisOp> ProgramMaterializer::resolveAxis(Value indexedValue,
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

FailureOr<plan::AxisOp>
ProgramMaterializer::packedScalarAxis(Value indexedValue,
                                      Operation &consumer) {
  FailureOr<target::ScalarIndexSource> source =
      target::traceScalarIndexSource(indexedValue, consumer);
  if (failed(source))
    return failure();
  if (!source->domain)
    return plan::AxisOp();
  FailureOr<int64_t> node =
      target::getNodeID(*source->domain, "packed scalar projection");
  auto axis = succeeded(node) ? planIndex.axes.find(*node) : planIndex.axes.end();
  if (failed(node))
    return failure();
  if (axis == planIndex.axes.end() ||
      !target::lowering::isPackedScalarAxis(axis->second))
    return plan::AxisOp();
  return axis->second;
}

FailureOr<SmallVector<unsigned>>
ProgramMaterializer::packedScalarInsertions(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  target::TensorIndexGroup tensorIndices =
      target::tensorIndexGroup(operation, *relation);
  SmallVector<unsigned> insertions;
  llvm::DenseSet<int64_t> packedAxes;
  unsigned logicalAxis = 0;
  bool tensorGroupConsumed = false;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "new_axis" || term.kind == "full_slice") {
      ++logicalAxis;
      continue;
    }
    if (term.kind == "static_index")
      continue;
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "packed scalar projection requires value-bound index terms");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index") {
      if (isa<RankedTensorType>(indexed.getType())) {
        if (tensorIndices.requiresBroadcastProjection()) {
          if (!tensorGroupConsumed) {
            logicalAxis += tensorIndices.rank;
            tensorGroupConsumed = true;
          }
        } else {
          ++logicalAxis;
        }
        continue;
      }
      FailureOr<plan::AxisOp> packed = packedScalarAxis(indexed, operation);
      if (failed(packed))
        return failure();
      if (*packed && packedAxes.insert(packed->getNode()).second)
        insertions.push_back(logicalAxis);
      continue;
    }
    if (term.kind != "region_index")
      return operation.emitOpError(
          "has no mechanical packed scalar index projection");
    if (target::lowering::isSequentialIterator(indexed))
      continue;
    FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
    if (failed(axis))
      return failure();
    if (!axis->isScalar())
      ++logicalAxis;
  }
  return insertions;
}

FailureOr<std::string> ProgramMaterializer::dimensionName(Operation &domain) {
  return target::lowering::plannedDomainExtent(domain, planIndex);
}

FailureOr<std::string>
ProgramMaterializer::indexExpression(plan::AxisOp axis, bool store,
                               unsigned tensorAxis, unsigned tensorRank,
                               Operation &consumer) {
  std::string base = axisIndices.lookup(axis.getNode());
  if (base.empty()) {
    consumer.emitOpError()
        << "has no active physical index for logical axis " << axis.getNode()
        << " with roles " << axis.getRoles();
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

std::string ProgramMaterializer::addressIndex(StringRef expression) const {
  return "tl.cast((" + expression.str() + "), tl.int64)";
}

std::string ProgramMaterializer::physicalExtent(StringRef logicalExtent) const {
  auto extent = planIndex.blockExtents.find(logicalExtent);
  if (extent == planIndex.blockExtents.end())
    return logicalExtent.str();
  return "triton.next_power_of_2(" + logicalExtent.str() + ")";
}

bool ProgramMaterializer::usesScaledContraction() const {
  return llvm::any_of(planIndex.contracts, [](const auto &entry) {
    return entry.second.getLowering() == "tl.dot_scaled";
  });
}

bool ProgramMaterializer::usesDescriptorCandidates() const {
  return llvm::any_of(planIndex.transferForms, [](const auto &entry) {
    return entry.second == "pointer_or_descriptor";
  });
}

bool ProgramMaterializer::usesLinearDescriptorCandidates() const {
  return llvm::any_of(planIndex.descriptorLayouts, [](const auto &entry) {
    return entry.second == "linear";
  });
}

bool ProgramMaterializer::usesStreamPipelineCandidates() const {
  for (const auto &entry : planIndex.streams) {
    plan::StreamOp stream = entry.second;
    auto form = stream.physical
                    ? stream.physical->getAttrOfType<StringAttr>(streamFormAttr)
                    : StringAttr();
    if (form && form.getValue() == "pipeline_candidates")
      return true;
  }
  return false;
}

std::string ProgramMaterializer::descriptorName(Operation &operation) const {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "Triton descriptor name");
  return failed(node) ? std::string()
                      : "descriptor_" + std::to_string(*node);
}

FailureOr<SmallVector<std::string>>
ProgramMaterializer::descriptorBlockShape(Operation &operation, ABIView &view) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<int64_t> node =
      target::getNodeID(operation, "Triton descriptor block shape");
  auto axes = succeeded(node) ? planIndex.descriptorBlockAxes.find(*node)
                              : planIndex.descriptorBlockAxes.end();
  if (failed(relation) || failed(node) ||
      axes == planIndex.descriptorBlockAxes.end() ||
      axes->second.size() != view.shape.size() ||
      relation->size() != view.shape.size())
    return operation.emitOpError("has no typed Triton descriptor block shape");
  SmallVector<std::string> shape;
  shape.reserve(axes->second.size());
  for (auto [position, axisNode] : llvm::enumerate(axes->second)) {
    if (axisNode == -1) {
      shape.push_back("1");
      continue;
    }
    if (axisNode == -2) {
      shape.push_back(physicalExtent(view.shape[position]));
      continue;
    }
    const target::IndexTerm &term = (*relation)[position];
    if (term.kind == "region_index" && term.operands.size() == 1 &&
        term.operands.front()) {
      Value indexed = operation.getOperand(*term.operands.front());
      FailureOr<std::optional<target::lowering::RegionRangeBinding>> selected =
          target::lowering::selectedRegionValueRange(planIndex, kernel, indexed,
                                                     operation);
      if (failed(selected) || !*selected ||
          static_cast<int64_t>((*selected)->axis.getNode()) != axisNode)
        return operation.emitOpError(
            "has no selected descriptor region block shape");
      shape.push_back((*selected)->range.getTile().str());
      continue;
    }
    plan::AxisOp axis = planIndex.axes.lookup(axisNode);
    FailureOr<std::string> tile =
        axis ? physicalAxisTile(axis) : FailureOr<std::string>(failure());
    if (failed(tile))
      return operation.emitOpError("has an unbound descriptor block axis");
    shape.push_back(*tile);
  }
  auto layout = planIndex.descriptorLayouts.find(*node);
  if (layout == planIndex.descriptorLayouts.end())
    return operation.emitOpError("has no selected Triton descriptor layout");
  if (layout->second == "linear") {
    auto block = llvm::find_if(axes->second,
                               [](int64_t axis) { return axis >= 0; });
    if (block == axes->second.end() || shape.empty())
      return operation.emitOpError("has no linear descriptor block axis");
    return SmallVector<std::string>{
        shape[block - axes->second.begin()], shape.back()};
  }
  if (layout->second != "strided")
    return operation.emitOpError("has an unknown Triton descriptor layout");
  return shape;
}

FailureOr<std::string> ProgramMaterializer::descriptorLinearRowOffset(
    Operation &operation, ABIView &view) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation) || relation->size() != view.shape.size() ||
      relation->empty() || relation->back().kind != "full_slice")
    return operation.emitOpError("has no linear Triton descriptor relation");
  std::string row;
  for (auto [position, term] : llvm::enumerate(*relation)) {
    if (position + 1 == relation->size())
      break;
    std::string index;
    if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError("has an invalid descriptor static index");
      index = std::to_string(*term.staticValues.front());
    } else if (term.kind == "value_index" && term.operands.size() == 1 &&
               term.operands.front() &&
               !isa<RankedTensorType>(
                   operation.getOperand(*term.operands.front()).getType())) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      index = exact->str();
    } else {
      if (term.kind != "region_index" || term.operands.size() != 1 ||
          !term.operands.front())
        return operation.emitOpError("has a non-linear descriptor index");
      Value indexed = operation.getOperand(*term.operands.front());
      FailureOr<std::optional<target::lowering::RegionRangeBinding>> selected =
          target::lowering::selectedRegionValueRange(planIndex, kernel, indexed,
                                                     operation);
      if (failed(selected) || !*selected)
        return failure();
      if ((*selected)->axis.isScalar()) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        index = exact->str();
      } else {
        auto start = selectedRegionStarts.find(indexed);
        if (start == selectedRegionStarts.end())
          return operation.emitOpError(
              "has no active descriptor block start for its region index");
        index = start->second;
      }
    }
    row = row.empty() ? index
                      : "(" + row + ") * " + view.shape[position] + " + (" +
                            index + ")";
  }
  return row;
}

FailureOr<std::string> ProgramMaterializer::descriptorTensorOrigin(
    Value value, int64_t axisNode, Operation &consumer) {
  if (!isa<RankedTensorType>(value.getType())) {
    auto named = valueNames.find(value);
    if (named == valueNames.end())
      return consumer.emitOpError(
          "has no emitted scalar descriptor-origin operand");
    return named->second;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return consumer.emitOpError("has an opaque tensor descriptor origin");
  StringRef name = target::semanticOperationName(*definition);
  if (name == "intent.indices") {
    std::string start = axisStarts.lookup(axisNode);
    if (start.empty())
      return consumer.emitOpError("has no active descriptor-axis origin");
    return start;
  }
  if (name == "intent.broadcast" || name == "intent.reshape" ||
      name == "intent.transpose" || name == "intent.cast" ||
      (name == "intent.gather" && isDescriptorShapeProjection(*definition))) {
    if (definition->getNumOperands() == 0)
      return consumer.emitOpError("has an empty descriptor-origin projection");
    return descriptorTensorOrigin(definition->getOperand(0), axisNode, consumer);
  }
  if (name != "intent.binary" || definition->getNumOperands() != 2)
    return consumer.emitOpError(
        "has no mechanical affine descriptor-origin projection");
  auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
  FailureOr<std::string> lhs =
      descriptorTensorOrigin(definition->getOperand(0), axisNode, consumer);
  FailureOr<std::string> rhs =
      descriptorTensorOrigin(definition->getOperand(1), axisNode, consumer);
  if (!logical || failed(lhs) || failed(rhs))
    return failure();
  StringRef spelling = logical.getValue() == "add"
                           ? "+"
                           : logical.getValue() == "subtract"
                                 ? "-"
                                 : logical.getValue() == "multiply" ? "*" : "";
  if (spelling.empty())
    return consumer.emitOpError(
        "has a non-affine descriptor-origin operation");
  return "(" + *lhs + ") " + spelling.str() + " (" + *rhs + ")";
}

FailureOr<std::string>
ProgramMaterializer::descriptorOffsets(Operation &operation, ABIView &view) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<int64_t> node =
      target::getNodeID(operation, "Triton descriptor offsets");
  auto axes = succeeded(node) ? planIndex.descriptorBlockAxes.find(*node)
                              : planIndex.descriptorBlockAxes.end();
  if (failed(relation) || failed(node) || relation->size() != view.shape.size() ||
      axes == planIndex.descriptorBlockAxes.end() ||
      axes->second.size() != relation->size())
    return operation.emitOpError("has no rectangular Triton descriptor relation");
  auto layout = planIndex.descriptorLayouts.find(*node);
  if (layout == planIndex.descriptorLayouts.end())
    return operation.emitOpError("has no selected Triton descriptor layout");
  if (layout->second == "linear") {
    FailureOr<std::string> row = descriptorLinearRowOffset(operation, view);
    if (failed(row))
      return failure();
    return "tl.cast((" + *row + "), tl.int32), 0";
  }
  if (layout->second != "strided")
    return operation.emitOpError("has an unknown Triton descriptor layout");
  std::string offsets;
  for (auto [position, term] : llvm::enumerate(*relation)) {
    std::string index;
    if (term.kind == "full_slice") {
      index = "0";
    } else if (term.kind == "static_index") {
      if (term.staticValues.size() != 1 || !term.staticValues.front())
        return operation.emitOpError("has an invalid descriptor static index");
      index = std::to_string(*term.staticValues.front());
    } else if ((term.kind == "value_index" ||
                term.kind == "region_index") &&
               term.operands.size() == 1 && term.operands.front()) {
      Value indexed = operation.getOperand(*term.operands.front());
      if (isa<RankedTensorType>(indexed.getType())) {
        if (axes->second[position] < 0)
          return operation.emitOpError(
              "has a tensor descriptor index without a block axis");
        FailureOr<std::string> origin = descriptorTensorOrigin(
            indexed, axes->second[position], operation);
        if (failed(origin))
          return failure();
        index = *origin;
      } else if (axes->second[position] >= 0 &&
                 term.kind == "region_index") {
        index = axisStarts.lookup(axes->second[position]);
        if (index.empty())
          return operation.emitOpError(
              "has no active descriptor block start for its region index");
      } else {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        index = exact->str();
      }
    } else {
      return operation.emitOpError("has a non-rectangular descriptor index");
    }
    if (index.empty())
      return operation.emitOpError("has an empty descriptor offset");
    if (!offsets.empty())
      offsets += ", ";
    offsets += "tl.cast((" + index + "), tl.int32)";
  }
  return offsets;
}

LogicalResult ProgramMaterializer::emitDescriptorDefinitions() {
  for (const auto &entry : planIndex.transferForms) {
    if (entry.second != "pointer_or_descriptor")
      continue;
    Operation *operation = kernel.nodes.lookup(entry.first);
    FailureOr<ABIView *> view =
        operation && operation->getNumOperands() > 0
            ? lookupView(operation->getOperand(0), *operation)
            : FailureOr<ABIView *>(failure());
    FailureOr<SmallVector<std::string>> blockShape =
        succeeded(view) ? descriptorBlockShape(*operation, **view)
                        : FailureOr<SmallVector<std::string>>(failure());
    auto layout = planIndex.descriptorLayouts.find(entry.first);
    if (!operation || failed(view) || failed(blockShape) ||
        layout == planIndex.descriptorLayouts.end())
      return failure();
    auto emitList = [&](ArrayRef<std::string> values) {
      std::string result;
      for (auto [index, value] : llvm::enumerate(values)) {
        if (index)
          result += ", ";
        result += value;
      }
      return result;
    };
    line("if USE_TMA:");
    ++indentation;
    if (layout->second == "linear") {
      std::string rows;
      for (size_t axis = 0; axis + 1 < (*view)->shape.size(); ++axis)
        rows = rows.empty() ? (*view)->shape[axis]
                            : "(" + rows + ") * " + (*view)->shape[axis];
      line(descriptorName(*operation) + " = tl.make_tensor_descriptor(" +
           (*view)->pointer + ", [" + rows + ", " + (*view)->shape.back() +
           "], [" + (*view)->shape.back() + ", 1], [" +
           (*blockShape).front() + ", " + (*blockShape).back() + "])");
    } else if (layout->second == "strided") {
      line(descriptorName(*operation) + " = tl.make_tensor_descriptor(" +
           (*view)->pointer + ", [" + emitList((*view)->shape) + "], [" +
           emitList((*view)->strides) + "], [" + emitList(*blockShape) + "])");
    } else {
      return operation->emitOpError("has an unknown Triton descriptor layout");
    }
    --indentation;
  }
  return success();
}

FailureOr<std::string> ProgramMaterializer::physicalAxisTile(plan::AxisOp axis) {
  const target::lowering::RangeBinding *range =
      target::lowering::canonicalDomainRange(axis);
  if (!range)
    return axis.emitOpError("has no canonical Triton domain range");
  if (axis.getReuseWorker() ||
      !range->getTileRole().starts_with("row_vector"))
    return range->getTile().str();
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

FailureOr<std::string>
ProgramMaterializer::emitPointerExpression(Operation &operation, ABIView &view,
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
  target::TensorIndexGroup tensorIndices =
      target::tensorIndexGroup(operation, *relation);
  std::string expression = view.pointer;
  unsigned vectorAxis = 0;
  unsigned sourceAxis = 0;
  std::optional<unsigned> tensorGroupAxis;
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
          FailureOr<plan::AxisOp> packed =
              packedScalarAxis(indexed, operation);
          if (failed(packed))
            return failure();
          if (*packed) {
            if (vectorAxis >= *tensorRank)
              return operation.emitOpError(
                  "packed scalar index exceeds the emitted Triton tensor rank");
            index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
          } else {
            index = "(" + exact->str() + ")";
          }
        } else {
          if (tensorIndices.requiresBroadcastProjection()) {
            if (!tensorGroupAxis) {
              if (vectorAxis + tensorIndices.rank > *tensorRank)
                return operation.emitOpError(
                    "Triton broadcasted tensor index exceeds the emitted tensor rank");
              tensorGroupAxis = vectorAxis;
              vectorAxis += tensorIndices.rank;
            }
            if (tensor.getRank() > static_cast<int64_t>(tensorIndices.rank))
              return operation.emitOpError(
                  "Triton broadcasted tensor index exceeds the emitted tensor rank");
            index = projectTensorIndex(*exact, tensor.getRank(),
                                       tensorIndices.rank, *tensorGroupAxis,
                                       *tensorRank);
          } else {
            if (tensor.getRank() != 1 || vectorAxis >= *tensorRank)
              return operation.emitOpError(
                  "Triton indirect tensor indices currently require one logical axis");
            index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
          }
        }
      } else if (term.kind == "region_index" && isa<BlockArgument>(indexed)) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
        if (failed(exact) || failed(axis))
          return failure();
        if (axis->isScalar()) {
          index = "(" + exact->str() + ")";
        } else {
          index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
        }
      } else if (target::lowering::isSequentialIterator(indexed)) {
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
ProgramMaterializer::emitMaskExpression(Operation &operation, bool store) {
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
  target::TensorIndexGroup tensorIndices =
      target::tensorIndexGroup(operation, *relation);
  SmallVector<std::string> predicates;
  unsigned vectorAxis = 0;
  unsigned sourceAxis = 0;
  std::optional<unsigned> tensorGroupAxis;
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
    if (plan::PartitionBindingOp partition =
            target::lowering::countPartitionForPartValue(planIndex, kernel,
                                                         indexed)) {
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      Value countValue = kernel.values.lookup(partition.getCountValue());
      Operation *source = kernel.nodes.lookup(partition.getPartitionNode());
      FailureOr<std::string> count =
          countValue && source
              ? target::lowering::sourceIntegerSpelling(countValue, kernel,
                                                        *source)
              : FailureOr<std::string>(failure());
      if (failed(exact) || failed(count))
        return failure();
      predicates.push_back("(" + exact->str() + " < " + *count + ")");
      continue;
    }
    if (term.kind == "value_index" &&
        isa<RankedTensorType>(indexed.getType())) {
      auto tensor = cast<RankedTensorType>(indexed.getType());
      FailureOr<StringRef> exact =
          lookupValue(operation, *term.operands.front());
      if (failed(exact))
        return failure();
      std::string index;
      if (tensorIndices.requiresBroadcastProjection()) {
        if (!tensorGroupAxis) {
          if (vectorAxis + tensorIndices.rank > *tensorRank)
            return operation.emitOpError(
                "Triton broadcasted tensor bounds exceed the emitted tensor rank");
          tensorGroupAxis = vectorAxis;
          vectorAxis += tensorIndices.rank;
        }
        if (tensor.getRank() > static_cast<int64_t>(tensorIndices.rank))
          return operation.emitOpError(
              "Triton broadcasted tensor bounds exceed the emitted tensor rank");
        index = projectTensorIndex(*exact, tensor.getRank(), tensorIndices.rank,
                                   *tensorGroupAxis, *tensorRank);
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
      FailureOr<plan::AxisOp> packed = packedScalarAxis(indexed, operation);
      if (failed(packed))
        return failure();
      bool packedScalar = static_cast<bool>(*packed);
      if (source->hasDomain() && (source->transformed || packedScalar)) {
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        std::string index;
        if (packedScalar) {
          if (vectorAxis >= *tensorRank)
            return operation.emitOpError(
                "packed scalar bounds exceed the emitted Triton tensor rank");
          index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
        } else {
          index = "(" + exact->str() + ")";
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
      }
      continue;
    }
    if (term.kind == "region_index" &&
        target::lowering::isSequentialIterator(indexed))
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
    if (target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode())) {
      FailureOr<int64_t> ordered = target::lowering::representativeOrderedAxis(
          planIndex, axis->getNode(), operation);
      if (failed(ordered))
        return failure();
      extent = "sequence_end_" + std::to_string(*ordered);
    }
    if (auto active = activeTraversalEnds.find(axis->getNode());
        active != activeTraversalEnds.end())
      extent = active->second;
    if (failed(extent))
      return failure();
    FailureOr<std::string> index =
        term.kind == "region_index" && isa<BlockArgument>(indexed)
            ? [&]() -> FailureOr<std::string> {
                FailureOr<StringRef> exact =
                    lookupValue(operation, *term.operands.front());
                if (failed(exact))
                  return failure();
                return axis->isScalar()
                           ? exact->str()
                           : broadcastIndex(*exact, vectorAxis, *tensorRank);
              }()
            : indexExpression(*axis, store, vectorAxis, *tensorRank, operation);
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

FailureOr<std::string> ProgramMaterializer::emitValidityExpression(
    ArrayRef<int64_t> tensorAxes, ArrayRef<int64_t> domainNodes, Value value,
    Operation &consumer) {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || tensorAxes.size() != domainNodes.size())
    return consumer.emitOpError(
        "has no ranked Triton value-validity binding");
  SmallVector<unsigned> packedInsertions;
  if (consumer.hasAttr("intent.index")) {
    FailureOr<SmallVector<unsigned>> projected =
        packedScalarInsertions(consumer);
    if (failed(projected))
      return failure();
    packedInsertions = std::move(*projected);
  } else if (llvm::any_of(domainNodes, [&](int64_t node) {
               auto axis = planIndex.axes.find(node);
               return axis != planIndex.axes.end() &&
                      target::lowering::isPackedScalarAxis(axis->second);
             })) {
    return consumer.emitOpError(
        "has padded packed-scalar results without an explicit physical-axis projection");
  }
  unsigned emittedRank = tensor.getRank() + packedInsertions.size();
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
    if (target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> ordered = target::lowering::representativeOrderedAxis(
          planIndex, axis.getNode(), consumer);
      if (failed(ordered))
        return failure();
      extent = "sequence_end_" + std::to_string(*ordered);
    }
    if (auto active = activeTraversalEnds.find(axis.getNode());
        active != activeTraversalEnds.end())
      extent = active->second;
    unsigned physicalTensorAxis = tensorAxis;
    for (unsigned insertion : packedInsertions)
      physicalTensorAxis += insertion <= static_cast<unsigned>(tensorAxis);
    FailureOr<std::optional<target::lowering::ResultAxisRegionRangeBinding>>
        region = target::lowering::selectedResultAxisRegionRange(
            planIndex, kernel, value, tensorAxis, consumer);
    if (failed(region))
      return failure();
    FailureOr<std::string> index = failure();
    if (*region) {
      if ((*region)->selected.axis.getNode() != domainNode)
        return consumer.emitOpError(
            "binds one result axis to conflicting logical and physical axes");
      auto projected = valueNames.find((*region)->argument);
      if (projected == valueNames.end())
        return consumer.emitOpError(
            "has no active Triton region projection for its validity axis");
      index = broadcastIndex(projected->second, physicalTensorAxis, emittedRank);
    } else {
      index =
          indexExpression(axis, false, physicalTensorAxis, emittedRank, consumer);
    }
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

FailureOr<std::string> ProgramMaterializer::padExpression(
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
  unsigned rank = 0;
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    rank = tensor.getRank();
  else if (!type.isIntOrIndexOrFloat())
    return operation.emitOpError("boundary value is neither a tensor nor a scalar");

  FailureOr<SmallVector<unsigned>> insertions =
      packedScalarInsertions(operation);
  if (failed(insertions))
    return failure();
  return rank + insertions->size();
}

FailureOr<std::string>
ProgramMaterializer::emitTensorShape(Operation &operation, unsigned resultIndex) {
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
    FailureOr<std::optional<target::lowering::RegionRangeBinding>>
        selected = target::lowering::selectedResultAxisPhysicalRange(
            planIndex, kernel, resultValue, tensorAxis, operation);
    if (failed(selected))
      return failure();
    if (*selected) {
      const target::lowering::RegionRangeBinding &binding = **selected;
      const target::lowering::RangeBinding &range = binding.range;
      bool rounded = !binding.axis.getReuseWorker() &&
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

std::string ProgramMaterializer::broadcastIndex(StringRef base, unsigned axis,
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

std::string ProgramMaterializer::programGrid(StringRef extent) {
  int64_t axis = planIndex.axesByRole.lookup("program_0").getWorkerAxis();
  if (axis == 0)
    return "(" + extent.str() + ", 1, 1)";
  if (axis == 1)
    return "(1, " + extent.str() + ", 1)";
  return "(1, 1, " + extent.str() + ")";
}

void ProgramMaterializer::line(StringRef text) {
  output.indent(indentation * 4) << text << "\n";
}

LogicalResult materializeProgramSource(
    target::KernelModel kernel,
    intent::plan::ProgramOp physicalProgram,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<PhysicalProgramIndex> indexed = indexPhysicalProgram(physicalProgram, kernel);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return ProgramMaterializer(std::move(kernel), physicalProgram, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .materialize();
}

} // namespace intent::triton::lowering

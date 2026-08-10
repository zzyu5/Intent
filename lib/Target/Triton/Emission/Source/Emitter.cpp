#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::triton::emission {
namespace {

StringRef torchDtype(Type type) {
  if (type.isF16())
    return "torch.float16";
  if (type.isF32())
    return "torch.float32";
  if (type.isBF16())
    return "torch.bfloat16";
  if (type.isInteger(8))
    return "torch.int8";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 32)
    return "torch.int32";
  return {};
}

FailureOr<std::string> tileSpelling(Operation *operation, StringRef role) {
  if (role == "one")
    return std::string("1");
  if (role == "row_vector")
    return std::string("BLOCK_SIZE");
  if (role.starts_with("row_vector_"))
    return "BLOCK_SIZE_V" + role.drop_front(11).str();
  if (role == "program_m" || role == "ragged_member")
    return std::string("BLOCK_SIZE_M");
  if (role.starts_with("ragged_member_"))
    return "BLOCK_SIZE_R" + role.drop_front(14).str();
  if (role == "program_n")
    return std::string("BLOCK_SIZE_N");
  if (role.starts_with("program_"))
    return "BLOCK_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return std::string("BLOCK_SIZE_K");
  if (role.starts_with("reduction_"))
    return "BLOCK_SIZE_K" + role.drop_front(10).str();
  if (role == "query")
    return std::string("BLOCK_SIZE_Q");
  if (role.starts_with("query_"))
    return "BLOCK_SIZE_Q" + role.drop_front(6).str();
  if (role == "stream")
    return std::string("BLOCK_SIZE_K");
  if (role == "stream_contract")
    return std::string("BLOCK_SIZE_K");
  if (role.starts_with("stream_contract_"))
    return "BLOCK_SIZE_C" + role.drop_front(16).str();
  if (role.starts_with("stream_"))
    return "BLOCK_SIZE_S" + role.drop_front(7).str();
  operation->emitOpError("has no Triton tile spelling for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwiseSpelling(Operation *operation, StringRef role) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "counter_random_f32")
    return StringRef("counter_xorshift32");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return StringRef("tl.cast");
  if (role == "reshape")
    return StringRef("tl.reshape");
  if (role == "unary_exp")
    return StringRef("tl.exp");
  if (role == "unary_exp2")
    return StringRef("tl.exp2");
  if (role == "unary_log")
    return StringRef("tl.log");
  if (role == "unary_rsqrt")
    return StringRef("tl.rsqrt");
  if (role == "unary_sigmoid")
    return StringRef("tl.sigmoid");
  if (role == "unary_negate")
    return StringRef("python_negate");
  if (role == "binary_add")
    return StringRef("python_add");
  if (role == "binary_subtract")
    return StringRef("python_subtract");
  if (role == "binary_multiply")
    return StringRef("python_multiply");
  if (role == "binary_true_divide")
    return StringRef("python_true_divide");
  if (role == "binary_floor_divide")
    return StringRef("python_floor_divide");
  if (role == "binary_remainder")
    return StringRef("python_remainder");
  if (role == "binary_maximum")
    return StringRef("tl.maximum");
  if (role == "binary_minimum")
    return StringRef("tl.minimum");
  if (role == "compare_equal")
    return StringRef("python_equal");
  if (role == "compare_not_equal")
    return StringRef("python_not_equal");
  if (role == "compare_less")
    return StringRef("python_less");
  if (role == "compare_less_equal")
    return StringRef("python_less_equal");
  if (role == "compare_greater")
    return StringRef("python_greater");
  if (role == "compare_greater_equal")
    return StringRef("python_greater_equal");
  if (role == "mask")
    return StringRef("tl.where");
  if (role == "full")
    return StringRef("tl.full");
  if (role == "zeros")
    return StringRef("tl.zeros");
  if (role == "members")
    return StringRef("tl.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("tl.indirect_gather");
  operation->emitOpError("has no Triton pointwise spelling for role ") << role;
  return failure();
}

FailureOr<std::string> parameterSpelling(Operation *operation, StringRef role) {
  if (role == "program_m" || role == "ragged_member")
    return std::string("BLOCK_SIZE_M");
  if (role.starts_with("ragged_member_"))
    return "BLOCK_SIZE_R" + role.drop_front(14).str();
  if (role == "program_n" || role == "feature")
    return std::string("BLOCK_SIZE_N");
  if (role.starts_with("program_"))
    return "BLOCK_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return std::string("BLOCK_SIZE_K");
  if (role.starts_with("reduction_"))
    return "BLOCK_SIZE_K" + role.drop_front(10).str();
  if (role == "group_m")
    return std::string("GROUP_SIZE_M");
  if (role.starts_with("group_"))
    return "GROUP_SIZE_G" + role.drop_front(6).str();
  if (role == "query")
    return std::string("BLOCK_SIZE_Q");
  if (role.starts_with("query_"))
    return "BLOCK_SIZE_Q" + role.drop_front(6).str();
  if (role == "stream")
    return std::string("BLOCK_SIZE_K");
  if (role == "stream_contract")
    return std::string("BLOCK_SIZE_K");
  if (role.starts_with("stream_contract_"))
    return "BLOCK_SIZE_C" + role.drop_front(16).str();
  if (role.starts_with("stream_"))
    return "BLOCK_SIZE_S" + role.drop_front(7).str();
  operation->emitOpError("has no Triton tuner parameter for role ") << role;
  return failure();
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
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (auto value = dyn_cast<intent::plan::DeviceOp>(operation)) {
      index.target.operation = value;
    } else if (auto value = dyn_cast<intent::plan::AxisOp>(operation)) {
      FailureOr<std::string> tile = tileSpelling(value, value.getTile());
      if (failed(tile))
        return failure();
      plan::AxisOp binding;
      binding.operation = value;
      binding.tile = *tile;
      if (std::optional<StringRef> group = value.getGroup()) {
        FailureOr<std::string> spelling = parameterSpelling(value, *group);
        if (failed(spelling))
          return failure();
        binding.group = *spelling;
      }
      index.axes.try_emplace(value.getNode(), binding);
    } else if (auto value = dyn_cast<intent::plan::ProgramOp>(operation)) {
      index.program.operation = value;
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
      binding.lowering = "tl.dot";
      binding.lhsSpace = value.getLhsSpace().str();
      binding.rhsSpace = value.getRhsSpace().str();
      binding.accumulatorSpace = value.getAccumulatorSpace().str();
      index.contracts[value.getNode()] = binding;
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
    }
  }
  if (failed(target::emission::indexCanonicalStructure(index, kernel)))
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
    binding.lowering = *role == "reduce_argmax"
                           ? "tl.max_with_index"
                       : *role == "reduce_maximum" ? "tl.max"
                                                    : "tl.sum";
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = *axis;
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::ScanOp value : scans) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::scanRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<int64_t> axis =
        operation ? target::emission::scanAxis(*operation)
                  : FailureOr<int64_t>(failure());
    if (failed(role) || failed(axis) || *role != "scan_inclusive_add")
      return value.emitOpError("does not bind a canonical scan");
    plan::ScanOp binding;
    binding.operation = value;
    binding.lowering = "tl.cumsum";
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = *axis;
    index.scans[value.getNode()] = binding;
  }
  for (intent::plan::PointwiseOp value : pointwise) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<StringRef> lowering =
        succeeded(role) ? pointwiseSpelling(value, *role)
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
    FailureOr<std::string> tile =
        axis ? tileSpelling(binding.operation, axis.getTileRole())
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
                  operation->getName().getStringRef() == "intent.atomic_add");
    if (!load && !store)
      return value.emitOpError("does not bind a canonical transfer");
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = load ? "load" : "store";
    binding.resultSpace = value.getResultSpace().str();
    binding.defer = load && target::emission::feedsContraction(*operation) &&
                    (!index.components.groups.empty() ||
                     target::emission::feedsStagedContraction(index, *operation));
    index.boundaries[value.getNode()] = binding;
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
        FailureOr<std::string> spelling = parameterSpelling(autotune, role);
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
  if (!planIndex.stages.empty())
    return prepareRaggedStages();
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
  }
  for (auto &entry : planIndex.paddings) {
    Value value = kernel.values.lookup(entry.first);
    Operation *definition = value ? value.getDefiningOp() : nullptr;
    if (!definition || definition->getNumResults() != 1 ||
        definition->getResult(0) != value ||
        (definition->getName().getStringRef() != "intent.binary" &&
         definition->getName().getStringRef() != "intent.mask"))
      return entry.second.emitOpError(
          "does not bind a producer-fusible pointwise value");
  }
  for (const target::RegionNode &region : kernel.regions.nodes) {
    Operation *operation = region.operation;
    StringRef name = operation->getName().getStringRef();
    if (name != "intent.parallel" && name != "intent.state_stream")
      continue;
    Operation *domain = nullptr;
    if (name == "intent.state_stream") {
      domain = operation->getNumOperands() > 0
                   ? operation->getOperand(0).getDefiningOp()
                   : nullptr;
    } else if (operation->getNumOperands() > 0) {
      Operation *source = operation->getOperand(0).getDefiningOp();
      if (source && source->getName().getStringRef() == "intent.partition" &&
          source->getNumOperands() == 1)
        source = source->getOperand(0).getDefiningOp();
      domain = source;
    }
    FailureOr<int64_t> domainNode =
        domain ? target::getNodeID(*domain, "region tile indexing")
               : FailureOr<int64_t>(failure());
    plan::AxisOp axis = succeeded(domainNode)
                            ? planIndex.axes.lookup(*domainNode)
                            : plan::AxisOp();
    auto regions = operation->getAttrOfType<ArrayAttr>(
        "intent.region_argument_nodes");
    auto blocks = regions && !regions.empty() ? dyn_cast<ArrayAttr>(regions[0])
                                              : ArrayAttr();
    auto arguments = blocks && !blocks.empty() ? dyn_cast<ArrayAttr>(blocks[0])
                                               : ArrayAttr();
    auto argument = arguments && !arguments.empty()
                        ? dyn_cast<IntegerAttr>(arguments[0])
                        : IntegerAttr();
    if (failed(domainNode) || !axis || !argument)
      return operation->emitOpError(
          "cannot index its region shape against the physical plan");
    regionTiles["?region_" + std::to_string(argument.getInt()) + "_0"] =
        axis.getTile().str();
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
  if ((!planIndex.components.groups.empty() || !planIndex.streams.empty() ||
       !planIndex.stages.empty()) &&
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
    if (!runtime.relation ||
        runtime.relation->getName().getStringRef() != "intent.ragged" ||
        !runtime.outer ||
        runtime.outer->getName().getStringRef() != "intent.ragged_outer" ||
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
    stageRaggedRuntime[position] = runtimeIndex;
    stageFeatureDimensions[position] = feature.getExtent().str();
    stageMemberDimensions[position] = member.getExtent().str();
    stageReductionDimensions[position] = reduction.getExtent().str();
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
       " + member_offsets[:, None] * " + feature +
       " + offs_feature[None, :], " + name.str() +
       ", mask=member_mask[:, None] & feature_mask[None, :])");
}

void SourceEmitter::emitImports() {
  output << "import torch\n";
  output << "import triton\n";
  output << "import triton.language as tl\n";
  if (!planIndex.components.reusedAxes.empty()) {
    output << "from triton.runtime import driver\n";
    output << "from intent.runtime.tuning.triton import row_configuration, row_program_count\n";
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
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP)\n";
  }
  output << "\n\n";
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
                  (compact && dimension == experts ? ": tl.constexpr" : ""));
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
      stageLine(stage, "pid_feature = tl.program_id(axis=" +
                           std::to_string(stageFeatureWorkers.lookup(stage)) +
                           ")");
      stageLine(stage, "pid_expert_route = tl.program_id(axis=" +
                           std::to_string(stageMemberWorkers.lookup(stage)) +
                           ")");
      if (compact) {
        stageLine(stage, "expert = 0");
        stageLine(stage, "route_tile = 0");
        stageLine(stage, "tile_cursor = 0");
        stageLine(stage, "for candidate in range(0, " + experts + "):");
        stageLine(stage, "candidate_begin = tl.load(" + offsets->pointer +
                             " + candidate * " + offsets->strides[0] + ")",
                  2);
        stageLine(stage, "candidate_end = tl.load(" + offsets->pointer +
                             " + (candidate + 1) * " +
                             offsets->strides[0] + ")",
                  2);
        stageLine(stage,
                  "candidate_tiles = tl.cdiv(candidate_end - candidate_begin, "
                  "BLOCK_SIZE_M)",
                  2);
        stageLine(stage,
                  "owns_tile = (pid_expert_route >= tile_cursor) & "
                  "(pid_expert_route < tile_cursor + candidate_tiles)",
                  2);
        stageLine(stage, "expert = tl.where(owns_tile, candidate, expert)", 2);
        stageLine(stage,
                  "route_tile = tl.where(owns_tile, pid_expert_route - "
                  "tile_cursor, route_tile)",
                  2);
        stageLine(stage, "tile_cursor += candidate_tiles", 2);
      } else {
        stageLine(stage,
                  "num_route_tiles = tl.cdiv(MAX_ROUTES, BLOCK_SIZE_M)");
        stageLine(stage, "expert = pid_expert_route // num_route_tiles");
        stageLine(stage, "route_tile = pid_expert_route % num_route_tiles");
      }
      stageLine(stage, "route_begin = tl.load(" + offsets->pointer +
                           " + expert * " + offsets->strides[0] + ")");
      stageLine(stage, "route_end = tl.load(" + offsets->pointer +
                           " + (expert + 1) * " + offsets->strides[0] +
                           ")");
      stageLine(stage,
                "member_offsets = route_begin + route_tile * BLOCK_SIZE_M + "
                "tl.arange(0, BLOCK_SIZE_M)");
      stageLine(stage, "member_mask = member_offsets < route_end");
      if (indices)
        stageLine(stage, "routes = tl.load(" + indices->pointer +
                             " + member_offsets * " + indices->strides[0] +
                             ", mask=member_mask, other=0)");
      else
        stageLine(stage, "routes = member_offsets");
      stageLine(stage,
                "offs_feature = pid_feature * BLOCK_SIZE_N + "
                "tl.arange(0, BLOCK_SIZE_N)");
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
          "num_stages: tl.constexpr"})
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
  for (const std::string &dimension : dimensionOrder) {
    bool physicalDimension = llvm::any_of(
        roleDimensions,
        [&](const auto &binding) { return binding.getValue() == dimension; });
    emitParameter(dimension +
                  (physicalDimension ? "" : ": tl.constexpr"));
  }
  for (ABIView &view : views)
    for (const std::string &stride : view.strides)
      emitParameter(stride);
  if (searchIndex.autotune)
    for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
      emitParameter(parameter.getName().getValue().str() + ": tl.constexpr");
  output << "):\n";
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
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
      output << "    grid_stage_" << stage
             << " = lambda META: (triton.cdiv(" << feature
             << ", META['BLOCK_SIZE_N']), ";
      if (compact)
        output << "sum(triton.cdiv(length, META['BLOCK_SIZE_M']) for length "
                  "in route_lengths_"
               << suffix << ")";
      else
        output << experts
               << " * triton.cdiv(max_routes_" << suffix
               << ", META['BLOCK_SIZE_M'])";
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
    output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
           << ")\n";
    output << "_PROPERTIES = driver.active.utils.get_device_properties(_DEVICE.index)\n";
    output << "_WARP_SIZE = torch.cuda.get_device_properties(_DEVICE).warp_size\n\n\n";
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
    output << "    configuration = row_configuration(n_cols, _PROPERTIES)\n";
    output << "    kernel = " << kernelName << ".warmup(";
    bool first = true;
    auto emitArgument = [&](StringRef argument) {
      if (!first)
        output << ", ";
      output << argument;
      first = false;
    };
    for (ABIView &view : views)
      emitArgument(view.argument->name);
    for (ABIScalar &scalar : scalars)
      emitArgument(scalar.name);
    for (const std::string &dimension : dimensionOrder)
      if (dimension != roleDimensions.lookup("program_0") &&
          dimension != roleDimensions.lookup("lane_0"))
        emitArgument(dimension);
    for (ABIView &view : views)
      emitArgument(view.argument->name + ".stride(0)");
    for (StringRef argument : {"n_rows", "n_cols"})
      emitArgument(argument);
    output << ", BLOCK_SIZE=configuration.tile_size, "
              "num_stages=configuration.num_stages, "
              "num_warps=configuration.num_warps, grid=" << programGrid("1")
           << ")\n";
    output << "    kernel._init_handles()\n";
    output << "    num_programs = row_program_count(n_rows, kernel, _PROPERTIES, "
              "_WARP_SIZE, configuration)\n";
    output << "    return " << kernelName << "[" << programGrid("num_programs")
           << "](";
    first = true;
    for (ABIView &view : views)
      emitArgument(view.argument->name);
    for (ABIScalar &scalar : scalars)
      emitArgument(scalar.name);
    for (const std::string &dimension : dimensionOrder)
      if (dimension != roleDimensions.lookup("program_0") &&
          dimension != roleDimensions.lookup("lane_0"))
        emitArgument(dimension);
    for (ABIView &view : views)
      emitArgument(view.argument->name + ".stride(0)");
    for (StringRef argument : {"n_rows", "n_cols", "configuration.tile_size",
                               "configuration.num_stages"})
      emitArgument(argument);
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
  output << "    return " << kernelName << "[grid](";
  bool first = true;
  auto emitArgument = [&](StringRef argument) {
    if (!first)
      output << ", ";
    output << argument;
    first = false;
  };
  for (ABIView &view : views)
    emitArgument(view.argument->name);
  for (ABIScalar &scalar : scalars)
    emitArgument(scalar.name);
  for (const std::string &dimension : dimensionOrder)
    emitArgument(dimension);
  for (ABIView &view : views)
    for (int64_t axis = 0; axis < view.tensor.getRank(); ++axis)
      emitArgument(view.argument->name + ".stride(" + std::to_string(axis) + ")");
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
  first = true;
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
  if (Operation *definition = indexedValue.getDefiningOp()) {
    if (definition->getName().getStringRef() == "intent.domain" ||
        definition->getName().getStringRef() == "intent.ragged_outer" ||
        definition->getName().getStringRef() == "intent.ragged_member")
      return definition;
  }
  auto argument = dyn_cast<BlockArgument>(indexedValue);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (owner && owner->getName().getStringRef() == "intent.state_stream" &&
      argument.getArgNumber() == 0 && owner->getNumOperands() > 0) {
    Operation *domain = owner->getOperand(0).getDefiningOp();
    if (domain &&
        (domain->getName().getStringRef() == "intent.domain" ||
         domain->getName().getStringRef() == "intent.ragged_outer" ||
         domain->getName().getStringRef() == "intent.ragged_member"))
      return domain;
    consumer.emitOpError("cannot resolve a stream to its source domain");
    return failure();
  }
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("cannot resolve index ownership during emission");
    return failure();
  }
  Operation *source = owner->getOperand(0).getDefiningOp();
  if (source &&
      (source->getName().getStringRef() == "intent.domain" ||
       source->getName().getStringRef() == "intent.ragged_outer" ||
       source->getName().getStringRef() == "intent.ragged_member"))
    return source;
  if (source && source->getName().getStringRef() == "intent.partition" &&
      source->getNumOperands() == 1) {
    Operation *domain = source->getOperand(0).getDefiningOp();
    if (domain &&
        (domain->getName().getStringRef() == "intent.domain" ||
         domain->getName().getStringRef() == "intent.ragged_member"))
      return domain;
  }
  consumer.emitOpError("cannot resolve a partition to its source domain");
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
  StringRef name = domain.getName().getStringRef();
  if (name == "intent.ragged_outer") {
    Operation *relation = domain.getOperand(0).getDefiningOp();
    Operation *source = relation &&
                                (relation->getNumOperands() == 3 ||
                                 relation->getNumOperands() == 4)
                            ? relation->getOperand(0).getDefiningOp()
                            : nullptr;
    if (!source)
      return failure();
    return dimensionName(*source);
  }
  if (name == "intent.ragged_member") {
    Operation *relation = domain.getOperand(0).getDefiningOp();
    Operation *source = relation &&
                                (relation->getNumOperands() == 3 ||
                                 relation->getNumOperands() == 4)
                            ? relation->getOperand(1).getDefiningOp()
                            : nullptr;
    if (!source)
      return failure();
    return dimensionName(*source);
  }
  if (domain.getNumOperands() < 2)
    return failure();
  Operation *dim = domain.getOperand(1).getDefiningOp();
  auto axis = dim ? dim->getAttrOfType<IntegerAttr>("intent.axis") : IntegerAttr();
  if (!dim || dim->getName().getStringRef() != "intent.dim" || !axis ||
      dim->getNumOperands() != 1)
    return failure();
  FailureOr<ABIView *> view = lookupView(dim->getOperand(0), domain);
  if (failed(view) || axis.getInt() < 0 ||
      static_cast<size_t>(axis.getInt()) >= (*view)->shape.size())
    return failure();
  return (*view)->shape[axis.getInt()];
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

FailureOr<std::string>
SourceEmitter::emitPointerExpression(Operation &operation, ABIView &view,
                                     bool store) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation) || relation->size() != view.strides.size())
    return failure();
  FailureOr<unsigned> tensorRank = emittedTensorRank(operation, store);
  if (failed(tensorRank))
    return failure();
  std::string expression = view.pointer;
  unsigned vectorAxis = 0;
  for (auto [axisNumber, term] : llvm::enumerate(*relation)) {
    std::string index;
    if (term.kind == "full_slice") {
      index = broadcastIndex("tl.arange(0, " + view.shape[axisNumber] + ")",
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
          if (tensor.getRank() != 1 || vectorAxis >= *tensorRank)
            return operation.emitOpError(
                "Triton indirect tensor indices currently require one logical axis");
          index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
        }
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
      stride = "1";
      for (unsigned trailing = axisNumber + 1; trailing < view.shape.size();
           ++trailing)
        stride += " * " + view.shape[trailing];
    }
    expression += " + " + index + " * (" + stride + ")";
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
  if (failed(view) || relation->size() != (*view)->shape.size())
    return failure();
  FailureOr<unsigned> tensorRank = emittedTensorRank(operation, store);
  if (failed(tensorRank))
    return failure();
  SmallVector<std::string> predicates;
  unsigned vectorAxis = 0;
  for (auto [axisNumber, term] : llvm::enumerate(*relation)) {
    if (term.kind == "full_slice") {
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
      if (tensor.getRank() != 1 || failed(exact) || vectorAxis >= *tensorRank)
        return operation.emitOpError(
            "Triton indirect tensor bounds require one logical axis");
      std::string index = broadcastIndex(*exact, vectorAxis++, *tensorRank);
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
      if (source->domain && source->transformed) {
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
  StringRef fill = padding.getFill() == "negative_infinity"
                       ? StringRef("-float('inf')")
                       : StringRef("0.0");
  return "tl.where(" + *predicate + ", " + expression.str() + ", " +
         fill.str() + ")";
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
    else if (label.getValue().starts_with("?region_"))
      return operation.emitOpError(
          "tensor shape region has no physical tile binding");
    else
      extents.push_back(label.getValue().str());
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

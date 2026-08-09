#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cutile::emission {
namespace {

FailureOr<std::string> tileSpelling(Operation *operation, StringRef role) {
  if (role == "one")
    return std::string("1");
  if (role == "row_vector")
    return std::string("TILE_SIZE");
  if (role.starts_with("row_vector_"))
    return "TILE_SIZE_V" + role.drop_front(11).str();
  if (role == "program_m" || role == "ragged_member" || role == "query")
    return std::string("TILE_SIZE_M");
  if (role.starts_with("ragged_member_"))
    return "TILE_SIZE_R" + role.drop_front(14).str();
  if (role.starts_with("query_"))
    return "TILE_SIZE_Q" + role.drop_front(6).str();
  if (role == "program_n" || role == "stream")
    return std::string("TILE_SIZE_N");
  if (role.starts_with("stream_"))
    return "TILE_SIZE_S" + role.drop_front(7).str();
  if (role.starts_with("program_"))
    return "TILE_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return std::string("TILE_SIZE_K");
  if (role.starts_with("reduction_"))
    return "TILE_SIZE_K" + role.drop_front(10).str();
  operation->emitOpError("has no cuTile tile spelling for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwiseSpelling(Operation *operation, StringRef role,
                                       StringRef resultSpace) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return resultSpace == "private_scalar" ? StringRef("ct.full_cast")
                                            : StringRef("ct.astype");
  if (role == "unary_exp")
    return StringRef("ct.exp");
  if (role == "unary_exp2")
    return StringRef("ct.exp2");
  if (role == "unary_log")
    return StringRef("ct.log");
  if (role == "unary_rsqrt")
    return StringRef("ct.rsqrt");
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
  if (role == "binary_maximum")
    return StringRef("ct.maximum");
  if (role == "compare_greater_equal")
    return StringRef("python_greater_equal");
  if (role == "mask")
    return StringRef("ct.where");
  if (role == "full")
    return StringRef("ct.full");
  if (role == "zeros")
    return StringRef("ct.zeros");
  if (role == "members")
    return StringRef("ct.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("ct.indirect_gather");
  operation->emitOpError("has no cuTile pointwise spelling for role ") << role;
  return failure();
}

FailureOr<std::string> parameterSpelling(Operation *operation, StringRef role) {
  if (role == "program_m" || role == "ragged_member" || role == "query")
    return std::string("TILE_SIZE_M");
  if (role.starts_with("ragged_member_"))
    return "TILE_SIZE_R" + role.drop_front(14).str();
  if (role.starts_with("query_"))
    return "TILE_SIZE_Q" + role.drop_front(6).str();
  if (role == "program_n" || role == "feature" || role == "stream")
    return std::string("TILE_SIZE_N");
  if (role.starts_with("stream_"))
    return "TILE_SIZE_S" + role.drop_front(7).str();
  if (role.starts_with("program_"))
    return "TILE_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return std::string("TILE_SIZE_K");
  if (role.starts_with("reduction_"))
    return "TILE_SIZE_K" + role.drop_front(10).str();
  if (role == "group_m")
    return std::string("GROUP_SIZE_M");
  if (role.starts_with("group_"))
    return "GROUP_SIZE_G" + role.drop_front(6).str();
  operation->emitOpError("has no cuTile tuner parameter for role ") << role;
  return failure();
}

bool workerReuse(const RealizationIndex &index) {
  return llvm::any_of(index.axesByRole, [](const auto &binding) {
    return binding.getValue().getReuseWorker();
  });
}

} // namespace

FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization,
                 const target::KernelModel &kernel) {
  RealizationIndex index;
  SmallVector<intent::plan::ReductionOp> reductions;
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
    } else if (auto value = dyn_cast<intent::plan::PointwiseOp>(operation)) {
      pointwise.push_back(value);
    } else if (auto value = dyn_cast<intent::plan::ContractOp>(operation)) {
      plan::ContractOp binding;
      binding.operation = value;
      binding.lowering = "ct.mma";
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
    plan::ReductionOp binding;
    binding.operation = value;
    binding.lowering = *role == "reduce_maximum" ? "ct.max" : "ct.sum";
    binding.resultSpace = value.getResultSpace().str();
    binding.axis = *axis;
    index.reductions[value.getNode()] = binding;
  }
  for (intent::plan::PointwiseOp value : pointwise) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    FailureOr<std::string> role =
        operation ? target::emission::pointwiseRole(*operation)
                  : FailureOr<std::string>(failure());
    FailureOr<StringRef> lowering =
        succeeded(role)
            ? pointwiseSpelling(value, *role, value.getResultSpace())
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
  for (intent::plan::TransferOp value : transfers) {
    Operation *operation = kernel.nodes.lookup(value.getNode());
    bool load = operation &&
                operation->getName().getStringRef() == "intent.view_load";
    bool store = operation &&
                 (operation->getName().getStringRef() == "intent.view_store" ||
                  operation->getName().getStringRef() == "intent.scatter_unique");
    if (!load && !store)
      return value.emitOpError("does not bind a canonical transfer");
    bool vectorized = llvm::any_of(value.getDomainNodes(), [&](int64_t node) {
      auto axis = index.axes.find(node);
      return axis != index.axes.end() && !axis->second.isScalar();
    });
    bool raggedBound = llvm::any_of(value.getDomainNodes(), [&](int64_t axis) {
      return target::emission::isRaggedBoundAxis(index.components, axis);
    });
    plan::BoundaryOp binding;
    binding.operation = value;
    binding.access = (rowStrided || raggedBound) && vectorized
                         ? (load ? "gather" : "scatter")
                         : (load ? "load" : "store");
    binding.resultSpace = value.getResultSpace().str();
    binding.defer = load && target::emission::feedsContraction(*operation) &&
                    (!index.components.groups.empty() ||
                     target::emission::feedsStagedContraction(index, *operation));
    binding.explicitBounds =
        rowStrided || raggedBound || (!index.stages.empty() && store);
    index.boundaries[value.getNode()] = binding;
  }
  if (!index.target || !index.program) {
    realization.emitOpError("lacks cuTile target or program choices");
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
    valueNames[argument.value] = argument.name;
  }
  if (views.empty())
    return kernel.entry.emitOpError("cuTile emitter requires external views");
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
          "cannot index its region shape against the cuTile plan");
    regionTiles["?region_" + std::to_string(argument.getInt()) + "_0"] =
        axis.getTile().str();
  }

  if (!planIndex.components.reusedAxes.empty()) {
    auto program = planIndex.axesByRole.find("program_0");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (program == planIndex.axesByRole.end() ||
        lane == planIndex.axesByRole.end())
      return realization.emitOpError(
          "persistent rows require program_0 and lane_0 choices");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if (view.view.getAccess() == "out" && !fixedOutput)
        fixedOutput = &view;
    }
    if (!fixedOutput || fixedOutput->tensor.getRank() < 1 ||
        fixedOutput->tensor.getRank() > 2 || searchSpace)
      return realization.emitOpError(
          "fixed persistent rows require one rank-one or rank-two output and no search space");
  }
  if ((!planIndex.components.groups.empty() || !planIndex.streams.empty() ||
       !planIndex.stages.empty()) &&
      (!searchSpace || !searchIndex.autotune))
    return realization.emitOpError(
        "tiled physical components require a delegated cuTile tuner");
  if (planIndex.components.reusedAxes.empty()) {
    for (const std::string &dimension : dimensionOrder) {
      bool physicalDimension = llvm::any_of(
          roleDimensions,
          [&](const auto &binding) { return binding.getValue() == dimension; });
      if (!physicalDimension)
        kernelConstants.push_back(dimension);
    }
  }
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
    Operation *memberSource = runtime.relation->getOperand(1).getDefiningOp();
    Operation *memberDim = memberSource && memberSource->getNumOperands() >= 2
                               ? memberSource->getOperand(1).getDefiningOp()
                               : nullptr;
    FailureOr<ABIView *> membersView =
        memberDim && memberDim->getName().getStringRef() == "intent.dim" &&
                memberDim->getNumOperands() == 1
            ? lookupView(memberDim->getOperand(0), *runtime.relation)
            : FailureOr<ABIView *>(failure());
    if (failed(offsets) || (*offsets)->tensor.getRank() != 1 ||
        failed(membersView))
      return runtime.relation->emitOpError(
          "requires canonical offsets and member-source views");
    runtime.offsets = *offsets;
    runtime.membersView = *membersView;
    if (runtime.relation->getNumOperands() == 4) {
      Operation *indicesLoad = runtime.relation->getOperand(3).getDefiningOp();
      FailureOr<ABIView *> indices =
          indicesLoad && indicesLoad->getNumOperands() == 1
              ? lookupView(indicesLoad->getOperand(0), *runtime.relation)
              : FailureOr<ABIView *>(failure());
      if (failed(indices) || (*indices)->tensor.getRank() != 1)
        return runtime.relation->emitOpError(
            "requires a canonical rank-one member-index view");
      runtime.indices = *indices;
    }
    bool ordered = llvm::any_of(ragged.getMemberNodes(), [&](int64_t node) {
      return planIndex.components.orderedRaggedAxes.contains(node);
    });
    if (ordered && runtime.indices)
      return runtime.relation->emitOpError(
          "ordered ragged traversal cannot project an indirect member map");
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
          "workspace_" + std::to_string(valueID);
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
  line("ct.scatter(" + workspaceNames.lookup(value) +
       ", (safe_member_offsets[:, None], offs_feature[None, :]), " +
       name.str() + ", check_bounds=True)");
}

void SourceEmitter::emitImports() {
  output << "import math\n";
  output << "import cuda.tile as ct\n";
  output << "import torch\n";
  if (!planIndex.components.reusedAxes.empty())
    output << "from intent.runtime.tuning.cutile import ROW_OCCUPANCY, row_configuration, row_program_count\n";
  if (searchSpace) {
    output << "from math import ceil\n";
    output << "from cuda.tile.tune import exhaustive_search\n";
    output << "from intent.runtime.tuning.cutile import autotune_configurations, autotune_timeout\n";
    output << "\n_PARAMETER_MAP = {";
    for (auto [index, mapping] :
         llvm::enumerate(searchIndex.autotune.getParameterMap())) {
      if (index)
        output << ", ";
      output << "'" << mapping.getName().getValue() << "': '"
             << cast<StringAttr>(mapping.getValue()).getValue() << "'";
    }
    output << "}\n_CONFIGS = autotune_configurations(_PARAMETER_MAP)\n";
    output << "_TUNE_TIMEOUT = autotune_timeout(_PARAMETER_MAP)\n";
  }
  output << "\nConstInt = ct.Constant[int]\n\n\n";
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  if (!planIndex.stages.empty()) {
    if (!searchIndex.autotune)
      return realization.emitOpError(
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
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          parameter(workspaceNames.lookup(kernel.values.lookup(valueID)));
      if (!compact)
        parameter("MAX_ROUTES: ConstInt");
      for (NamedAttribute config : searchIndex.autotune.getParameterMap())
        parameter(config.getName().getValue().str() + ": ConstInt");
      source << "):\n";
      source.flush();

      std::string feature = stageFeatureDimensions.lookup(stage);
      stageLine(stage, "bid_feature = ct.bid(" +
                           std::to_string(stageFeatureWorkers.lookup(stage)) +
                           ")");
      stageLine(stage, "bid_expert_route = ct.bid(" +
                           std::to_string(stageMemberWorkers.lookup(stage)) +
                           ")");
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
        stageLine(stage, "expert = 0");
        stageLine(stage, "route_tile = 0");
        stageLine(stage, "tile_cursor = 0");
        stageLine(stage, "for candidate in range(" + experts + "):");
        stageLine(stage, "candidate_begin = ct.load(" +
                             offsets->argument->name +
                             ", index=candidate, shape=())",
                  2);
        stageLine(stage, "candidate_end = ct.load(" +
                             offsets->argument->name +
                             ", index=candidate + 1, shape=())",
                  2);
        stageLine(stage,
                  "candidate_tiles = ct.cdiv(candidate_end - candidate_begin, "
                  "TILE_SIZE_M)",
                  2);
        stageLine(stage,
                  "owns_tile = (bid_expert_route >= tile_cursor) & "
                  "(bid_expert_route < tile_cursor + candidate_tiles)",
                  2);
        stageLine(stage, "expert = ct.where(owns_tile, candidate, expert)", 2);
        stageLine(stage,
                  "route_tile = ct.where(owns_tile, bid_expert_route - "
                  "tile_cursor, route_tile)",
                  2);
        stageLine(stage, "tile_cursor += candidate_tiles", 2);
      } else {
        stageLine(stage,
                  "num_route_tiles = ct.cdiv(MAX_ROUTES, TILE_SIZE_M)");
        stageLine(stage, "expert = bid_expert_route // num_route_tiles");
        stageLine(stage, "route_tile = bid_expert_route % num_route_tiles");
      }
      stageLine(stage, "route_begin = ct.load(" +
                           offsets->argument->name +
                           ", index=expert, shape=())");
      stageLine(stage, "route_end = ct.load(" +
                           offsets->argument->name +
                           ", index=expert + 1, shape=())");
      stageLine(stage,
                "member_offsets = route_begin + route_tile * TILE_SIZE_M + "
                "ct.arange(TILE_SIZE_M, dtype=ct.int32)");
      stageLine(stage, "member_mask = member_offsets < route_end");
      stageLine(stage,
                "safe_member_offsets = ct.where(member_mask, member_offsets, " +
                    stageMemberDimensions.lookup(stage) + ")");
      if (indices) {
        stageLine(stage, "routes = ct.gather(" + indices->argument->name +
                             ", member_offsets, check_bounds=True, "
                             "padding_value=0)");
        stageLine(stage, "routes = ct.where(member_mask, routes, 0)");
      } else {
        stageLine(stage, "routes = member_offsets");
      }
      stageLine(stage,
                "offs_feature = bid_feature * TILE_SIZE_N + "
                "ct.arange(TILE_SIZE_N, dtype=ct.int32)");
      stageLine(stage, "feature_mask = offs_feature < " + feature);
    }
    return success();
  }
  if (!planIndex.components.reusedAxes.empty()) {
    output << "@ct.kernel(occupancy=ROW_OCCUPANCY)\n";
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
  if (!planIndex.components.reusedAxes.empty()) {
    for (StringRef parameter : {"N_ROWS: ConstInt", "TILE_SIZE: ConstInt",
                                "DIM_COLS: ConstInt"})
      emitParameter(parameter);
  } else {
    for (const std::string &dimension : kernelConstants)
      emitParameter(dimension + ": ConstInt");
    for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
      emitParameter(parameter.getName().getValue().str() + ": ConstInt");
  }
  output << "):\n";
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
  auto torchDtype = [&](Type type) -> StringRef {
    if (type.isF16())
      return "torch.float16";
    if (type.isF32())
      return "torch.float32";
    if (type.isBF16())
      return "torch.bfloat16";
    if (auto integer = dyn_cast<IntegerType>(type);
        integer && integer.getWidth() == 32)
      return "torch.int32";
    return {};
  };
  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n";

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
             << " / cfg.TILE_SIZE_N), ";
      if (compact)
        output << "sum(ceil(length / cfg.TILE_SIZE_M) for length in "
                  "route_lengths_"
               << suffix << ")";
      else
        output << experts << " * ceil(max_routes_" << suffix
               << " / cfg.TILE_SIZE_M)";
      output << ", 1),\n";
      output << "                " << kernelName << "_stage_" << stage
             << ",\n";
      output << "                lambda cfg: (";
      for (ABIView &view : views) {
        if (planIndex.stages[stage].getOutputs().empty() && &view == merge)
          output << (target::emission::stageUsesScatterReduction(
                         planIndex.stages[stage], kernel)
                         ? "torch.zeros_like("
                         : "torch.empty_like(")
                 << view.argument->name << "), ";
        else
          output << view.argument->name << ", ";
      }
      for (ABIScalar &scalar : scalars)
        output << scalar.name << ", ";
      for (const std::string &dimension : dimensionOrder)
        output << dimension << ", ";
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
      if (!compact)
        output << "max_routes_" << suffix << ", ";
      for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
        output << "cfg." << parameter.getName().getValue() << ", ";
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
             << " / best.TILE_SIZE_N), ";
      if (compact)
        output << "sum(ceil(length / best.TILE_SIZE_M) for length in "
                  "route_lengths_"
               << suffix << ")";
      else
        output << experts << " * ceil(max_routes_" << suffix
               << " / best.TILE_SIZE_M)";
      output << ", 1)\n";
      output << "    ct.launch(stream, grid, tuned_kernel, (";
      for (ABIView &view : views)
        output << view.argument->name << ", ";
      for (ABIScalar &scalar : scalars)
        output << scalar.name << ", ";
      for (const std::string &dimension : dimensionOrder)
        output << dimension << ", ";
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          output << workspaceNames.lookup(kernel.values.lookup(valueID)) << ", ";
      if (!compact)
        output << "max_routes_" << suffix << ", ";
      for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
        output << "best." << parameter.getName().getValue() << ", ";
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
      if (view.view.getAccess() == "in")
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
    if (outputs.empty())
      return kernel.entry.emitOpError(
          "persistent-row wrapper requires an output view");
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
      output << "    if not " << view.argument->name << ".is_contiguous():\n";
      output << "        raise ValueError('persistent-row views must be contiguous')\n";
    }
    output << "    n_rows = "
           << dimensionOwners.lookup(roleDimensions.lookup("program_0")) << "\n";
    output << "    n_cols = "
           << dimensionOwners.lookup(roleDimensions.lookup("lane_0")) << "\n";
    output << "    configuration = row_configuration(n_cols)\n";
    output << "    num_programs = row_program_count(n_rows, _DEVICE, configuration.occupancy)\n";
    output << "    return ct.launch(torch.cuda.current_stream(), (num_programs, 1, 1), "
           << kernelName << ", (";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    for (ABIScalar &scalar : scalars)
      output << ", " << scalar.name;
    output << ", n_rows, configuration.tile_size, n_cols))\n\n\n";
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
    if (view.view.getAccess() == "in")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out")
      outputs.push_back(&view);
    else
      return kernel.entry.emitOpError(
          "autotuned cuTile wrapper supports input and output views");
  }
  if (outputs.empty())
    return kernel.entry.emitOpError("autotuned cuTile wrapper has no output views");

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
  output << "    stream = torch.cuda.current_stream()\n";
  SmallVector<plan::AxisOp> programAxes =
      target::emission::orderedProgramAxes(planIndex);
  SmallVector<plan::AxisOp> dynamicRaggedAxes;
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
    dynamicRaggedAxes.push_back(axis);
  }
  output << "    cache_key = (";
  for (auto [index, dimension] : llvm::enumerate(dimensionOrder)) {
    if (index)
      output << ", ";
    output << dimension;
  }
  for (plan::AxisOp axis : dynamicRaggedAxes)
    output << ", max_member_length_" << axis.getNode();
  for (ABIView *input : inputs)
    output << ", " << input->argument->name << ".dtype";
  output << ", str(_DEVICE))\n";
  output << "    if cache_key not in _TUNE_CACHE:\n";
  output << "        with ct.compiler_timeout(_TUNE_TIMEOUT):\n";
  output << "            result = exhaustive_search(\n";
  output << "                _CONFIGS,\n                stream,\n";
  std::array<std::string, 3> candidateGrid =
      target::emission::projectProgramGrid(
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
  output << "                lambda cfg: (" << candidateGrid[0] << ", "
         << candidateGrid[1] << ", " << candidateGrid[2] << "),\n";
  output << "                " << kernelName << ",\n";
  output << "                lambda cfg: (";
  for (ABIView &view : views)
    output << view.argument->name << ", ";
  for (ABIScalar &scalar : scalars)
    output << scalar.name << ", ";
  for (const std::string &dimension : kernelConstants)
    output << dimension << ", ";
  for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
    output << "cfg." << parameter.getName().getValue() << ", ";
  output << "),\n";
  output << "                lambda cfg: {'num_ctas': cfg.num_ctas, 'occupancy': cfg.occupancy},\n";
  output << "            )\n";
  output << "        best = result.best.config\n";
  output << "        _TUNE_CACHE[cache_key] = (best, " << kernelName
         << ".replace_hints(num_ctas=best.num_ctas, occupancy=best.occupancy))\n";
  output << "    best, tuned_kernel = _TUNE_CACHE[cache_key]\n";
  std::array<std::string, 3> selectedGrid =
      target::emission::projectProgramGrid(
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
  output << "    grid = (" << selectedGrid[0] << ", " << selectedGrid[1]
         << ", " << selectedGrid[2] << ")\n";
  output << "    return ct.launch(stream, grid, tuned_kernel, (";
  for (ABIView &view : views)
    output << view.argument->name << ", ";
  for (ABIScalar &scalar : scalars)
    output << scalar.name << ", ";
  for (const std::string &dimension : kernelConstants)
    output << dimension << ", ";
  for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
    output << "best." << parameter.getName().getValue() << ", ";
  output << "))\n\n\n";
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

FailureOr<StringRef> SourceEmitter::lookupValue(Operation &consumer,
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
  if (Operation *definition = indexedValue.getDefiningOp())
    if (definition->getName().getStringRef() == "intent.domain" ||
        definition->getName().getStringRef() == "intent.ragged_outer" ||
        definition->getName().getStringRef() == "intent.ragged_member")
      return definition;
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
    consumer.emitOpError("cannot resolve a cuTile stream source domain");
    return failure();
  }
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("cannot resolve index ownership during cuTile emission");
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
  plan::AxisOp axis =
      succeeded(node) ? planIndex.axes.lookup(*node) : plan::AxisOp();
  if (failed(node) || !axis) {
    consumer.emitOpError("indexes a domain without a cuTile axis binding");
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
  auto axis = dim ? dim->getAttrOfType<IntegerAttr>("intent.axis")
                  : IntegerAttr();
  if (!dim || dim->getName().getStringRef() != "intent.dim" || !axis ||
      dim->getNumOperands() != 1)
    return failure();
  FailureOr<ABIView *> view = lookupView(dim->getOperand(0), domain);
  if (failed(view) || axis.getInt() < 0 ||
      static_cast<size_t>(axis.getInt()) >= (*view)->shape.size())
    return failure();
  return (*view)->shape[axis.getInt()];
}

FailureOr<std::string> SourceEmitter::indexTuple(Operation &operation,
                                                 bool reductionLoop) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  bool raggedMatrix = false;
  if (relation->size() == 2 &&
      ((*relation)[0].kind == "region_index" ||
       (*relation)[0].kind == "value_index") &&
      (*relation)[0].operands.size() == 1 &&
      (*relation)[0].operands.front()) {
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*(*relation)[0].operands.front()),
                    operation);
    raggedMatrix = succeeded(axis) && target::emission::isRaggedBoundAxis(
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
    return "(" + rows + "[:, None], ct.arange(" +
           (*view)->shape[1] + ", dtype=ct.int32)[None, :])";
  }
  SmallVector<std::string> indices;
  for (const target::IndexTerm &term : *relation) {
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
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
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
    indices.push_back(std::move(index));
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

FailureOr<std::string> SourceEmitter::tileShape(Operation &operation) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view) || relation->size() != (*view)->shape.size())
    return failure();
  SmallVector<std::string> extents;
  for (auto [axisNumber, term] : llvm::enumerate(*relation)) {
    if (term.kind == "full_slice") {
      extents.push_back((*view)->shape[axisNumber]);
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
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
    if (failed(axis))
      return failure();
    extents.push_back(axis->getTile().str());
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

FailureOr<std::string> SourceEmitter::emitValidityExpression(
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
    std::string base = axisIndices.lookup(axis.getNode());
    if (base.empty())
      return consumer.emitOpError(
          "has no active cuTile index for its planned validity axis");
    bool direct = target::emission::isRaggedBoundAxis(
                      planIndex.components, axis.getNode()) ||
                  (!planIndex.components.reusedAxes.empty() &&
                   kernel.nodes.lookup(axis.getNode()) == vectorDomain);
    std::string index = direct
                            ? base
                            : base + " * " + axis.getTile().str() +
                                  " + ct.arange(" + axis.getTile().str() +
                                  ", dtype=ct.int32)";
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
    if (target::emission::isRaggedBoundAxis(planIndex.components,
                                            axis.getNode())) {
      FailureOr<int64_t> ordered = target::emission::representativeOrderedAxis(
          planIndex, axis.getNode(), consumer);
      if (failed(ordered))
        return failure();
      extent = "sequence_end_" + std::to_string(*ordered);
    }
    if (failed(extent))
      return failure();
    if (!target::emission::isRaggedBoundAxis(planIndex.components,
                                             axis.getNode()))
      *extent = dimensionOwners.lookup(*extent);
    predicates.push_back("(" + index + " < " + *extent + ")");
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
  StringRef fill = padding.getFill() == "negative_infinity"
                       ? StringRef("-math.inf")
                       : StringRef("0.0");
  return "ct.where(" + *predicate + ", " + expression.str() + ", " +
         fill.str() + ")";
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
          "tensor shape region has no cuTile tile binding");
    else
      extents.push_back(label.getValue().str());
  }
  if (target::emission::touchesStateStream(operation) && extents.size() == 1 &&
      extents.front() == "TILE_SIZE_M")
    extents.push_back("1");
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

std::string SourceEmitter::dtypeName(Type type, Operation &consumer) {
  if (type.isF16())
    return "ct.float16";
  if (type.isF32())
    return "ct.float32";
  if (type.isBF16())
    return "ct.bfloat16";
  consumer.emitOpError("uses an unsupported cuTile dtype");
  return {};
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

void SourceEmitter::line(StringRef text) {
  if (!planIndex.stages.empty()) {
    for (unsigned stage : activeStages)
      stageLine(stage, text, indentation);
    return;
  }
  output.indent(indentation * 4) << text << "\n";
}

LogicalResult emitRealizedKernelSource(
    target::KernelModel kernel, intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization, kernel);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return SourceEmitter(std::move(kernel), realization, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .emit();
}

} // namespace intent::cutile::emission

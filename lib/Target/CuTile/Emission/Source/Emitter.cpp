#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cutile::emission {

FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization) {
  RealizationIndex index;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (auto value = dyn_cast<plan::TargetOp>(operation))
      index.target = value;
    else if (auto value = dyn_cast<plan::AxisOp>(operation)) {
      index.axes[value.getNode()] = value;
      index.axesByRole[value.getRole()] = value;
    } else if (auto value = dyn_cast<plan::ProgramOp>(operation))
      index.program = value;
    else if (auto value = dyn_cast<plan::LaunchOp>(operation))
      index.launch = value;
    else if (auto value = dyn_cast<plan::StorageOp>(operation))
      index.storage[value.getValue()] = value;
    else if (auto value = dyn_cast<plan::LayoutOp>(operation))
      index.layouts[value.getValue()] = value;
    else if (auto value = dyn_cast<plan::ReductionOp>(operation))
      index.reductions[value.getNode()] = value;
    else if (auto value = dyn_cast<plan::PointwiseOp>(operation))
      index.pointwise[value.getNode()] = value;
    else if (auto value = dyn_cast<plan::ContractOp>(operation))
      index.contracts[value.getNode()] = value;
    else if (auto value = dyn_cast<plan::StreamOp>(operation))
      index.streams[value.getNode()] = value;
    else if (auto value = dyn_cast<plan::BoundaryOp>(operation))
      index.boundaries[value.getNode()] = value;
    else if (auto value = dyn_cast<plan::RaggedOp>(operation))
      index.ragged = value;
    else if (auto value = dyn_cast<plan::StageOp>(operation))
      index.stages.push_back(value);
    else if (auto value = dyn_cast<plan::AtomicOp>(operation))
      index.atomics[value.getNode()] = value;
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
    if (auto autotune = dyn_cast<plan::AutotuneOp>(operation))
      index.autotune = autotune;
    else if (auto config = dyn_cast<plan::ConfigOp>(operation))
      index.configs.push_back(config);
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
  if (failed(resolvePhysicalBindings()))
    return failure();
  if (isRaggedStages())
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
    if (!planIndex.storage.count(argument.valueID) ||
        !planIndex.layouts.count(argument.valueID))
      return kernel.entry.emitOpError()
             << "ABI value " << argument.valueID
             << " lacks cuTile storage/layout realization";
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
  bool multiAxis = planIndex.program.getMapping() == "multi_axis_stream";
  bool raggedStages = isRaggedStages();
  if ((!multiAxis && !raggedStages &&
       planIndex.program.getWorkerAxes().size() != 1) ||
      ((multiAxis || raggedStages) &&
       planIndex.program.getWorkerAxes().size() != 2))
    return planIndex.program.emitOpError(
        "block-axis count does not match the cuTile program mapping");
  for (auto &entry : planIndex.axes) {
    Operation *domain = kernel.nodes.lookup(entry.first);
    if (!domain ||
        (domain->getName().getStringRef() != "intent.domain" &&
         domain->getName().getStringRef() != "intent.ragged_outer" &&
         domain->getName().getStringRef() != "intent.ragged_member"))
      return entry.second.emitOpError("does not bind a logical domain op");
    FailureOr<std::string> dimension = dimensionName(*domain);
    if (failed(dimension))
      return entry.second.emitOpError("cannot resolve its source dimension");
    roleDimensions[entry.second.getRole()] = *dimension;
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

  if (planIndex.program.getMapping() == "persistent_rows") {
    auto program = planIndex.axesByRole.find("program_0");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (program == planIndex.axesByRole.end() ||
        lane == planIndex.axesByRole.end() || !planIndex.launch)
      return realization.emitOpError(
          "persistent rows require program_0, lane_0 and launch choices");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if (view.view.getAccess() == "out" && !fixedOutput)
        fixedOutput = &view;
      if (view.tensor.getRank() != 2)
        return kernel.entry.emitOpError(
            "persistent-row cuTile views must be rank two");
    }
    if (!fixedOutput || searchSpace)
      return realization.emitOpError(
          "fixed persistent rows require one output and no search space");
    if (planIndex.launch.getGridPolicy() != "static_persistent")
      return planIndex.launch.emitOpError(
          "has no persistent-row cuTile source emitter");
  } else if (planIndex.program.getMapping() == "grouped_2d_tiles") {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty())
      return realization.emitOpError(
          "grouped cuTile program requires backend autotune candidates");
    if (planIndex.launch)
      return realization.emitOpError(
          "autotuned grouped cuTile program cannot carry fixed launch hints");
    for (StringRef role : {"program_0", "program_1", "reduction_0"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "grouped cuTile program lacks " << role << " axis";
    SmallVector<StringRef> expectedKeys = {
        roleDimensions.lookup("program_0"), roleDimensions.lookup("program_1"),
        roleDimensions.lookup("reduction_0")};
    if (searchIndex.autotune.getKey().size() != expectedKeys.size())
      return searchIndex.autotune.emitOpError(
          "does not specialize every tiled dimension");
    for (auto [attribute, expected] :
         llvm::zip(searchIndex.autotune.getKey(), expectedKeys))
      if (cast<StringAttr>(attribute).getValue() != expected)
        return searchIndex.autotune.emitOpError(
            "key order does not match program/reduction roles");
  } else if (planIndex.program.getMapping() == "multi_axis_stream") {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty())
      return realization.emitOpError(
          "ordered cuTile stream requires backend autotune candidates");
    if (planIndex.launch || planIndex.streams.size() != 1)
      return realization.emitOpError(
          "ordered cuTile stream requires one stream and no fixed launch");
    for (StringRef role : {"program_0", "program_1", "program_2", "stream_0"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "ordered cuTile stream lacks " << role << " axis";
    SmallVector<StringRef> expectedKeys = {
        roleDimensions.lookup("program_2"),
        roleDimensions.lookup("stream_0")};
    if (searchIndex.autotune.getKey().size() != expectedKeys.size())
      return searchIndex.autotune.emitOpError(
          "does not specialize the tiled program and stream dimensions");
    for (auto [attribute, expected] :
         llvm::zip(searchIndex.autotune.getKey(), expectedKeys))
      if (cast<StringAttr>(attribute).getValue() != expected)
        return searchIndex.autotune.emitOpError(
            "key order does not match program_2/stream_0");
    kernelConstants.push_back(roleDimensions.lookup("program_1"));
    for (const std::string &dimension : dimensionOrder) {
      bool physicalDimension = llvm::any_of(
          roleDimensions,
          [&](const auto &binding) { return binding.getValue() == dimension; });
      if (!physicalDimension)
        kernelConstants.push_back(dimension);
    }
  } else if (raggedStages) {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty() ||
        !planIndex.ragged || planIndex.stages.empty() ||
        planIndex.atomics.empty())
      return realization.emitOpError(
          "ragged staging requires target traversal, stages, atomic merge, and backend candidates");
    for (StringRef role : {"program_0", "program_1"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "ragged staging lacks " << role << " axis";
    if (planIndex.launch)
      return realization.emitOpError(
          "ragged staging cannot carry a fixed launch choice");
  } else {
    return planIndex.program.emitOpError("has no cuTile program emitter");
  }
  return success();
}

LogicalResult SourceEmitter::prepareRaggedStages() {
  raggedRelation = kernel.nodes.lookup(planIndex.ragged.getNode());
  raggedOuter = kernel.nodes.lookup(planIndex.ragged.getOuterNode());
  raggedMember = kernel.nodes.lookup(planIndex.ragged.getMemberNode());
  if (!raggedRelation ||
      raggedRelation->getName().getStringRef() != "intent.ragged" ||
      !raggedOuter ||
      raggedOuter->getName().getStringRef() != "intent.ragged_outer" ||
      !raggedMember ||
      raggedMember->getName().getStringRef() != "intent.ragged_member")
    return planIndex.ragged.emitOpError(
        "does not resolve to canonical ragged operations");

  kernel.entry.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() == "intent.members" &&
        !membersOperation)
      membersOperation = operation;
  });
  if (!membersOperation)
    return raggedRelation->emitOpError("has no member enumeration operation");

  llvm::sort(planIndex.stages, [](plan::StageOp lhs, plan::StageOp rhs) {
    return lhs.getOrdinal() < rhs.getOrdinal();
  });
  stageBodies.resize(planIndex.stages.size());
  auto shapeLabel = [&](Value value, unsigned axis,
                        Operation &consumer) -> FailureOr<std::string> {
    auto result = dyn_cast<OpResult>(value);
    Operation *definition = result ? result.getOwner() : nullptr;
    auto shapes = definition
                      ? definition->getAttrOfType<ArrayAttr>(
                            "intent.result_shapes")
                      : ArrayAttr();
    auto shape = shapes && result.getResultNumber() < shapes.size()
                     ? dyn_cast<ArrayAttr>(shapes[result.getResultNumber()])
                     : ArrayAttr();
    auto label = shape && axis < shape.size() ? dyn_cast<StringAttr>(shape[axis])
                                             : StringAttr();
    if (!label || label.getValue().empty()) {
      consumer.emitOpError("stage axis has no canonical symbolic extent");
      return failure();
    }
    return label.getValue().str();
  };

  for (auto [position, stage] : llvm::enumerate(planIndex.stages)) {
    if (stage.getOrdinal() != position)
      return stage.emitOpError("stage ordinals must be contiguous from zero");
    Operation *contract = kernel.nodes.lookup(stage.getNode());
    if (!contract || contract->getName().getStringRef() != "intent.contract" ||
        contract->getNumResults() != 1)
      return stage.emitOpError("does not bind one canonical contraction");
    auto resultType = dyn_cast<RankedTensorType>(contract->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2)
      return stage.emitOpError("requires a rank-two staged contraction result");
    FailureOr<std::string> feature =
        shapeLabel(contract->getResult(0), 1, *contract);
    auto reduce = contract->getAttrOfType<ArrayAttr>("intent.reduce");
    auto pair = reduce && reduce.size() == 1
                    ? dyn_cast<ArrayAttr>(reduce[0])
                    : ArrayAttr();
    auto lhsAxis = pair && pair.size() == 2
                       ? dyn_cast<IntegerAttr>(pair[0])
                       : IntegerAttr();
    FailureOr<std::string> reduction =
        lhsAxis && lhsAxis.getInt() >= 0
            ? shapeLabel(contract->getOperand(0), lhsAxis.getInt(), *contract)
            : FailureOr<std::string>(failure());
    if (failed(feature) || failed(reduction))
      return failure();
    stageFeatureDimensions[position] = *feature;
    stageReductionDimensions[position] = *reduction;

    llvm::DenseSet<Value> inputs;
    for (int64_t valueID : stage.getInputs()) {
      auto found = kernel.values.find(valueID);
      if (found == kernel.values.end())
        return stage.emitOpError("references an unknown stage input value");
      inputs.insert(found->second);
    }
    if (!stage.getOutputs().empty()) {
      for (int64_t valueID : stage.getOutputs()) {
        auto found = kernel.values.find(valueID);
        if (found == kernel.values.end() ||
            !isa<RankedTensorType>(found->second.getType()) ||
            !stageOutputOwners.try_emplace(found->second, position).second)
          return stage.emitOpError("has an invalid stage output value");
        workspaceNames[found->second] =
            "workspace_" + std::to_string(valueID);
        llvm::DenseSet<Value> visited;
        collectStageValue(found->second, position, inputs, visited);
      }
    } else {
      for (const auto &binding : planIndex.atomics) {
        Operation *atomic = kernel.nodes.lookup(binding.first);
        if (!atomic)
          return raggedRelation->emitOpError(
              "has a plan atomic without a canonical operation");
        operationStages[atomic].push_back(position);
        for (Value operand : atomic->getOperands()) {
          llvm::DenseSet<Value> visited;
          collectStageValue(operand, position, inputs, visited);
        }
      }
    }
    if (!llvm::is_contained(operationStages.lookup(contract), position))
      return stage.emitOpError("stage roots do not depend on its contraction");
  }
  return success();
}

void SourceEmitter::collectStageValue(Value value, unsigned stage,
                                      const llvm::DenseSet<Value> &inputs,
                                      llvm::DenseSet<Value> &visited) {
  if (inputs.contains(value) || !visited.insert(value).second)
    return;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  auto &stages = operationStages[definition];
  if (!llvm::is_contained(stages, stage))
    stages.push_back(stage);
  for (Value operand : definition->getOperands())
    collectStageValue(operand, stage, inputs, visited);
}

bool SourceEmitter::selectOperation(Operation &operation) {
  if (!isRaggedStages())
    return true;
  activeStages = operationStages.lookup(&operation);
  return !activeStages.empty();
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
  if (!isRaggedStages() || outputStage == stageOutputOwners.end())
    return;
  line("ct.scatter(" + workspaceNames.lookup(value) +
       ", (safe_member_offsets[:, None], offs_feature[None, :]), " +
       name.str() + ", check_bounds=True)");
}

void SourceEmitter::emitImports() {
  output << "import math\n";
  output << "import cuda.tile as ct\n";
  output << "import torch\n";
  if (searchSpace) {
    output << "from math import ceil\n";
    output << "from types import SimpleNamespace\n";
    output << "from cuda.tile.tune import exhaustive_search\n";
  }
  output << "\nConstInt = ct.Constant[int]\n\n\n";
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  if (isRaggedStages()) {
    Operation *offsetsLoad = raggedRelation->getOperand(1).getDefiningOp();
    Operation *indicesLoad = raggedRelation->getOperand(2).getDefiningOp();
    FailureOr<ABIView *> offsets =
        offsetsLoad && offsetsLoad->getNumOperands() == 1
            ? lookupView(offsetsLoad->getOperand(0), *raggedRelation)
            : FailureOr<ABIView *>(failure());
    FailureOr<ABIView *> indices =
        indicesLoad && indicesLoad->getNumOperands() == 1
            ? lookupView(indicesLoad->getOperand(0), *raggedRelation)
            : FailureOr<ABIView *>(failure());
    if (failed(offsets) || failed(indices) || searchIndex.configs.empty())
      return raggedRelation->emitOpError(
          "cannot resolve staged ragged metadata or search candidates");

    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
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
      parameter("MAX_ROUTES: ConstInt");
      for (NamedAttribute config : searchIndex.configs.front().getParameters())
        parameter(config.getName().getValue().str() + ": ConstInt");
      source << "):\n";
      source.flush();

      std::string feature = stageFeatureDimensions.lookup(stage);
      stageLine(stage, "bid_feature = ct.bid(0)");
      stageLine(stage, "bid_expert_route = ct.bid(1)");
      stageLine(stage,
                "num_route_tiles = ct.cdiv(MAX_ROUTES, TILE_SIZE_M)");
      stageLine(stage, "expert = bid_expert_route // num_route_tiles");
      stageLine(stage, "route_tile = bid_expert_route % num_route_tiles");
      stageLine(stage, "route_begin = ct.load(" +
                           (*offsets)->argument->name +
                           ", index=expert, shape=())");
      stageLine(stage, "route_end = ct.load(" +
                           (*offsets)->argument->name +
                           ", index=expert + 1, shape=())");
      stageLine(stage,
                "member_offsets = route_begin + route_tile * TILE_SIZE_M + "
                "ct.arange(TILE_SIZE_M, dtype=ct.int32)");
      stageLine(stage, "member_mask = member_offsets < route_end");
      stageLine(stage,
                "safe_member_offsets = ct.where(member_mask, member_offsets, " +
                    (*indices)->argument->name + ".shape[0])");
      stageLine(stage, "routes = ct.gather(" + (*indices)->argument->name +
                           ", member_offsets, check_bounds=True, "
                           "padding_value=0)");
      stageLine(stage, "routes = ct.where(member_mask, routes, 0)");
      stageLine(stage,
                "offs_feature = bid_feature * TILE_SIZE_N + "
                "ct.arange(TILE_SIZE_N, dtype=ct.int32)");
      stageLine(stage, "feature_mask = offs_feature < " + feature);
    }
    return success();
  }
  if (planIndex.program.getMapping() == "persistent_rows") {
    output << "@ct.kernel(occupancy=" << planIndex.launch.getOccupancy()
           << ")\n";
    programIndex = makeRegionArgumentName(*programRoot, 0);
    vectorIndex = makeResultName(*vectorDomain, 0);
    valueNames[programRoot->getRegion(0).front().getArgument(0)] = programIndex;
    valueNames[vectorDomain->getResult(0)] = vectorIndex;
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
  if (planIndex.program.getMapping() == "persistent_rows") {
    for (StringRef parameter : {"N_ROWS: ConstInt", "TILE_SIZE: ConstInt",
                                "DIM_COLS: ConstInt"})
      emitParameter(parameter);
  } else {
    for (const std::string &dimension : kernelConstants)
      emitParameter(dimension + ": ConstInt");
    if (searchIndex.configs.empty())
      return realization.emitOpError(
          "autotuned cuTile emission has no physical configurations");
    for (NamedAttribute parameter : searchIndex.configs.front().getParameters())
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

  if (isRaggedStages()) {
    SmallVector<ABIView *> inputs;
    ABIView *merge = nullptr;
    for (ABIView &view : views) {
      if (view.view.getAccess() == "in")
        inputs.push_back(&view);
      else if (view.view.getAccess() == "inout" && !merge)
        merge = &view;
      else
        return kernel.entry.emitOpError(
            "ragged stages require input views and one inout merge view");
    }
    if (!merge)
      return kernel.entry.emitOpError("ragged stages have no merge destination");
    for (const std::string &body : stageBodies)
      output << "\n" << body;

    output << "\n_CONFIGS = (\n";
    for (plan::ConfigOp config : searchIndex.configs) {
      output << "    SimpleNamespace(";
      for (NamedAttribute parameter : config.getParameters())
        output << parameter.getName().getValue() << "="
               << cast<IntegerAttr>(parameter.getValue()).getInt() << ", ";
      output << "num_ctas=" << config.getNumCtas()
             << ", occupancy=" << config.getOccupancy() << "),\n";
    }
    output << ")\n_TUNE_CACHE = {}\n\n\n";

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
    Operation *offsetsLoad = raggedRelation->getOperand(1).getDefiningOp();
    FailureOr<ABIView *> offsets =
        offsetsLoad && offsetsLoad->getNumOperands() == 1
            ? lookupView(offsetsLoad->getOperand(0), *raggedRelation)
            : FailureOr<ABIView *>(failure());
    if (failed(offsets))
      return failure();
    output << "    max_routes = int((" << (*offsets)->argument->name
           << "[1:] - " << (*offsets)->argument->name
           << "[:-1]).max().item())\n";
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        Value value = kernel.values.lookup(valueID);
        auto tensor = dyn_cast<RankedTensorType>(value.getType());
        StringRef dtype =
            tensor ? torchDtype(tensor.getElementType()) : StringRef();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported workspace tensor");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty((R, "
               << stageFeatureDimensions.lookup(stage.getOrdinal())
               << "), device=_DEVICE, dtype=" << dtype << ")\n";
      }
    output << "    stream = torch.cuda.current_stream()\n";
    std::string experts = roleDimensions.lookup("program_0");
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      std::string feature = stageFeatureDimensions.lookup(stage);
      output << "    cache_key_" << stage << " = (" << stage;
      for (const std::string &dimension : dimensionOrder)
        output << ", " << dimension;
      output << ", " << inputs.front()->argument->name
             << ".dtype, str(_DEVICE))\n";
      output << "    if cache_key_" << stage << " not in _TUNE_CACHE:\n";
      output << "        with ct.compiler_timeout(10):\n";
      output << "            result = exhaustive_search(\n";
      output << "                _CONFIGS,\n                stream,\n";
      output << "                lambda cfg: (ceil(" << feature
             << " / cfg.TILE_SIZE_N), " << experts
             << " * ceil(max_routes / cfg.TILE_SIZE_M), 1),\n";
      output << "                " << kernelName << "_stage_" << stage
             << ",\n";
      output << "                lambda cfg: (";
      for (ABIView &view : views) {
        if (planIndex.stages[stage].getOutputs().empty() && &view == merge)
          output << "torch.zeros_like(" << view.argument->name << "), ";
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
      output << "max_routes, ";
      for (NamedAttribute parameter : searchIndex.configs.front().getParameters())
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
             << " / best.TILE_SIZE_N), " << experts
             << " * ceil(max_routes / best.TILE_SIZE_M), 1)\n";
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
      output << "max_routes, ";
      for (NamedAttribute parameter : searchIndex.configs.front().getParameters())
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
    output << "    " << merge->argument->name << " = torch.zeros((";
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

  if (planIndex.program.getMapping() == "persistent_rows") {
    SmallVector<ABIView *> inputs;
    for (ABIView &view : views)
      if (view.view.getAccess() == "in")
        inputs.push_back(&view);
      else if (view.view.getAccess() != "out" || &view != fixedOutput)
        return kernel.entry.emitOpError(
            "persistent-row wrapper supports inputs and one output");
    if (inputs.empty())
      return kernel.entry.emitOpError(
          "persistent-row wrapper requires an input view");
    ABIView *shapeOwner = inputs.front();
    output << "_OCCUPANCY = " << planIndex.launch.getOccupancy() << "\n\n\n";
    output << "def launch(";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    output << "):\n";
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
      output << "    if " << view.argument->name << ".ndim != 2";
      if (&view != shapeOwner)
        output << " or " << view.argument->name << ".shape != "
               << shapeOwner->argument->name << ".shape";
      output << ":\n";
      output << "        raise ValueError('persistent-row views require one rank-two shape')\n";
      output << "    if not " << view.argument->name << ".is_contiguous():\n";
      output << "        raise ValueError('persistent-row views must be contiguous')\n";
    }
    output << "    n_rows, n_cols = " << shapeOwner->argument->name << ".shape\n";
    output << "    tile_size = 1 << (n_cols - 1).bit_length()\n";
    output << "    num_sms = torch.cuda.get_device_properties(_DEVICE).multi_processor_count\n";
    output << "    num_programs = min(num_sms * _OCCUPANCY, n_rows)\n";
    output << "    return ct.launch(torch.cuda.current_stream(), (num_programs, 1, 1), "
           << kernelName << ", (";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    output << ", n_rows, tile_size, n_cols))\n\n\n";
    output << "def run(";
    for (auto [index, view] : llvm::enumerate(inputs)) {
      if (index)
        output << ", ";
      output << view->argument->name;
    }
    output << "):\n";
    StringRef outputDtype = torchDtype(fixedOutput->tensor.getElementType());
    output << "    " << fixedOutput->argument->name << " = torch.empty_like("
           << shapeOwner->argument->name << ", dtype=" << outputDtype << ")\n";
    output << "    launch(";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    output << ")\n    return " << fixedOutput->argument->name << "\n";
    return success();
  }

  SmallVector<ABIView *> inputs;
  ABIView *outputView = nullptr;
  for (ABIView &view : views) {
    if (view.view.getAccess() == "in")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out" && !outputView)
      outputView = &view;
    else
      return kernel.entry.emitOpError(
          "autotuned cuTile wrapper supports inputs and one output");
  }
  if (!outputView)
    return kernel.entry.emitOpError("autotuned cuTile wrapper has no output view");

  output << "_CONFIGS = (\n";
  for (plan::ConfigOp config : searchIndex.configs) {
    output << "    SimpleNamespace(";
    for (NamedAttribute parameter : config.getParameters())
      output << parameter.getName().getValue() << "="
             << cast<IntegerAttr>(parameter.getValue()).getInt() << ", ";
    output << "num_ctas=" << config.getNumCtas()
           << ", occupancy=" << config.getOccupancy() << "),\n";
  }
  output << ")\n_TUNE_CACHE = {}\n\n\n";
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
    output << "):\n";
    output << "        raise ValueError('" << view.argument->name
           << " shape violates the kernel symbols')\n";
  }
  output << "    stream = torch.cuda.current_stream()\n";
  output << "    cache_key = (";
  for (auto [index, dimension] : llvm::enumerate(dimensionOrder)) {
    if (index)
      output << ", ";
    output << dimension;
  }
  output << ", " << inputs.front()->argument->name << ".dtype, str(_DEVICE))\n";
  output << "    if cache_key not in _TUNE_CACHE:\n";
  output << "        with ct.compiler_timeout("
         << (planIndex.program.getMapping() == "multi_axis_stream" ? 10 : 5)
         << "):\n";
  output << "            result = exhaustive_search(\n";
  output << "                _CONFIGS,\n                stream,\n";
  if (planIndex.program.getMapping() == "grouped_2d_tiles") {
    output << "                lambda cfg: (ceil("
           << roleDimensions.lookup("program_0")
           << " / cfg.TILE_SIZE_M) * ceil("
           << roleDimensions.lookup("program_1")
           << " / cfg.TILE_SIZE_N), 1, 1),\n";
  } else if (planIndex.program.getMapping() == "multi_axis_stream") {
    output << "                lambda cfg: (ceil("
           << roleDimensions.lookup("program_2")
           << " / cfg.TILE_SIZE_M), "
           << roleDimensions.lookup("program_0") << " * "
           << roleDimensions.lookup("program_1") << ", 1),\n";
  } else {
    return planIndex.program.emitOpError(
        "has no autotuned cuTile grid emitter");
  }
  output << "                " << kernelName << ",\n";
  output << "                lambda cfg: (";
  for (ABIView &view : views)
    output << view.argument->name << ", ";
  for (ABIScalar &scalar : scalars)
    output << scalar.name << ", ";
  for (const std::string &dimension : kernelConstants)
    output << dimension << ", ";
  for (NamedAttribute parameter : searchIndex.configs.front().getParameters())
    output << "cfg." << parameter.getName().getValue() << ", ";
  output << "),\n";
  output << "                lambda cfg: {'num_ctas': cfg.num_ctas, 'occupancy': cfg.occupancy},\n";
  output << "            )\n";
  output << "        best = result.best.config\n";
  output << "        _TUNE_CACHE[cache_key] = (best, " << kernelName
         << ".replace_hints(num_ctas=best.num_ctas, occupancy=best.occupancy))\n";
  output << "    best, tuned_kernel = _TUNE_CACHE[cache_key]\n";
  if (planIndex.program.getMapping() == "grouped_2d_tiles")
    output << "    grid = (ceil(" << roleDimensions.lookup("program_0")
           << " / best.TILE_SIZE_M) * ceil("
           << roleDimensions.lookup("program_1")
           << " / best.TILE_SIZE_N), 1, 1)\n";
  else
    output << "    grid = (ceil(" << roleDimensions.lookup("program_2")
           << " / best.TILE_SIZE_M), "
           << roleDimensions.lookup("program_0") << " * "
           << roleDimensions.lookup("program_1") << ", 1)\n";
  output << "    return ct.launch(stream, grid, tuned_kernel, (";
  for (ABIView &view : views)
    output << view.argument->name << ", ";
  for (ABIScalar &scalar : scalars)
    output << scalar.name << ", ";
  for (const std::string &dimension : kernelConstants)
    output << dimension << ", ";
  for (NamedAttribute parameter : searchIndex.configs.front().getParameters())
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
  StringRef outputDtype = torchDtype(outputView->tensor.getElementType());
  output << "    " << outputView->argument->name << " = torch.empty((";
  for (auto [axis, extent] : llvm::enumerate(outputView->shape)) {
    if (axis)
      output << ", ";
    output << extent;
  }
  output << "), device=_DEVICE, dtype=" << outputDtype << ")\n";
  output << "    launch(";
  for (auto [index, view] : llvm::enumerate(views)) {
    if (index)
      output << ", ";
    output << view.argument->name;
  }
  for (ABIScalar &scalar : scalars)
    output << ", " << scalar.name;
  output << ")\n    return " << outputView->argument->name << "\n";
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
    if (domain && domain->getName().getStringRef() == "intent.domain")
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
    Operation *source = relation && relation->getNumOperands() == 3
                            ? relation->getOperand(0).getDefiningOp()
                            : nullptr;
    if (!source)
      return failure();
    return dimensionName(*source);
  }
  if (name == "intent.ragged_member") {
    Operation *relation = domain.getOperand(0).getDefiningOp();
    Operation *indices = relation && relation->getNumOperands() == 3
                             ? relation->getOperand(2).getDefiningOp()
                             : nullptr;
    if (!indices || indices->getName().getStringRef() != "intent.view_load" ||
        indices->getNumOperands() != 1)
      return failure();
    FailureOr<ABIView *> view = lookupView(indices->getOperand(0), domain);
    if (failed(view) || (*view)->shape.empty())
      return failure();
    return (*view)->shape.front();
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
    StringRef role = axis->getRole();
    if (role == "program_0")
      indices.push_back(
          planIndex.program.getMapping() == "persistent_rows"
              ? programIndex
              : planIndex.program.getMapping() == "multi_axis_stream"
                    ? "index_program_0"
                    : "bid_m");
    else if (role == "program_1")
      indices.push_back(planIndex.program.getMapping() == "multi_axis_stream"
                            ? "index_program_1"
                            : "bid_n");
    else if (role == "program_2")
      indices.push_back("bid_program_2");
    else if (role == "stream_0")
      indices.push_back("stream_tile");
    else if (role == "lane_0")
      indices.push_back(vectorIndex);
    else if (role == "reduction_0" && reductionLoop)
      indices.push_back("k_tile");
    else {
      operation.emitOpError() << "has no cuTile index for role " << role;
      return failure();
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
    if (axis->getTile() == "one")
      extents.push_back("1");
    else if (axis->getRole() == "program_0")
      extents.push_back("TILE_SIZE_M");
    else if (axis->getRole() == "program_1")
      extents.push_back("TILE_SIZE_N");
    else if (axis->getRole() == "program_2")
      extents.push_back("TILE_SIZE_M");
    else if (axis->getRole() == "stream_0")
      extents.push_back("TILE_SIZE_N");
    else if (axis->getRole() == "reduction_0")
      extents.push_back("TILE_SIZE_K");
    else {
      operation.emitOpError("has no cuTile tile extent for physical axis");
      return failure();
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
  if (planIndex.program.getMapping() == "multi_axis_stream" &&
      extents.size() == 1 && extents.front() == "TILE_SIZE_M")
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
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return operation.emitOpError("boundary value is not a ranked tensor");
  return static_cast<unsigned>(tensor.getRank());
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
  if (isRaggedStages()) {
    for (unsigned stage : activeStages)
      stageLine(stage, text, indentation);
    return;
  }
  output.indent(indentation * 4) << text << "\n";
}

bool SourceEmitter::isRaggedStages() {
  return planIndex.program &&
         planIndex.program.getMapping() == "ragged_stages";
}

LogicalResult emitRealizedKernelSource(
    target::KernelModel kernel, intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return SourceEmitter(std::move(kernel), realization, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .emit();
}

} // namespace intent::cutile::emission

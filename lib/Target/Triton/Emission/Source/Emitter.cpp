#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::triton::emission {

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
    else if (auto value = dyn_cast<plan::StorageOp>(operation))
      index.storage[value.getValue()] = value;
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
    if (auto autotune = dyn_cast<plan::AutotuneOp>(operation))
      index.autotune = autotune;
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
               << " has no Triton runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("Triton emitter requires ranked views");
    if (!planIndex.storage.count(argument.valueID))
      return kernel.entry.emitOpError()
             << "ABI value " << argument.valueID
             << " lacks global storage realization";
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
  bool multiAxis = planIndex.program.getMapping() == "multi_axis_stream";
  bool raggedStages = isRaggedStages();
  if ((!multiAxis && !raggedStages &&
       planIndex.program.getWorkerAxes().size() != 1) ||
      ((multiAxis || raggedStages) &&
       planIndex.program.getWorkerAxes().size() != 2))
    return planIndex.program.emitOpError(
        "worker-axis count does not match the Triton program mapping");
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
          "cannot index its region shape against the physical plan");
    regionTiles["?region_" + std::to_string(argument.getInt()) + "_0"] =
        axis.getTile().str();
  }
  if (planIndex.program.getMapping() == "grid_stride") {
    auto program = planIndex.axesByRole.find("program_0");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (program == planIndex.axesByRole.end() ||
        lane == planIndex.axesByRole.end())
      return realization.emitOpError(
          "grid-stride plan requires program_0 and lane_0 axes");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if (view.view.getAccess() == "out" && !fixedOutput)
        fixedOutput = &view;
    }
    if (!fixedOutput || fixedOutput->tensor.getRank() != 2)
      return kernel.entry.emitOpError(
          "grid-stride Triton program requires one rank-two output view");
    if (searchSpace)
      return searchSpace.emitOpError(
          "fixed grid-stride scheduling cannot consume an autotune space");
  } else if (planIndex.program.getMapping() == "grouped_2d_tiles") {
    if (!searchSpace || !searchIndex.autotune)
      return realization.emitOpError(
          "grouped tiled program requires a delegated backend tuner");
    for (StringRef role : {"program_0", "program_1", "reduction_0"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "grouped tiled program lacks " << role << " axis";
    SmallVector<StringRef> expectedKeys = {
        roleDimensions.lookup("program_0"), roleDimensions.lookup("program_1"),
        roleDimensions.lookup("reduction_0")};
    if (searchIndex.autotune.getKey().size() != expectedKeys.size())
      return searchIndex.autotune.emitOpError(
          "does not specialize every tiled program/reduction dimension");
    for (auto [attribute, expected] :
         llvm::zip(searchIndex.autotune.getKey(), expectedKeys))
      if (cast<StringAttr>(attribute).getValue() != expected)
        return searchIndex.autotune.emitOpError(
            "key order does not match program_0/program_1/reduction_0");
  } else if (planIndex.program.getMapping() == "multi_axis_stream") {
    if (!searchSpace || !searchIndex.autotune)
      return realization.emitOpError(
          "ordered stream requires a delegated backend tuner");
    if (planIndex.streams.size() != 1)
      return realization.emitOpError(
          "ordered stream requires one stream binding");
    for (StringRef role : {"program_0", "program_1", "program_2", "stream_0"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "ordered stream lacks " << role << " axis";
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
  } else if (raggedStages) {
    if (!searchSpace || !searchIndex.autotune ||
        !planIndex.ragged || planIndex.stages.empty() ||
        planIndex.atomics.empty())
      return realization.emitOpError(
          "ragged staging requires traversal, stages, atomic merge, and a delegated tuner");
    for (StringRef role : {"program_0", "program_1"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "ragged staging lacks " << role << " axis";
  } else {
    return planIndex.program.emitOpError("has no Triton program emitter");
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
            "workspace_" + std::to_string(valueID) + "_ptr";
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
  if (planIndex.program.getMapping() == "grid_stride") {
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
    if (failed(offsets) || failed(indices) || !searchIndex.autotune)
      return raggedRelation->emitOpError(
          "cannot resolve staged ragged metadata or delegated tuner");
    ABIView *atomicDestination = nullptr;
    for (const auto &binding : planIndex.atomics) {
      Operation *atomic = kernel.nodes.lookup(binding.first);
      FailureOr<ABIView *> destination =
          atomic && atomic->getNumOperands() > 0
              ? lookupView(atomic->getOperand(0), *atomic)
              : FailureOr<ABIView *>(failure());
      if (failed(destination) || atomicDestination)
        return raggedRelation->emitOpError(
            "requires exactly one staged atomic destination");
      atomicDestination = *destination;
    }

    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      llvm::raw_string_ostream source(stageBodies[stage]);
      source << "@triton.autotune(\n    configs=_CONFIGS,\n    key=[";
      for (auto [index, attribute] :
           llvm::enumerate(searchIndex.autotune.getKey())) {
        if (index)
          source << ", ";
        source << "'" << cast<StringAttr>(attribute).getValue() << "'";
      }
      source << "]";
      if (planIndex.stages[stage].getOutputs().empty())
        source << ",\n    restore_value=['" << atomicDestination->pointer << "']";
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
      for (const std::string &dimension : dimensionOrder)
        parameter(dimension);
      for (ABIView &view : views)
        for (const std::string &stride : view.strides)
          parameter(stride);
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs())
          parameter(workspaceNames.lookup(kernel.values.lookup(valueID)));
      parameter("MAX_ROUTES");
      for (NamedAttribute config : searchIndex.autotune.getParameterMap())
        parameter(config.getName().getValue().str() + ": tl.constexpr");
      source << "):\n";
      source.flush();

      std::string feature = stageFeatureDimensions.lookup(stage);
      stageLine(stage, "pid_feature = tl.program_id(axis=0)");
      stageLine(stage, "pid_expert_route = tl.program_id(axis=1)");
      stageLine(stage,
                "num_route_tiles = tl.cdiv(MAX_ROUTES, BLOCK_SIZE_M)");
      stageLine(stage, "expert = pid_expert_route // num_route_tiles");
      stageLine(stage, "route_tile = pid_expert_route % num_route_tiles");
      stageLine(stage, "route_begin = tl.load(" + (*offsets)->pointer +
                           " + expert * " + (*offsets)->strides[0] + ")");
      stageLine(stage, "route_end = tl.load(" + (*offsets)->pointer +
                           " + (expert + 1) * " + (*offsets)->strides[0] +
                           ")");
      stageLine(stage,
                "member_offsets = route_begin + route_tile * BLOCK_SIZE_M + "
                "tl.arange(0, BLOCK_SIZE_M)");
      stageLine(stage, "member_mask = member_offsets < route_end");
      stageLine(stage, "routes = tl.load(" + (*indices)->pointer +
                           " + member_offsets * " + (*indices)->strides[0] +
                           ", mask=member_mask, other=0)");
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
  if (planIndex.program.getMapping() == "grid_stride") {
    programIndex = makeRegionArgumentName(*programRoot, 0);
    vectorIndex = makeResultName(*vectorDomain, 0);
    valueNames[programRoot->getRegion(0).front().getArgument(0)] = programIndex;
    valueNames[vectorDomain->getResult(0)] = vectorIndex;
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
  for (NamedAttribute parameter : searchIndex.autotune.getParameterMap())
    emitParameter(parameter.getName().getValue().str() + ": tl.constexpr");
  output << "):\n";
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
  if (isRaggedStages()) {
    for (const std::string &body : stageBodies)
      output << body << "\n";
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
        StringRef dtype = tensor ? torchDtype(tensor.getElementType()) : StringRef();
        if (!tensor || tensor.getRank() != 2 || dtype.empty())
          return stage.emitOpError("has an unsupported workspace tensor");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty((R, "
               << stageFeatureDimensions.lookup(stage.getOrdinal())
               << "), device=_DEVICE, dtype=" << dtype << ")\n";
      }
    std::string experts = roleDimensions.lookup("program_0");
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      std::string feature = stageFeatureDimensions.lookup(stage);
      output << "    grid_stage_" << stage
             << " = lambda META: (triton.cdiv(" << feature
             << ", META['BLOCK_SIZE_N']), " << experts
             << " * triton.cdiv(max_routes, META['BLOCK_SIZE_M']), 1)\n";
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
      argument("max_routes");
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
    output << "    " << merge->argument->name << " = torch.zeros((";
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
  if (planIndex.program.getMapping() == "grid_stride") {
    SmallVector<ABIView *> inputs;
    for (ABIView &view : views)
      if (view.view.getAccess() == "in")
        inputs.push_back(&view);
      else if (view.view.getAccess() != "out" || &view != fixedOutput)
        return kernel.entry.emitOpError(
            "fixed grid-stride wrapper supports inputs and one output");
    if (inputs.empty())
      return kernel.entry.emitOpError(
          "fixed grid-stride wrapper requires an input view");
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
      StringRef dtype = view.tensor.getElementType().isF16() ? "torch.float16"
                                                             : "torch.float32";
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
    output << "    n_rows = "
           << dimensionOwners.lookup(roleDimensions.lookup("program_0")) << "\n";
    output << "    n_cols = "
           << dimensionOwners.lookup(roleDimensions.lookup("lane_0")) << "\n";
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
    output << "    " << fixedOutput->argument->name << " = torch.empty((";
    for (auto [axis, extent] : llvm::enumerate(fixedOutput->shape)) {
      if (axis)
        output << ", ";
      output << extent;
    }
    output << "), device=_DEVICE, dtype="
           << (fixedOutput->tensor.getElementType().isF16() ? "torch.float16"
                                                            : "torch.float32")
           << ")\n";
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
    output << ")\n";
    output << "    return " << fixedOutput->argument->name << "\n";
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
          "autotuned wrapper supports input views and one output view");
  }
  if (!outputView)
    return kernel.entry.emitOpError("autotuned wrapper has no output view");

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
    StringRef dtype = view.tensor.getElementType().isF16() ? "torch.float16"
                                                           : "torch.float32";
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
  if (planIndex.program.getMapping() == "grouped_2d_tiles") {
    std::string gridExtent =
        "triton.cdiv(" + roleDimensions.lookup("program_0") +
        ", META['BLOCK_SIZE_M']) * triton.cdiv(" +
        roleDimensions.lookup("program_1") + ", META['BLOCK_SIZE_N'])";
    output << "    grid = lambda META: " << programGrid(gridExtent) << "\n";
  } else if (planIndex.program.getMapping() == "multi_axis_stream") {
    SmallVector<std::string> grid = {"1", "1", "1"};
    int64_t tiledWorker = planIndex.program.getWorkerAxes()[0];
    int64_t outerWorker = planIndex.program.getWorkerAxes()[1];
    if (tiledWorker < 0 || tiledWorker >= 3 || outerWorker < 0 ||
        outerWorker >= 3 || tiledWorker == outerWorker)
      return planIndex.program.emitOpError(
          "has invalid multi-axis Triton grid dimensions");
    grid[tiledWorker] =
        "triton.cdiv(" + roleDimensions.lookup("program_2") +
        ", META['BLOCK_SIZE_Q'])";
    grid[outerWorker] = roleDimensions.lookup("program_0") + " * " +
                        roleDimensions.lookup("program_1");
    output << "    grid = lambda META: (" << grid[0] << ", " << grid[1]
           << ", " << grid[2] << ")\n";
  } else {
    return planIndex.program.emitOpError(
        "has no autotuned Triton grid emitter");
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
  output << "    " << outputView->argument->name << " = torch.empty((";
  for (auto [axis, extent] : llvm::enumerate(outputView->shape)) {
    if (axis)
      output << ", ";
    output << extent;
  }
  output << "), device=_DEVICE, dtype="
         << (outputView->tensor.getElementType().isF16() ? "torch.float16"
                                                         : "torch.float32")
         << ")\n";
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
  output << ")\n    return " << outputView->argument->name << "\n";
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
    if (domain && domain->getName().getStringRef() == "intent.domain")
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
  StringRef role = axis.getRole();
  std::string base;
  if (planIndex.program.getMapping() == "grid_stride") {
    if (role == "program_0")
      return programIndex;
    if (role == "lane_0")
      return vectorIndex;
    consumer.emitOpError()
        << "has no grid-stride index expression for axis role " << role;
    return failure();
  }
  if (planIndex.program.getMapping() == "multi_axis_stream") {
    if (role == "program_0")
      base = "index_program_0";
    else if (role == "program_1")
      base = "index_program_1";
    else if (role == "program_2")
      base = "offs_program_2";
    else if (role == "stream_0")
      base = "offs_stream_0";
    else {
      consumer.emitOpError()
          << "has no streamed index expression for axis role " << role;
      return failure();
    }
  } else if (role == "program_0")
    base = "offs_program_0";
  else if (role == "program_1")
    base = "offs_program_1";
  else if (role == "reduction_0")
    base = "offs_reduction_0";
  else if (role == "lane_0")
    base = vectorIndex;
  else {
    consumer.emitOpError() << "has no index expression for axis role " << role;
    return failure();
  }
  if (axis.getTile() == "one")
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
      FailureOr<plan::AxisOp> axis =
          resolveAxis(operation.getOperand(*term.operands.front()), operation);
      if (failed(axis))
        return failure();
      bool vector = axis->getTile() != "one";
      FailureOr<std::string> resolved =
          indexExpression(*axis, store, vectorAxis, *tensorRank, operation);
      if (failed(resolved))
        return failure();
      index = *resolved;
      if (vector)
        ++vectorAxis;
    }
    StringRef stride = planIndex.program.getMapping() == "grid_stride" &&
                               axisNumber == 1
                           ? StringRef("1")
                           : StringRef(view.strides[axisNumber]);
    expression += " + " + index + " * " + stride.str();
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
  FailureOr<unsigned> tensorRank = emittedTensorRank(operation, store);
  if (failed(tensorRank))
    return failure();
  SmallVector<std::string> predicates;
  unsigned vectorAxis = 0;
  for (const target::IndexTerm &term : *relation) {
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
    FailureOr<Operation *> domain =
        resolveDomain(operation.getOperand(*term.operands.front()), operation);
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
    if (failed(domain) || failed(axis))
      return failure();
    bool vector = axis->getTile() != "one";
    if (!vector)
      continue;
    bool gridStride = planIndex.program.getMapping() == "grid_stride";
    FailureOr<std::string> extent = gridStride && axis->getRole() == "program_0"
                                        ? FailureOr<std::string>(std::string("n_rows"))
                                    : gridStride && axis->getRole() == "lane_0"
                                        ? FailureOr<std::string>(std::string("n_cols"))
                                        : dimensionName(**domain);
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
  int64_t axis = planIndex.program.getWorkerAxes().front();
  if (axis == 0)
    return "(" + extent.str() + ", 1, 1)";
  if (axis == 1)
    return "(1, " + extent.str() + ", 1)";
  return "(1, 1, " + extent.str() + ")";
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
    target::KernelModel kernel,
    intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization);
  FailureOr<SearchIndex> indexedSearch = indexSearchSpace(searchSpace);
  if (failed(indexed) || failed(indexedSearch))
    return failure();
  return SourceEmitter(std::move(kernel), realization, searchSpace,
                       std::move(*indexed), std::move(*indexedSearch), output)
      .emit();
}

} // namespace intent::triton::emission

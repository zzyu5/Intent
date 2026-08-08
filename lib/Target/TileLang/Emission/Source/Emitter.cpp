#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::tilelang::emission {

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
    else if (auto value = dyn_cast<plan::PipelineOp>(operation))
      index.pipeline = value;
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
  StringAttr mapping = planIndex.program.getMappingAttr();
  if (!mapping || mapping.getValue().empty())
    return planIndex.program.emitOpError("has no resolved TileLang mapping");
  programMapping = mapping.getValue().str();
  if (failed(indexABI()) || failed(resolvePhysicalBindings()))
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
               << " has no TileLang runtime-scalar binding";
      scalars.push_back(ABIScalar{&argument, argument.type, argument.name});
      valueNames[argument.value] = argument.name;
      continue;
    }
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return kernel.entry.emitOpError("TileLang emitter requires ranked views");
    plan::StorageOp storage = planIndex.storage.lookup(argument.valueID);
    plan::LayoutOp layout = planIndex.layouts.lookup(argument.valueID);
    if (!storage || storage.getSpace() != "global" || !layout ||
        layout.getKind() != "row_major")
      return kernel.entry.emitOpError()
             << "ABI value " << argument.valueID
             << " lacks a global row-major TileLang binding";
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
    return kernel.entry.emitOpError("TileLang emitter requires external views");
  return success();
}

LogicalResult SourceEmitter::resolvePhysicalBindings() {
  programRoot = kernel.nodes.lookup(planIndex.program.getLoopNode());
  if (!programRoot || programRoot->getName().getStringRef() != "intent.parallel")
    return planIndex.program.emitOpError("does not bind an intent.parallel op");
  bool multiAxis = programMapping == "multi_axis_stream";
  bool raggedStages = isRaggedStages();
  if ((!multiAxis && !raggedStages &&
       planIndex.program.getWorkerAxes().size() != 1) ||
      ((multiAxis || raggedStages) &&
       planIndex.program.getWorkerAxes().size() != 2))
    return planIndex.program.emitOpError(
        "grid-axis count does not match the TileLang program mapping");
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
    if (domain->getNumResults() != 1)
      return entry.second.emitOpError("binds a domain without one canonical value");
    FailureOr<int64_t> valueID = target::getValueID(
        domain->getResult(0), kernel, *domain, "TileLang domain tile binding");
    if (failed(valueID))
      return failure();
    regionTiles["?region_" + std::to_string(*valueID) + "_0"] =
        entry.second.getTile().str();
  }
  for (const target::RegionNode &region : kernel.regions.nodes) {
    Operation *operation = region.operation;
    StringRef name = operation->getName().getStringRef();
    if (name != "intent.parallel" && name != "intent.state_stream")
      continue;
    Operation *domain = nullptr;
    if (name == "intent.state_stream")
      domain = operation->getNumOperands() > 0
                   ? operation->getOperand(0).getDefiningOp()
                   : nullptr;
    else if (operation->getNumOperands() > 0) {
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
          "cannot index its region shape against the TileLang plan");
    regionTiles["?region_" + std::to_string(argument.getInt()) + "_0"] =
        axis.getTile().str();
  }

  StringRef mapping = programMapping;
  if (mapping == "persistent_rows") {
    if (!planIndex.pipeline || !planIndex.launch || searchSpace)
      return realization.emitOpError(
          "fixed TileLang rows require launch/pipeline and no search space");
    auto lane = planIndex.axesByRole.find("lane_0");
    if (!planIndex.axesByRole.count("program_0") ||
        lane == planIndex.axesByRole.end())
      return realization.emitOpError(
          "fixed TileLang rows require program_0 and lane_0 axes");
    vectorDomain = kernel.nodes.lookup(lane->second.getNode());
    for (ABIView &view : views) {
      if (view.view.getAccess() == "out" && !fixedOutput)
        fixedOutput = &view;
      if (view.tensor.getRank() != 2)
        return kernel.entry.emitOpError(
            "fixed-row TileLang views must be rank two");
    }
    if (!fixedOutput)
      return kernel.entry.emitOpError(
          "fixed-row TileLang program requires one output view");
  } else if (mapping == "grouped_2d_tiles") {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty())
      return realization.emitOpError(
          "tiled TileLang program requires backend autotune candidates");
    for (StringRef role : {"program_0", "program_1", "reduction_0"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "tiled TileLang program lacks " << role << " axis";
  } else if (mapping == "multi_axis_stream") {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty() ||
        planIndex.streams.size() != 1)
      return realization.emitOpError(
          "streamed TileLang program requires one stream and autotune candidates");
    for (StringRef role : {"program_0", "program_1", "program_2", "stream_0"})
      if (!planIndex.axesByRole.count(role))
        return realization.emitOpError()
               << "streamed TileLang program lacks " << role << " axis";
  } else if (raggedStages) {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty() ||
        !planIndex.ragged || planIndex.stages.empty() ||
        planIndex.atomics.empty())
      return realization.emitOpError(
          "ragged TileLang staging requires traversal, stages, atomic merge, and candidates");
  } else {
    return planIndex.program.emitOpError("has no TileLang program emitter");
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

void SourceEmitter::emitImports() {
  output << "import torch\n";
  output << "import tilelang\n";
  output << "import tilelang.language as T\n";
  if (searchSpace) {
    output << "from tilelang.autotuner import autotune, set_autotune_inputs\n";
    output << "\n_CONFIGS = [\n";
    for (plan::ConfigOp config : searchIndex.configs) {
      output << "    {";
      bool first = true;
      for (NamedAttribute parameter : config.getParameters()) {
        if (!first)
          output << ", ";
        output << "'" << parameter.getName().getValue() << "': "
               << cast<IntegerAttr>(parameter.getValue()).getInt();
        first = false;
      }
      if (!first)
        output << ", ";
      output << "'num_stages': " << config.getNumStages()
             << ", 'threads': " << config.getThreads() << "},\n";
    }
    output << "]\n";
  }
  output << "\n\n";
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  auto emitBuilderParameters = [&](raw_ostream &stream, bool ragged) {
    bool first = true;
    auto parameter = [&](StringRef value) {
      if (!first)
        stream << ", ";
      stream << value;
      first = false;
    };
    for (const std::string &dimension : dimensionOrder)
      parameter(dimension);
    if (ragged)
      parameter("MAX_ROUTES");
    if (programMapping == "persistent_rows") {
      parameter("TILE_SIZE");
      parameter("num_stages=1");
      parameter("threads=128");
    } else {
      for (NamedAttribute config : searchIndex.configs.front().getParameters())
        parameter(config.getName().getValue().str() + "=1");
      parameter("num_stages=1");
      parameter("threads=128");
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
        parameter(workspaceNames.lookup(value) + ": T.Tensor((R, " +
                  stageFeatureDimensions.lookup(stage.getOrdinal()) + "), " +
                  dtype + ")");
      }
    return success();
  };

  auto emitDecorator = [&](raw_ostream &stream) {
    if (searchSpace)
      stream << "@autotune(configs=_CONFIGS, warmup=3, rep=10, "
                "timeout=100, skip_check=True)\n";
    stream << "@tilelang.jit\n";
  };

  if (isRaggedStages()) {
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      llvm::raw_string_ostream source(stageBodies[stage]);
      emitDecorator(source);
      source << "def " << kernelName << "_stage_" << stage << "(";
      emitBuilderParameters(source, true);
      source << "):\n    @T.prim_func\n    def main(";
      if (failed(emitMainParameters(source)))
        return failure();
      source << "):\n";
      std::string feature = stageFeatureDimensions.lookup(stage);
      source << "        with T.Kernel(T.ceildiv(" << feature
             << ", TILE_SIZE_N), E * T.ceildiv(MAX_ROUTES, TILE_SIZE_M), "
                "threads=threads) as (bid_feature, bid_expert_route):\n";
      source.flush();
      stageLine(stage,
                "num_route_tiles = T.ceildiv(MAX_ROUTES, TILE_SIZE_M)");
      stageLine(stage, "expert = bid_expert_route // num_route_tiles");
      stageLine(stage, "route_tile = bid_expert_route % num_route_tiles");
      stageLine(stage, "route_begin = route_offsets[expert]");
      stageLine(stage, "route_end = route_offsets[expert + 1]");
      stageLine(stage,
                "member_start = route_begin + route_tile * TILE_SIZE_M");
    }
    return success();
  }

  emitDecorator(output);
  output << "def " << kernelName << "(";
  emitBuilderParameters(output, false);
  output << "):\n    @T.prim_func\n    def main(";
  if (failed(emitMainParameters(output)))
    return failure();
  output << "):\n";
  StringRef mapping = programMapping;
  if (mapping == "persistent_rows") {
    output << "        with T.Kernel(M, threads=threads) as program_index:\n";
  } else if (mapping == "grouped_2d_tiles") {
    output << "        with T.Kernel(T.ceildiv(N, TILE_SIZE_N), "
              "T.ceildiv(M, TILE_SIZE_M), threads=threads) as (bid_n, bid_m):\n";
  } else if (mapping == "multi_axis_stream") {
    output << "        with T.Kernel(T.ceildiv(Q, TILE_SIZE_M), H, B, "
              "threads=threads) as (bid_program_2, index_program_1, "
              "index_program_0):\n";
  } else {
    return planIndex.program.emitOpError("has no TileLang kernel header");
  }
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
    for (plan::StageOp stage : planIndex.stages)
      for (int64_t valueID : stage.getOutputs()) {
        if (!first)
          output << ", ";
        output << workspaceNames.lookup(kernel.values.lookup(valueID));
        first = false;
      }
  };
  auto emitBuilderArguments = [&](bool ragged) {
    bool first = true;
    for (const std::string &dimension : dimensionOrder) {
      if (!first)
        output << ", ";
      output << dimension;
      first = false;
    }
    if (ragged) {
      if (!first)
        output << ", ";
      output << "max_routes";
    }
  };

  if (isRaggedStages()) {
    for (const std::string &body : stageBodies)
      output << body << "    return main\n\n\n";
  } else
    output << "    return main\n\n\n";
  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n_KERNEL_CACHE = {}\n\n\n";
  output << "def launch(";
  emitLaunchSignature();
  output << "):\n";
  if (failed(emitValidation()))
    return failure();

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
            "ragged TileLang stages require inputs and one inout merge view");
    }
    if (!merge)
      return kernel.entry.emitOpError("ragged TileLang stages have no merge view");
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
          return stage.emitOpError("has an unsupported TileLang workspace");
        output << "    " << workspaceNames.lookup(value) << " = torch.empty((R, "
               << stageFeatureDimensions.lookup(stage.getOrdinal())
               << "), device=_DEVICE, dtype=" << dtype << ")\n";
      }
    for (unsigned stage = 0; stage < planIndex.stages.size(); ++stage) {
      output << "    cache_key = (" << stage;
      for (const std::string &dimension : dimensionOrder)
        output << ", " << dimension;
      output << ", max_routes, " << inputs.front()->argument->name
             << ".dtype, str(_DEVICE))\n";
      output << "    if cache_key not in _KERNEL_CACHE:\n";
      output << "        with set_autotune_inputs(";
      bool first = true;
      for (ABIView &view : views) {
        if (!first)
          output << ", ";
        if (planIndex.stages[stage].getOutputs().empty() && &view == merge)
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
      for (plan::StageOp binding : planIndex.stages)
        for (int64_t valueID : binding.getOutputs()) {
          if (!first)
            output << ", ";
          output << workspaceNames.lookup(kernel.values.lookup(valueID));
          first = false;
        }
      output << "):\n            compiled = " << kernelName << "_stage_" << stage
             << "(";
      emitBuilderArguments(true);
      output << ")\n        _KERNEL_CACHE[cache_key] = compiled\n";
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
    output << "    " << merge->argument->name << " = torch.zeros((";
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
  ABIView *result = nullptr;
  for (ABIView &view : views) {
    if (view.view.getAccess() == "in")
      inputs.push_back(&view);
    else if (view.view.getAccess() == "out" && !result)
      result = &view;
    else
      return kernel.entry.emitOpError(
          "TileLang wrapper supports inputs and one output view");
  }
  if (!result)
    return kernel.entry.emitOpError("TileLang wrapper has no output view");
  output << "    cache_key = (";
  for (auto [index, dimension] : llvm::enumerate(dimensionOrder)) {
    if (index)
      output << ", ";
    output << dimension;
  }
  output << ", " << inputs.front()->argument->name << ".dtype, str(_DEVICE))\n";
  output << "    if cache_key not in _KERNEL_CACHE:\n";
  if (searchSpace) {
    output << "        with set_autotune_inputs(";
    emitKernelArguments();
    output << "):\n            compiled = " << kernelName << "(";
    emitBuilderArguments(false);
    output << ")\n";
  } else {
    output << "        compiled = " << kernelName << "(";
    emitBuilderArguments(false);
    if (!dimensionOrder.empty())
      output << ", ";
    output << "1 << (N - 1).bit_length(), num_stages="
           << planIndex.pipeline.getNumStages() << ", threads="
           << planIndex.launch.getThreads() << ")\n";
  }
  output << "        _KERNEL_CACHE[cache_key] = compiled\n";
  output << "    compiled = _KERNEL_CACHE[cache_key]\n    compiled(";
  emitKernelArguments();
  output << ")\n    return compiled\n\n\n";
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
  output << "    " << result->argument->name << " = torch.empty((";
  for (auto [axis, extent] : llvm::enumerate(result->shape)) {
    if (axis)
      output << ", ";
    output << extent;
  }
  output << "), device=_DEVICE, dtype="
         << torchDtype(result->tensor.getElementType()) << ")\n    launch(";
  first = true;
  for (ABIView &view : views) {
    if (!first)
      output << ", ";
    output << view.argument->name;
    first = false;
  }
  for (ABIScalar &scalar : scalars)
    output << ", " << scalar.name;
  output << ")\n    return " << result->argument->name << "\n";
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
    consumer.emitOpError("cannot resolve a TileLang stream source domain");
    return failure();
  }
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("cannot resolve index ownership during TileLang emission");
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
    consumer.emitOpError("indexes a domain without a TileLang axis binding");
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

FailureOr<std::string> SourceEmitter::accessIndices(Operation &operation,
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
            "static TileLang index has no canonical value");
      indices.push_back(std::to_string(*term.staticValues.front()));
      continue;
    }
    if ((term.kind != "region_index" && term.kind != "value_index") ||
        term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "TileLang access has no mechanical index relation");
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
    if (failed(axis))
      return failure();
    StringRef role = axis->getRole();
    StringRef mapping = programMapping;
    if (role == "program_0")
      indices.push_back(mapping == "persistent_rows"   ? "program_index"
                        : mapping == "multi_axis_stream" ? "index_program_0"
                                                          : "bid_m * TILE_SIZE_M");
    else if (role == "program_1")
      indices.push_back(mapping == "multi_axis_stream"
                            ? "index_program_1"
                            : "bid_n * TILE_SIZE_N");
    else if (role == "program_2")
      indices.push_back("bid_program_2 * TILE_SIZE_M");
    else if (role == "stream_0")
      indices.push_back("stream_tile * TILE_SIZE_N");
    else if (role == "lane_0")
      indices.push_back("0");
    else if (role == "reduction_0" && reductionLoop)
      indices.push_back("k_tile * TILE_SIZE_K");
    else {
      operation.emitOpError() << "has no TileLang index for role " << role;
      return failure();
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

FailureOr<SmallVector<std::string>>
SourceEmitter::tensorExtents(Operation &operation, unsigned resultIndex) {
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
          "tensor shape region has no TileLang tile binding");
    else
      extents.push_back(label.getValue().str());
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
  if (type.isBF16())
    return "T.bfloat16";
  if (type.isInteger(1))
    return "T.bool";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 32)
    return "T.int32";
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
  if (!isRaggedStages() || owner == stageOutputOwners.end())
    return;
  std::string feature = stageFeatureDimensions.lookup(owner->second);
  line("for store_i, store_j in T.Parallel(TILE_SIZE_M, TILE_SIZE_N):");
  ++indentation;
  line("if member_start + store_i < route_end and "
       "bid_feature * TILE_SIZE_N + store_j < " + feature + ":");
  ++indentation;
  line(workspaceNames.lookup(value) +
       "[member_start + store_i, bid_feature * TILE_SIZE_N + store_j] = " +
       name.str() + "[store_i, store_j]");
  --indentation;
  --indentation;
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

void SourceEmitter::line(StringRef text) {
  if (isRaggedStages()) {
    for (unsigned stage : activeStages)
      stageLine(stage, text, indentation);
    return;
  }
  output.indent(indentation * 4) << text << "\n";
}

bool SourceEmitter::isRaggedStages() {
  return programMapping == "ragged_stages";
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

} // namespace intent::tilelang::emission

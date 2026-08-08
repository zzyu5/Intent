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
    else if (auto value = dyn_cast<plan::BoundaryOp>(operation))
      index.boundaries[value.getNode()] = value;
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
  if (failed(indexABI()) || failed(resolvePhysicalBindings()))
    return failure();
  emitImports();
  if (failed(emitKernelHeader()))
    return failure();
  target::OperationHandlerRegistry registry;
  if (failed(registerEmissionHandlers(registry, *this)) ||
      failed(target::traverseKernel(kernel.entry, registry,
                                    "cuTile target emission")))
    return failure();
  output << "\n\n";
  return emitWrapper();
}

LogicalResult SourceEmitter::indexABI() {
  views.reserve(kernel.abi.arguments.size());
  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view)
      continue;
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
  if (planIndex.program.getWorkerAxes().size() != 1)
    return planIndex.program.emitOpError(
        "current cuTile mappings require one block axis");
  for (auto &entry : planIndex.axes) {
    Operation *domain = kernel.nodes.lookup(entry.first);
    if (!domain || domain->getName().getStringRef() != "intent.domain")
      return entry.second.emitOpError("does not bind an intent.domain op");
    FailureOr<std::string> dimension = dimensionName(*domain);
    if (failed(dimension))
      return entry.second.emitOpError("cannot resolve its source dimension");
    roleDimensions[entry.second.getRole()] = *dimension;
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
  } else {
    return planIndex.program.emitOpError("has no cuTile program emitter");
  }
  return success();
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
  if (planIndex.program.getMapping() == "persistent_rows") {
    for (StringRef parameter : {"N_ROWS: ConstInt", "TILE_SIZE: ConstInt",
                                "DIM_COLS: ConstInt"})
      emitParameter(parameter);
  } else {
    for (StringRef parameter : {"TILE_SIZE_M: ConstInt",
                                "TILE_SIZE_N: ConstInt",
                                "TILE_SIZE_K: ConstInt"})
      emitParameter(parameter);
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
    return {};
  };
  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n";

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
          "grouped cuTile wrapper supports inputs and one output");
  }
  if (!outputView)
    return kernel.entry.emitOpError("grouped cuTile wrapper has no output view");

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
  output << "        with ct.compiler_timeout(5):\n";
  output << "            result = exhaustive_search(\n";
  output << "                _CONFIGS,\n                stream,\n";
  output << "                lambda cfg: (ceil("
         << roleDimensions.lookup("program_0")
         << " / cfg.TILE_SIZE_M) * ceil("
         << roleDimensions.lookup("program_1")
         << " / cfg.TILE_SIZE_N), 1, 1),\n";
  output << "                " << kernelName << ",\n";
  output << "                lambda cfg: (";
  for (ABIView &view : views)
    output << view.argument->name << ", ";
  output << "cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K),\n";
  output << "                lambda cfg: {'num_ctas': cfg.num_ctas, 'occupancy': cfg.occupancy},\n";
  output << "            )\n";
  output << "        best = result.best.config\n";
  output << "        _TUNE_CACHE[cache_key] = (best, " << kernelName
         << ".replace_hints(num_ctas=best.num_ctas, occupancy=best.occupancy))\n";
  output << "    best, tuned_kernel = _TUNE_CACHE[cache_key]\n";
  output << "    grid = (ceil(" << roleDimensions.lookup("program_0")
         << " / best.TILE_SIZE_M) * ceil("
         << roleDimensions.lookup("program_1")
         << " / best.TILE_SIZE_N), 1, 1)\n";
  output << "    return ct.launch(stream, grid, tuned_kernel, (";
  for (ABIView &view : views)
    output << view.argument->name << ", ";
  output << "best.TILE_SIZE_M, best.TILE_SIZE_N, best.TILE_SIZE_K))\n\n\n";
  output << "def run(";
  for (auto [index, view] : llvm::enumerate(inputs)) {
    if (index)
      output << ", ";
    output << view->argument->name;
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
    if (definition->getName().getStringRef() == "intent.domain")
      return definition;
  auto argument = dyn_cast<BlockArgument>(indexedValue);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("cannot resolve index ownership during cuTile emission");
    return failure();
  }
  Operation *source = owner->getOperand(0).getDefiningOp();
  if (source && source->getName().getStringRef() == "intent.domain")
    return source;
  if (source && source->getName().getStringRef() == "intent.partition" &&
      source->getNumOperands() == 1) {
    Operation *domain = source->getOperand(0).getDefiningOp();
    if (domain && domain->getName().getStringRef() == "intent.domain")
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
    if (term.operands.size() != 1 || !term.operands.front()) {
      operation.emitOpError("cuTile access requires value-bound index terms");
      return failure();
    }
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
    if (failed(axis))
      return failure();
    StringRef role = axis->getRole();
    if (role == "program_0")
      indices.push_back(planIndex.program.getMapping() == "persistent_rows"
                            ? programIndex
                            : "bid_m");
    else if (role == "program_1")
      indices.push_back("bid_n");
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
  SmallVector<std::string> extents;
  for (const target::IndexTerm &term : *relation) {
    if (term.operands.size() != 1 || !term.operands.front())
      return failure();
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
    if (failed(axis))
      return failure();
    if (axis->getRole() == "program_0")
      extents.push_back("TILE_SIZE_M");
    else if (axis->getRole() == "program_1")
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
  output.indent(indentation * 4) << text << "\n";
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

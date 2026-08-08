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
    else if (auto value = dyn_cast<plan::BoundaryOp>(operation))
      index.boundaries[value.getNode()] = value;
  }
  if (!index.target || !index.program) {
    realization.emitOpError("lacks target or program realization choices");
    return failure();
  }
  if (index.program.getMapping() == "grid_stride" &&
      (!index.pipeline || !index.launch)) {
    realization.emitOpError("grid-stride emission requires fixed pipeline/launch");
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
                                    "Triton target emission")))
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
      return kernel.entry.emitOpError("Triton emitter requires ranked views");
    if (!planIndex.storage.count(argument.valueID) ||
        !planIndex.layouts.count(argument.valueID))
      return kernel.entry.emitOpError()
             << "ABI value " << argument.valueID
             << " lacks storage/layout realization";
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
  if (planIndex.program.getWorkerAxes().size() != 1)
    return planIndex.program.emitOpError(
        "current Triton program mappings require exactly one worker axis");
  for (auto &entry : planIndex.axes) {
    Operation *domain = kernel.nodes.lookup(entry.first);
    if (!domain || domain->getName().getStringRef() != "intent.domain")
      return entry.second.emitOpError("does not bind an intent.domain op");
    FailureOr<std::string> dimension = dimensionName(*domain);
    if (failed(dimension))
      return entry.second.emitOpError("cannot resolve its source dimension");
    roleDimensions[entry.second.getRole()] = *dimension;
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
      if (view.tensor.getRank() != 2)
        return kernel.entry.emitOpError(
            "grid-stride Triton views must be rank two");
    }
    if (!fixedOutput)
      return kernel.entry.emitOpError(
          "grid-stride Triton program requires one output view");
    if (searchSpace)
      return searchSpace.emitOpError(
          "fixed grid-stride scheduling cannot consume an autotune space");
    if (planIndex.pipeline.getLoopNode() != planIndex.program.getLoopNode() ||
        planIndex.launch.getLoopNode() != planIndex.program.getLoopNode())
      return realization.emitOpError(
          "fixed pipeline/launch must bind the resolved program root");
    if (planIndex.launch.getGridPolicy() != "persistent_occupancy")
      return planIndex.launch.emitOpError(
          "has no fixed grid-stride source emitter");
  } else if (planIndex.program.getMapping() == "grouped_2d_tiles") {
    if (!searchSpace || !searchIndex.autotune || searchIndex.configs.empty())
      return realization.emitOpError(
          "grouped tiled program requires a backend autotune search space");
    if (planIndex.pipeline || planIndex.launch)
      return realization.emitOpError(
          "autotuned grouped scheduling cannot carry fixed pipeline/launch choices");
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
  } else {
    return planIndex.program.emitOpError("has no Triton program emitter");
  }
  return success();
}

void SourceEmitter::emitImports() {
  output << "import torch\n";
  output << "import triton\n";
  output << "import triton.language as tl\n";
  if (planIndex.program.getMapping() == "grid_stride")
    output << "from triton.runtime import driver\n";
  output << "\n\n";
}

LogicalResult SourceEmitter::emitKernelHeader() {
  kernelName = (kernel.entry.getName() + "_kernel").str();
  if (searchSpace) {
    output << "@triton.autotune(\n    configs=[\n";
    for (plan::ConfigOp config : searchIndex.configs) {
      output << "        triton.Config({";
      for (auto [index, parameter] :
           llvm::enumerate(config.getParameters())) {
        if (index)
          output << ", ";
        output << "'" << parameter.getName().getValue() << "': "
               << cast<IntegerAttr>(parameter.getValue()).getInt();
      }
      output << "}, num_stages=" << config.getNumStages()
             << ", num_warps=" << config.getNumWarps() << "),\n";
    }
    output << "    ],\n    key=[";
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
  for (const std::string &dimension : dimensionOrder)
    emitParameter(dimension);
  for (ABIView &view : views)
    for (const std::string &stride : view.strides)
      emitParameter(stride);
  for (StringRef meta : {"BLOCK_SIZE_M: tl.constexpr",
                         "BLOCK_SIZE_N: tl.constexpr",
                         "BLOCK_SIZE_K: tl.constexpr",
                         "GROUP_SIZE_M: tl.constexpr"})
    emitParameter(meta);
  output << "):\n";
  return success();
}

LogicalResult SourceEmitter::emitWrapper() {
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
    ABIView *shapeOwner = inputs.front();
    output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
           << ")\n";
    output << "_PROPERTIES = driver.active.utils.get_device_properties(_DEVICE.index)\n";
    output << "_NUM_SM = _PROPERTIES['multiprocessor_count']\n";
    output << "_NUM_REGS = _PROPERTIES['max_num_regs']\n";
    output << "_SIZE_SMEM = _PROPERTIES['max_shared_mem']\n";
    output << "_WARP_SIZE = " << planIndex.target.getWarpSize() << "\n";
    output << "_NUM_WARPS = " << planIndex.launch.getNumWarps() << "\n";
    output << "_LOW_STAGES = " << planIndex.pipeline.getLowStages() << "\n";
    output << "_HIGH_STAGES = " << planIndex.pipeline.getHighStages() << "\n";
    output << "_SMEM_THRESHOLD = " << planIndex.pipeline.getSmemThreshold()
           << "\n\n\n";
    output << "def launch(";
    for (auto [index, view] : llvm::enumerate(views)) {
      if (index)
        output << ", ";
      output << view.argument->name;
    }
    output << "):\n";
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
      output << "    if " << view.argument->name << ".ndim != 2";
      if (&view != shapeOwner)
        output << " or " << view.argument->name << ".shape != "
               << shapeOwner->argument->name << ".shape";
      output << ":\n";
      output << "        raise ValueError('grid-stride views require one rank-two shape')\n";
      output << "    if " << view.argument->name << ".stride(1) != 1:\n";
      output << "        raise ValueError('grid-stride views require contiguous rows')\n";
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
    output << "    n_rows, n_cols = " << shapeOwner->argument->name << ".shape\n";
    output << "    block_size = triton.next_power_of_2(n_cols)\n";
    output << "    num_stages = _HIGH_STAGES if _SIZE_SMEM > _SMEM_THRESHOLD else _LOW_STAGES\n";
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
    for (ABIView &view : views)
      emitArgument(view.argument->name + ".stride(0)");
    for (StringRef argument : {"n_rows", "n_cols"})
      emitArgument(argument);
    output << ", BLOCK_SIZE=block_size, num_stages=num_stages, "
              "num_warps=_NUM_WARPS, grid=" << programGrid("1") << ")\n";
    output << "    kernel._init_handles()\n";
    output << "    occupancy = _NUM_REGS // (kernel.n_regs * _WARP_SIZE * _NUM_WARPS)\n";
    output << "    occupancy = min(occupancy, _SIZE_SMEM // kernel.metadata.shared)\n";
    output << "    num_programs = min(_NUM_SM * occupancy, n_rows)\n";
    output << "    return " << kernelName << "[" << programGrid("num_programs")
           << "](";
    first = true;
    for (ABIView &view : views)
      emitArgument(view.argument->name);
    for (ABIView &view : views)
      emitArgument(view.argument->name + ".stride(0)");
    for (StringRef argument : {"n_rows", "n_cols", "block_size", "num_stages"})
      emitArgument(argument);
    output << ")\n\n\n";
    output << "def run(";
    for (auto [index, view] : llvm::enumerate(inputs)) {
      if (index)
        output << ", ";
      output << view->argument->name;
    }
    output << "):\n";
    output << "    " << fixedOutput->argument->name << " = torch.empty_like("
           << shapeOwner->argument->name << ", dtype="
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
          "grouped tiled wrapper supports input views and one output view");
  }
  if (!outputView)
    return kernel.entry.emitOpError("grouped tiled wrapper has no output view");

  output << "_DEVICE = torch.device('cuda', " << planIndex.target.getDevice()
         << ")\n\n\n";
  output << "def launch(";
  for (auto [index, view] : llvm::enumerate(views)) {
    if (index)
      output << ", ";
    output << view.argument->name;
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
  std::string gridExtent =
      "triton.cdiv(" + roleDimensions.lookup("program_0") +
      ", META['BLOCK_SIZE_M']) * triton.cdiv(" +
      roleDimensions.lookup("program_1") + ", META['BLOCK_SIZE_N'])";
  output << "    grid = lambda META: " << programGrid(gridExtent) << "\n";
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
  for (const std::string &dimension : dimensionOrder)
    emitArgument(dimension);
  for (ABIView &view : views)
    for (int64_t axis = 0; axis < view.tensor.getRank(); ++axis)
      emitArgument(view.argument->name + ".stride(" + std::to_string(axis) + ")");
  output << ")\n\n\n";
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
    if (definition->getName().getStringRef() == "intent.domain")
      return definition;
  }
  auto argument = dyn_cast<BlockArgument>(indexedValue);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("cannot resolve index ownership during emission");
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
  plan::AxisOp axis = succeeded(node) ? planIndex.axes.lookup(*node) : plan::AxisOp();
  if (failed(node) || !axis) {
    consumer.emitOpError("indexes a domain without a physical axis binding");
    return failure();
  }
  return axis;
}

FailureOr<std::string> SourceEmitter::dimensionName(Operation &domain) {
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
  if (role == "program_0")
    base = store ? "offs_program_0" : "load_program_0";
  else if (role == "program_1")
    base = store ? "offs_program_1" : "load_program_1";
  else if (role == "reduction_0")
    base = "offs_reduction_0";
  else if (role == "lane_0")
    base = vectorIndex;
  else {
    consumer.emitOpError() << "has no index expression for axis role " << role;
    return failure();
  }
  if (tensorRank != 2) {
    consumer.emitOpError(
        "grouped tiled pointer emission currently requires rank-two views");
    return failure();
  }
  return base + (tensorAxis == 0 ? "[:, None]" : "[None, :]");
}

FailureOr<std::string>
SourceEmitter::emitPointerExpression(Operation &operation, ABIView &view,
                                     bool store) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation) || relation->size() != view.strides.size())
    return failure();
  std::string expression = view.pointer;
  for (auto [axisNumber, term] : llvm::enumerate(*relation)) {
    if (term.operands.size() != 1 || !term.operands.front()) {
      operation.emitOpError(
          "Triton pointer emission requires value-bound index terms");
      return failure();
    }
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getOperand(*term.operands.front()), operation);
    if (failed(axis))
      return failure();
    FailureOr<std::string> index = indexExpression(
        *axis, store, axisNumber, view.tensor.getRank(), operation);
    if (failed(index))
      return failure();
    StringRef stride = planIndex.program.getMapping() == "grid_stride" &&
                               axisNumber == 1
                           ? StringRef("1")
                           : StringRef(view.strides[axisNumber]);
    expression += " + " + *index + " * " + stride.str();
  }
  return expression;
}

FailureOr<std::string>
SourceEmitter::emitMaskExpression(Operation &operation, bool store) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<std::string> predicates;
  for (const target::IndexTerm &term : *relation) {
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
    bool gridStride = planIndex.program.getMapping() == "grid_stride";
    FailureOr<std::string> extent = gridStride && axis->getRole() == "program_0"
                                        ? FailureOr<std::string>(std::string("n_rows"))
                                    : gridStride && axis->getRole() == "lane_0"
                                        ? FailureOr<std::string>(std::string("n_cols"))
                                        : dimensionName(**domain);
    if (failed(extent))
      return failure();
    if ((gridStride && axis->getRole() == "program_0") ||
        (!gridStride && !store &&
         (axis->getRole() == "program_0" ||
          axis->getRole() == "program_1")))
      continue;
    std::string index;
    if (gridStride && axis->getRole() == "lane_0")
      index = vectorIndex;
    else if (axis->getRole() == "program_0")
      index = "offs_program_0[:, None]";
    else if (axis->getRole() == "program_1")
      index = "offs_program_1[None, :]";
    else if (axis->getRole() == "reduction_0") {
      bool firstTensorAxis = &term == &relation->front();
      index = std::string("offs_reduction_0") +
              (firstTensorAxis ? "[:, None]" : "[None, :]");
    } else if (axis->getRole() == "lane_0")
      index = vectorIndex;
    else {
      operation.emitOpError("has no mask expression for physical axis");
      return failure();
    }
    predicates.push_back("(" + index + " < " + *extent + ")");
  }
  if (predicates.empty())
    return std::string("True");
  std::string combined = predicates.front();
  for (StringRef predicate : llvm::drop_begin(predicates))
    combined += " & " + predicate.str();
  return combined;
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
  output.indent(indentation * 4) << text << "\n";
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

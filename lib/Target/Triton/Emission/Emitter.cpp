#include "Emitter.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <cmath>
#include <optional>
#include <string>

using namespace mlir;

namespace intent::triton {
namespace {

struct RealizationIndex {
  plan::TargetOp target;
  plan::AxisOp rowAxis;
  plan::AxisOp columnAxis;
  plan::ProgramOp program;
  plan::PipelineOp pipeline;
  plan::LaunchOp launch;
  DenseMap<int64_t, plan::StorageOp> storage;
  DenseMap<int64_t, plan::LayoutOp> layouts;
  DenseMap<int64_t, plan::ReductionOp> reductions;
  DenseMap<int64_t, plan::PointwiseOp> pointwise;
  DenseMap<int64_t, plan::BoundaryOp> boundaries;
};

FailureOr<int64_t> nodeID(Operation *operation) {
  auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
  if (!node) {
    operation->emitOpError("requires intent.node for target emission");
    return failure();
  }
  return node.getInt();
}

FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization) {
  RealizationIndex index;
  for (Operation &operation : realization.getBody().front()) {
    if (isa<intent::plan::YieldOp>(operation))
      continue;
    if (auto value = dyn_cast<plan::TargetOp>(operation))
      index.target = value;
    else if (auto value = dyn_cast<plan::AxisOp>(operation)) {
      if (value.getRole() == "row")
        index.rowAxis = value;
      else if (value.getRole() == "column")
        index.columnAxis = value;
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
    else if (auto value = dyn_cast<plan::BoundaryOp>(operation))
      index.boundaries[value.getNode()] = value;
  }
  if (!index.target || !index.rowAxis || !index.columnAxis || !index.program ||
      !index.pipeline || !index.launch) {
    realization.emitOpError("lacks required Triton realization choices");
    return failure();
  }
  return index;
}

std::string uniqueName(StringRef candidate, int64_t node,
                       llvm::StringSet<> &usedNames) {
  std::string result = candidate.str();
  if (!usedNames.insert(result).second) {
    result += "_" + std::to_string(node);
    usedNames.insert(result);
  }
  return result;
}

std::string resultName(Operation *operation, unsigned index,
                       llvm::StringSet<> &usedNames) {
  auto names = operation->getAttrOfType<ArrayAttr>("intent.result_names");
  auto name = names && index < names.size() ? dyn_cast<StringAttr>(names[index])
                                           : StringAttr();
  std::string candidate = name ? name.getValue().str()
                               : "v" + std::to_string(index);
  return uniqueName(candidate, *nodeID(operation), usedNames);
}

std::string regionArgumentName(Operation *operation, unsigned argumentIndex,
                               llvm::StringSet<> &usedNames) {
  auto regions =
      operation->getAttrOfType<ArrayAttr>("intent.region_argument_names");
  auto blocks = regions && !regions.empty() ? dyn_cast<ArrayAttr>(regions[0])
                                            : ArrayAttr();
  auto arguments = blocks && !blocks.empty() ? dyn_cast<ArrayAttr>(blocks[0])
                                             : ArrayAttr();
  auto name = arguments && argumentIndex < arguments.size()
                  ? dyn_cast<StringAttr>(arguments[argumentIndex])
                  : StringAttr();
  std::string candidate = name ? name.getValue().str()
                               : "arg" + std::to_string(argumentIndex);
  return uniqueName(candidate, *nodeID(operation), usedNames);
}

StringRef expectedReduction(Operation *operation) {
  auto combine = operation->getAttrOfType<StringAttr>("intent.combine");
  auto axes = operation->getAttrOfType<ArrayAttr>("intent.axes");
  auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                      : IntegerAttr();
  if (!combine || !axis || axis.getInt() != 0)
    return {};
  if (combine.getValue() == "maximum")
    return "tl.max";
  if (combine.getValue() == "add")
    return "tl.sum";
  return {};
}

StringRef expectedPointwise(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  if (name == "intent.broadcast")
    return "alias";
  auto logical = operation->getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return {};
  if (name == "intent.unary") {
    if (logical.getValue() == "exp")
      return "tl.exp";
    if (logical.getValue() == "exp2")
      return "tl.exp2";
    if (logical.getValue() == "log")
      return "tl.log";
    if (logical.getValue() == "rsqrt")
      return "tl.rsqrt";
    if (logical.getValue() == "negate")
      return "python_negate";
    return {};
  }
  if (name == "intent.binary") {
    if (logical.getValue() == "add")
      return "python_add";
    if (logical.getValue() == "subtract")
      return "python_subtract";
    if (logical.getValue() == "multiply")
      return "python_multiply";
    if (logical.getValue() == "true_divide")
      return "python_true_divide";
  }
  return {};
}

bool hasRowColumnIndex(Operation *operation, BlockArgument row, Value columns,
                       unsigned rowOperand, unsigned columnOperand) {
  auto relation = operation->getAttrOfType<ArrayAttr>("intent.index");
  if (!relation || relation.size() != 2 ||
      rowOperand >= operation->getNumOperands() ||
      columnOperand >= operation->getNumOperands())
    return false;
  auto rowTerm = dyn_cast<DictionaryAttr>(relation[0]);
  auto columnTerm = dyn_cast<DictionaryAttr>(relation[1]);
  auto rowPositions = rowTerm ? rowTerm.getAs<ArrayAttr>("operands") : ArrayAttr();
  auto columnPositions =
      columnTerm ? columnTerm.getAs<ArrayAttr>("operands") : ArrayAttr();
  auto rowKind = rowTerm ? rowTerm.getAs<StringAttr>("kind") : StringAttr();
  auto columnKind =
      columnTerm ? columnTerm.getAs<StringAttr>("kind") : StringAttr();
  auto encodedRow = rowPositions && rowPositions.size() == 1
                        ? dyn_cast<IntegerAttr>(rowPositions[0])
                        : IntegerAttr();
  auto encodedColumn = columnPositions && columnPositions.size() == 1
                           ? dyn_cast<IntegerAttr>(columnPositions[0])
                           : IntegerAttr();
  return rowKind && rowKind.getValue() == "value_index" && columnKind &&
         columnKind.getValue() == "region_index" && encodedRow &&
         encodedRow.getInt() == rowOperand && encodedColumn &&
         encodedColumn.getInt() == columnOperand &&
         operation->getOperand(rowOperand) == row &&
         operation->getOperand(columnOperand) == columns;
}

class Emitter {
public:
  Emitter(ModuleOp module, intent::plan::RealizationOp realization,
          RealizationIndex index, llvm::raw_ostream &output)
      : module(module), realization(realization), planIndex(std::move(index)),
        output(output) {}

  LogicalResult emit() {
    entry = module.lookupSymbol<func::FuncOp>(realization.getEntry());
    if (!entry)
      return realization.emitOpError("entry function disappeared before emission");
    if (failed(indexKernelNodes()) || failed(indexABI()) ||
        failed(resolvePhysicalBindings()))
      return failure();
    emitImports();
    emitKernelHeader();
    for (Operation &operation : entry.getBody().front())
      if (failed(emitTopLevelOperation(operation)))
        return failure();
    output << "\n\n";
    emitWrapper();
    return success();
  }

private:
  LogicalResult indexKernelNodes() {
    WalkResult result = entry.walk([&](Operation *operation) {
      auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
      if (!node)
        return WalkResult::advance();
      if (!kernelNodes.try_emplace(node.getInt(), operation).second) {
        operation->emitOpError("duplicates an intent.node during target emission");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult indexABI() {
    auto parameterNodes =
        entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
    auto parameters = entry->getAttrOfType<ArrayAttr>("intent.parameters");
    if (!parameterNodes || !parameters ||
        parameterNodes.size() != entry.getNumArguments() ||
        parameters.size() != entry.getNumArguments())
      return entry.emitOpError("has incomplete ABI metadata");
    for (auto [index, pair] : llvm::enumerate(llvm::zip(parameterNodes, parameters))) {
      auto node = dyn_cast<IntegerAttr>(std::get<0>(pair));
      auto metadata = dyn_cast<DictionaryAttr>(std::get<1>(pair));
      auto name = metadata ? metadata.getAs<StringAttr>("name") : StringAttr();
      auto view = dyn_cast<intent::ViewType>(entry.getArgument(index).getType());
      if (!node || !name || !view)
        return entry.emitOpError("Triton emitter requires view ABI metadata");
      int64_t valueID = node.getInt();
      if (!planIndex.storage.count(valueID) ||
          !planIndex.layouts.count(valueID))
        return entry.emitOpError()
               << "ABI value " << valueID << " lacks storage/layout realization";
      if (view.getAccess() == "in") {
        if (inputIndex)
          return entry.emitOpError("initial Triton emitter supports one input view");
        inputIndex = index;
        inputName = name.getValue().str();
      } else if (view.getAccess() == "out") {
        if (outputIndex)
          return entry.emitOpError("initial Triton emitter supports one output view");
        outputIndex = index;
        outputName = name.getValue().str();
      } else {
        return entry.emitOpError("initial Triton emitter does not support InOut ABI");
      }
    }
    if (!inputIndex || !outputIndex)
      return entry.emitOpError("requires one input and one output view");
    return success();
  }

  LogicalResult resolvePhysicalBindings() {
    parallel = kernelNodes.lookup(planIndex.program.getLoopNode());
    rowDomain = kernelNodes.lookup(planIndex.rowAxis.getNode());
    columnDomain = kernelNodes.lookup(planIndex.columnAxis.getNode());
    if (!parallel || parallel->getName().getStringRef() != "intent.parallel")
      return planIndex.program.emitOpError("does not bind an intent.parallel op");
    if (!rowDomain || !columnDomain ||
        rowDomain->getName().getStringRef() != "intent.domain" ||
        columnDomain->getName().getStringRef() != "intent.domain")
      return realization.emitOpError("axis choices do not bind Intent domains");
    if (parallel->getNumRegions() != 1 ||
        !llvm::hasSingleElement(parallel->getRegion(0)) ||
        parallel->getRegion(0).front().getNumArguments() != 1)
      return parallel->emitOpError("has no mechanical rowwise Triton translation");
    if (parallel->getNumOperands() != 1 ||
        parallel->getOperand(0) != rowDomain->getResult(0))
      return parallel->emitOpError("program ownership does not bind its row domain");
    return success();
  }

  void emitImports() {
    output << "import torch\n";
    output << "import triton\n";
    output << "import triton.language as tl\n";
    output << "from triton.runtime import driver\n\n\n";
  }

  void emitKernelHeader() {
    kernelName = (entry.getName() + "_kernel").str();
    inputPointer = inputName + "_ptr";
    outputPointer = outputName + "_ptr";
    inputStride = inputName + "_row_stride";
    outputStride = outputName + "_row_stride";
    usedNames.insert(inputPointer);
    usedNames.insert(outputPointer);
    usedNames.insert("row_start");
    usedNames.insert("row_step");
    usedNames.insert("valid");
    rowName = regionArgumentName(parallel, 0, usedNames);
    columnName = resultName(columnDomain, 0, usedNames);
    valueNames[entry.getArgument(*inputIndex)] = inputPointer;
    valueNames[entry.getArgument(*outputIndex)] = outputPointer;
    valueNames[parallel->getRegion(0).front().getArgument(0)] = rowName;

    output << "@triton.jit\n";
    output << "def " << kernelName << "(" << inputPointer << ", "
           << outputPointer << ", " << inputStride << ", " << outputStride
           << ", n_rows, n_cols, BLOCK_SIZE: tl.constexpr, "
              "num_stages: tl.constexpr):\n";
  }

  LogicalResult emitTopLevelOperation(Operation &operation) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.constant" || name == "intent.dim" ||
        name == "intent.domain" || name == "intent.return")
      return success();
    if (name != "intent.parallel")
      return operation.emitOpError("has no top-level Triton emission handler");
    if (&operation != parallel)
      return operation.emitOpError("is not owned by the resolved Triton program");
    output << "    row_start = tl.program_id(" << planIndex.program.getWorkerAxis()
           << ")\n";
    output << "    row_step = tl.num_programs(" << planIndex.program.getWorkerAxis()
           << ")\n";
    output << "    for " << rowName
           << " in tl.range(row_start, n_rows, row_step, "
              "num_stages=num_stages):\n";
    output << "        " << columnName << " = tl.arange(0, BLOCK_SIZE)\n";
    output << "        valid = " << columnName << " < n_cols\n";
    for (Operation &nested : parallel->getRegion(0).front())
      if (failed(emitBodyOperation(nested)))
        return failure();
    return success();
  }

  LogicalResult emitBodyOperation(Operation &operation) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.constant")
      return emitConstant(operation);
    if (name == "intent.view_load")
      return emitLoad(operation);
    if (name == "intent.reduce")
      return emitReduction(operation);
    if (name == "intent.broadcast")
      return emitBroadcast(operation);
    if (name == "intent.unary")
      return emitUnary(operation);
    if (name == "intent.binary")
      return emitBinary(operation);
    if (name == "intent.view_store")
      return emitStore(operation);
    if (name == "intent.yield")
      return success();
    return operation.emitOpError("has no Triton emission handler");
  }

  LogicalResult emitConstant(Operation &operation) {
    if (operation.getNumResults() != 1)
      return operation.emitOpError("constant emission requires one result");
    std::string result = resultName(&operation, 0, usedNames);
    Attribute value = operation.getAttr("intent.value");
    output << "        " << result << " = ";
    if (auto floating = dyn_cast<FloatAttr>(value)) {
      double number = floating.getValueAsDouble();
      if (std::isinf(number))
        output << (number < 0 ? "-float('inf')" : "float('inf')");
      else if (std::isnan(number))
        output << "float('nan')";
      else
        output << number;
    } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
      output << integer.getInt();
    } else {
      return operation.emitOpError("has an unsupported Triton constant value");
    }
    output << "\n";
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  LogicalResult emitLoad(Operation &operation) {
    FailureOr<int64_t> node = nodeID(&operation);
    plan::BoundaryOp boundary =
        succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
    if (failed(node) || !boundary)
      return operation.emitOpError("lacks a resolved boundary mechanism");
    if (boundary.getDomainNode() != planIndex.columnAxis.getNode())
      return boundary.emitOpError("does not bind the emitted column domain");
    if (operation.getNumOperands() != 3 || operation.getNumResults() != 1 ||
        !hasRowColumnIndex(&operation, parallel->getRegion(0).front().getArgument(0),
                           columnDomain->getResult(0), 1, 2))
      return operation.emitOpError("has no row-major Triton load handler");
    auto parameter = dyn_cast<BlockArgument>(operation.getOperand(0));
    if (!parameter || parameter.getOwner() != &entry.getBody().front())
      return operation.emitOpError("load base must be a realized ABI view");
    std::string baseName =
        parameter.getArgNumber() == *inputIndex ? inputName : outputName;
    std::string pointer = valueNames.lookup(operation.getOperand(0));
    std::string stride =
        parameter.getArgNumber() == *inputIndex ? inputStride : outputStride;
    std::string offsets = uniqueName(baseName + "_offsets", *node, usedNames);
    std::string result = resultName(&operation, 0, usedNames);
    output << "        " << offsets << " = " << rowName << " * " << stride
           << " + " << columnName << "\n";
    output << "        " << result << " = tl.load(" << pointer << " + "
           << offsets << ", mask=valid, other=-float('inf'))\n";
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  LogicalResult emitReduction(Operation &operation) {
    FailureOr<int64_t> node = nodeID(&operation);
    plan::ReductionOp binding =
        succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
    StringRef expected = expectedReduction(&operation);
    if (failed(node) || !binding || expected.empty() ||
        binding.getLowering() != expected)
      return operation.emitOpError("has no semantics-preserving Triton reduction binding");
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    if (failed(operand))
      return failure();
    std::string result = resultName(&operation, 0, usedNames);
    output << "        " << result << " = " << binding.getLowering() << "("
           << *operand << ", axis=" << binding.getAxis() << ")\n";
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  LogicalResult emitBroadcast(Operation &operation) {
    FailureOr<int64_t> node = nodeID(&operation);
    plan::PointwiseOp binding =
        succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
    if (failed(node) || !binding || binding.getLowering() != "alias" ||
        expectedPointwise(&operation) != "alias")
      return operation.emitOpError("has no Triton broadcast binding");
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    if (failed(operand))
      return failure();
    valueNames[operation.getResult(0)] = operand->str();
    return success();
  }

  LogicalResult emitUnary(Operation &operation) {
    FailureOr<int64_t> node = nodeID(&operation);
    plan::PointwiseOp binding =
        succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
    StringRef expected = expectedPointwise(&operation);
    if (failed(node) || !binding || expected.empty() ||
        binding.getLowering() != expected)
      return operation.emitOpError("has no semantics-preserving Triton unary binding");
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    if (failed(operand))
      return failure();
    std::string result = resultName(&operation, 0, usedNames);
    output << "        " << result << " = ";
    if (binding.getLowering() == "python_negate")
      output << "-(" << *operand << ")\n";
    else
      output << binding.getLowering() << "(" << *operand << ")\n";
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  LogicalResult emitBinary(Operation &operation) {
    FailureOr<int64_t> node = nodeID(&operation);
    plan::PointwiseOp binding =
        succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
    StringRef expected = expectedPointwise(&operation);
    if (failed(node) || !binding || expected.empty() ||
        binding.getLowering() != expected)
      return operation.emitOpError("has no semantics-preserving Triton binary binding");
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    if (failed(lhs) || failed(rhs))
      return failure();
    StringRef symbol;
    if (binding.getLowering() == "python_add")
      symbol = "+";
    else if (binding.getLowering() == "python_subtract")
      symbol = "-";
    else if (binding.getLowering() == "python_multiply")
      symbol = "*";
    else if (binding.getLowering() == "python_true_divide")
      symbol = "/";
    else
      return operation.emitOpError("uses an unsupported binary target lowering");
    std::string result = resultName(&operation, 0, usedNames);
    output << "        " << result << " = " << *lhs << " " << symbol << " "
           << *rhs << "\n";
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  LogicalResult emitStore(Operation &operation) {
    auto valueIndex =
        operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
    if (!valueIndex || operation.getNumOperands() != 4 ||
        !hasRowColumnIndex(&operation, parallel->getRegion(0).front().getArgument(0),
                           columnDomain->getResult(0), 2, 3))
      return operation.emitOpError("has no row-major Triton store handler");
    FailureOr<StringRef> stored = lookupValue(operation, valueIndex.getInt());
    if (failed(stored))
      return failure();
    auto parameter = dyn_cast<BlockArgument>(operation.getOperand(0));
    if (!parameter || parameter.getOwner() != &entry.getBody().front())
      return operation.emitOpError("store base must be a realized ABI view");
    FailureOr<int64_t> node = nodeID(&operation);
    if (failed(node))
      return failure();
    std::string baseName =
        parameter.getArgNumber() == *outputIndex ? outputName : inputName;
    std::string pointer = valueNames.lookup(operation.getOperand(0));
    std::string stride =
        parameter.getArgNumber() == *outputIndex ? outputStride : inputStride;
    std::string offsets = uniqueName(baseName + "_offsets", *node, usedNames);
    output << "        " << offsets << " = " << rowName << " * " << stride
           << " + " << columnName << "\n";
    output << "        tl.store(" << pointer << " + " << offsets << ", "
           << *stored << ", mask=valid)\n";
    return success();
  }

  FailureOr<StringRef> lookupValue(Operation &consumer, unsigned operandIndex) {
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

  void emitWrapper() {
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

    output << "def launch(" << inputName << ", " << outputName << "):\n";
    output << "    if " << inputName << ".device != _DEVICE or " << outputName
           << ".device != _DEVICE:\n";
    output << "        raise ValueError('tensors must reside on the realized CUDA device')\n";
    output << "    if " << inputName << ".dtype != torch.float32 or "
           << outputName << ".dtype != torch.float32:\n";
    output << "        raise ValueError('realized Triton views require f32')\n";
    output << "    if " << inputName << ".ndim != 2 or " << outputName
           << ".shape != " << inputName << ".shape:\n";
    output << "        raise ValueError('realized Triton views require equal rank-two shapes')\n";
    output << "    if " << inputName << ".stride(1) != 1 or " << outputName
           << ".stride(1) != 1:\n";
    output << "        raise ValueError('realized Triton views require contiguous rows')\n";
    output << "    input_start = " << inputName << ".data_ptr()\n";
    output << "    input_end = input_start + " << inputName << ".numel() * "
           << inputName << ".element_size()\n";
    output << "    output_start = " << outputName << ".data_ptr()\n";
    output << "    output_end = output_start + " << outputName << ".numel() * "
           << outputName << ".element_size()\n";
    output << "    if max(input_start, output_start) < min(input_end, output_end):\n";
    output << "        raise ValueError('realized views violate noalias')\n";
    output << "    n_rows, n_cols = " << inputName << ".shape\n";
    output << "    block_size = triton.next_power_of_2(n_cols)\n";
    output << "    num_stages = _HIGH_STAGES if _SIZE_SMEM > _SMEM_THRESHOLD else _LOW_STAGES\n";
    output << "    kernel = " << kernelName << ".warmup(" << inputName << ", "
           << outputName << ", " << inputName << ".stride(0), " << outputName
           << ".stride(0), n_rows, n_cols, BLOCK_SIZE=block_size, "
              "num_stages=num_stages, num_warps=_NUM_WARPS, grid=(1,))\n";
    output << "    kernel._init_handles()\n";
    output << "    occupancy = _NUM_REGS // (kernel.n_regs * _WARP_SIZE * _NUM_WARPS)\n";
    output << "    occupancy = min(occupancy, _SIZE_SMEM // kernel.metadata.shared)\n";
    output << "    num_programs = min(_NUM_SM * occupancy, n_rows)\n";
    output << "    return " << kernelName << "[(num_programs, 1, 1)]("
           << inputName << ", " << outputName << ", " << inputName
           << ".stride(0), " << outputName
           << ".stride(0), n_rows, n_cols, block_size, num_stages)\n\n\n";
    output << "def run(" << inputName << "):\n";
    output << "    " << outputName << " = torch.empty_like(" << inputName
           << ")\n";
    output << "    launch(" << inputName << ", " << outputName << ")\n";
    output << "    return " << outputName << "\n";
  }

  ModuleOp module;
  intent::plan::RealizationOp realization;
  RealizationIndex planIndex;
  llvm::raw_ostream &output;
  func::FuncOp entry;
  DenseMap<int64_t, Operation *> kernelNodes;
  DenseMap<Value, std::string> valueNames;
  std::optional<unsigned> inputIndex;
  std::optional<unsigned> outputIndex;
  Operation *parallel = nullptr;
  Operation *rowDomain = nullptr;
  Operation *columnDomain = nullptr;
  llvm::StringSet<> usedNames;
  std::string inputName;
  std::string outputName;
  std::string kernelName;
  std::string inputPointer;
  std::string outputPointer;
  std::string inputStride;
  std::string outputStride;
  std::string rowName;
  std::string columnName;
};

} // namespace

LogicalResult emitRealizedKernelSource(
    ModuleOp module, intent::plan::RealizationOp realization,
    llvm::raw_ostream &output) {
  FailureOr<RealizationIndex> indexed = indexRealization(realization);
  if (failed(indexed))
    return failure();
  return Emitter(module, realization, std::move(*indexed), output).emit();
}

} // namespace intent::triton

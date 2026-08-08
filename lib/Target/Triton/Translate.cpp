#include "Intent/Target/Triton/Translate.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent {
namespace triton {
namespace {

int64_t getNodeID(Operation *operation) {
  return operation->getAttrOfType<IntegerAttr>("intent.node").getInt();
}

StringRef getStringAttribute(Operation *operation, StringRef name) {
  auto attribute = operation->getAttrOfType<StringAttr>(name);
  return attribute ? attribute.getValue() : StringRef();
}

LogicalResult requireOperation(Operation *operation, StringRef name,
                               StringRef attribute = {},
                               StringRef value = {}) {
  if (!operation || operation->getName().getStringRef() != name)
    return failure();
  if (!attribute.empty() && getStringAttribute(operation, attribute) != value)
    return failure();
  return success();
}

Operation *findPrimitive(plan::PlanOp plan, StringRef kind,
                         StringRef operatorName,
                         const DenseMap<int64_t, Operation *> &nodes) {
  Operation *result = nullptr;
  for (auto primitive : plan.getBody().front().getOps<plan::PrimitiveOp>()) {
    if (primitive.getKind() != kind || primitive.getOperatorName() != operatorName)
      continue;
    if (result)
      return nullptr;
    result = nodes.lookup(primitive.getNode());
  }
  return result;
}

LogicalResult verifyAxisZero(Operation *operation, StringRef name) {
  auto axes = operation->getAttrOfType<ArrayAttr>(name);
  if (!axes || axes.size() != 1)
    return failure();
  auto axis = dyn_cast<IntegerAttr>(axes[0]);
  return success(axis && axis.getInt() == 0);
}

std::string getResultName(Operation *operation, unsigned index,
                          llvm::StringSet<> &usedNames) {
  auto names = operation->getAttrOfType<ArrayAttr>("intent.result_names");
  std::string candidate = cast<StringAttr>(names[index]).getValue().str();
  if (!usedNames.insert(candidate).second) {
    candidate += "_" + std::to_string(getNodeID(operation));
    usedNames.insert(candidate);
  }
  return candidate;
}

std::string getRegionArgumentName(Operation *operation, unsigned regionIndex,
                                  unsigned blockIndex, unsigned argumentIndex,
                                  llvm::StringSet<> &usedNames) {
  auto regions =
      operation->getAttrOfType<ArrayAttr>("intent.region_argument_names");
  auto blocks = cast<ArrayAttr>(regions[regionIndex]);
  auto arguments = cast<ArrayAttr>(blocks[blockIndex]);
  std::string candidate =
      cast<StringAttr>(arguments[argumentIndex]).getValue().str();
  if (!usedNames.insert(candidate).second) {
    candidate += "_" + std::to_string(getNodeID(operation));
    usedNames.insert(candidate);
  }
  return candidate;
}

LogicalResult verifyIndexRelation(Operation *operation, BlockArgument row,
                                  Value columns, unsigned rowOperand,
                                  unsigned columnOperand) {
  auto relation = operation->getAttrOfType<ArrayAttr>("intent.index");
  if (!relation || relation.size() != 2)
    return failure();
  auto rowTerm = dyn_cast<DictionaryAttr>(relation[0]);
  auto columnTerm = dyn_cast<DictionaryAttr>(relation[1]);
  if (!rowTerm || !columnTerm)
    return failure();
  auto rowKind = rowTerm.getAs<StringAttr>("kind");
  auto columnKind = columnTerm.getAs<StringAttr>("kind");
  if (!rowKind || !columnKind || rowKind.getValue() != "value_index" ||
      columnKind.getValue() != "region_index")
    return failure();
  auto rowPositions = rowTerm.getAs<ArrayAttr>("operands");
  auto columnPositions = columnTerm.getAs<ArrayAttr>("operands");
  if (!rowPositions || !columnPositions || rowPositions.size() != 1 ||
      columnPositions.size() != 1)
    return failure();
  auto encodedRowOperand = dyn_cast<IntegerAttr>(rowPositions[0]);
  auto encodedColumnOperand = dyn_cast<IntegerAttr>(columnPositions[0]);
  if (!encodedRowOperand || !encodedColumnOperand ||
      encodedRowOperand.getInt() != rowOperand ||
      encodedColumnOperand.getInt() != columnOperand ||
      rowOperand >= operation->getNumOperands() ||
      columnOperand >= operation->getNumOperands())
    return failure();
  return success(operation->getOperand(rowOperand) == row &&
                 operation->getOperand(columnOperand) == columns);
}

LogicalResult verifyStableSoftmax(func::FuncOp entry, plan::PlanOp plan) {
  DenseMap<int64_t, Operation *> nodes;
  entry.walk([&](Operation *operation) {
    if (operation != entry.getOperation())
      nodes[getNodeID(operation)] = operation;
  });

  auto launch = *plan.getBody().front().getOps<plan::LaunchOp>().begin();
  Operation *rowLoop = nodes.lookup(launch.getLoopNode());
  if (failed(requireOperation(rowLoop, "intent.parallel")) ||
      rowLoop->getNumRegions() != 1 ||
      !llvm::hasSingleElement(rowLoop->getRegion(0)))
    return plan.emitOpError("launch does not reference one parallel row loop");

  Operation *reduceMax = findPrimitive(plan, "reduction", "maximum", nodes);
  Operation *subtract = findPrimitive(plan, "pointwise", "subtract", nodes);
  Operation *exponential = findPrimitive(plan, "pointwise", "exp", nodes);
  Operation *reduceSum = findPrimitive(plan, "reduction", "add", nodes);
  Operation *divide = findPrimitive(plan, "pointwise", "true_divide", nodes);
  SmallVector<Operation *> broadcasts;
  for (auto primitive : plan.getBody().front().getOps<plan::PrimitiveOp>())
    if (primitive.getKind() == "pointwise" && primitive.getOperatorName() == "broadcast")
      broadcasts.push_back(nodes.lookup(primitive.getNode()));
  if (broadcasts.size() != 2)
    return plan.emitOpError("stable softmax requires two scalar broadcasts");

  if (failed(requireOperation(reduceMax, "intent.reduce", "intent.combine", "maximum")) ||
      failed(requireOperation(subtract, "intent.binary", "intent.operator", "subtract")) ||
      failed(requireOperation(exponential, "intent.unary", "intent.operator", "exp")) ||
      failed(requireOperation(reduceSum, "intent.reduce", "intent.combine", "add")) ||
      failed(requireOperation(divide, "intent.binary", "intent.operator", "true_divide")) ||
      failed(verifyAxisZero(reduceMax, "intent.axes")) ||
      failed(verifyAxisZero(reduceSum, "intent.axes")))
    return plan.emitOpError("primitive bindings do not describe stable softmax");

  Block &body = rowLoop->getRegion(0).front();
  if (body.getNumArguments() != 1)
    return plan.emitOpError("stable softmax row loop requires one logical index");
  SmallVector<Operation *> loads;
  SmallVector<Operation *> stores;
  for (Operation &operation : body) {
    if (operation.getName().getStringRef() == "intent.view_load")
      loads.push_back(&operation);
    if (operation.getName().getStringRef() == "intent.view_store")
      stores.push_back(&operation);
  }
  if (loads.size() != 1 || stores.size() != 1)
    return plan.emitOpError("stable softmax requires one row load and one row store");
  Operation *load = loads.front();
  Operation *store = stores.front();

  plan::ExtentOp columnExtent;
  for (auto extent : plan.getBody().front().getOps<plan::ExtentOp>())
    if (extent.getAxis() == 1)
      columnExtent = extent;
  Operation *columnDomain = nodes.lookup(columnExtent.getNode());
  if (!columnDomain || columnDomain->getNumResults() != 1 ||
      failed(verifyIndexRelation(load, body.getArgument(0),
                                 columnDomain->getResult(0), 1, 2)) ||
      failed(verifyIndexRelation(store, body.getArgument(0),
                                 columnDomain->getResult(0), 2, 3)))
    return plan.emitOpError("memory index relation is not row-major softmax");

  Operation *maxBroadcast = nullptr;
  Operation *sumBroadcast = nullptr;
  for (Operation *broadcast : broadcasts) {
    if (!broadcast || broadcast->getName().getStringRef() != "intent.broadcast" ||
        broadcast->getNumOperands() != 1 || broadcast->getNumResults() != 1)
      return plan.emitOpError("broadcast primitive references an incompatible node");
    if (broadcast->getOperand(0) == reduceMax->getResult(0))
      maxBroadcast = broadcast;
    if (broadcast->getOperand(0) == reduceSum->getResult(0))
      sumBroadcast = broadcast;
  }
  if (!maxBroadcast || !sumBroadcast || load->getNumResults() != 1 ||
      reduceMax->getOperand(0) != load->getResult(0) ||
      subtract->getOperand(0) != load->getResult(0) ||
      subtract->getOperand(1) != maxBroadcast->getResult(0) ||
      exponential->getOperand(0) != subtract->getResult(0) ||
      reduceSum->getOperand(0) != exponential->getResult(0) ||
      divide->getOperand(0) != exponential->getResult(0) ||
      divide->getOperand(1) != sumBroadcast->getResult(0))
    return plan.emitOpError("Kernel IR def-use chain is not stable softmax");

  auto valueIndex = store->getAttrOfType<IntegerAttr>("intent.value_operand_index");
  if (!valueIndex || valueIndex.getInt() < 0 ||
      static_cast<unsigned>(valueIndex.getInt()) >= store->getNumOperands() ||
      store->getOperand(valueIndex.getInt()) != divide->getResult(0))
    return plan.emitOpError("stable softmax store does not write normalized values");

  auto parameterNodes = entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  if (!parameterNodes || parameterNodes.size() != 2)
    return plan.emitOpError("stable softmax ABI requires two view parameters");
  int64_t inputValue = cast<IntegerAttr>(parameterNodes[0]).getInt();
  int64_t outputValue = cast<IntegerAttr>(parameterNodes[1]).getInt();
  bool sawInputStorage = false;
  bool sawOutputStorage = false;
  for (auto storage : plan.getBody().front().getOps<plan::StorageOp>()) {
    sawInputStorage |= static_cast<int64_t>(storage.getValue()) == inputValue &&
                       storage.getAccess() == "read";
    sawOutputStorage |= static_cast<int64_t>(storage.getValue()) == outputValue &&
                        storage.getAccess() == "write";
  }
  if (!sawInputStorage || !sawOutputStorage ||
      load->getOperand(0) != entry.getArgument(0) ||
      store->getOperand(0) != entry.getArgument(1))
    return plan.emitOpError("Plan storage does not preserve the softmax ABI flow");
  return success();
}

LogicalResult emitSource(func::FuncOp entry, plan::PlanOp plan,
                         llvm::raw_ostream &output) {
  auto pipeline = *plan.getBody().front().getOps<plan::PipelineOp>().begin();
  auto launch = *plan.getBody().front().getOps<plan::LaunchOp>().begin();
  auto ownership = *plan.getBody().front().getOps<plan::OwnershipOp>().begin();
  plan::ExtentOp columnExtent;
  for (auto extent : plan.getBody().front().getOps<plan::ExtentOp>())
    if (extent.getAxis() == 1)
      columnExtent = extent;
  plan::BoundaryOp columnBoundary;
  for (auto boundary : plan.getBody().front().getOps<plan::BoundaryOp>())
    if (boundary.getNode() == columnExtent.getNode())
      columnBoundary = boundary;
  DenseMap<int64_t, plan::PrimitiveOp> primitives;
  for (auto primitive : plan.getBody().front().getOps<plan::PrimitiveOp>())
    primitives[primitive.getNode()] = primitive;

  auto parameterNodes = entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  auto parameterMetadata = entry->getAttrOfType<ArrayAttr>("intent.parameters");
  std::string inputName;
  std::string outputName;
  unsigned inputIndex = 0;
  unsigned outputIndex = 0;
  unsigned index = 0;
  for (auto [nodeAttribute, metadataAttribute] :
       llvm::zip(parameterNodes, parameterMetadata)) {
    int64_t valueID = cast<IntegerAttr>(nodeAttribute).getInt();
    auto metadata = cast<DictionaryAttr>(metadataAttribute);
    StringRef name = metadata.getAs<StringAttr>("name").getValue();
    for (auto storage : plan.getBody().front().getOps<plan::StorageOp>()) {
      if (static_cast<int64_t>(storage.getValue()) != valueID)
        continue;
      if (storage.getAccess() == "read") {
        inputName = name.str();
        inputIndex = index;
      } else if (storage.getAccess() == "write") {
        outputName = name.str();
        outputIndex = index;
      }
    }
    ++index;
  }
  if (inputName.empty() || outputName.empty())
    return plan.emitOpError("cannot derive input/output names from storage bindings");

  DenseMap<int64_t, Operation *> nodes;
  entry.walk([&](Operation *operation) {
    if (operation != entry.getOperation())
      nodes[getNodeID(operation)] = operation;
  });
  Operation *rowLoop = nodes.lookup(launch.getLoopNode());
  Operation *columnDomain = nodes.lookup(columnExtent.getNode());
  Block &body = rowLoop->getRegion(0).front();
  std::string kernelName = (entry.getName() + "_kernel").str();
  std::string inputPointer = inputName + "_ptr";
  std::string outputPointer = outputName + "_ptr";
  std::string inputStride = inputName + "_row_stride";
  std::string outputStride = outputName + "_row_stride";
  llvm::StringSet<> usedNames;
  usedNames.insert(inputPointer);
  usedNames.insert(outputPointer);
  usedNames.insert("row_start");
  usedNames.insert("row_step");
  usedNames.insert("valid");
  std::string rowIndex =
      getRegionArgumentName(rowLoop, 0, 0, 0, usedNames);
  std::string columns = getResultName(columnDomain, 0, usedNames);

  output << "import torch\n";
  output << "import triton\n";
  output << "import triton.language as tl\n";
  output << "from triton.runtime import driver\n\n\n";
  output << "@triton.jit\n";
  output << "def " << kernelName
         << "(" << inputPointer << ", " << outputPointer << ", "
         << inputStride << ", " << outputStride << ", "
            "n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):\n";
  output << "    row_start = tl.program_id(" << ownership.getWorkerAxis() << ")\n";
  output << "    row_step = tl.num_programs(" << ownership.getWorkerAxis() << ")\n";
  output << "    for " << rowIndex << " in tl.range(row_start, n_rows, row_step, "
            "num_stages=num_stages):\n";
  output << "        " << columns << " = tl.arange(0, BLOCK_SIZE)\n";
  output << "        valid = " << columns << " < n_cols\n";

  DenseMap<Value, std::string> valueNames;
  valueNames[entry.getArgument(inputIndex)] = inputPointer;
  valueNames[entry.getArgument(outputIndex)] = outputPointer;
  valueNames[body.getArgument(0)] = rowIndex;

  for (Operation &operation : body) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.constant" || name == "intent.yield")
      continue;
    if (name == "intent.view_load") {
      std::string result = getResultName(&operation, 0, usedNames);
      std::string pointer = valueNames.lookup(operation.getOperand(0));
      if (pointer != inputPointer)
        return operation.emitOpError("load storage is not the planned input view");
      output << "        " << inputName << "_offsets = " << rowIndex << " * "
             << inputStride << " + " << columns << "\n";
      output << "        " << result << " = tl.load(" << pointer << " + "
             << inputName << "_offsets, mask=valid, other=";
      output << (columnBoundary.getLoadFill() == "negative_infinity"
                     ? "-float('inf')"
                     : "0.0");
      output << ")\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    auto primitive = primitives.lookup(getNodeID(&operation));
    if (name == "intent.broadcast") {
      valueNames[operation.getResult(0)] = valueNames.lookup(operation.getOperand(0));
      continue;
    }
    if (name == "intent.reduce") {
      std::string result = getResultName(&operation, 0, usedNames);
      std::string source = valueNames.lookup(operation.getOperand(0));
      StringRef reduction = primitive.getOperatorName();
      if (reduction == "maximum" && primitive.getIdentity() != "negative_infinity")
        return primitive.emitOpError("maximum reduction requires negative-infinity identity");
      if (reduction == "add" && primitive.getIdentity() != "zero")
        return primitive.emitOpError("sum reduction requires zero identity");
      StringRef target = reduction == "maximum" ? "tl.max" : "tl.sum";
      output << "        " << result << " = " << target << "(" << source
             << ", axis=" << primitive.getAxis() << ")\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.unary") {
      std::string result = getResultName(&operation, 0, usedNames);
      if (primitive.getOperatorName() != "exp")
        return primitive.emitOpError("initial Triton unary lowering supports exp only");
      output << "        " << result << " = tl.exp("
             << valueNames.lookup(operation.getOperand(0)) << ")\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.binary") {
      std::string result = getResultName(&operation, 0, usedNames);
      StringRef operatorName = primitive.getOperatorName();
      StringRef symbol = operatorName == "subtract"     ? "-"
                         : operatorName == "true_divide" ? "/"
                                                         : StringRef();
      if (symbol.empty())
        return primitive.emitOpError("initial Triton binary lowering supports subtract/divide only");
      output << "        " << result << " = "
             << valueNames.lookup(operation.getOperand(0)) << " " << symbol
             << " " << valueNames.lookup(operation.getOperand(1)) << "\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.view_store") {
      auto valueIndex = operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
      std::string pointer = valueNames.lookup(operation.getOperand(0));
      if (pointer != outputPointer)
        return operation.emitOpError("store storage is not the planned output view");
      output << "        " << outputName << "_offsets = " << rowIndex << " * "
             << outputStride << " + " << columns << "\n";
      output << "        tl.store(" << pointer << " + " << outputName
             << "_offsets, "
             << valueNames.lookup(operation.getOperand(valueIndex.getInt()))
             << ", mask=valid)\n";
      continue;
    }
    return operation.emitOpError("has no Triton lowering in the initial Plan translator");
  }
  output << "\n\n";

  output << "_BACKEND = '" << plan.getBackend() << "'\n";
  output << "_ARCHITECTURE = '" << plan.getArchitecture() << "'\n";
  output << "_DEVICE = torch.device('cuda', " << plan.getDevice() << ")\n";
  output << "_PROPERTIES = driver.active.utils.get_device_properties(_DEVICE.index)\n";
  output << "_NUM_SM = _PROPERTIES['multiprocessor_count']\n";
  output << "_NUM_REGS = _PROPERTIES['max_num_regs']\n";
  output << "_SIZE_SMEM = _PROPERTIES['max_shared_mem']\n";
  output << "_WARP_SIZE = " << plan.getWarpSize() << "\n";
  output << "_NUM_WARPS = " << launch.getNumWarps() << "\n";
  output << "_LOW_STAGES = " << pipeline.getLowStages() << "\n";
  output << "_HIGH_STAGES = " << pipeline.getHighStages() << "\n";
  output << "_SMEM_THRESHOLD = " << pipeline.getSmemThreshold() << "\n\n\n";

  output << "def launch(" << inputName << ", " << outputName << "):\n";
  output << "    if " << inputName << ".device != _DEVICE or " << outputName
         << ".device != _DEVICE:\n";
  output << "        raise ValueError('softmax tensors must reside on the planned CUDA device')\n";
  output << "    if " << inputName << ".dtype != torch.float32 or " << outputName
         << ".dtype != torch.float32:\n";
  output << "        raise ValueError('softmax Plan requires f32 input/output')\n";
  output << "    if " << inputName << ".ndim != 2 or " << outputName
         << ".shape != " << inputName << ".shape:\n";
  output << "        raise ValueError('softmax Plan requires equal rank-two input/output')\n";
  output << "    if " << inputName << ".stride(1) != 1 or " << outputName
         << ".stride(1) != 1:\n";
  output << "        raise ValueError('softmax Plan requires contiguous row elements')\n";
  output << "    input_start = " << inputName << ".data_ptr()\n";
  output << "    input_end = input_start + " << inputName << ".numel() * "
         << inputName << ".element_size()\n";
  output << "    output_start = " << outputName << ".data_ptr()\n";
  output << "    output_end = output_start + " << outputName << ".numel() * "
         << outputName << ".element_size()\n";
  output << "    if max(input_start, output_start) < min(input_end, output_end):\n";
  output << "        raise ValueError('softmax input/output violate noalias')\n";
  output << "    n_rows, n_cols = " << inputName << ".shape\n";
  output << "    block_size = ";
  output << (columnExtent.getTile() == "next_power_of_two"
                 ? "triton.next_power_of_2(n_cols)"
                 : "n_cols");
  output << "\n";
  output << "    num_stages = _HIGH_STAGES if _SIZE_SMEM > _SMEM_THRESHOLD else _LOW_STAGES\n";
  output << "    kernel = " << kernelName
         << ".warmup(" << inputName << ", " << outputName << ", "
         << inputName << ".stride(0), " << outputName << ".stride(0), "
            "n_rows, n_cols, BLOCK_SIZE=block_size, num_stages=num_stages, "
            "num_warps=_NUM_WARPS, grid=(1,))\n";
  output << "    kernel._init_handles()\n";
  output << "    occupancy = _NUM_REGS // (kernel.n_regs * _WARP_SIZE * _NUM_WARPS)\n";
  output << "    occupancy = min(occupancy, _SIZE_SMEM // kernel.metadata.shared)\n";
  output << "    num_programs = min(_NUM_SM * occupancy, n_rows)\n";
  output << "    return " << kernelName
         << "[(num_programs, 1, 1)](" << inputName << ", " << outputName
         << ", " << inputName << ".stride(0), " << outputName
         << ".stride(0), n_rows, n_cols, block_size, num_stages)\n\n\n";
  output << "def run(" << inputName << "):\n";
  output << "    " << outputName << " = torch.empty_like(" << inputName << ")\n";
  output << "    launch(" << inputName << ", " << outputName << ")\n";
  output << "    return " << outputName << "\n";
  return success();
}

} // namespace

LogicalResult emitTritonSource(ModuleOp module, llvm::raw_ostream &output) {
  if (failed(verifyKernelModule(module)) || failed(verify(module)))
    return failure();
  SmallVector<plan::PlanOp> plans(module.getOps<plan::PlanOp>());
  if (plans.size() != 1)
    return module.emitError("Triton translation requires exactly one Physical Plan");
  plan::PlanOp physicalPlan = plans.front();
  auto entry = module.lookupSymbol<func::FuncOp>(physicalPlan.getEntry());
  if (!entry || failed(verifyStableSoftmax(entry, physicalPlan)))
    return failure();
  return emitSource(entry, physicalPlan, output);
}

} // namespace triton
} // namespace intent

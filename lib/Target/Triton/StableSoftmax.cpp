#include "StableSoftmax.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent {
namespace triton {
namespace {

std::string getResultName(Operation *operation, unsigned index,
                          llvm::StringSet<> &usedNames) {
  auto names = operation->getAttrOfType<ArrayAttr>("intent.result_names");
  std::string candidate = cast<StringAttr>(names[index]).getValue().str();
  if (!usedNames.insert(candidate).second) {
    candidate += "_" + std::to_string(getIntentNodeID(operation));
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
    candidate += "_" + std::to_string(getIntentNodeID(operation));
    usedNames.insert(candidate);
  }
  return candidate;
}

} // namespace

LogicalResult emitStableSoftmaxSource(StableSoftmaxMatch &softmax,
                                      plan::PlanOp physicalPlan,
                                      llvm::raw_ostream &output) {
  Block &planBody = physicalPlan.getBody().front();
  auto pipeline = *planBody.getOps<plan::PipelineOp>().begin();
  auto launch = *planBody.getOps<plan::LaunchOp>().begin();
  auto ownership = *planBody.getOps<plan::OwnershipOp>().begin();
  DenseMap<int64_t, plan::PrimitiveOp> primitives;
  for (auto primitive : planBody.getOps<plan::PrimitiveOp>())
    primitives[primitive.getNodeAttr().getInt()] = primitive;

  auto parameterNodes =
      softmax.entry->getAttrOfType<ArrayAttr>("intent.parameter_nodes");
  auto parameterMetadata =
      softmax.entry->getAttrOfType<ArrayAttr>("intent.parameters");
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
    for (auto storage : planBody.getOps<plan::StorageOp>()) {
      if (storage.getValueAttr().getInt() != valueID)
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

  Block &body = softmax.rowLoop->getRegion(0).front();
  std::string kernelName = (softmax.entry.getName() + "_kernel").str();
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
      getRegionArgumentName(softmax.rowLoop, 0, 0, 0, usedNames);
  std::string columns = getResultName(softmax.columnDomain, 0, usedNames);

  output << "import torch\n";
  output << "import triton\n";
  output << "import triton.language as tl\n";
  output << "from triton.runtime import driver\n\n\n";
  output << "@triton.jit\n";
  output << "def " << kernelName << "(" << inputPointer << ", "
         << outputPointer << ", " << inputStride << ", " << outputStride
         << ", n_rows, n_cols, BLOCK_SIZE: tl.constexpr, "
            "num_stages: tl.constexpr):\n";
  output << "    row_start = tl.program_id(" << ownership.getWorkerAxis()
         << ")\n";
  output << "    row_step = tl.num_programs(" << ownership.getWorkerAxis()
         << ")\n";
  output << "    for " << rowIndex
         << " in tl.range(row_start, n_rows, row_step, "
            "num_stages=num_stages):\n";
  output << "        " << columns << " = tl.arange(0, BLOCK_SIZE)\n";
  output << "        valid = " << columns << " < n_cols\n";

  DenseMap<Value, std::string> valueNames;
  valueNames[softmax.entry.getArgument(inputIndex)] = inputPointer;
  valueNames[softmax.entry.getArgument(outputIndex)] = outputPointer;
  valueNames[body.getArgument(0)] = rowIndex;

  for (Operation &operation : body) {
    StringRef name = operation.getName().getStringRef();
    if (name == "intent.constant" || name == "intent.yield")
      continue;
    if (name == "intent.view_load") {
      std::string result = getResultName(&operation, 0, usedNames);
      output << "        " << inputName << "_offsets = " << rowIndex
             << " * " << inputStride << " + " << columns << "\n";
      output << "        " << result << " = tl.load(" << inputPointer
             << " + " << inputName
             << "_offsets, mask=valid, other=-float('inf'))\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.broadcast") {
      valueNames[operation.getResult(0)] =
          valueNames.lookup(operation.getOperand(0));
      continue;
    }
    if (name == "intent.reduce") {
      plan::PrimitiveOp primitive =
          primitives.lookup(getIntentNodeID(&operation));
      std::string result = getResultName(&operation, 0, usedNames);
      StringRef target;
      if (&operation == softmax.reduceMax)
        target = "tl.max";
      else if (&operation == softmax.reduceSum)
        target = "tl.sum";
      else
        return operation.emitOpError(
            "is not a supported stable softmax reduction");
      output << "        " << result << " = " << target << "("
             << valueNames.lookup(operation.getOperand(0)) << ", axis="
             << primitive.getAxis() << ")\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.unary") {
      std::string result = getResultName(&operation, 0, usedNames);
      output << "        " << result << " = tl.exp("
             << valueNames.lookup(operation.getOperand(0)) << ")\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.binary") {
      std::string result = getResultName(&operation, 0, usedNames);
      StringRef symbol;
      if (&operation == softmax.subtract)
        symbol = "-";
      else if (&operation == softmax.divide)
        symbol = "/";
      else
        return operation.emitOpError(
            "is not a supported stable softmax binary operation");
      output << "        " << result << " = "
             << valueNames.lookup(operation.getOperand(0)) << " " << symbol
             << " " << valueNames.lookup(operation.getOperand(1)) << "\n";
      valueNames[operation.getResult(0)] = result;
      continue;
    }
    if (name == "intent.view_store") {
      auto valueIndex =
          operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
      output << "        " << outputName << "_offsets = " << rowIndex
             << " * " << outputStride << " + " << columns << "\n";
      output << "        tl.store(" << outputPointer << " + " << outputName
             << "_offsets, "
             << valueNames.lookup(operation.getOperand(valueIndex.getInt()))
             << ", mask=valid)\n";
      continue;
    }
    return operation.emitOpError(
        "has no Triton lowering in the stable softmax emitter");
  }
  output << "\n\n";

  output << "_DEVICE = torch.device('cuda', " << physicalPlan.getDevice()
         << ")\n";
  output << "_PROPERTIES = driver.active.utils.get_device_properties(_DEVICE.index)\n";
  output << "_NUM_SM = _PROPERTIES['multiprocessor_count']\n";
  output << "_NUM_REGS = _PROPERTIES['max_num_regs']\n";
  output << "_SIZE_SMEM = _PROPERTIES['max_shared_mem']\n";
  output << "_WARP_SIZE = " << physicalPlan.getWarpSize() << "\n";
  output << "_NUM_WARPS = " << launch.getNumWarps() << "\n";
  output << "_LOW_STAGES = " << pipeline.getLowStages() << "\n";
  output << "_HIGH_STAGES = " << pipeline.getHighStages() << "\n";
  output << "_SMEM_THRESHOLD = " << pipeline.getSmemThreshold() << "\n\n\n";

  output << "def launch(" << inputName << ", " << outputName << "):\n";
  output << "    if " << inputName << ".device != _DEVICE or " << outputName
         << ".device != _DEVICE:\n";
  output << "        raise ValueError('softmax tensors must reside on the planned CUDA device')\n";
  output << "    if " << inputName << ".dtype != torch.float32 or "
         << outputName << ".dtype != torch.float32:\n";
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
  return success();
}

} // namespace triton
} // namespace intent

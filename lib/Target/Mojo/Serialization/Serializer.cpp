#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Target/Mojo/Serialization/Scalar.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Serialization/NativeABI.h"
#include "Intent/Serialization/Source.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace intent::mojo {
namespace {

struct Memory {
  std::string pointer;
  SmallVector<std::string> sizes;
  SmallVector<std::string> strides;
  std::string base;
  std::string offset;
};

std::string join(ArrayRef<std::string> values, llvm::StringRef separator = ", ") {
  return llvm::join(values, separator);
}

std::string dtype(Type type) { return scalarDType(type); }

std::string memoryElement(Type type) {
  return "SIMD[DType." + dtype(type) + ", 1]";
}

std::string valueType(Type type) { return scalarValueType(type); }

std::string ordering(AtomicOrdering order) {
  switch (order) {
  case AtomicOrdering::Relaxed: return "Ordering.RELAXED";
  case AtomicOrdering::Acquire: return "Ordering.ACQUIRE";
  case AtomicOrdering::Release: return "Ordering.RELEASE";
  case AtomicOrdering::AcquireRelease: return "Ordering.ACQUIRE_RELEASE";
  }
  llvm_unreachable("unknown atomic ordering");
}

class Serializer : public SourceEmitter {
public:
  explicit Serializer(llvm::raw_ostream &out) : SourceEmitter(out) {}

  LogicalResult function(func::FuncOp function, const NativeABI &abi) {
    ScopedValues local(*this);
    memories.clear(); allocations.clear(); scope.clear();
    SmallVector<std::string> signature;
    for (auto [number, argument] : llvm::enumerate(function.getArguments())) {
      std::string name = "a" + std::to_string(number);
      values[argument] = name;
      if (auto type = dyn_cast<MemRefType>(argument.getType())) {
        Memory memory{name, SmallVector<std::string>(type.getRank()),
                      SmallVector<std::string>(type.getRank()), name, "0"};
        SmallVector<int64_t> staticStrides;
        int64_t staticOffset;
        if (mlir::failed(type.getStridesAndOffset(staticStrides, staticOffset)))
          return function.emitError("Mojo entry requires a strided memory descriptor");
        for (unsigned axis = 0; axis < type.getRank(); ++axis) {
          if (!type.isDynamicDim(axis)) memory.sizes[axis] = std::to_string(type.getDimSize(axis));
          if (!ShapedType::isDynamic(staticStrides[axis])) memory.strides[axis] = std::to_string(staticStrides[axis]);
        }
        memories[argument] = std::move(memory);
      } else {
        if (argument.getType().isIndex()) values[argument] = "Int(" + name + ")";
      }
    }
    for (const NativeSlot &slot : abi.slots) {
      Value argument = function.getArgument(slot.parameter);
      std::string name = slot.name();
      scope.push_back(name);
      if (slot.role == NativeSlotRole::Pointer) {
        signature.push_back(name + ": Pointer[" + memoryElement(slot.element) + ", MutUntrackedOrigin]");
      } else if (slot.role == NativeSlotRole::Scalar) {
        signature.push_back(name + ": " + valueType(slot.carrier));
      } else {
        signature.push_back(name + ": Int64");
        auto &memory = memories.find(argument)->second;
        auto &entry = slot.role == NativeSlotRole::Extent ? memory.sizes[*slot.axis] : memory.strides[*slot.axis];
        if (entry.empty()) entry = "Int(" + name + ")";
      }
    }
    line("@export(\"" + function.getName().str() + "\")");
    line("@no_inline");
    line("def " + function.getName().str() + "(" + join(signature) + ") abi(\"C\"):");
    ++indent;
    line("initialize_runtime()");
    if (mlir::failed(block(function.front()))) return failure();
    --indent;
    line("");
    return failure(hasFailed());
  }

private:
  std::string name(Value value) { return valueString(value); }

  void bind(Value value, const std::string &name) {
    SourceEmitter::bind(value, name);
    scope.push_back(name);
  }

  std::string fresh(Value value) {
    std::string result = newName();
    bind(value, result);
    return result;
  }

  void assign(Value value, const std::string &expression, bool constant = false) {
    std::string identifier = fresh(value);
    if (constant) scope.pop_back();
    line(std::string(constant ? "comptime " : "var ") + identifier + " = " + expression);
  }

  std::string fold(OpFoldResult value) {
    if (auto ssa = dyn_cast<Value>(value)) return name(ssa);
    return std::to_string(cast<IntegerAttr>(cast<Attribute>(value)).getInt());
  }

  std::string offset(Value memory, ValueRange indices) {
    const Memory &descriptor = memories.at(memory);
    SmallVector<std::string> terms;
    for (auto [axis, index] : llvm::enumerate(indices))
      terms.push_back("(" + name(index) + ") * (" + descriptor.strides[axis] + ")");
    return terms.empty() ? "0" : join(terms, " + ");
  }

  std::string pointer(Value memory, ValueRange indices) {
    return memories.at(memory).pointer + ".unsafe_offset(" + offset(memory, indices) + ")";
  }

  LogicalResult block(Block &body) {
    for (Operation &operation : body.without_terminator())
      if (mlir::failed(emit(&operation))) return failure();
    return success();
  }

  LogicalResult forLoop(scf::ForOp loop) {
    SmallVector<std::string> carries;
    for (auto [argument, initial] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
      assign(argument, name(initial));
      carries.push_back(name(argument));
    }
    auto saved = scope.size();
    std::string iv = fresh(loop.getInductionVar());
    line("for " + iv + " in range(" + name(loop.getLowerBound()) + ", " + name(loop.getUpperBound()) + ", " + name(loop.getStep()) + "):");
    ++indent;
    if (mlir::failed(block(*loop.getBody()))) return failure();
    SmallVector<std::string> nextValues;
    for (Value value : loop.getBody()->getTerminator()->getOperands()) {
      std::string temporary = newName("next_");
      line("var " + temporary + " = " + name(value));
      nextValues.push_back(temporary);
    }
    for (auto [carry, value] : llvm::zip(carries, nextValues)) line(carry + " = " + value);
    if (loop.getBody()->getOperations().size() == 1 && carries.empty()) line("pass");
    --indent;
    scope.resize(saved);
    for (auto [result, carry] : llvm::zip(loop.getResults(), carries)) values[result] = carry;
    return success();
  }

  LogicalResult dispatch(cpu::TaskDispatchOp dispatch) {
    Block &body = dispatch.getBody().front();
    SmallVector<std::string> captures;
    for (const std::string &value : scope) captures.push_back("imm " + value);
    auto saved = scope.size();
    std::string task = newName("task_");
    std::string iv = fresh(body.getArgument(0));
    line("def " + task + "(" + iv + ": Int) {" + join(captures) + "}:");
    ++indent;
    if (mlir::failed(block(body))) return failure();
    if (body.getOperations().size() == 1) line("pass");
    --indent;
    scope.resize(saved);
    line("parallelize(" + task + ", " + name(dispatch.getCount()) + ", " + name(dispatch.getWorkerCount()) + ")");
    return success();
  }

  LogicalResult whileLoop(scf::WhileOp loop) {
    Block &before = loop.getBefore().front(), &after = loop.getAfter().front();
    for (auto [argument, initial] : llvm::zip(before.getArguments(), loop.getInits()))
      assign(argument, name(initial));
    for (BlockArgument argument : after.getArguments())
      line("var " + fresh(argument) + ": " + valueType(argument.getType()));
    auto saved = scope.size();
    auto transfer = [&](ValueRange from, ValueRange to) {
      SmallVector<std::string> nextValues;
      for (Value value : from) {
        std::string temporary = newName("next_");
        line("var " + temporary + " = " + name(value));
        nextValues.push_back(temporary);
      }
      for (auto [value, temporary] : llvm::zip(to, nextValues)) line(name(value) + " = " + temporary);
    };
    line("while True:");
    ++indent;
    if (mlir::failed(block(before))) return failure();
    auto condition = cast<scf::ConditionOp>(before.getTerminator());
    transfer(condition.getArgs(), after.getArguments());
    line("if not " + name(condition.getCondition()) + ":");
    ++indent; line("break"); --indent;
    if (mlir::failed(block(after))) return failure();
    transfer(after.getTerminator()->getOperands(), before.getArguments());
    --indent;
    scope.resize(saved);
    for (auto [result, argument] : llvm::zip(loop.getResults(), after.getArguments())) values[result] = name(argument);
    return success();
  }

  LogicalResult conditional(scf::IfOp operation) {
    SmallVector<std::string> results;
    for (Value result : operation.getResults()) {
      results.push_back(fresh(result));
      line("var " + results.back() + ": " + valueType(result.getType()));
    }
    auto saved = scope.size();
    auto branch = [&](Block *body) {
      if (mlir::failed(block(*body))) return failure();
      for (auto [result, value] : llvm::zip(results, body->getTerminator()->getOperands()))
        line(result + " = " + name(value));
      if (body->getOperations().size() == 1 && results.empty()) line("pass");
      return success();
    };
    line("if " + name(operation.getCondition()) + ":");
    ++indent;
    if (mlir::failed(branch(operation.thenBlock()))) return failure();
    --indent;
    scope.resize(saved);
    if (!operation.getElseRegion().empty()) {
      line("else:");
      ++indent;
      if (mlir::failed(branch(operation.elseBlock()))) return failure();
      --indent;
      scope.resize(saved);
    }
    return success();
  }

  LogicalResult allocation(Operation *operation, Value memory, ValueRange dynamicSizes, bool stack) {
    auto type = cast<MemRefType>(memory.getType());
    if (!type.getLayout().isIdentity())
      return operation->emitError("Mojo allocation requires an explicit dense storage layout");
    if (stack && !type.hasStaticShape())
      return operation->emitError("Mojo stack allocation requires static extents");
    std::string element = memoryElement(type.getElementType());
    std::string value = fresh(memory);
    std::string storage = "storage_" + value;
    Memory descriptor{value, {}, {}, value, "0"};
    unsigned dynamicAxis = 0;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      descriptor.sizes.push_back(type.isDynamicDim(axis) ? name(dynamicSizes[dynamicAxis++])
                                                        : std::to_string(type.getDimSize(axis)));
    std::string stride = "1";
    descriptor.strides.resize(type.getRank());
    for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
      descriptor.strides[axis] = stride;
      stride = "(" + stride + ") * (" + descriptor.sizes[axis] + ")";
    }
    if (stack) {
      int64_t elementBytes = type.getElementType().isIndex() ? 8 : (type.getElementTypeBitWidth() + 7) / 8;
      int64_t alignment = cast<memref::AllocaOp>(operation).getAlignment().value_or(elementBytes);
      line("var " + value + " = unsafe_stack_allocation[" +
          std::to_string(type.getNumElements()) + ", " + element + ", alignment=" +
          std::to_string(alignment) + "]()");
    } else {
      auto alignment = cast<memref::AllocOp>(operation).getAlignment();
      std::string layout = "Layout[" + element + "]";
      if (alignment) layout += ".aligned[" + std::to_string(*alignment) + "]";
      line("var " + storage + " = alloc(" + layout + "(count=" + stride + "))");
      line("var " + value + " = " + storage + ".unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()");
      allocations[memory] = storage;
    }
    memories[memory] = std::move(descriptor);
    return success();
  }

  LogicalResult emit(Operation *operation) {
    if (isStandardScalarOperation(operation)) {
      SmallVector<std::string> operands;
      for (Value value : operation->getOperands()) operands.push_back(name(value));
      auto expression = emitScalar(operation, operands);
      if (mlir::failed(expression)) return failure();
      assign(operation->getResult(0), *expression, isa<arith::ConstantOp>(operation));
      return success();
    }
    if (auto op = dyn_cast<cpu::InvokeOp>(operation)) {
      SmallVector<std::string> arguments;
      for (Value argument : op.getArguments())
        arguments.push_back(isa<MemRefType>(argument.getType())
            ? pointer(argument, {}) : name(argument));
      line("external_call[\"" + op.getCallee().str() + "\", NoneType](" +
           join(arguments, ", ") + ")");
    } else if (auto op = dyn_cast<func::CallOp>(operation)) {
      if (op.getCallee() == "intent_cpu_enter_ieee") {
        assign(op.getResult(0), "external_call[\"intent_cpu_enter_ieee\", UInt32]()");
      } else if (op.getCallee() == "intent_cpu_leave_ieee") {
        line("external_call[\"intent_cpu_leave_ieee\", NoneType](" + name(op.getOperand(0)) + ")");
      } else return op.emitError("Mojo runtime call has no declared C ABI spelling");
    } else if (auto op = dyn_cast<memref::DimOp>(operation)) {
      auto axis = op.getConstantIndex();
      if (!axis) return op.emitError("Mojo memory descriptor dimension must be static");
      assign(op.getResult(), memories.at(op.getSource()).sizes[*axis]);
    } else if (auto op = dyn_cast<memref::ExtractStridedMetadataOp>(operation)) {
      const Memory memory = memories.at(op.getSource());
      if (!op.getBaseBuffer().use_empty()) {
        assign(op.getBaseBuffer(), memory.base);
        memories[op.getBaseBuffer()] = {name(op.getBaseBuffer()), {}, {}, memory.base, "0"};
      }
      if (!op.getOffset().use_empty()) assign(op.getOffset(), memory.offset);
      for (auto [value, size] : llvm::zip(op.getSizes(), memory.sizes))
        if (!value.use_empty()) assign(value, size);
      for (auto [value, stride] : llvm::zip(op.getStrides(), memory.strides))
        if (!value.use_empty()) assign(value, stride);
    } else if (auto op = dyn_cast<memref::SubViewOp>(operation)) {
      const Memory source = memories.at(op.getSource());
      auto offsets = op.getMixedOffsets(), sizes = op.getMixedSizes(), strides = op.getMixedStrides();
      auto dropped = op.getDroppedDims();
      SmallVector<std::string> terms;
      Memory result;
      for (auto [axis, offset] : llvm::enumerate(offsets)) {
        terms.push_back("(" + fold(offset) + ") * (" + source.strides[axis] + ")");
        if (!dropped.test(axis)) {
          result.sizes.push_back(fold(sizes[axis]));
          result.strides.push_back("(" + fold(strides[axis]) + ") * (" + source.strides[axis] + ")");
        }
      }
      assign(op.getResult(), source.pointer + ".unsafe_offset(" + join(terms, " + ") + ")");
      result.pointer = name(op.getResult());
      result.base = source.base;
      result.offset = "(" + source.offset + ") + (" + join(terms, " + ") + ")";
      memories[op.getResult()] = std::move(result);
    } else if (auto op = dyn_cast<memref::ReinterpretCastOp>(operation)) {
      Memory result;
      result.base = memories.at(op.getSource()).base;
      result.offset = fold(op.getMixedOffsets()[0]);
      for (OpFoldResult size : op.getMixedSizes()) result.sizes.push_back(fold(size));
      for (OpFoldResult stride : op.getMixedStrides()) result.strides.push_back(fold(stride));
      assign(op.getResult(), result.base + ".unsafe_offset(" + result.offset + ")");
      result.pointer = name(op.getResult());
      memories[op.getResult()] = std::move(result);
    } else if (auto op = dyn_cast<memref::CastOp>(operation)) {
      memories[op.getResult()] = memories.at(op.getSource());
      values[op.getResult()] = name(op.getSource());
    } else if (auto op = dyn_cast<memref::LoadOp>(operation)) {
      std::string expression = pointer(op.getMemref(), op.getIndices()) + ".unsafe_load()";
      if (op.getType().isIndex()) expression = "Int(" + expression + ")";
      if (op.getType().isInteger(1)) expression = "Bool(" + expression + ")";
      assign(op.getResult(), expression);
    } else if (auto op = dyn_cast<memref::StoreOp>(operation)) {
      std::string value = name(op.getValue());
      if (op.getValue().getType().isIndex()) value = "Int64(" + value + ")";
      if (op.getValue().getType().isInteger(1)) value = "SIMD[DType.bool, 1](" + value + ")";
      line(pointer(op.getMemref(), op.getIndices()) + ".unsafe_store(" + value + ")");
    } else if (auto op = dyn_cast<cpu::AtomicLoadOp>(operation)) {
      assign(op.getValue(), "Atomic[DType." + dtype(op.getValue().getType()) + "].load[ordering=" +
          ordering(op.getOrdering()) + "](" + pointer(op.getTarget(), op.getIndices()) + ")");
    } else if (auto op = dyn_cast<cpu::AtomicStoreOp>(operation)) {
      line("Atomic[DType." + dtype(op.getValue().getType()) + "].store[ordering=" + ordering(op.getOrdering()) +
          "](" + pointer(op.getTarget(), op.getIndices()) + ", " + name(op.getValue()) + ")");
    } else if (auto op = dyn_cast<cpu::AtomicRMWOp>(operation)) {
      if (op.getKind() != AtomicRMWKind::Add) return op.emitError("Mojo atomic RMW was not expanded before serialization");
      assign(op.getOldValue(), "Atomic[DType." + dtype(op.getValue().getType()) + "].fetch_add[ordering=" +
          ordering(op.getOrdering()) + "](" + pointer(op.getTarget(), op.getIndices()) + ", " + name(op.getValue()) + ")");
    } else if (auto op = dyn_cast<cpu::AtomicCompareExchangeOp>(operation)) {
      assign(op.getOldValue(), name(op.getExpected()));
      auto failureOrder = op.getOrdering() == AtomicOrdering::Acquire || op.getOrdering() == AtomicOrdering::AcquireRelease
          ? AtomicOrdering::Acquire : AtomicOrdering::Relaxed;
      assign(op.getSuccess(), "Atomic[DType." + dtype(op.getExpected().getType()) + "].compare_exchange[success_ordering=" +
          ordering(op.getOrdering()) + ", failure_ordering=" + ordering(failureOrder) + ", weak=False](" +
          pointer(op.getTarget(), op.getIndices()) + ", " + name(op.getOldValue()) + ", " + name(op.getDesired()) + ")");
    } else if (auto op = dyn_cast<vector::LoadOp>(operation)) {
      assign(op.getResult(), pointer(op.getBase(), op.getIndices()) + ".unsafe_load[width=" + std::to_string(op.getVectorType().getNumElements()) + "]()");
    } else if (auto op = dyn_cast<vector::StoreOp>(operation)) {
      line(pointer(op.getBase(), op.getIndices()) + ".unsafe_store(" + name(op.getValueToStore()) + ")");
    } else if (auto op = dyn_cast<memref::PrefetchOp>(operation)) {
      line("prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](" +
          pointer(op.getMemref(), op.getIndices()) + ")");
    } else if (auto op = dyn_cast<vector::BroadcastOp>(operation)) {
      assign(op.getResult(), valueType(op.getType()) + "(" +
          (op.getType().getElementType().isInteger(1) ? "fill=" : "") + name(op.getSource()) + ")");
    } else if (auto op = dyn_cast<vector::FromElementsOp>(operation)) {
      SmallVector<std::string> lanes;
      Type element = op.getType().getElementType();
      for (Value value : op.getElements())
        lanes.push_back(element.isInteger(1) || element.isIndex()
            ? memoryElement(element) + "(" + name(value) + ")" : name(value));
      assign(op.getResult(), valueType(op.getType()) + "(" + join(lanes) + ")");
    } else if (auto op = dyn_cast<vector::StepOp>(operation)) {
      SmallVector<std::string> lanes;
      for (int64_t lane = 0; lane < op.getType().getNumElements(); ++lane) lanes.push_back(std::to_string(lane));
      assign(op.getResult(), valueType(op.getType()) + "(" + join(lanes) + ")", true);
    } else if (auto op = dyn_cast<vector::ShuffleOp>(operation)) {
      SmallVector<std::string> lanes;
      int64_t lhsSize = cast<VectorType>(op.getV1().getType()).getNumElements();
      for (int64_t lane : op.getMask())
        lanes.push_back(name(lane < lhsSize ? op.getV1() : op.getV2()) + "[" + std::to_string(lane < lhsSize ? lane : lane - lhsSize) + "]");
      assign(op.getResult(), valueType(op.getType()) + "(" + join(lanes) + ")");
    } else if (auto op = dyn_cast<vector::ExtractElementOp>(operation)) {
      std::string expression = name(op.getVector()) + "[" + name(op.getPosition()) + "]";
      if (op.getResult().getType().isIndex()) expression = "Int(" + expression + ")";
      assign(op.getResult(), expression);
    } else if (auto op = dyn_cast<memref::AllocaOp>(operation)) {
      return allocation(operation, op.getResult(), op.getDynamicSizes(), true);
    } else if (auto op = dyn_cast<memref::AllocOp>(operation)) {
      return allocation(operation, op.getResult(), op.getDynamicSizes(), false);
    } else if (auto op = dyn_cast<memref::DeallocOp>(operation)) {
      line("dealloc(" + allocations.at(op.getMemref()) + "^)");
    } else if (auto op = dyn_cast<scf::ForOp>(operation)) {
      return forLoop(op);
    } else if (auto op = dyn_cast<scf::WhileOp>(operation)) {
      return whileLoop(op);
    } else if (auto op = dyn_cast<cpu::TaskDispatchOp>(operation)) {
      return dispatch(op);
    } else if (auto op = dyn_cast<scf::IfOp>(operation)) {
      return conditional(op);
    } else return operation->emitError("Mojo serialization has no spelling for this realized CPU operation");
    return success();
  }

  llvm::DenseMap<Value, Memory> memories;
  llvm::DenseMap<Value, std::string> allocations;
  SmallVector<std::string> scope;
};

}

LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  auto options = readCompileOptions(module);
  if (mlir::failed(options)) return failure();
  if (mlir::failed(cpu::verifyCPUProgram(module, true))) return failure();
  llvm::raw_string_ostream output(source);
  output << "from std.ffi import external_call\n"
            "from std.atomic import Atomic, Ordering\n"
            "from std.memory import Layout, alloc, dealloc, unsafe_stack_allocation, bitcast\n"
            "from std.sys import prefetch, llvm_intrinsic\n"
            "from std.sys.intrinsics import PrefetchOptions\n"
            "from std.math import fma, sqrt, exp, exp2, log, tanh, sin, cos, floor, erf, abs, min, max\n"
            "from std.runtime import initialize_runtime\n"
            "from max.algorithm import parallelize\n\n";
  Serializer serializer(output);
  llvm::json::Object interface;
  auto capabilities = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
  interface["provider"] = "mojo";
  interface["compile_options"] = serializeCompileOptions(*options);
  interface["target"] = llvm::json::Object{
      {"family", "cpu"}, {"vector_bits", capabilities.getVectorBits()},
      {"workers", capabilities.getWorkers()},
      {"private_bytes", capabilities.getPrivateBytes()},
      {"matrix_i8_i32", capabilities.getMatrixI8I32()}};
  interface["native_dependencies"] = llvm::json::Array{"std", "max"};
  interface["source_prelude_end"] = static_cast<int64_t>(output.tell());
  llvm::json::Array candidates;
  bool first = true;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    auto nativeABI = queryNativeABI(function, getPublicInterface(function), [](Type type) -> FailureOr<Type> {
      if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
      if (auto integer = dyn_cast<IntegerType>(type))
        return IntegerType::get(type.getContext(), integer.getWidth());
      if (type.isF32() || type.isF64()) return type;
      return failure();
    });
    if (mlir::failed(nativeABI)) return failure();
    int64_t sourceBegin = output.tell();
    if (mlir::failed(serializer.function(function, *nativeABI))) return failure();
    int64_t sourceEnd = output.tell();
    if (first) {
      auto abi = serializePublicInterface(function, getPublicInterface(function));
      if (mlir::failed(abi)) return failure();
      auto entry = function->getAttrOfType<cpu::EntryRequirementsAttr>(cpu::entryRequirementsAttr);
      auto requirements = queryNativeEntryRequirements(function, getPublicInterface(function),
          entry.getDisjointOutputs(), [&](unsigned, intent::ViewType view) -> FailureOr<NativeViewRequirements> {
            auto layout = entry.getContiguousViews() || view.getAccess() != 0
                ? NativeViewLayout::Contiguous : NativeViewLayout::Strided;
            return NativeViewRequirements{layout, 1};
          });
      if (mlir::failed(requirements)) return failure();
      interface["entry_name"] = function.getName();
      interface["interface"] = std::move(*abi);
      interface["native"] = llvm::json::Object{{"requirements", requirements->serialize()},
          {"slots", nativeABI->serialize()}};
      first = false;
    }
    auto configuration = function->getAttrOfType<cpu::ConfigurationAttr>("intent_cpu.configuration");
    llvm::json::Array implementations;
    for (Attribute entry : function->getAttrOfType<ArrayAttr>("intent_cpu.implementations")) {
      auto binding = cast<cpu::ImplementationAttr>(entry);
      llvm::json::Object values;
      for (NamedAttribute parameter : binding.getParameters())
        values[parameter.getName().getValue()] = cast<IntegerAttr>(parameter.getValue()).getInt();
      implementations.push_back(llvm::json::Object{{"name", binding.getName().getValue().str()}, {"parameters", std::move(values)}});
    }
    candidates.push_back(llvm::json::Object{
        {"entry", function.getName().str()},
        {"source_range", llvm::json::Array{sourceBegin, sourceEnd}},
        {"values", llvm::json::Array{configuration.getTaskGrain(),
            configuration.getTileM(), configuration.getTileN(), configuration.getTileK(), configuration.getRegionSize()}},
        {"implementations", std::move(implementations)}});
  }
  interface["candidates"] = std::move(candidates);
  llvm::raw_string_ostream metadataOutput(metadata);
  metadataOutput << llvm::json::Value(std::move(interface));
  return success();
}

}

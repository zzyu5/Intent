#include "Intent/Target/Mojo/Serialization/Serializer.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
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

std::string dtype(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) type = vector.getElementType();
  if (type.isF16()) return "float16";
  if (type.isBF16()) return "bfloat16";
  if (type.isF32()) return "float32";
  if (type.isF64()) return "float64";
  if (isa<Float8E4M3FNType>(type)) return "float8_e4m3fn";
  if (isa<Float8E5M2Type>(type)) return "float8_e5m2";
  if (type.isInteger(1)) return "bool";
  if (type.isIndex()) return "int64";
  return "int" + std::to_string(cast<IntegerType>(type).getWidth());
}

std::string memoryElement(Type type) {
  return "SIMD[DType." + dtype(type) + ", 1]";
}

std::string valueType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return "SIMD[DType." + dtype(vector) + ", " + std::to_string(vector.getNumElements()) + "]";
  if (type.isIndex()) return "Int";
  if (type.isInteger(1)) return "Bool";
  return memoryElement(type);
}

std::string ordering(AtomicOrdering order) {
  switch (order) {
  case AtomicOrdering::Relaxed: return "Ordering.RELAXED";
  case AtomicOrdering::Acquire: return "Ordering.ACQUIRE";
  case AtomicOrdering::Release: return "Ordering.RELEASE";
  case AtomicOrdering::AcquireRelease: return "Ordering.ACQUIRE_RELEASE";
  }
  llvm_unreachable("unknown atomic ordering");
}

std::string abiDType(Type type) {
  if (type.isIndex()) return "i64";
  if (isa<Float8E4M3FNType>(type)) return "f8e4m3fn";
  if (isa<Float8E5M2Type>(type)) return "f8e5m2";
  std::string result;
  llvm::raw_string_ostream out(result);
  type.print(out);
  return result;
}

std::string floatingLiteral(Type element, const llvm::APFloat &value) {
  std::string type = valueType(element);
  if (!value.isFinite())
    return "(" + type + "(" + (value.isNaN() ? "0" : value.isNegative() ? "-1" : "1") + ") / " + type + "(0))";
  if (value.isZero() && value.isNegative()) return type + "(-0.0)";
  llvm::SmallString<32> literal;
  value.toString(literal);
  return type + "(" + literal.str().str() + ")";
}

llvm::json::Array json(DenseI64ArrayAttr attribute) {
  llvm::json::Array result;
  for (int64_t value : attribute.asArrayRef()) result.push_back(value);
  return result;
}

llvm::json::Array parameters(cpu::InterfaceAttr interface) {
  llvm::json::Array result;
  for (Attribute argument : interface.getArguments()) {
    if (auto view = dyn_cast<cpu::ViewArgumentAttr>(argument)) {
      llvm::json::Array strides;
      for (Attribute constraint : view.getStrides()) {
        if (auto fixed = dyn_cast<IntegerAttr>(constraint)) strides.push_back(fixed.getInt());
        else strides.push_back(nullptr);
      }
      result.push_back(llvm::json::Object{
          {"name", view.getName().getValue().str()}, {"kind", "view"},
          {"dtype", abiDType(view.getElementType())},
          {"access", static_cast<int64_t>(view.getAccess())}, {"shape", json(view.getShape())},
          {"dimensions", json(view.getDimensions())}, {"strides", std::move(strides)},
          {"alias", view.getAlias().getValue().str()},
          {"noalias", view.getNoalias()}});
    } else {
      auto scalar = cast<cpu::ScalarArgumentAttr>(argument);
      result.push_back(llvm::json::Object{{"name", scalar.getName().getValue().str()},
          {"kind", "scalar"}, {"dtype", abiDType(scalar.getType())}});
    }
  }
  return result;
}

class Serializer {
public:
  explicit Serializer(llvm::raw_ostream &out) : out(out) {}

  LogicalResult function(func::FuncOp function) {
    names.clear(); memories.clear(); allocations.clear(); scope.clear(); next = 0;
    SmallVector<std::string> signature;
    for (auto [number, argument] : llvm::enumerate(function.getArguments())) {
      std::string name = "a" + std::to_string(number);
      bind(argument, name);
      if (auto type = dyn_cast<MemRefType>(argument.getType())) {
        signature.push_back(name + ": Pointer[" + memoryElement(type.getElementType()) + ", MutUntrackedOrigin]");
        Memory memory{name, {}, {}, name, "0"};
        SmallVector<int64_t> staticStrides;
        int64_t staticOffset;
        if (failed(type.getStridesAndOffset(staticStrides, staticOffset)))
          return function.emitError("Mojo entry requires a strided memory descriptor");
        for (int64_t axis = 0; axis < type.getRank(); ++axis) {
          std::string dimension = name + "_d" + std::to_string(axis);
          signature.push_back(dimension + ": Int64");
          scope.push_back(dimension);
          memory.sizes.push_back(type.isDynamicDim(axis) ? "Int(" + dimension + ")"
                                                        : std::to_string(type.getDimSize(axis)));
        }
        for (int64_t axis = 0; axis < type.getRank(); ++axis) {
          std::string stride = name + "_s" + std::to_string(axis);
          signature.push_back(stride + ": Int64");
          scope.push_back(stride);
          memory.strides.push_back(ShapedType::isDynamic(staticStrides[axis])
                                       ? "Int(" + stride + ")" : std::to_string(staticStrides[axis]));
        }
        memories[argument] = std::move(memory);
      } else {
        signature.push_back(name + ": " + valueType(argument.getType().isIndex() ? IntegerType::get(function.getContext(), 64) : argument.getType()));
        if (argument.getType().isIndex()) names[argument] = "Int(" + name + ")";
      }
    }
    line("@export(\"" + function.getName().str() + "\")");
    line("@no_inline");
    line("def " + function.getName().str() + "(" + join(signature) + ") abi(\"C\"):");
    ++indent;
    line("initialize_runtime()");
    if (failed(block(function.front()))) return failure();
    --indent;
    line("");
    return success();
  }

private:
  void line(const std::string &text) { out.indent(indent * 4) << text << "\n"; }

  std::string name(Value value) { return names.at(value); }

  void bind(Value value, const std::string &name) {
    names[value] = name;
    scope.push_back(name);
  }

  std::string fresh(Value value) {
    std::string result = "v" + std::to_string(next++);
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

  std::string unsignedValue(Value value) {
    Type type = getElementTypeOrSelf(value.getType());
    std::string input = value.getType().isIndex() ? "Int64(" + name(value) + ")" : name(value);
    return "bitcast[DType.uint" + std::to_string(type.isIndex() ? 64 : type.getIntOrFloatBitWidth()) + "](" + input + ")";
  }

  LogicalResult block(Block &body) {
    for (Operation &operation : body.without_terminator())
      if (failed(emit(&operation))) return failure();
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
    if (failed(block(*loop.getBody()))) return failure();
    SmallVector<std::string> nextValues;
    for (Value value : loop.getBody()->getTerminator()->getOperands()) {
      std::string temporary = "next_" + std::to_string(next++);
      line("var " + temporary + " = " + name(value));
      nextValues.push_back(temporary);
    }
    for (auto [carry, value] : llvm::zip(carries, nextValues)) line(carry + " = " + value);
    if (loop.getBody()->getOperations().size() == 1 && carries.empty()) line("pass");
    --indent;
    scope.resize(saved);
    for (auto [result, carry] : llvm::zip(loop.getResults(), carries)) names[result] = carry;
    return success();
  }

  LogicalResult dispatch(cpu::TaskDispatchOp dispatch) {
    Block &body = dispatch.getBody().front();
    SmallVector<std::string> captures;
    for (const std::string &value : scope) captures.push_back("imm " + value);
    auto saved = scope.size();
    std::string task = "task_" + std::to_string(next++);
    std::string iv = fresh(body.getArgument(0));
    line("def " + task + "(" + iv + ": Int) {" + join(captures) + "}:");
    ++indent;
    if (failed(block(body))) return failure();
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
        std::string temporary = "next_" + std::to_string(next++);
        line("var " + temporary + " = " + name(value));
        nextValues.push_back(temporary);
      }
      for (auto [value, temporary] : llvm::zip(to, nextValues)) line(name(value) + " = " + temporary);
    };
    line("while True:");
    ++indent;
    if (failed(block(before))) return failure();
    auto condition = cast<scf::ConditionOp>(before.getTerminator());
    transfer(condition.getArgs(), after.getArguments());
    line("if not " + name(condition.getCondition()) + ":");
    ++indent; line("break"); --indent;
    if (failed(block(after))) return failure();
    transfer(after.getTerminator()->getOperands(), before.getArguments());
    --indent;
    scope.resize(saved);
    for (auto [result, argument] : llvm::zip(loop.getResults(), after.getArguments())) names[result] = name(argument);
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
      if (failed(block(*body))) return failure();
      for (auto [result, value] : llvm::zip(results, body->getTerminator()->getOperands()))
        line(result + " = " + name(value));
      if (body->getOperations().size() == 1 && results.empty()) line("pass");
      return success();
    };
    line("if " + name(operation.getCondition()) + ":");
    ++indent;
    if (failed(branch(operation.thenBlock()))) return failure();
    --indent;
    scope.resize(saved);
    if (!operation.getElseRegion().empty()) {
      line("else:");
      ++indent;
      if (failed(branch(operation.elseBlock()))) return failure();
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
    if (auto op = dyn_cast<func::CallOp>(operation)) {
      if (op.getCallee() == "intent_cpu_enter_ieee") {
        assign(op.getResult(0), "external_call[\"intent_cpu_enter_ieee\", UInt32]()");
      } else if (op.getCallee() == "intent_cpu_leave_ieee") {
        line("external_call[\"intent_cpu_leave_ieee\", NoneType](" + name(op.getOperand(0)) + ")");
      } else return op.emitError("Mojo runtime call has no declared C ABI spelling");
    } else if (auto op = dyn_cast<arith::ConstantOp>(operation)) {
      if (auto integer = dyn_cast<IntegerAttr>(op.getValue())) {
        if (op.getResult().getType().isInteger(1))
          assign(op.getResult(), integer.getValue().isZero() ? "False" : "True", true);
        else
          assign(op.getResult(), valueType(op.getType()) + "(" + std::to_string(integer.getInt()) + ")", true);
      } else if (auto floating = dyn_cast<FloatAttr>(op.getValue())) {
        assign(op.getResult(), floatingLiteral(op.getType(), floating.getValue()), true);
      } else if (auto dense = dyn_cast<DenseElementsAttr>(op.getValue())) {
        SmallVector<std::string> elements;
        if (isa<FloatType>(dense.getElementType())) {
          for (llvm::APFloat value : dense.getValues<llvm::APFloat>()) {
            elements.push_back(floatingLiteral(dense.getElementType(), value));
            if (dense.isSplat()) break;
          }
        } else {
          for (llvm::APInt value : dense.getValues<llvm::APInt>()) {
            elements.push_back(dense.getElementType().isInteger(1) ? (value.isZero() ? "False" : "True") : std::to_string(value.getSExtValue()));
            if (dense.isSplat()) break;
          }
        }
        std::string fill = dense.isSplat() && dense.getElementType().isInteger(1) ? "fill=" : "";
        assign(op.getResult(), valueType(op.getType()) + "(" + fill + join(elements) + ")", true);
      } else return op.emitError("unsupported Mojo constant");
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
      names[op.getResult()] = name(op.getSource());
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
      assign(op.getResult(), name(op.getVector()) + "[" + name(op.getPosition()) + "]");
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
    } else if (auto op = dyn_cast<arith::CmpIOp>(operation)) {
      StringRef token, method;
      bool isUnsigned = false;
      switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq: token = " == "; method = "eq"; break;
      case arith::CmpIPredicate::ne: token = " != "; method = "ne"; break;
      case arith::CmpIPredicate::slt: token = " < "; method = "lt"; break;
      case arith::CmpIPredicate::sle: token = " <= "; method = "le"; break;
      case arith::CmpIPredicate::sgt: token = " > "; method = "gt"; break;
      case arith::CmpIPredicate::sge: token = " >= "; method = "ge"; break;
      case arith::CmpIPredicate::ult: token = " < "; method = "lt"; isUnsigned = true; break;
      case arith::CmpIPredicate::ule: token = " <= "; method = "le"; isUnsigned = true; break;
      case arith::CmpIPredicate::ugt: token = " > "; method = "gt"; isUnsigned = true; break;
      case arith::CmpIPredicate::uge: token = " >= "; method = "ge"; isUnsigned = true; break;
      }
      std::string lhs = isUnsigned ? unsignedValue(op.getLhs()) : name(op.getLhs());
      std::string rhs = isUnsigned ? unsignedValue(op.getRhs()) : name(op.getRhs());
      assign(op.getResult(), isa<VectorType>(op.getLhs().getType())
          ? lhs + "." + method.str() + "(" + rhs + ")" : lhs + token.str() + rhs);
    } else if (auto op = dyn_cast<arith::CmpFOp>(operation)) {
      StringRef token, method;
      switch (op.getPredicate()) {
      case arith::CmpFPredicate::OEQ: token = " == "; method = "eq"; break;
      case arith::CmpFPredicate::UNE: token = " != "; method = "ne"; break;
      case arith::CmpFPredicate::OLT: token = " < "; method = "lt"; break;
      case arith::CmpFPredicate::OLE: token = " <= "; method = "le"; break;
      case arith::CmpFPredicate::OGT: token = " > "; method = "gt"; break;
      case arith::CmpFPredicate::OGE: token = " >= "; method = "ge"; break;
      default: return op.emitError("unsupported Mojo floating comparison");
      }
      assign(op.getResult(), isa<VectorType>(op.getLhs().getType())
          ? name(op.getLhs()) + "." + method.str() + "(" + name(op.getRhs()) + ")"
          : name(op.getLhs()) + token.str() + name(op.getRhs()));
    } else if (auto op = dyn_cast<arith::SelectOp>(operation)) {
      assign(op.getResult(), isa<VectorType>(op.getCondition().getType())
          ? name(op.getCondition()) + ".select(" + name(op.getTrueValue()) + ", " + name(op.getFalseValue()) + ")"
          : name(op.getTrueValue()) + " if " + name(op.getCondition()) + " else " + name(op.getFalseValue()));
    } else if (auto op = dyn_cast<math::FmaOp>(operation)) {
      assign(op.getResult(), "fma(" + name(op.getA()) + ", " + name(op.getB()) + ", " + name(op.getC()) + ")");
    } else if (isa<math::SqrtOp, math::ExpOp, math::Exp2Op, math::LogOp, math::TanhOp,
                   math::SinOp, math::CosOp, math::FloorOp, math::ErfOp, math::AbsFOp, math::AbsIOp>(operation)) {
      StringRef function = operation->getName().stripDialect();
      if (isa<math::AbsFOp, math::AbsIOp>(operation)) function = "abs";
      assign(operation->getResult(0), function.str() + "(" + name(operation->getOperand(0)) + ")");
    } else if (isa<arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp, arith::MinimumFOp>(operation)) {
      std::string type = valueType(operation->getResult(0).getType());
      StringRef intrinsic = isa<arith::MaxNumFOp>(operation) ? "llvm.maximumnum" :
          isa<arith::MinNumFOp>(operation) ? "llvm.minimumnum" :
          isa<arith::MaximumFOp>(operation) ? "llvm.maximum" : "llvm.minimum";
      assign(operation->getResult(0), "llvm_intrinsic[\"" + intrinsic.str() + "\", " + type + "](" +
          name(operation->getOperand(0)) + ", " + name(operation->getOperand(1)) + ")");
    } else if (auto op = dyn_cast<arith::BitcastOp>(operation)) {
      assign(op.getResult(), "bitcast[DType." + dtype(op.getType()) + "](" + name(op.getIn()) + ")");
    } else if (isa<arith::DivUIOp, arith::RemUIOp, arith::ShRUIOp, arith::MinUIOp, arith::MaxUIOp>(operation)) {
      std::string lhs = unsignedValue(operation->getOperand(0)), rhs = unsignedValue(operation->getOperand(1));
      std::string expression;
      if (isa<arith::MinUIOp, arith::MaxUIOp>(operation))
        expression = std::string(isa<arith::MinUIOp>(operation) ? "min(" : "max(") + lhs + ", " + rhs + ")";
      else expression = "(" + lhs + ") " + (isa<arith::DivUIOp>(operation) ? "/" : isa<arith::RemUIOp>(operation) ? "%" : ">>") + " (" + rhs + ")";
      assign(operation->getResult(0), "bitcast[DType." + dtype(operation->getResult(0).getType()) + "](" + expression + ")");
    } else if (isa<arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, arith::ExtFOp,
                   arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(operation)) {
      Value input = operation->getOperand(0), result = operation->getResult(0);
      std::string expression = name(input);
      if (input.getType().isInteger(1)) expression = "SIMD[DType.bool, 1](" + expression + ")";
      else if (!getElementTypeOrSelf(input.getType()).isInteger(1) && isa<arith::UIToFPOp, arith::ExtUIOp>(operation))
        expression = unsignedValue(input);
      if (isa<arith::FPToUIOp>(operation)) {
        auto bits = getElementTypeOrSelf(result.getType()).getIntOrFloatBitWidth();
        expression += ".cast[DType.uint" + std::to_string(bits) + "]()";
        assign(result, "bitcast[DType." + dtype(result.getType()) + "](" + expression + ")");
      } else assign(result, expression + ".cast[DType." + dtype(result.getType()) + "]()");
    } else if (auto op = dyn_cast<arith::NegFOp>(operation)) {
      assign(op.getResult(), "-" + name(op.getOperand()));
    } else if (auto op = dyn_cast<arith::RemSIOp>(operation)) {
      std::string lhs = name(op.getLhs()), rhs = name(op.getRhs());
      std::string quotient = op.getType().isIndex() ? "Int(Int64(" + lhs + ") / Int64(" + rhs + "))"
          : "(" + lhs + ") / (" + rhs + ")";
      assign(op.getResult(), "(" + lhs + ") - (" + quotient + ") * (" + rhs + ")");
    } else if (auto op = dyn_cast<arith::DivSIOp>(operation); op && op.getType().isIndex()) {
      assign(op.getResult(), "Int(Int64(" + name(op.getLhs()) + ") / Int64(" + name(op.getRhs()) + "))");
    } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(operation)) {
      Value result = operation->getResult(0);
      Value input = operation->getOperand(0);
      bool unsignedSource = isa<arith::IndexCastUIOp>(operation) && !getElementTypeOrSelf(input.getType()).isIndex();
      std::string expression = unsignedSource ? unsignedValue(input) : name(input);
      if (isa<VectorType>(result.getType()))
        assign(result, expression + ".cast[DType." + dtype(result.getType()) + "]()");
      else assign(result, valueType(result.getType()) + "(" + expression + ")");
    } else {
      std::string token;
      if (isa<arith::AddFOp, arith::AddIOp>(operation)) token = "+";
      else if (isa<arith::SubFOp, arith::SubIOp>(operation)) token = "-";
      else if (isa<arith::MulFOp, arith::MulIOp>(operation)) token = "*";
      else if (isa<arith::DivFOp>(operation)) token = "/";
      else if (isa<math::PowFOp>(operation)) token = "**";
      else if (isa<arith::DivSIOp>(operation)) token = "/";
      else if (isa<arith::AndIOp>(operation)) token = "&";
      else if (isa<arith::OrIOp>(operation)) token = "|";
      else if (isa<arith::XOrIOp>(operation)) token = "^";
      else if (isa<arith::ShLIOp>(operation)) token = "<<";
      else if (isa<arith::ShRSIOp>(operation)) token = ">>";
      if (!token.empty()) {
        assign(operation->getResult(0), "(" + name(operation->getOperand(0)) + ") " + token + " (" + name(operation->getOperand(1)) + ")");
      } else if (isa<arith::MinSIOp, arith::MaxSIOp>(operation)) {
        assign(operation->getResult(0), std::string(isa<arith::MinSIOp>(operation) ? "min(" : "max(") + name(operation->getOperand(0)) + ", " + name(operation->getOperand(1)) + ")");
      } else return operation->emitError("Mojo serialization has no spelling for this realized CPU operation");
    }
    return success();
  }

  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> names;
  llvm::DenseMap<Value, Memory> memories;
  llvm::DenseMap<Value, std::string> allocations;
  SmallVector<std::string> scope;
  unsigned indent = 0, next = 0;
};

}

LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  if (failed(cpu::verifyCPUProgram(module, true))) return failure();
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
  interface["source_prelude_end"] = static_cast<int64_t>(output.tell());
  llvm::json::Array candidates;
  bool first = true;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    int64_t sourceBegin = output.tell();
    if (failed(serializer.function(function))) return failure();
    int64_t sourceEnd = output.tell();
    if (first) {
      auto abi = function->getAttrOfType<cpu::InterfaceAttr>("intent_cpu.interface");
      interface["parameters"] = parameters(abi);
      interface["workers"] = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities").getWorkers();
      interface["contiguous_views"] = abi.getContiguousViews();
      interface["disjoint_outputs"] = abi.getDisjointOutputs();
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

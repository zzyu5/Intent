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
};

std::string join(ArrayRef<std::string> values, llvm::StringRef separator = ", ") {
  return llvm::join(values, separator);
}

std::string memoryElement(Type type) {
  if (type.isF32()) return "Float32";
  if (type.isInteger(1)) return "SIMD[DType.bool, 1]";
  assert(type.isIndex() || type.isInteger(64));
  return "Int64";
}

std::string floatingLiteral(const llvm::APFloat &value) {
  if (!value.isFinite())
    return std::string("(Float32(") + (value.isNaN() ? "0" : value.isNegative() ? "-1" : "1") + ") / Float32(0))";
  llvm::SmallString<32> literal;
  value.toString(literal);
  return "Float32(" + literal.str().str() + ")";
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
      result.push_back(llvm::json::Object{
          {"name", view.getName().getValue().str()}, {"kind", "view"},
          {"access", static_cast<int64_t>(view.getAccess())}, {"shape", json(view.getShape())},
          {"dimensions", json(view.getDimensions())}, {"alias", view.getAlias().getValue().str()},
          {"noalias", view.getNoalias()}});
    } else {
      auto scalar = cast<cpu::ScalarArgumentAttr>(argument);
      result.push_back(llvm::json::Object{{"name", scalar.getName().getValue().str()},
          {"kind", "scalar"}, {"dtype", scalar.getType().isF32() ? "f32" : "i64"}});
    }
  }
  return result;
}

class Serializer {
public:
  explicit Serializer(llvm::raw_ostream &out) : out(out) {}

  LogicalResult function(func::FuncOp function) {
    names.clear(); memories.clear(); allocations.clear(); scope.clear(); next = 0;
    workers = function->getParentOfType<ModuleOp>()->getAttrOfType<cpu::CapabilitiesAttr>(
        "intent_cpu.capabilities").getWorkers();
    SmallVector<std::string> signature;
    for (auto [number, argument] : llvm::enumerate(function.getArguments())) {
      std::string name = "a" + std::to_string(number);
      bind(argument, name);
      if (auto type = dyn_cast<MemRefType>(argument.getType())) {
        signature.push_back(name + ": Pointer[Float32, MutUntrackedOrigin]");
        Memory memory{name, {}, {}};
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
        signature.push_back(name + ": " + (argument.getType().isF32() ? "Float32" : "Int64"));
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

  LogicalResult parallel(scf::ParallelOp parallel) {
    if (parallel.getNumLoops() != 1 || parallel.getNumResults())
      return parallel.emitError("Mojo serialization requires a realized one-dimensional task region");
    SmallVector<std::string> captures;
    for (const std::string &value : scope) captures.push_back("imm " + value);
    auto saved = scope.size();
    std::string task = "task_" + std::to_string(next++);
    std::string iv = fresh(parallel.getInductionVars()[0]);
    line("def " + task + "(" + iv + ": Int) {" + join(captures) + "}:");
    ++indent;
    if (failed(block(*parallel.getBody()))) return failure();
    --indent;
    scope.resize(saved);
    line("parallelize(" + task + ", " + name(parallel.getUpperBound()[0]) + ", " + std::to_string(workers) + ")");
    return success();
  }

  LogicalResult conditional(scf::IfOp operation) {
    if (operation.getNumResults()) return operation.emitError("Mojo conditional SSA results are not implemented");
    auto saved = scope.size();
    line("if " + name(operation.getCondition()) + ":");
    ++indent;
    if (failed(block(*operation.thenBlock()))) return failure();
    if (operation.thenBlock()->getOperations().size() == 1) line("pass");
    --indent;
    scope.resize(saved);
    if (!operation.getElseRegion().empty()) {
      line("else:");
      ++indent;
      if (failed(block(*operation.elseBlock()))) return failure();
      if (operation.elseBlock()->getOperations().size() == 1) line("pass");
      --indent;
      scope.resize(saved);
    }
    return success();
  }

  LogicalResult allocation(Operation *operation, Value memory, ValueRange dynamicSizes, bool stack) {
    auto type = cast<MemRefType>(memory.getType());
    if (stack && !type.hasStaticShape())
      return operation->emitError("Mojo stack allocation requires static extents");
    std::string element = memoryElement(type.getElementType());
    std::string value = fresh(memory);
    std::string storage = "storage_" + value;
    Memory descriptor{value, {}, {}};
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
      int64_t alignment = cast<memref::AllocaOp>(operation).getAlignment().value_or(4);
      line("var " + value + " = unsafe_stack_allocation[" +
          std::to_string(type.getNumElements()) + ", " + element + ", alignment=" +
          std::to_string(alignment) + "]()");
    } else {
      line("var " + storage + " = alloc(Layout[" + element + "](count=" + stride + "))");
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
          assign(op.getResult(), std::string(op.getResult().getType().isIndex() ? "Int(" : "Int64(") + std::to_string(integer.getInt()) + ")", true);
      } else if (auto floating = dyn_cast<FloatAttr>(op.getValue())) {
        assign(op.getResult(), floatingLiteral(floating.getValue()), true);
      } else if (auto dense = dyn_cast<DenseFPElementsAttr>(op.getValue())) {
        SmallVector<std::string> elements;
        for (llvm::APFloat value : dense.getValues<llvm::APFloat>()) {
          elements.push_back(floatingLiteral(value));
          if (dense.isSplat()) break;
        }
        assign(op.getResult(), "SIMD[DType.float32, " +
            std::to_string(cast<VectorType>(op.getResult().getType()).getNumElements()) + "](" + join(elements) + ")", true);
      } else return op.emitError("unsupported Mojo constant");
    } else if (auto op = dyn_cast<memref::DimOp>(operation)) {
      auto axis = op.getConstantIndex();
      if (!axis) return op.emitError("Mojo memory descriptor dimension must be static");
      assign(op.getResult(), memories.at(op.getSource()).sizes[*axis]);
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
    } else if (auto op = dyn_cast<vector::LoadOp>(operation)) {
      assign(op.getResult(), pointer(op.getBase(), op.getIndices()) + ".unsafe_load[width=" + std::to_string(op.getVectorType().getNumElements()) + "]()");
    } else if (auto op = dyn_cast<vector::StoreOp>(operation)) {
      line(pointer(op.getBase(), op.getIndices()) + ".unsafe_store(" + name(op.getValueToStore()) + ")");
    } else if (auto op = dyn_cast<memref::PrefetchOp>(operation)) {
      line("prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](" +
          pointer(op.getMemref(), op.getIndices()) + ")");
    } else if (auto op = dyn_cast<vector::BroadcastOp>(operation)) {
      assign(op.getResult(), "SIMD[DType.float32, " + std::to_string(cast<VectorType>(op.getResult().getType()).getNumElements()) + "](" + name(op.getSource()) + ")");
    } else if (auto op = dyn_cast<vector::ShuffleOp>(operation)) {
      SmallVector<std::string> lanes;
      int64_t lhsSize = cast<VectorType>(op.getV1().getType()).getNumElements();
      for (int64_t lane : op.getMask())
        lanes.push_back(name(lane < lhsSize ? op.getV1() : op.getV2()) + "[" + std::to_string(lane < lhsSize ? lane : lane - lhsSize) + "]");
      assign(op.getResult(), "SIMD[DType.float32, " + std::to_string(lanes.size()) + "](" + join(lanes) + ")");
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
      if (op.getNumResults() || op.getNumOperands()) return op.emitError("Mojo while requires realized destination-passing state");
      auto saved = scope.size();
      line("while True:");
      ++indent;
      if (failed(block(op.getBefore().front()))) return failure();
      line("if not " + name(cast<scf::ConditionOp>(op.getBefore().front().getTerminator()).getCondition()) + ":");
      ++indent; line("break"); --indent;
      if (failed(block(op.getAfter().front()))) return failure();
      --indent;
      scope.resize(saved);
    } else if (auto op = dyn_cast<scf::ParallelOp>(operation)) {
      return parallel(op);
    } else if (auto op = dyn_cast<scf::IfOp>(operation)) {
      return conditional(op);
    } else if (auto op = dyn_cast<arith::CmpIOp>(operation)) {
      StringRef token;
      switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq: token = " == "; break;
      case arith::CmpIPredicate::ne: token = " != "; break;
      case arith::CmpIPredicate::slt: token = " < "; break;
      case arith::CmpIPredicate::sle: token = " <= "; break;
      case arith::CmpIPredicate::sgt: token = " > "; break;
      case arith::CmpIPredicate::sge: token = " >= "; break;
      default: return op.emitError("unsupported Mojo unsigned comparison");
      }
      assign(op.getResult(), name(op.getLhs()) + token.str() + name(op.getRhs()));
    } else if (auto op = dyn_cast<arith::CmpFOp>(operation)) {
      StringRef token;
      switch (op.getPredicate()) {
      case arith::CmpFPredicate::OEQ: token = " == "; break;
      case arith::CmpFPredicate::UNE: token = " != "; break;
      case arith::CmpFPredicate::OLT: token = " < "; break;
      case arith::CmpFPredicate::OLE: token = " <= "; break;
      case arith::CmpFPredicate::OGT: token = " > "; break;
      case arith::CmpFPredicate::OGE: token = " >= "; break;
      default: return op.emitError("unsupported Mojo floating comparison");
      }
      assign(op.getResult(), name(op.getLhs()) + token.str() + name(op.getRhs()));
    } else if (auto op = dyn_cast<arith::SelectOp>(operation)) {
      assign(op.getResult(), isa<VectorType>(op.getCondition().getType())
          ? name(op.getCondition()) + ".select(" + name(op.getTrueValue()) + ", " + name(op.getFalseValue()) + ")"
          : name(op.getTrueValue()) + " if " + name(op.getCondition()) + " else " + name(op.getFalseValue()));
    } else if (auto op = dyn_cast<math::FmaOp>(operation)) {
      assign(op.getResult(), "fma(" + name(op.getA()) + ", " + name(op.getB()) + ", " + name(op.getC()) + ")");
    } else if (auto op = dyn_cast<math::SqrtOp>(operation)) {
      assign(op.getResult(), "sqrt(" + name(op.getOperand()) + ")");
    } else if (auto op = dyn_cast<math::ExpOp>(operation)) {
      assign(op.getResult(), "exp(" + name(op.getOperand()) + ")");
    } else if (auto op = dyn_cast<math::Exp2Op>(operation)) {
      assign(op.getResult(), "exp2(" + name(op.getOperand()) + ")");
    } else if (isa<arith::MaxNumFOp, arith::MaximumFOp, arith::MinimumFOp>(operation)) {
      auto vector = dyn_cast<VectorType>(operation->getResult(0).getType());
      std::string type = vector ? "SIMD[DType.float32, " + std::to_string(vector.getNumElements()) + "]" : "Float32";
      StringRef intrinsic = isa<arith::MaxNumFOp>(operation) ? "llvm.maximumnum" :
          isa<arith::MaximumFOp>(operation) ? "llvm.maximum" : "llvm.minimum";
      assign(operation->getResult(0), "llvm_intrinsic[\"" + intrinsic.str() + "\", " + type + "](" +
          name(operation->getOperand(0)) + ", " + name(operation->getOperand(1)) + ")");
    } else if (auto op = dyn_cast<arith::SIToFPOp>(operation)) {
      assign(op.getResult(), "Float32(" + name(op.getIn()) + ")");
    } else if (auto op = dyn_cast<arith::NegFOp>(operation)) {
      assign(op.getResult(), "-" + name(op.getOperand()));
    } else if (isa<arith::IndexCastOp>(operation)) {
      assign(operation->getResult(0), std::string(operation->getResult(0).getType().isIndex() ? "Int(" : "Int64(") + name(operation->getOperand(0)) + ")");
    } else {
      std::string token;
      if (isa<arith::AddFOp, arith::AddIOp>(operation)) token = "+";
      else if (isa<arith::SubFOp, arith::SubIOp>(operation)) token = "-";
      else if (isa<arith::MulFOp, arith::MulIOp>(operation)) token = "*";
      else if (isa<arith::DivFOp>(operation)) token = "/";
      else if (isa<arith::DivSIOp, arith::FloorDivSIOp>(operation)) token = "//";
      else if (isa<arith::RemSIOp>(operation)) token = "%";
      else if (isa<arith::AndIOp>(operation)) token = "&";
      else if (isa<arith::OrIOp>(operation)) token = "|";
      else if (isa<arith::XOrIOp>(operation)) token = "^";
      if (!token.empty()) {
        assign(operation->getResult(0), "(" + name(operation->getOperand(0)) + ") " + token + " (" + name(operation->getOperand(1)) + ")");
      } else if (isa<arith::MinSIOp, arith::MaxSIOp>(operation)) {
        assign(operation->getResult(0), std::string(isa<arith::MinSIOp>(operation) ? "min(" : "max(") + name(operation->getOperand(0)) + ", " + name(operation->getOperand(1)) + ")");
      } else if (auto op = dyn_cast<arith::CeilDivSIOp>(operation)) {
        assign(op.getResult(), "(" + name(op.getLhs()) + " + " + name(op.getRhs()) + " - 1) // " + name(op.getRhs()));
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
  int64_t workers = 0;
};

}

LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  if (failed(cpu::verifyCPUProgram(module, true))) return failure();
  llvm::raw_string_ostream output(source);
  output << "from std.ffi import external_call\n"
            "from std.memory import Layout, alloc, dealloc, unsafe_stack_allocation\n"
            "from std.sys import prefetch, llvm_intrinsic\n"
            "from std.sys.intrinsics import PrefetchOptions\n"
            "from std.math import fma, sqrt, exp, exp2, min, max\n"
            "from std.runtime import initialize_runtime\n"
            "from max.algorithm import parallelize\n\n";
  Serializer serializer(output);
  llvm::json::Object interface;
  llvm::json::Array candidates;
  bool first = true;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function.isExternal()) continue;
    if (failed(serializer.function(function))) return failure();
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

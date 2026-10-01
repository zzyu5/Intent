#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "TaskABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace intent::weft_provider {

FailureOr<NativeABI> queryHostABI(func::FuncOp function) {
  return queryNativeABI(function, getPublicInterface(function), [](Type type) -> FailureOr<Type> {
    if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
    if (type.isInteger(1)) return IntegerType::get(type.getContext(), 32);
    if (type.isSignlessInteger() || type.isF32() || type.isF64()) return type;
    return failure();
  });
}

namespace {

struct Memory {
  std::string base;
  std::string offset;
  SmallVector<std::string> shape, strides;
};

class HostSerializer {
public:
  HostSerializer(ModuleOp module, llvm::raw_ostream &out) : module(module), out(out) {}

  LogicalResult run() {
    out << "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <math.h>\n";
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function.isExternal()) continue;
      out << "extern void " << function.getName() << "(";
      for (auto [i, type] : llvm::enumerate(function.getArgumentTypes())) {
        if (i) out << ", ";
        out << (isa<MemRefType>(type) ? "void *" : scalarType(type));
      }
      out << ");\n";
    }
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.isExternal()) continue;
      auto abi = queryHostABI(function);
      if (failed(abi)) return failure();
      values.clear(); memories.clear(); allocations.clear();
      out << "void " << function.getName() << "(";
      bool first = true;
      auto parameter = [&](const std::string &type, const std::string &name) {
        if (!first) out << ", ";
        out << type << " " << name; first = false;
      };
      for (auto [i, argument] : llvm::enumerate(function.getArguments())) {
        std::string name = "a" + std::to_string(i);
        if (auto type = dyn_cast<MemRefType>(argument.getType()))
          memories[argument] = Memory{name, "0", SmallVector<std::string>(type.getRank()),
                                     SmallVector<std::string>(type.getRank())};
        else values[argument] = name;
      }
      for (const NativeSlot &slot : abi->slots) {
        std::string name = slot.name();
        parameter(slot.role == NativeSlotRole::Pointer
            ? scalarType(slot.element) + " *" : scalarType(slot.carrier), name);
        if (slot.axis) {
          auto &memory = memories.find(function.getArgument(slot.parameter))->second;
          auto &entries = slot.role == NativeSlotRole::Extent ? memory.shape : memory.strides;
          entries[*slot.axis] = name;
        }
      }
      out << ") {\n";
      indent = 1;
      if (failed(block(function.front()))) return failure();
      out << "}\n";
    }
    return success();
  }

private:
  std::string scalarType(Type type) {
    if (type.isF32()) return "float";
    if (type.isF64()) return "double";
    if (type.isIndex()) return "int64_t";
    if (auto integer = dyn_cast<IntegerType>(type)) {
      if (integer.getWidth() == 1) return "int";
      return (integer.isUnsigned() ? "uint" : "int") + std::to_string(integer.getWidth()) + "_t";
    }
    llvm_unreachable("host scalar type must be legalized before serialization");
  }
  void line(const std::string &text) { out.indent(indent * 2) << text << '\n'; }
  std::string fresh() { return "v" + std::to_string(counter++); }
  std::string value(Value input) { return values.at(input); }
  std::string bound(OpFoldResult input) {
    return isa<Attribute>(input) ? std::to_string(cast<IntegerAttr>(cast<Attribute>(input)).getInt())
                                 : value(cast<Value>(input));
  }
  std::string address(Value memory, ValueRange indices) {
    auto &entry = memories.at(memory);
    std::string result = pointer(memory);
    for (auto [index, stride] : llvm::zip(indices, entry.strides))
      result += " + (" + value(index) + ") * (" + stride + ")";
    return "(" + result + ")";
  }
  std::string pointer(Value memory) {
    const auto &entry = memories.at(memory);
    return entry.offset == "0" ? entry.base : "(" + entry.base + " + (" + entry.offset + "))";
  }

  LogicalResult view(Operation *operation) {
    if (auto cast = dyn_cast<memref::CastOp>(operation)) {
      if (!isa<MemRefType>(cast.getSource().getType()) ||
          !isa<MemRefType>(cast.getResult().getType()))
        return cast.emitError("native host cast requires ranked memory descriptors");
      Memory source = memories.at(cast.getSource());
      memories[cast.getResult()] = std::move(source);
      if (auto allocation = allocations.find(cast.getSource()); allocation != allocations.end()) {
        std::string owner = allocation->second;
        allocations[cast.getResult()] = std::move(owner);
      }
      return success();
    }
    if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(operation)) {
      const Memory source = memories.at(metadata.getSource());
      memories[metadata.getBaseBuffer()] = Memory{source.base, "0", {}, {}};
      values[metadata.getOffset()] = source.offset;
      for (auto [result, size] : llvm::zip(metadata.getSizes(), source.shape)) values[result] = size;
      for (auto [result, stride] : llvm::zip(metadata.getStrides(), source.strides)) values[result] = stride;
      return success();
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(operation)) {
      if (!isa<MemRefType>(reinterpret.getSource().getType()))
        return reinterpret.emitError("native host reinterpretation requires a ranked source descriptor");
      // Reinterpretation replaces metadata relative to the storage base. It
      // must not add the source view's offset to the replacement offset.
      Memory target{memories.at(reinterpret.getSource()).base,
                    bound(reinterpret.getMixedOffsets().front()), {}, {}};
      for (OpFoldResult size : reinterpret.getMixedSizes()) target.shape.push_back(bound(size));
      for (OpFoldResult stride : reinterpret.getMixedStrides()) target.strides.push_back(bound(stride));
      memories[reinterpret.getResult()] = std::move(target);
      return success();
    }
    auto subview = cast<memref::SubViewOp>(operation);
    const Memory source = memories.at(subview.getSource());
    Memory target{source.base, source.offset, {}, {}};
    auto offsets = subview.getMixedOffsets(), sizes = subview.getMixedSizes(), steps = subview.getMixedStrides();
    auto dropped = subview.getDroppedDims();
    for (unsigned axis = 0; axis < offsets.size(); ++axis) {
      target.offset += " + (" + bound(offsets[axis]) + ") * (" + source.strides[axis] + ")";
      if (!dropped.test(axis)) {
        target.shape.push_back(bound(sizes[axis]));
        target.strides.push_back("(" + source.strides[axis] + ") * (" + bound(steps[axis]) + ")");
      }
    }
    target.offset = "(" + target.offset + ")";
    memories[subview.getResult()] = std::move(target);
    return success();
  }
  void bind(Value result, const std::string &expression) {
    std::string name = fresh();
    line(scalarType(result.getType()) + " " + name + " = " + expression + ";");
    values[result] = name;
  }
  LogicalResult block(Block &body) {
    for (auto &operation : body)
      if (failed(emit(&operation))) return failure();
    return success();
  }
  LogicalResult emit(Operation *operation) {
    if (isa<scf::YieldOp, scf::ReduceOp>(operation)) return success();
    if (isa<func::ReturnOp>(operation)) { line("return;"); return success(); }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      std::string expression;
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) expression = std::to_string(integer.getInt());
      else if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
        if (floating.getValue().isInfinity()) expression = floating.getValue().isNegative() ? "-INFINITY" : "INFINITY";
        else {
          llvm::SmallString<32> text;
          floating.getValue().toString(text); expression = "(" + scalarType(constant.getType()) + ")(" + text.str().str() + ")";
        }
      } else return operation->emitError("host constant has no scalar C representation");
      values[constant.getResult()] = expression;
      return success();
    }
    if (auto dim = dyn_cast<memref::DimOp>(operation)) {
      auto axis = dim.getConstantIndex();
      if (!axis) return dim.emitError("host dimension must name a constant axis");
      values[dim.getResult()] = memories.at(dim.getSource()).shape[*axis];
      return success();
    }
    if (isa<memref::CastOp, memref::ExtractStridedMetadataOp,
            memref::ReinterpretCastOp, memref::SubViewOp>(operation)) return view(operation);
    if (isa<memref::AllocOp, memref::AllocaOp>(operation)) {
      auto type = cast<MemRefType>(operation->getResult(0).getType());
      if (!type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != 0)
        return operation->emitError("native host allocation requires contiguous host storage");
      std::string name = fresh(), elements = "1";
      Memory memory{name, "0", {}, {}};
      unsigned dynamic = 0;
      for (int64_t size : type.getShape())
        memory.shape.push_back(ShapedType::isDynamic(size) ? value(operation->getOperand(dynamic++)) : std::to_string(size));
      memory.strides.resize(type.getRank());
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        memory.strides[axis] = elements;
        elements = "(" + elements + ") * (" + memory.shape[axis] + ")";
      }
      auto element = scalarType(type.getElementType());
      int64_t alignment = 16;
      if (auto attr = operation->getAttrOfType<IntegerAttr>("alignment")) alignment = std::max(alignment, attr.getInt());
      if (isa<memref::AllocaOp>(operation))
        line("_Alignas(" + std::to_string(alignment) + ") " + element + " " + name + "[" + elements + "];");
      else {
        std::string bytes = "sizeof(" + element + ") * (" + elements + ")";
        line(element + " *" + name + " = aligned_alloc(" + std::to_string(alignment) + ", ((" + bytes +
            " + " + std::to_string(alignment - 1) + ") / " + std::to_string(alignment) + ") * " + std::to_string(alignment) + ");");
        line("if (!" + name + " && (" + elements + ")) abort();");
        allocations[operation->getResult(0)] = name;
      }
      memories[operation->getResult(0)] = memory; return success();
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) {
      auto allocation = allocations.find(dealloc.getMemref());
      if (allocation == allocations.end())
        return dealloc.emitError("native host deallocation must reference its owning heap allocation");
      line("free(" + allocation->second + ");"); return success();
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      bind(load.getResult(), "*" + address(load.getMemref(), load.getIndices())); return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      line("*" + address(store.getMemref(), store.getIndices()) + " = " + value(store.getValue()) + ";"); return success();
    }
    if (auto call = dyn_cast<func::CallOp>(operation)) {
      if (call.getNumResults()) return call.emitError("native task call must return through explicit storage");
      std::string text = call.getCallee().str() + "(";
      for (auto [i, argument] : llvm::enumerate(call.getOperands())) {
        if (i) text += ", ";
        text += isa<MemRefType>(argument.getType()) ? pointer(argument) : value(argument);
      }
      line(text + ");"); return success();
    }
    if (auto loop = dyn_cast<scf::ParallelOp>(operation)) {
      if (loop.getNumLoops() != 1 || loop.getNumResults()) return loop.emitError("native task loop requires one linear workset");
      auto caps = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
      if (caps.getWorkers() != 1) line("#pragma omp parallel for num_threads(" + std::to_string(caps.getWorkers()) + ")");
      std::string iv = fresh(); values[loop.getInductionVars()[0]] = iv;
      line("for (int64_t " + iv + " = " + value(loop.getLowerBound()[0]) + "; " + iv + " < " +
          value(loop.getUpperBound()[0]) + "; " + iv + " += " + value(loop.getStep()[0]) + ") {");
      ++indent; if (failed(block(*loop.getBody()))) return failure(); --indent; line("}"); return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      for (auto [argument, initial] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) bind(argument, value(initial));
      std::string iv = fresh(); values[loop.getInductionVar()] = iv;
      line("for (int64_t " + iv + " = " + value(loop.getLowerBound()) + "; " + iv + " < " + value(loop.getUpperBound()) +
          "; " + iv + " += " + value(loop.getStep()) + ") {");
      ++indent; if (failed(block(*loop.getBody()))) return failure();
      SmallVector<std::string> next;
      for (Value yielded : loop.getBody()->getTerminator()->getOperands()) {
        next.push_back(fresh());
        line(scalarType(yielded.getType()) + " " + next.back() + " = " + value(yielded) + ";");
      }
      for (auto [argument, yielded] : llvm::zip(loop.getRegionIterArgs(), next))
        line(value(argument) + " = " + yielded + ";");
      --indent; line("}");
      for (auto [result, argument] : llvm::zip(loop.getResults(), loop.getRegionIterArgs())) values[result] = value(argument);
      return success();
    }
    if (auto condition = dyn_cast<scf::IfOp>(operation)) {
      for (Value result : condition.getResults()) {
        std::string name = fresh();
        line(scalarType(result.getType()) + " " + name + ";");
        values[result] = name;
      }
      auto branch = [&](Block &body) {
        if (failed(block(body))) return failure();
        for (auto [result, yielded] : llvm::zip(condition.getResults(), body.getTerminator()->getOperands()))
          line(value(result) + " = " + value(yielded) + ";");
        return success();
      };
      line("if (" + value(condition.getCondition()) + ") {");
      ++indent; if (failed(branch(*condition.thenBlock()))) return failure(); --indent;
      if (!condition.getElseRegion().empty()) {
        line("} else {"); ++indent;
        if (failed(branch(*condition.elseBlock()))) return failure(); --indent;
      }
      line("}"); return success();
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(operation)) {
      bind(cast.getResult(), "(" + scalarType(cast.getType()) + ")(" + value(cast.getIn()) + ")"); return success();
    }
    if (operation->getNumOperands() == 2 && operation->getNumResults() == 1) {
      auto a = value(operation->getOperand(0)), c = value(operation->getOperand(1));
      std::string op;
      if (isa<arith::AddIOp, arith::AddFOp>(operation)) op = "+";
      else if (isa<arith::SubIOp, arith::SubFOp>(operation)) op = "-";
      else if (isa<arith::MulIOp, arith::MulFOp>(operation)) op = "*";
      else if (isa<arith::DivSIOp, arith::DivFOp>(operation)) op = "/";
      else if (isa<arith::RemSIOp>(operation)) op = "%";
      else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
        if (compare.getPredicate() == arith::CmpIPredicate::eq) op = "==";
      }
      std::string expression;
      if (!op.empty()) expression = "(" + a + ") " + op + " (" + c + ")";
      else if (isa<arith::CeilDivSIOp>(operation)) expression = "((" + a + ") + (" + c + ") - 1) / (" + c + ")";
      else if (isa<arith::MinSIOp, arith::MaxSIOp>(operation)) expression = "(" + a + (isa<arith::MinSIOp>(operation) ? " < " : " > ") + c + ") ? " + a + " : " + c;
      if (!expression.empty()) { bind(operation->getResult(0), expression); return success(); }
    }
    return operation->emitError("operation has no legalized native host serialization: ") << operation->getName();
  }

  ModuleOp module;
  llvm::raw_ostream &out;
  llvm::DenseMap<Value, std::string> values;
  llvm::DenseMap<Value, Memory> memories;
  llvm::DenseMap<Value, std::string> allocations;
  unsigned counter = 0, indent = 0;
};

}

LogicalResult serializeHostProgram(ModuleOp program, std::string &source) {
  if (failed(verify(program))) return failure();
  llvm::raw_string_ostream stream(source);
  return HostSerializer(program, stream).run();
}

}

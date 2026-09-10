#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace intent::weft_provider {
namespace {

struct Memory {
  std::string pointer;
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
      values.clear(); memories.clear();
      out << "void " << function.getName() << "(";
      bool first = true;
      auto parameter = [&](const std::string &type, const std::string &name) {
        if (!first) out << ", ";
        out << type << " " << name; first = false;
      };
      for (auto [i, argument] : llvm::enumerate(function.getArguments())) {
        std::string name = "a" + std::to_string(i);
        if (auto type = dyn_cast<MemRefType>(argument.getType())) {
          parameter(scalarType(type.getElementType()) + " *", name);
          Memory memory{name, {}, {}};
          for (int64_t axis = 0; axis < type.getRank(); ++axis) {
            std::string extent = name + "_d" + std::to_string(axis);
            parameter("int64_t", extent); memory.shape.push_back(extent);
          }
          for (int64_t axis = 0; axis < type.getRank(); ++axis) {
            std::string stride = name + "_s" + std::to_string(axis);
            parameter("int64_t", stride); memory.strides.push_back(stride);
          }
          memories[argument] = memory;
        } else { parameter(scalarType(argument.getType()), name); values[argument] = name; }
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
    std::string result = entry.pointer;
    for (auto [index, stride] : llvm::zip(indices, entry.strides))
      result += " + (" + value(index) + ") * (" + stride + ")";
    return "(" + result + ")";
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
    if (auto cast = dyn_cast<memref::CastOp>(operation)) {
      memories[cast.getResult()] = memories.at(cast.getSource()); return success();
    }
    if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
      const auto source = memories.at(view.getSource());
      Memory target{source.pointer, {}, {}};
      auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes(), steps = view.getMixedStrides();
      auto dropped = view.getDroppedDims();
      for (unsigned axis = 0; axis < offsets.size(); ++axis) {
        target.pointer += " + (" + bound(offsets[axis]) + ") * (" + source.strides[axis] + ")";
        if (!dropped.test(axis)) {
          target.shape.push_back(bound(sizes[axis]));
          target.strides.push_back("(" + source.strides[axis] + ") * (" + bound(steps[axis]) + ")");
        }
      }
      target.pointer = "(" + target.pointer + ")";
      memories[view.getResult()] = target; return success();
    }
    if (isa<memref::AllocOp, memref::AllocaOp>(operation)) {
      auto type = cast<MemRefType>(operation->getResult(0).getType());
      std::string name = fresh(), elements = "1";
      Memory memory{name, {}, {}};
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
      }
      memories[operation->getResult(0)] = memory; return success();
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) {
      line("free(" + memories.at(dealloc.getMemref()).pointer + ");"); return success();
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
        text += isa<MemRefType>(argument.getType()) ? memories.at(argument).pointer : value(argument);
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
      for (auto [argument, yielded] : llvm::zip(loop.getRegionIterArgs(), loop.getBody()->getTerminator()->getOperands()))
        line(value(argument) + " = " + value(yielded) + ";");
      --indent; line("}");
      for (auto [result, argument] : llvm::zip(loop.getResults(), loop.getRegionIterArgs())) values[result] = value(argument);
      return success();
    }
    if (auto condition = dyn_cast<scf::IfOp>(operation)) {
      if (condition.getNumResults()) return condition.emitError("host conditional results require prior materialization");
      line("if (" + value(condition.getCondition()) + ") {");
      ++indent; if (failed(block(*condition.thenBlock()))) return failure(); --indent;
      if (!condition.getElseRegion().empty()) {
        line("} else {"); ++indent;
        if (failed(block(*condition.elseBlock()))) return failure(); --indent;
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
  unsigned counter = 0, indent = 0;
};

}

LogicalResult serializeHostProgram(ModuleOp program, std::string &source) {
  if (failed(verify(program))) return failure();
  llvm::raw_string_ostream stream(source);
  return HostSerializer(program, stream).run();
}

}

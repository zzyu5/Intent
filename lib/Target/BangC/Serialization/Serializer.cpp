#include "Intent/Target/BangC/Passes.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
using namespace mlir;
namespace intent::bangc {
namespace {
#include "../Runtime/TileImplementations.inc"
std::string ctype(Type type) {
  if (type.isF16()) return "half";
  if (type.isF32()) return "float";
  if (type.isF64()) return "double";
  if (type.isInteger(1)) return "bool";
  if (type.isInteger(32)) return "int32_t";
  return "int64_t";
}
std::string dtype(Type type) {
  if (type.isF16()) return "f16";
  if (type.isF32()) return "f32";
  if (type.isInteger(1)) return "bool";
  if (type.isInteger(32)) return "i32";
  return "i64";
}
class Serializer {
public:
  Serializer(func::FuncOp function, llvm::raw_ostream &output) : function(function), out(output) {}
  LogicalResult emit(llvm::json::Object &metadata) {
    auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
    auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
    llvm::json::Array parameters;
    for (auto [i, attribute] : llvm::enumerate(interface.getArguments())) {
      Value argument = function.getArgument(i);
      std::string name = "a" + std::to_string(i);
      names[argument] = name;
      if (auto view = dyn_cast<dsa::ViewArgumentAttr>(attribute)) {
        signature.push_back((view.getAccess() == 0 ? "const " : "") + ctype(view.getElement()) + " *" + name);
        call.push_back(name);
        llvm::json::Array shape, dimensions, constraints;
        for (int64_t value : view.getShape().asArrayRef()) shape.push_back(value);
        for (int64_t value : view.getDimensions().asArrayRef()) dimensions.push_back(value);
        for (Attribute value : view.getConstraints().getStrides()) {
          if (auto fixed = dyn_cast<IntegerAttr>(value)) constraints.push_back(fixed.getInt());
          else constraints.push_back(nullptr);
        }
        parameters.push_back(llvm::json::Object{{"kind", "view"}, {"name", view.getName().getValue()},
            {"dtype", dtype(view.getElement())}, {"shape", std::move(shape)}, {"dimensions", std::move(dimensions)},
            {"access", static_cast<int64_t>(view.getAccess())}, {"alias", view.getConstraints().getAlias().getValue()},
            {"noalias", view.getConstraints().getNoalias()}, {"strides", std::move(constraints)}});
        for (int64_t axis = 0; axis < cast<MemRefType>(argument.getType()).getRank(); ++axis) {
          std::string dimension = name + "_d" + std::to_string(axis);
          signature.push_back("int64_t " + dimension); call.push_back(dimension);
        }
        for (int64_t axis = 0; axis < cast<MemRefType>(argument.getType()).getRank(); ++axis) {
          std::string stride = name + "_s" + std::to_string(axis);
          signature.push_back("int64_t " + stride); call.push_back(stride);
        }
      } else {
        auto scalar = cast<dsa::ScalarArgumentAttr>(attribute);
        signature.push_back(ctype(scalar.getType()) + " " + name); call.push_back(name);
        parameters.push_back(llvm::json::Object{{"kind", "scalar"}, {"name", scalar.getName().getValue()}, {"dtype", dtype(scalar.getType())}});
      }
    }
    llvm::json::Array fullExtents;
    for (int64_t value : function->getAttrOfType<DenseI64ArrayAttr>("intent_dsa.full_extent_dimensions").asArrayRef())
      fullExtents.push_back(value);
    metadata = llvm::json::Object{{"provider", "bangc"}, {"architecture", "mtp_372"},
        {"parameters", std::move(parameters)}, {"entry", "intent_launch"},
        {"tile", config.getTile()}, {"tasks", config.getTasks()},
        {"tile_m", config.getTileM()}, {"tile_n", config.getTileN()}, {"tile_k", config.getTileK()},
        {"local_bytes", config.getLocalBytes()},
        {"full_extent_dimensions", std::move(fullExtents)}, {"disjoint_outputs", true}};
    out << tileImplementations << "\n__mlu_global__ void intent_device(" << llvm::join(signature, ", ") << ") {\n";
    auto bytes = [&](StringRef name) { return function->getAttrOfType<IntegerAttr>(name).getInt(); };
    if (bytes("bangc.nram_bytes")) line("__nram__ __attribute__((aligned(128))) unsigned char local_nram[" + std::to_string(bytes("bangc.nram_bytes")) + "];", 1);
    if (bytes("bangc.wram_bytes")) line("__wram__ __attribute__((aligned(128))) unsigned char local_wram[" + std::to_string(bytes("bangc.wram_bytes")) + "];", 1);
    if (failed(block(function.front(), 1))) return failure();
    out << "}\n\nextern \"C\" int intent_launch(void *stream";
    if (!signature.empty()) out << ", " << llvm::join(signature, ", ");
    out << ") {\n  cnrtDim3_t dim = {" << config.getTasks() << ", 1, 1};\n"
        << "  intent_device<<<dim, cnrtFuncTypeBlock, static_cast<cnrtQueue_t>(stream)>>>("
        << llvm::join(call, ", ") << ");\n  return static_cast<int>(cnrtGetLastError());\n}\n";
    return success();
  }
private:
  void line(const std::string &text, unsigned depth) { out.indent(depth * 2) << text << "\n"; }
  std::string name(Value value) { return names.lookup(value); }
  std::string bind(Value value) {
    std::string result = "v" + std::to_string(next++); names[value] = result; return result;
  }
  std::string count(Value value) { return std::to_string(cast<MemRefType>(value.getType()).getNumElements()); }
  std::string shape(Value value) {
    auto type = cast<MemRefType>(value.getType());
    return ctype(type.getElementType()) + ", " + std::to_string(type.getDimSize(0)) + ", " + std::to_string(type.getDimSize(1));
  }
  std::string floatLiteral(FloatAttr attribute) {
    double value = attribute.getValueAsDouble();
    if (std::isnan(value)) return "NAN";
    if (std::isinf(value)) return value < 0 ? "(-INFINITY)" : "INFINITY";
    SmallString<32> text;
    attribute.getValue().toString(text);
    std::string literal = text.str().str();
    if (literal.find_first_of(".eE") == std::string::npos) literal += ".0";
    return attribute.getType().isF64() ? literal : literal + "f";
  }
  LogicalResult block(Block &body, unsigned depth) {
    for (Operation &op : body) if (failed(operation(&op, depth))) return failure();
    return success();
  }
  LogicalResult operation(Operation *op, unsigned depth) {
    if (isa<func::ReturnOp, scf::YieldOp>(op)) return success();
    if (isa<dsa::SynchronizeOp>(op)) { line("__sync();", depth); return success(); }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults()) return op->emitError("BANG C conditional results must be materialized");
      line("if (" + name(branch.getCondition()) + ") {", depth);
      if (failed(block(branch.getThenRegion().front(), depth + 1))) return failure();
      if (!branch.getElseRegion().empty()) {
        line("} else {", depth);
        if (failed(block(branch.getElseRegion().front(), depth + 1))) return failure();
      }
      line("}", depth); return success();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      if (loop.getNumResults() || loop.getNumOperands()) return op->emitError("BANG C while state must be materialized");
      line("while (true) {", depth);
      if (failed(block(loop.getBefore().front(), depth + 1)) || failed(block(loop.getAfter().front(), depth + 1))) return failure();
      line("}", depth); return success();
    }
    if (auto condition = dyn_cast<scf::ConditionOp>(op)) {
      if (!condition.getArgs().empty()) return op->emitError("BANG C condition state must be materialized");
      line("if (!(" + name(condition.getCondition()) + ")) break;", depth); return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (loop.getNumResults()) return op->emitError("BANG C serialization requires materialized loop state");
      std::string iv = bind(loop.getInductionVar());
      line("for (int64_t " + iv + " = " + name(loop.getLowerBound()) + "; " + iv + " < " + name(loop.getUpperBound()) +
           "; " + iv + " += " + name(loop.getStep()) + ") {", depth);
      if (failed(block(*loop.getBody(), depth + 1))) return failure();
      line("}", depth); return success();
    }
    if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
      auto type = allocation.getType();
      std::string buffer = type.getMemorySpaceAsInt() == dsa::matrixSpace ? "local_wram" : "local_nram";
      line(ctype(type.getElementType()) + " *" + bind(allocation.getResult()) + " = reinterpret_cast<" +
          ctype(type.getElementType()) + " *>(" + buffer + " + " +
          std::to_string(op->getAttrOfType<IntegerAttr>("bangc.offset").getInt()) + ");", depth);
      return success();
    }
    if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
      line("intent_load_tile<" + shape(load.getOutput()) + ">(" + name(load.getOutput()) + ", " + name(load.getSource()) +
          ", " + name(load.getOffset()) + ", " + name(load.getRowStride()) + ", " + name(load.getColumnStride()) +
          ", " + name(load.getRows()) + ", " + name(load.getColumns()) + ");", depth); return success();
    }
    if (auto store = dyn_cast<dsa::StoreTileOp>(op)) {
      line("intent_store_tile<" + shape(store.getInput()) + ">(" + name(store.getDestination()) + ", " + name(store.getInput()) +
          ", " + name(store.getOffset()) + ", " + name(store.getRowStride()) + ", " + name(store.getColumnStride()) +
          ", " + name(store.getRows()) + ", " + name(store.getColumns()) + ");", depth); return success();
    }
    if (auto fill = dyn_cast<dsa::FillOp>(op)) {
      if (isa<FloatType>(fill.getValue().getType()))
        line("__bang_write_value(" + name(fill.getOutput()) + ", " + count(fill.getOutput()) + ", " + name(fill.getValue()) + ");", depth);
      else line("intent_fill_local<" + ctype(fill.getValue().getType()) + ", " + count(fill.getOutput()) + ">(" + name(fill.getOutput()) + ", " + name(fill.getValue()) + ");", depth);
      return success();
    }
    if (auto select = dyn_cast<dsa::SelectOp>(op)) {
      line("intent_select_local<" + ctype(cast<MemRefType>(select.getOutput().getType()).getElementType()) + ", " +
          count(select.getOutput()) + ">(" + name(select.getOutput()) + ", " + name(select.getCondition()) + ", " +
          name(select.getTrueValue()) + ", " + name(select.getFalseValue()) + ");", depth); return success();
    }
    if (auto store = dyn_cast<dsa::StoreScalarOp>(op)) {
      line(name(store.getDestination()) + "[" + name(store.getOffset()) + "] = " + name(store.getValue()) + ";", depth); return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto type = cast<MemRefType>(store.getMemref().getType());
      line("intent_write_local(" + name(store.getMemref()) + " + " + name(store.getIndices()[0]) + " * " + std::to_string(type.getDimSize(1)) +
          " + " + name(store.getIndices()[1]) + ", " + name(store.getValue()) + ");", depth); return success();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      line("intent_copy_local<" + ctype(cast<MemRefType>(copy.getSource().getType()).getElementType()) + ", " +
          count(copy.getSource()) + ">(" + name(copy.getTarget()) + ", " + name(copy.getSource()) + ");", depth); return success();
    }
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      if (binary.getKind() == BinaryOperator::TrueDivide) {
        line("intent_divide_local<" + ctype(cast<MemRefType>(binary.getOutput().getType()).getElementType()) + ", " +
            count(binary.getOutput()) + ">(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");", depth);
        return success();
      }
      std::string intrinsic;
      switch (binary.getKind()) {
      case BinaryOperator::Add: intrinsic = "__bang_add"; break;
      case BinaryOperator::Subtract: intrinsic = "__bang_sub"; break;
      case BinaryOperator::Multiply: intrinsic = "__bang_mul"; break;
      case BinaryOperator::Maximum: intrinsic = "__bang_nan_maximum"; break;
      case BinaryOperator::Minimum: intrinsic = "__bang_nan_minimum"; break;
      case BinaryOperator::MaximumNum: intrinsic = "__bang_maximum"; break;
      case BinaryOperator::MinimumNum: intrinsic = "__bang_minimum"; break;
      default: return op->emitError("unbound BANG binary operation");
      }
      line(intrinsic + "(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " +
          name(binary.getRhs()) + ", " + count(binary.getOutput()) + ");", depth); return success();
    }
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      std::string intrinsic;
      switch (unary.getKind()) {
      case UnaryOperator::Exp: intrinsic = "__bang_active_exp"; break;
      case UnaryOperator::Log: intrinsic = "__bang_active_log"; break;
      case UnaryOperator::Sqrt: intrinsic = "__bang_sqrt"; break;
      case UnaryOperator::Rsqrt: intrinsic = "__bang_rsqrt"; break;
      case UnaryOperator::Tanh: intrinsic = "__bang_active_tanh"; break;
      case UnaryOperator::Abs: intrinsic = "__bang_abs"; break;
      case UnaryOperator::Negate:
        line("__bang_mul_scalar(" + name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " +
            ctype(cast<MemRefType>(unary.getInput().getType()).getElementType()) + "(-1), " + count(unary.getOutput()) + ");", depth); return success();
      default: return op->emitError("unbound BANG unary operation");
      }
      line(intrinsic + "(" + name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + count(unary.getOutput()) + ");", depth); return success();
    }
    if (auto castOp = dyn_cast<dsa::CastOp>(op)) {
      auto from = cast<MemRefType>(castOp.getInput().getType()).getElementType();
      auto to = cast<MemRefType>(castOp.getOutput().getType()).getElementType();
      if (from == to) line("__memcpy(" + name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ", " +
          count(castOp.getOutput()) + " * sizeof(" + ctype(to) + "), NRAM2NRAM);", depth);
      else line(std::string(to.isF32() ? "__bang_half2float(" : "__bang_float2half_rn(") + name(castOp.getOutput()) + ", " +
          name(castOp.getInput()) + ", " + count(castOp.getOutput()) + ");", depth);
      return success();
    }
    if (auto reduce = dyn_cast<dsa::ReduceOp>(op)) {
      line("intent_reduce<" + count(reduce.getInput()) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
          name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
          name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");", depth); return success();
    }
    if (auto prepare = dyn_cast<dsa::PrepareMatrixOp>(op)) {
      line("intent_prepare_matrix<" + shape(prepare.getInput()) + ">(" + name(prepare.getOutput()) + ", " +
          name(prepare.getInput()) + ", " + name(prepare.getScratch()) + ");", depth); return success();
    }
    if (auto matrix = dyn_cast<dsa::MatMulOp>(op)) {
      auto a = cast<MemRefType>(matrix.getLhs().getType()), c = cast<MemRefType>(matrix.getAccumulator().getType());
      line("intent_matmul<" + ctype(a.getElementType()) + ", " + std::to_string(a.getDimSize(0)) + ", " +
          std::to_string(a.getDimSize(1)) + ", " + std::to_string(c.getDimSize(1)) + ">(" + name(matrix.getAccumulator()) + ", " +
          name(matrix.getLhs()) + ", " + name(matrix.getRhs()) + ", " + name(matrix.getScratch()) + ");", depth); return success();
    }
    std::string expression;
    if (isa<dsa::TaskIdOp>(op)) expression = "taskId";
    else if (isa<dsa::TaskCountOp>(op)) expression = "taskDim";
    else if (auto stride = dyn_cast<dsa::StrideOp>(op)) expression = name(stride.getSource()) + "_s" + std::to_string(stride.getAxis());
    else if (auto dim = dyn_cast<memref::DimOp>(op)) {
      auto axis = dim.getConstantIndex();
      if (!axis) return op->emitError("BANG C requires a bound view dimension axis");
      expression = name(dim.getSource()) + "_d" + std::to_string(*axis);
    } else if (auto scalar = dyn_cast<dsa::LoadScalarOp>(op)) expression = name(scalar.getSource()) + "[" + name(scalar.getOffset()) + "]";
    else if (auto scalar = dyn_cast<memref::LoadOp>(op)) {
      auto type = cast<MemRefType>(scalar.getMemref().getType());
      expression = "intent_read_local(" + name(scalar.getMemref()) + " + " + name(scalar.getIndices()[0]) + " * " + std::to_string(type.getDimSize(1)) + " + " + name(scalar.getIndices()[1]) + ")";
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) expression = floatLiteral(floating);
      else expression = std::to_string(cast<IntegerAttr>(constant.getValue()).getInt());
    } else if (isa<arith::ExtFOp, arith::TruncFOp, arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
                   arith::TruncIOp, arith::SIToFPOp, arith::FPToSIOp>(op))
      expression = "static_cast<" + ctype(op->getResult(0).getType()) + ">(" + name(op->getOperand(0)) + ")";
    else if (auto select = dyn_cast<arith::SelectOp>(op)) expression = name(select.getCondition()) + " ? " + name(select.getTrueValue()) + " : " + name(select.getFalseValue());
    else if (isa<arith::NegFOp>(op)) expression = "-" + name(op->getOperand(0));
    else if (op->getName().getDialectNamespace() == "math" && op->getNumOperands() == 1) {
      std::string callee;
      if (isa<math::ExpOp>(op)) callee = "expf";
      if (isa<math::Exp2Op>(op)) callee = "exp2f";
      if (isa<math::LogOp>(op)) callee = "logf";
      if (isa<math::SqrtOp>(op)) callee = "sqrtf";
      if (isa<math::RsqrtOp>(op)) callee = "1.0f / sqrtf";
      if (isa<math::TanhOp>(op)) callee = "tanhf";
      if (isa<math::AbsFOp>(op)) callee = "fabsf";
      if (op->getResult(0).getType().isF64() && !callee.empty()) callee.pop_back();
      if (!callee.empty()) expression = callee + "(" + name(op->getOperand(0)) + ")";
    }
    else if (op->getNumOperands() == 2 && op->getNumResults() == 1) {
      std::string lhs = name(op->getOperand(0)), rhs = name(op->getOperand(1)), symbol;
      if (isa<arith::AddIOp, arith::AddFOp>(op)) symbol = "+";
      if (isa<arith::SubIOp, arith::SubFOp>(op)) symbol = "-";
      if (isa<arith::MulIOp, arith::MulFOp>(op)) symbol = "*";
      if (isa<arith::DivSIOp, arith::DivFOp>(op)) symbol = "/";
      if (isa<arith::RemSIOp>(op)) symbol = "%";
      if (isa<arith::AndIOp>(op)) symbol = "&";
      if (isa<arith::OrIOp>(op)) symbol = "|";
      if (isa<arith::XOrIOp>(op)) symbol = "^";
      if (isa<arith::ShLIOp>(op)) symbol = "<<";
      if (isa<arith::ShRSIOp>(op)) symbol = ">>";
      if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq: symbol = "=="; break;
        case arith::CmpIPredicate::ne: symbol = "!="; break;
        case arith::CmpIPredicate::slt: symbol = "<"; break;
        case arith::CmpIPredicate::sle: symbol = "<="; break;
        case arith::CmpIPredicate::sgt: symbol = ">"; break;
        case arith::CmpIPredicate::sge: symbol = ">="; break;
        default: return op->emitError("BANG C integer comparison predicate is not bound");
        }
      }
      if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
        switch (cmp.getPredicate()) {
        case arith::CmpFPredicate::OEQ: symbol = "=="; break;
        case arith::CmpFPredicate::UNE: symbol = "!="; break;
        case arith::CmpFPredicate::OLT: symbol = "<"; break;
        case arith::CmpFPredicate::OLE: symbol = "<="; break;
        case arith::CmpFPredicate::OGT: symbol = ">"; break;
        case arith::CmpFPredicate::OGE: symbol = ">="; break;
        default: return op->emitError("BANG C floating comparison predicate is not bound");
        }
      }
      if (!symbol.empty()) expression = lhs + " " + symbol + " " + rhs;
      else if (isa<arith::MinSIOp>(op)) expression = "(" + lhs + " < " + rhs + " ? " + lhs + " : " + rhs + ")";
      else if (isa<arith::MaxSIOp>(op)) expression = "(" + lhs + " > " + rhs + " ? " + lhs + " : " + rhs + ")";
      else if (isa<arith::FloorDivSIOp>(op)) expression = "(" + lhs + " / " + rhs + " - (" + lhs + " % " + rhs + " != 0 && ((" + lhs + " < 0) != (" + rhs + " < 0))))";
      else if (isa<arith::CeilDivSIOp>(op)) expression = "(" + lhs + " / " + rhs + " + (" + lhs + " % " + rhs + " != 0 && ((" + lhs + " < 0) == (" + rhs + " < 0))))";
      else if (isa<arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp>(op)) {
        std::string callee = isa<arith::MaximumFOp, arith::MaxNumFOp>(op) ? "fmax" : "fmin";
        if (!op->getResult(0).getType().isF64()) callee += "f";
        expression = callee + "(" + lhs + ", " + rhs + ")";
        if (isa<arith::MaximumFOp, arith::MinimumFOp>(op)) expression = "(isnan(" + lhs + ") || isnan(" + rhs + ") ? NAN : " + expression + ")";
      }
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp>(op)) {
        Type type = op->getResult(0).getType();
        std::string unsignedType = type.isInteger(32) ? "uint32_t" : "uint64_t";
        expression = "static_cast<" + ctype(type) + ">(static_cast<" + unsignedType + ">(" + lhs + ") " + symbol + " static_cast<" + unsignedType + ">(" + rhs + "))";
      }
    }
    if (expression.empty() || op->getNumResults() != 1) return op->emitError("operation has no BANG C spelling");
    line(ctype(op->getResult(0).getType()) + " " + bind(op->getResult(0)) + " = " + expression + ";", depth);
    return success();
  }
  func::FuncOp function;
  llvm::raw_ostream &out;
  DenseMap<Value, std::string> names;
  unsigned next = 0;
  SmallVector<std::string> signature, call;
};
}
LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  if (failed(dsa::verifyProgram(module, true))) return failure();
  llvm::raw_string_ostream output(source);
  llvm::json::Object interface;
  Serializer serializer(*module.getOps<func::FuncOp>().begin(), output);
  if (failed(serializer.emit(interface))) return failure();
  llvm::raw_string_ostream metadataOutput(metadata);
  metadataOutput << llvm::json::Value(std::move(interface));
  return success();
}
}

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Target/TileLang/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/GPU/Serialization/Python.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "Intent/Target/TileLang/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <set>

using namespace mlir;

namespace intent::tilelang {
namespace {

std::string tileLangType(Type type) {
  static const gpu::PythonScalarSyntax syntax{
      "T.", "bool", "", "float8_e4m3fn", "float8_e5m2"};
  return gpu::pythonScalarType(type, syntax);
}

std::string expressionString(gpu::PhysicalExprAttr expression) {
  static const gpu::PythonExpressionSyntax syntax{
      "T.ceildiv", "T.min", "T.max", "T.if_then_else", "T.next_power_of_2", false};
  return gpu::pythonExpression(expression, syntax, [](gpu::PhysicalExprAttr leaf) {
    return leaf.getSymbol().getValue().str();
  });
}

std::string shape(ArrayAttr extents) {
  std::string result = "(";
  for (auto [index, extent] : llvm::enumerate(extents)) {
    if (index)
      result += ", ";
    result += expressionString(cast<gpu::PhysicalExprAttr>(extent));
  }
  if (extents.size() == 1)
    result += ",";
  return result + ")";
}

std::string literal(Attribute value) {
  return gpu::pythonLiteral(value, [](Type type) {
    std::string name = tileLangType(type);
    return name.empty() ? "float(\"inf\")" : "T.infinity(" + name + ")";
  });
}

class Serializer {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : kernel(kernel), output(output) {}

  LogicalResult emit(std::string &metadata) {
    bindArguments();
    if (failed) return failure();
    collectConfiguration();
    emitPreamble();
    emitBuilder();
    return failed ? failure() : emitMetadata(metadata);
  }

private:
  using ViewABI = gpu::ViewArgument;
  using ScalarABI = gpu::ScalarArgument;
  using MetadataABI = gpu::MetadataArgument;
  void bindArguments() {
    auto interface = gpu::readInterface(kernel);
    if (mlir::failed(interface)) {
      failed = true;
      return;
    }
    views = std::move(interface->views);
    scalars = std::move(interface->scalars);
    metadataArguments = std::move(interface->metadata);
    for (const ScalarABI &scalar : scalars)
      values[kernel.getArgument(scalar.abi)] = scalar.name;
    for (const MetadataABI &metadata : metadataArguments)
      values[kernel.getArgument(metadata.abi)] = metadata.name;
    for (const ViewABI &view : views)
      values[kernel.getArgument(view.abi)] = view.name;
    llvm::StringSet<> occupied;
    for (const auto &entry : values)
      occupied.insert(entry.second);
    kernel.walk([&](gpu::ParameterOp parameter) {
      occupied.insert(parameter.getParameter().getName().getValue());
    });
    auto fresh = [&](StringRef stem) {
      std::string name = stem.str();
      while (!occupied.insert(name).second)
        name += "_";
      return name;
    };
    kernel.walk([&](gpu::ViewOverlapOp overlap) {
      values[overlap.getResult()] = fresh("_intent_overlap");
      overlapFacts.push_back(overlap);
    });
  }

  void collectConfiguration() {
    kernel.walk([&](gpu::ParameterOp parameter) {
      std::string name = parameter.getParameter().getName().getValue().str();
      values[parameter.getResult()] = name;
      if (parameter->hasAttr(gpu::coverageDimensionAttr))
        coverageNames.insert(name);
      else
        parameterNames.push_back(name);
    });
  }

  void emitPreamble() {
    output << "import tilelang\nimport tilelang.language as T\n\n";
    std::map<std::string, std::pair<std::string, bool>> primitives;
    auto add = [&](StringRef mnemonic, bool flush, bool binary) {
      std::string suffix = flush ? "_ftz" : "";
      primitives["_intent_approx_" + mnemonic.str() + suffix] = {
          mnemonic.str() + ".approx" + (flush ? ".ftz" : "") + ".f32", binary};
    };
    kernel.walk([&](gpu::UnaryOp unary) {
      if (unary.getApproximate())
        add(unary.getOperatorKind() == UnaryOperator::Exp2 ? "ex2" : "tanh",
            unary.getFlushToZero(), false);
    });
    kernel.walk([&](gpu::BinaryOp binary) {
      if (binary.getApproximate())
        add("div", binary.getFlushToZero(), true);
    });
    hasApproximateMath = !primitives.empty();
    if (hasApproximateMath) {
      output << "_INTENT_MATH_SOURCE = r\"\"\"\n";
      for (const auto &[name, primitive] : primitives) {
        const auto &[instruction, binary] = primitive;
        output << "static __device__ __forceinline__ float " << name
               << "(float x" << (binary ? ", float y" : "") << ") {\n"
               << "  float result;\n  asm(\"" << instruction
               << " %0, %1" << (binary ? ", %2" : "")
               << ";\" : \"=f\"(result) : \"f\"(x)"
               << (binary ? ", \"f\"(y)" : "")
               << ");\n  return result;\n}\n";
      }
      output << "\"\"\"\n\n";
    }
  }

  void emitBuilder() {
    auto lowerPredicatedLoadStore =
        kernel->getAttrOfType<BoolAttr>(lowerPredicatedLoadStoreAttr);
    output << "@tilelang.jit(pass_configs={"
              "tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: "
           << (lowerPredicatedLoadStore.getValue() ? "True" : "False")
           << "})\ndef _intent_kernel(";
    bool first = true;
    auto argument = [&](StringRef text) {
      if (!first)
        output << ", ";
      first = false;
      output << text;
    };
    for (const MetadataABI &metadata : metadataArguments)
      argument(metadata.name);
    for (gpu::ViewOverlapOp overlap : overlapFacts)
      argument(valueString(overlap.getResult()));
    for (const std::string &name : parameterNames)
      argument(name);
    for (const std::string &name : coverageNames)
      argument(name);
    output << "):\n";
    output << "    @T.prim_func\n    def main(";
    first = true;
    for (const ViewABI &view : views) {
      if (!first)
        output << ", ";
      first = false;
      output << view.name << ": T.Tensor(" << viewShape(view) << ", "
             << tileLangType(view.type.getElementType()) << ")";
    }
    for (const ScalarABI &scalar : scalars) {
      if (!first)
        output << ", ";
      first = false;
      output << scalar.name << ": " << tileLangType(scalar.type);
    }
    output << "):\n";
    auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
    if (!space || space.size() != 1) {
      failed = true;
      return;
    }
    auto launches = kernel.getBody().front().getOps<LaunchConfigOp>();
    if (!llvm::hasSingleElement(launches)) {
      kernel.emitError(
          "terminal TileLang program has no unique top-level launch configuration");
      failed = true;
      return;
    }
    auto launch = *launches.begin();
    line("with T.Kernel(" +
             expressionString(cast<gpu::PhysicalExprAttr>(space[0])) +
             ", threads=" + valueString(launch.getThreads()) + ") as pid0:",
         2);
    indent = 3;
    if (hasApproximateMath)
      line("T.import_source(_INTENT_MATH_SOURCE)");
    emitBlock(kernel.getBody().front());
    line("return main", 1);
    output << "\n";
  }

  void emitBlock(Block &block) {
    uint64_t begin = output.tell();
    for (Operation &operation : block) {
      if (isa<scf::YieldOp, YieldOp, func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
    if (output.tell() == begin)
      line("pass");
  }

  void emitOperation(Operation &operation) {
    if (isa<gpu::ViewOverlapOp>(operation)) {
      return;
    } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      values[constant.getResult()] = literal(constant.getValue());
    } else if (auto parameter = dyn_cast<gpu::ParameterOp>(operation)) {
      values[parameter.getResult()] =
          parameter.getParameter().getName().getValue().str();
    } else if (auto physical = dyn_cast<gpu::PhysicalExprOp>(operation)) {
      assign(physical.getResult(), expressionString(physical.getExpression()));
    } else if (auto program = dyn_cast<gpu::ProgramIdOp>(operation)) {
      if (program.getAxis() != 0) {
        program.emitOpError("TileLang T.Kernel binding currently has one program axis");
        failed = true;
      } else {
        values[program.getResult()] = "pid0";
      }
    } else if (auto coordinate = dyn_cast<gpu::WorksetCoordinateOp>(operation)) {
      values[coordinate.getResult()] = valueString(coordinate.getCoordinate());
    } else if (auto dim = dyn_cast<gpu::DimOp>(operation)) {
      auto extent = cast<gpu::PhysicalExprAttr>(
          dim.getView().getType().getLayout().getExtents()[dim.getAxis()]);
      assign(dim.getResult(), expressionString(extent));
    } else if (auto range = dyn_cast<gpu::RangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) + ", " +
                                    valueString(range.getStop()) + ", " +
                                    valueString(range.getStep()) + ")");
    } else if (auto bound = dyn_cast<gpu::RangeBoundOp>(operation)) {
      assign(bound.getResult(), valueString(bound.getRange()) + "[" +
                                    std::to_string(bound.getBound()) + "]");
    } else if (auto mapping = dyn_cast<gpu::DelinearizeOp>(operation)) {
      std::string remaining = valueString(mapping.getLinear());
      SmallVector<std::string> coordinates(mapping.getNumResults());
      for (int64_t axis = mapping.getNumResults() - 1; axis >= 0; --axis) {
        std::string extent = valueString(mapping.getExtents()[axis]);
        coordinates[axis] = "(" + remaining + " % " + extent + ")";
        remaining = "(" + remaining + " // " + extent + ")";
      }
      for (auto [coordinate, text] : llvm::zip(mapping.getCoordinates(), coordinates))
        assign(coordinate, text);
    } else if (auto binary = dyn_cast<gpu::BinaryOp>(operation)) {
      assign(binary.getResult(), binaryExpression(binary));
    } else if (auto unary = dyn_cast<gpu::UnaryOp>(operation)) {
      assign(unary.getResult(), unaryExpression(unary));
    } else if (auto compare = dyn_cast<gpu::CompareOp>(operation)) {
      auto predicate = [&]() -> StringRef {
        switch (compare.getPredicate()) {
        case ComparePredicate::Eq: return "==";
        case ComparePredicate::Ne: return "!=";
        case ComparePredicate::Lt: return "<";
        case ComparePredicate::Le: return "<=";
        case ComparePredicate::Gt: return ">";
        case ComparePredicate::Ge: return ">=";
        }
        llvm_unreachable("unhandled Intent compare predicate");
      }();
      assign(compare.getResult(), "(" + valueString(compare.getLhs()) + " " +
                                      predicate.str() + " " +
                                      valueString(compare.getRhs()) + ")");
    } else if (auto select = dyn_cast<gpu::SelectOp>(operation)) {
      assign(select.getResult(), "T.if_then_else(" +
                                     valueString(select.getCondition()) + ", " +
                                     valueString(select.getTrueValue()) + ", " +
                                     valueString(select.getFalseValue()) + ")");
    } else if (auto cast = dyn_cast<gpu::CastOp>(operation)) {
      assign(cast.getResult(), "T.cast(" + valueString(cast.getValue()) + ", " +
                                   tileLangType(cast.getResult().getType()) + ")");
    } else if (auto bitcast = dyn_cast<gpu::BitcastOp>(operation)) {
      assign(bitcast.getResult(), "T.reinterpret(" +
                                      valueString(bitcast.getValue()) + ", " +
                                      tileLangType(bitcast.getResult().getType()) +
                                      ")");
    } else if (auto record = dyn_cast<gpu::MakeRecordOp>(operation)) {
      values[record.getResult()] = tuple(record.getFields());
    } else if (auto extract = dyn_cast<gpu::ExtractOp>(operation)) {
      assign(extract.getResult(), valueString(extract.getRecord()) + "[" +
                                      std::to_string(extract.getField()) + "]");
    } else if (isa<LaunchConfigOp>(operation)) {
      return;
    } else if (auto allocation = dyn_cast<AllocOp>(operation)) {
      auto type = allocation.getResult().getType();
      std::string function =
          type.getSpace().getValue() == BufferSpace::Shared
              ? "T.alloc_shared"
              : "T.alloc_fragment";
      assign(allocation.getResult(), function + "(" + shape(type.getShape()) +
                                         ", " + tileLangType(type.getElementType()) + ")");
    } else if (auto clear = dyn_cast<ClearOp>(operation)) {
      line("T.clear(" + valueString(clear.getBuffer()) + ")");
    } else if (auto fill = dyn_cast<FillOp>(operation)) {
      line("T.fill(" + valueString(fill.getBuffer()) + ", " +
           valueString(fill.getValue()) + ")");
    } else if (isa<SyncOp>(operation)) {
      line("T.sync_threads()");
    } else if (auto copy = dyn_cast<CopyInOp>(operation)) {
      line("T.copy(" + valueString(copy.getSource()) +
           regionIndex(copy.getOffsets(), copy.getSourceAxes(),
                       copy.getDestination().getType().getShape()) +
           ", " + valueString(copy.getDestination()) +
           (copy.getBoundaryAxes().empty() ? ")" : ", disable_tma=True)"));
    } else if (auto copy = dyn_cast<CopyOutOp>(operation)) {
      line("T.copy(" + valueString(copy.getSource()) + ", " +
           valueString(copy.getDestination()) +
           regionIndex(copy.getOffsets(), copy.getDestinationAxes(),
                       copy.getSource().getType().getShape()) +
           (copy.getBoundaryAxes().empty() ? ")" : ", disable_tma=True)"));
    } else if (auto copy = dyn_cast<CastCopyOutOp>(operation)) {
      line("T.copy(" + valueString(copy.getSource()) + ", " +
           valueString(copy.getDestination()) +
           regionIndex(copy.getOffsets(), copy.getDestinationAxes(),
                       copy.getSource().getType().getShape()) +
           (copy.getBoundaryAxes().empty() ? ")" : ", disable_tma=True)"));
    } else if (auto parallel = dyn_cast<ParallelOp>(operation)) {
      Block &body = parallel.getBody().front();
      std::string variables;
      std::string extents;
      for (auto [axis, argument] : llvm::enumerate(body.getArguments())) {
        if (axis) {
          variables += ", ";
          extents += ", ";
        }
        std::string name = "i" + std::to_string(counter++);
        values[argument] = name;
        variables += name;
        extents += expressionString(
            mlir::cast<gpu::PhysicalExprAttr>(parallel.getShape()[axis]));
      }
      line("for " + variables + " in T.Parallel(" + extents + "):");
      ++indent;
      emitBlock(body);
      --indent;
    } else if (auto load = dyn_cast<BufferLoadOp>(operation)) {
      assign(load.getResult(), valueString(load.getBuffer()) +
                                   index(load.getIndices()));
    } else if (auto store = dyn_cast<BufferStoreOp>(operation)) {
      line(valueString(store.getBuffer()) + index(store.getIndices()) + " = " +
           valueString(store.getValue()));
    } else if (auto load = dyn_cast<ViewLoadOp>(operation)) {
      std::string expression =
          valueString(load.getView()) + index(load.getIndices());
      if (load.getValid())
        expression = "T.if_then_else(" + valueString(load.getValid()) + ", " +
                     expression + ", " + valueString(load.getFill()) + ")";
      assign(load.getResult(), expression);
    } else if (auto store = dyn_cast<ViewStoreOp>(operation)) {
      if (store.getValid()) {
        line("if " + valueString(store.getValid()) + ":");
        ++indent;
      }
      line(valueString(store.getView()) + index(store.getIndices()) + " = " +
           valueString(store.getValue()));
      if (store.getValid())
        --indent;
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      auto nativeReduction = [](BinaryOperator kind) -> StringRef {
        switch (kind) {
        case BinaryOperator::Add: return "T.reduce_sum";
        case BinaryOperator::MaximumNum: return "T.reduce_max";
        case BinaryOperator::MinimumNum: return "T.reduce_min";
        case BinaryOperator::LogicalAnd:
        case BinaryOperator::BitwiseAnd: return "T.reduce_bitand";
        case BinaryOperator::LogicalOr:
        case BinaryOperator::BitwiseOr: return "T.reduce_bitor";
        case BinaryOperator::BitwiseXor: return "T.reduce_bitxor";
        default: llvm_unreachable("unverified TileLang native reduction kind");
        }
      };
      line(nativeReduction(reduce.getKind()).str() + "(" +
           valueString(reduce.getSource()) + ", " +
           valueString(reduce.getDestination()) + ", dim=" +
           std::to_string(reduce.getAxis()) + ", clear=False)");
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      std::string function = scan.getKind() == BinaryOperator::Add
                                 ? "T.cumsum"
                                 : "T.cummax";
      line(function + "(" + valueString(scan.getSource()) + ", " +
           valueString(scan.getDestination()) + ", dim=" +
           std::to_string(scan.getAxis()) + ", reverse=" +
           (scan.getReverse() ? "True" : "False") + ")");
    } else if (auto gemm = dyn_cast<GemmOp>(operation)) {
      line("T.gemm(" + valueString(gemm.getLhs()) + ", " +
           valueString(gemm.getRhs()) + ", " +
           valueString(gemm.getAccumulator()) + ", transpose_A=" +
           (gemm.getTransposeLhs() ? "True" : "False") + ", transpose_B=" +
           (gemm.getTransposeRhs() ? "True" : "False") + ")");
    } else if (auto gemm = dyn_cast<SparseGemmOp>(operation)) {
      line("T.gemm_sp(" + valueString(gemm.getCompressed()) + ", " +
           valueString(gemm.getMetadata()) + ", " +
           valueString(gemm.getRhs()) + ", " +
           valueString(gemm.getAccumulator()) + ", transpose_A=" +
           (gemm.getTransposeCompressed() ? "True" : "False") +
           ", transpose_E=" +
           (gemm.getTransposeMetadata() ? "True" : "False") +
           ", transpose_B=" +
           (gemm.getTransposeRhs() ? "True" : "False") + ")");
    } else if (auto pipeline = dyn_cast<PipelineOp>(operation)) {
      auto loop = mlir::cast<scf::ForOp>(pipeline.getBody().front().front());
      std::string induction = "iv" + std::to_string(counter++);
      std::string tile = induction + "_tile";
      values[loop.getInductionVar()] = induction;
      line("for " + tile + " in T.Pipelined(T.ceildiv((" +
           valueString(loop.getUpperBound()) + " - " +
           valueString(loop.getLowerBound()) + "), " +
           valueString(loop.getStep()) + "), num_stages=" +
           valueString(pipeline.getStages()) + "):");
      ++indent;
      line(induction + " = " + valueString(loop.getLowerBound()) + " + " +
           tile + " * " + valueString(loop.getStep()));
      emitBlock(*loop.getBody());
      --indent;
    } else if (auto choice = dyn_cast<scf::IfOp>(operation)) {
      if (choice.getNumResults() != 0) {
        choice.emitOpError("TileLang-local branch still returns an unbufferized SSA value");
        failed = true;
        return;
      }
      line("if " + valueString(choice.getCondition()) + ":");
      ++indent;
      emitBlock(choice.getThenRegion().front());
      --indent;
      if (!choice.getElseRegion().empty()) {
        line("else:");
        ++indent;
        emitBlock(choice.getElseRegion().front());
        --indent;
      }
    } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      if (loop.getNumResults() != 0) {
        loop.emitOpError("TileLang-local loop still carries an unbufferized SSA value");
        failed = true;
        return;
      }
      std::string induction = "iv" + std::to_string(counter++);
      values[loop.getInductionVar()] = induction;
      line("for " + induction + " in T.serial(" +
           valueString(loop.getLowerBound()) + ", " +
           valueString(loop.getUpperBound()) + ", " +
           valueString(loop.getStep()) + "):");
      ++indent;
      emitBlock(*loop.getBody());
      --indent;
    } else {
      operation.emitOpError("has no terminal TileLang spelling");
      failed = true;
    }
  }

  std::string binaryExpression(gpu::BinaryOp binary) {
    auto infix = [&](StringRef spelling) {
      return "(" + valueString(binary.getLhs()) + " " + spelling.str() + " " +
             valueString(binary.getRhs()) + ")";
    };
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: return infix("+");
    case BinaryOperator::Subtract: return infix("-");
    case BinaryOperator::Multiply: return infix("*");
    case BinaryOperator::TrueDivide:
      if (binary.getApproximate())
        return "T.call_pure_extern(\"float32\", \"_intent_approx_div" +
               std::string(binary.getFlushToZero() ? "_ftz" : "") + "\", " +
               valueString(binary.getLhs()) + ", " + valueString(binary.getRhs()) + ")";
      return infix("/");
    case BinaryOperator::FloorDivide: return infix("//");
    case BinaryOperator::Remainder: return infix("%");
    case BinaryOperator::Power: return infix("**");
    case BinaryOperator::MaximumNum:
      return "T.max(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    case BinaryOperator::MinimumNum:
      return "T.min(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    case BinaryOperator::Maximum:
    case BinaryOperator::Minimum:
      if (isa<IntegerType, IndexType>(binary.getResult().getType()))
        return std::string(binary.getOperatorKind() == BinaryOperator::Maximum
                               ? "T.max(" : "T.min(") +
               valueString(binary.getLhs()) + ", " +
               valueString(binary.getRhs()) + ")";
      binary.emitOpError(
          "TileLang has no spelling that preserves Intent NaN-propagating min/max");
      failed = true;
      return "<unsupported-nan-propagating-minmax>";
    case BinaryOperator::LogicalAnd:
    case BinaryOperator::BitwiseAnd: return infix("&");
    case BinaryOperator::LogicalOr:
    case BinaryOperator::BitwiseOr: return infix("|");
    case BinaryOperator::BitwiseXor: return infix("^");
    case BinaryOperator::LeftShift: return infix("<<");
    case BinaryOperator::RightShift: return infix(">>");
    }
    llvm_unreachable("unhandled Intent binary operator");
  }

  std::string unaryExpression(gpu::UnaryOp unary) {
    std::string input = valueString(unary.getInput());
    auto call = [&](StringRef function) {
      return function.str() + "(" + input + ")";
    };
    auto libraryCall = [&](StringRef function, bool external) {
      Type element = unary.getResult().getType();
      std::string argument = element.isF32()
                                 ? input : "T.cast(" + input + ", T.float32)";
      std::string result = external
          ? "T.call_pure_extern(\"float32\", \"" + function.str() + "\", " + argument + ")"
          : function.str() + "(" + argument + ")";
      return element.isF32()
                 ? result : "T.cast(" + result + ", " + tileLangType(element) + ")";
    };
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate: return "(-" + input + ")";
    case UnaryOperator::Not: return "(~" + input + ")";
    case UnaryOperator::Exp: return call("T.exp");
    case UnaryOperator::Exp2:
      if (unary.getApproximate())
        return "T.call_pure_extern(\"float32\", \"_intent_approx_ex2" +
               std::string(unary.getFlushToZero() ? "_ftz" : "") + "\", " + input + ")";
      return call("T.exp2");
    case UnaryOperator::Log: return call("T.log");
    case UnaryOperator::Log1p: return libraryCall("T.log1p", /*external=*/false);
    case UnaryOperator::Sin: return call("T.sin");
    case UnaryOperator::Asin: return libraryCall("T.asin", /*external=*/false);
    case UnaryOperator::Cos: return call("T.cos");
    case UnaryOperator::Floor: return call("T.floor");
    case UnaryOperator::Erf: return call("T.erf");
    case UnaryOperator::Erfc: return libraryCall("erfcf", /*external=*/true);
    case UnaryOperator::I0: return libraryCall("cyl_bessel_i0f", /*external=*/true);
    case UnaryOperator::Rsqrt: return call("T.rsqrt");
    case UnaryOperator::Sigmoid: return call("T.sigmoid");
    case UnaryOperator::Tanh:
      return unary.getApproximate()
                 ? "T.call_pure_extern(\"float32\", \"_intent_approx_tanh\", " + input + ")"
                 : call("T.tanh");
    case UnaryOperator::Abs: return call("T.abs");
    case UnaryOperator::Sqrt: return call("T.sqrt");
    case UnaryOperator::Lgamma:
      unary.emitOpError("lgamma is unsupported by the TileLang provider");
      failed = true;
      return "<unsupported-lgamma>";
    }
    llvm_unreachable("unhandled Intent unary operator");
  }

  std::string index(ValueRange offsets) {
    std::string result = "[";
    for (auto [axis, offset] : llvm::enumerate(offsets)) {
      if (axis)
        result += ", ";
      result += valueString(offset);
    }
    return result + "]";
  }

  std::string regionIndex(ValueRange offsets, ArrayRef<int64_t> viewAxes,
                          ArrayAttr bufferShape) {
    SmallVector<std::optional<unsigned>> viewToBuffer(offsets.size());
    for (auto [bufferAxis, viewAxis] : llvm::enumerate(viewAxes))
      viewToBuffer[viewAxis] = bufferAxis;
    std::string result = "[";
    for (auto [viewAxis, offset] : llvm::enumerate(offsets)) {
      if (viewAxis)
        result += ", ";
      std::string begin = valueString(offset);
      result += begin;
      if (std::optional<unsigned> bufferAxis = viewToBuffer[viewAxis])
        result += ":(" + begin + " + " +
                  expressionString(cast<gpu::PhysicalExprAttr>(
                      bufferShape[*bufferAxis])) +
                  ")";
    }
    return result + "]";
  }

  std::string tuple(ValueRange range) {
    std::string result = "(";
    for (auto [index, value] : llvm::enumerate(range)) {
      if (index)
        result += ", ";
      result += valueString(value);
    }
    if (range.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string viewShape(const ViewABI &view) {
    return shape(view.type.getLayout().getExtents());
  }

  std::string valueString(Value value) {
    auto found = values.find(value);
    if (found == values.end()) {
      failed = true;
      return "<missing>";
    }
    return found->second;
  }

  void assign(Value value, const std::string &expression) {
    std::string name = "v" + std::to_string(counter++);
    values[value] = name;
    line(name + " = " + expression);
  }

  void line(const std::string &text, unsigned explicitIndent = ~0U) {
    unsigned level = explicitIndent == ~0U ? indent : explicitIndent;
    output.indent(level * 4) << text << "\n";
  }

  LogicalResult emitMetadata(std::string &metadata) {
    auto artifact = gpu::serializeInterface(kernel, "tilelang", [&](Value value) {
      return valueString(value);
    });
    if (mlir::failed(artifact)) return failure();
    llvm::json::Array arguments, builderArguments;
    for (const ViewABI &view : views) arguments.push_back(view.name);
    for (const ScalarABI &scalar : scalars) arguments.push_back(scalar.name);
    for (const MetadataABI &argument : metadataArguments)
      builderArguments.push_back(argument.name);
    for (gpu::ViewOverlapOp overlap : overlapFacts)
      builderArguments.push_back(valueString(overlap.getResult()));
    for (const std::string &name : parameterNames) builderArguments.push_back(name);
    for (const std::string &name : coverageNames) builderArguments.push_back(name);
    (*artifact)["tilelang"] = llvm::json::Object{
        {"kernel", "_intent_kernel"}, {"kernel_arguments", std::move(arguments)},
        {"builder_arguments", std::move(builderArguments)}};
    llvm::raw_string_ostream(metadata) << llvm::json::Value(std::move(*artifact));
    return failed ? failure() : success();
  }

  func::FuncOp kernel;
  raw_ostream &output;
  llvm::DenseMap<Value, std::string> values;
  SmallVector<ViewABI> views;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  SmallVector<gpu::ViewOverlapOp> overlapFacts;
  SmallVector<std::string> parameterNames;
  std::set<std::string> coverageNames;
  unsigned indent = 0;
  unsigned counter = 0;
  bool failed = false;
  bool hasApproximateMath = false;
};

} // namespace

LogicalResult serializeProgram(ModuleOp module, std::string &source,
                               std::string &metadata) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (!(*kernel)->hasAttr("intent_tilelang.legalized"))
    return (*kernel).emitError("TileLang program was not provider-legalized");
  llvm::raw_string_ostream stream(source);
  Serializer serializer(*kernel, stream);
  LogicalResult result = serializer.emit(metadata);
  stream.flush();
  return result;
}

} // namespace intent::tilelang

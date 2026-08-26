#include "Intent/Target/TileLang/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <map>
#include <sstream>

using namespace mlir;

namespace intent::tilelang {
namespace {

std::string tileLangType(Type type, bool torch = false) {
  if (type.isIndex())
    return torch ? "torch.int64" : "T.int64";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return torch ? "torch.bool" : "T.bool";
    StringRef prefix = integer.isUnsigned() ? "uint" : "int";
    return ((torch ? "torch." : "T.") + prefix + Twine(integer.getWidth())).str();
  }
  if (isa<Float16Type>(type))
    return torch ? "torch.float16" : "T.float16";
  if (isa<BFloat16Type>(type))
    return torch ? "torch.bfloat16" : "T.bfloat16";
  if (isa<Float32Type>(type))
    return torch ? "torch.float32" : "T.float32";
  if (isa<Float8E4M3FNType>(type))
    return torch ? "torch.float8_e4m3fn" : "T.float8_e4m3fn";
  if (isa<Float8E5M2Type>(type))
    return torch ? "torch.float8_e5m2" : "T.float8_e5m2";
  return {};
}

std::string expressionString(gpu::PhysicalExprAttr expression) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return std::to_string(expression.getValue());
  if (kind == gpu::PhysicalExprKind::Parameter ||
      kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return expression.getSymbol().getValue().str();
  SmallVector<std::string> operands;
  for (Attribute operand : expression.getOperands())
    operands.push_back(expressionString(cast<gpu::PhysicalExprAttr>(operand)));
  if (kind == gpu::PhysicalExprKind::Add)
    return "(" + operands[0] + " + " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Subtract)
    return "(" + operands[0] + " - " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Multiply)
    return "(" + operands[0] + " * " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::CeilDiv)
    return "T.ceildiv(" + operands[0] + ", " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Minimum)
    return "T.min(" + operands[0] + ", " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Maximum)
    return "T.max(" + operands[0] + ", " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::FloorDiv)
    return "(" + operands[0] + " // " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Select)
    return "T.if_then_else(" + operands[0] + ", " + operands[1] + ", " +
           operands[2] + ")";
  if (kind == gpu::PhysicalExprKind::NextPowerOfTwo)
    return "T.next_power_of_2(" + operands[0] + ")";
  return {};
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
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (integer.getType().isInteger(1))
      return integer.getInt() ? "True" : "False";
    return std::to_string(integer.getInt());
  }
  if (auto floating = dyn_cast<FloatAttr>(value)) {
    std::ostringstream stream;
    stream << std::setprecision(17) << floating.getValueAsDouble();
    std::string result = stream.str();
    if (result.find_first_of(".eE") == std::string::npos)
      result += ".0";
    return result;
  }
  return {};
}

struct ParameterDomain {
  std::string name;
  SmallVector<int64_t> candidates;
};

class Serializer {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : kernel(kernel), output(output) {}

  LogicalResult emit() {
    bindArguments();
    collectConfiguration();
    emitPreamble();
    emitBuilder();
    emitLaunch();
    emitRun();
    return failed ? failure() : success();
  }

private:
  struct ViewABI {
    unsigned argument;
    std::string name;
    gpu::ViewType type;
  };
  struct MetadataABI {
    unsigned argument;
    std::string name;
    std::string kind;
    unsigned sourceABI;
    unsigned sourceAxis;
    int64_t dimension;
  };
  struct ScalarABI {
    unsigned argument;
    std::string name;
    std::string kind;
    Type type;
  };

  void bindArguments() {
    for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
      DictionaryAttr attrs = kernel.getArgAttrDict(index);
      std::string kind = attrs.getAs<StringAttr>(gpu::abiKindAttr).getValue().str();
      std::string name = attrs.getAs<StringAttr>(gpu::abiNameAttr).getValue().str();
      values[argument] = name;
      if (kind == "view") {
        views.push_back({static_cast<unsigned>(index), name,
                         cast<gpu::ViewType>(argument.getType())});
      } else if (kind == "scalar" || kind == "constexpr" || kind == "value") {
        if (kind == "constexpr") {
          if (!argument.use_empty()) {
            kernel.emitError(
                "live constexpr reached TileLang runtime ABI after specialization");
            failed = true;
          }
          continue;
        }
        scalars.push_back({static_cast<unsigned>(index), name, kind,
                           argument.getType()});
      } else {
        MetadataABI metadata{
            static_cast<unsigned>(index), name, kind,
            static_cast<unsigned>(attrs.getAs<IntegerAttr>(gpu::sourceABIAttr).getInt()),
            static_cast<unsigned>(attrs.getAs<IntegerAttr>(gpu::sourceAxisAttr).getInt()),
            attrs.getAs<IntegerAttr>(gpu::dimensionAttr)
                ? attrs.getAs<IntegerAttr>(gpu::dimensionAttr).getInt()
                : 0};
        metadataArguments.push_back(metadata);
        if (kind == "dimension")
          dimensionBindings[metadata.dimension] = metadata;
      }
    }
  }

  void collectConfiguration() {
    llvm::StringSet<> names;
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto schema = parameter.getParameter();
      auto role = static_cast<gpu::ParameterRole>(schema.getRole());
      if (role == gpu::ParameterRole::ProviderWarps ||
          role == gpu::ParameterRole::ProviderCTAs) {
        parameter.emitOpError(
            "TileLang source cannot bind a foreign provider parameter role");
        failed = true;
        return;
      }
      StringRef name = schema.getName().getValue();
      if (!names.insert(name).second) {
        parameter.emitOpError("duplicates a TileLang physical parameter");
        failed = true;
        return;
      }
      domains.push_back(
          {name.str(), SmallVector<int64_t>(schema.getCandidates().asArrayRef())});
      values[parameter.getResult()] = name.str();
    });
  }

  void emitPreamble() {
    output << "import torch\nimport tilelang\nimport tilelang.language as T\n"
              "from tilelang.autotuner import set_autotune_inputs\n\n";
  }

  SmallVector<std::map<std::string, int64_t>> configurations() {
    SmallVector<std::map<std::string, int64_t>> configs(1);
    auto expand = [&](StringRef name, ArrayRef<int64_t> candidates) {
      SmallVector<std::map<std::string, int64_t>> next;
      for (const auto &base : configs)
        for (int64_t candidate : candidates) {
          auto value = base;
          value[name.str()] = candidate;
          next.push_back(std::move(value));
        }
      configs = std::move(next);
    };
    for (const ParameterDomain &domain : domains)
      expand(domain.name, domain.candidates);
    return configs;
  }

  void emitBuilder() {
    output << "@tilelang.autotune(configs=[\n";
    for (const auto &config : configurations()) {
      output << "    {";
      for (auto [index, item] : llvm::enumerate(config)) {
        if (index)
          output << ", ";
        output << "\"" << item.first << "\": " << item.second;
      }
      output << "},\n";
    }
    output << "], warmup=3, rep=10)\n@tilelang.jit\ndef _intent_kernel(";
    bool first = true;
    auto argument = [&](StringRef text) {
      if (!first)
        output << ", ";
      first = false;
      output << text;
    };
    for (const MetadataABI &metadata : metadataArguments)
      argument(metadata.name);
    for (const ScalarABI &scalar : scalars)
      if (scalar.kind == "constexpr")
        argument(scalar.name);
    for (const ParameterDomain &domain : domains)
      argument(domain.name);
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
      if (scalar.kind == "constexpr")
        continue;
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
    emitBlock(kernel.getBody().front());
    line("return main", 1);
    output << "\n";
  }

  void emitLaunch() {
    output << "_KERNEL_CACHE = {}\n\ndef launch(" << joinViewNames(views);
    for (const ScalarABI &scalar : scalars)
      if (scalar.kind != "constexpr")
        output << (views.empty() ? "" : ", ") << scalar.name;
    output << "):\n";
    for (const MetadataABI &metadata : metadataArguments) {
      const ViewABI &source = viewByABI(metadata.sourceABI);
      line(metadata.name + " = " + source.name +
               (metadata.kind == "dimension" ? ".shape[" : ".stride(") +
               std::to_string(metadata.sourceAxis) +
               (metadata.kind == "dimension" ? "]" : ")"),
           1);
    }
    std::string key = "key = (";
    for (const ViewABI &view : views)
      key += "tuple(" + view.name + ".shape), " + view.name + ".dtype, str(" +
             view.name + ".device), ";
    for (const ScalarABI &scalar : scalars)
      if (scalar.kind != "constexpr")
        key += scalar.name + ", ";
    line(key + ")", 1);
    line("if key not in _KERNEL_CACHE:", 1);
    line("with set_autotune_inputs(" + joinKernelRuntimeArguments() + "):", 2);
    std::string compile = "_KERNEL_CACHE[key] = _intent_kernel.compile(";
    for (auto [index, metadata] : llvm::enumerate(metadataArguments)) {
      if (index)
        compile += ", ";
      compile += metadata.name + "=" + metadata.name;
    }
    compile += ")";
    line(compile, 3);
    line("kernel = _KERNEL_CACHE[key]", 1);
    line("kernel(" + joinKernelRuntimeArguments() + ")", 1);
    line("return kernel", 1);
    output << "\n";
  }

  void emitRun() {
    SmallVector<ViewABI> inputs;
    SmallVector<ViewABI> outputs;
    for (const ViewABI &view : views) {
      if (view.type.getAccess() != 1)
        inputs.push_back(view);
      if (view.type.getAccess() == 1)
        outputs.push_back(view);
    }
    output << "def run(" << joinViewNames(inputs);
    for (const ScalarABI &scalar : scalars)
      if (scalar.kind != "constexpr")
        output << (inputs.empty() ? "" : ", ") << scalar.name;
    output << "):\n";
    std::string device = inputs.empty() ? "'cuda'" : inputs.front().name + ".device";
    for (const ViewABI &view : outputs)
      line(view.name + " = torch.empty(" + outputShape(view) + ", device=" +
               device + ", dtype=" + tileLangType(view.type.getElementType(), true) +
               ")",
           1);
    line("launch(" + joinLaunchArguments() + ")", 1);
    if (outputs.empty())
      line("return None", 1);
    else if (outputs.size() == 1)
      line("return " + outputs.front().name, 1);
    else
      line("return (" + joinViewNames(outputs) + ")", 1);
  }

  void emitBlock(Block &block) {
    for (Operation &operation : block) {
      if (isa<scf::YieldOp, YieldOp, func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
  }

  void emitOperation(Operation &operation) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
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
      static constexpr const char *predicates[] = {"==", "!=", "<", "<=", ">", ">="};
      assign(compare.getResult(), "(" + valueString(compare.getLhs()) + " " +
                                      predicates[compare.getPredicate()] + " " +
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
      std::string function = type.getSpace() == 0 ? "T.alloc_shared" : "T.alloc_fragment";
      assign(allocation.getResult(), function + "(" + shape(type.getShape()) +
                                         ", " + tileLangType(type.getElementType()) + ")");
    } else if (auto clear = dyn_cast<ClearOp>(operation)) {
      line("T.clear(" + valueString(clear.getBuffer()) + ")");
    } else if (auto fill = dyn_cast<FillOp>(operation)) {
      line("T.fill(" + valueString(fill.getBuffer()) + ", " +
           valueString(fill.getValue()) + ")");
    } else if (auto copy = dyn_cast<CopyInOp>(operation)) {
      line("T.copy(" + valueString(copy.getSource()) + index(copy.getOffsets()) +
           ", " + valueString(copy.getDestination()) + ")");
    } else if (auto copy = dyn_cast<CopyOutOp>(operation)) {
      line("T.copy(" + valueString(copy.getSource()) + ", " +
           valueString(copy.getDestination()) + index(copy.getOffsets()) + ")");
    } else if (auto copy = dyn_cast<CastCopyOutOp>(operation)) {
      line("T.copy(" + valueString(copy.getSource()) + ", " +
           valueString(copy.getDestination()) + index(copy.getOffsets()) + ")");
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
      static constexpr const char *functions[] = {
          "T.reduce_sum", "T.reduce_max", "T.reduce_min"};
      line(std::string(functions[reduce.getKind()]) + "(" +
           valueString(reduce.getSource()) + ", " +
           valueString(reduce.getDestination()) + ", dim=" +
           std::to_string(reduce.getAxis()) + ", clear=False)");
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      std::string function = scan.getKind() == 0 ? "T.cumsum" : "T.cummax";
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
    static constexpr const char *operators[] = {
        "+", "-", "*", "/", "//", "%", "**", nullptr, nullptr,
        nullptr, nullptr, "&", "|", "&", "|", "^", "<<", ">>"};
    uint64_t kind = binary.getOperatorKind();
    if (kind == 7 || kind == 9)
      return "T.max(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    if (kind == 8 || kind == 10)
      return "T.min(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    return "(" + valueString(binary.getLhs()) + " " + operators[kind] + " " +
           valueString(binary.getRhs()) + ")";
  }

  std::string unaryExpression(gpu::UnaryOp unary) {
    std::string input = valueString(unary.getInput());
    static constexpr const char *functions[] = {
        nullptr, nullptr, "T.exp", "T.exp2", "T.log", "T.sin", "T.cos",
        "T.floor", "T.erf", "T.rsqrt", "T.sigmoid", "T.tanh", "T.abs"};
    if (unary.getOperatorKind() == 0)
      return "(-" + input + ")";
    if (unary.getOperatorKind() == 1)
      return "(~" + input + ")";
    return std::string(functions[unary.getOperatorKind()]) + "(" + input + ")";
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

  std::string outputShape(const ViewABI &view) {
    std::string result = "(";
    auto ids = view.type.getLayout().getDimensionIds();
    auto extents = view.type.getLayout().getExtents();
    for (auto [axis, dimension] : llvm::enumerate(ids.asArrayRef())) {
      if (axis)
        result += ", ";
      auto extent = cast<gpu::PhysicalExprAttr>(extents[axis]);
      if (extent.getKind() ==
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant)) {
        result += std::to_string(extent.getValue());
      } else {
        auto found = dimensionBindings.find(dimension);
        if (found == dimensionBindings.end()) {
          failed = true;
          return "<missing-output-shape>";
        }
        const MetadataABI &metadata = found->second;
        const ViewABI &source = viewByABI(metadata.sourceABI);
        result += source.name + ".shape[" + std::to_string(metadata.sourceAxis) + "]";
      }
    }
    if (ids.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string joinKernelRuntimeArguments() const {
    std::string result = joinViewNames(views);
    for (const ScalarABI &scalar : scalars)
      if (scalar.kind != "constexpr") {
        if (!result.empty())
          result += ", ";
        result += scalar.name;
      }
    return result;
  }

  std::string joinLaunchArguments() const {
    return joinKernelRuntimeArguments();
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

  const ViewABI &viewByABI(unsigned abi) const {
    for (const ViewABI &view : views)
      if (view.type.getAbiIndex() == abi)
        return view;
    llvm_unreachable("verified metadata references a missing view ABI");
  }

  template <typename Range>
  std::string joinViewNames(const Range &range) const {
    std::string result;
    for (auto [index, view] : llvm::enumerate(range)) {
      if (index)
        result += ", ";
      result += view.name;
    }
    return result;
  }

  func::FuncOp kernel;
  raw_ostream &output;
  llvm::DenseMap<Value, std::string> values;
  SmallVector<ViewABI> views;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  llvm::DenseMap<int64_t, MetadataABI> dimensionBindings;
  SmallVector<ParameterDomain> domains;
  unsigned indent = 0;
  unsigned counter = 0;
  bool failed = false;
};

} // namespace

LogicalResult serializeProgram(ModuleOp module, std::string &source) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (!(*kernel)->hasAttr("intent_tilelang.legalized"))
    return (*kernel).emitError("TileLang program was not provider-legalized");
  llvm::raw_string_ostream stream(source);
  Serializer serializer(*kernel, stream);
  LogicalResult result = serializer.emit();
  stream.flush();
  return result;
}

} // namespace intent::tilelang

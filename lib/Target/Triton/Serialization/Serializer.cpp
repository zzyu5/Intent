#include "Intent/Target/Triton/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>

using namespace mlir;

namespace intent::triton {
namespace {

constexpr llvm::StringLiteral reduceFormAttr = "intent_gpu.triton.reduce_form";

std::string pythonType(Type type, bool torch = false) {
  if (type.isIndex())
    return torch ? "torch.int64" : "tl.int64";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return torch ? "torch.bool" : "tl.int1";
    StringRef prefix = integer.isUnsigned() ? "u" : "";
    return ((torch ? "torch." : "tl.") + prefix + "int" +
            Twine(integer.getWidth()))
        .str();
  }
  if (isa<Float16Type>(type))
    return torch ? "torch.float16" : "tl.float16";
  if (isa<BFloat16Type>(type))
    return torch ? "torch.bfloat16" : "tl.bfloat16";
  if (isa<Float32Type>(type))
    return torch ? "torch.float32" : "tl.float32";
  if (isa<Float64Type>(type))
    return torch ? "torch.float64" : "tl.float64";
  if (isa<Float8E4M3FNType>(type))
    return torch ? "torch.float8_e4m3fn" : "tl.float8e4nv";
  if (isa<Float8E5M2Type>(type))
    return torch ? "torch.float8_e5m2" : "tl.float8e5";
  return {};
}

std::string expressionString(gpu::PhysicalExprAttr expression,
                             bool launchContext) {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return std::to_string(expression.getValue());
  if (kind == gpu::PhysicalExprKind::Parameter)
    return launchContext
               ? ("META[\"" + expression.getSymbol().getValue() + "\"]").str()
               : expression.getSymbol().getValue().str();
  if (kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return expression.getSymbol().getValue().str();
  SmallVector<std::string> operands;
  for (Attribute operand : expression.getOperands())
    operands.push_back(
        expressionString(cast<gpu::PhysicalExprAttr>(operand), launchContext));
  if (kind == gpu::PhysicalExprKind::Add)
    return "(" + operands[0] + " + " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Subtract)
    return "(" + operands[0] + " - " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Multiply)
    return "(" + operands[0] + " * " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::CeilDiv)
    return "triton.cdiv(" + operands[0] + ", " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Minimum)
    return "min(" + operands[0] + ", " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Maximum)
    return "max(" + operands[0] + ", " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::FloorDiv)
    return "(" + operands[0] + " // " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Select)
    return "(" + operands[1] + " if " + operands[0] + " else " +
           operands[2] + ")";
  if (kind == gpu::PhysicalExprKind::NextPowerOfTwo)
    return "triton.next_power_of_2(" + operands[0] + ")";
  return {};
}

std::string fragmentShape(gpu::FragmentType fragment) {
  SmallVector<std::string> extents;
  for (Attribute extent : fragment.getShape())
    extents.push_back(
        expressionString(cast<gpu::PhysicalExprAttr>(extent), false));
  std::string result = "(";
  for (auto [index, extent] : llvm::enumerate(extents)) {
    if (index)
      result += ", ";
    result += extent;
  }
  if (extents.size() == 1)
    result += ",";
  return result + ")";
}

struct Config {
  std::map<std::string, int64_t> kernelParameters;
  int64_t warps = 0;
  int64_t stages = 0;
  int64_t ctas = 0;
};

struct CoverageParameter {
  std::string dimension;
  SmallVector<int64_t> candidates;
};

FailureOr<SmallVector<Config>> parameterConfigs(func::FuncOp kernel) {
  auto encoded = kernel->getAttrOfType<ArrayAttr>(gpu::tritonConfigsAttr);
  if (!encoded || encoded.empty())
    return kernel.emitError(
        "Triton legalization did not materialize a legal candidate set");
  SmallVector<Config> configs;
  for (Attribute candidate : encoded) {
    auto dictionary = dyn_cast<DictionaryAttr>(candidate);
    auto parameters = dictionary
                          ? dictionary.getAs<DictionaryAttr>("parameters")
                          : DictionaryAttr();
    auto warps = dictionary ? dictionary.getAs<IntegerAttr>("num_warps")
                            : IntegerAttr();
    auto stages = dictionary ? dictionary.getAs<IntegerAttr>("num_stages")
                             : IntegerAttr();
    auto ctas = dictionary ? dictionary.getAs<IntegerAttr>("num_ctas")
                           : IntegerAttr();
    if (!dictionary || !parameters || !warps || !stages || !ctas)
      return kernel.emitError("contains a malformed legalized Triton candidate");
    Config config;
    config.warps = warps.getInt();
    config.stages = stages.getInt();
    config.ctas = ctas.getInt();
    for (NamedAttribute parameter : parameters)
      config.kernelParameters[parameter.getName().strref().str()] =
          cast<IntegerAttr>(parameter.getValue()).getInt();
    configs.push_back(std::move(config));
  }
  return configs;
}

std::string literal(Attribute value) {
  if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (integer.getType().isInteger(1))
      return integer.getInt() ? "True" : "False";
    return std::to_string(integer.getInt());
  }
  if (auto floating = dyn_cast<FloatAttr>(value)) {
    double number = floating.getValueAsDouble();
    if (std::isnan(number))
      return "float(\"nan\")";
    if (std::isinf(number))
      return std::signbit(number) ? "-float(\"inf\")"
                                  : "float(\"inf\")";
    std::ostringstream stream;
    stream << std::setprecision(17) << number;
    std::string result = stream.str();
    if (result.find_first_of(".eE") == std::string::npos)
      result += ".0";
    return result;
  }
  return {};
}

class Serializer {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : kernel(kernel), output(output) {}

  LogicalResult emit() {
    bindArguments();
    emitPreamble();
    emitHelpers();
    emitKernel();
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
        views.push_back(
            {static_cast<unsigned>(index), name, cast<gpu::ViewType>(argument.getType())});
        continue;
      }
      if (kind == "scalar" || kind == "constexpr" || kind == "value") {
        if (kind == "constexpr") {
          if (!argument.use_empty()) {
            kernel.emitError(
                "live constexpr reached Triton runtime ABI after specialization");
            failed = true;
          }
          continue;
        }
        scalars.push_back({static_cast<unsigned>(index), name, kind,
                           argument.getType()});
        continue;
      }
      MetadataABI metadata{
          static_cast<unsigned>(index), name, kind,
          static_cast<unsigned>(attrs.getAs<IntegerAttr>(gpu::sourceABIAttr).getInt()),
          static_cast<unsigned>(attrs.getAs<IntegerAttr>(gpu::sourceAxisAttr).getInt()),
          attrs.getAs<IntegerAttr>(gpu::dimensionAttr)
              ? attrs.getAs<IntegerAttr>(gpu::dimensionAttr).getInt()
              : 0};
      metadataByName[name] = metadata;
      metadataArguments.push_back(metadata);
      if (kind == "dimension")
        dimensionBindings[metadata.dimension] = metadata;
    }
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto schema = parameter.getParameter();
      std::string name = schema.getName().getValue().str();
      parameterRoles[name] = schema.getRole();
      if (auto logicalDimension =
              parameter->getAttrOfType<IntegerAttr>(gpu::dimensionAttr))
        parameterDimensions[name] = logicalDimension.getInt();
      auto dimension =
          parameter->getAttrOfType<IntegerAttr>(gpu::coverageDimensionAttr);
      if (!dimension)
        return;
      auto binding = dimensionBindings.find(dimension.getInt());
      if (binding == dimensionBindings.end()) {
        parameter.emitOpError(
            "full-coverage parameter references a non-ABI dimension");
        failed = true;
        return;
      }
      parameterDimensions[name] = dimension.getInt();
      fullCoverageParameters[name] = {
          binding->second.name,
          SmallVector<int64_t>(schema.getCandidates().asArrayRef())};
    });
  }

  void emitPreamble() {
    output << "import torch\nimport triton\nimport triton.language as tl\n"
              "from triton.language.extra import libdevice\n\n";
  }

  void emitHelper(Operation *owner, Region &region, StringRef role) {
    std::string name =
        ("_intent_" + role + "_" + Twine(helperCounter++)).str();
    helperNames[owner].push_back(name);
    Block &block = region.front();
    output << "@triton.jit\ndef " << name << "(";
    for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
      if (index)
        output << ", ";
      std::string argumentName = "a" + std::to_string(index);
      output << argumentName;
      values[argument] = argumentName;
    }
    output << "):\n";
    unsigned savedIndent = indent;
    bool savedEmittingHelper = emittingHelper;
    indent = 1;
    emittingHelper = true;
    for (Operation &operation : block.without_terminator())
      emitOperation(operation);
    auto yield = dyn_cast<gpu::YieldOp>(block.getTerminator());
    if (!yield) {
      failed = true;
    } else {
      std::string result = "return ";
      if (yield.getValues().size() > 1)
        result += "(";
      for (auto [index, value] : llvm::enumerate(yield.getValues())) {
        if (index)
          result += ", ";
        result += valueString(value);
      }
      if (yield.getValues().size() > 1)
        result += ")";
      line(result);
    }
    indent = savedIndent;
    emittingHelper = savedEmittingHelper;
    output << "\n";
  }

  void emitHelpers() {
    kernel.walk([&](Operation *operation) {
      if (auto reduce = dyn_cast<gpu::ReduceOp>(operation)) {
        if (!reduce->hasAttr(reduceFormAttr))
          emitHelper(operation, reduce.getCombine(), "reduce");
      }
      else if (auto scan = dyn_cast<gpu::ScanOp>(operation))
        emitHelper(operation, scan.getCombine(), "scan");
    });
  }

  void emitKernel() {
    FailureOr<SmallVector<Config>> configs = parameterConfigs(kernel);
    if (mlir::failed(configs)) {
      failed = true;
      return;
    }
    for (const auto &[parameter, coverage] : fullCoverageParameters) {
      output << "def _intent_cover_" << parameter << "(args):\n"
             << "    bound = int(args[\"" << coverage.dimension << "\"])\n"
             << "    for extent in (";
      for (int64_t candidate : coverage.candidates)
        output << candidate << ", ";
      output << "):\n"
             << "        if extent >= bound:\n"
             << "            return extent\n"
             << "    raise ValueError(\"no legal full-coverage extent for "
             << parameter << "\")\n\n";
    }
    output << "@triton.autotune(\n    configs=[\n";
    for (const Config &config : *configs) {
      output << "        triton.Config({";
      bool first = true;
      for (const auto &[name, value] : config.kernelParameters) {
        if (!first)
          output << ", ";
        first = false;
        output << "\"" << name << "\": "
               << value;
      }
      output << "}, num_warps=" << config.warps
             << ", num_stages=" << config.stages
             << ", num_ctas=" << config.ctas
             << "),\n";
    }
    output << "    ],\n    key=[";
    bool firstKey = true;
    for (const MetadataABI &metadata : metadataArguments) {
      if (metadata.kind != "dimension")
        continue;
      if (!firstKey)
        output << ", ";
      firstKey = false;
      output << "\"" << metadata.name << "\"";
    }
    output << "],\n";
    output << ")\n";
    if (!fullCoverageParameters.empty()) {
      output << "@triton.heuristics({\n";
      for (const auto &[parameter, coverage] : fullCoverageParameters)
        output << "    \"" << parameter << "\": _intent_cover_" << parameter
               << ",\n";
      output << "})\n";
    }
    output << "@triton.jit\ndef _intent_kernel(";
    bool first = true;
    for (const ViewABI &view : views) {
      if (!first)
        output << ", ";
      first = false;
      output << view.name;
    }
    for (const ScalarABI &scalar : scalars) {
      if (!first)
        output << ", ";
      first = false;
      output << scalar.name;
      if (scalar.kind == "constexpr")
        output << ": tl.constexpr";
    }
    for (const MetadataABI &metadata : metadataArguments) {
      if (!first)
        output << ", ";
      first = false;
      output << metadata.name << ": tl.constexpr";
    }
    SmallVector<std::string> parameters;
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto role = static_cast<gpu::ParameterRole>(
          parameter.getParameter().getRole());
      if (role == gpu::ParameterRole::ProviderWarps ||
          role == gpu::ParameterRole::ProviderStages ||
          role == gpu::ParameterRole::ProviderCTAs)
        return;
      std::string name = parameter.getParameter().getName().getValue().str();
      parameters.push_back(name);
      values[parameter.getResult()] = name;
    });
    for (StringRef parameter : parameters)
      output << ", " << parameter << ": tl.constexpr";
    output << "):\n";
    indent = 1;
    emitBlock(kernel.getBody().front(), /*isLoop=*/false, {});
    output << "\n";
  }

  void emitLaunch() {
    output << "_intent_parameter_roles = {";
    bool firstRole = true;
    for (const auto &[name, role] : parameterRoles) {
      if (!firstRole)
        output << ", ";
      firstRole = false;
      output << "\"" << name << "\": " << role;
    }
    output << "}\n";
    output << "_intent_parameter_dimensions = {";
    bool firstDimension = true;
    for (const auto &[name, dimension] : parameterDimensions) {
      if (!firstDimension)
        output << ", ";
      firstDimension = false;
      output << "\"" << name << "\": " << dimension;
    }
    output << "}\n"
              "for _intent_config in _intent_kernel.configs:\n"
              "    _intent_config.intent_parameter_roles = _intent_parameter_roles\n"
              "    _intent_config.intent_parameter_dimensions = _intent_parameter_dimensions\n\n";
    output << "def launch(";
    bool firstArgument = true;
    for (auto [index, view] : llvm::enumerate(views)) {
      if (!firstArgument)
        output << ", ";
      firstArgument = false;
      output << view.name;
    }
    for (const ScalarABI &scalar : scalars) {
      if (!firstArgument)
        output << ", ";
      firstArgument = false;
      output << scalar.name;
    }
    output << "):\n";
    for (const MetadataABI &metadata : metadataArguments) {
      const ViewABI &source = viewByABI(metadata.sourceABI);
      line(metadata.name + " = " + source.name +
           (metadata.kind == "dimension" ? ".shape[" : ".stride(") +
           std::to_string(metadata.sourceAxis) +
           (metadata.kind == "dimension" ? "]" : ")"), 1);
    }
    auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
    std::string grid = "grid = lambda META: (";
    for (auto [index, extent] : llvm::enumerate(space)) {
      if (index)
        grid += ", ";
      grid += expressionString(cast<gpu::PhysicalExprAttr>(extent), true);
    }
    if (space.size() == 1)
      grid += ",";
    line(grid + ")", 1);
    std::string call = "return _intent_kernel[grid](";
    bool first = true;
    for (const ViewABI &view : views) {
      if (!first)
        call += ", ";
      first = false;
      call += view.name;
    }
    for (const ScalarABI &scalar : scalars) {
      if (!first)
        call += ", ";
      first = false;
      call += scalar.name;
    }
    for (const MetadataABI &metadata : metadataArguments) {
      call += ", " + metadata.name;
    }
    line(call + ")", 1);
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
    output << "def run(";
    for (auto [index, view] : llvm::enumerate(inputs)) {
      if (index)
        output << ", ";
      output << view.name;
    }
    for (const ScalarABI &scalar : scalars) {
      if (!inputs.empty() || &scalar != &scalars.front())
        output << ", ";
      output << scalar.name;
    }
    output << "):\n";
    if (outputs.empty()) {
      line("launch(" + joinLaunchArguments() + ")", 1);
      line("return None", 1);
      return;
    }
    std::string device = inputs.empty() ? "'cuda'" : inputs.front().name + ".device";
    for (const ViewABI &view : outputs) {
      std::string shape = "(";
      auto ids = view.type.getLayout().getDimensionIds();
      auto extents = view.type.getLayout().getExtents();
      for (auto [axis, dimension] : llvm::enumerate(ids.asArrayRef())) {
        if (axis)
          shape += ", ";
        auto extent = cast<gpu::PhysicalExprAttr>(extents[axis]);
        if (extent.getKind() ==
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant)) {
          shape += std::to_string(extent.getValue());
          continue;
        }
        auto binding = dimensionBindings.find(dimension);
        if (binding == dimensionBindings.end()) {
          failed = true;
          return;
        }
        const MetadataABI &metadata = binding->second;
        const ViewABI &source = viewByABI(metadata.sourceABI);
        shape += source.name + ".shape[" + std::to_string(metadata.sourceAxis) + "]";
      }
      if (ids.size() == 1)
        shape += ",";
      shape += ")";
      line(view.name + " = torch.empty(" + shape + ", device=" + device +
               ", dtype=" + pythonType(view.type.getElementType(), true) + ")",
           1);
    }
    line("launch(" + joinLaunchArguments() + ")", 1);
    if (outputs.size() == 1)
      line("return " + outputs.front().name, 1);
    else
      line("return (" + joinViewNames(outputs) + ")", 1);
  }

  void emitBlock(Block &block, bool isLoop,
                 ArrayRef<std::string> loopResults) {
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (isLoop || !loopResults.empty())
          for (auto [name, value] : llvm::zip(loopResults, yield.getOperands()))
            line(name + " = " + valueString(value));
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
  }

  void emitOperation(Operation &operation) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      values[constant.getResult()] = literal(constant.getValue());
      return;
    }
    if (auto parameter = dyn_cast<gpu::ParameterOp>(operation)) {
      values[parameter.getResult()] =
          parameter.getParameter().getName().getValue().str();
      return;
    }
    if (auto physical = dyn_cast<gpu::PhysicalExprOp>(operation)) {
      assign(physical.getResult(),
             expressionString(physical.getExpression(), false));
      return;
    }
    if (auto program = dyn_cast<gpu::ProgramIdOp>(operation)) {
      assign(program.getResult(),
             "tl.program_id(" + std::to_string(program.getAxis()) + ")");
      return;
    }
    if (auto coordinate = dyn_cast<gpu::WorksetCoordinateOp>(operation)) {
      values[coordinate.getResult()] = valueString(coordinate.getCoordinate());
      return;
    }
    if (auto dim = dyn_cast<gpu::DimOp>(operation)) {
      auto view = dim.getView().getType();
      auto extent = cast<gpu::PhysicalExprAttr>(
          view.getLayout().getExtents()[dim.getAxis()]);
      assign(dim.getResult(), expressionString(extent, false));
      return;
    }
    if (auto range = dyn_cast<gpu::RangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) + ", " +
                                    valueString(range.getStop()) + ", " +
                                    valueString(range.getStep()) + ")");
      return;
    }
    if (auto bound = dyn_cast<gpu::RangeBoundOp>(operation)) {
      assign(bound.getResult(), valueString(bound.getRange()) + "[" +
                                    std::to_string(bound.getBound()) + "]");
      return;
    }
    if (auto mapping = dyn_cast<gpu::DelinearizeOp>(operation)) {
      std::string remaining = valueString(mapping.getLinear());
      SmallVector<std::string> coordinates(mapping.getCoordinates().size());
      for (int64_t axis = static_cast<int64_t>(mapping.getCoordinates().size()) - 1;
           axis >= 0; --axis) {
        std::string extent = valueString(mapping.getExtents()[axis]);
        coordinates[axis] = "(" + remaining + " % " + extent + ")";
        remaining = "(" + remaining + " // " + extent + ")";
      }
      for (auto [coordinate, expression] :
           llvm::zip(mapping.getCoordinates(), coordinates))
        assign(coordinate, expression);
      return;
    }
    if (auto binary = dyn_cast<gpu::BinaryOp>(operation)) {
      assign(binary.getResult(), binaryExpression(binary));
      return;
    }
    if (auto unary = dyn_cast<gpu::UnaryOp>(operation)) {
      assign(unary.getResult(), unaryExpression(unary));
      return;
    }
    if (auto compare = dyn_cast<gpu::CompareOp>(operation)) {
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
      return;
    }
    if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = cast<gpu::FragmentType>(range.getResult().getType());
      auto physicalExtent =
          cast<gpu::PhysicalExprAttr>(fragment.getShape()[0]);
      assign(range.getResult(), "(" + valueString(range.getStart()) +
                                    " + tl.arange(0, " +
                                    expressionString(physicalExtent, false) + ") * " +
                                    valueString(range.getStep()) + ")");
      return;
    }
    if (auto splat = dyn_cast<gpu::SplatOp>(operation)) {
      // Triton reduce/scan helpers receive accumulator values directly and
      // scalar operands broadcast through ordinary elementwise expressions.
      // Their ABI has no outer-kernel constexpr shape parameters, so spelling
      // a typed scalar splat with fragmentShape() would reference symbols that
      // are intentionally outside the helper scope.
      if (emittingHelper) {
        assign(splat.getResult(), valueString(splat.getValue()));
        return;
      }
      auto type = splat.getResult().getType();
      assign(splat.getResult(), "tl.full(" + fragmentShape(type) + ", " +
                                    valueString(splat.getValue()) + ", " +
                                    pythonType(type.getElementType()) + ")");
      return;
    }
    if (auto broadcast = dyn_cast<gpu::BroadcastOp>(operation)) {
      assign(broadcast.getResult(), broadcastValue(broadcast.getValue(),
                                                   broadcast.getResult().getType()));
      return;
    }
    if (auto cast = dyn_cast<gpu::CastOp>(operation)) {
      assign(cast.getResult(), "tl.cast(" + valueString(cast.getValue()) +
                                   ", " +
                                   pythonType(elementType(cast.getResult().getType())) +
                                   ")");
      return;
    }
    if (auto bitcast = dyn_cast<gpu::BitcastOp>(operation)) {
      assign(bitcast.getResult(), "tl.cast(" + valueString(bitcast.getValue()) +
                                      ", " +
                                      pythonType(elementType(bitcast.getResult().getType())) +
                                      ", bitcast=True)");
      return;
    }
    if (auto select = dyn_cast<gpu::SelectOp>(operation)) {
      assign(select.getResult(), "tl.where(" + valueString(select.getCondition()) +
                                      ", " + valueString(select.getTrueValue()) +
                                      ", " + valueString(select.getFalseValue()) + ")");
      return;
    }
    if (auto reshape = dyn_cast<gpu::ReshapeOp>(operation)) {
      auto target = cast<gpu::FragmentType>(reshape.getResult().getType());
      assign(reshape.getResult(), "tl.reshape(" + valueString(reshape.getValue()) +
                                       ", " + fragmentShape(target) +
                                       ", can_reorder=False)");
      return;
    }
    if (auto transpose = dyn_cast<gpu::TransposeOp>(operation)) {
      std::string permutation = "(";
      for (auto [index, axis] : llvm::enumerate(transpose.getPermutation())) {
        if (index)
          permutation += ", ";
        permutation += std::to_string(axis);
      }
      if (transpose.getPermutation().size() == 1)
        permutation += ",";
      permutation += ")";
      assign(transpose.getResult(), "tl.permute(" + valueString(transpose.getValue()) +
                                         ", " + permutation + ")") ;
      return;
    }
    if (auto join = dyn_cast<gpu::JoinOp>(operation)) {
      assign(join.getResult(), "tl.join(" + valueString(join.getLhs()) + ", " +
                                   valueString(join.getRhs()) + ")");
      return;
    }
    if (auto record = dyn_cast<gpu::MakeRecordOp>(operation)) {
      std::string tuple = "(";
      for (auto [index, field] : llvm::enumerate(record.getFields())) {
        if (index)
          tuple += ", ";
        tuple += valueString(field);
      }
      if (record.getFields().size() == 1)
        tuple += ",";
      assign(record.getResult(), tuple + ")");
      return;
    }
    if (auto extract = dyn_cast<gpu::ExtractOp>(operation)) {
      assign(extract.getResult(), valueString(extract.getRecord()) + "[" +
                                       std::to_string(extract.getField()) + "]");
      return;
    }
    if (auto load = dyn_cast<gpu::LoadOp>(operation)) {
      std::string call = "tl.load(" + pointer(load.getResource(),
                                               load.getCoordinates(),
                                               load.getSourceAxes(),
                                               load.getResult().getType());
      if (load.getValid())
        call += ", mask=" + valueString(load.getValid()) +
                ", other=" + valueString(load.getFill());
      assign(load.getResult(), call + ")");
      return;
    }
    if (auto gather = dyn_cast<gpu::GatherOp>(operation)) {
      std::string call = "tl.gather(" + valueString(gather.getSource()) + ", " +
                         valueString(gather.getCoordinates().front()) +
                         ", axis=" +
                         std::to_string(gather.getSourceAxes().front()) + ")";
      assign(gather.getResult(), call);
      return;
    }
    if (auto assumption = dyn_cast<gpu::AssumeInBoundsOp>(operation)) {
      auto view = cast<gpu::ViewType>(assumption.getResource().getType());
      auto extent = cast<gpu::PhysicalExprAttr>(
          view.getLayout().getExtents()[assumption.getAxis()]);
      std::string predicate =
          "((" + valueString(assumption.getIndex()) + " >= 0) & (" +
          valueString(assumption.getIndex()) + " < " +
          expressionString(extent, false) + "))";
      if (auto fragment =
              dyn_cast<gpu::FragmentType>(assumption.getIndex().getType())) {
        predicate = "(" + predicate + ").to(tl.int32)";
        for (int64_t axis = static_cast<int64_t>(fragment.getShape().size()) - 1;
             axis >= 0; --axis)
          predicate = "tl.min(" + predicate + ", axis=" +
                      std::to_string(axis) + ")";
        predicate = "(" + predicate + " != 0)";
      }
      line("tl.assume(" + predicate + ")");
      return;
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      assign(contract.getResult(), "tl.dot(" + valueString(contract.getLhs()) +
                                      ", " + valueString(contract.getRhs()) +
                                      ", " + valueString(contract.getAccumulator()) +
                                      ")");
      return;
    }
    if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      auto format = [](ScaledFormat value) -> StringRef {
        switch (value) {
        case ScaledFormat::E2M1: return "e2m1";
        case ScaledFormat::E4M3: return "e4m3";
        case ScaledFormat::E8M0: return "e8m0";
        }
        llvm_unreachable("unhandled Intent scaled format");
      };
      assign(contract.getResult(),
             "tl.dot_scaled(" + valueString(contract.getLhs()) + ", " +
                 valueString(contract.getLhsScale()) + ", \"" +
                 format(contract.getLhsFormat()).str() + "\", " +
                 valueString(contract.getRhs()) + ", " +
                 valueString(contract.getRhsScale()) + ", \"" +
                 format(contract.getRhsFormat()).str() + "\", " +
                 valueString(contract.getAccumulator()) + ")");
      return;
    }
    if (auto reduce = dyn_cast<gpu::ReduceOp>(operation)) {
      std::string sources;
      ValueRange sourceValues =
          reduce.getInputs().take_front(reduce.getSourceCount());
      if (sourceValues.size() == 1) {
        sources = valueString(sourceValues.front());
      } else {
        sources = "(";
        for (auto [index, source] : llvm::enumerate(sourceValues)) {
          if (index)
            sources += ", ";
          sources += valueString(source);
        }
        sources += ")";
      }
      std::string call;
      if (auto form = reduce->getAttrOfType<StringAttr>(reduceFormAttr)) {
        if (form.getValue() != "sum") {
          reduce.emitOpError("has an unknown Triton native reduction form");
          failed = true;
          return;
        }
        call = "tl.sum(" + sources + ", axis=" +
               std::to_string(reduce.getAxes().front()) + ")";
      } else {
        call = "tl.reduce(" + sources + ", axis=" +
               std::to_string(reduce.getAxes().front()) +
               ", combine_fn=" + helperNames.lookup(&operation).front() + ")";
      }
      assignResults(reduce.getResults(), call);
      return;
    }
    if (auto scan = dyn_cast<gpu::ScanOp>(operation)) {
      std::string sources;
      ValueRange sourceValues = scan.getInputs().take_front(scan.getSourceCount());
      if (sourceValues.size() == 1) {
        sources = valueString(sourceValues.front());
      } else {
        sources = "(";
        for (auto [index, source] : llvm::enumerate(sourceValues)) {
          if (index)
            sources += ", ";
          sources += valueString(source);
        }
        sources += ")";
      }
      std::string call =
          "tl.associative_scan(" + sources + ", axis=" +
          std::to_string(scan.getAxis()) + ", combine_fn=" +
          helperNames.lookup(&operation).front() +
          ", reverse=" + (scan.getReverse() ? "True" : "False") + ")";
      assignResults(scan.getResults(), call);
      return;
    }
    if (auto histogram = dyn_cast<gpu::HistogramOp>(operation)) {
      assign(histogram.getResult(), "tl.histogram(" +
                                         valueString(histogram.getValues()) + ", " +
                                         valueString(histogram.getBins()) +
                                         ", mask=" + valueString(histogram.getValid()) +
                                         ")");
      return;
    }
    if (auto random = dyn_cast<gpu::RandomBitsOp>(operation)) {
      assign(random.getResult(), "tl.randint(" + valueString(random.getSeed()) +
                                      ", " + valueString(random.getCounter()) +
                                      ").to(tl.uint32, bitcast=True)");
      return;
    }
    if (auto atomic = dyn_cast<gpu::AtomicStoreOp>(operation)) {
      std::string call = "tl.atomic_xchg(" +
                         pointer(atomic.getResource(), atomic.getCoordinates(),
                                 atomic.getSourceAxes(),
                                 atomic.getValue().getType()) +
                         ", " + valueString(atomic.getValue());
      if (atomic.getValid())
        call += ", mask=" + valueString(atomic.getValid());
      call += ", sem=\"" + atomicSemantics(atomic.getOrdering()) +
              "\", scope=\"" + atomicScope(atomic.getSharing()) + "\")";
      line(call);
      return;
    }
    if (auto atomic = dyn_cast<gpu::AtomicRMWOp>(operation)) {
      auto operationName = [](AtomicRMWKind kind) -> StringRef {
        switch (kind) {
        case AtomicRMWKind::Exchange: return "xchg";
        case AtomicRMWKind::Add: return "add";
        case AtomicRMWKind::Maximum: return "max";
        case AtomicRMWKind::Minimum: return "min";
        case AtomicRMWKind::BitwiseAnd: return "and";
        case AtomicRMWKind::BitwiseOr: return "or";
        case AtomicRMWKind::BitwiseXor: return "xor";
        }
        llvm_unreachable("unhandled Intent atomic RMW kind");
      };
      std::string call = "tl.atomic_" + operationName(atomic.getKind()).str() +
                         "(" +
                         pointer(atomic.getResource(), atomic.getCoordinates(),
                                 atomic.getSourceAxes(),
                                 atomic.getValue().getType()) +
                         ", " + valueString(atomic.getValue());
      if (atomic.getValid())
        call += ", mask=" + valueString(atomic.getValid());
      call += ", sem=\"" + atomicSemantics(atomic.getOrdering()) +
              "\", scope=\"" + atomicScope(atomic.getSharing()) + "\")";
      assign(atomic.getResult(), call);
      return;
    }
    if (auto atomic = dyn_cast<gpu::AtomicCompareExchangeOp>(operation)) {
      std::string old = newName();
      line(old + " = tl.atomic_cas(" +
           pointer(atomic.getResource(), atomic.getCoordinates(),
                   atomic.getSourceAxes(),
                   atomic.getExpected().getType()) +
           ", " + valueString(atomic.getExpected()) + ", " +
           valueString(atomic.getDesired()) + ", sem=\"" +
           atomicSemantics(atomic.getOrdering()) + "\", scope=\"" +
           atomicScope(atomic.getSharing()) + "\")");
      assign(atomic.getResult(), "(" + old + ", (" + old + " == " +
                                     valueString(atomic.getExpected()) + "))");
      return;
    }
    if (auto store = dyn_cast<gpu::StoreOp>(operation)) {
      std::string call = "tl.store(" +
                         pointer(store.getResource(), store.getCoordinates(),
                                 store.getSourceAxes(),
                                 store.getValue().getType()) +
                         ", " + valueString(store.getValue());
      if (store.getValid())
        call += ", mask=" + valueString(store.getValid());
      line(call + ")");
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      SmallVector<std::string> results;
      for (auto [result, initial] : llvm::zip(loop.getResults(), loop.getInitArgs())) {
        std::string name = newName();
        values[result] = name;
        line(name + " = " + valueString(initial));
        results.push_back(name);
      }
      std::string induction = "iv" + std::to_string(counter++);
      values[loop.getInductionVar()] = induction;
      for (auto [argument, name] : llvm::zip(loop.getRegionIterArgs(), results))
        values[argument] = name;
      line("for " + induction + " in range(" + valueString(loop.getLowerBound()) +
           ", " + valueString(loop.getUpperBound()) + ", " +
           valueString(loop.getStep()) + "):");
      ++indent;
      emitBlock(*loop.getBody(), true, results);
      --indent;
      return;
    }
    if (auto ifOperation = dyn_cast<scf::IfOp>(operation)) {
      SmallVector<std::string> results;
      for (Value result : ifOperation.getResults()) {
        std::string name = newName();
        values[result] = name;
        results.push_back(name);
      }
      line("if " + valueString(ifOperation.getCondition()) + ":");
      ++indent;
      emitBlock(ifOperation.getThenRegion().front(), false, results);
      --indent;
      if (!ifOperation.getElseRegion().empty() &&
          !ifOperation.getElseRegion().front().empty()) {
        line("else:");
        ++indent;
        emitBlock(ifOperation.getElseRegion().front(), false, results);
        --indent;
      }
      return;
    }
    if (auto whileOperation = dyn_cast<scf::WhileOp>(operation)) {
      SmallVector<std::string> carries;
      for (auto [result, initial] :
           llvm::zip(whileOperation.getResults(), whileOperation.getInits())) {
        std::string name = newName();
        values[result] = name;
        line(name + " = " + valueString(initial));
        carries.push_back(name);
      }
      Block &before = whileOperation.getBefore().front();
      for (auto [argument, name] : llvm::zip(before.getArguments(), carries))
        values[argument] = name;
      line("while True:");
      ++indent;
      for (Operation &nested : before.without_terminator())
        emitOperation(nested);
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      line("if not " + valueString(condition.getCondition()) + ":");
      ++indent;
      line("break");
      --indent;
      Block &after = whileOperation.getAfter().front();
      for (auto [argument, forwarded] :
           llvm::zip(after.getArguments(), condition.getArgs()))
        values[argument] = valueString(forwarded);
      emitBlock(after, true, carries);
      --indent;
      return;
    }
    operation.emitOpError("has no terminal Triton spelling");
    failed = true;
  }

  Type elementType(Type type) const {
    if (auto fragment = dyn_cast<gpu::FragmentType>(type))
      return fragment.getElementType();
    return type;
  }

  std::string binaryExpression(gpu::BinaryOp binary) {
    auto infix = [&](StringRef spelling) {
      return "(" + valueString(binary.getLhs()) + " " + spelling.str() + " " +
             valueString(binary.getRhs()) + ")";
    };
    auto call = [&](StringRef function, StringRef propagateNan = {}) {
      std::string result = function.str() + "(" + valueString(binary.getLhs()) +
                           ", " + valueString(binary.getRhs());
      if (!propagateNan.empty())
        result += ", propagate_nan=tl.PropagateNan." + propagateNan.str();
      return result + ")";
    };
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: return infix("+");
    case BinaryOperator::Subtract: return infix("-");
    case BinaryOperator::Multiply: return infix("*");
    case BinaryOperator::TrueDivide: return infix("/");
    case BinaryOperator::FloorDivide: return infix("//");
    case BinaryOperator::Remainder: return infix("%");
    case BinaryOperator::Power: return call("libdevice.pow");
    case BinaryOperator::Maximum: return call("tl.maximum", "ALL");
    case BinaryOperator::Minimum: return call("tl.minimum", "ALL");
    case BinaryOperator::MaximumNum: return call("tl.maximum", "NONE");
    case BinaryOperator::MinimumNum: return call("tl.minimum", "NONE");
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
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate:
      return "(-" + input + ")";
    case UnaryOperator::Not:
      return "(~" + input + ")";
    case UnaryOperator::Exp:
      return "tl.exp(" + input + ")";
    case UnaryOperator::Exp2:
      return "tl.exp2(" + input + ")";
    case UnaryOperator::Log:
      return "tl.log(" + input + ")";
    case UnaryOperator::Sin:
      return "tl.sin(" + input + ")";
    case UnaryOperator::Cos:
      return "tl.cos(" + input + ")";
    case UnaryOperator::Floor:
      return "tl.floor(" + input + ")";
    case UnaryOperator::Erf:
      return "tl.erf(" + input + ")";
    case UnaryOperator::Rsqrt:
      return "tl.rsqrt(" + input + ")";
    case UnaryOperator::Sigmoid:
      return "tl.sigmoid(" + input + ")";
    case UnaryOperator::Tanh:
      return "tl.libdevice.tanh(" + input + ")";
    case UnaryOperator::Abs:
      return "tl.abs(" + input + ")";
    case UnaryOperator::Sqrt:
      return "tl.sqrt(" + input + ")";
    default:
      failed = true;
      return "<unsupported-unary>";
    }
  }

  std::string atomicSemantics(AtomicOrdering ordering) const {
    switch (ordering) {
    case AtomicOrdering::Relaxed: return "relaxed";
    case AtomicOrdering::Acquire: return "acquire";
    case AtomicOrdering::Release: return "release";
    case AtomicOrdering::AcquireRelease: return "acq_rel";
    }
    llvm_unreachable("unhandled Intent atomic ordering");
  }

  std::string atomicScope(gpu::AtomicSharingDomain sharing) const {
    switch (sharing) {
    case gpu::AtomicSharingDomain::ProgramInstance: return "cta";
    case gpu::AtomicSharingDomain::KernelInvocation: return "gpu";
    }
    llvm_unreachable("unhandled atomic sharing domain");
  }

  std::string broadcastValue(Value value, gpu::FragmentType target) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return valueString(value);
    if (source == target)
      return valueString(value);
    if (source.getShape().size() > target.getShape().size()) {
      kernel.emitError("Triton broadcast source rank exceeds target rank");
      failed = true;
      return {};
    }
    // Triton broadcasts singleton extents implicitly.  An equal-rank
    // intent_gpu.broadcast therefore changes only the logical extent carried
    // by the typed program; emitting another indexing expression would insert
    // a new rank instead of expanding the existing singleton axis.
    if (source.getShape().size() == target.getShape().size())
      return valueString(value);
    SmallVector<std::optional<unsigned>> projection(target.getShape().size());
    SmallVector<bool> sourceUsed(source.getShape().size(), false);
    for (auto [targetIndex, targetMapping] :
         llvm::enumerate(target.getAxisMaps())) {
      auto targetAxis = cast<gpu::AxisMapAttr>(targetMapping);
      for (auto [sourceIndex, sourceMapping] :
           llvm::enumerate(source.getAxisMaps())) {
        auto sourceAxis = cast<gpu::AxisMapAttr>(sourceMapping);
        if (sourceAxis.getSourceId() == targetAxis.getSourceId() &&
            sourceAxis.getSourceAxis() == targetAxis.getSourceAxis()) {
          if (sourceUsed[sourceIndex]) {
            kernel.emitError("Triton broadcast reuses one source axis ambiguously");
            failed = true;
            return {};
          }
          projection[targetIndex] = sourceIndex;
          sourceUsed[sourceIndex] = true;
          break;
        }
      }
    }
    unsigned offset = target.getShape().size() - source.getShape().size();
    for (unsigned sourceIndex = 0; sourceIndex < source.getShape().size();
         ++sourceIndex) {
      if (sourceUsed[sourceIndex])
        continue;
      unsigned targetIndex = offset + sourceIndex;
      if (projection[targetIndex]) {
        kernel.emitError("Triton broadcast axis mapping is ambiguous");
        failed = true;
        return {};
      }
      projection[targetIndex] = sourceIndex;
    }
    SmallVector<std::string> selectors;
    for (std::optional<unsigned> sourceIndex : projection)
      selectors.push_back(sourceIndex ? ":" : "None");
    std::string result = valueString(value) + "[";
    for (auto [index, selector] : llvm::enumerate(selectors)) {
      if (index)
        result += ", ";
      result += selector;
    }
    return result + "]";
  }

  std::string pointer(Value resource, ValueRange coordinates,
                      ArrayRef<int64_t> sourceAxes, Type valueType) {
    auto argument = dyn_cast<BlockArgument>(resource);
    if (!argument) {
      failed = true;
      return {};
    }
    auto view = cast<gpu::ViewType>(resource.getType());
    auto strides = view.getLayout().getStrides();
    auto fragment = dyn_cast<gpu::FragmentType>(valueType);
    std::string result = values.lookup(resource);
    for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
      auto stride =
          cast<StringAttr>(strides[sourceAxes[axis]]).getValue().str();
      std::string coordinateExpression =
          fragment ? broadcastValue(coordinate, fragment) : valueString(coordinate);
      result += " + (" + coordinateExpression + ") * " + stride;
    }
    return "(" + result + ")";
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
    std::string name = newName();
    values[value] = name;
    line(name + " = " + expression);
  }

  void assignResults(ResultRange results, const std::string &expression) {
    SmallVector<std::string> names;
    for (Value result : results) {
      names.push_back(newName());
      values[result] = names.back();
    }
    std::string statement;
    for (auto [index, name] : llvm::enumerate(names)) {
      if (index)
        statement += ", ";
      statement += name;
    }
    line(statement + " = " + expression);
  }

  std::string newName() { return "v" + std::to_string(counter++); }

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

  std::string joinLaunchArguments() const {
    std::string result = joinViewNames(views);
    for (const ScalarABI &scalar : scalars) {
      if (!result.empty())
        result += ", ";
      result += scalar.name;
    }
    return result;
  }

  func::FuncOp kernel;
  raw_ostream &output;
  llvm::DenseMap<Value, std::string> values;
  SmallVector<ViewABI> views;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  llvm::StringMap<MetadataABI> metadataByName;
  llvm::DenseMap<int64_t, MetadataABI> dimensionBindings;
  std::map<std::string, CoverageParameter> fullCoverageParameters;
  std::map<std::string, uint32_t> parameterRoles;
  std::map<std::string, int64_t> parameterDimensions;
  llvm::DenseMap<Operation *, SmallVector<std::string>> helperNames;
  unsigned indent = 0;
  unsigned counter = 0;
  unsigned helperCounter = 0;
  bool emittingHelper = false;
  bool failed = false;
};

} // namespace

LogicalResult serializeProgram(ModuleOp module, std::string &source) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (function->hasAttr(gpu::kernelAttr))
      kernels.push_back(function);
  if (kernels.size() != 1)
    return module.emitError(
        "Triton serialization requires one provider-legalized kernel");
  func::FuncOp kernel = kernels.front();
  if (!kernel->hasAttr("intent_gpu.triton.legalized"))
    return kernel.emitError("Triton program was not provider-legalized");
  llvm::raw_string_ostream stream(source);
  Serializer serializer(kernel, stream);
  LogicalResult result = serializer.emit();
  stream.flush();
  return result;
}

} // namespace intent::triton

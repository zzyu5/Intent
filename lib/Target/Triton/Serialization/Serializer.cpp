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

#include <iomanip>
#include <map>
#include <sstream>

using namespace mlir;

namespace intent::triton {
namespace {

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

struct ParameterDomain {
  std::string name;
  gpu::ParameterRole role;
  SmallVector<int64_t> candidates;
};

struct Config {
  std::map<std::string, int64_t> kernelParameters;
  int64_t warps = 0;
  int64_t stages = 0;
  int64_t ctas = 0;
};

FailureOr<SmallVector<ParameterDomain>> parameterDomains(func::FuncOp kernel) {
  SmallVector<ParameterDomain> domains;
  llvm::StringSet<> names;
  WalkResult result = kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    StringRef name = schema.getName().getValue();
    if (!names.insert(name).second) {
      parameter.emitOpError("duplicates a Triton physical/provider parameter");
      return WalkResult::interrupt();
    }
    domains.push_back({name.str(), static_cast<gpu::ParameterRole>(schema.getRole()),
                       SmallVector<int64_t>(schema.getCandidates().asArrayRef())});
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return domains;
}

FailureOr<SmallVector<Config>> parameterConfigs(func::FuncOp kernel) {
  FailureOr<SmallVector<ParameterDomain>> domains = parameterDomains(kernel);
  if (failed(domains))
    return failure();
  SmallVector<Config> configs(1);
  for (const ParameterDomain &domain : *domains) {
    SmallVector<Config> expanded;
    for (const Config &base : configs) {
      for (int64_t candidate : domain.candidates) {
        Config next = base;
        switch (domain.role) {
        case gpu::ParameterRole::ProviderWarps:
          next.warps = candidate;
          break;
        case gpu::ParameterRole::ProviderStages:
          next.stages = candidate;
          break;
        case gpu::ParameterRole::ProviderCTAs:
          next.ctas = candidate;
          break;
        default:
          next.kernelParameters[domain.name] = candidate;
          break;
        }
        expanded.push_back(std::move(next));
      }
    }
    configs = std::move(expanded);
  }
  for (const Config &config : configs)
    if (config.warps <= 0 || config.stages <= 0 || config.ctas <= 0)
      return kernel.emitError(
          "Triton provider parameter domains are incomplete");
  return configs;
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
    return stream.str();
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
  }

  void emitPreamble() {
    output << "import torch\nimport triton\nimport triton.language as tl\n\n";
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
    indent = 1;
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
    output << "\n";
  }

  void emitHelpers() {
    kernel.walk([&](Operation *operation) {
      if (auto reduce = dyn_cast<gpu::ReduceOp>(operation))
        emitHelper(operation, reduce.getCombine(), "reduce");
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
    output << "],\n)\n@triton.jit\ndef _intent_kernel(";
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
      static constexpr const char *predicates[] = {"==", "!=", "<", "<=", ">", ">="};
      assign(compare.getResult(), "(" + valueString(compare.getLhs()) + " " +
                                      predicates[compare.getPredicate()] + " " +
                                      valueString(compare.getRhs()) + ")");
      return;
    }
    if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) +
                                    " + tl.arange(0, " +
                                    valueString(range.getExtent()) + ") * " +
                                    valueString(range.getStep()) + ")");
      return;
    }
    if (auto splat = dyn_cast<gpu::SplatOp>(operation)) {
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
      line("tl.assume((" + valueString(assumption.getIndex()) + " >= 0) & (" +
           valueString(assumption.getIndex()) + " < " +
           expressionString(extent, false) + "))");
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
      static constexpr const char *formats[] = {"e2m1", "e4m3", "e8m0"};
      assign(contract.getResult(),
             "tl.dot_scaled(" + valueString(contract.getLhs()) + ", " +
                 valueString(contract.getLhsScale()) + ", \"" +
                 formats[contract.getLhsFormat()] + "\", " +
                 valueString(contract.getRhs()) + ", " +
                 valueString(contract.getRhsScale()) + ", \"" +
                 formats[contract.getRhsFormat()] + "\", " +
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
      std::string call = "tl.reduce(" + sources + ", axis=" +
                         std::to_string(reduce.getAxes().front()) +
                         ", combine_fn=" + helperNames.lookup(&operation).front() +
                         ")";
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
      static constexpr const char *operations[] = {
          "xchg", "add", "max", "min", "and", "or", "xor"};
      std::string call = "tl.atomic_" + std::string(operations[atomic.getKind()]) +
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
    static constexpr const char *operators[] = {
        "+", "-", "*", "/", "//", "%", "**", nullptr, nullptr,
        nullptr, nullptr, "&", "|", "&", "|", "^", "<<", ">>"};
    uint64_t kind = binary.getOperatorKind();
    if (kind == 7 || kind == 9)
      return "tl.maximum(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    if (kind == 8 || kind == 10)
      return "tl.minimum(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    return "(" + valueString(binary.getLhs()) + " " + operators[kind] + " " +
           valueString(binary.getRhs()) + ")";
  }

  std::string unaryExpression(gpu::UnaryOp unary) {
    std::string input = valueString(unary.getInput());
    switch (unary.getOperatorKind()) {
    case 0:
      return "(-" + input + ")";
    case 1:
      return "(~" + input + ")";
    case 2:
      return "tl.exp(" + input + ")";
    case 3:
      return "tl.exp2(" + input + ")";
    case 4:
      return "tl.log(" + input + ")";
    case 5:
      return "tl.sin(" + input + ")";
    case 6:
      return "tl.cos(" + input + ")";
    case 7:
      return "tl.floor(" + input + ")";
    case 8:
      return "tl.erf(" + input + ")";
    case 9:
      return "tl.rsqrt(" + input + ")";
    case 10:
      return "tl.sigmoid(" + input + ")";
    case 11:
      return "tl.libdevice.tanh(" + input + ")";
    case 12:
      return "tl.abs(" + input + ")";
    default:
      failed = true;
      return "<unsupported-unary>";
    }
  }

  std::string atomicSemantics(uint64_t ordering) const {
    static constexpr const char *semantics[] = {
        "relaxed", "acquire", "release", "acq_rel"};
    return semantics[ordering];
  }

  std::string atomicScope(uint64_t sharing) const {
    static constexpr const char *scopes[] = {"cta", "gpu", "sys"};
    return scopes[sharing];
  }

  std::string broadcastValue(Value value, gpu::FragmentType target) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return valueString(value);
    if (source == target)
      return valueString(value);
    SmallVector<std::string> selectors;
    for (Attribute targetMapping : target.getAxisMaps()) {
      auto targetAxis = cast<gpu::AxisMapAttr>(targetMapping);
      bool found = false;
      for (Attribute sourceMapping : source.getAxisMaps()) {
        auto sourceAxis = cast<gpu::AxisMapAttr>(sourceMapping);
        if (sourceAxis.getSourceId() == targetAxis.getSourceId() &&
            sourceAxis.getSourceAxis() == targetAxis.getSourceAxis()) {
          selectors.push_back(":");
          found = true;
          break;
        }
      }
      if (!found)
        selectors.push_back("None");
    }
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
  llvm::DenseMap<Operation *, SmallVector<std::string>> helperNames;
  unsigned indent = 0;
  unsigned counter = 0;
  unsigned helperCounter = 0;
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

#include "Intent/Target/Triton/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
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
  gpu::PhysicalExprAttr bound;
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
    for (NamedAttribute parameter : parameters) {
      auto value = dyn_cast<IntegerAttr>(parameter.getValue());
      if (!value)
        return kernel.emitError(
            "contains a non-integer legalized Triton parameter binding");
      config.kernelParameters[parameter.getName().strref().str()] =
          value.getInt();
    }
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
    emitDescriptorHelpers();
    emitConfigPruner();
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
    bool workspace;
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
  struct DescriptorABI {
    TensorDescriptorOp operation;
    std::string name;
  };

  void bindArguments() {
    llvm::StringSet<> argumentNames;
    for (unsigned index = 0; index < kernel.getNumArguments(); ++index)
      argumentNames.insert(kernel.getArgAttrDict(index)
                               .getAs<StringAttr>(gpu::abiNameAttr).getValue());
    for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
      DictionaryAttr attrs = kernel.getArgAttrDict(index);
      std::string kind = attrs.getAs<StringAttr>(gpu::abiKindAttr).getValue().str();
      std::string name = attrs.getAs<StringAttr>(gpu::abiNameAttr).getValue().str();
      // Triton's launch kwargs also enter autotune hook argument maps.  Keep
      // view names out of that namespace (for example, an input named grid).
      if (kind == "view" || kind == "workspace") {
        name = "_intent_view_" + std::to_string(index);
        while (!argumentNames.insert(name).second)
          name += "_";
      }
      values[argument] = name;
      if (kind == "view" || kind == "workspace") {
        views.push_back({static_cast<unsigned>(index), name,
                         cast<gpu::ViewType>(argument.getType()),
                         kind == "workspace"});
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
      constexprValues.insert(argument);
      if (kind == "dimension")
        dimensionBindings[metadata.dimension] = metadata;
    }
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto schema = parameter.getParameter();
      std::string name = schema.getName().getValue().str();
      argumentNames.insert(name);
      auto dimension =
          parameter->getAttrOfType<IntegerAttr>(gpu::coverageDimensionAttr);
      if (!dimension)
        return;
      auto bound = parameter->getAttrOfType<gpu::PhysicalExprAttr>(
          gpu::coverageBoundAttr);
      if (!bound) {
        parameter.emitOpError(
            "full-coverage parameter has no typed bound expression");
        failed = true;
        return;
      }
      fullCoverageParameters[name] = {
          bound,
          SmallVector<int64_t>(schema.getCandidates().asArrayRef())};
    });
    kernel.walk([&](TensorDescriptorChoiceOp choice) {
      descriptorChoice = choice;
      values[choice.getResult()] = choice.getConfigParameter().str();
      argumentNames.insert(choice.getConfigParameter());
      argumentNames.insert(choice.getEligibilityArgument());
    });
    kernel.walk([&](TensorDescriptorAllocatorOp allocator) {
      descriptorAllocator = allocator;
    });
    kernel.walk([&](TensorDescriptorOp descriptor) {
      std::string name = "_intent_descriptor_" + std::to_string(descriptors.size());
      while (!argumentNames.insert(name).second)
        name += "_";
      values[descriptor.getResult()] = name;
      descriptors.push_back({descriptor, name});
    });
    while (!argumentNames.insert(overlapFunction).second)
      overlapFunction += "_";
    while (!argumentNames.insert(overlapSpanFunction).second)
      overlapSpanFunction += "_";
    while (!argumentNames.insert(overlapArgument).second)
      overlapArgument += "_";
    kernel.walk([&](ViewOverlapOp overlap) {
      values[overlap.getResult()] = overlapArgument + "[" +
                                   std::to_string(overlapFacts.size()) + "]";
      overlapFacts.push_back(overlap);
      constexprValues.insert(overlap.getResult());
    });
    if (descriptorChoice && !descriptorAllocator) {
      kernel.emitError(
          "tensor descriptor form has no declared allocator requirement");
      failed = true;
    }
  }

  void emitPreamble() {
    output << "import torch\nimport triton\nimport triton.language as tl\n"
              "from triton.language.extra import libdevice\n"
              "from triton.tools.tensor_descriptor import TensorDescriptor\n"
              "from intent.runtime.triton import TuningHooks\n"
              "from intent.runtime.triton_math import contract_fma\n\n";
    if (!overlapFacts.empty())
      output << "from intent.runtime.tuning import byte_spans_overlap as "
             << overlapFunction << ", view_byte_span as "
             << overlapSpanFunction << "\n\n";
  }

  void emitDescriptorHelpers() {
    if (!descriptorChoice)
      return;
    output << "def _intent_tensor_descriptor_legal(\n"
              "    tensor, shape, strides, source_rank,\n"
              "    aligned_stride_axes, unit_stride_axes, require_positive_shape,\n"
              "    require_positive_strides, alignment, maximum_shape_extent,\n"
              "):\n"
              "    if tensor.ndim != source_rank:\n"
              "        return False\n"
              "    if tensor.data_ptr() % alignment != 0:\n"
              "        return False\n"
              "    if require_positive_shape and any(extent <= 0 for extent in shape):\n"
              "        return False\n"
              "    if any(extent > maximum_shape_extent for extent in shape):\n"
              "        return False\n"
              "    if require_positive_strides and any(stride <= 0 for stride in strides):\n"
              "        return False\n"
              "    if strides[-1] != 1:\n"
              "        return False\n"
              "    if any(tensor.stride(axis) != 1 for axis in unit_stride_axes):\n"
              "        return False\n"
              "    if any((tensor.stride(axis) * tensor.element_size()) % alignment != 0 "
              "for axis in aligned_stride_axes):\n"
              "        return False\n"
              "    return True\n\n"
              "def _intent_tensor_descriptor_block_shape_legal(\n"
              "    block_shape, element_size, minimum_contiguous_bytes,\n"
              "    require_power_of_two, maximum_block_elements,\n"
              "    pipeline_block_alignment, num_stages,\n"
              "):\n"
              "    elements = 1\n"
              "    for extent in block_shape:\n"
              "        if extent <= 0 or (require_power_of_two and extent & (extent - 1)):\n"
              "            return False\n"
              "        elements *= extent\n"
              "    return (\n"
              "        elements <= maximum_block_elements\n"
              "        and block_shape[-1] * element_size >= minimum_contiguous_bytes\n"
              "        and (num_stages <= 1 or elements * element_size % pipeline_block_alignment == 0)\n"
              "    )\n\n"
              "def _intent_tensor_descriptor_shapes_legal(args, num_stages):\n"
              "    descriptors = (\n";
    for (const DescriptorABI &descriptor : descriptors) {
      TensorDescriptorOp operation = descriptor.operation;
      output << "        (" << descriptorBlockShape(descriptor)
             << ", args[\"" << valueString(operation.getBase())
             << "\"].element_size(), "
             << operation.getMinimumContiguousBytes() << ", "
             << (operation.getRequirePowerOfTwoBlockShape() ? "True" : "False")
             << ", " << operation.getMaximumBlockElements() << ", "
             << operation.getPipelineBlockAlignment() << ", num_stages),\n";
    }
    output << "    )\n"
              "    for contract in descriptors:\n"
              "        if not _intent_tensor_descriptor_block_shape_legal(*contract):\n"
              "            return False\n"
              "    return True\n\n"
              "def _intent_tensor_descriptor_allocator(size, alignment, stream):\n"
              "    buffer = torch.empty(size, dtype=torch.int8, device=\"cuda\")\n"
              "    if buffer.data_ptr() % alignment != 0:\n"
              "        raise RuntimeError(\"Triton descriptor allocator returned a misaligned buffer\")\n"
              "    return buffer\n\n";
    for (const DescriptorABI &descriptor : descriptors) {
      TensorDescriptorOp operation = descriptor.operation;
      std::string base = "args[\"" + valueString(operation.getBase()) + "\"]";
      SmallVector<std::string> shape;
      SmallVector<std::string> strides;
      for (Value extent : operation.getShape())
        shape.push_back(descriptorHostValue(extent, /*argumentMap=*/true));
      for (Value stride : operation.getStrides())
        strides.push_back(descriptorHostValue(stride, /*argumentMap=*/true));
      output << "def _intent_bind" << descriptor.name << "(args):\n"
             << "    if not args[\"" << descriptorChoice.getConfigParameter()
             << "\"]:\n"
             << "        return " << base << "\n"
             << "    return TensorDescriptor(" << base
             << ", shape=" << stringList(shape)
             << ", strides=" << stringList(strides)
             << ", block_shape=" << descriptorBlockShape(descriptor)
             << ", padding=\"" << operation.getPadding() << "\")\n\n";
    }
  }

  void emitConfigPruner() {
    auto boundExpression = [&](Value value) -> std::string {
      if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
        return std::to_string(constant.value());
      if (auto physical = value.getDefiningOp<gpu::PhysicalExprOp>())
        if (isConstexprExpression(physical.getExpression()))
          return descriptorArgumentExpression(physical.getExpression());
      return {};
    };
    SmallVector<std::string> bounds;
    // The same current-IR assertions also guard compilation. Evaluate their
    // constexpr bounds before the native tuner spends its candidate budget.
    for (auto assertion : kernel.front().getOps<cf::AssertOp>()) {
      auto comparison = assertion.getArg().getDefiningOp<gpu::CompareOp>();
      if (!comparison || comparison.getPredicate() != ComparePredicate::Le)
        continue;
      std::string lhs = boundExpression(comparison.getLhs());
      std::string rhs = boundExpression(comparison.getRhs());
      if (!lhs.empty() && !rhs.empty())
        bounds.push_back("(" + lhs + " <= " + rhs + ")");
    }
    hasConfigPruner = descriptorChoice || !bounds.empty();
    if (!hasConfigPruner)
      return;
    output << "def _intent_prune_configs(configs, named_args, **kwargs):\n"
              "    retained = []\n"
              "    for config in configs:\n"
              "        args = {**named_args, **kwargs, **config.kwargs}\n";
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto role = static_cast<gpu::ParameterRole>(
          parameter.getParameter().getRole());
      StringRef field;
      if (role == gpu::ParameterRole::ProviderWarps)
        field = "num_warps";
      else if (role == gpu::ParameterRole::ProviderStages)
        field = "num_stages";
      else if (role == gpu::ParameterRole::ProviderCTAs)
        field = "num_ctas";
      if (!field.empty())
        output << "        args[\"" << parameter.getParameter().getName().getValue()
               << "\"] = config." << field << "\n";
    });
    if (descriptorChoice)
      output << "        if args[\"" << descriptorChoice.getConfigParameter()
             << "\"] and (not args[\""
             << descriptorChoice.getEligibilityArgument()
             << "\"] or not _intent_tensor_descriptor_shapes_legal(args, config.num_stages)):\n"
                "            continue\n";
    for (const std::string &bound : bounds)
      output << "        if not " << bound << ":\n"
                "            continue\n";
    output << "        retained.append(config)\n"
              "    return retained\n\n";
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
      if (auto reduce = dyn_cast<ReduceOp>(operation))
        emitHelper(operation, reduce.getCombine(), "reduce");
      else if (auto scan = dyn_cast<ScanOp>(operation))
        emitHelper(operation, scan.getCombine(), "scan");
      else if (auto map = dyn_cast<MapElementwiseOp>(operation))
        emitHelper(operation, map.getBody(), "map");
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
             << "    bound = int(" << descriptorArgumentExpression(coverage.bound)
             << ")\n"
             << "    for extent in (";
      for (int64_t candidate : coverage.candidates)
        output << candidate << ", ";
      output << "):\n"
             << "        if extent >= bound:\n"
             << "            return extent\n"
             << "    raise ValueError(\"no legal full-coverage extent for "
             << parameter << "\")\n\n";
    }
    output << "_intent_tuning_hooks = TuningHooks((";
    for (const ViewABI &view : views)
      if (!view.workspace)
        output << "\"" << view.name << "\", ";
    output << "), (";
    for (const ViewABI &view : views)
      if (!view.workspace)
        output << (view.type.getAccess() != 0 ? "True, " : "False, ");
    output << "), (";
    for (const ViewABI &view : views)
      if (!view.workspace)
        output << (view.type.getAccess() != 1 ? "True, " : "False, ");
    output << "))\n\n@triton.autotune(\n    configs=[\n";
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
             << ", num_ctas=" << config.ctas << "),\n";
    }
    output << "    ],\n    key=[";
    bool firstKey = true;
    for (const MetadataABI &metadata : metadataArguments) {
      if (!firstKey)
        output << ", ";
      firstKey = false;
      output << "\"" << metadata.name << "\"";
    }
    if (!overlapFacts.empty()) {
      if (!firstKey)
        output << ", ";
      firstKey = false;
      output << "\"" << overlapArgument << "\"";
    }
    if (descriptorChoice) {
      if (!firstKey)
        output << ", ";
      output << "\"" << descriptorChoice.getEligibilityArgument() << "\"";
    }
    output << "],\n";
    output << "    pre_hook=_intent_tuning_hooks.before,\n"
              "    post_hook=_intent_tuning_hooks.after,\n";
    if (hasConfigPruner)
      output << "    prune_configs_by={\"early_config_prune\": "
                "_intent_prune_configs},\n";
    output << ")\n";
    if (!fullCoverageParameters.empty() || !descriptors.empty()) {
      output << "@triton.heuristics({\n";
      for (const auto &[parameter, coverage] : fullCoverageParameters)
        output << "    \"" << parameter << "\": _intent_cover_" << parameter
               << ",\n";
      for (const DescriptorABI &descriptor : descriptors)
        output << "    \"" << descriptor.name << "\": _intent_bind"
               << descriptor.name << ",\n";
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
      else
        output << ": " << pythonType(scalar.type);
    }
    for (const MetadataABI &metadata : metadataArguments) {
      if (!first)
        output << ", ";
      first = false;
      output << metadata.name << ": tl.constexpr";
    }
    if (!overlapFacts.empty()) {
      if (!first)
        output << ", ";
      first = false;
      output << overlapArgument << ": tl.constexpr";
    }
    if (descriptorChoice) {
      if (!first)
        output << ", ";
      first = false;
      output << descriptorChoice.getEligibilityArgument() << ": tl.constexpr";
      output << ", " << descriptorChoice.getConfigParameter()
             << ": tl.constexpr";
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
    for (const DescriptorABI &descriptor : descriptors)
      output << ", " << descriptor.name;
    output << "):\n";
    indent = 1;
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (parameter.getParameter().getRole() ==
          static_cast<uint32_t>(gpu::ParameterRole::ProviderWarps))
        line(parameter.getParameter().getName().getValue().str() +
             ": tl.constexpr = tl.extra.cuda.num_warps()");
    });
    emitBlock(kernel.getBody().front(), /*isLoop=*/false, {});
    output << "\n";
  }

  void emitViewChecks(ArrayRef<ViewABI> arguments) {
    std::map<int64_t, std::pair<std::string, std::string>> dimensions;
    for (const ViewABI &view : arguments) {
      if (view.workspace)
        continue;
      std::string name = kernel.getArgAttrDict(view.argument)
                             .getAs<StringAttr>(gpu::abiNameAttr)
                             .getValue().str();
      std::string dtype = pythonType(view.type.getElementType(), true);
      line("if not isinstance(" + view.name + ", torch.Tensor):", 1);
      line("raise TypeError(\"" + name + " must be a torch.Tensor\")", 2);
      line("if " + view.name + ".dtype != " + dtype + ":", 1);
      line("raise TypeError(f\"" + name + " must have dtype " + dtype +
               ", got {" + view.name + ".dtype}\")", 2);
      std::string rank = std::to_string(view.type.getRank());
      line("if " + view.name + ".ndim != " + rank + ":", 1);
      line("raise ValueError(f\"" + name + " must have rank " + rank +
               ", got {" + view.name + ".ndim}\")", 2);
      auto layout = view.type.getLayout();
      for (auto [axis, dimension] :
           llvm::enumerate(layout.getDimensionIds().asArrayRef())) {
        std::string suffix = ".shape[" + std::to_string(axis) + "]";
        std::string actual = view.name + suffix;
        std::string label = name + suffix;
        auto extent = cast<gpu::PhysicalExprAttr>(layout.getExtents()[axis]);
        if (extent.getKind() ==
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant)) {
          std::string expected = std::to_string(extent.getValue());
          line("if " + actual + " != " + expected + ":", 1);
          line("raise ValueError(f\"" + label + " must equal " + expected +
                   ", got {" + actual + "}\")", 2);
        }
        if (dimension <= 0)
          continue;
        auto [binding, inserted] = dimensions.try_emplace(
            dimension, std::make_pair(actual, label));
        if (inserted)
          continue;
        const auto &[expected, source] = binding->second;
        line("if " + actual + " != " + expected + ":", 1);
        line("raise ValueError(f\"" + label + " must equal " + source +
                 ", got {" + actual + "} and {" + expected + "}\")", 2);
      }
    }
  }

  void emitLaunch() {
    output << "def launch(" << joinLaunchArguments() << "):\n";
    emitViewChecks(views);
    line("return _intent_launch(" + joinLaunchArguments() + ")", 1);
    output << "\ndef _intent_launch(" << joinLaunchArguments() << "):\n";
    if (descriptorAllocator)
      line("triton.set_allocator(_intent_tensor_descriptor_allocator)", 1);
    for (const MetadataABI &metadata : metadataArguments) {
      const ViewABI &source = viewByABI(metadata.sourceABI);
      if (source.workspace)
        continue;
      line(metadata.name + " = " + source.name +
           (metadata.kind == "dimension" ? ".shape[" : ".stride(") +
           std::to_string(metadata.sourceAxis) +
           (metadata.kind == "dimension" ? "]" : ")"), 1);
    }
    for (const ViewABI &view : views) {
      if (!view.workspace)
        continue;
      auto deviceSource = llvm::find_if(views, [](const ViewABI &candidate) {
        return !candidate.workspace;
      });
      if (deviceSource == views.end()) {
        kernel.emitError("workspace allocation requires a public view device");
        failed = true;
        return;
      }
      std::string shape = "(";
      for (Attribute extent : view.type.getLayout().getExtents())
        shape +=
            expressionString(cast<gpu::PhysicalExprAttr>(extent), false) + ", ";
      shape += ")";
      line(view.name + " = torch.empty(" + shape + ", device=" +
               deviceSource->name + ".device, dtype=" +
               pythonType(view.type.getElementType(), true) + ")", 1);
    }
    for (const MetadataABI &metadata : metadataArguments) {
      const ViewABI &source = viewByABI(metadata.sourceABI);
      if (source.workspace)
        line(metadata.name + " = " + source.name + ".stride(" +
                 std::to_string(metadata.sourceAxis) + ")", 1);
    }
    llvm::DenseMap<Value, std::string> overlapSpans;
    SmallVector<std::string> overlapChecks;
    for (ViewOverlapOp overlap : overlapFacts) {
      for (Value view : overlap.getOperands()) {
        if (overlapSpans.count(view))
          continue;
        std::string name = newName();
        overlapSpans[view] = name;
        line(name + " = " + overlapSpanFunction + "(" + valueString(view) +
                 ")", 1);
      }
      overlapChecks.push_back(overlapFunction + "(" +
                              overlapSpans.lookup(overlap.getLhs()) + ", " +
                              overlapSpans.lookup(overlap.getRhs()) + ")");
    }
    if (!overlapChecks.empty())
      line(overlapArgument + " = " + stringTuple(overlapChecks), 1);
    if (descriptorChoice) {
      std::string eligibility;
      for (const DescriptorABI &descriptor : descriptors) {
        if (!eligibility.empty())
          eligibility += " and ";
        eligibility += descriptorContractCall(descriptor,
                                              /*argumentMap=*/false);
      }
      line(descriptorChoice.getEligibilityArgument().str() + " = (" +
               eligibility + ")",
           1);
    }
    auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
    std::string gridName = newName();
    std::string grid = gridName + " = lambda META: (";
    for (auto [index, extent] : llvm::enumerate(space)) {
      if (index)
        grid += ", ";
      grid += expressionString(cast<gpu::PhysicalExprAttr>(extent), true);
    }
    if (space.size() == 1)
      grid += ",";
    line(grid + ")", 1);
    std::string call = "return _intent_kernel[" + gridName + "](";
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
    if (!overlapFacts.empty())
      call += ", " + overlapArgument;
    if (descriptorChoice)
      call += ", " + descriptorChoice.getEligibilityArgument().str();
    // Ordinary arithmetic permits FMA while preserving subnormals.
    if (!first)
      call += ", ";
    call += "enable_fp_fusion=True, enable_reflect_ftz=False";
    line("with _intent_tuning_hooks:", 1);
    line(call + ")", 2);
    output << "\n";
  }

  void emitRun() {
    SmallVector<ViewABI> inputs;
    SmallVector<ViewABI> outputs;
    for (const ViewABI &view : views) {
      if (view.workspace)
        continue;
      if (view.type.getAccess() != 1)
        inputs.push_back(view);
      if (view.type.getAccess() == 1)
        outputs.push_back(view);
    }
    output << "def run(" << joinLaunchArguments(/*includeOutputs=*/false) << "):\n";
    if (outputs.empty()) {
      line("launch(" + joinLaunchArguments() + ")", 1);
      line("return None", 1);
      return;
    }
    emitViewChecks(inputs);
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
    line("_intent_launch(" + joinLaunchArguments() + ")", 1);
    if (outputs.size() == 1)
      line("return " + outputs.front().name, 1);
    else
      line("return (" + joinViewNames(outputs) + ")", 1);
  }

  void emitBlock(Block &block, bool isLoop,
                 ArrayRef<std::string> loopResults) {
    auto begin = output.tell();
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (isLoop || !loopResults.empty())
          for (auto [name, value] : llvm::zip(loopResults, yield.getOperands()))
            line(name + " = " + controlValueString(value));
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
    if (output.tell() == begin)
      line("pass");
  }

  void emitOperation(Operation &operation) {
    if (auto assertion = dyn_cast<cf::AssertOp>(operation)) {
      if (!constexprValues.contains(assertion.getArg())) {
        assertion.emitOpError("Triton static assertion condition is not constexpr");
        failed = true;
        return;
      }
      std::string message;
      llvm::raw_string_ostream(message) << llvm::json::Value(assertion.getMsg());
      line("tl.static_assert(" + valueString(assertion.getArg()) + ", " + message + ")");
      return;
    }
    if (isa<CtaBarrierOp>(operation)) {
      line("tl.debug_barrier()");
      return;
    }
    if (isa<ViewOverlapOp, TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp>(operation))
      return;
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      std::string value = literal(constant.getValue());
      if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
        value = "tl.full((), " + value + ", " +
                pythonType(constant.getType()) + ")";
        // Triton's scalar constructor canonicalizes both zero signs to +0.
        if (floating.getValue().isNegZero())
          value = "(-" + value + ")";
        assign(constant.getResult(), value);
      } else {
        values[constant.getResult()] = value;
        constexprValues.insert(constant.getResult());
      }
      return;
    }
    if (auto parameter = dyn_cast<gpu::ParameterOp>(operation)) {
      values[parameter.getResult()] =
          parameter.getParameter().getName().getValue().str();
      constexprValues.insert(parameter.getResult());
      return;
    }
    if (auto physical = dyn_cast<gpu::PhysicalExprOp>(operation)) {
      assign(physical.getResult(),
             expressionString(physical.getExpression(), false),
             isConstexprExpression(physical.getExpression()));
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
      assign(dim.getResult(), expressionString(extent, false),
             isConstexprExpression(extent));
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
      assign(binary.getResult(), binaryExpression(binary),
             (binary.getResult().getType().isIndex() ||
              binary.getResult().getType().isInteger(1)) &&
                 constexprValues.contains(binary.getLhs()) &&
                 constexprValues.contains(binary.getRhs()));
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
                                      valueString(compare.getRhs()) + ")",
             compare.getResult().getType().isInteger(1) &&
                 constexprValues.contains(compare.getLhs()) &&
                 constexprValues.contains(compare.getRhs()));
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
                                      ", " + controlValueString(select.getTrueValue()) +
                                      ", " + controlValueString(select.getFalseValue()) + ")");
      return;
    }
    if (auto map = dyn_cast<MapElementwiseOp>(operation)) {
      std::string call = "tl.map_elementwise(" + helperNames[&operation].front();
      for (Value input : map.getInputs())
        call += ", " + controlValueString(input);
      assign(map.getResult(), call + ")");
      return;
    }
    if (auto reshape = dyn_cast<gpu::ReshapeOp>(operation)) {
      auto source = cast<gpu::FragmentType>(reshape.getValue().getType());
      auto target = cast<gpu::FragmentType>(reshape.getResult().getType());
      if (source.getShape().empty()) {
        assign(reshape.getResult(), "tl.broadcast_to(" +
                                         valueString(reshape.getValue()) + ", " +
                                         fragmentShape(target) + ")");
        return;
      }
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
    if (auto split = dyn_cast<SplitOp>(operation)) {
      assignResults(split.getResults(),
                    "tl.split(" + valueString(split.getSource()) + ")");
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
    if (auto load = dyn_cast<DescriptorLoadOp>(operation)) {
      assign(load.getResult(),
             valueString(load.getDescriptor()) + ".load(" +
                 descriptorOffsets(load.getOffsets()) + ")");
      return;
    }
    if (auto load = dyn_cast<BlockLoadOp>(operation)) {
      auto fragment = load.getResult().getType();
      std::string call = "tl.load(" +
                         blockPointer(load.getView(), load.getOffsets(),
                                      load.getBlockAxes(), load.getOrder(),
                                      fragment);
      if (!load.getBoundaryAxes().empty())
        call += ", boundary_check=" + axisTuple(load.getBoundaryAxes()) +
                ", padding_option=\"" + load.getPadding().str() + "\"";
      call += ")";
      if (fragment.getElementType().isInteger(1))
        call = "tl.cast(" + call + ", tl.int1)";
      assign(load.getResult(), call);
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
      Value coordinate = gather.getCoordinates().front();
      std::string indices = valueString(coordinate);
      bool scalar = !isa<gpu::FragmentType>(gather.getResult().getType());
      if (scalar)
        indices = "tl.full((1,), " + indices + ", " +
                  pythonType(coordinate.getType()) + ")";
      std::string call = "tl.gather(" + valueString(gather.getSource()) + ", " +
                         indices +
                         ", axis=" +
                         std::to_string(gather.getSourceAxes().front()) + ")";
      if (scalar)
        call = "tl.reshape(" + call + ", ())";
      assign(gather.getResult(), call);
      return;
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      auto form = contract->getAttrOfType<StringAttr>(
          "intent_gpu.triton.contract_form");
      if (form && form.getValue() == "fma") {
        assign(contract.getResult(),
               "contract_fma(" + valueString(contract.getLhs()) + ", " +
                   valueString(contract.getRhs()) + ", " +
                   valueString(contract.getAccumulator()) + ")");
        return;
      }
      if (form && form.getValue() == "multiply_sum") {
        unsigned rank = contract.getLhs().getType().getShape().size();
        std::string element = pythonType(elementType(contract.getResult().getType()));
        std::string lhs = "tl.expand_dims(" + valueString(contract.getLhs()) +
                          ", axis=" + std::to_string(rank) + ").to(" +
                          element + ")";
        std::string rhs = "tl.expand_dims(" + valueString(contract.getRhs()) +
                          ", axis=" + std::to_string(rank - 2) + ").to(" +
                          element + ")";
        std::string reduced =
            "tl.sum((" + lhs + " * " + rhs + "), axis=" +
            std::to_string(rank - 1) + ")";
        assign(contract.getResult(), "(" + reduced + " + " +
                                         valueString(contract.getAccumulator()) +
                                         ")");
        return;
      }
      assign(contract.getResult(), "tl.dot(" + valueString(contract.getLhs()) +
                                      ", " + valueString(contract.getRhs()) +
                                      ", " + valueString(contract.getAccumulator()) +
                                      ", input_precision=\"ieee\", max_num_imprecise_acc=0, out_dtype=" +
                                      pythonType(contract.getResult()
                                                     .getType()
                                                     .getElementType()) + ")");
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
      auto extent = [](gpu::FragmentType type, unsigned axis) {
        return expressionString(
            cast<gpu::PhysicalExprAttr>(type.getShape()[axis]), false);
      };
      auto lhs = contract.getLhs().getType();
      auto rhs = contract.getRhs().getType();
      // The verified scaled-contract schema already fixes contiguous groups
      // and packed carriers. dot_scaled spells that grouped K as one axis.
      std::string lhsValue =
          "tl.reshape(" + valueString(contract.getLhs()) + ", (" +
          extent(lhs, 0) + ", " + extent(lhs, 1) + " * " + extent(lhs, 2) + "))";
      std::string rhsValue =
          "tl.reshape(" + valueString(contract.getRhs()) + ", (" +
          extent(rhs, 0) + " * " + extent(rhs, 1) + ", " + extent(rhs, 2) + "))";
      assign(contract.getResult(),
             "tl.dot_scaled(" + lhsValue + ", " +
                 valueString(contract.getLhsScale()) + ", \"" +
                 format(contract.getLhsFormat()).str() + "\", " +
                 rhsValue + ", " +
                 valueString(contract.getRhsScale()) + ", \"" +
                 format(contract.getRhsFormat()).str() + "\", " +
                 valueString(contract.getAccumulator()) + ")");
      return;
    }
    if (isa<ReduceOp, ScanOp>(operation)) {
      auto reduce = dyn_cast<ReduceOp>(operation);
      auto scan = dyn_cast<ScanOp>(operation);
      unsigned count = reduce ? reduce.getSourceCount() : scan.getSourceCount();
      int64_t axis = reduce ? reduce.getAxis() : scan.getAxis();
      std::string sources = count == 1 ? valueString(operation.getOperand(0)) : "(";
      if (count != 1) {
        for (unsigned i = 0; i < count; ++i) {
          if (i) sources += ", ";
          sources += valueString(operation.getOperand(i));
        }
        sources += ")";
      }
      std::string call = (reduce ? "tl.reduce(" : "tl.associative_scan(") + sources +
          ", axis=" + std::to_string(axis) + ", combine_fn=" + helperNames.lookup(&operation).front();
      if (scan) call += std::string(", reverse=") + (scan.getReverse() ? "True" : "False");
      assignResults(operation.getResults(), call + ")");
      return;
    }
    if (auto histogram = dyn_cast<gpu::HistogramOp>(operation)) {
      std::string counts = "tl.histogram(" +
                           valueString(histogram.getValues()) + ", " +
                           valueString(histogram.getBins()) + ", mask=" +
                           valueString(histogram.getValid()) + ")";
      Type resultElement = histogram.getResult().getType().getElementType();
      assign(histogram.getResult(),
             "tl.cast(" + counts + ", " + pythonType(resultElement) + ")");
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
    if (auto store = dyn_cast<DescriptorStoreOp>(operation)) {
      auto fragment = cast<gpu::FragmentType>(store.getValue().getType());
      line(valueString(store.getDescriptor()) + ".store(" +
           descriptorOffsets(store.getOffsets()) +
           ", tl.cast(" + valueString(store.getValue()) + ", " +
           pythonType(fragment.getElementType()) + "))");
      return;
    }
    if (auto store = dyn_cast<BlockStoreOp>(operation)) {
      auto fragment = cast<gpu::FragmentType>(store.getValue().getType());
      std::string storageType = fragment.getElementType().isInteger(1)
                                    ? "tl.int8"
                                    : pythonType(fragment.getElementType());
      std::string call =
          "tl.store(" +
          blockPointer(store.getView(), store.getOffsets(),
                       store.getBlockAxes(), store.getOrder(), fragment) +
          ", tl.cast(" + valueString(store.getValue()) + ", " +
          storageType + ")";
      if (!store.getBoundaryAxes().empty())
        call += ", boundary_check=" + axisTuple(store.getBoundaryAxes());
      line(call + ")");
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
        line(name + " = " + controlValueString(initial));
        results.push_back(name);
      }
      std::string induction = "iv" + std::to_string(counter++);
      values[loop.getInductionVar()] = induction;
      for (auto [argument, name] : llvm::zip(loop.getRegionIterArgs(), results))
        values[argument] = name;
      auto unroll = loop->getAttrOfType<IntegerAttr>(
          "intent_gpu.triton.loop_unroll_factor");
      std::string range = unroll ? "tl.range(" : "range(";
      range += valueString(loop.getLowerBound()) + ", " +
               valueString(loop.getUpperBound()) + ", " +
               valueString(loop.getStep());
      if (unroll)
        range += ", loop_unroll_factor=" + std::to_string(unroll.getInt());
      line("for " + induction + " in " + range + "):");
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
        line(name + " = " + controlValueString(initial));
        carries.push_back(name);
      }
      Block &before = whileOperation.getBefore().front();
      for (auto [argument, name] : llvm::zip(before.getArguments(), carries))
        values[argument] = name;
      std::string active = newName();
      line(active + " = tl.full((), True, tl.int1)");
      line("while " + active + ":");
      ++indent;
      for (Operation &nested : before.without_terminator())
        emitOperation(nested);
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      line(active + " = " + valueString(condition.getCondition()));
      line("if " + active + ":");
      ++indent;
      Block &after = whileOperation.getAfter().front();
      for (auto [argument, forwarded] :
           llvm::zip(after.getArguments(), condition.getArgs()))
        values[argument] = controlValueString(forwarded);
      emitBlock(after, true, carries);
      --indent;
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
    if (binary.getResult().getType().isIndex() &&
        constexprValues.contains(binary.getLhs()) &&
        constexprValues.contains(binary.getRhs())) {
      if (binary.getOperatorKind() == BinaryOperator::Maximum)
        return call("max");
      if (binary.getOperatorKind() == BinaryOperator::Minimum)
        return call("min");
    }
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: return infix("+");
    case BinaryOperator::Subtract: return infix("-");
    case BinaryOperator::Multiply: return infix("*");
    case BinaryOperator::TrueDivide: {
      if (binary.getApproximate())
        return "tl.inline_asm_elementwise(\"div.approx" +
               std::string(binary.getFlushToZero() ? ".ftz" : "") +
               ".f32 $0, $1, $2;\", constraints=\"=f,f,f\", args=[" +
               valueString(binary.getLhs()) + ", " + valueString(binary.getRhs()) +
               "], dtype=tl.float32, is_pure=True, pack=1)";
      Type element = elementType(binary.getResult().getType());
      auto floating = cast<FloatType>(element);
      std::string computation = floating.getWidth() < 32
                                    ? "tl.float32" : pythonType(element);
      std::string result =
          "libdevice.div_rn(tl.cast(" + valueString(binary.getLhs()) + ", " +
          computation + "), tl.cast(" + valueString(binary.getRhs()) + ", " +
          computation + "))";
      return floating.getWidth() < 32
                 ? "tl.cast(" + result + ", " + pythonType(element) + ")"
                 : result;
    }
    case BinaryOperator::FloorDivide:
    case BinaryOperator::Remainder: {
      Type element = elementType(binary.getResult().getType());
      auto integer = dyn_cast<IntegerType>(element);
      bool remainder = binary.getOperatorKind() == BinaryOperator::Remainder;
      if (integer && integer.isUnsigned())
        return infix(remainder ? "%" : "//");
      std::string rem = infix("%");
      std::string rhs = valueString(binary.getRhs());
      // Triton integer division truncates. A Python constexpr remainder already
      // has the divisor's sign, so this correction also preserves constexprs.
      std::string adjust = "((" + rem + " != 0) & ((" + rem +
                           " < 0) != (" + rhs + " < 0)))";
      return remainder ? "(" + rem + " + " + adjust + " * " + rhs + ")"
                       : "(" + infix("//") + " - " + adjust + ")";
    }
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
    auto libraryCall = [&](StringRef function) {
      Type element = elementType(unary.getResult().getType());
      auto floating = cast<FloatType>(element);
      std::string computation = floating.getWidth() < 32
                                    ? "tl.float32" : pythonType(element);
      std::string result = function.str() + "(tl.cast(" + input + ", " +
                           computation + "))";
      return floating.getWidth() < 32
                 ? "tl.cast(" + result + ", " + pythonType(element) + ")"
                 : result;
    };
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate:
      return "(-" + input + ")";
    case UnaryOperator::Not:
      return "(~" + input + ")";
    case UnaryOperator::Exp:
      return libraryCall("libdevice.exp");
    case UnaryOperator::Exp2:
      if (unary.getApproximate())
        return "tl.inline_asm_elementwise(\"ex2.approx" +
               std::string(unary.getFlushToZero() ? ".ftz" : "") +
               ".f32 $0, $1;\", constraints=\"=f,f\", args=[" + input +
               "], dtype=tl.float32, is_pure=True, pack=1)";
      return libraryCall("libdevice.exp2");
    case UnaryOperator::Log:
      return libraryCall("libdevice.log");
    case UnaryOperator::Log1p:
      return libraryCall("libdevice.log1p");
    case UnaryOperator::Lgamma:
      return libraryCall("libdevice.lgamma");
    case UnaryOperator::Sin:
      return libraryCall("libdevice.sin");
    case UnaryOperator::Cos:
      return libraryCall("libdevice.cos");
    case UnaryOperator::Floor:
      return "tl.floor(" + input + ")";
    case UnaryOperator::Erf:
      return libraryCall("libdevice.erf");
    case UnaryOperator::Erfc:
      return libraryCall("libdevice.erfc");
    case UnaryOperator::I0:
      return libraryCall("libdevice.cyl_bessel_i0");
    case UnaryOperator::Rsqrt:
      return libraryCall("libdevice.rsqrt");
    case UnaryOperator::Sigmoid:
      return "tl.sigmoid(" + input + ")";
    case UnaryOperator::Tanh:
      if (unary.getApproximate())
        return "tl.inline_asm_elementwise(\"tanh.approx.f32 $0, $1;\", "
               "constraints=\"=f,f\", args=[" + input +
               "], dtype=tl.float32, is_pure=True, pack=1)";
      return libraryCall("libdevice.tanh");
    case UnaryOperator::Abs:
      return "tl.abs(" + input + ")";
    case UnaryOperator::Sqrt:
      return libraryCall("libdevice.sqrt_rn");
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

  std::string broadcastValue(Value value, gpu::FragmentType target,
                             std::optional<unsigned> coordinateAxis = std::nullopt) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return "tl.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", " + pythonType(elementType(value.getType())) + ")";
    if (source == target)
      return valueString(value);
    gpu::BroadcastProjection projection;
    if (coordinateAxis) {
      projection.state = gpu::BroadcastProjectionState::Exact;
      projection.targetToSource.resize(target.getShape().size());
      projection.targetToSource[*coordinateAxis] = 0;
    } else {
      projection = gpu::queryBroadcastProjection(source, target);
    }
    if (!projection.isExact()) {
      kernel.emitError("Triton broadcast lost its shared axis projection")
          << "; source=" << source << "; target=" << target
          << "; value=" << value;
      failed = true;
      return {};
    }
    if (source.getShape().size() == target.getShape().size())
      return "tl.broadcast_to(" + valueString(value) + ", " +
             fragmentShape(target) + ")";
    SmallVector<std::string> selectors;
    for (std::optional<unsigned> sourceIndex : projection.targetToSource)
      selectors.push_back(sourceIndex ? ":" : "None");
    std::string result = valueString(value) + "[";
    for (auto [index, selector] : llvm::enumerate(selectors)) {
      if (index)
        result += ", ";
      result += selector;
    }
    return "tl.broadcast_to(" + result + "], " + fragmentShape(target) + ")";
  }

  std::string axisTuple(ArrayRef<int64_t> axes) const {
    std::string result = "(";
    for (auto [index, axis] : llvm::enumerate(axes)) {
      if (index)
        result += ", ";
      result += std::to_string(axis);
    }
    if (axes.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string stringTuple(ArrayRef<std::string> values) const {
    std::string result = "(";
    for (auto [index, value] : llvm::enumerate(values)) {
      if (index)
        result += ", ";
      result += value;
    }
    if (values.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string stringList(ArrayRef<std::string> values) const {
    std::string result = "[";
    for (auto [index, value] : llvm::enumerate(values)) {
      if (index)
        result += ", ";
      result += value;
    }
    return result + "]";
  }

  std::string descriptorOffsets(ValueRange offsets) {
    SmallVector<std::string> expressions;
    for (Value offset : offsets)
      expressions.push_back("tl.cast(" + valueString(offset) + ", tl.int32)");
    return stringList(expressions);
  }

  std::string descriptorArgumentExpression(
      gpu::PhysicalExprAttr expression) const {
    auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
    if (kind == gpu::PhysicalExprKind::Constant)
      return std::to_string(expression.getValue());
    if (kind == gpu::PhysicalExprKind::Parameter) {
      std::string name = expression.getSymbol().getValue().str();
      if (fullCoverageParameters.count(name))
        return "_intent_cover_" + name + "(args)";
      return "args[\"" + name + "\"]";
    }
    if (kind == gpu::PhysicalExprKind::Dimension ||
        kind == gpu::PhysicalExprKind::ScalarABI)
      return ("args[\"" + expression.getSymbol().getValue() + "\"]").str();
    SmallVector<std::string> operands;
    for (Attribute operand : expression.getOperands())
      operands.push_back(descriptorArgumentExpression(
          cast<gpu::PhysicalExprAttr>(operand)));
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

  std::string descriptorHostValue(Value value, bool argumentMap) {
    if (auto dimension = value.getDefiningOp<gpu::DimOp>()) {
      auto view = cast<gpu::ViewType>(dimension.getView().getType());
      auto expression = cast<gpu::PhysicalExprAttr>(
          view.getLayout().getExtents()[dimension.getAxis()]);
      return argumentMap ? descriptorArgumentExpression(expression)
                         : expressionString(expression, false);
    }
    if (auto physical = value.getDefiningOp<gpu::PhysicalExprOp>())
      return argumentMap
                 ? descriptorArgumentExpression(physical.getExpression())
                 : expressionString(physical.getExpression(), false);
    if (auto constant = value.getDefiningOp<arith::ConstantOp>())
      return literal(constant.getValue());
    if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
      std::string lhs = descriptorHostValue(binary.getLhs(), argumentMap);
      std::string rhs = descriptorHostValue(binary.getRhs(), argumentMap);
      auto infix = [&](StringRef spelling) {
        return "(" + lhs + " " + spelling.str() + " " + rhs + ")";
      };
      switch (binary.getOperatorKind()) {
      case BinaryOperator::Add:
        return infix("+");
      case BinaryOperator::Subtract:
        return infix("-");
      case BinaryOperator::Multiply:
        return infix("*");
      case BinaryOperator::FloorDivide:
        return infix("//");
      case BinaryOperator::Maximum:
        return "max(" + lhs + ", " + rhs + ")";
      case BinaryOperator::Minimum:
        return "min(" + lhs + ", " + rhs + ")";
      default:
        break;
      }
    }
    if (isa<BlockArgument>(value))
      return argumentMap ? "args[\"" + valueString(value) + "\"]"
                         : valueString(value);
    failed = true;
    return "<unsupported-descriptor-launch-value>";
  }

  std::string descriptorContractCall(const DescriptorABI &descriptor,
                                     bool argumentMap) {
    TensorDescriptorOp operation = descriptor.operation;
    SmallVector<std::string> shape;
    SmallVector<std::string> strides;
    for (Value extent : operation.getShape())
      shape.push_back(descriptorHostValue(extent, argumentMap));
    for (Value stride : operation.getStrides())
      strides.push_back(descriptorHostValue(stride, argumentMap));
    std::string base = valueString(operation.getBase());
    if (argumentMap)
      base = "args[\"" + base + "\"]";
    auto view = cast<gpu::ViewType>(operation.getBase().getType());
    return "_intent_tensor_descriptor_legal(" + base + ", " +
           stringList(shape) + ", " + stringList(strides) + ", " +
           std::to_string(view.getRank()) + ", " +
           axisTuple(operation.getAlignedStrideAxes()) + ", " +
           axisTuple(operation.getUnitStrideAxes()) + ", " +
           (operation.getRequirePositiveShape() ? "True" : "False") + ", " +
           (operation.getRequirePositiveStrides() ? "True" : "False") +
           ", " + std::to_string(operation.getAlignment()) + ", " +
           std::to_string(operation.getMaximumShapeExtent()) + ")";
  }

  std::string descriptorBlockShape(const DescriptorABI &descriptor) const {
    SmallVector<std::string> shape;
    TensorDescriptorOp operation = descriptor.operation;
    for (Value extent : operation.getBlockShape()) {
      auto physical = extent.getDefiningOp<gpu::PhysicalExprOp>();
      if (!physical)
        return {};
      shape.push_back(
          descriptorArgumentExpression(physical.getExpression()));
    }
    return stringList(shape);
  }

  std::string blockPointer(Value viewValue, ValueRange offsets,
                           ArrayRef<int64_t> blockAxes,
                           ArrayRef<int64_t> order,
                           gpu::FragmentType fragment) {
    auto view = cast<gpu::ViewType>(viewValue.getType());
    auto strides = view.getLayout().getStrides();
    auto strideString = [&](int64_t axis) -> std::string {
      Attribute stride = strides[axis];
      if (auto symbol = dyn_cast<StringAttr>(stride))
        return symbol.getValue().str();
      if (auto integer = dyn_cast<IntegerAttr>(stride))
        return std::to_string(integer.getInt());
      failed = true;
      return {};
    };

    SmallVector<std::string> baseOffsets;
    SmallVector<std::string> offsetLimits(view.getRank());
    for (auto [blockAxis, viewAxis] : llvm::enumerate(blockAxes))
      offsetLimits[viewAxis] = "(2147483648 - " +
          expressionString(cast<gpu::PhysicalExprAttr>(
                               fragment.getShape()[blockAxis]), false) + ")";
    std::string base = valueString(viewValue);
    for (int64_t viewAxis = 0;
         viewAxis < static_cast<int64_t>(view.getRank()); ++viewAxis) {
      std::string offset = "tl.cast(" + valueString(offsets[viewAxis]) +
                           ", tl.int64)";
      // Keep ordinary coordinates in the native i32 offset. Only rebase when
      // the end of the block would exceed that range, preserving a fixed view
      // shape for native loop-bound and memory-pipeline optimization.
      if (!offsetLimits[viewAxis].empty())
        offset = "(tl.maximum(" + offset + ", " + offsetLimits[viewAxis] +
                 ") - " + offsetLimits[viewAxis] + ")";
      baseOffsets.push_back(offset);
      base += " + " + offset + " * " + strideString(viewAxis);
    }

    SmallVector<std::string> shape;
    SmallVector<std::string> blockStrides;
    SmallVector<std::string> offsetExpressions;
    for (int64_t viewAxis : blockAxes) {
      shape.push_back(
          "tl.maximum((" +
          expressionString(cast<gpu::PhysicalExprAttr>(
                               view.getLayout().getExtents()[viewAxis]),
                           false) +
          " - " + baseOffsets[viewAxis] + "), 0)");
      blockStrides.push_back(strideString(viewAxis));
      // A block has at most 2^20 lanes, so an offset below INT32_MIN is
      // entirely padding. Clamping it preserves that fact without wrapping.
      offsetExpressions.push_back(
          "tl.cast(tl.maximum(tl.minimum(tl.cast(" +
          valueString(offsets[viewAxis]) +
          ", tl.int64), " + offsetLimits[viewAxis] +
          "), -2147483648), tl.int32)");
    }
    return "tl.make_block_ptr(base=(" + base + ")" +
           ", shape=" + stringTuple(shape) +
           ", strides=" + stringTuple(blockStrides) +
           ", offsets=" + stringTuple(offsetExpressions) +
           ", block_shape=" + fragmentShape(fragment) +
           ", order=" + axisTuple(order) + ")";
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
    SmallVector<Value> fragmentCoordinates;
    for (Value coordinate : coordinates)
      if (isa<gpu::FragmentType>(coordinate.getType()))
        fragmentCoordinates.push_back(coordinate);
    bool cartesian = fragment &&
        fragmentCoordinates.size() == fragment.getShape().size() &&
        llvm::all_of(fragmentCoordinates, [](Value coordinate) {
          return cast<gpu::FragmentType>(coordinate.getType()).getShape().size() == 1;
        });
    unsigned coordinateSlot = 0;
    std::string result = values.lookup(resource);
    for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
      Attribute strideAttribute = strides[sourceAxes[axis]];
      std::string stride;
      if (auto symbol = dyn_cast<StringAttr>(strideAttribute))
        stride = symbol.getValue().str();
      else if (auto constant = dyn_cast<IntegerAttr>(strideAttribute))
        stride = std::to_string(constant.getInt());
      else {
        failed = true;
        return {};
      }
      std::optional<unsigned> coordinateAxis;
      if (cartesian && isa<gpu::FragmentType>(coordinate.getType())) {
        auto source = cast<gpu::FragmentType>(coordinate.getType());
        auto sourceMap = cast<gpu::AxisMapAttr>(source.getAxisMaps()[0]);
        auto targetMap = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[coordinateSlot]);
        // A Cartesian coordinate slot already defines its result axis, even
        // when two slots carry the same logical source provenance.
        if (sourceMap.getSourceId() == targetMap.getSourceId() &&
            sourceMap.getSourceAxis() == targetMap.getSourceAxis() &&
            sourceMap.getDerived() == targetMap.getDerived() &&
            sourceMap.getDimensionId() == targetMap.getDimensionId() &&
            source.getShape()[0] == fragment.getShape()[coordinateSlot])
          coordinateAxis = coordinateSlot;
        ++coordinateSlot;
      }
      std::string coordinateExpression = fragment
          ? broadcastValue(coordinate, fragment, coordinateAxis)
          : valueString(coordinate);
      result += " + (" + coordinateExpression + ") * " + stride;
    }
    return "(" + result + ")";
  }

  std::string controlValueString(Value value) {
    std::string result = valueString(value);
    if (isa<IntegerType, IndexType, FloatType>(value.getType())) {
      if (value.getDefiningOp<arith::ConstantOp>())
        return "tl.full((), " + result + ", " + pythonType(value.getType()) + ")";
      return "tl.cast(" + result + ", " + pythonType(value.getType()) + ")";
    }
    return result;
  }

  std::string valueString(Value value) {
    auto found = values.find(value);
    if (found == values.end()) {
      failed = true;
      return "<missing>";
    }
    return found->second;
  }

  bool isConstexprExpression(gpu::PhysicalExprAttr expression) {
    if (expression.getKind() ==
        static_cast<uint32_t>(gpu::PhysicalExprKind::ScalarABI))
      return false;
    return llvm::all_of(expression.getOperands(), [&](Attribute operand) {
      return isConstexprExpression(cast<gpu::PhysicalExprAttr>(operand));
    });
  }

  void assign(Value value, const std::string &expression, bool compileTime = false) {
    std::string name = newName();
    values[value] = name;
    if (compileTime)
      constexprValues.insert(value);
    line(name + (compileTime ? ": tl.constexpr = " : " = ") + expression);
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

  std::string newName() {
    std::string name;
    do {
      name = "v" + std::to_string(counter++);
    } while (llvm::any_of(values, [&](const auto &entry) {
      return entry.second == name;
    }));
    return name;
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

  std::string joinLaunchArguments(bool includeOutputs = true) const {
    std::map<unsigned, std::string> arguments;
    for (const ViewABI &view : views)
      if (!view.workspace && (includeOutputs || view.type.getAccess() != 1))
        arguments.emplace(view.argument, view.name);
    for (const ScalarABI &scalar : scalars)
      arguments.emplace(scalar.argument, scalar.name);
    std::string result;
    for (const auto &[index, name] : arguments) {
      if (!result.empty())
        result += ", ";
      result += name;
    }
    return result;
  }

  func::FuncOp kernel;
  raw_ostream &output;
  llvm::DenseMap<Value, std::string> values;
  llvm::DenseSet<Value> constexprValues;
  SmallVector<ViewABI> views;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  SmallVector<ViewOverlapOp> overlapFacts;
  std::string overlapArgument = "_intent_overlaps";
  std::string overlapFunction = "_intent_byte_spans_overlap";
  std::string overlapSpanFunction = "_intent_view_byte_span";
  llvm::StringMap<MetadataABI> metadataByName;
  llvm::DenseMap<int64_t, MetadataABI> dimensionBindings;
  std::map<std::string, CoverageParameter> fullCoverageParameters;
  llvm::DenseMap<Operation *, SmallVector<std::string>> helperNames;
  TensorDescriptorChoiceOp descriptorChoice;
  TensorDescriptorAllocatorOp descriptorAllocator;
  SmallVector<DescriptorABI> descriptors;
  unsigned indent = 0;
  unsigned counter = 0;
  unsigned helperCounter = 0;
  bool emittingHelper = false;
  bool hasConfigPruner = false;
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

#include "Intent/Target/Triton/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
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
    emitDescriptorPruner();
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
  struct DescriptorABI {
    TensorDescriptorOp operation;
    std::string name;
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
      fullCoverageParameters[name] = {
          binding->second.name,
          SmallVector<int64_t>(schema.getCandidates().asArrayRef())};
    });
    kernel.walk([&](TensorDescriptorChoiceOp choice) {
      descriptorChoice = choice;
      values[choice.getResult()] = choice.getConfigParameter().str();
    });
    kernel.walk([&](TensorDescriptorAllocatorOp allocator) {
      descriptorAllocator = allocator;
    });
    kernel.walk([&](TensorDescriptorOp descriptor) {
      std::string name =
          "_intent_descriptor_" + std::to_string(descriptors.size());
      descriptors.push_back({descriptor, name});
      values[descriptor.getResult()] = name;
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
              "from intent.runtime.triton import TuningHooks\n\n";
  }

  void emitDescriptorPruner() {
    if (!descriptorChoice)
      return;
    output << "def _intent_tensor_descriptor_legal(\n"
              "    tensor, shape, strides, source_rank, flattened_contiguous_axes,\n"
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
              "    if any(tensor.stride(axis) != tensor.stride(axis + 1) * tensor.shape[axis + 1] "
              "for axis in flattened_contiguous_axes):\n"
              "        return False\n"
              "    return True\n\n"
              "def _intent_tensor_descriptor_block_shape_legal(\n"
              "    block_shape, element_size, minimum_contiguous_bytes,\n"
              "    require_power_of_two, maximum_block_elements,\n"
              "):\n"
              "    elements = 1\n"
              "    for extent in block_shape:\n"
              "        if extent <= 0 or (require_power_of_two and extent & (extent - 1)):\n"
              "            return False\n"
              "        elements *= extent\n"
              "    return (\n"
              "        elements <= maximum_block_elements\n"
              "        and block_shape[-1] * element_size >= minimum_contiguous_bytes\n"
              "    )\n\n"
              "def _intent_prune_tensor_descriptor_configs(configs, named_args, **kwargs):\n"
              "    if not named_args[\""
           << descriptorChoice.getEligibilityArgument()
           << "\"]:\n"
              "        return [config for config in configs "
              "if not config.kwargs[\""
           << descriptorChoice.getConfigParameter()
           << "\"]]\n"
              "    retained = []\n"
              "    for config in configs:\n"
              "        if not config.kwargs[\""
           << descriptorChoice.getConfigParameter()
           << "\"]:\n"
              "            retained.append(config)\n"
              "            continue\n"
              "        args = dict(named_args)\n"
              "        args.update(config.kwargs)\n"
              "        if _intent_tensor_descriptor_shapes_legal(args):\n"
              "            retained.append(config)\n"
              "    return retained\n\n"
              "def _intent_tensor_descriptor_shapes_legal(args):\n"
              "    descriptors = (\n";
    for (const DescriptorABI &descriptor : descriptors) {
      TensorDescriptorOp operation = descriptor.operation;
      output << "        (" << descriptorBlockShape(descriptor)
             << ", args[\"" << valueString(operation.getBase())
             << "\"].element_size(), "
             << operation.getMinimumContiguousBytes() << ", "
             << (operation.getRequirePowerOfTwoBlockShape() ? "True" : "False")
             << ", " << operation.getMaximumBlockElements() << "),\n";
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
              "    return buffer\n\n"
              "def _intent_host_tensor_descriptor_pre_hook(args):\n";
    if (!descriptors.empty())
      output << "    if not _intent_tensor_descriptor_shapes_legal(args):\n"
                "        return\n"
                "    if not isinstance(args[\""
             << descriptors.front().name
             << "\"], TensorDescriptor):\n"
                "        return\n";
    for (const DescriptorABI &descriptor : descriptors)
      output << "    args[\"" << descriptor.name << "\"].block_shape = "
             << descriptorBlockShape(descriptor) << "\n";
    output << "\n";
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
    output << "_intent_tuning_hooks = TuningHooks((";
    for (const ViewABI &view : views)
      output << "\"" << view.name << "\", ";
    output << "), (";
    for (const ViewABI &view : views)
      output << (view.type.getAccess() != 0 ? "True, " : "False, ");
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
             << ", num_ctas=" << config.ctas;
      if (descriptorChoice)
        output << ", pre_hook=_intent_host_tensor_descriptor_pre_hook";
      output
             << "),\n";
    }
    output << "    ],\n    key=[";
    bool firstKey = true;
    for (const MetadataABI &metadata : metadataArguments) {
      if (!firstKey)
        output << ", ";
      firstKey = false;
      output << "\"" << metadata.name << "\"";
    }
    if (descriptorChoice) {
      if (!firstKey)
        output << ", ";
      output << "\"" << descriptorChoice.getEligibilityArgument() << "\"";
    }
    output << "],\n";
    output << "    pre_hook=_intent_tuning_hooks.before,\n"
              "    post_hook=_intent_tuning_hooks.after,\n";
    if (descriptorChoice)
      output << "    prune_configs_by={\"early_config_prune\": "
                "_intent_prune_tensor_descriptor_configs},\n";
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
    for (const DescriptorABI &descriptor : descriptors) {
      if (!first)
        output << ", ";
      first = false;
      output << descriptor.name;
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
    if (descriptorAllocator)
      line("triton.set_allocator(_intent_tensor_descriptor_allocator)", 1);
    for (const MetadataABI &metadata : metadataArguments) {
      const ViewABI &source = viewByABI(metadata.sourceABI);
      line(metadata.name + " = " + source.name +
           (metadata.kind == "dimension" ? ".shape[" : ".stride(") +
           std::to_string(metadata.sourceAxis) +
           (metadata.kind == "dimension" ? "]" : ")"), 1);
    }
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
    for (const DescriptorABI &descriptor : descriptors) {
      TensorDescriptorOp declaration = descriptor.operation;
      std::string view = valueString(declaration.getBase());
      SmallVector<std::string> shape;
      SmallVector<std::string> strides;
      SmallVector<std::string> initialBlockShape;
      for (Value extent : declaration.getShape())
        shape.push_back(descriptorLaunchValue(extent));
      for (Value stride : declaration.getStrides())
        strides.push_back(descriptorLaunchValue(stride));
      for (int64_t extent : declaration.getInitialBlockShape())
        initialBlockShape.push_back(std::to_string(extent));
      line(descriptor.name + " = (TensorDescriptor(" + view +
               ", shape=" + stringList(shape) + ", strides=" +
               stringList(strides) + ", block_shape=" +
               stringList(initialBlockShape) + ", padding=\"" +
               declaration.getPadding().str() + "\") if " +
               descriptorChoice.getEligibilityArgument().str() + " else " + view +
               ")",
           1);
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
    for (const DescriptorABI &descriptor : descriptors) {
      if (!first)
        call += ", ";
      first = false;
      call += descriptor.name;
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
    if (descriptorChoice)
      call += ", " + descriptorChoice.getEligibilityArgument().str();
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
    if (isa<TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp>(operation))
      return;
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
      std::string call = "tl.gather(" + valueString(gather.getSource()) + ", " +
                         valueString(gather.getCoordinates().front()) +
                         ", axis=" +
                         std::to_string(gather.getSourceAxes().front()) + ")";
      assign(gather.getResult(), call);
      return;
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      auto form = contract->getAttrOfType<StringAttr>(
          "intent_gpu.triton.contract_form");
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
                                      ", input_precision=\"ieee\")");
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
        StringRef primitive;
        if (form.getValue() == "sum")
          primitive = "tl.sum";
        else if (form.getValue() == "max")
          primitive = "tl.max";
        else if (form.getValue() == "min")
          primitive = "tl.min";
        else {
          reduce.emitOpError("has an unknown Triton native reduction form");
          failed = true;
          return;
        }
        call = primitive.str() + "(" + sources + ", axis=" +
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
          "tl.fdiv(tl.cast(" + valueString(binary.getLhs()) + ", " +
          computation + "), tl.cast(" + valueString(binary.getRhs()) + ", " +
          computation + "), ieee_rounding=True)";
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
    case UnaryOperator::Sin:
      return libraryCall("libdevice.sin");
    case UnaryOperator::Cos:
      return libraryCall("libdevice.cos");
    case UnaryOperator::Floor:
      return "tl.floor(" + input + ")";
    case UnaryOperator::Erf:
      return libraryCall("libdevice.erf");
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
      return libraryCall("libdevice.sqrt");
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
      return "tl.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", " + pythonType(elementType(value.getType())) + ")";
    if (source == target)
      return valueString(value);
    gpu::BroadcastProjection projection =
        gpu::queryBroadcastProjection(source, target);
    if (!projection.isExact()) {
      kernel.emitError("Triton broadcast lost its shared axis projection");
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

  std::string descriptorLaunchValue(Value value) {
    return descriptorHostValue(value, /*argumentMap=*/false);
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
           axisTuple(operation.getFlattenedContiguousAxes()) + ", " +
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

    std::string base = valueString(viewValue);
    for (int64_t viewAxis = 0;
         viewAxis < static_cast<int64_t>(view.getRank()); ++viewAxis) {
      base += " + tl.cast(" + valueString(offsets[viewAxis]) +
              ", tl.int64) * " + strideString(viewAxis);
    }

    SmallVector<std::string> shape;
    SmallVector<std::string> blockStrides;
    SmallVector<std::string> offsetExpressions;
    for (int64_t viewAxis : blockAxes) {
      shape.push_back(
          "(" +
          expressionString(cast<gpu::PhysicalExprAttr>(
                               view.getLayout().getExtents()[viewAxis]),
                           false) +
          " - tl.cast(" + valueString(offsets[viewAxis]) + ", tl.int64))");
      blockStrides.push_back(strideString(viewAxis));
      offsetExpressions.push_back("0");
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
  llvm::DenseMap<Operation *, SmallVector<std::string>> helperNames;
  TensorDescriptorChoiceOp descriptorChoice;
  TensorDescriptorAllocatorOp descriptorAllocator;
  SmallVector<DescriptorABI> descriptors;
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

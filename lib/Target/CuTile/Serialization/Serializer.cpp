#include "Intent/Target/CuTile/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

using namespace mlir;

namespace intent::cutile {
namespace {

std::string pythonType(Type type, bool torch = false) {
  if (type.isIndex())
    return torch ? "torch.int64" : "ct.int64";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return torch ? "torch.bool" : "ct.bool_";
    StringRef prefix = integer.isUnsigned() ? "u" : "";
    return ((torch ? "torch." : "ct.") + prefix + "int" +
            Twine(integer.getWidth()))
        .str();
  }
  if (isa<Float16Type>(type))
    return torch ? "torch.float16" : "ct.float16";
  if (isa<BFloat16Type>(type))
    return torch ? "torch.bfloat16" : "ct.bfloat16";
  if (isa<Float32Type>(type))
    return torch ? "torch.float32" : "ct.float32";
  if (isa<Float8E4M3FNType>(type))
    return torch ? "torch.float8_e4m3fn" : "ct.float8_e4m3fn";
  if (isa<Float8E5M2Type>(type))
    return torch ? "torch.float8_e5m2" : "ct.float8_e5m2";
  return {};
}

std::string expressionString(gpu::PhysicalExprAttr expression,
                             bool configContext,
                             const llvm::StringSet<> *fullCoverage = nullptr,
                             StringRef configName = "cfg") {
  auto kind = static_cast<gpu::PhysicalExprKind>(expression.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return std::to_string(expression.getValue());
  if (kind == gpu::PhysicalExprKind::Parameter) {
    StringRef name = expression.getSymbol().getValue();
    return configContext && (!fullCoverage || !fullCoverage->contains(name))
               ? (Twine(configName) + "." + name).str()
               : name.str();
  }
  if (kind == gpu::PhysicalExprKind::Dimension ||
      kind == gpu::PhysicalExprKind::ScalarABI)
    return expression.getSymbol().getValue().str();
  SmallVector<std::string> operands;
  for (Attribute operand : expression.getOperands())
    operands.push_back(expressionString(cast<gpu::PhysicalExprAttr>(operand),
                                        configContext, fullCoverage,
                                        configName));
  if (kind == gpu::PhysicalExprKind::Add)
    return "(" + operands[0] + " + " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Subtract)
    return "(" + operands[0] + " - " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::Multiply)
    return "(" + operands[0] + " * " + operands[1] + ")";
  if (kind == gpu::PhysicalExprKind::CeilDiv)
    return "ct.cdiv(" + operands[0] + ", " + operands[1] + ")";
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
    return "ct.next_power_of_2(" + operands[0] + ")";
  return {};
}

std::string fragmentShape(gpu::FragmentType fragment) {
  std::string result = "(";
  for (auto [index, extent] : llvm::enumerate(fragment.getShape())) {
    if (index)
      result += ", ";
    result += expressionString(cast<gpu::PhysicalExprAttr>(extent), false);
  }
  if (fragment.getShape().size() == 1)
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

struct CoverageParameter {
  std::string dimension;
  SmallVector<int64_t> candidates;
};

bool isCuTileProviderRole(gpu::ParameterRole role) {
  return role == gpu::ParameterRole::ProviderAccessForm ||
         role == gpu::ParameterRole::ProviderOccupancy ||
         role == gpu::ParameterRole::ProviderLoadPolicy ||
         role == gpu::ParameterRole::ProviderWarps ||
         role == gpu::ParameterRole::ProviderCTAs;
}

StringRef providerHint(gpu::ParameterOp parameter) {
  auto role = static_cast<gpu::ParameterRole>(parameter.getParameter().getRole());
  if (role == gpu::ParameterRole::ProviderOccupancy)
    return "occupancy";
  if (role == gpu::ParameterRole::ProviderCTAs)
    return "num_ctas";
  if (role == gpu::ParameterRole::ProviderWarps)
    return "num_worker_warps";
  return {};
}

FailureOr<SmallVector<std::map<std::string, int64_t>>>
parameterConfigs(func::FuncOp kernel) {
  llvm::StringMap<gpu::ParameterOp> parameters;
  WalkResult result = kernel.walk([&](gpu::ParameterOp parameter) {
    auto schema = parameter.getParameter();
    auto role = static_cast<gpu::ParameterRole>(schema.getRole());
    auto category =
        static_cast<gpu::ParameterCategory>(schema.getCategory());
    bool provider = category == gpu::ParameterCategory::Provider;
    if (provider != isCuTileProviderRole(role)) {
      parameter.emitOpError(
          "cuTile source cannot bind a foreign provider parameter role");
      return WalkResult::interrupt();
    }
    StringRef name = schema.getName().getValue();
    if (!parameters.try_emplace(name, parameter).second) {
      parameter.emitOpError("duplicates a cuTile physical parameter");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  auto encoded =
      kernel->getAttrOfType<ArrayAttr>(gpu::cuTileConfigsAttr);
  if (!encoded || encoded.empty())
    return kernel.emitError("cuTile source requires closed provider configs");
  SmallVector<std::map<std::string, int64_t>> configs;
  for (Attribute attribute : encoded) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    size_t boundParameters = llvm::count_if(parameters, [](const auto &entry) {
      return !entry.getValue()->hasAttr(gpu::coverageDimensionAttr);
    });
    if (!tuple || tuple.size() != boundParameters)
      return kernel.emitError("contains a malformed cuTile provider config");
    std::map<std::string, int64_t> config;
    for (NamedAttribute binding : tuple) {
      auto found = parameters.find(binding.getName().getValue());
      auto value = dyn_cast<IntegerAttr>(binding.getValue());
      if (found == parameters.end() || !value ||
          found->second->hasAttr(gpu::coverageDimensionAttr) ||
          !llvm::is_contained(
              found->second.getParameter().getCandidates().asArrayRef(),
              value.getInt()))
        return kernel.emitError(
            "cuTile provider config contains an invalid binding");
      config[binding.getName().strref().str()] = value.getInt();
    }
    configs.push_back(std::move(config));
  }
  return configs;
}

class Serializer {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : kernel(kernel), output(output) {}

  LogicalResult emit() {
    bindArguments();
    emitPreamble();
    emitCollectiveHelpers();
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
    ArrayAttr indexTileBounds;
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
  struct ArrayViewABI {
    ArrayViewOp operation;
    unsigned sourceView;
    std::string name;
    std::string eligible;
    std::string trialName;
    std::string trialEligible;
  };

  void bindArguments() {
    auto indexBounds = kernel->getAttrOfType<ArrayAttr>(arrayIndexTileBoundsAttr);
    for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
      DictionaryAttr attrs = kernel.getArgAttrDict(index);
      std::string kind = attrs.getAs<StringAttr>(gpu::abiKindAttr).getValue().str();
      std::string name = attrs.getAs<StringAttr>(gpu::abiNameAttr).getValue().str();
      values[argument] = name;
      if (kind == "view") {
        views.push_back({static_cast<unsigned>(index), name,
                         cast<gpu::ViewType>(argument.getType()),
                         indexBounds ? cast<ArrayAttr>(indexBounds[index]) : ArrayAttr()});
      } else if (kind == "scalar" || kind == "constexpr" || kind == "value") {
        if (kind == "constexpr") {
          if (!argument.use_empty()) {
            kernel.emitError(
                "live constexpr reached cuTile runtime ABI after specialization");
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
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (StringRef hint = providerHint(parameter); !hint.empty()) {
        if (!providerHintParameters.emplace(
                hint.str(), parameter.getParameter().getName().getValue().str()).second) {
          parameter.emitOpError("duplicates a cuTile compiler hint");
          failed = true;
          return;
        }
        return;
      }
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
      auto schema = parameter.getParameter();
      fullCoverageParameters[schema.getName().getValue().str()] = {
          binding->second.name,
          SmallVector<int64_t>(schema.getCandidates().asArrayRef())};
      fullCoverageParameterNames.insert(schema.getName().getValue());
    });
    llvm::StringSet<> occupied;
    for (const auto &entry : values)
      occupied.insert(entry.second);
    kernel.walk([&](gpu::ParameterOp parameter) {
      occupied.insert(parameter.getParameter().getName().getValue());
    });
    auto fresh = [&](StringRef stem) {
      unsigned suffix = 0;
      std::string name;
      do {
        name = (Twine(stem) + Twine(suffix++)).str();
      } while (!occupied.insert(name).second);
      return name;
    };
    kernel.walk([&](ArrayViewOp array) {
      unsigned argument = cast<BlockArgument>(array.getBase()).getArgNumber();
      for (auto [index, view] : llvm::enumerate(views)) {
        if (view.argument != argument)
          continue;
        ArrayViewABI binding{
            array, static_cast<unsigned>(index), fresh("_intent_array_view_"),
            fresh("_intent_array_valid_"), fresh("_intent_trial_array_view_"),
            fresh("_intent_trial_array_valid_")};
        values[array.getResult()] = binding.name;
        values[array.getEligible()] = binding.eligible;
        arrayViews.push_back(std::move(binding));
      }
    });
  }

  void emitPreamble() {
    output << "from types import SimpleNamespace\n"
              "from typing import Annotated\n"
              "import torch\n"
              "import cuda.tile as ct\n"
              "from cuda.tile.tune import exhaustive_search\n"
              "from intent.runtime.artifact import ParameterRole, TuningConfiguration, TuningParameter\n"
              "from intent.runtime.cutile import array_index_kernels, bind_array_view, can_use_i32_array_indices\n"
              "from intent.runtime.tuning import TuningState\n\n"
              "ConstInt = ct.Constant[int]\n\n";
  }

  Attribute scalarConstant(Value value) {
    while (true) {
      if (auto cast = value.getDefiningOp<gpu::CastOp>()) {
        if (cast.getValue().getType() != cast.getResult().getType())
          return {};
        value = cast.getValue();
        continue;
      }
      if (auto splat = value.getDefiningOp<gpu::SplatOp>()) {
        value = splat.getValue();
        continue;
      }
      if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>()) {
        value = broadcast.getValue();
        continue;
      }
      if (auto extract = value.getDefiningOp<gpu::ExtractOp>()) {
        auto record = extract.getRecord().getDefiningOp<gpu::MakeRecordOp>();
        if (record && extract.getField() < record.getFields().size()) {
          value = record.getFields()[extract.getField()];
          continue;
        }
      }
      break;
    }
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    return constant ? constant.getValue() : Attribute();
  }

  void emitCollectiveHelpers() {
    SmallVector<Operation *> collectives;
    kernel.walk([&](Operation *operation) {
      if (isa<gpu::ReduceOp, gpu::ScanOp>(operation))
        collectives.push_back(operation);
    });
    for (Operation *collective : collectives) {
      std::string helper = "_intent_combine_" +
                           std::to_string(collectiveHelpers.size());
      collectiveHelpers[collective] = helper;
      output << "@ct.function\ndef " << helper << "(";
      Block &block = collective->getRegion(0).front();
      for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
        if (index)
          output << ", ";
        std::string name = "arg" + std::to_string(index);
        values[argument] = name;
        output << name;
      }
      output << "):\n";
      indent = 1;
      for (Operation &operation : block) {
        if (auto yield = dyn_cast<gpu::YieldOp>(operation)) {
          if (yield.getValues().size() == 1)
            line("return " + valueString(yield.getValues().front()));
          else
            line("return " + tuple(yield.getValues()));
          continue;
        }
        emitOperation(operation);
      }
      output << "\n";
      indent = 0;
    }
  }

  void emitKernel() {
    if (!kernel->hasAttr(arrayIndexTileBoundsAttr))
      output << "@ct.kernel\n";
    output << "def _intent_kernel(";
    bool first = true;
    auto argument = [&](StringRef text) {
      if (!first)
        output << ", ";
      first = false;
      output << text;
    };
    auto arrayArgument = [&](StringRef name, unsigned rank) {
      // Shapes already specialize the metadata ABI and launch cache. Preserve
      // the same facts in the provider array type used for access lowering.
      std::string annotation = name.str() +
          ": Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=(";
      for (unsigned axis = 0; axis < rank; ++axis)
        annotation += std::to_string(axis) + ", ";
      argument(annotation + "))]");
    };
    for (const ViewABI &view : views)
      arrayArgument(view.name, view.type.getLayout().getExtents().size());
    for (ArrayViewABI view : arrayViews) {
      arrayArgument(view.name, view.operation.getGroupEnds().size());
      argument(view.eligible + ": ct.Constant[bool]");
    }
    for (const ScalarABI &scalar : scalars) {
      auto integer = dyn_cast<IntegerType>(scalar.type);
      bool wide = scalar.type.isIndex() ||
                  (integer && integer.getWidth() == 64);
      argument(scalar.name + (wide ? ": ct.ScalarInt64" : ""));
    }
    for (const MetadataABI &metadata : metadataArguments)
      argument(metadata.name + ": ConstInt");
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (!providerHint(parameter).empty())
        return;
      std::string name = parameter.getParameter().getName().getValue().str();
      values[parameter.getResult()] = name;
      argument(name + ": ConstInt");
    });
    output << "):\n";
    indent = 1;
    emitBlock(kernel.getBody().front(), false, {});
    output << "\n";
    if (kernel->hasAttr(arrayIndexTileBoundsAttr)) {
      output << "_intent_i32_kernel, _intent_kernel = array_index_kernels(_intent_kernel, (";
      for (const ViewABI &view : views) {
        llvm::json::OStream(output).value(view.name);
        output << ", ";
      }
      for (const ArrayViewABI &view : arrayViews) {
        llvm::json::OStream(output).value(view.name);
        output << ", ";
      }
      output << "))\n\n";
    }
  }

  void emitArgumentBindings() {
    for (const MetadataABI &metadata : metadataArguments) {
      const ViewABI &source = viewByABI(metadata.sourceABI);
      line(metadata.name + " = " + source.name +
               (metadata.kind == "dimension" ? ".shape[" : ".stride(") +
               std::to_string(metadata.sourceAxis) +
               (metadata.kind == "dimension" ? "]" : ")"),
           1);
    }
    for (const auto &[parameter, coverage] : fullCoverageParameters) {
      std::string candidates = "(";
      for (int64_t candidate : coverage.candidates)
        candidates += std::to_string(candidate) + ", ";
      candidates += ")";
      line(parameter + " = next((extent for extent in " + candidates +
               " if extent >= " + coverage.dimension + "), None)",
           1);
      line("if " + parameter + " is None:", 1);
      line("raise ValueError(\"no legal full-coverage extent for " + parameter +
               "\")",
           2);
    }
  }

  void emitArrayBindings(StringRef trialState, unsigned level) {
    for (ArrayViewABI &view : arrayViews) {
      bool trial = !trialState.empty();
      std::string source = trial
          ? trialState.str() + ".views[" + std::to_string(view.sourceView) + "]"
          : views[view.sourceView].name;
      std::string groups = "(";
      for (int64_t end : view.operation.getGroupEnds())
        groups += std::to_string(end) + ", ";
      line((trial ? view.trialName : view.name) + ", " +
               (trial ? view.trialEligible : view.eligible) +
               " = bind_array_view(" + source + ", " + groups + "))", level);
    }
  }

  void emitTuningConfigurations() {
    output << "_TUNING_PARAMETERS = (\n";
    std::string bindings = "(";
    kernel.walk([&](gpu::ParameterOp parameter) {
      auto schema = parameter.getParameter();
      gpu::PhysicalParameterBinding binding = gpu::queryParameterBinding(parameter);
      if (binding.state == gpu::PhysicalFactState::Ambiguous) {
        parameter.emitOpError("has contradictory physical parameter bindings");
        failed = true;
        return;
      }
      output << "    TuningParameter(";
      llvm::json::OStream(output).value(schema.getName().getValue());
      output << ", ParameterRole(" << schema.getRole() << "), "
             << schema.getCategory() << ", (";
      for (int64_t candidate : schema.getCandidates().asArrayRef())
        output << candidate << ", ";
      output << "), ";
      if (binding.dimension)
        output << *binding.dimension;
      else
        output << "None";
      output << ", ";
      if (binding.source)
        output << "(" << binding.source->sourceId << ", "
               << binding.source->sourceAxis << ", "
               << (binding.source->derived ? "True" : "False") << ")";
      else
        output << "None";
      output << ", ";
      auto metadata = binding.dimension ? dimensionBindings.find(*binding.dimension)
                                        : dimensionBindings.end();
      if (metadata == dimensionBindings.end()) {
        output << "None";
      } else {
        for (auto [index, view] : llvm::enumerate(views))
          if (view.argument == metadata->second.sourceABI) {
            unsigned runtimeIndex = index + llvm::count_if(scalars, [&](const ScalarABI &scalar) {
              return scalar.argument < view.argument;
            });
            output << "(" << runtimeIndex << ", " << metadata->second.sourceAxis << ")";
          }
      }
      output << "),\n";
      std::string name = schema.getName().getValue().str();
      bindings += fullCoverageParameterNames.contains(name)
                      ? name + ", "
                      : "_intent_config." + name + ", ";
    });
    output << ")\n\ndef tuning_configurations(" << joinLaunchArguments() << "):\n";
    emitArgumentBindings();
    line("return tuple(TuningConfiguration(_TUNING_PARAMETERS, " + bindings +
             ")) for _intent_config in _CONFIGS)",
         1);
    output << "\n";
  }

  void emitLaunch() {
    FailureOr<SmallVector<std::map<std::string, int64_t>>> configs =
        parameterConfigs(kernel);
    if (mlir::failed(configs)) {
      failed = true;
      return;
    }
    output << "_CONFIGS = (\n";
    for (const auto &config : *configs) {
      output << "    SimpleNamespace(";
      for (auto [index, item] : llvm::enumerate(config)) {
        if (index)
          output << ", ";
        output << item.first << "=" << item.second;
      }
      output << "),\n";
    }
    output << ")\n_TUNE_CACHE = {}\n\n";
    emitTuningConfigurations();
    output << "def launch(" << joinLaunchArguments() << "):\n";
    emitArgumentBindings();
    emitArrayBindings("", 1);

    llvm::StringSet<> occupiedNames;
    for (const ViewABI &view : views)
      occupiedNames.insert(view.name);
    for (const ScalarABI &scalar : scalars)
      occupiedNames.insert(scalar.name);
    for (const MetadataABI &metadata : metadataArguments)
      occupiedNames.insert(metadata.name);
    for (const ArrayViewABI &view : arrayViews) {
      occupiedNames.insert(view.name);
      occupiedNames.insert(view.eligible);
      occupiedNames.insert(view.trialName);
      occupiedNames.insert(view.trialEligible);
    }
    kernel.walk([&](gpu::ParameterOp parameter) {
      occupiedNames.insert(parameter.getParameter().getName().getValue());
    });
    auto freshName = [&](StringRef stem) {
      std::string candidate = stem.str();
      unsigned suffix = 0;
      while (!occupiedNames.insert(candidate).second)
        candidate = (Twine(stem) + "_" + Twine(++suffix)).str();
      return candidate;
    };
    std::string tuneKeyName = freshName("_intent_tune_key");
    std::string streamName = freshName("_intent_stream");
    std::string searchResultName = freshName("_intent_search_result");
    std::string configName = freshName("_intent_cfg");
    std::string tunedKernelName = freshName("_intent_tuned_kernel");
    std::string trialStateName = freshName("_intent_trial_state");
    std::string selectedKernelName = freshName("_intent_selected_kernel");
    std::string boundGridName = freshName("_intent_bound_grid");
    std::string boundArgumentsName = freshName("_intent_bound_arguments");
    std::string boundLaunchName = freshName("_intent_bound_launch");

    std::string key = tuneKeyName + " = (";
    for (const ViewABI &view : views)
      key += "tuple(" + view.name + ".shape), tuple(" + view.name + ".stride()), " + view.name + ".dtype, str(" +
             view.name + ".device), ";
    for (const ScalarABI &scalar : scalars)
      key += scalar.name + ", ";
    line(key + ")", 1);
    line(streamName + " = torch.cuda.current_stream()", 1);
    line("if " + tuneKeyName + " not in _TUNE_CACHE:", 1);
    if (kernel->hasAttr(arrayIndexTileBoundsAttr)) {
      std::string boundArguments = "(";
      std::string viewArguments = "(";
      for (const ViewABI &view : views) {
        viewArguments += view.name + ", ";
        boundArguments += "(";
        for (Attribute bound : view.indexTileBounds)
          boundArguments += expressionString(cast<gpu::PhysicalExprAttr>(bound), false) + ", ";
        boundArguments += "), ";
      }
      line(selectedKernelName + " = _intent_i32_kernel if can_use_i32_array_indices(" +
               viewArguments + "), " + boundArguments + ")) else _intent_kernel", 2);
    } else {
      line(selectedKernelName + " = _intent_kernel", 2);
    }
    std::string trialState = trialStateName + " = TuningState((";
    for (const ViewABI &view : views)
      trialState += view.name + ", ";
    trialState += "), (";
    for (const ViewABI &view : views)
      trialState += view.type.getAccess() != 0 ? "True, " : "False, ";
    line(trialState + "))", 2);
    emitArrayBindings(trialStateName, 2);
    auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
    std::string grid = "lambda " + configName + ": (";
    for (Attribute extent : space)
      grid += expressionString(cast<gpu::PhysicalExprAttr>(extent), true,
                               &fullCoverageParameterNames, configName) +
              ", ";
    grid += "1, 1)";
    std::string hints;
    if (!providerHintParameters.empty())
      hints = ", lambda " + configName + ": " + compilerHints(configName);
    line(searchResultName + " = exhaustive_search(_CONFIGS, " + streamName +
             ", " + grid + ", " + selectedKernelName + ", lambda " + configName +
             ": " + trialStateName + ".arguments((" + joinKernelArguments(configName, true) + "))" + hints +
             ", quiet=True)",
         2);
    std::string tunedKernel = selectedKernelName;
    if (!providerHintParameters.empty())
      tunedKernel += ".replace_hints(**" + compilerHints(searchResultName + ".best.config") + ")";
    line("_TUNE_CACHE[" + tuneKeyName + "] = (" + searchResultName +
             ".best.config, " + tunedKernel + ")",
         2);
    line(configName + ", " + tunedKernelName + " = _TUNE_CACHE[" +
             tuneKeyName + "]",
         1);
    std::string launchGrid = "(";
    for (Attribute extent : space)
      launchGrid += expressionString(cast<gpu::PhysicalExprAttr>(extent), true,
                                     &fullCoverageParameterNames, configName) +
                    ", ";
    launchGrid += "1, 1)";
    line(boundGridName + " = " + launchGrid, 1);
    line(boundArgumentsName + " = (" + joinKernelArguments(configName) + ")", 1);
    line("def " + boundLaunchName + "():", 1);
    line("return ct.launch(torch.cuda.current_stream(), " + boundGridName +
             ", " + tunedKernelName + ", " + boundArgumentsName + ")", 2);
    line(boundLaunchName + "()", 1);
    line("return " + boundLaunchName, 1);
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
    output << "def run(" << joinLaunchArguments(/*includeOutputs=*/false) << "):\n";
    std::string device = inputs.empty() ? "'cuda'" : inputs.front().name + ".device";
    for (const ViewABI &view : outputs) {
      std::string shape = outputShape(view);
      line(view.name + " = torch.empty(" + shape + ", device=" + device +
               ", dtype=" + pythonType(view.type.getElementType(), true) + ")",
           1);
    }
    line("launch(" + joinLaunchArguments() + ")", 1);
    if (outputs.empty())
      line("return None", 1);
    else if (outputs.size() == 1)
      line("return " + outputs.front().name, 1);
    else
      line("return (" + joinViewNames(outputs) + ")", 1);
  }

  void emitBlock(Block &block, bool isLoop,
                 ArrayRef<std::string> loopResults) {
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (isLoop && !loopResults.empty()) {
          std::string names;
          for (auto [index, name] : llvm::enumerate(loopResults)) {
            if (index)
              names += ", ";
            names += name;
          }
          line(names + " = " +
               (yield.getOperands().size() == 1
                    ? valueString(yield.getOperands().front())
                    : tuple(yield.getOperands())));
        }
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
  }

  void emitIfBranch(Block &block, ArrayRef<std::string> resultNames) {
    bool emitted = false;
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (yield.getOperands().size() != resultNames.size()) {
          yield.emitOpError(
              "cuTile if yield/result arity changed after provider legalization");
          failed = true;
          continue;
        }
        if (resultNames.empty())
          continue;
        std::string names;
        for (auto [index, name] : llvm::enumerate(resultNames)) {
          if (index)
            names += ", ";
          names += name;
        }
        line(names + " = " +
             (yield.getOperands().size() == 1
                  ? valueString(yield.getOperands().front())
                  : tuple(yield.getOperands())));
        emitted = true;
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
      emitted = true;
    }
    if (!emitted)
      line("pass");
  }

  void emitOperation(Operation &operation) {
    if (isa<ArrayViewOp>(operation)) {
      return;
    } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      values[constant.getResult()] = literal(constant.getValue());
    } else if (auto parameter = dyn_cast<gpu::ParameterOp>(operation)) {
      values[parameter.getResult()] =
          parameter.getParameter().getName().getValue().str();
    } else if (auto physical = dyn_cast<gpu::PhysicalExprOp>(operation)) {
      assign(physical.getResult(), expressionString(physical.getExpression(), false));
    } else if (auto program = dyn_cast<gpu::ProgramIdOp>(operation)) {
      assign(program.getResult(), "ct.astype(ct.bid(" +
                                      std::to_string(program.getAxis()) + "), " +
                                      pythonType(program.getResult().getType()) + ")");
    } else if (auto coordinate = dyn_cast<gpu::WorksetCoordinateOp>(operation)) {
      values[coordinate.getResult()] = valueString(coordinate.getCoordinate());
    } else if (auto dim = dyn_cast<gpu::DimOp>(operation)) {
      auto extent = cast<gpu::PhysicalExprAttr>(
          dim.getView().getType().getLayout().getExtents()[dim.getAxis()]);
      assign(dim.getResult(), expressionString(extent, false));
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
    } else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) +
                                    " + ct.arange(" + valueString(range.getExtent()) +
                                    ", dtype=" +
                                    pythonType(range.getResult().getType().getElementType()) + ") * " +
                                    valueString(range.getStep()) + ")");
    } else if (auto splat = dyn_cast<gpu::SplatOp>(operation)) {
      auto type = splat.getResult().getType();
      assign(splat.getResult(), "ct.full(" + fragmentShape(type) + ", " +
                                    valueString(splat.getValue()) + ", dtype=" +
                                    pythonType(type.getElementType()) + ")");
    } else if (auto broadcast = dyn_cast<gpu::BroadcastOp>(operation)) {
      assign(broadcast.getResult(), broadcastValue(broadcast.getValue(),
                                                   broadcast.getResult().getType()));
    } else if (auto cast = dyn_cast<gpu::CastOp>(operation)) {
      assign(cast.getResult(), "ct.astype(" + valueString(cast.getValue()) +
                                   ", " + pythonType(elementType(cast.getResult().getType())) +
                                   ")");
    } else if (auto bitcast = dyn_cast<gpu::BitcastOp>(operation)) {
      assign(bitcast.getResult(), "ct.bitcast(" + valueString(bitcast.getValue()) +
                                      ", " +
                                      pythonType(elementType(bitcast.getResult().getType())) +
                                      ")");
    } else if (auto reshape = dyn_cast<gpu::ReshapeOp>(operation)) {
      assign(reshape.getResult(), "ct.reshape(" + valueString(reshape.getValue()) +
                                      ", " +
                                      fragmentShape(mlir::cast<gpu::FragmentType>(
                                          reshape.getResult().getType())) +
                                      ")");
    } else if (auto transpose = dyn_cast<gpu::TransposeOp>(operation)) {
      std::string permutation = "(";
      for (auto [index, axis] : llvm::enumerate(transpose.getPermutation())) {
        if (index)
          permutation += ", ";
        permutation += std::to_string(axis);
      }
      if (transpose.getPermutation().size() == 1)
        permutation += ",";
      permutation += ")";
      assign(transpose.getResult(), "ct.permute(" +
                                        valueString(transpose.getValue()) + ", " +
                                        permutation + ")");
    } else if (auto join = dyn_cast<gpu::JoinOp>(operation)) {
      auto input = join.getLhs().getType();
      std::string expanded = "(";
      for (Attribute extent : input.getShape())
        expanded += expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(extent), false) +
                    ", ";
      expanded += "1)";
      assign(join.getResult(),
             "ct.cat((ct.reshape(" + valueString(join.getLhs()) + ", " +
                 expanded + "), ct.reshape(" + valueString(join.getRhs()) +
                 ", " + expanded + ")), axis=" +
                 std::to_string(join.getAxis()) + ")");
    } else if (auto record = dyn_cast<gpu::MakeRecordOp>(operation)) {
      values[record.getResult()] = tuple(record.getFields());
    } else if (auto extract = dyn_cast<gpu::ExtractOp>(operation)) {
      assign(extract.getResult(), valueString(extract.getRecord()) + "[" +
                                      std::to_string(extract.getField()) + "]");
    } else if (auto select = dyn_cast<gpu::SelectOp>(operation)) {
      assign(select.getResult(), "ct.where(" + valueString(select.getCondition()) +
                                      ", " + valueString(select.getTrueValue()) +
                                      ", " + valueString(select.getFalseValue()) + ")");
    } else if (auto load = dyn_cast<TileLoadOp>(operation)) {
      std::string padding = "ct.PaddingMode.ZERO";
      if (Value fullTiles = load.getFullTiles())
        padding = "(ct.PaddingMode.UNDETERMINED if " + valueString(fullTiles) +
                  " else ct.PaddingMode.ZERO)";
      std::string call = "ct.load(" + valueString(load.getResource()) +
                         ", index=" + tuple(load.getTileIndices()) +
                         ", shape=" + fragmentShape(load.getResult().getType()) +
                         ", padding_mode=" + padding + ", allow_tma=" +
                         valueString(load.getAllowTma());
      if (auto latency = load.getLatencyPolicy())
        call += ", latency=(None if " + valueString(latency) + " == " +
                std::to_string(inferredLoadPolicy) + " else " +
                valueString(latency) + ")";
      call += ")";
      assign(load.getResult(), call);
    } else if (auto load = dyn_cast<ScalarLoadOp>(operation)) {
      std::string call = "ct.gather(" + valueString(load.getResource()) + ", " +
                         tuple(load.getIndices());
      if (load.getValid())
        call += ", mask=" + valueString(load.getValid()) +
                ", padding_value=" + valueString(load.getFill());
      call += load.getInBounds() ? ", check_bounds=False)"
                                 : ", check_bounds=True)";
      assign(load.getResult(), call);
    } else if (auto gather = dyn_cast<GatherLoadOp>(operation)) {
      std::string call = "ct.gather(" + valueString(gather.getResource()) +
                         ", " + tuple(gather.getCoordinates());
      if (gather.getValid())
        call += ", mask=" + valueString(gather.getValid());
      if (gather.getFill())
        call += ", padding_value=" + valueString(gather.getFill());
      if (auto latency = gather.getLatencyPolicy())
        call += ", latency=(None if " + valueString(latency) + " == " +
                std::to_string(inferredLoadPolicy) + " else " +
                valueString(latency) + ")";
      call += gather.getInBounds() ? ", check_bounds=False)"
                                   : ", check_bounds=True)";
      assign(gather.getResult(), call);
    } else if (auto mma = dyn_cast<MMAOp>(operation)) {
      assign(mma.getResult(), "ct.mma(" + valueString(mma.getLhs()) + ", " +
                                   valueString(mma.getRhs()) + ", " +
                                   valueString(mma.getAccumulator()) + ")");
    } else if (auto mma = dyn_cast<ScaledMMAOp>(operation)) {
      auto lhs = mma.getLhs().getType();
      auto rhs = mma.getRhs().getType();
      std::string lhsK =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[1]), false) +
          " * " + expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[2]),
                        false) +
          ")";
      std::string rhsK =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[0]), false) +
          " * " + expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[1]),
                        false) +
          ")";
      std::string lhsShape =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[0]), false) +
          ", " + lhsK + ")";
      std::string rhsShape =
          "(" + rhsK + ", " +
          expressionString(
              mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[2]), false) +
          ")";
      assign(mma.getResult(),
             "ct.mma_scaled(ct.reshape(" + valueString(mma.getLhs()) + ", " +
                 lhsShape + "), ct.bitcast(" + valueString(mma.getLhsScale()) +
                 ", ct.float8_e8m0fnu)" +
                 ", ct.reshape(" + valueString(mma.getRhs()) + ", " + rhsShape +
                 "), ct.bitcast(" + valueString(mma.getRhsScale()) +
                 ", ct.float8_e8m0fnu), " +
                 valueString(mma.getAccumulator()) + ")");
    } else if (isa<gpu::ReduceOp, gpu::ScanOp>(operation)) {
      auto reduce = dyn_cast<gpu::ReduceOp>(operation);
      auto scan = dyn_cast<gpu::ScanOp>(operation);
      unsigned count = reduce ? reduce.getSourceCount() : scan.getSourceCount();
      ValueRange sources = operation.getOperands().take_front(count);
      ValueRange identities = operation.getOperands().slice(count, count);
      std::string source = count == 1 ? valueString(sources.front())
                                      : tuple(sources);
      std::string identity;
      if (count == 1) {
        identity = literal(scalarConstant(identities.front()));
      } else {
        identity = "(";
        for (auto [index, value] : llvm::enumerate(identities)) {
          if (index)
            identity += ", ";
          identity += literal(scalarConstant(value));
        }
        identity += ")";
      }
      std::string call = std::string(reduce ? "ct.reduce(" : "ct.scan(") +
                         source + ", axis=" +
                         std::to_string(reduce ? reduce.getAxes().front()
                                               : scan.getAxis()) +
                         ", func=" +
                         collectiveHelpers.lookup(&operation) +
                         ", identity=" + identity;
      if (scan)
        call += std::string(", reverse=") + (scan.getReverse() ? "True" : "False");
      call += ")";
      if (count == 1) {
        assign(operation.getResult(0), call);
      } else {
        std::string resultNames;
        for (auto [index, result] : llvm::enumerate(operation.getResults())) {
          if (index)
            resultNames += ", ";
          std::string name = newName();
          values[result] = name;
          resultNames += name;
        }
        line(resultNames + " = " + call);
      }
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      auto nativeReduction = [](BinaryOperator kind) -> StringRef {
        switch (kind) {
        case BinaryOperator::Add: return "ct.sum";
        case BinaryOperator::MaximumNum:
        case BinaryOperator::LogicalOr: return "ct.max";
        case BinaryOperator::MinimumNum:
        case BinaryOperator::LogicalAnd: return "ct.min";
        default: llvm_unreachable("unverified cuTile native reduction kind");
        }
      };
      std::string expression =
          nativeReduction(reduce.getKind()).str() + "(" +
          valueString(reduce.getSource()) + ", axis=" +
          std::to_string(reduce.getAxis()) + ")";
      if (elementType(reduce.getResult().getType()).isInteger(1))
        expression = "ct.astype(" + expression + ", ct.bool_)";
      assign(reduce.getResult(), expression);
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      assign(scan.getResult(), "ct.cumsum(" + valueString(scan.getSource()) +
                                   ", axis=" + std::to_string(scan.getAxis()) +
                                   ", reverse=" +
                                   (scan.getReverse() ? "True" : "False") + ")");
    } else if (auto atomic = dyn_cast<AtomicRMWOp>(operation)) {
      auto atomicOperation = [](AtomicRMWKind kind) -> StringRef {
        switch (kind) {
        case AtomicRMWKind::Exchange: return "xchg";
        case AtomicRMWKind::Add: return "add";
        case AtomicRMWKind::Maximum: return "max";
        case AtomicRMWKind::Minimum: return "min";
        case AtomicRMWKind::BitwiseAnd: return "and";
        case AtomicRMWKind::BitwiseOr: return "or";
        case AtomicRMWKind::BitwiseXor: return "xor";
        }
        llvm_unreachable("unhandled atomic RMW kind");
      };
      auto atomicOrder = [](AtomicOrdering ordering) -> StringRef {
        switch (ordering) {
        case AtomicOrdering::Relaxed: return "RELAXED";
        case AtomicOrdering::Acquire: return "ACQUIRE";
        case AtomicOrdering::Release: return "RELEASE";
        case AtomicOrdering::AcquireRelease: return "ACQ_REL";
        }
        llvm_unreachable("unhandled atomic ordering");
      };
      auto atomicScope = [](gpu::AtomicSharingDomain sharing) -> StringRef {
        switch (sharing) {
        case gpu::AtomicSharingDomain::ProgramInstance: return "BLOCK";
        case gpu::AtomicSharingDomain::KernelInvocation: return "DEVICE";
        }
        llvm_unreachable("unhandled atomic sharing domain");
      };
      assign(atomic.getResult(),
             "ct.atomic_" + atomicOperation(atomic.getKind()).str() + "(" +
                 valueString(atomic.getResource()) + ", " +
                 tuple(atomic.getCoordinates()) + ", " +
                 valueString(atomic.getValue()) +
                 ", check_bounds=True, memory_order=ct.MemoryOrder." +
                 atomicOrder(atomic.getOrdering()).str() +
                 ", memory_scope=ct.MemoryScope." +
                 atomicScope(atomic.getSharing()).str() + ")");
    } else if (auto extract = dyn_cast<ExtractOp>(operation)) {
      std::string shape = "(";
      for (Attribute extent : extract.getExtractionShape())
        shape += expressionString(mlir::cast<gpu::PhysicalExprAttr>(extent), false) + ", ";
      shape += ")";
      std::string result = "ct.extract(" + valueString(extract.getSource()) +
                           ", index=" + tuple(extract.getCoordinates()) +
                           ", shape=" + shape + ")";
      if (auto fragment = dyn_cast<gpu::FragmentType>(extract.getResult().getType()))
        result += ".reshape(" + fragmentShape(fragment) + ")";
      else
        result += ".item()";
      assign(extract.getResult(), result);
    } else if (auto store = dyn_cast<TileStoreOp>(operation)) {
      line("ct.store(" + valueString(store.getResource()) + ", index=" +
           tuple(store.getTileIndices()) + ", tile=" +
           valueString(store.getValue()) + ", allow_tma=" +
           valueString(store.getAllowTma()) + ")");
    } else if (auto store = dyn_cast<ScalarStoreOp>(operation)) {
      std::string call = "ct.scatter(" + valueString(store.getResource()) +
                         ", " + tuple(store.getIndices()) + ", " +
                         valueString(store.getValue());
      if (store.getValid())
        call += ", mask=" + valueString(store.getValid());
      line(call + (store.getInBounds() ? ", check_bounds=False)"
                                       : ", check_bounds=True)"));
    } else if (auto scatter = dyn_cast<ScatterStoreOp>(operation)) {
      std::string call = "ct.scatter(" + valueString(scatter.getResource()) +
                         ", " + tuple(scatter.getCoordinates()) +
                         ", " +
                         valueString(scatter.getValue());
      if (scatter.getValid())
        call += ", mask=" + valueString(scatter.getValid());
      line(call + (scatter.getInBounds() ? ", check_bounds=False)"
                                         : ", check_bounds=True)"));
    } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
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
      std::string indexType = pythonType(loop.getInductionVar().getType());
      line("for " + induction + " in range(ct.astype(" +
           valueString(loop.getLowerBound()) + ", " + indexType + "), ct.astype(" +
           valueString(loop.getUpperBound()) + ", " + indexType + "), ct.astype(" +
           valueString(loop.getStep()) + ", " + indexType + ")):");
      ++indent;
      emitBlock(*loop.getBody(), true, results);
      --indent;
    } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      SmallVector<std::string> carries;
      for (auto [result, initial] : llvm::zip(loop.getResults(), loop.getInits())) {
        std::string name = newName();
        values[result] = name;
        std::string initialValue = valueString(initial);
        if (initial.getType().isIntOrIndex())
          initialValue = "ct.astype(" + initialValue + ", " +
                         pythonType(initial.getType()) + ")";
        line(name + " = " + initialValue);
        carries.push_back(name);
      }
      Block &before = loop.getBefore().front();
      for (auto [argument, name] : llvm::zip(before.getArguments(), carries))
        values[argument] = name;
      line("while True:");
      ++indent;
      for (Operation &nested : before.without_terminator())
        emitOperation(nested);
      auto condition = mlir::cast<scf::ConditionOp>(before.getTerminator());
      line("if not " + valueString(condition.getCondition()) + ":");
      ++indent;
      line("break");
      --indent;
      Block &after = loop.getAfter().front();
      for (auto [argument, forwarded] : llvm::zip(after.getArguments(), condition.getArgs()))
        values[argument] = valueString(forwarded);
      emitBlock(after, true, carries);
      --indent;
    } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
      SmallVector<std::string> results;
      for (Value result : branch.getResults()) {
        std::string name = newName();
        values[result] = name;
        results.push_back(name);
      }
      line("if " + valueString(branch.getCondition()) + ":");
      ++indent;
      emitIfBranch(branch.getThenRegion().front(), results);
      --indent;
      if (!branch.getElseRegion().empty()) {
        line("else:");
        ++indent;
        emitIfBranch(branch.getElseRegion().front(), results);
        --indent;
      }
    } else {
      operation.emitOpError("has no terminal cuTile spelling");
      failed = true;
    }
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
    auto nativeMinMax = [&](StringRef spelling) {
      return spelling.str() + "(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    };
    auto propagatingMinMax = [&](StringRef spelling) {
      std::string lhs = valueString(binary.getLhs());
      std::string rhs = valueString(binary.getRhs());
      std::string native = spelling.str() + "(" + lhs + ", " + rhs + ")";
      return "ct.where(ct.isnan(" + lhs + "), " + lhs +
             ", ct.where(ct.isnan(" + rhs + "), " + rhs + ", " + native + "))";
    };
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: return infix("+");
    case BinaryOperator::Subtract: return infix("-");
    case BinaryOperator::Multiply: return infix("*");
    case BinaryOperator::TrueDivide:
      if (binary.getApproximate())
        return "ct.truediv(" + valueString(binary.getLhs()) + ", " +
               valueString(binary.getRhs()) +
               ", rounding_mode=ct.RoundingMode.APPROX, flush_to_zero=" +
               (binary.getFlushToZero() ? "True" : "False") + ")";
      return infix("/");
    case BinaryOperator::FloorDivide: return infix("//");
    case BinaryOperator::Remainder: return infix("%");
    case BinaryOperator::Power: return infix("**");
    case BinaryOperator::MaximumNum: return nativeMinMax("ct.maximum");
    case BinaryOperator::MinimumNum: return nativeMinMax("ct.minimum");
    case BinaryOperator::Maximum:
      return elementType(binary.getResult().getType()).isIntOrIndex()
                 ? nativeMinMax("ct.maximum")
                 : propagatingMinMax("ct.maximum");
    case BinaryOperator::Minimum:
      return elementType(binary.getResult().getType()).isIntOrIndex()
                 ? nativeMinMax("ct.minimum")
                 : propagatingMinMax("ct.minimum");
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
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate: return "(-" + input + ")";
    case UnaryOperator::Not: return "(~" + input + ")";
    case UnaryOperator::Exp: return call("ct.exp");
    case UnaryOperator::Exp2:
      if (unary.getApproximate())
        return "ct.exp2(" + input + ", flush_to_zero=" +
               (unary.getFlushToZero() ? "True" : "False") + ")";
      return call("ct.exp2");
    case UnaryOperator::Log: return call("ct.log");
    case UnaryOperator::Sin: return call("ct.sin");
    case UnaryOperator::Cos: return call("ct.cos");
    case UnaryOperator::Floor: return call("ct.floor");
    case UnaryOperator::Erf: return call("ct.erf");
    case UnaryOperator::Rsqrt: return call("ct.rsqrt");
    case UnaryOperator::Sigmoid: return call("ct.sigmoid");
    case UnaryOperator::Tanh:
      return unary.getApproximate()
                 ? "ct.tanh(" + input + ", rounding_mode=ct.RoundingMode.APPROX)"
                 : call("ct.tanh");
    case UnaryOperator::Abs: return call("ct.abs");
    case UnaryOperator::Sqrt: return call("ct.sqrt");
    }
    llvm_unreachable("unhandled Intent unary operator");
  }

  std::string broadcastValue(Value value, gpu::FragmentType target) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return "ct.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", dtype=" + pythonType(target.getElementType()) + ")";
    if (source == target)
      return valueString(value);
    gpu::BroadcastProjection projection =
        gpu::queryBroadcastProjection(source, target);
    if (!projection.isExact()) {
      kernel.emitError("cuTile broadcast lost its shared axis projection");
      failed = true;
      return "<invalid-cutile-broadcast>";
    }
    if (source.getShape().size() == target.getShape().size())
      return "ct.broadcast_to(" + valueString(value) + ", " +
             fragmentShape(target) + ")";
    SmallVector<int64_t> targetForSource(source.getShape().size(), -1);
    for (auto [targetIndex, sourceIndex] :
         llvm::enumerate(projection.targetToSource))
      if (sourceIndex)
        targetForSource[*sourceIndex] = targetIndex;

    SmallVector<unsigned> sourceOrder(source.getShape().size());
    std::iota(sourceOrder.begin(), sourceOrder.end(), 0);
    llvm::sort(sourceOrder, [&](unsigned lhs, unsigned rhs) {
      return targetForSource[lhs] < targetForSource[rhs];
    });
    std::string input = valueString(value);
    bool permuted = llvm::any_of(
        llvm::enumerate(sourceOrder),
        [](auto item) { return item.index() != item.value(); });
    if (permuted) {
      std::string permutation = "(";
      for (unsigned axis : sourceOrder)
        permutation += std::to_string(axis) + ", ";
      permutation += ")";
      input = "ct.permute(" + input + ", " + permutation + ")";
    }

    std::string reshape = "(";
    unsigned orderedSource = 0;
    for (unsigned targetAxis = 0; targetAxis < target.getShape().size();
         ++targetAxis) {
      if (orderedSource < sourceOrder.size() &&
          targetForSource[sourceOrder[orderedSource]] ==
              static_cast<int64_t>(targetAxis)) {
        reshape += expressionString(
                       cast<gpu::PhysicalExprAttr>(
                           source.getShape()[sourceOrder[orderedSource]]),
                       false) +
                   ", ";
        ++orderedSource;
      } else {
        reshape += "1, ";
      }
    }
    reshape += ")";
    return "ct.broadcast_to(ct.reshape(" + input + ", " + reshape + "), " +
           fragmentShape(target) + ")";
  }

  std::string tuple(ValueRange valuesRange) {
    std::string result = "(";
    for (auto [index, value] : llvm::enumerate(valuesRange)) {
      if (index)
        result += ", ";
      result += valueString(value);
    }
    if (valuesRange.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string outputShape(const ViewABI &view) {
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
      } else {
        const MetadataABI &metadata = dimensionBindings.lookup(dimension);
        const ViewABI &source = viewByABI(metadata.sourceABI);
        shape += source.name + ".shape[" + std::to_string(metadata.sourceAxis) + "]";
      }
    }
    if (ids.size() == 1)
      shape += ",";
    return shape + ")";
  }

  std::string compilerHints(StringRef configName) {
    std::string result = "{";
    for (const auto &[hint, parameter] : providerHintParameters) {
      std::string value = configName.str() + "." + parameter;
      if (hint == "num_worker_warps")
        value = "(None if " + value + " == " +
                std::to_string(inferredWorkerWarps) + " else " + value + ")";
      result += "\"" + hint + "\": " + value + ", ";
    }
    return result + "}";
  }

  std::string joinKernelArguments(StringRef configName, bool trial = false) {
    std::string result = joinViewNames(views);
    for (const ArrayViewABI &view : arrayViews)
      result += ", " + (trial ? view.trialName : view.name) + ", " +
                (trial ? view.trialEligible : view.eligible);
    for (const ScalarABI &scalar : scalars) {
      if (!result.empty())
        result += ", ";
      result += scalar.name;
    }
    for (const MetadataABI &metadata : metadataArguments)
      result += ", " + metadata.name;
    SmallVector<std::string> parameters;
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (!providerHint(parameter).empty())
        return;
      parameters.push_back(parameter.getParameter().getName().getValue().str());
    });
    for (StringRef parameter : parameters)
      result += ", " +
                (fullCoverageParameters.count(parameter.str())
                     ? parameter.str()
                     : configName.str() + "." + parameter.str());
    return result;
  }

  std::string valueString(Value value) {
    auto found = values.find(value);
    if (found == values.end()) {
      if (Operation *producer = value.getDefiningOp())
        producer->emitOpError(
            "cuTile terminal translation encountered an unmapped SSA value");
      else
        kernel.emitError(
            "cuTile terminal translation encountered an unmapped block argument");
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

  std::string joinLaunchArguments(bool includeOutputs = true) const {
    std::map<unsigned, std::string> arguments;
    for (const ViewABI &view : views)
      if (includeOutputs || view.type.getAccess() != 1)
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
  llvm::DenseMap<Operation *, std::string> collectiveHelpers;
  SmallVector<ViewABI> views;
  SmallVector<ArrayViewABI> arrayViews;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  llvm::DenseMap<int64_t, MetadataABI> dimensionBindings;
  std::map<std::string, CoverageParameter> fullCoverageParameters;
  llvm::StringSet<> fullCoverageParameterNames;
  std::map<std::string, std::string> providerHintParameters;
  unsigned indent = 0;
  unsigned counter = 0;
  bool failed = false;
};

} // namespace

LogicalResult serializeProgram(ModuleOp module, std::string &source) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (!(*kernel)->hasAttr("intent_cutile.legalized"))
    return (*kernel).emitError("cuTile program was not provider-legalized");
  llvm::raw_string_ostream stream(source);
  Serializer serializer(*kernel, stream);
  LogicalResult result = serializer.emit();
  stream.flush();
  return result;
}

} // namespace intent::cutile

#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::cpu {

std::optional<std::string> checkInputRequirement(
    Value source, Value element, const InputRequirement &requirement) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || requirement.panelSize <= 0 || requirement.alignment <= 0 ||
      requirement.windowAlignment <= 0 || requirement.panelSize % requirement.windowAlignment != 0 ||
      !llvm::isPowerOf2_64(requirement.alignment))
    return "implementation has an invalid input representation requirement";
  if (!requirement.elementType || !requirement.elementType.isIntOrIndexOrFloat())
    return "implementation input requires an explicit scalar representation type";
  if (requirement.elementType != type.getElementType()) {
    if (element.use_empty() || !llvm::all_of(element.getUsers(), [&](Operation *user) {
          auto widen = dyn_cast<arith::ExtFOp>(user);
          return widen && widen.getType() == requirement.elementType;
        }))
      return "input representation must preserve the consumer's explicit floating extension";
  }
  int64_t bits = requirement.elementType.isIndex() ? 64 : requirement.elementType.getIntOrFloatBitWidth();
  if (requirement.panelAxis >= static_cast<unsigned>(type.getRank()) || requirement.alignment < (bits + 7) / 8)
    return "implementation input panel does not match its typed source";
  return std::nullopt;
}

std::optional<std::string> checkInputRequirements(
    linalg::GenericOp operation, ArrayRef<InputRequirement> requirements) {
  SmallVector<unsigned> operands;
  for (const auto &requirement : requirements) {
    if (requirement.operand >= operation.getInputs().size() ||
        llvm::is_contained(operands, requirement.operand))
      return "implementation has an invalid or repeated input representation operand";
    operands.push_back(requirement.operand);
    if (auto reason = checkInputRequirement(operation.getInputs()[requirement.operand],
            operation.getRegion().front().getArgument(requirement.operand), requirement))
      return "operand " + std::to_string(requirement.operand) + ": " + *reason;
  }
  return std::nullopt;
}

SmallVector<InputRequirement> Implementation::inputRequirements(
    linalg::GenericOp operation, ConfigurationAttr configuration, ImplementationAttr binding) const {
  return inputs ? inputs(operation, configuration, binding) : SmallVector<InputRequirement>{};
}

bool ContractionRequirements::acceptsInputLayout(unsigned operand, MemRefType type) const {
  if (operand >= unitInnerStride.size()) return false;
  if (!unitInnerStride[operand]) return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.getRank() && succeeded(type.getStridesAndOffset(strides, offset)) && strides.back() == 1;
}

namespace {
std::optional<std::string> checkInputLayouts(Operation *operation, const ContractionRequirements &requirements) {
  if (llvm::none_of(requirements.unitInnerStride, [](bool required) { return required; })) return std::nullopt;
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  if (!generic || generic.getInputs().size() != requirements.unitInnerStride.size())
    return "contraction input layout requirements need two shaped inputs";
  for (auto [operand, input] : llvm::enumerate(generic.getInputs())) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || !requirements.acceptsInputLayout(operand, type))
      return "operand " + std::to_string(operand) + " requires a proven unit inner stride";
  }
  return std::nullopt;
}

struct ImplementationMatch {
  const Implementation *implementation;
  ImplementationAttr binding;
  bool supplyFree;
};

ConfigurationAttr sharedConfiguration(MLIRContext *context, const Configuration &configuration) {
  return ConfigurationAttr::get(context, configuration.taskGrain, configuration.tileM,
      configuration.tileN, configuration.tileK, configuration.regionSize);
}

std::optional<std::string> matchImplementation(
    const Implementation &implementation, Operation *operation,
    CapabilitiesAttr capabilities, const Configuration &configuration,
    ImplementationMatch &match) {
  if (!implementation.applicable(operation)) return "implementation does not match this computation";
  if (auto reason = checkInputLayouts(operation, implementation.contraction)) return reason;
  if (auto reason = implementation.check(operation, capabilities, configuration)) return reason;
  Builder builder(operation->getContext());
  auto binding = ImplementationAttr::get(builder.getContext(), builder.getStringAttr(implementation.name),
      implementation.parameters(builder, configuration));
  bool supplyFree = true;
  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    auto requirements = implementation.inputRequirements(
        generic, sharedConfiguration(builder.getContext(), configuration), binding);
    if (auto reason = checkInputRequirements(generic, requirements)) return reason;
    supplyFree = requirements.empty();
  } else if (implementation.inputs) {
    return "input representation requirements need a structured linalg computation";
  }
  match = {&implementation, binding, supplyFree};
  return std::nullopt;
}
} // namespace

void ImplementationRegistry::addProfile(StringRef name,
                                         ArrayRef<StringRef> localParameters) {
  profiles.push_back({name, llvm::to_vector(localParameters)});
}

std::optional<ArrayRef<StringRef>>
ImplementationRegistry::profileParameters(StringRef name) const {
  const ProfileSchema *selected = nullptr;
  for (const ProfileSchema &schema : profiles) {
    if (schema.name != name)
      continue;
    if (selected)
      return std::nullopt;
    selected = &schema;
  }
  return selected ? std::optional<ArrayRef<StringRef>>(selected->localParameters)
                  : std::nullopt;
}

bool needsImplementation(Operation *operation) {
  if (auto function = dyn_cast<func::FuncOp>(operation)) {
    bool computation = false;
    function.walk([&](Operation *nested) {
      computation |= isa<linalg::GenericOp, ReduceOp, ScanOp, HistogramOp, QuantizeOp, QuantizedDotOp>(nested);
    });
    return !function.isExternal() && !computation;
  }
  return isa<linalg::GenericOp, ReduceOp, ScanOp, HistogramOp, QuantizeOp, QuantizedDotOp>(operation);
}

FailureOr<const Implementation *> ImplementationRegistry::lookup(Operation *operation) const {
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (binding)
    for (const auto &implementation : implementations)
      if (implementation.name == binding.getName().getValue() && implementation.applicable(operation) &&
          !checkInputLayouts(operation, implementation.contraction))
        return &implementation;
  operation->emitError("CPU computation has lost its selected implementation binding");
  return failure();
}

SmallVector<SmallVector<ImplementationAttr>> ImplementationRegistry::candidates(
    func::FuncOp function, CapabilitiesAttr capabilities,
    const Configuration &configuration,
    llvm::function_ref<void(Operation *, StringRef, StringRef)> rejected) const {
  struct Computation {
    SmallVector<ImplementationMatch> choices;
    unsigned anchor = 0;
  };
  SmallVector<Computation> computations;
  function.walk([&](Operation *operation) {
    if (!needsImplementation(operation)) return;
    auto &computation = computations.emplace_back();
    SmallVector<std::pair<StringRef, std::string>> rejections;
    for (const auto &implementation : implementations) {
      if (!implementation.applicable(operation)) continue;
      ImplementationMatch match;
      if (auto reason = matchImplementation(implementation, operation, capabilities, configuration, match)) {
        rejections.emplace_back(implementation.name, std::move(*reason));
        continue;
      }
      computation.choices.push_back(match);
    }
    if (computation.choices.empty()) {
      if (rejections.empty())
        rejected(operation, {}, "no registered implementation matches this computation");
      else
        for (const auto &[implementation, reason] : rejections)
          rejected(operation, implementation, reason);
      return;
    }
    // Use current input requirements to anchor each contraction independently.
    // Prepared alternatives remain in the portfolio: supply-free is not a cost
    // model for conversion, reuse, cache locality or the eventual winner.
    for (auto [index, choice] : llvm::enumerate(computation.choices))
      if (choice.implementation->formTile && choice.supplyFree) {
        computation.anchor = index;
        break;
      }
  });
  SmallVector<SmallVector<ImplementationAttr>> result;
  if (llvm::any_of(computations, [](const auto &computation) { return computation.choices.empty(); })) return result;
  auto candidate = [&](const Implementation *preferred) {
    SmallVector<ImplementationAttr> bindings;
    SmallVector<StringAttr> consumed;
    for (const auto &computation : computations) {
      auto selected = llvm::find_if(computation.choices, [&](const ImplementationMatch &choice) {
        return choice.implementation == preferred;
      });
      const auto &binding = selected == computation.choices.end()
          ? computation.choices[computation.anchor].binding : selected->binding;
      for (NamedAttribute parameter : binding.getParameters()) consumed.push_back(parameter.getName());
      bindings.push_back(binding);
    }
    for (NamedAttribute parameter : configuration.local)
      if (!llvm::is_contained(consumed, parameter.getName())) {
        rejected(function, preferred ? preferred->name : StringRef(),
            "candidate does not consume local parameter '" + parameter.getName().getValue().str() + "'");
        return;
      }
    if (!llvm::is_contained(result, bindings)) result.push_back(std::move(bindings));
  };
  candidate(nullptr);
  // Correlate a registered alternative across every consumer it can serve.
  // This finite portfolio covers each legal implementation without forming
  // an independent Cartesian product for every operation in the program.
  for (const auto &implementation : implementations) candidate(&implementation);
  return result;
}

LogicalResult ImplementationRegistry::bind(func::FuncOp function, CapabilitiesAttr capabilities,
                                            const Configuration &configuration,
                                            ArrayRef<ImplementationAttr> bindings) const {
  Builder builder(function.getContext());
  unsigned ordinal = 0;
  bool matrix = false;
  SmallVector<std::pair<Operation *, ImplementationAttr>> selected;
  SmallVector<Attribute> distinctBindings;
  auto status = function.walk([&](Operation *operation) -> WalkResult {
    if (!needsImplementation(operation)) return WalkResult::advance();
    if (ordinal == bindings.size()) {
      operation->emitError("CPU candidate is missing an implementation binding");
      return WalkResult::interrupt();
    }
    auto binding = bindings[ordinal++];
    auto implementation = llvm::find_if(implementations, [&](const Implementation &candidate) {
      return candidate.name == binding.getName().getValue();
    });
    if (implementation == implementations.end()) {
      operation->emitError("CPU candidate names an unregistered implementation: ") << binding.getName();
      return WalkResult::interrupt();
    }
    ImplementationMatch match;
    if (auto reason = matchImplementation(*implementation, operation, capabilities, configuration, match)) {
      operation->emitError("CPU candidate implementation '") << implementation->name << "' rejected: " << *reason;
      return WalkResult::interrupt();
    }
    if (match.binding != binding) {
      operation->emitError("CPU candidate parameters do not match the selected implementation");
      return WalkResult::interrupt();
    }
    selected.emplace_back(operation, binding);
    if (!llvm::is_contained(distinctBindings, Attribute(binding)))
      distinctBindings.push_back(binding);
    matrix |= implementation->requiresMatrixI8I32;
    return WalkResult::advance();
  });
  if (status.wasInterrupted()) return failure();
  if (ordinal != bindings.size())
    return function.emitError("CPU candidate implementation bindings do not match its current computations");
  for (auto [operation, binding] : selected)
    operation->setAttr("intent_cpu.implementation", binding);
  function->setAttr("intent_cpu.configuration", sharedConfiguration(function.getContext(), configuration));
  // This preserves the selected implementation/parameter identity after local
  // expansion. Mojo's coordinated vector materialization and provider artifact
  // metadata consume it; it is not a reclassification of the expanded graph.
  function->setAttr("intent_cpu.implementations", builder.getArrayAttr(distinctBindings));
  function->setAttr("intent_cpu.requires_matrix_i8_i32", builder.getBoolAttr(matrix));
  return success();
}

int64_t implementationParameter(ImplementationAttr binding, StringRef name) {
  return cast<IntegerAttr>(binding.getParameters().get(name)).getInt();
}

}

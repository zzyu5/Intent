#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"

using namespace mlir;
namespace intent::cpu {

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
      if (implementation.name == binding.getName().getValue() && implementation.applicable(operation))
        return &implementation;
  operation->emitError("CPU computation has lost its selected implementation binding");
  return failure();
}

SmallVector<SmallVector<ImplementationAttr>> ImplementationRegistry::candidates(
    func::FuncOp function, CapabilitiesAttr capabilities,
    const Configuration &configuration) const {
  SmallVector<SmallVector<const Implementation *>> legal;
  function.walk([&](Operation *operation) {
    if (!needsImplementation(operation)) return;
    auto &choices = legal.emplace_back();
    for (const auto &implementation : implementations)
      if (implementation.applicable(operation) && implementation.legal(operation, capabilities, configuration))
        choices.push_back(&implementation);
  });
  SmallVector<SmallVector<ImplementationAttr>> result;
  if (llvm::any_of(legal, [](const auto &choices) { return choices.empty(); })) return result;
  Builder builder(function.getContext());
  auto candidate = [&](const Implementation *preferred) {
    SmallVector<ImplementationAttr> bindings;
    SmallVector<StringAttr> consumed;
    for (const auto &choices : legal) {
      const auto *selected = llvm::is_contained(choices, preferred) ? preferred : choices.front();
      auto parameters = selected->parameters(builder, configuration);
      for (NamedAttribute parameter : parameters) consumed.push_back(parameter.getName());
      bindings.push_back(ImplementationAttr::get(function.getContext(), builder.getStringAttr(selected->name), parameters));
    }
    for (NamedAttribute parameter : configuration.local)
      if (!llvm::is_contained(consumed, parameter.getName())) return;
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
  bool invalid = false;
  unsigned ordinal = 0;
  bool matrix = false;
  SmallVector<std::pair<Operation *, ImplementationAttr>> selected;
  SmallVector<Attribute> distinctBindings;
  function.walk([&](Operation *operation) {
    if (!needsImplementation(operation)) return;
    if (ordinal == bindings.size()) {
      invalid = true;
      return;
    }
    auto binding = bindings[ordinal++];
    auto implementation = llvm::find_if(implementations, [&](const Implementation &candidate) {
      return candidate.name == binding.getName().getValue() && candidate.applicable(operation);
    });
    if (implementation == implementations.end() ||
        !implementation->legal(operation, capabilities, configuration) ||
        implementation->parameters(builder, configuration) != binding.getParameters()) {
      invalid = true;
      return;
    }
    selected.emplace_back(operation, binding);
    if (!llvm::is_contained(distinctBindings, Attribute(binding)))
      distinctBindings.push_back(binding);
    matrix |= implementation->requiresMatrixI8I32;
  });
  if (invalid || ordinal != bindings.size())
    return function.emitError("CPU candidate implementation bindings do not match its current computations");
  for (auto [operation, binding] : selected)
    operation->setAttr("intent_cpu.implementation", binding);
  function->setAttr("intent_cpu.configuration", ConfigurationAttr::get(
      function.getContext(), configuration.taskGrain, configuration.tileM,
      configuration.tileN, configuration.tileK, configuration.regionSize));
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

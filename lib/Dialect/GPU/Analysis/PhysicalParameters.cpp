#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {

FailureOr<PhysicalParameterSpace>
PhysicalParameterSpace::read(func::FuncOp kernel) {
  PhysicalParameterSpace result;
  result.kernel = kernel;
  llvm::StringSet<> names;
  auto walked = kernel.walk([&](ParameterOp parameter) {
    auto definition = parameter.getParameter();
    if (!names.insert(definition.getName().getValue()).second) {
      parameter.emitOpError("duplicates a physical parameter name");
      return WalkResult::interrupt();
    }
    if (definition.getCandidates().empty()) {
      parameter.emitOpError("has an empty physical parameter domain");
      return WalkResult::interrupt();
    }
    result.parameters.push_back({
        parameter, definition,
        parameter->hasAttr(coverageDimensionAttr) ||
            definition.getCategory() ==
                static_cast<uint32_t>(ParameterCategory::Coverage),
        definition.getCategory() ==
            static_cast<uint32_t>(ParameterCategory::Provider)});
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();
  return result;
}

const PhysicalParameterDomain *
PhysicalParameterSpace::find(ParameterRole role) const {
  for (const auto &domain : parameters)
    if (domain.role() == role)
      return &domain;
  return nullptr;
}

LogicalResult PhysicalParameterSpace::verifyBindings(
    DictionaryAttr bindings, ParameterBindingScope scope) const {
  if (!bindings)
    return kernel->emitError("physical parameter config must be a dictionary");
  unsigned expected = 0;
  for (const auto &domain : parameters) {
    if (domain.coverage ||
        (scope == ParameterBindingScope::Shared && domain.provider))
      continue;
    ++expected;
    auto value = bindings.getAs<IntegerAttr>(domain.name());
    if (!value || !llvm::is_contained(domain.candidates(), value.getInt()))
      return kernel->emitError("physical config has no in-domain binding for ")
             << domain.name().getValue();
  }
  if (bindings.size() != expected)
    return kernel->emitError("physical config contains an undeclared or deferred binding");
  return success();
}

FailureOr<SmallVector<DictionaryAttr>>
PhysicalParameterSpace::sharedConfigurations() const {
  auto encoded = kernel->getAttrOfType<ArrayAttr>(sharedConfigTuplesAttr);
  if (!encoded || encoded.empty())
    return kernel->emitError("provider legalization requires shared config tuples");
  SmallVector<DictionaryAttr> result;
  for (Attribute attribute : encoded) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (failed(verifyBindings(tuple, ParameterBindingScope::Shared)))
      return failure();
    result.push_back(tuple);
  }
  return result;
}

} // namespace intent::gpu

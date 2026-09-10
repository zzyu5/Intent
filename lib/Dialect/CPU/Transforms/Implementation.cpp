#include "Intent/Dialect/CPU/Transforms/Implementation.h"

using namespace mlir;
namespace intent::cpu {

bool needsImplementation(Operation *operation) {
  return isa<linalg::GenericOp, ReduceOp, QuantizeOp, QuantizedDotOp>(operation);
}

FailureOr<const Implementation *> ImplementationRegistry::select(Operation *operation) const {
  for (const auto &implementation : implementations)
    if (implementation.applicable(operation)) return &implementation;
  operation->emitError("no registered CPU implementation accepts this structured computation");
  return failure();
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

bool ImplementationRegistry::legal(func::FuncOp function, CapabilitiesAttr capabilities,
                                    const Configuration &configuration) const {
  bool valid = true;
  function.walk([&](Operation *operation) {
    if (!needsImplementation(operation)) return;
    auto implementation = select(operation);
    valid &= succeeded(implementation) && (*implementation)->legal(capabilities, configuration);
  });
  return valid;
}

LogicalResult ImplementationRegistry::bind(func::FuncOp function, CapabilitiesAttr capabilities,
                                            const Configuration &configuration) const {
  Builder builder(function.getContext());
  bool invalid = false;
  SmallVector<StringAttr> consumed;
  function.walk([&](Operation *operation) {
    if (!needsImplementation(operation)) return;
    auto implementation = select(operation);
    if (failed(implementation) || !(*implementation)->legal(capabilities, configuration)) {
      invalid = true;
      return;
    }
    auto parameters = (*implementation)->parameters(builder, configuration);
    for (NamedAttribute parameter : parameters) consumed.push_back(parameter.getName());
    operation->setAttr("intent_cpu.implementation", ImplementationAttr::get(
        function.getContext(), builder.getStringAttr((*implementation)->name),
        parameters));
  });
  for (NamedAttribute parameter : configuration.local)
    if (!llvm::is_contained(consumed, parameter.getName())) {
      function.emitError("selected CPU implementations do not consume parameter ") << parameter.getName();
      invalid = true;
    }
  return failure(invalid);
}

int64_t implementationParameter(ImplementationAttr binding, StringRef name) {
  return cast<IntegerAttr>(binding.getParameters().get(name)).getInt();
}

}

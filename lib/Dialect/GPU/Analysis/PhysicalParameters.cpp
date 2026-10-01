#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/Configurations.h"

using namespace mlir;

namespace intent::gpu {

FailureOr<PhysicalParameterSpace>
PhysicalParameterSpace::read(func::FuncOp kernel) {
  PhysicalParameterSpace result;
  result.kernel = kernel;
  auto declarations = ConfigurationSpace::read(kernel);
  if (failed(declarations))
    return failure();
  for (auto declaration : declarations->parameters()) {
    auto parameter = dyn_cast<ParameterOp>(declaration.getOperation());
    if (!parameter)
      continue;
    auto phase = parameter.getConfigurationBindingPhase();
    result.parameters.push_back({
        parameter, parameter.getParameter(),
        phase == ConfigurationBindingPhase::Deferred,
        phase == ConfigurationBindingPhase::Provider});
  }
  return result;
}

const PhysicalParameterDomain *
PhysicalParameterSpace::find(ParameterRole role) const {
  for (const auto &domain : parameters)
    if (domain.role() == role)
      return &domain;
  return nullptr;
}

FailureOr<SmallVector<DictionaryAttr>>
PhysicalParameterSpace::sharedConfigurations() const {
  auto space = ConfigurationSpace::read(kernel);
  if (failed(space))
    return failure();
  return space->configurations(ConfigurationStage::Shared);
}

} // namespace intent::gpu

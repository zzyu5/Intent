#include "Intent/Dialect/GPU/Analysis/Configurations.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {

FailureOr<ConfigurationSpace> ConfigurationSpace::read(func::FuncOp kernel) {
  ConfigurationSpace result;
  result.kernel = kernel;
  llvm::StringSet<> names;
  auto walk = kernel.walk([&](ConfigurationParameterOpInterface parameter) {
    auto name = parameter.getConfigurationName();
    if (!name || name.empty() || !names.insert(name.getValue()).second) {
      parameter.emitOpError("configuration symbol must have one nonempty declaration");
      return WalkResult::interrupt();
    }
    auto candidates = parameter.getConfigurationCandidates();
    llvm::DenseSet<int64_t> unique;
    if (candidates.empty() || llvm::any_of(candidates, [&](int64_t value) {
          return !unique.insert(value).second;
        })) {
      parameter.emitOpError("configuration domain must be nonempty and unique");
      return WalkResult::interrupt();
    }
    result.declarations.push_back(parameter);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  return result;
}

LogicalResult ConfigurationSpace::verifyBindings(DictionaryAttr bindings,
                                                 ConfigurationStage stage) const {
  if (!bindings)
    return kernel->emitError("configuration requires a binding dictionary");
  unsigned expected = 0;
  for (auto parameter : declarations) {
    auto phase = parameter.getConfigurationBindingPhase();
    if (phase == ConfigurationBindingPhase::Deferred ||
        (stage == ConfigurationStage::Shared &&
         phase == ConfigurationBindingPhase::Provider))
      continue;
    ++expected;
    auto name = parameter.getConfigurationName();
    auto value = bindings.getAs<IntegerAttr>(name);
    if (!value || !value.getType().isSignlessInteger(64) ||
        !llvm::is_contained(parameter.getConfigurationCandidates(), value.getInt()))
      return kernel->emitError("configuration has no in-domain binding for ")
             << name.getValue();
  }
  if (bindings.size() != expected)
    return kernel->emitError("configuration contains an undeclared or deferred binding");
  return success();
}

FailureOr<SmallVector<DictionaryAttr>>
ConfigurationSpace::configurations(ConfigurationStage stage) const {
  auto set = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  if (!set || set.getStage() != stage)
    return kernel->emitError("requires a ") << stringifyConfigurationStage(stage)
           << " configuration set for the current program";
  SmallVector<DictionaryAttr> result;
  llvm::DenseSet<Attribute> unique;
  if (set.getRows().empty())
    return kernel->emitError("configuration set cannot be empty");
  for (Attribute attribute : set.getRows()) {
    auto bindings = dyn_cast<DictionaryAttr>(attribute);
    if (failed(verifyBindings(bindings, stage)))
      return failure();
    if (!unique.insert(bindings).second)
      return kernel->emitError("configuration set contains duplicate bindings");
    result.push_back(bindings);
  }
  return result;
}

} // namespace intent::gpu

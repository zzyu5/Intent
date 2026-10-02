#include "Intent/Dialect/GPU/Transforms/Resources.h"
#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {

SmallVector<ConfigurationRequirementAttr> collectReductionRequirements(
    func::FuncOp kernel, ArrayRef<ValueRange> sourceGroups,
    ReductionRequirementScope scope) {
  SmallVector<ConfigurationRequirementAttr> requirements;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return requirements;
  Builder builder(kernel.getContext());
  auto limit = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Constant,
      capabilities.getRegistersPerUnit(), builder.getStringAttr(""), builder.getArrayAttr({}));
  for (ValueRange sources : sourceGroups) {
    auto footprint = reductionRegisterFootprint(sources, kernel);
    if (!footprint) continue;
    auto requirement = ConfigurationRequirementAttr::get(kernel.getContext(),
        ConfigurationRequirementKind::NominalBudget,
        ConfigurationRequirementMetric::FragmentRegisterWords, footprint, limit,
        builder.getStringAttr("reduction source exceeds the candidate register budget"));
    if (scope == ReductionRequirementScope::InvocationDependent) {
      // Preserve the provider's specialization-only policy. A concrete sample
      // from each nondeferred domain tests evaluability, not a resource bound.
      // Unknown arithmetic and host metadata remain for specialization too.
      auto known = evaluateConfigurationRequirement(requirement,
          [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
            if (expression.getKind() != PhysicalExprKind::Parameter) return std::nullopt;
            auto parameter = lookupParameter(kernel, expression.getParameterReference());
            if (!parameter || parameter.isDeferred() ||
                parameter.getCategory() == ParameterCategory::Coverage)
              return std::nullopt;
            return parameter.getCandidates().asArrayRef().front();
          });
      if (known.usage) continue;
    }
    if (!llvm::is_contained(requirements, requirement)) requirements.push_back(requirement);
  }
  return requirements;
}

FailureOr<SmallVector<DictionaryAttr>> filterConfigurationRequirements(
    func::FuncOp kernel, ArrayRef<DictionaryAttr> rows,
    ArrayRef<ConfigurationRequirementAttr> requirements) {
  auto options = readCompileOptions(kernel);
  if (failed(options)) return failure();
  struct Rejection {
    DictionaryAttr row;
    ConfigurationRequirementAttr requirement;
    RequirementEvaluation evaluation;
  };
  SmallVector<Rejection> rejected;
  SmallVector<DictionaryAttr> accepted;
  for (DictionaryAttr row : rows) {
    bool valid = true;
    for (ConfigurationRequirementAttr requirement : requirements) {
      auto evaluation = evaluateConfigurationRequirement(requirement, row);
      if (evaluation.bound == FootprintBound::Unknown &&
          options->getOptimizationRemarks()) {
        kernel.emitRemark("configuration requirement deferred to invocation binding: ")
            << stringifyConfigurationRequirementKind(requirement.getKind())
            << ", " << stringifyConfigurationRequirementMetric(requirement.getMetric())
            << "; " << requirement.getMessage().getValue() << "; bindings=" << row;
      }
      if (evaluation.bound != FootprintBound::Exceeds &&
          evaluation.bound != FootprintBound::Invalid)
        continue;
      valid = false;
      rejected.push_back({row, requirement, evaluation});
      if (options->getOptimizationRemarks()) {
        auto remark = kernel.emitRemark("configuration rejected: ");
        remark << stringifyConfigurationRequirementKind(requirement.getKind())
               << ", " << stringifyConfigurationRequirementMetric(requirement.getMetric())
               << "; " << requirement.getMessage().getValue() << "; bindings=" << row;
        if (evaluation.usage) remark << "; usage=" << *evaluation.usage;
        if (evaluation.limit) remark << "; limit=" << *evaluation.limit;
      }
    }
    if (valid) accepted.push_back(row);
  }
  if (accepted.empty()) {
    auto diagnostic = kernel.emitError("no candidate satisfies the current configuration requirements");
    for (const Rejection &rejection : rejected) {
      auto &note = diagnostic.attachNote(kernel.getLoc());
      note << stringifyConfigurationRequirementKind(rejection.requirement.getKind())
           << ", " << stringifyConfigurationRequirementMetric(rejection.requirement.getMetric())
           << "; " << rejection.requirement.getMessage().getValue()
           << "; bindings=" << rejection.row;
      if (rejection.evaluation.usage) note << "; usage=" << *rejection.evaluation.usage;
      if (rejection.evaluation.limit) note << "; limit=" << *rejection.evaluation.limit;
    }
    return failure();
  }
  return accepted;
}

} // namespace intent::gpu

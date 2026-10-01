#include "ConfigurationPolicy.h"
#include "Intent/Dialect/GPU/Analysis/Configurations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

LogicalResult materializeCoverageBound(func::FuncOp kernel, ParameterOp parameter) {
  auto schema = parameter.getParameter();
  auto role = static_cast<ParameterRole>(schema.getRole());
  auto category = static_cast<ParameterCategory>(schema.getCategory());
  const bool coverage = parameter->hasAttr(coverageDimensionAttr);
  if ((role == ParameterRole::FullCoverage ||
       category == ParameterCategory::Coverage) &&
      (!coverage || category != ParameterCategory::Coverage)) {
    parameter.emitOpError(
        "full-coverage parameter lacks its typed coverage category or dimension");
    return failure();
  }
  if (coverage && !parameter->hasAttr(coverageBoundAttr)) {
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    PhysicalExprAttr bound;
    if (binding.isExact() && binding.dimension)
      for (BlockArgument argument : kernel.getArguments()) {
        DictionaryAttr attributes =
            kernel.getArgAttrDict(argument.getArgNumber());
        auto kind = attributes.getAs<StringAttr>(abiKindAttr);
        auto dimension = attributes.getAs<IntegerAttr>(dimensionAttr);
        if (kind && kind.getValue() == "dimension" && dimension &&
            dimension.getInt() == *binding.dimension)
          bound = queryLaunchExpression(argument);
      }
    if (!bound) {
      parameter.emitOpError(
          "full-coverage parameter has no launch-visible bound expression");
      return failure();
    }
    parameter->setAttr(coverageBoundAttr, bound);
  }
  return success();
}

} // namespace

LogicalResult materializeSharedConfigTuples(func::FuncOp kernel) {
  auto tables = TuningProfiles::from(kernel->getParentOfType<ModuleOp>());
  if (failed(tables))
    return failure();
  auto declarations = PhysicalParameterSpace::read(kernel);
  if (failed(declarations))
    return failure();
  SmallVector<ParameterOp> parameters;
  for (const PhysicalParameterDomain &domain : declarations->domains()) {
    if (failed(materializeCoverageBound(kernel, domain.operation)))
      return failure();
    if (!domain.coverage && !domain.provider)
      parameters.push_back(domain.operation);
  }

  Builder builder(kernel.getContext());
  const FragmentResourceAnalysis resources(kernel);
  const auto facts =
      configuration::analyzeConfigurationPolicy(kernel, parameters, resources);
  SmallVector<DictionaryAttr> tuples;
  auto append = [&](DictionaryAttr tuple) {
    if (!llvm::is_contained(tuples, tuple))
      tuples.push_back(tuple);
  };
  bool invalidFootprint = false;
  auto project = configuration::projectConfigurationProfiles(
      kernel, facts, *tables,
      [&](NamedAttrList &bindings, configuration::ProfileLookup profileFor,
          bool splitInnerAxis) {
        configuration::bindContractionFreeExtents(
            facts.freeExtents, bindings, profileFor, builder, splitInnerAxis);
        if (failed(configuration::bindTraversalFragmentFootprints(
                kernel, parameters, resources, bindings, builder))) {
          invalidFootprint = true;
          return;
        }
        append(bindings.getDictionary(kernel.getContext()));
        configuration::appendFullResultContractionTuples(
            kernel, facts.fullResultContractions, bindings, parameters,
            resources, profileFor, builder, append);
      });
  if (failed(project))
    return failure();
  if (invalidFootprint && tuples.empty())
    return kernel.emitOpError(
        "physical traversal has no profile within the fragment register budget");
  return writeConfigurations(kernel, tuples, ConfigurationStage::Shared);
}

LogicalResult verifySharedConfigTuples(func::FuncOp kernel) {
  auto coverage = kernel.walk([&](ParameterOp parameter) -> WalkResult {
    if (parameter->hasAttr(coverageDimensionAttr) &&
        !parameter->getAttrOfType<PhysicalExprAttr>(coverageBoundAttr)) {
      parameter.emitOpError(
          "full-coverage parameter requires a typed bound expression");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (coverage.wasInterrupted())
    return failure();
  auto space = ConfigurationSpace::read(kernel);
  if (failed(space))
    return failure();
  auto tuples = space->configurations(ConfigurationStage::Shared);
  return failed(tuples) ? failure() : success();
}

} // namespace intent::gpu

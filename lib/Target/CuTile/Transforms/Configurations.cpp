#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"

#include "Configurations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/Resources.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/CuTile/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
namespace intent::cutile {
namespace {

FailureOr<SmallVector<gpu::ConfigurationRequirementAttr>>
collectConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = gpu::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  SmallVector<ValueRange> reductionSources;
  kernel.walk([&](ReduceOp reduce) {
    reductionSources.push_back(reduce.getSources());
  });
  auto requirements = gpu::collectReductionRequirements(
      kernel, reductionSources,
      gpu::ReductionRequirementScope::InvocationDependent);
  auto resident = parameters->find(gpu::ParameterRole::ResidentWorkers);
  auto ctas = parameters->find(gpu::ParameterRole::ProviderCTAs);
  auto occupancy = parameters->find(gpu::ParameterRole::ProviderOccupancy);
  if (resident && ctas && occupancy) {
    auto capabilities =
        kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
    if (!capabilities || capabilities.getComputeUnits() <= 0)
      return kernel.emitError("cuTile resident binding requires a positive compute-unit count"),
             failure();
    Builder builder(kernel.getContext());
    auto expression = [&](gpu::PhysicalExprKind kind, int64_t value,
                          Attribute symbol, ArrayRef<Attribute> operands) {
      return gpu::PhysicalExprAttr::get(kernel.getContext(), kind, value,
                                       symbol, builder.getArrayAttr(operands));
    };
    auto parameter = [&](gpu::ParameterAttr declaration) {
      return expression(gpu::PhysicalExprKind::Parameter, 0,
                        declaration.getReference(), {});
    };
    auto computeUnits = expression(gpu::PhysicalExprKind::Constant,
        capabilities.getComputeUnits(), builder.getStringAttr(""), {});
    auto clusters = expression(gpu::PhysicalExprKind::FloorDiv, 0,
        builder.getStringAttr(""), {computeUnits, parameter(ctas)});
    auto capacity = expression(gpu::PhysicalExprKind::Multiply, 0,
        builder.getStringAttr(""), {clusters, parameter(occupancy)});
    requirements.push_back(gpu::ConfigurationRequirementAttr::get(
        kernel.getContext(), gpu::ConfigurationRequirementKind::Legality,
        gpu::ConfigurationRequirementMetric::ResidentWorkers,
        gpu::ConfigurationRequirementPredicate::Equal, parameter(resident),
        capacity, gpu::ParameterRefAttr(),
        builder.getStringAttr("cuTile resident workers must match the CTA and occupancy binding")));
  }
  return requirements;
}

} // namespace

const gpu::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {"value"};
  static const gpu::TuningProfileSchema schema{"cutile", columns};
  return schema;
}

bool isLegalAccessForm(int64_t value) {
  return value == nativeAccessForm || value == gatherAccessForm ||
         value == nativeNoTMAForm;
}

bool isLegalOccupancy(int64_t value) { return value >= 1 && value <= 32; }

bool isLegalWorkerWarps(int64_t value) {
  return value == inferredWorkerWarps || value == 4 || value == 8;
}

bool isLegalCTAs(int64_t value) {
  return value >= 1 && value <= 16 && llvm::isPowerOf2_64(value);
}

bool isCuTileProviderRole(gpu::ParameterRole role) {
  return role == gpu::ParameterRole::ProviderAccessForm ||
         role == gpu::ParameterRole::ProviderOccupancy ||
         role == gpu::ParameterRole::ProviderLoadPolicy ||
         role == gpu::ParameterRole::ProviderWarps ||
         role == gpu::ParameterRole::ProviderCTAs;
}

FailureOr<gpu::ParameterRefAttr> declareProviderParameter(
    func::FuncOp kernel, const gpu::TuningProfiles &profiles, StringRef family,
    StringRef name, gpu::ParameterRole role, bool (*isLegal)(int64_t)) {
  auto rows = profiles.get(tuningProfileSchema(), family, kernel.getLoc());
  if (failed(rows))
    return failure();
  SmallVector<int64_t> candidates;
  for (const auto &row : *rows)
    if (isLegal(row.front()))
      candidates.push_back(row.front());
  if (candidates.empty())
    return kernel.emitError("cuTile tuning profile has no legal hints for ") << name;
  return gpu::getOrCreatePhysicalParameter(
      kernel, name, role, gpu::ParameterCategory::Provider, 0, candidates);
}

LogicalResult materializeClosedConfigs(func::FuncOp kernel) {
  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  for (gpu::ParameterAttr domain : space->declarations())
    if (!domain.isExtent() ||
        (domain.getPhase() == gpu::ConfigurationBindingPhase::Provider) !=
            isCuTileProviderRole(domain.getRole()))
      return kernel.emitError("cuTile program contains a foreign provider parameter");
  auto shared = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(shared))
    return failure();
  Builder builder(kernel.getContext());
  SmallVector<SmallVector<NamedAttribute>> configurations;
  for (DictionaryAttr tuple : *shared)
    configurations.emplace_back(tuple.getValue());
  SmallVector<NamedAttribute> launchBaseline;
  SmallVector<NamedAttribute> launchEndpoint;
  SmallVector<gpu::ParameterAttr> launchOptions;
  gpu::ParameterAttr accessForm;
  gpu::ParameterAttr loadPolicy;
  for (gpu::ParameterAttr definition : space->declarations()) {
    if (definition.getPhase() != gpu::ConfigurationBindingPhase::Provider)
      continue;
    if (definition.getRole() ==
        gpu::ParameterRole::ProviderAccessForm) {
      accessForm = definition;
      continue;
    }
    if (definition.getRole() ==
        gpu::ParameterRole::ProviderLoadPolicy) {
      loadPolicy = definition;
      continue;
    }
    auto candidates = definition.getCandidates().asArrayRef();
    launchOptions.push_back(definition);
    launchBaseline.push_back(builder.getNamedAttr(
        definition.getName(), builder.getI64IntegerAttr(candidates.front())));
    launchEndpoint.push_back(builder.getNamedAttr(
        definition.getName(), builder.getI64IntegerAttr(candidates.back())));
  }

  // Launch hints are correlated candidates, not another Cartesian search over
  // every shared tile. Retain each declared value and the joint endpoint,
  // including the lower compiler's inferred worker count.
  SmallVector<SmallVector<NamedAttribute>> launchConfigurations{launchBaseline};
  for (gpu::ParameterAttr definition : launchOptions) {
    for (int64_t candidate : definition.getCandidates().asArrayRef().drop_front()) {
      auto bindings = launchBaseline;
      for (NamedAttribute &binding : bindings)
        if (binding.getName() == definition.getName())
          binding = builder.getNamedAttr(
              definition.getName(), builder.getI64IntegerAttr(candidate));
      launchConfigurations.push_back(std::move(bindings));
    }
  }
  if (!llvm::is_contained(launchConfigurations, launchEndpoint))
    launchConfigurations.push_back(std::move(launchEndpoint));

  SmallVector<SmallVector<NamedAttribute>> memoryConfigurations(1);
  for (gpu::ParameterAttr definition : {loadPolicy, accessForm}) {
    if (!definition)
      continue;
    auto forms = definition.getCandidates().asArrayRef();
    SmallVector<SmallVector<NamedAttribute>> expanded;
    for (const auto &base : memoryConfigurations)
      for (int64_t form : forms) {
        auto configuration = base;
        configuration.push_back(builder.getNamedAttr(
            definition.getName(), builder.getI64IntegerAttr(form)));
        expanded.push_back(std::move(configuration));
      }
    memoryConfigurations = std::move(expanded);
  }

  SmallVector<SmallVector<NamedAttribute>> providerConfigurations;
  for (auto configuration : launchConfigurations) {
    configuration.append(memoryConfigurations.front());
    providerConfigurations.push_back(std::move(configuration));
  }
  // Preserve access-form/load-latency interactions at the launch baseline.
  for (const auto &memory : llvm::drop_begin(memoryConfigurations)) {
    auto configuration = launchBaseline;
    configuration.append(memory);
    providerConfigurations.push_back(std::move(configuration));
  }
  SmallVector<SmallVector<NamedAttribute>> expanded;
  for (const auto &base : configurations)
    for (const auto &provider : providerConfigurations) {
      auto configuration = base;
      configuration.append(provider);
      expanded.push_back(std::move(configuration));
    }
  configurations = std::move(expanded);

  auto resident = space->find(gpu::ParameterRole::ResidentWorkers);
  auto ctas = space->find(gpu::ParameterRole::ProviderCTAs);
  auto occupancy = space->find(gpu::ParameterRole::ProviderOccupancy);
  bool bindResidentCapacity = resident && ctas && occupancy;
  if (bindResidentCapacity) {
    auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
    if (!capabilities || capabilities.getComputeUnits() <= 0)
      return kernel.emitError("cuTile resident binding requires a positive compute-unit count");
    SmallVector<int64_t> counts;
    for (auto &configuration : configurations) {
      NamedAttrList bindings(configuration);
      int64_t cluster = cast<IntegerAttr>(bindings.get(ctas.getName())).getInt();
      int64_t capacity = cast<IntegerAttr>(bindings.get(occupancy.getName())).getInt();
      if (!isLegalCTAs(cluster) || !isLegalOccupancy(capacity))
        return kernel.emitError("cuTile resident binding requires legal CTA and occupancy options");
      int64_t count;
      if (llvm::MulOverflow(capabilities.getComputeUnits() / cluster, capacity, count) || count <= 0)
        return kernel.emitError("cuTile resident capacity is not a positive representable count");
      bindings.set(resident.getName(), builder.getI64IntegerAttr(count));
      configuration.assign(bindings.begin(), bindings.end());
      if (!llvm::is_contained(counts, count))
        counts.push_back(count);
    }
    llvm::sort(counts);
    if (failed(gpu::updateParameter(kernel, resident.withCandidates(
            DenseI64ArrayAttr::get(kernel.getContext(), counts)))))
      return failure();
  }

  SmallVector<DictionaryAttr> encoded;
  for (const auto &bindings : configurations) {
    DictionaryAttr candidate = builder.getDictionaryAttr(bindings);
    if (!llvm::is_contained(encoded, candidate))
      encoded.push_back(candidate);
  }
  if (encoded.empty())
    return kernel.emitError("cuTile legalization produced no provider config");
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  auto accepted = gpu::filterConfigurationRequirements(kernel, encoded, *requirements);
  if (failed(accepted))
    return failure();
  return gpu::writeConfigurations(kernel, *accepted,
                                 gpu::ConfigurationStage::Complete, *requirements);
}

LogicalResult finalizeConfigurationRequirements(func::FuncOp kernel) {
  auto parameters = gpu::ParameterSpace::read(kernel);
  if (failed(parameters))
    return failure();
  auto rows = parameters->configurations(gpu::ConfigurationStage::Complete);
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(rows) || failed(requirements))
    return failure();
  auto accepted = gpu::filterConfigurationRequirements(kernel, *rows, *requirements);
  if (failed(accepted))
    return failure();
  return gpu::writeConfigurations(kernel, *accepted,
                                 gpu::ConfigurationStage::Complete, *requirements);
}

LogicalResult verifyClosedConfigs(func::FuncOp kernel) {
  auto requirements = collectConfigurationRequirements(kernel);
  if (failed(requirements))
    return failure();
  return gpu::verifyConfigurationRequirements(kernel, *requirements);
}

} // namespace intent::cutile

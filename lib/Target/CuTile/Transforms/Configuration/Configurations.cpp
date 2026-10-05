#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"

#include "Configurations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/Resources.h"
#include "Intent/Target/CuTile/Analysis/Program.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Target/CuTile/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
namespace intent::cutile {
const gpu::TuningProfileSchema &tuningProfileSchema() {
  static const StringRef columns[] = {"value"};
  static const gpu::TuningProfileSchema schema{"cutile", columns};
  return schema;
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

namespace {

bool containsFragment(Type type) {
  if (isa<gpu::FragmentType>(type))
    return true;
  auto record = dyn_cast<gpu::RecordType>(type);
  return record && llvm::any_of(record.getFieldTypes(), [](Attribute field) {
           return containsFragment(cast<TypeAttr>(field).getValue());
         });
}

bool hasLoopCarriedFragment(func::FuncOp kernel) {
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    found |= llvm::any_of(loop.getInitArgs(), [](Value value) {
      return containsFragment(value.getType());
    });
  });
  kernel.walk([&](scf::WhileOp loop) {
    found |= llvm::any_of(loop.getInits(), [](Value value) {
      return containsFragment(value.getType());
    });
  });
  return found;
}

StringRef occupancyProfileFamily(func::FuncOp kernel,
                                  gpu::CapabilitiesAttr capabilities) {
  for (Attribute attribute : gpu::getParameterDeclarations(kernel))
    if (cast<gpu::ParameterAttr>(attribute).getRole() ==
        gpu::ParameterRole::ResidentWorkers)
      return "occupancy_persistent";
  if (hasLoopCarriedFragment(kernel))
    return "occupancy_loop";
  return capabilities.getComputeCapabilityMajor() < 9
             ? "occupancy_legacy" : "occupancy_modern";
}

SmallVector<SmallVector<NamedAttribute>>
launchConfigurations(const gpu::ParameterSpace &space, Builder &builder) {
  SmallVector<NamedAttribute> launchBaseline;
  SmallVector<NamedAttribute> launchEndpoint;
  SmallVector<gpu::ParameterAttr> launchOptions;
  for (gpu::ParameterAttr definition : space.declarations()) {
    if (definition.getPhase() != gpu::ConfigurationBindingPhase::Provider)
      continue;
    if (definition.getRole() == gpu::ParameterRole::ProviderAccessForm ||
        definition.getRole() == gpu::ParameterRole::ProviderLoadPolicy)
      continue;
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
  SmallVector<SmallVector<NamedAttribute>> configurations{launchBaseline};
  for (gpu::ParameterAttr definition : launchOptions) {
    for (int64_t candidate : definition.getCandidates().asArrayRef().drop_front()) {
      auto bindings = launchBaseline;
      for (NamedAttribute &binding : bindings)
        if (binding.getName() == definition.getName())
          binding = builder.getNamedAttr(
              definition.getName(), builder.getI64IntegerAttr(candidate));
      configurations.push_back(std::move(bindings));
    }
  }
  if (!llvm::is_contained(configurations, launchEndpoint))
    configurations.push_back(std::move(launchEndpoint));
  return configurations;
}

} // namespace

LogicalResult prepareLaunchConfigurations(
    func::FuncOp kernel, const gpu::TuningProfiles &profiles) {
  NativeProgramFeatures features = queryNativeProgramFeatures(kernel);
  auto capabilities =
      kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, profiles, occupancyProfileFamily(kernel, capabilities),
          occupancyParameter, gpu::ParameterRole::ProviderOccupancy,
          isLegalOccupancy)))
    return failure();
  if (features.matrixCompute &&
      failed(declareProviderParameter(
          kernel, profiles, "ctas", ctasParameter,
          gpu::ParameterRole::ProviderCTAs, isLegalCTAs)))
    return failure();
  if (features.occupancySensitive &&
      failed(declareProviderParameter(
          kernel, profiles, "worker_warps", workerWarpsParameter,
          gpu::ParameterRole::ProviderWarps, isLegalWorkerWarps)))
    return failure();

  auto space = gpu::ParameterSpace::read(kernel);
  if (failed(space)) return failure();
  auto shared = space->configurations(gpu::ConfigurationStage::Shared);
  if (failed(shared)) return failure();
  auto resident = space->find(gpu::ParameterRole::ResidentWorkers);
  auto ctas = space->find(gpu::ParameterRole::ProviderCTAs);
  auto occupancy = space->find(gpu::ParameterRole::ProviderOccupancy);
  if (!resident || !ctas || !occupancy) return success();
  if (!capabilities || capabilities.getComputeUnits() <= 0)
    return kernel.emitError("cuTile resident binding requires a positive compute-unit count");

  Builder builder(kernel.getContext());
  SmallVector<int64_t> counts;
  for (const auto &configuration : launchConfigurations(*space, builder)) {
    NamedAttrList bindings(configuration);
    int64_t cluster = cast<IntegerAttr>(bindings.get(ctas.getName())).getInt();
    int64_t capacity = cast<IntegerAttr>(bindings.get(occupancy.getName())).getInt();
    if (!isLegalCTAs(cluster) || !isLegalOccupancy(capacity))
      return kernel.emitError("cuTile resident binding requires legal CTA and occupancy options");
    int64_t count;
    if (llvm::MulOverflow(capabilities.getComputeUnits() / cluster, capacity, count) || count <= 0)
      return kernel.emitError("cuTile resident capacity is not a positive representable count");
    if (!llvm::is_contained(counts, count)) counts.push_back(count);
  }
  llvm::sort(counts);
  SmallVector<DictionaryAttr> projected;
  for (DictionaryAttr tuple : *shared)
    for (int64_t count : counts) {
      NamedAttrList bindings(tuple);
      bindings.set(resident.getName(), builder.getI64IntegerAttr(count));
      auto row = bindings.getDictionary(kernel.getContext());
      if (!llvm::is_contained(projected, row)) projected.push_back(row);
    }
  // Publish the actual shared binding domain before any allocation envelope is
  // formed. The complete rows later restore launch correlation through the
  // resident/CTA/occupancy requirement; they never rebind this value.
  if (failed(gpu::updateParameter(kernel, resident.withCandidates(
          DenseI64ArrayAttr::get(kernel.getContext(), counts)))))
    return failure();
  return gpu::writeConfigurations(kernel, projected,
                                   gpu::ConfigurationStage::Shared);
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
  auto resident = space->find(gpu::ParameterRole::ResidentWorkers);
  if (!space->find(gpu::ParameterRole::ProviderCTAs) ||
      !space->find(gpu::ParameterRole::ProviderOccupancy))
    resident = {};
  llvm::MapVector<DictionaryAttr, SmallVector<DictionaryAttr>> sharedGroups;
  for (DictionaryAttr tuple : *shared) {
    NamedAttrList identity(tuple);
    if (resident) identity.erase(resident.getName());
    sharedGroups[identity.getDictionary(kernel.getContext())].push_back(tuple);
  }
  auto launches = launchConfigurations(*space, builder);
  auto accessForm = space->find(gpu::ParameterRole::ProviderAccessForm);
  auto loadPolicy = space->find(gpu::ParameterRole::ProviderLoadPolicy);

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
  for (const auto &memory : memoryConfigurations) {
    ArrayRef<SmallVector<NamedAttribute>> selectedLaunches(launches);
    // Every access form retains the correlated launch portfolio. An additional
    // legal form must not remove launch choices from an existing primitive.
    // Nonbaseline load policies still vary only at the launch baseline.
    if (loadPolicy &&
        cast<IntegerAttr>(NamedAttrList(memory).get(loadPolicy.getName())).getInt() !=
            loadPolicy.getCandidates().asArrayRef().front())
      selectedLaunches = selectedLaunches.take_front();
    for (const auto &launch : selectedLaunches) {
      auto configuration = launch;
      configuration.append(memory);
      providerConfigurations.push_back(std::move(configuration));
    }
  }
  // Group only the resident alternatives introduced by preparation. Keep the
  // original shared-tuple/launch order after the equality requirement selects
  // the matching resident alternative below.
  SmallVector<DictionaryAttr> encoded;
  for (const auto &group : sharedGroups)
    for (const auto &provider : providerConfigurations) {
      for (DictionaryAttr base : group.second) {
        NamedAttrList bindings(base);
        bindings.append(provider);
        DictionaryAttr candidate = bindings.getDictionary(kernel.getContext());
        if (!llvm::is_contained(encoded, candidate)) encoded.push_back(candidate);
      }
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

} // namespace intent::cutile

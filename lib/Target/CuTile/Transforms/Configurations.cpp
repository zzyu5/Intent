#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"

#include "Configurations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
namespace intent::cutile {
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

FailureOr<gpu::ParameterOp> declareProviderParameter(
    func::FuncOp kernel, const gpu::TuningProfiles &profiles, StringRef family,
    StringRef name, gpu::ParameterRole role, bool (*isLegal)(int64_t)) {
  auto rows = profiles.get("cutile", family, kernel.getLoc());
  if (failed(rows))
    return failure();
  SmallVector<int64_t> candidates;
  for (const auto &row : *rows)
    if (isLegal(row.front()))
      candidates.push_back(row.front());
  if (candidates.empty())
    return kernel.emitError("cuTile tuning profile has no legal hints for ") << name;
  bool nameCollision = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    nameCollision |= parameter.getParameter().getName().getValue() == name;
  });
  if (nameCollision)
    return kernel.emitError("cuTile hint parameter name is already owned: ") << name;
  OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = gpu::ParameterAttr::get(
      kernel.getContext(), entry.getStringAttr(name), static_cast<uint32_t>(role),
      static_cast<uint32_t>(gpu::ParameterCategory::Provider),
      /*elementBitWidth=*/0, DenseI64ArrayAttr::get(kernel.getContext(), candidates));
  return entry.create<gpu::ParameterOp>(kernel.getLoc(), entry.getIndexType(), schema);
}

LogicalResult materializeClosedConfigs(func::FuncOp kernel) {
  auto space = gpu::PhysicalParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  for (const auto &domain : space->domains())
    if (domain.provider != isCuTileProviderRole(domain.role()))
      return domain.operation->emitOpError("cuTile program contains a foreign provider parameter");
  auto shared = space->sharedConfigurations();
  if (failed(shared))
    return failure();
  Builder builder(kernel.getContext());
  SmallVector<SmallVector<NamedAttribute>> configurations;
  for (DictionaryAttr tuple : *shared)
    configurations.emplace_back(tuple.getValue());
  SmallVector<StringAttr> sharedNames;
  for (const auto &domain : space->domains())
    if (!domain.provider && !domain.coverage)
      sharedNames.push_back(domain.name());

  SmallVector<NamedAttribute> launchBaseline;
  SmallVector<NamedAttribute> launchEndpoint;
  SmallVector<gpu::ParameterOp> launchOptions;
  gpu::ParameterOp accessForm;
  gpu::ParameterOp loadPolicy;
  for (const auto &domain : space->domains()) {
    if (!domain.provider)
      continue;
    gpu::ParameterAttr definition = domain.definition;
    if (definition.getRole() ==
        static_cast<uint32_t>(gpu::ParameterRole::ProviderAccessForm)) {
      accessForm = domain.operation;
      continue;
    }
    if (definition.getRole() ==
        static_cast<uint32_t>(gpu::ParameterRole::ProviderLoadPolicy)) {
      loadPolicy = domain.operation;
      continue;
    }
    auto candidates = definition.getCandidates().asArrayRef();
    launchOptions.push_back(domain.operation);
    launchBaseline.push_back(builder.getNamedAttr(
        definition.getName(), builder.getI64IntegerAttr(candidates.front())));
    launchEndpoint.push_back(builder.getNamedAttr(
        definition.getName(), builder.getI64IntegerAttr(candidates.back())));
  }

  // Launch hints are correlated candidates, not another Cartesian search over
  // every shared tile. Retain each declared value and the joint endpoint,
  // including the lower compiler's inferred worker count.
  SmallVector<SmallVector<NamedAttribute>> launchConfigurations{launchBaseline};
  for (gpu::ParameterOp option : launchOptions) {
    auto definition = option.getParameter();
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
  for (gpu::ParameterOp option : {loadPolicy, accessForm}) {
    if (!option)
      continue;
    auto definition = option.getParameter();
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

  gpu::ParameterOp resident;
  gpu::ParameterOp ctas;
  gpu::ParameterOp occupancy;
  for (const auto &domain : space->domains()) {
    auto role = domain.role();
    if (role == gpu::ParameterRole::ResidentWorkers)
      resident = domain.operation;
    if (role == gpu::ParameterRole::ProviderCTAs)
      ctas = domain.operation;
    if (role == gpu::ParameterRole::ProviderOccupancy)
      occupancy = domain.operation;
  }
  bool bindResidentCapacity = resident && ctas && occupancy;
  if (bindResidentCapacity) {
    auto capabilities = kernel->getAttrOfType<gpu::CapabilitiesAttr>(gpu::capabilitiesAttr);
    if (!capabilities || capabilities.getComputeUnits() <= 0)
      return kernel.emitError("cuTile resident binding requires a positive compute-unit count");
    SmallVector<int64_t> counts;
    auto definition = resident.getParameter();
    for (auto &configuration : configurations) {
      NamedAttrList bindings(configuration);
      int64_t cluster = cast<IntegerAttr>(bindings.get(ctas.getParameter().getName())).getInt();
      int64_t capacity = cast<IntegerAttr>(bindings.get(occupancy.getParameter().getName())).getInt();
      if (!isLegalCTAs(cluster) || !isLegalOccupancy(capacity))
        return kernel.emitError("cuTile resident binding requires legal CTA and occupancy options");
      int64_t count;
      if (llvm::MulOverflow(capabilities.getComputeUnits() / cluster, capacity, count) || count <= 0)
        return kernel.emitError("cuTile resident capacity is not a positive representable count");
      bindings.set(definition.getName(), builder.getI64IntegerAttr(count));
      configuration.assign(bindings.begin(), bindings.end());
      if (!llvm::is_contained(counts, count))
        counts.push_back(count);
    }
    llvm::sort(counts);
    resident.setParameterAttr(gpu::ParameterAttr::get(
        kernel.getContext(), definition.getName(), definition.getRole(),
        definition.getCategory(), definition.getElementBitWidth(),
        DenseI64ArrayAttr::get(kernel.getContext(), counts)));
  }

  SmallVector<Attribute> encoded;
  for (const auto &bindings : configurations) {
    DictionaryAttr candidate = builder.getDictionaryAttr(bindings);
    if (!llvm::is_contained(encoded, Attribute(candidate)))
      encoded.push_back(candidate);
  }
  if (encoded.empty())
    return kernel.emitError("cuTile legalization produced no provider config");
  kernel->setAttr(gpu::cuTileConfigsAttr, builder.getArrayAttr(encoded));
  if (bindResidentCapacity) {
    SmallVector<Attribute> projected;
    for (Attribute attribute : encoded) {
      auto candidate = cast<DictionaryAttr>(attribute);
      SmallVector<NamedAttribute> bindings;
      for (StringAttr name : sharedNames)
        bindings.push_back(builder.getNamedAttr(name, candidate.get(name)));
      auto tuple = builder.getDictionaryAttr(bindings);
      if (!llvm::is_contained(projected, Attribute(tuple)))
        projected.push_back(tuple);
    }
    kernel->setAttr(gpu::sharedConfigTuplesAttr, builder.getArrayAttr(projected));
  }
  return success();
}

LogicalResult verifyClosedConfigs(func::FuncOp kernel) {
  auto space = gpu::PhysicalParameterSpace::read(kernel);
  if (failed(space))
    return failure();
  auto encoded = kernel->getAttrOfType<ArrayAttr>(gpu::cuTileConfigsAttr);
  if (!encoded || encoded.empty())
    return kernel.emitError(
        "cuTile legalization did not materialize closed provider configs");
  llvm::SmallDenseSet<Attribute, 8> unique;
  for (Attribute attribute : encoded) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (failed(space->verifyBindings(tuple, gpu::ParameterBindingScope::Complete)))
      return failure();
    if (!unique.insert(attribute).second)
      return kernel.emitError("contains a duplicate cuTile provider config");
  }
  return success();
}

} // namespace intent::cutile

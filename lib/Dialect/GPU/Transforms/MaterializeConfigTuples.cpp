#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <limits>
using namespace mlir;

namespace intent::gpu {
namespace {

struct TuningProfile {
  int64_t ownershipM;
  int64_t ownershipN;
  int64_t reduction;
  int64_t scan;
  int64_t traversalWorkers;
  int64_t traversalGroup;
};

bool isProviderRole(ParameterRole role) {
  return role == ParameterRole::ProviderWarps ||
         role == ParameterRole::ProviderStages ||
         role == ParameterRole::ProviderCTAs ||
         role == ParameterRole::ProviderThreads;
}

bool isSharedStaticParameter(ParameterOp parameter) {
  auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
  auto category = static_cast<ParameterCategory>(
      parameter.getParameter().getCategory());
  return !isProviderRole(role) && category != ParameterCategory::Coverage &&
         !parameter->hasAttr(coverageDimensionAttr);
}

SmallVector<TuningProfile, 3>
profilesFor(func::FuncOp kernel, ParameterCategory category, unsigned width) {
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool matrix = capabilities && capabilities.getMatrixUnits();
  bool narrow = width <= 16;
  if (category == ParameterCategory::Contraction && matrix && narrow)
    return {{128, 128, 32, 128, 1, 8},
            {64, 128, 64, 128, 1, 8},
            {128, 64, 32, 128, 1, 8}};
  if (category == ParameterCategory::Contraction)
    return {{64, 64, 32, 128, 1, 8},
            {32, 64, 64, 128, 1, 8},
            {64, 32, 32, 128, 1, 8}};
  if (category == ParameterCategory::Scan)
    return {{128, 256, 64, 256, 1, 8},
            {64, 128, 64, 128, 1, 8},
            {256, 512, 32, 512, 1, 8}};
  if (category == ParameterCategory::Reduction)
    return {{128, 128, 64, 128, 1, 8},
            {64, 128, 128, 128, 1, 8},
            {256, 64, 32, 128, 1, 8}};
  if (category == ParameterCategory::Execution)
    return {{1, 1, 1, 1, 1, 8},
            {1, 1, 1, 1, 2, 4},
            {1, 1, 1, 1, 4, 2}};
  int64_t lane = narrow ? 256 : 128;
  return {{lane, lane, 32, 128, 1, 8},
          {lane / 2, lane / 2, 32, 128, 1, 8},
          {64, 64, 32, 128, 1, 8}};
}

int64_t requestedValue(const TuningProfile &profile, ParameterRole role) {
  switch (role) {
  case ParameterRole::OwnershipM:
    return profile.ownershipM;
  case ParameterRole::OwnershipN:
    return profile.ownershipN;
  case ParameterRole::Reduction:
    return profile.reduction;
  case ParameterRole::ScanChunk:
    return profile.scan;
  case ParameterRole::TraversalWorkers:
    return profile.traversalWorkers;
  case ParameterRole::TraversalGroup:
    return profile.traversalGroup;
  case ParameterRole::ResidentWorkers:
    return std::numeric_limits<int64_t>::max();
  case ParameterRole::FullCoverage:
  case ParameterRole::ProviderWarps:
  case ParameterRole::ProviderStages:
  case ParameterRole::ProviderCTAs:
  case ParameterRole::ProviderThreads:
    llvm_unreachable("non-static parameter entered shared tuning table");
  }
  llvm_unreachable("unknown physical parameter role");
}

int64_t selectCandidate(ArrayRef<int64_t> candidates, int64_t requested) {
  int64_t selected = *std::min_element(candidates.begin(), candidates.end());
  for (int64_t candidate : candidates) {
    if (candidate == requested)
      return candidate;
    if (candidate <= requested && candidate > selected)
      selected = candidate;
  }
  return selected;
}

} // namespace

LogicalResult materializeSharedConfigTuples(func::FuncOp kernel) {
  SmallVector<ParameterOp> parameters;
  bool invalidParameter = false;
  kernel.walk([&](ParameterOp parameter) {
    ParameterAttr schema = parameter.getParameter();
    auto role = static_cast<ParameterRole>(schema.getRole());
    auto category =
        static_cast<ParameterCategory>(schema.getCategory());
    const bool providerRole = isProviderRole(role);
    const bool coverage = parameter->hasAttr(coverageDimensionAttr);
    if (providerRole != (category == ParameterCategory::Provider)) {
      parameter.emitOpError(
          "provider parameter role and tuning category disagree");
      invalidParameter = true;
      return;
    }
    if ((role == ParameterRole::FullCoverage ||
         category == ParameterCategory::Coverage) &&
        (!coverage || category != ParameterCategory::Coverage)) {
      parameter.emitOpError(
          "full-coverage parameter lacks its typed coverage category or dimension");
      invalidParameter = true;
      return;
    }
    if (isSharedStaticParameter(parameter))
      parameters.push_back(parameter);
  });
  if (invalidParameter)
    return failure();
  Builder builder(kernel.getContext());
  SmallVector<Attribute> tuples;
  for (unsigned profileIndex = 0; profileIndex < 3; ++profileIndex) {
    SmallVector<NamedAttribute> bindings;
    for (ParameterOp parameter : parameters) {
      ParameterAttr schema = parameter.getParameter();
      auto role = static_cast<ParameterRole>(schema.getRole());
      auto category =
          static_cast<ParameterCategory>(schema.getCategory());
      SmallVector<TuningProfile, 3> profiles = profilesFor(
          kernel, category, schema.getElementBitWidth());
      int64_t selected = selectCandidate(schema.getCandidates().asArrayRef(),
                                         requestedValue(profiles[profileIndex], role));
      bindings.push_back(builder.getNamedAttr(
          schema.getName(), builder.getI64IntegerAttr(selected)));
    }
    DictionaryAttr tuple = builder.getDictionaryAttr(bindings);
    if (!llvm::is_contained(tuples, Attribute(tuple)))
      tuples.push_back(tuple);
  }
  if (tuples.empty())
    tuples.push_back(builder.getDictionaryAttr({}));
  kernel->setAttr(sharedConfigTuplesAttr, builder.getArrayAttr(tuples));
  return success();
}

LogicalResult verifySharedConfigTuples(func::FuncOp kernel) {
  auto tuples = kernel->getAttrOfType<ArrayAttr>(sharedConfigTuplesAttr);
  if (!tuples || tuples.empty())
    return kernel.emitError(
        "shared physical program requires complete config tuples");
  llvm::StringMap<ParameterOp> parameters;
  kernel.walk([&](ParameterOp parameter) {
    if (isSharedStaticParameter(parameter))
      parameters.try_emplace(parameter.getParameter().getName().getValue(),
                             parameter);
  });
  llvm::SmallDenseSet<Attribute, 8> unique;
  for (Attribute attribute : tuples) {
    auto tuple = dyn_cast<DictionaryAttr>(attribute);
    if (!tuple || tuple.size() != parameters.size())
      return kernel.emitError(
          "shared config tuple does not bind every static physical parameter");
    if (!unique.insert(attribute).second)
      return kernel.emitError("shared config tuple is duplicated");
    for (NamedAttribute binding : tuple) {
      auto found = parameters.find(binding.getName().getValue());
      auto value = dyn_cast<IntegerAttr>(binding.getValue());
      if (found == parameters.end() || !value || value.getInt() <= 0)
        return kernel.emitError(
            "shared config tuple has an unknown or invalid binding");
      ArrayRef<int64_t> candidates =
          found->second.getParameter().getCandidates().asArrayRef();
      if (!llvm::is_contained(candidates, value.getInt()))
        return found->second.emitOpError(
            "shared config tuple value is outside the typed domain");
    }
  }
  return success();
}

} // namespace intent::gpu

#include "ConfigurationPolicy.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace intent::gpu {

const TuningProfileSchema &sharedTuningProfileSchema() {
  static const StringRef columns[] = {"ownership_m", "ownership_n", "reduction",
      "reduction_outer", "scan", "traversal_workers", "traversal_group"};
  static const TuningProfileSchema schema{"shared", columns};
  return schema;
}

namespace configuration {
namespace {

FailureOr<SmallVector<TuningProfile, 5>>
readProfiles(func::FuncOp kernel, StringRef family, const TuningProfiles &tables) {
  auto rows = tables.get(sharedTuningProfileSchema(), family, kernel.getLoc());
  if (failed(rows))
    return failure();
  SmallVector<TuningProfile, 5> profiles;
  for (const TuningProfiles::Row &row : *rows)
    profiles.push_back({row[0], row[1], row[2], row[3], row[4], row[5], row[6]});
  return profiles;
}

FailureOr<SmallVector<TuningProfile, 5>>
profilesFor(func::FuncOp kernel, TuningClass kind, unsigned width,
            bool twoAxisPointwise, bool fixedPointwiseLocal,
            bool pointwiseOnlyProgram, bool smallRegionRows,
            bool multipleRegionMatrixAccumulators, const TuningProfiles &tables) {
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool matrix = capabilities && capabilities.getMatrixUnits();
  bool narrow = width <= 16;
  StringRef family;
  switch (kind) {
  case TuningClass::PersistentContraction:
    family = matrix && narrow ? "persistent_contraction_narrow" : "persistent_contraction";
    break;
  case TuningClass::Contraction:
    family = matrix && narrow ? "contraction_narrow" : "contraction";
    break;
  case TuningClass::RegionContraction:
    family = smallRegionRows ? "region_contraction_small_rows"
                             : "region_contraction";
    break;
  case TuningClass::RegionReduction: family = "region_reduction"; break;
  case TuningClass::Scan: family = "scan"; break;
  case TuningClass::MultiAxisReduction: family = "multi_axis_reduction"; break;
  case TuningClass::Histogram: family = "histogram"; break;
  case TuningClass::Reduction: family = "reduction"; break;
  case TuningClass::StatefulReduction: family = "stateful_reduction"; break;
  case TuningClass::Execution: family = "execution"; break;
  case TuningClass::OnlineMoment: family = "online_moment"; break;
  case TuningClass::PointwiseReduction:
    family = twoAxisPointwise ? "pointwise_reduction_two_axis" : "pointwise_reduction";
    break;
  case TuningClass::Pointwise:
    if (twoAxisPointwise && fixedPointwiseLocal)
      family = pointwiseOnlyProgram ? "pointwise_only_fixed_local" : "pointwise_fixed_local";
    else if (twoAxisPointwise)
      family = narrow ? "pointwise_two_axis_narrow" : "pointwise_two_axis";
    else
      family = narrow ? "pointwise_narrow" : "pointwise";
    break;
  }
  if (multipleRegionMatrixAccumulators &&
      (kind == TuningClass::Contraction || kind == TuningClass::RegionContraction))
    family = "region_contraction_multi_accumulator";
  return readProfiles(kernel, family, tables);
}

} // namespace

int64_t requestedValue(const TuningProfile &profile, ParameterRole role) {
  switch (role) {
  case ParameterRole::OwnershipM:
    return profile.ownershipM;
  case ParameterRole::OwnershipN:
    return profile.ownershipN;
  case ParameterRole::Reduction:
    return profile.reduction;
  case ParameterRole::ReductionInner:
    return profile.reduction;
  case ParameterRole::ReductionOuter:
    return profile.reductionOuter;
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
  case ParameterRole::ProviderAccessForm:
  case ParameterRole::ProviderOccupancy:
  case ParameterRole::ProviderLoadPolicy:
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

LogicalResult projectConfigurationProfiles(
    func::FuncOp kernel, const ConfigurationFacts &facts,
    const TuningProfiles &tables, ProfileBindingConsumer consume) {
  ArrayRef<ParameterAttr> parameters = facts.parameters;
  Builder builder(kernel.getContext());
  unsigned profileCount = 0;
  llvm::DenseMap<ParameterAttr, SmallVector<TuningProfile, 5>> parameterProfiles;
  for (ParameterAttr parameter : parameters) {
    const auto &classification = facts.classifications.find(parameter)->second;
    auto profiles = profilesFor(
        kernel, classification.kind, classification.width,
        facts.hasTwoAxisPointwiseOwnership, facts.hasFixedPointwiseLocal,
        facts.pointwiseOnlyProgram, facts.smallRegionRows,
        facts.multipleRegionMatrixAccumulators, tables);
    if (failed(profiles))
      return failure();
    profileCount = std::max<unsigned>(profileCount, profiles->size());
    parameterProfiles.try_emplace(parameter, std::move(*profiles));
  }
  if (profileCount == 0)
    profileCount = 1;
  int64_t largestReduction = 0;
  bool hasReductionRows = false;
  for (ParameterAttr parameter : parameters) {
    hasReductionRows |= facts.classifications.find(parameter)->second.reductionRow;
    auto role = parameter.getRole();
    if (role == ParameterRole::Reduction || role == ParameterRole::ReductionInner ||
        role == ParameterRole::ReductionOuter)
      for (const TuningProfile &profile : parameterProfiles.find(parameter)->second)
        largestReduction = std::max(largestReduction, requestedValue(profile, role));
  }
  unsigned rowChoiceCount = 1;
  for (const auto &group : facts.rowGroups)
    rowChoiceCount = std::max<unsigned>(rowChoiceCount, group.second.size());
  unsigned freeAxisChoices = llvm::any_of(facts.freeExtents, [](const auto &group) {
    return group.parameters.size() > 1;
  }) ? 2 : 1;
  auto appendTuple = [&](ProfileLookup profileFor, bool compactRows = false) {
    for (unsigned choice = 0; choice < rowChoiceCount * freeAxisChoices; ++choice) {
      llvm::SmallDenseSet<ParameterAttr> selectedRows;
      for (const auto &group : facts.rowGroups)
        selectedRows.insert(group.second[(choice / freeAxisChoices) % group.second.size()]);
      NamedAttrList bindings;
      for (ParameterAttr parameter : parameters) {
        auto schema = parameter;
        auto role = schema.getRole();
        int64_t requested = requestedValue(profileFor(parameter), role);
        // A one-axis pointwise profile budgets the entire fragment. Static
        // local axes consume that budget even without their own parameters.
        if (auto local = facts.pointwiseLocalMultiplicity.find(parameter);
            local != facts.pointwiseLocalMultiplicity.end())
          requested = std::max<int64_t>(1, requested / local->second);
        if (!selectedRows.contains(parameter) &&
            llvm::is_contained(facts.pointwiseRowAxes, parameter))
          requested = 1;
        if (compactRows && facts.classifications.find(parameter)->second.reductionRow) {
          for (const TuningProfile &profile : parameterProfiles.find(parameter)->second)
            requested = std::min(requested, requestedValue(profile, role));
        } else if (compactRows &&
                   (facts.classifications.find(parameter)->second.pointwiseTraversal ||
                    role == ParameterRole::Reduction ||
                    role == ParameterRole::ReductionInner ||
                    role == ParameterRole::ReductionOuter)) {
          requested = largestReduction;
        }
        int64_t selected = selectCandidate(schema.getCandidates().asArrayRef(), requested);
        bindings.set(schema.getName(), builder.getI64IntegerAttr(selected));
      }
      consume(bindings, profileFor, choice % freeAxisChoices != 0);
    }
  };
  for (unsigned profileIndex = 0; profileIndex < profileCount;
       ++profileIndex) {
    appendTuple([&](ParameterAttr parameter) -> const TuningProfile & {
      const auto &profiles = parameterProfiles.find(parameter)->second;
      unsigned selectedProfile = std::min<unsigned>(profileIndex,
                                                     profiles.size() - 1);
      return profiles[selectedProfile];
    });
  }
  // Independent rows and reduction/traversal chunks consume different axes of the
  // resource budget. Keep one correlated small-row/large-chunk tuple instead
  // of pairing both granularities solely by their profile row number.
  if (!facts.hasContraction && hasReductionRows && largestReduction > 0)
    appendTuple([&](ParameterAttr parameter) -> const TuningProfile & {
      return parameterProfiles.find(parameter)->second.front();
    }, true);
  for (const ReductionProfileParameters &correlated : facts.reductionProfiles) {
    for (auto [index, profile] :
         llvm::enumerate(parameterProfiles.find(correlated.chunk)->second))
      appendTuple([&](ParameterAttr parameter) -> const TuningProfile & {
        if (parameter == correlated.chunk || llvm::is_contained(correlated.rows, parameter))
          return profile;
        const auto &profiles = parameterProfiles.find(parameter)->second;
        return profiles[std::min<size_t>(index, profiles.size() - 1)];
      });
  }
  for (const CorrelatedProfileParameters &correlated : facts.correlatedProfiles) {
    const auto &matrixProfiles =
        parameterProfiles.find(correlated.contraction)->second;
    for (auto [index, matrix] : llvm::enumerate(matrixProfiles)) {
      // The consumer reduces the matrix's N axis. Bind its chunk and retained
      // M rows from that same matrix profile, regardless of their tuning class.
      // Zipping independent profile row numbers can omit these legal tiles.
      TuningProfile rows = matrix;
      rows.ownershipN = matrix.ownershipM;
      TuningProfile columns = matrix;
      columns.reduction = matrix.ownershipN;
      appendTuple([&](ParameterAttr parameter) -> const TuningProfile & {
        if (parameter == correlated.pointwise)
          return rows;
        if (parameter == correlated.reduction)
          return columns;
        const auto &profiles = parameterProfiles.find(parameter)->second;
        return profiles[std::min<size_t>(index, profiles.size() - 1)];
      });
    }
  }
  if (!facts.indirectRowGroups.empty()) {
    auto indirectProfiles = readProfiles(kernel, "indirect_row", tables);
    if (failed(indirectProfiles))
      return failure();
    for (const TuningProfile &indirectRowProfile : *indirectProfiles) {
      appendTuple([&](ParameterAttr parameter) -> const TuningProfile & {
        ParameterAttr schema = parameter;
        const auto &profiles = parameterProfiles.find(parameter)->second;
        bool indirectContraction =
            schema.getCategory() ==
                ParameterCategory::Contraction &&
            llvm::is_contained(facts.indirectRowGroups,
                               parameter.getBinding().getGroup());
        return indirectContraction ? indirectRowProfile : profiles.front();
      });
    }
  }
  return success();
}

} // namespace configuration
} // namespace intent::gpu

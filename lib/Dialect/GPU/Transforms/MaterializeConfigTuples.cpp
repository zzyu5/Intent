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
  int64_t reductionOuter;
  int64_t scan;
  int64_t traversalWorkers;
  int64_t traversalGroup;
};

enum class TuningClass {
  Pointwise,
  PointwiseReduction,
  Reduction,
  MultiAxisReduction,
  RegionReduction,
  RegionContraction,
  Scan,
  Contraction,
  PersistentContraction,
  Execution,
};

bool expressionReferencesParameter(PhysicalExprAttr expression,
                                   StringAttr parameter) {
  if (expression.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
      expression.getSymbol() == parameter)
    return true;
  return llvm::any_of(expression.getOperands(), [&](Attribute operand) {
    return expressionReferencesParameter(cast<PhysicalExprAttr>(operand),
                                         parameter);
  });
}

bool isFixedReductionFreeAxis(func::FuncOp kernel, ParameterOp parameter) {
  auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
  if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
    return false;
  StringAttr name = parameter.getParameter().getName();
  bool found = false;
  kernel.walk([&](ReduceOp reduce) {
    if (found || reduce.getAxes().empty() || reduce.getSourceCount() != 1 ||
        reduce.getIdentityCount() != 1 || reduce.getNumResults() != 1)
      return;
    for (Value source :
         reduce.getInputs().take_front(reduce.getSourceCount())) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      bool fixedReduction = llvm::all_of(reduce.getAxes(), [&](int64_t axis) {
        if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
          return false;
        auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
        return extent.getKind() ==
                   static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() > 0;
      });
      if (!fixedReduction)
        continue;
      for (auto [axis, extent] : llvm::enumerate(fragment.getShape())) {
        if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
          continue;
        found |= expressionReferencesParameter(cast<PhysicalExprAttr>(extent),
                                               name);
      }
    }
  });
  return found;
}

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

TuningClass tuningClass(func::FuncOp kernel, ParameterOp parameter) {
  ParameterAttr schema = parameter.getParameter();
  switch (static_cast<ParameterCategory>(schema.getCategory())) {
  case ParameterCategory::Reduction:
    return schema.getRole() ==
                       static_cast<uint32_t>(ParameterRole::ReductionOuter) ||
                   schema.getRole() == static_cast<uint32_t>(
                                           ParameterRole::ReductionInner)
               ? TuningClass::MultiAxisReduction
               : TuningClass::Reduction;
  case ParameterCategory::Scan:
    return TuningClass::Scan;
  case ParameterCategory::Contraction:
    return TuningClass::Contraction;
  case ParameterCategory::PersistentContraction:
    return TuningClass::PersistentContraction;
  case ParameterCategory::RegionReduction:
    return TuningClass::RegionReduction;
  case ParameterCategory::RegionContraction:
    return TuningClass::RegionContraction;
  case ParameterCategory::Execution:
    return TuningClass::Execution;
  case ParameterCategory::Pointwise:
    return isFixedReductionFreeAxis(kernel, parameter)
               ? TuningClass::PointwiseReduction
               : TuningClass::Pointwise;
  case ParameterCategory::Coverage:
  case ParameterCategory::Provider:
    return TuningClass::Pointwise;
  }
  llvm_unreachable("unknown physical parameter category");
}

SmallVector<TuningProfile, 5>
profilesFor(func::FuncOp kernel, TuningClass kind, unsigned width,
            bool twoAxisPointwise, bool fixedPointwiseLocal) {
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  bool matrix = capabilities && capabilities.getMatrixUnits();
  bool narrow = width <= 16;
  if (kind == TuningClass::PersistentContraction && matrix && narrow)
    return {{128, 128, 32, 1, 128, 8, 8},
            {64, 128, 64, 1, 128, 16, 8},
            {128, 64, 32, 1, 128, 8, 8},
            {128, 256, 64, 1, 128, 8, 8}};
  if (kind == TuningClass::PersistentContraction)
    return {{64, 64, 32, 1, 128, 8, 8},
            {32, 64, 64, 1, 128, 16, 8},
            {64, 32, 32, 1, 128, 8, 8}};
  if (kind == TuningClass::Contraction && matrix && narrow)
    return {{128, 128, 32, 1, 128, 1, 8},
            {64, 128, 64, 1, 128, 1, 8},
            {128, 64, 32, 1, 128, 1, 8},
            {128, 256, 64, 1, 128, 1, 8}};
  if (kind == TuningClass::Contraction)
    return {{64, 64, 32, 1, 128, 1, 8},
            {32, 64, 64, 1, 128, 1, 8},
            {64, 32, 32, 1, 128, 1, 8}};
  if (kind == TuningClass::RegionContraction)
    return {{128, 128, 64, 1, 128, 1, 8},
            {64, 128, 64, 1, 64, 1, 8},
            {128, 64, 32, 1, 32, 1, 8},
            {128, 16, 32, 1, 32, 1, 8}};
  if (kind == TuningClass::RegionReduction)
    return {{128, 128, 64, 1, 32768, 1, 8},
            {128, 128, 64, 1, 16384, 1, 8},
            {128, 128, 64, 1, 8192, 1, 8}};
  if (kind == TuningClass::Scan)
    return {{128, 256, 64, 1, 256, 1, 8},
            {64, 128, 64, 1, 128, 1, 8},
            {256, 512, 32, 1, 512, 1, 8}};
  if (kind == TuningClass::MultiAxisReduction)
    return {{128, 128, 512, 32, 128, 1, 8},
            {128, 128, 1024, 16, 128, 1, 8},
            {128, 128, 256, 64, 128, 1, 8}};
  if (kind == TuningClass::Reduction)
    return {{128, 128, 64, 1, 128, 1, 8},
            {64, 128, 128, 1, 128, 1, 8},
            {256, 64, 32, 1, 128, 1, 8}};
  if (kind == TuningClass::Execution)
    return {{1, 1, 1, 1, 1, 1, 8},
            {1, 1, 1, 1, 1, 2, 4},
            {1, 1, 1, 1, 1, 4, 2}};
  if (kind == TuningClass::PointwiseReduction)
    return {{64, 128, 32, 1, 128, 1, 8},
            {64, 64, 32, 1, 128, 1, 8},
            {64, 32, 32, 1, 128, 1, 8},
            {64, 16, 32, 1, 128, 1, 8}};
  if (twoAxisPointwise && fixedPointwiseLocal)
    return {{8, 2, 32, 1, 128, 1, 8},
            {8, 4, 32, 1, 128, 1, 8},
            {4, 4, 32, 1, 128, 1, 8},
            {16, 2, 32, 1, 128, 1, 8}};
  if (twoAxisPointwise)
    return {{1, narrow ? 512 : 256, 32, 1, 128, 1, 8},
            {64, 64, 32, 1, 128, 1, 8},
            {16, 16, 32, 1, 128, 1, 8},
            {8, 16, 32, 1, 128, 1, 8},
            {8, 8, 32, 1, 128, 1, 8},
            {8, 2, 32, 1, 128, 1, 8}};
  int64_t lane = narrow ? 512 : 256;
  return {{64, lane, 32, 1, 128, 1, 8},
          {32, lane / 4, 32, 1, 128, 1, 8},
          {64, std::max<int64_t>(lane / 16, 16), 32, 1, 128, 1, 8},
          {64, narrow ? 4096 : 8192, 32, 1, 128, 1, 8}};
}

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
  bool hasTwoAxisPointwiseOwnership = llvm::any_of(
      parameters, [](ParameterOp parameter) {
        ParameterAttr schema = parameter.getParameter();
        return schema.getCategory() ==
                   static_cast<uint32_t>(ParameterCategory::Pointwise) &&
               schema.getRole() ==
               static_cast<uint32_t>(ParameterRole::OwnershipM);
      });
  bool hasFixedPointwiseLocal = llvm::any_of(
      parameters, [](ParameterOp parameter) {
        ParameterAttr schema = parameter.getParameter();
        return schema.getCategory() ==
                   static_cast<uint32_t>(ParameterCategory::Pointwise) &&
               schema.getCandidates().size() == 1 &&
               parameter->hasAttr(pointwiseLocalAttr);
      });
  unsigned profileCount = 0;
  for (ParameterOp parameter : parameters) {
    ParameterAttr schema = parameter.getParameter();
    profileCount = std::max<unsigned>(
        profileCount,
        profilesFor(kernel, tuningClass(kernel, parameter),
                    schema.getElementBitWidth(),
                    hasTwoAxisPointwiseOwnership, hasFixedPointwiseLocal)
            .size());
  }
  if (profileCount == 0)
    profileCount = 1;
  for (unsigned profileIndex = 0; profileIndex < profileCount; ++profileIndex) {
    SmallVector<NamedAttribute> bindings;
    for (ParameterOp parameter : parameters) {
      ParameterAttr schema = parameter.getParameter();
      auto role = static_cast<ParameterRole>(schema.getRole());
      SmallVector<TuningProfile, 5> profiles = profilesFor(
          kernel, tuningClass(kernel, parameter), schema.getElementBitWidth(),
          hasTwoAxisPointwiseOwnership, hasFixedPointwiseLocal);
      unsigned selectedProfile = std::min<unsigned>(profileIndex,
                                                     profiles.size() - 1);
      int64_t selected = selectCandidate(schema.getCandidates().asArrayRef(),
                                         requestedValue(profiles[selectedProfile], role));
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

#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

struct CorrelatedProfileParameters {
  ParameterOp pointwise;
  ParameterOp reduction;
  ParameterOp contraction;
};

enum class TuningClass {
  Pointwise,
  PointwiseReduction,
  StatefulReduction,
  OnlineMoment,
  Reduction,
  MultiAxisReduction,
  RegionReduction,
  RegionContraction,
  Scan,
  Contraction,
  PersistentContraction,
  Histogram,
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

bool isBlockedReductionFreeAxis(func::FuncOp kernel, ParameterOp parameter) {
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
      bool blockedReduction = llvm::all_of(reduce.getAxes(), [&](int64_t axis) {
        if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
          return false;
        auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
        auto kind = static_cast<PhysicalExprKind>(extent.getKind());
        return (kind == PhysicalExprKind::Constant && extent.getValue() > 0) ||
               kind == PhysicalExprKind::Parameter;
      });
      if (!blockedReduction)
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
         role == ParameterRole::ProviderThreads ||
         role == ParameterRole::ProviderAccessForm ||
         role == ParameterRole::ProviderOccupancy;
}

bool isSharedStaticParameter(ParameterOp parameter) {
  auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
  auto category = static_cast<ParameterCategory>(
      parameter.getParameter().getCategory());
  return !isProviderRole(role) && category != ParameterCategory::Coverage &&
         !parameter->hasAttr(coverageDimensionAttr);
}

bool axisReferencesParameter(FragmentType fragment, int64_t axis,
                             StringAttr parameter) {
  return axis >= 0 && axis < static_cast<int64_t>(fragment.getShape().size()) &&
         expressionReferencesParameter(
             cast<PhysicalExprAttr>(fragment.getShape()[axis]), parameter);
}

bool fragmentReferencesParameter(FragmentType fragment, StringAttr parameter) {
  return llvm::any_of(fragment.getShape(), [&](Attribute extent) {
    return expressionReferencesParameter(cast<PhysicalExprAttr>(extent),
                                         parameter);
  });
}

bool reducedAxesReferenceParameter(FragmentType fragment,
                                   ArrayRef<int64_t> axes,
                                   StringAttr parameter) {
  return llvm::any_of(axes, [&](int64_t axis) {
    return axisReferencesParameter(fragment, axis, parameter);
  });
}

bool freeAxesReferenceParameter(FragmentType fragment, ArrayRef<int64_t> axes,
                                StringAttr parameter) {
  return llvm::any_of(llvm::enumerate(fragment.getShape()), [&](auto indexed) {
    return !llvm::is_contained(axes, static_cast<int64_t>(indexed.index())) &&
           expressionReferencesParameter(
               cast<PhysicalExprAttr>(indexed.value()), parameter);
  });
}

bool valueDependsOn(Value value, Value producer,
                    llvm::DenseSet<Value> &visited) {
  if (value == producer)
    return true;
  if (!visited.insert(value).second)
    return false;

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument.getArgNumber() == 0)
      return false;
    unsigned resultIndex = argument.getArgNumber() - 1;
    if (resultIndex >= loop.getInitArgs().size())
      return false;
    if (valueDependsOn(loop.getInitArgs()[resultIndex], producer, visited))
      return true;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return valueDependsOn(yield.getOperand(resultIndex), producer, visited);
  }

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (auto loop = dyn_cast<scf::ForOp>(definition)) {
    unsigned resultIndex = cast<OpResult>(value).getResultNumber();
    if (resultIndex >= loop.getInitArgs().size())
      return false;
    if (valueDependsOn(loop.getInitArgs()[resultIndex], producer, visited))
      return true;
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return valueDependsOn(yield.getOperand(resultIndex), producer, visited);
  }
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, producer, visited);
  });
}

bool isOnlineMomentOwnership(func::FuncOp kernel, ParameterOp parameter) {
  auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
  if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
    return false;
  StringAttr name = parameter.getParameter().getName();
  bool found = false;
  kernel.walk([&](scf::ForOp loop) {
    if (found || !loop->hasAttr(reductionSourcesAttr) ||
        loop.getNumResults() < 3)
      return;
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getNumResults())
      return;
    SmallVector<unsigned> carryingResults;
    for (auto [index, result] : llvm::enumerate(loop.getResults())) {
      auto fragment = dyn_cast<FragmentType>(result.getType());
      if (fragment && fragmentReferencesParameter(fragment, name))
        carryingResults.push_back(index);
    }
    if (carryingResults.size() != 1)
      return;
    unsigned carry = carryingResults.front();
    loop.getBody()->walk([&](ContractOp contract) {
      auto result = dyn_cast<FragmentType>(contract.getResult().getType());
      if (found || !result || !fragmentReferencesParameter(result, name))
        return;
      llvm::DenseSet<Value> visited;
      found = valueDependsOn(yield.getOperand(carry), contract.getResult(),
                             visited);
    });
  });
  return found;
}

bool isContractionOwnership(func::FuncOp kernel, ParameterOp parameter) {
  auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
  if (role != ParameterRole::OwnershipM && role != ParameterRole::OwnershipN)
    return false;
  StringAttr name = parameter.getParameter().getName();
  bool found = false;
  kernel.walk([&](ContractOp contract) {
    found |= fragmentReferencesParameter(contract.getResult().getType(), name);
  });
  return found;
}

bool isStatefulReduction(func::FuncOp kernel, ParameterOp parameter) {
  auto role = static_cast<ParameterRole>(parameter.getParameter().getRole());
  if (role != ParameterRole::Reduction)
    return false;
  bool found = false;
  // A chunked multi-component reduction carries one coupled accumulator state.
  // Its chunk controls that state directly, so it needs a profile distinct from
  // ordinary scalar reductions even though both use the Reduction role.
  kernel.walk([&](scf::ForOp loop) {
    if (found || loop.getStep() != parameter.getResult() ||
        !loop->hasAttr(reductionSourcesAttr) || loop.getNumResults() < 2)
      return;
    loop.getBody()->walk([&](ReduceOp reduce) {
      found |= reduce.getSourceCount() > 1 &&
               reduce.getNumResults() == loop.getNumResults();
    });
  });
  return found;
}

TuningClass tuningClass(func::FuncOp kernel, ParameterOp parameter) {
  ParameterAttr schema = parameter.getParameter();
  switch (static_cast<ParameterCategory>(schema.getCategory())) {
  case ParameterCategory::Reduction:
    if (isStatefulReduction(kernel, parameter))
      return TuningClass::StatefulReduction;
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
  case ParameterCategory::Histogram:
    return TuningClass::Histogram;
  case ParameterCategory::RegionReduction:
    return TuningClass::RegionReduction;
  case ParameterCategory::RegionContraction:
    return TuningClass::RegionContraction;
  case ParameterCategory::Execution:
    return TuningClass::Execution;
  case ParameterCategory::Pointwise:
    if (isOnlineMomentOwnership(kernel, parameter))
      return TuningClass::OnlineMoment;
    if (isContractionOwnership(kernel, parameter))
      return TuningClass::Contraction;
    return isBlockedReductionFreeAxis(kernel, parameter)
               ? TuningClass::PointwiseReduction
               : TuningClass::Pointwise;
  case ParameterCategory::Coverage:
  case ParameterCategory::Provider:
    return TuningClass::Pointwise;
  }
  llvm_unreachable("unknown physical parameter category");
}

SmallVector<CorrelatedProfileParameters>
correlatedReductionContractionParameters(
    func::FuncOp kernel, ArrayRef<ParameterOp> parameters,
    bool twoAxisPointwise, bool fixedPointwiseLocal) {
  SmallVector<CorrelatedProfileParameters> correlated;
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || !capabilities.getMatrixUnits() || twoAxisPointwise ||
      fixedPointwiseLocal)
    return correlated;

  SmallVector<ParameterOp> pointwise;
  SmallVector<ParameterOp> reductions;
  SmallVector<ParameterOp> contractions;
  SmallVector<ContractOp> contracts;
  for (ParameterOp parameter : parameters) {
    ParameterAttr schema = parameter.getParameter();
    auto role = static_cast<ParameterRole>(schema.getRole());
    if (tuningClass(kernel, parameter) == TuningClass::Pointwise &&
        (role == ParameterRole::OwnershipM ||
         role == ParameterRole::OwnershipN) &&
        schema.getElementBitWidth() > 16)
      pointwise.push_back(parameter);
    if (tuningClass(kernel, parameter) == TuningClass::Reduction &&
        role == ParameterRole::Reduction)
      reductions.push_back(parameter);
    if (tuningClass(kernel, parameter) == TuningClass::Contraction &&
        role == ParameterRole::Reduction &&
        schema.getElementBitWidth() <= 16)
      contractions.push_back(parameter);
  }
  kernel.walk([&](ContractOp contract) { contracts.push_back(contract); });

  kernel.walk([&](ReduceOp reduce) {
    if (reduce.getSourceCount() != 1 || reduce.getNumResults() != 1 ||
        reduce.getAxes().empty())
      return;
    Value source = reduce.getInputs().front();
    auto sourceType = dyn_cast<FragmentType>(source.getType());
    auto resultType = dyn_cast<FragmentType>(reduce.getResult(0).getType());
    if (!sourceType || !resultType)
      return;

    for (ParameterOp pointwiseParameter : pointwise) {
      StringAttr pointwiseName = pointwiseParameter.getParameter().getName();
      if (!freeAxesReferenceParameter(sourceType, reduce.getAxes(),
                                      pointwiseName) ||
          !fragmentReferencesParameter(resultType, pointwiseName))
        continue;
      for (ParameterOp reductionParameter : reductions) {
        StringAttr reductionName = reductionParameter.getParameter().getName();
        if (!reducedAxesReferenceParameter(sourceType, reduce.getAxes(),
                                           reductionName))
          continue;
        for (ContractOp contract : contracts) {
          auto lhsType = dyn_cast<FragmentType>(contract.getLhs().getType());
          auto rhsType = dyn_cast<FragmentType>(contract.getRhs().getType());
          auto contractResultType =
              dyn_cast<FragmentType>(contract.getResult().getType());
          if (!lhsType || !rhsType || !contractResultType ||
              !fragmentReferencesParameter(contractResultType,
                                           pointwiseName) ||
              !fragmentReferencesParameter(contractResultType, reductionName))
            continue;
          llvm::DenseSet<Value> visited;
          if (!valueDependsOn(source, contract.getResult(), visited))
            continue;
          for (ParameterOp contractionParameter : contractions) {
            StringAttr contractionName =
                contractionParameter.getParameter().getName();
            if (!reducedAxesReferenceParameter(
                    lhsType, contract.getLhsReductionAxes(), contractionName) ||
                !reducedAxesReferenceParameter(
                    rhsType, contract.getRhsReductionAxes(), contractionName))
              continue;
            CorrelatedProfileParameters profile{pointwiseParameter,
                                                reductionParameter,
                                                contractionParameter};
            if (!llvm::any_of(correlated, [&](const auto &existing) {
                  return existing.pointwise == profile.pointwise &&
                         existing.reduction == profile.reduction &&
                         existing.contraction == profile.contraction;
                }))
              correlated.push_back(profile);
          }
        }
      }
    }
  });
  return correlated;
}

FailureOr<SmallVector<TuningProfile, 5>>
profilesFor(func::FuncOp kernel, TuningClass kind, unsigned width,
            bool twoAxisPointwise, bool fixedPointwiseLocal,
            bool pointwiseOnlyProgram, bool smallRegionRows,
            const TuningProfiles &tables) {
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
  auto rows = tables.get("shared", family, kernel.getLoc());
  if (failed(rows))
    return failure();
  SmallVector<TuningProfile, 5> profiles;
  for (const TuningProfiles::Row &row : *rows)
    profiles.push_back({row[0], row[1], row[2], row[3], row[4], row[5], row[6]});
  return profiles;
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
  case ParameterRole::ProviderAccessForm:
  case ParameterRole::ProviderOccupancy:
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

bool hasSmallRegionRows(func::FuncOp kernel, ArrayRef<ParameterOp> parameters) {
  bool found = false;
  for (ParameterOp parameter : parameters) {
    ParameterAttr schema = parameter.getParameter();
    if (schema.getCategory() !=
            static_cast<uint32_t>(ParameterCategory::RegionContraction) ||
        schema.getRole() != static_cast<uint32_t>(ParameterRole::OwnershipM))
      continue;
    bool small = false;
    kernel.walk([&](MakeRangeOp range) {
      auto fragment = cast<FragmentType>(range.getResult().getType());
      if (fragment.getShape().size() != 1 ||
          !axisReferencesParameter(fragment, 0, schema.getName()))
        return;
      auto start = range.getLogicalStart().getDefiningOp<arith::ConstantIndexOp>();
      auto stop = range.getLogicalStop().getDefiningOp<arith::ConstantIndexOp>();
      if (start && stop && start.value() == 0 &&
          stop.value() > 0 && stop.value() <= 8)
        small = true;
    });
    if (!small)
      return false;
    found = true;
  }
  return found;
}

} // namespace

LogicalResult materializeSharedConfigTuples(func::FuncOp kernel, const TuningProfiles &tables) {
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
  bool pointwiseOnlyProgram = true;
  kernel.walk([&](Operation *operation) {
    pointwiseOnlyProgram &=
        !isa<ContractOp, ReduceOp, ScanOp, RegionFoldOp, RegionScanOp,
             ScaledContractOp, SparseContractOp, HistogramOp, ScatterReduceOp>(
            operation);
  });
  SmallVector<CorrelatedProfileParameters> correlatedProfiles =
      correlatedReductionContractionParameters(
          kernel, parameters, hasTwoAxisPointwiseOwnership,
          hasFixedPointwiseLocal);
  SmallVector<Attribute> indirectRowGroups;
  auto capabilities =
      kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (capabilities && capabilities.getMatrixUnits())
    for (ParameterOp parameter : parameters) {
      ParameterAttr schema = parameter.getParameter();
      Attribute group = parameter->getAttr(parameterGroupAttr);
      if (schema.getCategory() !=
              static_cast<uint32_t>(ParameterCategory::Contraction) ||
          schema.getRole() !=
              static_cast<uint32_t>(ParameterRole::TraversalWorkers) ||
          !isa_and_nonnull<ArrayAttr>(group) ||
          llvm::is_contained(indirectRowGroups, group))
        continue;
      indirectRowGroups.push_back(group);
    }
  unsigned profileCount = 0;
  bool smallRegionRows = hasSmallRegionRows(kernel, parameters);
  llvm::DenseMap<Operation *, SmallVector<TuningProfile, 5>> parameterProfiles;
  for (ParameterOp parameter : parameters) {
    ParameterAttr schema = parameter.getParameter();
    auto profiles = profilesFor(
        kernel, tuningClass(kernel, parameter), schema.getElementBitWidth(),
        hasTwoAxisPointwiseOwnership, hasFixedPointwiseLocal,
        pointwiseOnlyProgram, smallRegionRows, tables);
    if (failed(profiles))
      return failure();
    profileCount = std::max<unsigned>(profileCount, profiles->size());
    parameterProfiles.try_emplace(parameter, std::move(*profiles));
  }
  if (profileCount == 0)
    profileCount = 1;
  for (unsigned profileIndex = 0; profileIndex < profileCount;
       ++profileIndex) {
    SmallVector<NamedAttribute> bindings;
    for (ParameterOp parameter : parameters) {
      ParameterAttr schema = parameter.getParameter();
      auto role = static_cast<ParameterRole>(schema.getRole());
      const auto &profiles = parameterProfiles.find(parameter)->second;
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
  for (const CorrelatedProfileParameters &correlated : correlatedProfiles) {
    SmallVector<NamedAttribute> bindings;
    for (ParameterOp parameter : parameters) {
      ParameterAttr schema = parameter.getParameter();
      auto role = static_cast<ParameterRole>(schema.getRole());
      const auto &profiles = parameterProfiles.find(parameter)->second;
      const TuningProfile &profile =
          parameter == correlated.reduction ? profiles.back()
                                            : profiles.front();
      int64_t selected = selectCandidate(schema.getCandidates().asArrayRef(),
                                         requestedValue(profile, role));
      bindings.push_back(builder.getNamedAttr(
          schema.getName(), builder.getI64IntegerAttr(selected)));
    }
    DictionaryAttr tuple = builder.getDictionaryAttr(bindings);
    if (!llvm::is_contained(tuples, Attribute(tuple)))
      tuples.push_back(tuple);
  }
  if (!indirectRowGroups.empty()) {
    auto rows = tables.get("shared", "indirect_row", kernel.getLoc());
    if (failed(rows))
      return failure();
    for (const TuningProfiles::Row &row : *rows) {
      const TuningProfile indirectRowProfile{
          row[0], row[1], row[2], row[3], row[4], row[5], row[6]};
      SmallVector<NamedAttribute> bindings;
      for (ParameterOp parameter : parameters) {
        ParameterAttr schema = parameter.getParameter();
        auto role = static_cast<ParameterRole>(schema.getRole());
        const auto &profiles = parameterProfiles.find(parameter)->second;
        bool indirectContraction =
            schema.getCategory() ==
                static_cast<uint32_t>(ParameterCategory::Contraction) &&
            llvm::is_contained(indirectRowGroups,
                               parameter->getAttr(parameterGroupAttr));
        int64_t selected = selectCandidate(
            schema.getCandidates().asArrayRef(),
            requestedValue(indirectContraction ? indirectRowProfile
                                               : profiles.front(),
                           role));
        bindings.push_back(builder.getNamedAttr(
            schema.getName(), builder.getI64IntegerAttr(selected)));
      }
      DictionaryAttr tuple = builder.getDictionaryAttr(bindings);
      if (!llvm::is_contained(tuples, Attribute(tuple)))
        tuples.push_back(tuple);
    }
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

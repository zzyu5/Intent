#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu {

PhysicalExprAttr fragmentElementCount(FragmentType fragment) {
  MLIRContext *context = fragment.getContext();
  auto footprint = PhysicalExprAttr::get(
      context, PhysicalExprKind::Constant,
      1, StringAttr::get(context, ""),
      ArrayAttr::get(context, {}));
  for (Attribute extent : fragment.getShape())
    footprint = PhysicalExprAttr::get(
        context, PhysicalExprKind::Multiply, 0,
        StringAttr::get(context, ""), ArrayAttr::get(context, {footprint, extent}));
  return footprint;
}

PhysicalExprAttr fragmentRegisterFootprint(FragmentType fragment) {
  Type element = fragment.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  auto elements = fragmentElementCount(fragment);
  int64_t words = std::max(1u, (bits + 31) / 32);
  if (words == 1) return elements;
  MLIRContext *context = fragment.getContext();
  auto width = PhysicalExprAttr::get(context, PhysicalExprKind::Constant,
      words, StringAttr::get(context, ""), ArrayAttr::get(context, {}));
  return PhysicalExprAttr::get(context, PhysicalExprKind::Multiply, 0,
      StringAttr::get(context, ""), ArrayAttr::get(context, {width, elements}));
}

std::optional<int64_t> minimumFragmentRegisterFootprint(
    func::FuncOp kernel, Value value, int64_t limit,
    FragmentFootprintScope scope) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || !fragment.getElementType().isIntOrIndexOrFloat() ||
      limit < 0 || limit == std::numeric_limits<int64_t>::max())
    return std::nullopt;
  std::optional<PhysicalProgramAnalysis> analysis;
  if (scope == FragmentFootprintScope::FullScalarSeedCapacity)
    analysis.emplace(kernel);
  Type element = fragment.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  int64_t words = std::max(1u, (bits + 31) / 32);
  const __int128 saturation = static_cast<__int128>(limit) + 1;
  for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    int64_t minimum;
    if (scope == FragmentFootprintScope::FullScalarSeedCapacity &&
        analysis->axisRealization(value, axis).constructionScalarSeed) {
      auto range = queryExactLogicalRange(analysis->axisRanges(value, axis));
      auto capacity = succeeded(range) ? queryLogicalRangeCapacity(*range)
                                       : PhysicalExprAttr();
      auto count = capacity ? constantPhysicalExpression(capacity) : std::nullopt;
      if (!count || *count <= 0) return std::nullopt;
      auto rounded = static_cast<__int128>(
          llvm::PowerOf2Ceil(static_cast<uint64_t>(*count)));
      minimum = static_cast<int64_t>(std::min(saturation, rounded));
    } else if (extent.getKind() == PhysicalExprKind::Constant) {
      minimum = extent.getValue();
    } else if (extent.getKind() == PhysicalExprKind::Parameter) {
      auto parameter = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      if (failed(parameter)) return std::nullopt;
      minimum = *llvm::min_element(parameter->getCandidates().asArrayRef());
    } else {
      return std::nullopt;
    }
    if (minimum <= 0) return std::nullopt;
    words = static_cast<int64_t>(std::min(saturation, static_cast<__int128>(words) * minimum));
  }
  return words;
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf,
    llvm::function_ref<std::optional<int64_t>(ParameterRefAttr)>
        resolveActivation) {
  using Status = RequirementStatus;
  using Predicate = ConfigurationRequirementPredicate;
  if (auto reference = requirement.getActivation()) {
    auto activation =
        resolveActivation ? resolveActivation(reference) : std::nullopt;
    if (!activation)
      return {Status::Unknown, std::nullopt, std::nullopt};
    if (*activation == 0)
      return {Status::Inactive, std::nullopt, std::nullopt};
    if (*activation != 1)
      return {Status::Invalid, std::nullopt, std::nullopt};
  }
  auto usage = evaluatePhysicalExpression(requirement.getUsage(), resolveLeaf);
  auto limit = requirement.getLimit()
                   ? evaluatePhysicalExpression(requirement.getLimit(),
                                                resolveLeaf)
                   : std::nullopt;
  auto predicate = requirement.getPredicate();
  if ((usage && *usage <= 0) || (limit && *limit < 0) ||
      ((predicate == Predicate::MultipleOf || predicate == Predicate::Equal) &&
       limit && *limit == 0))
    return {Status::Invalid, usage, limit};
  if (predicate == Predicate::Positive)
    return {usage ? Status::Satisfied : Status::Unknown, usage, limit};
  if (predicate == Predicate::PowerOfTwo)
    return {!usage ? Status::Unknown
                   : llvm::isPowerOf2_64(*usage) ? Status::Satisfied
                                               : Status::Violated,
            usage, limit};
  if (predicate == Predicate::MultipleOf)
    return {!usage || !limit ? Status::Unknown
                            : *usage % *limit == 0 ? Status::Satisfied
                                                  : Status::Violated,
            usage, limit};
  if (predicate == Predicate::Equal)
    return {!usage || !limit ? Status::Unknown
                            : *usage == *limit ? Status::Satisfied
                                              : Status::Violated,
            usage, limit};
  if (!usage && limit) {
    // Preserve a finite upper-budget decision for positive products whose exact
    // count exceeds i64. Missing leaves and unproved arithmetic remain unknown.
    const __int128 cap = static_cast<__int128>(*limit) + 1;
    std::function<std::optional<__int128>(PhysicalExprAttr)> bounded =
        [&](PhysicalExprAttr expression) -> std::optional<__int128> {
      if (auto exact = evaluatePhysicalExpression(expression, resolveLeaf))
        return *exact < 0
                   ? std::nullopt
                   : std::optional<__int128>(
                         std::min(cap, static_cast<__int128>(*exact)));
      auto kind = expression.getKind();
      if (kind != PhysicalExprKind::Add && kind != PhysicalExprKind::Multiply)
        return std::nullopt;
      auto operands = expression.getOperands();
      auto left = bounded(cast<PhysicalExprAttr>(operands[0]));
      auto right = bounded(cast<PhysicalExprAttr>(operands[1]));
      if (!left || !right)
        return std::nullopt;
      return std::min(cap, kind == PhysicalExprKind::Add ? *left + *right
                                                       : *left * *right);
    };
    if (auto count = bounded(requirement.getUsage()); count && *count > *limit)
      return {Status::Violated, usage, limit};
  }
  if (!usage || !limit)
    return {Status::Unknown, usage, limit};
  return {*usage <= *limit ? Status::Satisfied : Status::Violated,
          usage, limit};
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, DictionaryAttr bindings) {
  return evaluateConfigurationRequirement(
      requirement,
      [&](PhysicalExprAttr leaf) -> std::optional<int64_t> {
        if (leaf.getKind() != PhysicalExprKind::Parameter || !bindings)
          return std::nullopt;
        auto value =
            bindings.getAs<IntegerAttr>(leaf.getParameterReference().getName());
        return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
      },
      [&](ParameterRefAttr reference) -> std::optional<int64_t> {
        auto value = bindings ? bindings.getAs<IntegerAttr>(reference.getName())
                              : IntegerAttr();
        // Configuration rows store every declaration, including i1 choices,
        // as i64 bindings. Do not sign-extend an SSA i1 true into -1 here.
        return value && value.getType().isSignlessInteger(64)
                   ? std::optional<int64_t>(value.getInt())
                   : std::nullopt;
      });
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, DictionaryAttr bindings,
    func::FuncOp kernel) {
  auto exact = evaluateConfigurationRequirement(requirement, bindings);
  if (exact.status != RequirementStatus::Unknown || !exact.limit ||
      requirement.getActivation() ||
      requirement.getKind() != ConfigurationRequirementKind::NominalBudget ||
      requirement.getMetric() != ConfigurationRequirementMetric::FragmentRegisterWords ||
      requirement.getPredicate() != ConfigurationRequirementPredicate::LessEqual)
    return exact;
  std::function<bool(PhysicalExprAttr)> monotone = [&](PhysicalExprAttr value) {
    auto kind = value.getKind();
    if (kind == PhysicalExprKind::Constant) return value.getValue() >= 0;
    if (kind == PhysicalExprKind::Parameter) return true;
    return (kind == PhysicalExprKind::Add || kind == PhysicalExprKind::Multiply) &&
        llvm::all_of(value.getOperands(), [&](Attribute operand) {
          return monotone(cast<PhysicalExprAttr>(operand));
        });
  };
  if (!monotone(requirement.getUsage())) return exact;
  auto lower = evaluateConfigurationRequirement(requirement,
      [&](PhysicalExprAttr leaf) -> std::optional<int64_t> {
        if (leaf.getKind() != PhysicalExprKind::Parameter) return std::nullopt;
        auto name = leaf.getParameterReference().getName();
        if (auto value = bindings ? bindings.getAs<IntegerAttr>(name) : IntegerAttr())
          return value.getInt() > 0 ? std::optional<int64_t>(value.getInt()) : std::nullopt;
        auto parameter = lookupParameter(kernel, leaf.getParameterReference());
        if (!parameter || !parameter.isExtent() || !parameter.isDeferred())
          return std::nullopt;
        auto candidates = parameter.getCandidates().asArrayRef();
        if (candidates.empty()) return std::nullopt;
        int64_t minimum = *llvm::min_element(candidates);
        return minimum > 0 ? std::optional<int64_t>(minimum) : std::nullopt;
      });
  return lower.status == RequirementStatus::Violated ? lower : exact;
}

namespace {

// Shape views keep their underlying payload alive; they do not allocate a
// second full fragment. Numeric casts and computations remain distinct values.
void collectPayloads(Value value, llvm::DenseSet<Value> &payloads) {
  if (auto view = value.getDefiningOp<BroadcastOp>())
    return collectPayloads(view.getValue(), payloads);
  if (auto view = value.getDefiningOp<ReshapeOp>())
    return collectPayloads(view.getValue(), payloads);
  if (auto view = value.getDefiningOp<SplatOp>())
    return collectPayloads(view.getValue(), payloads);
  if (auto record = value.getDefiningOp<MakeRecordOp>()) {
    for (Value field : record.getFields()) collectPayloads(field, payloads);
    return;
  }
  if (auto field = value.getDefiningOp<ExtractOp>()) {
    if (auto record = field.getRecord().getDefiningOp<MakeRecordOp>())
      return collectPayloads(record.getFields()[field.getField()], payloads);
    // An opaque carried record owns its fields together. Count it once even
    // when several field views and the record itself span the collective.
    return collectPayloads(field.getRecord(), payloads);
  }
  if (isa<FragmentType, RecordType>(value.getType())) payloads.insert(value);
}

PhysicalExprAttr reductionRegisterFootprint(
    Operation *reduction, const Liveness &liveness, ArrayRef<Value> orderedValues) {
  auto *block = liveness.getLiveness(reduction->getBlock());
  if (!block) return {};
  llvm::DenseSet<Value> payloads;
  for (Value value : block->currentlyLiveValues(reduction))
    collectPayloads(value, payloads);
  // A collective in a branch also spans values retained by its enclosing
  // operation for later consumers. Block-local liveness alone omits those.
  for (Operation *parent = reduction->getParentOp();
       parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
    auto *parentBlock = liveness.getLiveness(parent->getBlock());
    if (!parentBlock) continue;
    for (Value value : parentBlock->currentlyLiveValues(parent)) {
      if (value.getDefiningOp() == parent || liveness.isDeadAfter(value, parent))
        continue;
      collectPayloads(value, payloads);
    }
  }
  PhysicalExprAttr registers;
  bool tunable = false;
  auto kernel = reduction->getParentOfType<func::FuncOp>();
  auto append = [&](FragmentType fragment) {
    AttrTypeWalker parameters;
    parameters.addWalk([&](PhysicalExprAttr expression) {
      if (expression.getKind() != PhysicalExprKind::Parameter) return;
      auto parameter = lookupParameter(kernel, expression.getParameterReference());
      if (!parameter) return;
      auto role = parameter.getRole();
      tunable |= role == ParameterRole::OwnershipM ||
                 role == ParameterRole::OwnershipN ||
                 role == ParameterRole::Reduction ||
                 role == ParameterRole::ReductionOuter ||
                 role == ParameterRole::ReductionInner;
    });
    parameters.walk(fragment.getShape());
    auto footprint = fragmentRegisterFootprint(fragment);
    registers = !registers ? footprint : PhysicalExprAttr::get(
        reduction->getContext(), PhysicalExprKind::Add, 0,
        StringAttr::get(reduction->getContext(), ""),
        ArrayAttr::get(reduction->getContext(), {registers, footprint}));
  };
  std::function<void(Type)> appendType = [&](Type type) {
    if (auto fragment = dyn_cast<FragmentType>(type)) append(fragment);
    else if (auto record = dyn_cast<RecordType>(type))
      for (Attribute field : record.getFieldTypes())
        appendType(cast<TypeAttr>(field).getValue());
  };
  // Liveness sets are unordered. Preserve lexical value order in the published
  // expression, including distinct values that happen to have the same type.
  for (Value value : orderedValues) {
    if (payloads.contains(value)) appendType(value.getType());
  }
  // This projects existing traversal candidates. It does not reject a fixed
  // provider program on the basis of a nominal register-allocation estimate.
  return tunable ? registers : PhysicalExprAttr();
}

} // namespace

FootprintBound checkFragmentFootprint(
    FragmentType fragment, int64_t limit, int64_t wordsPerElement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf) {
  if (limit < 0 || wordsPerElement <= 0)
    return FootprintBound::Invalid;
  bool unknown = false;
  // Saturating the product after each dimension preserves a proof that the
  // bound was exceeded without overflowing even for high-rank fragments.
  __int128 count = wordsPerElement;
  const __int128 saturation = static_cast<__int128>(limit) + 1;
  for (Attribute attribute : fragment.getShape()) {
    auto extent = evaluatePhysicalExpression(cast<PhysicalExprAttr>(attribute),
                                             resolveLeaf);
    if (!extent) {
      unknown = true;
      continue;
    }
    if (*extent <= 0)
      return FootprintBound::Invalid;
    count = std::min(saturation, count * *extent);
  }
  if (unknown)
    return FootprintBound::Unknown;
  return count > limit ? FootprintBound::Exceeds : FootprintBound::Within;
}

FragmentResourceAnalysis::FragmentResourceAnalysis(func::FuncOp kernel) {
  llvm::DenseSet<FragmentType> seenValues, seenPayloads;
  AttrTypeWalker valueTypes;
  valueTypes.addWalk([&](FragmentType fragment) {
    if (seenValues.insert(fragment).second)
      values.push_back(fragment);
  });
  kernel.walk([&](Operation *operation) {
    for (Type type : operation->getOperandTypes())
      valueTypes.walk(type);
    for (Type type : operation->getResultTypes())
      valueTypes.walk(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          valueTypes.walk(argument.getType());
    if (isa<BroadcastOp, SplatOp, ReshapeOp>(operation))
      return;
    for (Type type : operation->getResultTypes()) {
      auto fragment = dyn_cast<FragmentType>(type);
      if (!fragment || !seenPayloads.insert(fragment).second)
        continue;
      llvm::DenseSet<StringAttr> symbols;
      AttrTypeWalker parameters;
      parameters.addWalk([&](PhysicalExprAttr expression) {
        if (expression.getKind() ==
            PhysicalExprKind::Parameter)
          symbols.insert(expression.getParameterReference().getName());
      });
      parameters.walk(fragment.getShape());
      for (StringAttr name : symbols)
        payloads[name].push_back(fragment);
    }
  });
  SmallVector<Operation *> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  collectives = collectReductionRequirements(
      kernel, reductions, ReductionRequirementScope::AllCandidates);
}

ArrayRef<FragmentType>
FragmentResourceAnalysis::materializedTypesUsing(StringAttr name) const {
  auto found = payloads.find(name);
  return found == payloads.end() ? ArrayRef<FragmentType>()
                                : ArrayRef<FragmentType>(found->second);
}

namespace {

bool dependsOnInvocation(ConfigurationRequirementAttr requirement,
                         func::FuncOp kernel) {
  // Sampling the nondeferred domains only asks whether all leaves are known;
  // it does not establish a bound for the other candidates.
  auto known = evaluateConfigurationRequirement(requirement,
      [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
        if (expression.getKind() != PhysicalExprKind::Parameter) return std::nullopt;
        auto parameter = lookupParameter(kernel, expression.getParameterReference());
        if (!parameter || parameter.isDeferred() ||
            parameter.getCategory() == ParameterCategory::Coverage)
          return std::nullopt;
        return parameter.getCandidates().asArrayRef().front();
      });
  return !known.usage;
}

} // namespace

bool isPointwiseTraversalParameter(ParameterAttr parameter) {
  return parameter.getBinding().getPointwiseChunk() ||
      (parameter.getCategory() == ParameterCategory::Pointwise &&
       (parameter.getRole() == ParameterRole::OwnershipM ||
        parameter.getRole() == ParameterRole::OwnershipN));
}

SmallVector<ConfigurationRequirementAttr> collectPointwiseRequirements(
    func::FuncOp kernel, const FragmentResourceAnalysis &resources) {
  SmallVector<ConfigurationRequirementAttr> requirements;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0) return requirements;
  Builder builder(kernel.getContext());
  auto limit = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Constant,
      capabilities.getRegistersPerUnit() - 1, builder.getStringAttr(""), builder.getArrayAttr({}));
  llvm::DenseSet<FragmentType> seen;
  for (Attribute attribute : getParameterDeclarations(kernel)) {
    auto parameter = cast<ParameterAttr>(attribute);
    if (!isPointwiseTraversalParameter(parameter)) continue;
    for (FragmentType fragment : resources.materializedTypesUsing(parameter.getName())) {
      if (!seen.insert(fragment).second) continue;
      auto requirement = ConfigurationRequirementAttr::get(kernel.getContext(),
          ConfigurationRequirementKind::NominalBudget,
          ConfigurationRequirementMetric::FragmentRegisterWords,
          ConfigurationRequirementPredicate::LessEqual,
          fragmentRegisterFootprint(fragment), limit, ParameterRefAttr(),
          builder.getStringAttr("pointwise payload exceeds the candidate register budget"));
      if (dependsOnInvocation(requirement, kernel) &&
          !llvm::is_contained(requirements, requirement))
        requirements.push_back(requirement);
    }
  }
  return requirements;
}

SmallVector<ConfigurationRequirementAttr> collectReductionRequirements(
    func::FuncOp kernel, ArrayRef<Operation *> reductions,
    ReductionRequirementScope scope) {
  SmallVector<ConfigurationRequirementAttr> requirements;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return requirements;
  if (reductions.empty()) return requirements;
  Builder builder(kernel.getContext());
  auto limit = PhysicalExprAttr::get(kernel.getContext(), PhysicalExprKind::Constant,
      capabilities.getRegistersPerUnit(), builder.getStringAttr(""), builder.getArrayAttr({}));
  Liveness liveness(kernel);
  SmallVector<Value> values;
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    llvm::append_range(values, operation->getResults());
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        llvm::append_range(values, block.getArguments());
  });
  for (Operation *reduction : reductions) {
    auto footprint = reductionRegisterFootprint(reduction, liveness, values);
    if (!footprint) continue;
    auto requirement = ConfigurationRequirementAttr::get(kernel.getContext(),
        ConfigurationRequirementKind::NominalBudget,
        ConfigurationRequirementMetric::FragmentRegisterWords,
        ConfigurationRequirementPredicate::LessEqual, footprint, limit,
        ParameterRefAttr(),
        builder.getStringAttr("collective live payloads exceed the candidate register budget"));
    if (scope == ReductionRequirementScope::InvocationDependent &&
        !dependsOnInvocation(requirement, kernel)) continue;
    if (!llvm::is_contained(requirements, requirement)) requirements.push_back(requirement);
  }
  return requirements;
}

LogicalResult verifyConfigurationRequirements(
    func::FuncOp kernel, ArrayRef<ConfigurationRequirementAttr> expected) {
  auto space = ParameterSpace::read(kernel);
  if (failed(space) || failed(space->verifyRequirements(expected)))
    return failure();
  auto rows = space->configurations(ConfigurationStage::Complete);
  auto requirements = space->requirements();
  if (failed(rows) || failed(requirements))
    return failure();
  llvm::DenseSet<Attribute> remaining;
  for (ConfigurationRequirementAttr requirement : expected)
    if (!remaining.insert(requirement).second)
      return kernel.emitError(
          "current program produced a duplicate candidate requirement");
  for (ConfigurationRequirementAttr requirement : *requirements)
    if (!remaining.erase(requirement))
      return kernel.emitError(
                 "candidate requirement is not justified by the current program: ")
             << requirement;
  if (!remaining.empty())
    return kernel.emitError(
               "candidate requirements omit a current program condition: ")
           << *remaining.begin();
  for (DictionaryAttr row : *rows)
    for (ConfigurationRequirementAttr requirement : *requirements) {
      auto evaluation = evaluateConfigurationRequirement(requirement, row, kernel);
      if (evaluation.status != RequirementStatus::Violated &&
          evaluation.status != RequirementStatus::Invalid)
        continue;
      auto diagnostic = kernel.emitError(
          "candidate violates a current program requirement: ");
      diagnostic << requirement.getMessage().getValue()
                 << "; predicate="
                 << stringifyConfigurationRequirementPredicate(
                        requirement.getPredicate())
                 << "; bindings=" << row;
      if (evaluation.usage)
        diagnostic << "; usage=" << *evaluation.usage;
      if (evaluation.limit)
        diagnostic << "; limit=" << *evaluation.limit;
      return failure();
    }
  return success();
}

} // namespace intent::gpu

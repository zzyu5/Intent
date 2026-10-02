#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
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
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf) {
  auto usage = evaluatePhysicalExpression(requirement.getUsage(), resolveLeaf);
  auto limit = evaluatePhysicalExpression(requirement.getLimit(), resolveLeaf);
  if ((usage && *usage <= 0) || (limit && *limit < 0))
    return {FootprintBound::Invalid, usage, limit};
  if (!usage && limit) {
    // Preserve a finite upper-budget decision for positive products whose exact
    // count exceeds i64. Missing leaves and unproved arithmetic remain unknown.
    const __int128 cap = static_cast<__int128>(*limit) + 1;
    std::function<std::optional<__int128>(PhysicalExprAttr)> bounded =
        [&](PhysicalExprAttr expression) -> std::optional<__int128> {
      if (auto exact = evaluatePhysicalExpression(expression, resolveLeaf))
        return *exact < 0 ? std::nullopt
                         : std::optional<__int128>(std::min(cap, static_cast<__int128>(*exact)));
      auto kind = expression.getKind();
      if (kind != PhysicalExprKind::Add && kind != PhysicalExprKind::Multiply)
        return std::nullopt;
      auto operands = expression.getOperands();
      auto left = bounded(cast<PhysicalExprAttr>(operands[0]));
      auto right = bounded(cast<PhysicalExprAttr>(operands[1]));
      if (!left || !right) return std::nullopt;
      return std::min(cap, kind == PhysicalExprKind::Add ? *left + *right : *left * *right);
    };
    if (auto count = bounded(requirement.getUsage()); count && *count > *limit)
      return {FootprintBound::Exceeds, usage, limit};
  }
  if (!usage || !limit)
    return {FootprintBound::Unknown, usage, limit};
  return {*usage <= *limit ? FootprintBound::Within : FootprintBound::Exceeds,
          usage, limit};
}

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, DictionaryAttr bindings) {
  return evaluateConfigurationRequirement(requirement,
      [&](PhysicalExprAttr leaf) -> std::optional<int64_t> {
        if (leaf.getKind() != PhysicalExprKind::Parameter || !bindings)
          return std::nullopt;
        auto value = bindings.getAs<IntegerAttr>(leaf.getParameterReference().getName());
        return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
      });
}

PhysicalExprAttr reductionRegisterFootprint(ValueRange sources,
                                          func::FuncOp kernel) {
  std::function<bool(PhysicalExprAttr)> isTunableExtent =
      [&](PhysicalExprAttr extent) {
    if (extent.getKind() ==
        PhysicalExprKind::Parameter) {
      auto parameter = queryParameterBySymbol(kernel, extent.getParameterReference().getName());
      if (failed(parameter))
        return false;
      auto role = parameter->getRole();
      return role == ParameterRole::OwnershipM ||
             role == ParameterRole::OwnershipN ||
             role == ParameterRole::Reduction ||
             role == ParameterRole::ReductionOuter ||
             role == ParameterRole::ReductionInner;
    }
    return llvm::any_of(extent.getOperands(), [&](Attribute operand) {
      return isTunableExtent(cast<PhysicalExprAttr>(operand));
    });
  };
  PhysicalExprAttr registers;
  for (Value source : sources) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || !llvm::any_of(fragment.getShape(), [&](Attribute extent) {
          return isTunableExtent(cast<PhysicalExprAttr>(extent));
        }))
      continue;
    auto footprint = fragmentRegisterFootprint(fragment);
    registers = !registers ? footprint : PhysicalExprAttr::get(
        kernel.getContext(), PhysicalExprKind::Add, 0,
        StringAttr::get(kernel.getContext(), ""),
        ArrayAttr::get(kernel.getContext(), {registers, footprint}));
  }
  return registers;
}

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
}

ArrayRef<FragmentType>
FragmentResourceAnalysis::materializedTypesUsing(StringAttr name) const {
  auto found = payloads.find(name);
  return found == payloads.end() ? ArrayRef<FragmentType>()
                                : ArrayRef<FragmentType>(found->second);
}

} // namespace intent::gpu

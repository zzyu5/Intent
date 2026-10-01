#include "ConfigurationPolicy.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;

namespace intent::gpu::configuration {
namespace {

bool fitsFragmentFootprints(
    ArrayRef<FragmentType> fragments, CapabilitiesAttr capabilities,
    const NamedAttrList &bindings, StringAttr selected = {}, int64_t candidate = 0) {
  auto resolve = [&](PhysicalExprAttr expression) -> std::optional<int64_t> {
    if (expression.getKind() != static_cast<uint32_t>(PhysicalExprKind::Parameter))
      return std::nullopt;
    auto binding = dyn_cast_or_null<IntegerAttr>(
        bindings.get(expression.getSymbol().getValue()));
    if (!binding) return std::nullopt;
    return expression.getSymbol() == selected ? candidate : binding.getInt();
  };
  // This bounds individual payloads, not provider register allocation or
  // machine occupancy. ABI-dependent Unknown bounds remain for specialization.
  for (FragmentType fragment : fragments) {
    Type element = fragment.getElementType();
    unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
    auto bound = checkFragmentFootprint(
        fragment, capabilities.getRegistersPerUnit() - 1,
        std::max(1u, (bits + 31) / 32), resolve);
    if (bound == FootprintBound::Exceeds || bound == FootprintBound::Invalid)
      return false;
  }
  return true;
}

} // namespace

void bindContractionFreeExtents(
    ArrayRef<ContractionFreeExtent> groups, NamedAttrList &bindings,
    ProfileLookup profileFor, Builder &builder, bool splitInnerAxis) {
  for (const ContractionFreeExtent &group : groups) {
    int64_t budget = std::numeric_limits<int64_t>::max();
    for (ParameterOp parameter : group.profileParameters)
      budget = std::min(budget, requestedValue(profileFor(parameter), group.role));
    auto fits = [&](ParameterOp selected, int64_t candidate) {
      std::function<std::optional<int64_t>(PhysicalExprAttr)> evaluate =
          [&](PhysicalExprAttr extent) -> std::optional<int64_t> {
        auto kind = static_cast<PhysicalExprKind>(extent.getKind());
        if (kind == PhysicalExprKind::Constant)
          return extent.getValue() <= budget
                     ? std::optional<int64_t>(extent.getValue()) : std::nullopt;
        if (kind == PhysicalExprKind::Parameter) {
          int64_t value = selected && extent.getSymbol() == selected.getParameter().getName()
                              ? candidate
                              : cast<IntegerAttr>(bindings.get(extent.getSymbol().getValue())).getInt();
          return value <= budget ? std::optional<int64_t>(value) : std::nullopt;
        }
        int64_t product = 1;
        for (Attribute operand : extent.getOperands()) {
          auto factor = evaluate(cast<PhysicalExprAttr>(operand));
          if (!factor || product > budget / *factor) return std::nullopt;
          product *= *factor;
        }
        return product;
      };
      int64_t product = 1;
      for (PhysicalExprAttr extent : group.extents) {
        auto factor = evaluate(extent);
        if (!factor || product > budget / *factor) return false;
        product *= *factor;
      }
      return true;
    };
    // Keep the innermost coordinates wide; the profile targets their combined
    // M/N extent, not each independently tiled factor of that extent.
    for (ParameterOp parameter : group.parameters) {
      if (fits({}, 0)) break;
      auto schema = parameter.getParameter();
      int64_t current = cast<IntegerAttr>(bindings.get(schema.getName().getValue())).getInt();
      auto candidates = schema.getCandidates().asArrayRef();
      int64_t selected = *std::min_element(candidates.begin(), candidates.end());
      for (int64_t candidate : candidates)
        if (candidate <= current && candidate > selected && fits(parameter, candidate))
          selected = candidate;
      bindings.set(schema.getName(), builder.getI64IntegerAttr(selected));
    }
    // A flattened free side can span adjacent logical axes. Offer the same
    // tile budget across those axes as well as along its innermost coordinate.
    if (splitInnerAxis && group.parameters.size() > 1) {
      ParameterOp outerParameter = group.parameters[group.parameters.size() - 2];
      ParameterOp innerParameter = group.parameters.back();
      ParameterAttr outer = outerParameter.getParameter();
      ParameterAttr inner = innerParameter.getParameter();
      int64_t outerValue = cast<IntegerAttr>(bindings.get(outer.getName().getValue())).getInt();
      int64_t innerValue = cast<IntegerAttr>(bindings.get(inner.getName().getValue())).getInt();
      if (innerValue > 1 && innerValue % 2 == 0 && outerValue <= budget / 2 &&
          llvm::is_contained(outer.getCandidates().asArrayRef(), outerValue * 2) &&
          llvm::is_contained(inner.getCandidates().asArrayRef(), innerValue / 2)) {
        bindings.set(outer.getName(), builder.getI64IntegerAttr(outerValue * 2));
        bindings.set(inner.getName(), builder.getI64IntegerAttr(innerValue / 2));
        if (!fits({}, 0)) {
          bindings.set(outer.getName(), builder.getI64IntegerAttr(outerValue));
          bindings.set(inner.getName(), builder.getI64IntegerAttr(innerValue));
        }
      }
    }
  }
}

LogicalResult bindTraversalFragmentFootprints(
    func::FuncOp kernel, ArrayRef<ParameterOp> parameters,
    const FragmentResourceAnalysis &resources, NamedAttrList &bindings,
    Builder &builder) {
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!capabilities || capabilities.getRegistersPerUnit() <= 0)
    return success();
  SmallVector<ArrayRef<FragmentType>> budgeted;
  for (ParameterOp parameter : parameters) {
    auto schema = parameter.getParameter();
    auto category = static_cast<ParameterCategory>(schema.getCategory());
    auto role = static_cast<ParameterRole>(schema.getRole());
    bool reduction = category == ParameterCategory::Reduction &&
                     (role == ParameterRole::Reduction ||
                      role == ParameterRole::ReductionInner ||
                      role == ParameterRole::ReductionOuter);
    bool pointwise =
        category == ParameterCategory::Pointwise &&
        (role == ParameterRole::OwnershipM || role == ParameterRole::OwnershipN);
    if (!parameter->hasAttr(pointwiseChunkAttr) && !reduction && !pointwise)
      continue;
    auto fragments = resources.materializedTypesUsing(schema.getName());
    auto fits = [&](int64_t candidate) {
      return fitsFragmentFootprints(
          fragments, capabilities, bindings, schema.getName(), candidate);
    };
    int64_t requested = cast<IntegerAttr>(
        bindings.get(schema.getName().getValue())).getInt();
    std::optional<int64_t> selected;
    for (int64_t candidate : schema.getCandidates().asArrayRef())
      if (candidate <= requested && (!selected || candidate > *selected) &&
          fits(candidate))
        selected = candidate;
    // Other traversal axes may still exceed their budget. Reach this axis's
    // legal minimum before binding those axes, then validate the complete tuple.
    if (!selected)
      selected = *llvm::min_element(schema.getCandidates().asArrayRef());
    bindings.set(schema.getName(), builder.getI64IntegerAttr(*selected));
    budgeted.push_back(fragments);
  }
  // Every axis has now been projected. Recheck those same payload sets against
  // the complete tuple, since another traversal axis may have changed it.
  return success(llvm::all_of(budgeted, [&](ArrayRef<FragmentType> fragments) {
    return fitsFragmentFootprints(fragments, capabilities, bindings);
  }));
}

void appendFullResultContractionTuples(
    func::FuncOp kernel, ArrayRef<FullResultContraction> contractions,
    const NamedAttrList &bindings, ArrayRef<ParameterOp> parameters,
    const FragmentResourceAnalysis &resources, ProfileLookup profileFor,
    Builder &builder, llvm::function_ref<void(DictionaryAttr)> append) {
  for (const FullResultContraction &contraction : contractions) {
    ParameterOp parameter = contraction.parameter;
    auto schema = parameter.getParameter();
    auto other = contraction.otherExtent;
    int64_t otherExtent;
    if (other.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant))
      otherExtent = other.getValue();
    else if (other.getKind() ==
             static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      auto binding = dyn_cast_or_null<IntegerAttr>(
          bindings.get(other.getSymbol().getValue()));
      if (!binding)
        continue;
      otherExtent = binding.getInt();
    } else
      continue;
    if (otherExtent <= 0)
      continue;
    // The full result is already live. Widen its producer slice within the
    // existing accumulator budget to reduce repeated gather assembly.
    const TuningProfile &profile = profileFor(parameter);
    __int128 budget = static_cast<__int128>(profile.ownershipM) *
                      profile.ownershipN / otherExtent;
    int64_t requested = static_cast<int64_t>(std::min<__int128>(
        budget, std::numeric_limits<int64_t>::max()));
    int64_t selected =
        selectCandidate(schema.getCandidates().asArrayRef(), requested);
    auto current = cast<IntegerAttr>(bindings.get(schema.getName().getValue()));
    if (selected <= current.getInt())
      continue;
    NamedAttrList widened(bindings);
    widened.set(schema.getName(), builder.getI64IntegerAttr(selected));
    // Widened producer tiles are candidates of the same current program. They
    // must pass the same budget binding as the ordinary profile projection.
    if (failed(bindTraversalFragmentFootprints(kernel, parameters, resources,
                                              widened, builder)))
      continue;
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    if (capabilities && capabilities.getRegistersPerUnit() > 0 &&
        !fitsFragmentFootprints(resources.materializedTypesUsing(schema.getName()),
                                capabilities, widened))
      continue;
    append(widened.getDictionary(kernel.getContext()));
  }
}

} // namespace intent::gpu::configuration

#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

using namespace mlir;

namespace intent::gpu {

PhysicalExprAttr fragmentRegisterFootprint(FragmentType fragment) {
  Type element = fragment.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  MLIRContext *context = fragment.getContext();
  auto footprint = PhysicalExprAttr::get(
      context, static_cast<uint32_t>(PhysicalExprKind::Constant),
      std::max(1u, (bits + 31) / 32), StringAttr::get(context, ""),
      ArrayAttr::get(context, {}));
  for (Attribute extent : fragment.getShape())
    footprint = PhysicalExprAttr::get(
        context, static_cast<uint32_t>(PhysicalExprKind::Multiply), 0,
        StringAttr::get(context, ""), ArrayAttr::get(context, {footprint, extent}));
  return footprint;
}

PhysicalExprAttr reductionRegisterFootprint(ValueRange sources,
                                          func::FuncOp kernel) {
  std::function<bool(PhysicalExprAttr)> isTunableExtent =
      [&](PhysicalExprAttr extent) {
    if (static_cast<PhysicalExprKind>(extent.getKind()) ==
        PhysicalExprKind::Parameter) {
      auto parameter = queryParameterBySymbol(kernel, extent.getSymbol());
      if (failed(parameter))
        return false;
      auto role = static_cast<ParameterRole>(parameter->getParameter().getRole());
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
        kernel.getContext(), static_cast<uint32_t>(PhysicalExprKind::Add), 0,
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
            static_cast<uint32_t>(PhysicalExprKind::Parameter))
          symbols.insert(expression.getSymbol());
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

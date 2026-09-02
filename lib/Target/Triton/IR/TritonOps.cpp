#include "Intent/Target/Triton/IR/TritonOps.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace intent::triton {
namespace {

LogicalResult verifyBlockAccess(Operation *owner, Value viewValue,
                                gpu::FragmentType fragment,
                                ValueRange offsets, ArrayRef<int64_t> blockAxes,
                                ArrayRef<int64_t> order,
                                ArrayRef<int64_t> boundaryAxes) {
  auto view = cast<gpu::ViewType>(viewValue.getType());
  if (!isa<BlockArgument>(viewValue))
    return owner->emitOpError(
        "requires an external-view kernel argument as its pointer base");
  if (!view.getLayout().getHasStrides() ||
      view.getLayout().getStrides().size() != view.getRank())
    return owner->emitOpError(
        "requires one explicit stride per external-view axis");
  if (offsets.size() != view.getRank())
    return owner->emitOpError(
        "requires one source-ordered offset per external-view axis");
  if (view.getElementType() != fragment.getElementType() ||
      blockAxes.size() != fragment.getShape().size())
    return owner->emitOpError(
        "block axes and fragment type do not match the external view");
  llvm::SmallBitVector blocked(view.getRank());
  for (int64_t axis : blockAxes) {
    if (axis < 0 || axis >= static_cast<int64_t>(view.getRank()) ||
        blocked.test(axis))
      return owner->emitOpError(
          "block axes must be distinct axes of the external view");
    blocked.set(axis);
  }
  unsigned blockRank = fragment.getShape().size();
  if (order.size() != blockRank)
    return owner->emitOpError(
        "block-pointer order must cover every fragment axis");
  llvm::SmallBitVector ordered(blockRank);
  for (int64_t axis : order) {
    if (axis < 0 || axis >= static_cast<int64_t>(blockRank) ||
        ordered.test(axis))
      return owner->emitOpError(
          "block-pointer order must be a permutation of the fragment axes");
    ordered.set(axis);
  }
  for (unsigned index = 1; index < order.size(); ++index)
    if (blockAxes[order[index - 1]] <= blockAxes[order[index]])
      return owner->emitOpError(
          "block-pointer order must follow descending external-view data order");
  llvm::SmallBitVector checked(blockRank);
  for (int64_t axis : boundaryAxes) {
    if (axis < 0 || axis >= static_cast<int64_t>(blockRank) ||
        checked.test(axis))
      return owner->emitOpError(
          "boundary axes must be distinct block-pointer axes");
    checked.set(axis);
  }
  return success();
}

} // namespace

LogicalResult BlockLoadOp::verify() {
  if (getPadding() != "zero")
    return emitOpError("supports only zero padding for proven tail loads");
  return verifyBlockAccess(*this, getView(), getResult().getType(),
                           getOffsets(), getBlockAxes(), getOrder(),
                           getBoundaryAxes());
}

void BlockLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult BlockStoreOp::verify() {
  return verifyBlockAccess(*this, getView(), getValue().getType(),
                           getOffsets(), getBlockAxes(), getOrder(),
                           getBoundaryAxes());
}

void BlockStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

} // namespace intent::triton

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"

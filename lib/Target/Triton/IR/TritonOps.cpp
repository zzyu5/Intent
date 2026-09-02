#include "Intent/Target/Triton/IR/TritonOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"

#include <algorithm>

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

LogicalResult verifyDescriptorAccess(Operation *owner, Value descriptorValue,
                                     gpu::FragmentType fragment,
                                     ValueRange offsets,
                                     ArrayRef<int64_t> boundaryAxes) {
  auto descriptor = descriptorValue.getDefiningOp<TensorDescriptorOp>();
  if (!descriptor)
    return owner->emitOpError(
        "requires an explicit tensor-descriptor value");
  auto base = cast<gpu::ViewType>(descriptor.getBase().getType());
  if (base.getElementType() != fragment.getElementType())
    return owner->emitOpError(
        "tensor descriptor and fragment element types must match");
  if (offsets.size() != descriptor.getShape().size() ||
      fragment.getShape().size() != descriptor.getBlockShape().size())
    return owner->emitOpError(
        "descriptor offsets and fragment rank must match its declared shape");
  for (auto [extent, blockShape] :
       llvm::zip(fragment.getShape(), descriptor.getBlockShape())) {
    auto expression = blockShape.getDefiningOp<gpu::PhysicalExprOp>();
    if (!expression || expression.getExpression() != extent)
      return owner->emitOpError(
          "fragment extents must equal the declared descriptor block shape");
  }
  llvm::SmallBitVector checked(fragment.getShape().size());
  for (int64_t axis : boundaryAxes) {
    if (axis < 0 || axis >= static_cast<int64_t>(checked.size()) ||
        checked.test(axis))
      return owner->emitOpError(
          "descriptor boundary axes must be distinct fragment axes");
    checked.set(axis);
  }
  return success();
}

bool collectFlattenedDimensions(Value value, Value base,
                                SmallVectorImpl<unsigned> &axes) {
  if (auto dimension = value.getDefiningOp<gpu::DimOp>()) {
    if (dimension.getView() != base)
      return false;
    axes.push_back(dimension.getAxis());
    return true;
  }
  auto product = value.getDefiningOp<gpu::BinaryOp>();
  if (!product || product.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  return collectFlattenedDimensions(product.getLhs(), base, axes) &&
         collectFlattenedDimensions(product.getRhs(), base, axes);
}

bool hasStrideBinding(func::FuncOp kernel, gpu::ViewType view, unsigned axis) {
  Attribute stride = view.getLayout().getStrides()[axis];
  if (auto constant = dyn_cast<IntegerAttr>(stride))
    return constant.getInt() > 0;
  auto symbol = dyn_cast<StringAttr>(stride);
  if (!symbol)
    return false;
  unsigned matches = 0;
  for (auto [index, argument] : llvm::enumerate(kernel.getArguments())) {
    auto name = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiNameAttr);
    auto kind = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiKindAttr);
    auto source =
        kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceABIAttr);
    auto sourceAxis =
        kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceAxisAttr);
    if (name == symbol && kind && kind.getValue() == "stride" && source &&
        source.getInt() == view.getAbiIndex() && sourceAxis &&
        sourceAxis.getInt() == axis && argument.getType().isIndex())
      ++matches;
  }
  return matches == 1;
}

bool hasContiguousDescriptorLayout(func::FuncOp kernel, gpu::ViewType view) {
  auto layout = view.getLayout();
  if (!layout.getHasStrides() || layout.getStrides().size() != view.getRank())
    return false;
  for (unsigned axis = 0; axis < view.getRank(); ++axis)
    if (!hasStrideBinding(kernel, view, axis))
      return false;

  auto lastStride = dyn_cast<IntegerAttr>(
      layout.getStrides()[layout.getStrides().size() - 1]);
  if (lastStride && lastStride.getInt() != 1)
    return false;
  for (unsigned axis = 0; axis + 1 < view.getRank(); ++axis) {
    auto stride = dyn_cast<IntegerAttr>(layout.getStrides()[axis]);
    auto nextStride = dyn_cast<IntegerAttr>(layout.getStrides()[axis + 1]);
    auto nextExtent =
        dyn_cast<gpu::PhysicalExprAttr>(layout.getExtents()[axis + 1]);
    if (!stride || !nextStride || !nextExtent ||
        nextExtent.getKind() !=
            static_cast<uint32_t>(gpu::PhysicalExprKind::Constant))
      continue;
    __int128 expected = static_cast<__int128>(nextStride.getInt()) *
                        nextExtent.getValue();
    if (expected != stride.getInt())
      return false;
  }
  return true;
}

} // namespace

LogicalResult TensorDescriptorChoiceOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(gpu::kernelAttr))
    return emitOpError("must belong to a physical GPU kernel");
  unsigned choices = 0;
  kernel.walk([&](TensorDescriptorChoiceOp) { ++choices; });
  if (choices != 1)
    return emitOpError(
        "requires exactly one tensor descriptor choice per physical kernel");
  if (getConstruction() != "host")
    return emitOpError("supports only explicit host descriptor binding");
  return success();
}

LogicalResult TensorDescriptorAllocatorOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(gpu::kernelAttr))
    return emitOpError("must belong to a physical GPU kernel");
  unsigned allocators = 0;
  unsigned descriptors = 0;
  kernel.walk([&](TensorDescriptorAllocatorOp) { ++allocators; });
  kernel.walk([&](TensorDescriptorOp) { ++descriptors; });
  if (allocators != 1 || descriptors == 0)
    return emitOpError(
        "requires one allocator declaration for a non-empty descriptor set");
  if (getSizeArgument() != 0 || getAlignmentArgument() != 1 ||
      getStreamArgument() != 2)
    return emitOpError(
        "allocator ABI must bind runtime size, alignment, and stream in order");
  if (getLifetime() != "launch")
    return emitOpError("supports only launch-lifetime descriptor allocation");
  return success();
}

LogicalResult TensorDescriptorOp::verify() {
  auto kernel = (*this)->getParentOfType<func::FuncOp>();
  if (!kernel || !kernel->hasAttr(gpu::kernelAttr))
    return emitOpError("must belong to a physical GPU kernel");
  auto base = cast<gpu::ViewType>(getBase().getType());
  auto baseArgument = dyn_cast<BlockArgument>(getBase());
  if (!baseArgument || baseArgument.getOwner() != &kernel.getBody().front() ||
      getResult().getType() != base)
    return emitOpError(
        "must preserve one external-view kernel argument as its base");
  if (base.getRank() < 2 || base.getRank() > 5 || getShape().size() != 2 ||
      getStrides().size() != 2 || getBlockShape().size() != 2)
    return emitOpError(
        "host tensor descriptor must declare a two-axis flattening of a rank-2-to-5 view");
  if (getSourceBlockAxes() != ArrayRef<int64_t>{
                                  static_cast<int64_t>(base.getRank()) - 2,
                                  static_cast<int64_t>(base.getRank()) - 1})
    return emitOpError(
        "tensor descriptor flattening requires the final two source axes");
  int64_t elementBytes =
      static_cast<int64_t>(base.getElementType().getIntOrFloatBitWidth() / 8);
  if (getInitialBlockShape().size() != 2 ||
      getInitialBlockShape().front() != 1 ||
      getInitialBlockShape().back() <= 0 ||
      !llvm::isPowerOf2_64(getInitialBlockShape().back()) ||
      getInitialBlockShape().back() * elementBytes <
          static_cast<int64_t>(getMinimumContiguousBytes()))
    return emitOpError(
        "host tensor descriptor must declare a legal provider dummy block shape");
  if (getBaseLayout() != "contiguous" || getPadding() != "zero" ||
      getAlignment() != 16 || getMinimumContiguousBytes() != 16)
    return emitOpError(
        "host tensor descriptor requires contiguous base, zero padding, 16-byte alignment, and a 16-byte contiguous block");
  if (!hasContiguousDescriptorLayout(kernel, base))
    return emitOpError(
        "host tensor descriptor base must carry a complete row-major stride ABI");
  unsigned bitWidth = base.getElementType().getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return emitOpError(
        "tensor descriptor element type must occupy whole bytes");
  auto lastDimension = getShape().back().getDefiningOp<gpu::DimOp>();
  SmallVector<unsigned> flattenedAxes;
  if (!lastDimension || lastDimension.getView() != getBase() ||
      lastDimension.getAxis() + 1 != base.getRank() ||
      getStrides().front() != getShape().back() ||
      !collectFlattenedDimensions(getShape().front(), getBase(),
                                  flattenedAxes))
    return emitOpError(
        "descriptor shape and strides must explicitly flatten all leading base axes");
  SmallVector<unsigned> expectedAxes;
  for (unsigned axis = 0; axis + 1 < base.getRank(); ++axis)
    expectedAxes.push_back(axis);
  if (flattenedAxes != expectedAxes)
    return emitOpError(
        "descriptor row shape must preserve every leading base dimension in order");
  auto unitStride = getStrides().back().getDefiningOp<arith::ConstantIndexOp>();
  if (!unitStride || unitStride.value() != 1)
    return emitOpError("tensor descriptor final stride must be one");
  for (Value extent : getBlockShape())
    if (!extent.getDefiningOp<gpu::PhysicalExprOp>())
      return emitOpError(
          "descriptor block shape must use declared physical expressions");
  unsigned choices = 0;
  unsigned allocators = 0;
  kernel.walk([&](TensorDescriptorChoiceOp) { ++choices; });
  kernel.walk([&](TensorDescriptorAllocatorOp) { ++allocators; });
  if (choices != 1 || allocators != 1)
    return emitOpError(
        "requires one descriptor choice and one allocator declaration");
  return success();
}

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

LogicalResult DescriptorLoadOp::verify() {
  return verifyDescriptorAccess(*this, getDescriptor(), getResult().getType(),
                                getOffsets(), getBoundaryAxes());
}

void DescriptorLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get());
}

LogicalResult DescriptorStoreOp::verify() {
  return verifyDescriptorAccess(*this, getDescriptor(), getValue().getType(),
                                getOffsets(), getBoundaryAxes());
}

void DescriptorStoreOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult SplitOp::verify() {
  auto source = cast<gpu::FragmentType>(getSource().getType());
  auto low = cast<gpu::FragmentType>(getLow().getType());
  auto high = cast<gpu::FragmentType>(getHigh().getType());
  if (low != high || source.getShape().size() != low.getShape().size() + 1 ||
      source.getElementType() != low.getElementType() ||
      source.getValidity() != low.getValidity() ||
      source.getOwner() != low.getOwner())
    return emitOpError(
        "requires two equal prefix fragments from one trailing pair axis");
  auto trailing = dyn_cast<gpu::PhysicalExprAttr>(
      source.getShape()[source.getShape().size() - 1]);
  if (!trailing ||
      trailing.getKind() !=
          static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
      trailing.getValue() != 2)
    return emitOpError("requires a constant trailing extent of two");
  if (!std::equal(low.getShape().begin(), low.getShape().end(),
                  source.getShape().begin()) ||
      !std::equal(low.getAxisMaps().begin(), low.getAxisMaps().end(),
                  source.getAxisMaps().begin()))
    return emitOpError(
        "results must preserve the source prefix shape and axis relations");
  return success();
}

} // namespace intent::triton

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.cpp.inc"

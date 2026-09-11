#include "Intent/Target/Triton/IR/TritonOps.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace intent::triton {
namespace {

LogicalResult verifyElementwiseCollective(Operation *owner, ValueRange inputs,
                                         ResultRange results, Region &combine,
                                         unsigned count, int64_t axis,
                                         bool scan) {
  if (!count || inputs.size() != 2 * count || results.size() != count ||
      !llvm::hasSingleElement(combine))
    return owner->emitOpError("requires paired sources/identities and one scalar combine block");
  auto first = dyn_cast<gpu::FragmentType>(inputs.front().getType());
  if (!first || axis < 0 || axis >= static_cast<int64_t>(first.getShape().size()))
    return owner->emitOpError("collective axis is outside its source fragment");
  Block &block = combine.front();
  auto yield = dyn_cast<gpu::YieldOp>(block.getTerminator());
  if (block.getNumArguments() != 2 * count || !yield || yield.getValues().size() != count)
    return owner->emitOpError("scalar combine arity disagrees with sources");
  for (unsigned i = 0; i < count; ++i) {
    auto source = dyn_cast<gpu::FragmentType>(inputs[i].getType());
    if (!source || source.getShape() != first.getShape() ||
        inputs[count + i].getType() != results[i].getType())
      return owner->emitOpError("native collective requires equal source shapes and exact identity/result types");
    Type element = source.getElementType();
    if (block.getArgument(i).getType() != element ||
        block.getArgument(count + i).getType() != element ||
        yield.getValues()[i].getType() != element)
      return owner->emitOpError("callback arguments and yields must be source element types");
    SmallVector<Attribute> shape, mappings;
    for (auto [position, extent] : llvm::enumerate(source.getShape())) {
      if (!scan && position == static_cast<unsigned>(axis))
        continue;
      shape.push_back(extent);
      auto mapping = cast<gpu::AxisMapAttr>(source.getAxisMaps()[position]);
      mappings.push_back(gpu::AxisMapAttr::get(owner->getContext(),
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDimensionId(),
          mappings.size(), mapping.getDerived()));
    }
    Type expected = shape.empty() ? element : Type(gpu::FragmentType::get(
        owner->getContext(), element, ArrayAttr::get(owner->getContext(), shape),
        ArrayAttr::get(owner->getContext(), mappings), source.getValidity(), source.getOwner()));
    if (results[i].getType() != expected)
      return owner->emitOpError("collective result must preserve its source free-axis relation");
  }
  for (Operation &operation : block) {
    if (operation.getNumRegions())
      return owner->emitOpError("native callback must be a closed elementwise block");
    for (Value operand : operation.getOperands())
      if (operand.getParentBlock() != &block)
        return owner->emitOpError("native callback cannot capture enclosing values");
    for (Value result : operation.getResults())
      if (isa<gpu::FragmentType>(result.getType()))
        return owner->emitOpError("native callback still contains a fragment value");
  }
  return success();
}

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

bool matchesStrideBinding(func::FuncOp kernel, gpu::ViewType view, unsigned axis,
                          Value value) {
  Attribute stride = view.getLayout().getStrides()[axis];
  if (auto constant = dyn_cast<IntegerAttr>(stride)) {
    auto operation = value.getDefiningOp<arith::ConstantIndexOp>();
    return operation && operation.value() == constant.getInt();
  }
  auto symbol = dyn_cast<StringAttr>(stride);
  auto argument = dyn_cast<BlockArgument>(value);
  if (!symbol || !argument || argument.getOwner() != &kernel.getBody().front())
    return false;
  unsigned index = argument.getArgNumber();
  auto name = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiNameAttr);
  auto kind = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiKindAttr);
  auto source = kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceABIAttr);
  auto sourceAxis =
      kernel.getArgAttrOfType<IntegerAttr>(index, gpu::sourceAxisAttr);
  return name == symbol && kind && kind.getValue() == "stride" && source &&
         source.getInt() == view.getAbiIndex() && sourceAxis &&
         sourceAxis.getInt() == axis && argument.getType().isIndex();
}

bool hasFlattenableDescriptorLayout(func::FuncOp kernel, gpu::ViewType view) {
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
  // The final two source axes become the descriptor row/column axes.  A padded
  // row stride is representable directly; only axes flattened into the row
  // coordinate must be mutually contiguous.
  for (unsigned axis = 0; axis + 2 < view.getRank(); ++axis) {
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

LogicalResult ReduceOp::verify() {
  if (getReverse())
    return emitOpError("native reduction does not reverse logical order");
  return verifyElementwiseCollective(getOperation(), getInputs(), getResults(),
                                    getCombine(), getSourceCount(), getAxis(), false);
}

LogicalResult ScanOp::verify() {
  return verifyElementwiseCollective(getOperation(), getInputs(), getResults(),
                                    getCombine(), getSourceCount(), getAxis(), true);
}

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
  if (getSelection() != "all_eligible")
    return emitOpError(
        "supports only an all-descriptor runtime eligibility contract");
  if (getConfigParameter().empty() || getEligibilityArgument().empty() ||
      getConfigParameter() == getEligibilityArgument())
    return emitOpError(
        "requires distinct non-empty config and runtime eligibility names");
  if (getDescriptors().empty())
    return emitOpError("requires a non-empty explicit descriptor set");
  llvm::SmallPtrSet<Operation *, 8> declared;
  for (Value value : getDescriptors()) {
    auto descriptor = value.getDefiningOp<TensorDescriptorOp>();
    if (!descriptor || descriptor->getParentOfType<func::FuncOp>() != kernel ||
        !declared.insert(descriptor.getOperation()).second)
      return emitOpError(
          "descriptor operands must uniquely enumerate this physical kernel's declarations");
  }
  unsigned descriptors = 0;
  bool completeDescriptorSet = true;
  kernel.walk([&](TensorDescriptorOp descriptor) {
    ++descriptors;
    if (!declared.contains(descriptor.getOperation()))
      completeDescriptorSet = false;
  });
  if (!completeDescriptorSet || descriptors != declared.size())
    return emitOpError(
        "descriptor operands must enumerate every descriptor declaration exactly once");
  for (unsigned index = 0; index < kernel.getNumArguments(); ++index) {
    auto name = kernel.getArgAttrOfType<StringAttr>(index, gpu::abiNameAttr);
    if (name && (name.getValue() == getConfigParameter() ||
                 name.getValue() == getEligibilityArgument()))
      return emitOpError(
          "descriptor config and eligibility names must not collide with the physical ABI");
  }
  bool parameterCollision = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    StringRef name = parameter.getParameter().getName().getValue();
    parameterCollision |= name == getConfigParameter() ||
                          name == getEligibilityArgument();
  });
  if (parameterCollision)
    return emitOpError(
        "descriptor config and eligibility names must not collide with physical parameters");
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
  if (getImplementation() != "torch_cuda_current_device")
    return emitOpError(
        "supports only the bound current-device Torch allocator implementation");
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
  if (getBaseLayout() != "flattened_row_major" || getPadding() != "zero" ||
      getAlignment() != 16 || getMinimumContiguousBytes() != 16 ||
      !getRequirePositiveShape() || !getRequirePositiveStrides() ||
      !getRequirePowerOfTwoBlockShape() ||
      getMaximumShapeExtent() != std::numeric_limits<int32_t>::max() ||
      getMaximumBlockElements() != (1 << 20))
    return emitOpError(
        "host tensor descriptor requires the complete Triton 3.6 flattened-row-major runtime contract");
  SmallVector<int64_t> expectedContiguousAxes;
  for (unsigned axis = 0; axis + 2 < base.getRank(); ++axis)
    expectedContiguousAxes.push_back(axis);
  if (getFlattenedContiguousAxes() !=
      ArrayRef<int64_t>(expectedContiguousAxes))
    return emitOpError(
        "runtime contract must name every source axis flattened into descriptor rows");
  if (getAlignedStrideAxes() != ArrayRef<int64_t>{
                                    static_cast<int64_t>(base.getRank()) - 2})
    return emitOpError(
        "runtime contract must align the represented descriptor row stride");
  if (getUnitStrideAxes() !=
      ArrayRef<int64_t>{static_cast<int64_t>(base.getRank()) - 1})
    return emitOpError(
        "runtime contract must require the represented column stride to be one");
  if (!hasFlattenableDescriptorLayout(kernel, base))
    return emitOpError(
        "host tensor descriptor base must carry a complete flattenable stride ABI");
  unsigned bitWidth = base.getElementType().getIntOrFloatBitWidth();
  if (bitWidth < 8 || bitWidth % 8 != 0)
    return emitOpError(
        "tensor descriptor element type must occupy whole bytes");
  auto lastDimension = getShape().back().getDefiningOp<gpu::DimOp>();
  SmallVector<unsigned> flattenedAxes;
  if (!lastDimension || lastDimension.getView() != getBase() ||
      lastDimension.getAxis() + 1 != base.getRank() ||
      !matchesStrideBinding(kernel, base, base.getRank() - 2,
                            getStrides().front()) ||
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
  bool declaredByChoice = false;
  kernel.walk([&](TensorDescriptorChoiceOp choice) {
    ++choices;
    declaredByChoice |= llvm::is_contained(choice.getDescriptors(), getResult());
  });
  kernel.walk([&](TensorDescriptorAllocatorOp) { ++allocators; });
  if (choices != 1 || allocators != 1 || !declaredByChoice)
    return emitOpError(
        "requires one explicit descriptor choice and one allocator declaration");
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

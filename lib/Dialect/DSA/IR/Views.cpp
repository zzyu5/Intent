#include "Intent/Dialect/DSA/IR/Views.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace intent::dsa {
namespace {

bool localCapacity(MemRefType type) {
  if (!type || !type.hasStaticShape() || type.getNumElements() <= 0 ||
      !type.getLayout().isIdentity() ||
      type.getMemorySpaceAsInt() != nramSpace)
    return false;
  Type element = type.getElementType();
  return element.isF16() || element.isBF16() || element.isF32() ||
         element.isInteger(1) || element.isInteger(32) || element.isInteger(64);
}

SmallVector<int64_t> contiguousStrides(MemRefType type) {
  SmallVector<int64_t> strides(type.getRank());
  int64_t stride = 1;
  for (int64_t axis = type.getRank(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= type.getDimSize(axis - 1);
  }
  return strides;
}

bool completeContiguousView(memref::ReinterpretCastOp view) {
  auto input = cast<MemRefType>(view.getSource().getType());
  auto output = view.getType();
  return input.hasStaticShape() && output.hasStaticShape() &&
         input.getLayout().isIdentity() && output.getLayout().isIdentity() &&
         input.getMemorySpace() == output.getMemorySpace() &&
         input.getElementType() == output.getElementType() &&
         input.getNumElements() == output.getNumElements() &&
         view.getOffsets().empty() && view.getSizes().empty() &&
         view.getStrides().empty() &&
         view.getStaticOffsets() == ArrayRef<int64_t>({0}) &&
         view.getStaticSizes() == output.getShape() &&
         view.getStaticStrides() == ArrayRef<int64_t>(contiguousStrides(output));
}

} // namespace

bool isCompleteLocalStorageView(Value value) {
  if (!localCapacity(dyn_cast<MemRefType>(value.getType()))) return false;
  while (true) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
    } else if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      if (!completeContiguousView(view))
        return false;
      value = view.getSource();
    } else {
      break;
    }
    if (!localCapacity(dyn_cast<MemRefType>(value.getType())))
      return false;
  }
  if (value.getDefiningOp<memref::AllocaOp>()) return true;
  auto formal = dyn_cast<BlockArgument>(value);
  return formal && isCollectiveBorrowedArgument(formal);
}

bool isCompleteStorageViewOf(Value value, Value origin) {
  if (!value || !origin || !isa<BaseMemRefType>(value.getType()) ||
      !isa<BaseMemRefType>(origin.getType()))
    return false;
  while (value != origin) {
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
    } else if (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) {
      if (!completeContiguousView(view))
        return false;
      value = view.getSource();
    } else {
      return false;
    }
  }
  return true;
}

FailureOr<Value> materializeCollectiveView(OpBuilder &builder, Location location,
                                          Value source, MemRefType requested) {
  auto original = dyn_cast<MemRefType>(source.getType());
  if (!isCompleteLocalStorageView(source) || !localCapacity(requested) ||
      requested.getElementType() != original.getElementType() ||
      requested.getNumElements() != original.getNumElements())
    return emitError(location,
                     "collective view must preserve complete contiguous local storage"),
           failure();
  if (source.getType() == requested) return source;
  Value backing = source;
  while (true) {
    if (auto view = backing.getDefiningOp<memref::ReinterpretCastOp>())
      backing = view.getSource();
    else if (auto cast = backing.getDefiningOp<memref::CastOp>())
      backing = cast.getSource();
    else
      break;
    if (backing.getType() == requested) return backing;
  }
  return builder.create<memref::ReinterpretCastOp>(
      location, requested, backing, int64_t(0), requested.getShape(),
      contiguousStrides(requested)).getResult();
}

} // namespace intent::dsa

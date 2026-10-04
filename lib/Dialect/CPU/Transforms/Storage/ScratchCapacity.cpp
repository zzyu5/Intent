#include "ScratchStorage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::cpu::detail {

std::optional<ScratchAllocation> scratchAllocation(Operation *operation,
                                                   StorageAnalysis &storage) {
  if (!isa<memref::AllocOp, memref::AllocaOp>(operation)) return std::nullopt;
  Value memory = operation->getResult(0);
  auto type = cast<MemRefType>(memory.getType());
  if (!type.getLayout().isIdentity()) return std::nullopt;
  auto aliases = storage.aliases(memory);
  if (!aliases.complete) return std::nullopt;
  bool stack = isa<memref::AllocaOp>(operation);
  memref::DeallocOp release;
  if (!stack) {
    auto lifetime = storage.lifetime(cast<memref::AllocOp>(operation));
    if (!lifetime) return std::nullopt;
    release = lifetime->end;
  } else {
    Operation *owner = operation->getParentOp();
    while (owner && !owner->hasTrait<OpTrait::AutomaticAllocationScope>())
      owner = owner->getParentOp();
    if (!owner) return std::nullopt;
  }
  Block *block = operation->getBlock();
  // In particular, a descriptor yielded through this block's loop backedge or
  // out of its region is not a per-iteration scratch value.
  for (Value alias : aliases.values) {
    if (alias == memory) continue;
    Operation *owner = alias.getDefiningOp();
    if (!owner) owner = cast<BlockArgument>(alias).getOwner()->getParentOp();
    if (!block->findAncestorOpInBlock(*owner)) return std::nullopt;
  }
  Operation *last = operation;
  for (Operation *user : aliases.users) {
    if (user == release) continue;
    if (isa<memref::DeallocOp, memref::ExtractAlignedPointerAsIndexOp>(user))
      return std::nullopt;
    Operation *anchor = block->findAncestorOpInBlock(*user);
    if (!anchor || anchor == operation || !operation->isBeforeInBlock(anchor))
      return std::nullopt;
    auto effects = storage.effects(user);
    if (!effects.complete) return std::nullopt;
    if (last == operation || last->isBeforeInBlock(anchor)) last = anchor;
  }
  if (last == operation) return std::nullopt;
  return ScratchAllocation{operation, memory, type, last, release,
                           std::move(aliases), stack};
}

std::optional<int64_t> scratchCapacity(const ScratchAllocation &scratch,
                                      int64_t byteLimit) {
  Type element = scratch.type.getElementType();
  if (!element.isIntOrIndexOrFloat()) return std::nullopt;
  int64_t bytes = element.isIndex() ? 8 : (element.getIntOrFloatBitWidth() + 7) / 8;
  if (bytes <= 0 || byteLimit <= 0) return std::nullopt;
  int64_t capacity = 1;
  for (int64_t axis = 0; axis < scratch.type.getRank(); ++axis) {
    auto extent = constantDimensionUpperBound(scratch.memory, axis);
    if (!extent || *extent < 0 || llvm::MulOverflow(capacity, *extent, capacity))
      return std::nullopt;
  }
  if (capacity <= 0 || capacity > byteLimit / bytes) return std::nullopt;
  return capacity;
}

int64_t scratchAlignment(const ScratchAllocation &scratch) {
  if (auto allocation = dyn_cast<memref::AllocOp>(scratch.operation))
    return allocation.getAlignment().value_or(0);
  return cast<memref::AllocaOp>(scratch.operation).getAlignment().value_or(0);
}

Value createScratchBacking(OpBuilder &builder, const ScratchAllocation &scratch,
                           int64_t capacity, int64_t alignment) {
  auto type = MemRefType::get({capacity}, scratch.type.getElementType(),
                             MemRefLayoutAttrInterface{},
                             scratch.type.getMemorySpace());
  if (scratch.stack) {
    auto allocation = builder.create<memref::AllocaOp>(scratch.operation->getLoc(), type);
    if (alignment) allocation.setAlignment(alignment);
    return allocation;
  }
  auto allocation = builder.create<memref::AllocOp>(scratch.operation->getLoc(), type);
  if (alignment) allocation.setAlignment(alignment);
  return allocation;
}

Value scratchDescriptor(OpBuilder &builder, const ScratchAllocation &scratch,
                        Value backing) {
  if (scratch.type.hasStaticShape() && backing.getType() == scratch.type)
    return backing;
  Location loc = scratch.operation->getLoc();
  ValueRange dynamic = scratch.stack
      ? ValueRange(cast<memref::AllocaOp>(scratch.operation).getDynamicSizes())
      : ValueRange(cast<memref::AllocOp>(scratch.operation).getDynamicSizes());
  SmallVector<OpFoldResult> sizes, strides(scratch.type.getRank());
  unsigned next = 0;
  for (int64_t extent : scratch.type.getShape())
    sizes.push_back(ShapedType::isDynamic(extent) ? OpFoldResult(dynamic[next++])
                                                : builder.getIndexAttr(extent));
  OpFoldResult stride = builder.getIndexAttr(1);
  for (int64_t axis = scratch.type.getRank(); axis-- > 0;) {
    strides[axis] = stride;
    if (!axis) break;
    auto lhs = getConstantIntValue(stride), rhs = getConstantIntValue(sizes[axis]);
    int64_t product;
    if (lhs && rhs && !llvm::MulOverflow(*lhs, *rhs, product))
      stride = builder.getIndexAttr(product);
    else {
      Value left = getValueOrCreateConstantIndexOp(builder, loc, stride);
      Value right = getValueOrCreateConstantIndexOp(builder, loc, sizes[axis]);
      stride = builder.createOrFold<arith::MulIOp>(loc, left, right);
    }
  }
  return builder.create<memref::ReinterpretCastOp>(loc, scratch.type, backing,
      builder.getIndexAttr(0), sizes, strides);
}

} // namespace intent::cpu::detail

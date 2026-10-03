#include "Intent/Dialect/CPU/Transforms/Vector/VectorReductions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cpu {

SmallVector<Value> horizontalReduce(
    OpBuilder &builder, Location location, ValueRange values,
    llvm::function_ref<SmallVector<Value>(ValueRange, ValueRange, int64_t)> combine) {
  assert(!values.empty());
  auto type = cast<VectorType>(values.front().getType());
  int64_t width = type.getDimSize(0);
  assert(type.getRank() == 1 && !type.isScalable() && llvm::isPowerOf2_64(width));
  for (Value value : values) {
    auto component = cast<VectorType>(value.getType());
    assert(component.getShape() == type.getShape() && !component.isScalable());
    (void)component;
  }
  SmallVector<Value> partial(values);
  for (int64_t count = width; count > 1; count /= 2) {
    SmallVector<int64_t> even, odd;
    for (int64_t lane = 0; lane < count; lane += 2) {
      even.push_back(lane);
      odd.push_back(lane + 1);
    }
    SmallVector<Value> left, right;
    for (Value value : partial) {
      left.push_back(builder.create<vector::ShuffleOp>(location, value, value, even));
      right.push_back(builder.create<vector::ShuffleOp>(location, value, value, odd));
    }
    partial = combine(left, right, count / 2);
  }
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  for (Value &value : partial)
    value = builder.create<vector::ExtractElementOp>(location, value, zero);
  return partial;
}

} // namespace intent::cpu

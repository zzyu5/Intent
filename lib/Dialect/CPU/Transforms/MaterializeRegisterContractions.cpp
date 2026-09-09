#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace intent::cpu {

LogicalResult materializeRegisterContractions(func::FuncOp function) {
  SmallVector<linalg::GenericOp> operations;
  function.walk([&](linalg::GenericOp operation) {
    if (operation->hasAttr("intent_cpu.microtile")) operations.push_back(operation);
  });
  for (auto operation : operations) {
    auto tile = operation->getAttrOfType<MicrotileAttr>("intent_cpu.microtile");
    auto outputType = cast<MemRefType>(operation.getOutputs()[0].getType());
    if (!isMatrixContraction(operation) || outputType.getShape() !=
        ArrayRef<int64_t>({tile.getRows(), tile.getColumns()}))
      return operation.emitError("CPU register contraction does not match its formed microtile");
    Value lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
    Value output = operation.getOutputs()[0];
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    int64_t width = tile.getVectorWidth(), columns = tile.getColumns() / width;
    auto vectorType = VectorType::get({width}, b.getF32Type());
    SmallVector<Value> rows, offsets, accumulators;
    for (int64_t row = 0; row < tile.getRows(); ++row) rows.push_back(index(b, loc, row));
    for (int64_t column = 0; column < columns; ++column) offsets.push_back(index(b, loc, column * width));
    auto load = [&](Value source, Value row, Value column) -> Value {
      if (width == 1) return b.create<memref::LoadOp>(loc, source, ValueRange{row, column});
      return b.create<vector::LoadOp>(loc, vectorType, source, ValueRange{row, column});
    };
    for (Value row : rows)
      for (Value offset : offsets) accumulators.push_back(load(output, row, offset));
    Value depth = b.create<memref::DimOp>(loc, lhs, 1);
    auto reduction = b.create<scf::ForOp>(loc, index(b, loc, 0), depth, index(b, loc, 1), accumulators);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(reduction.getBody());
      Value k = reduction.getInductionVar();
      SmallVector<Value> right;
      for (Value offset : offsets) right.push_back(load(rhs, k, offset));
      SmallVector<Value> next;
      for (auto [number, row] : llvm::enumerate(rows)) {
        Value left = b.create<memref::LoadOp>(loc, lhs, ValueRange{row, k});
        if (width != 1) left = b.create<vector::BroadcastOp>(loc, vectorType, left);
        for (int64_t column = 0; column < columns; ++column)
          next.push_back(b.create<math::FmaOp>(loc, left, right[column],
              reduction.getRegionIterArgs()[number * columns + column]));
      }
      b.create<scf::YieldOp>(loc, next);
    }
    for (auto [number, row] : llvm::enumerate(rows))
      for (int64_t column = 0; column < columns; ++column) {
        Value value = reduction.getResult(number * columns + column);
        if (width == 1) b.create<memref::StoreOp>(loc, value, output, ValueRange{row, offsets[column]});
        else b.create<vector::StoreOp>(loc, value, output, ValueRange{row, offsets[column]});
      }
    operation.erase();
  }
  return success();
}

}

#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace intent::cpu {
namespace {

Value subview(OpBuilder &b, Location loc, Value source,
              ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  auto type = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));
  return b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
}

LogicalResult block(linalg::GenericOp operation, const Configuration &config) {
  Value lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
  Value output = operation.getOutputs()[0];
  linalg::FillOp initialization;
  for (Operation *user : output.getUsers()) {
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (initialization || fill->getBlock() != operation->getBlock() ||
          !fill->isBeforeInBlock(operation))
        return operation.emitError("CPU contraction requires one dominating initialization");
      initialization = fill;
    } else if (user != operation && !isa<memref::DeallocOp, memref::DimOp>(user)) {
      return operation.emitError("CPU contraction output has an intervening or escaping use");
    }
  }
  if (!initialization)
    return operation.emitError("CPU contraction accumulator initialization is missing");
  Value initial = initialization.getInputs()[0];
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value mSize = b.create<memref::DimOp>(loc, lhs, 0);
  Value nSize = b.create<memref::DimOp>(loc, rhs, 1);
  Value kSize = b.create<memref::DimOp>(loc, lhs, 1);
  Value bm = index(b, loc, config.tileM), bn = index(b, loc, config.tileN);
  Value bk = index(b, loc, config.tileK);
  Value mTasks = b.create<arith::CeilDivSIOp>(loc, mSize, bm);
  Value nTasks = b.create<arith::CeilDivSIOp>(loc, nSize, bn);
  auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero, zero},
      ValueRange{mTasks, nTasks}, ValueRange{one, one});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(parallel.getBody());
    Value mBegin = multiply(b, loc, parallel.getInductionVars()[0], bm);
    Value nBegin = multiply(b, loc, parallel.getInductionVars()[1], bn);
    Value mCount = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, mSize, mBegin), bm);
    Value nCount = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, nSize, nBegin), bn);
    Value mEnd = add(b, loc, mBegin, mCount);
    Value outputTile = subview(b, loc, output, {mBegin, nBegin}, {mCount, nCount});
    b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{outputTile});
    Value packed = b.create<memref::AllocaOp>(loc,
        MemRefType::get({config.tileK, config.tileN}, b.getF32Type()));
    loop(b, loc, zero, kSize, config.tileK, [&](Value kBegin) {
      Value depth = b.create<arith::MinSIOp>(loc,
          b.create<arith::SubIOp>(loc, kSize, kBegin), bk);
      Value source = subview(b, loc, rhs, {kBegin, nBegin}, {depth, nCount});
      Value destination = subview(b, loc, packed,
          {b.getIndexAttr(0), b.getIndexAttr(0)}, {depth, nCount});
      b.create<memref::CopyOp>(loc, source, destination);
      auto micro = [&](Value m, Value n, int64_t rows, int64_t columns, int64_t width) {
        Value left = subview(b, loc, lhs, {m, kBegin}, {b.getIndexAttr(rows), depth});
        Value right = subview(b, loc, packed,
            {b.getIndexAttr(0), n}, {depth, b.getIndexAttr(columns)});
        Value out = subview(b, loc, output,
            {m, add(b, loc, nBegin, n)}, {b.getIndexAttr(rows), b.getIndexAttr(columns)});
        auto contract = b.create<linalg::GenericOp>(loc, ValueRange{left, right},
            ValueRange{out}, operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
            [](OpBuilder &nested, Location loc, ValueRange arguments) {
              Value value = nested.create<math::FmaOp>(loc, arguments[0], arguments[1], arguments[2]);
              nested.create<linalg::YieldOp>(loc, value);
            });
        contract->setAttr("intent_cpu.microtile", MicrotileAttr::get(b.getContext(), rows, columns, width));
      };
      auto columns = [&](Value m, int64_t rows) {
        int64_t registerColumns = config.vectorWidth * config.microN;
        Value registerStep = index(b, loc, registerColumns);
        Value full = multiply(b, loc, b.create<arith::DivSIOp>(loc, nCount, registerStep), registerStep);
        loop(b, loc, zero, full, registerColumns,
            [&](Value n) { micro(m, n, rows, registerColumns, config.vectorWidth); });
        Value width = index(b, loc, config.vectorWidth);
        Value vectorEnd = multiply(b, loc, b.create<arith::DivSIOp>(loc, nCount, width), width);
        loop(b, loc, full, vectorEnd, config.vectorWidth,
            [&](Value n) { micro(m, n, rows, config.vectorWidth, config.vectorWidth); });
        loop(b, loc, vectorEnd, nCount, 1, [&](Value n) { micro(m, n, rows, 1, 1); });
      };
      Value mr = index(b, loc, config.microM);
      Value fullM = add(b, loc, mBegin, multiply(b, loc, b.create<arith::DivSIOp>(loc, mCount, mr), mr));
      loop(b, loc, mBegin, fullM, config.microM, [&](Value m) { columns(m, config.microM); });
      loop(b, loc, fullM, mEnd, 1, [&](Value m) { columns(m, 1); });
    });
  }
  initialization.erase();
  operation.erase();
  return success();
}

}

LogicalResult blockContractions(func::FuncOp function, const Configuration &configuration) {
  SmallVector<linalg::GenericOp> contractions;
  function.walk([&](linalg::GenericOp operation) {
    if (isMatrixContraction(operation) && !operation->hasAttr("intent_cpu.microtile"))
      contractions.push_back(operation);
  });
  for (linalg::GenericOp operation : contractions)
    if (failed(block(operation, configuration))) return failure();
  return success();
}

}

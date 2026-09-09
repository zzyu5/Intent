#include "Intent/Transforms/CPU/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool isContraction(linalg::GenericOp operation) {
  if (operation.getInputs().size() != 2 || operation.getOutputs().size() != 1)
    return false;
  AffineExpr m, n, k;
  bindDims(operation.getContext(), m, n, k);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {m, k}, operation.getContext()),
      AffineMap::get(3, 0, {k, n}, operation.getContext()),
      AffineMap::get(3, 0, {m, n}, operation.getContext())};
  if (operation.getIndexingMapsArray() != maps ||
      operation.getIteratorTypesArray() != SmallVector<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction}) return false;
  Block &body = operation.getRegion().front();
  auto fma = body.getTerminator()->getOperand(0).getDefiningOp<math::FmaOp>();
  return fma && fma.getA() == body.getArgument(0) &&
      fma.getB() == body.getArgument(1) && fma.getC() == body.getArgument(2);
}

LogicalResult block(linalg::GenericOp operation, const Configuration &config) {
  if (!isContraction(operation))
    return operation.emitError("CPU blocking requires explicit paired-axis multiply-add semantics");
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
  Location loc = operation.getLoc();
  OpBuilder b(operation);
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
    Value mEnd = b.create<arith::MinSIOp>(loc, add(b, loc, mBegin, bm), mSize);
    Value nEnd = b.create<arith::MinSIOp>(loc, add(b, loc, nBegin, bn), nSize);
    Value nCount = b.create<arith::SubIOp>(loc, nEnd, nBegin);
    Value packed = b.create<memref::AllocaOp>(loc,
        MemRefType::get({config.tileK, config.tileN}, b.getF32Type()));
    loop(b, loc, mBegin, mEnd, 1, [&](Value m) {
      loop(b, loc, nBegin, nEnd, 1, [&](Value n) {
        b.create<memref::StoreOp>(loc, initial, output, ValueRange{m, n});
      });
    });
    loop(b, loc, zero, kSize, config.tileK, [&](Value kBegin) {
      Value depth = b.create<arith::MinSIOp>(loc,
          b.create<arith::SubIOp>(loc, kSize, kBegin), bk);
      loop(b, loc, zero, depth, 1, [&](Value k) {
        loop(b, loc, zero, nCount, 1, [&](Value n) {
          Value value = b.create<memref::LoadOp>(loc, rhs,
              ValueRange{add(b, loc, kBegin, k), add(b, loc, nBegin, n)});
          b.create<memref::StoreOp>(loc, value, packed, ValueRange{k, n});
        });
      });
      auto micro = [&](Value m, Value n, int64_t rows, int64_t width) {
        SmallVector<Value> accumulators, rowCoordinates;
        auto vectorType = VectorType::get({width}, b.getF32Type());
        Value outputN = add(b, loc, nBegin, n);
        for (int64_t row = 0; row < rows; ++row) {
          Value currentM = add(b, loc, m, index(b, loc, row));
          rowCoordinates.push_back(currentM);
          accumulators.push_back(width == 1
              ? Value(b.create<memref::LoadOp>(loc, output, ValueRange{currentM, outputN}))
              : Value(b.create<vector::LoadOp>(loc, vectorType, output, ValueRange{currentM, outputN})));
        }
        auto reduction = b.create<scf::ForOp>(loc, zero, depth, one, accumulators);
        {
          OpBuilder::InsertionGuard reductionGuard(b);
          b.setInsertionPointToStart(reduction.getBody());
          Value k = reduction.getInductionVar();
          Value right = width == 1
              ? Value(b.create<memref::LoadOp>(loc, packed, ValueRange{k, n}))
              : Value(b.create<vector::LoadOp>(loc, vectorType, packed, ValueRange{k, n}));
          SmallVector<Value> next;
          for (auto [row, currentM] : llvm::enumerate(rowCoordinates)) {
            Value left = b.create<memref::LoadOp>(loc, lhs,
                ValueRange{currentM, add(b, loc, kBegin, k)});
            if (width != 1) left = b.create<vector::BroadcastOp>(loc, vectorType, left);
            next.push_back(b.create<math::FmaOp>(loc, left, right, reduction.getRegionIterArgs()[row]));
          }
          b.create<scf::YieldOp>(loc, next);
        }
        for (auto [row, currentM] : llvm::enumerate(rowCoordinates)) {
          if (width == 1)
            b.create<memref::StoreOp>(loc, reduction.getResult(row), output, ValueRange{currentM, outputN});
          else
            b.create<vector::StoreOp>(loc, reduction.getResult(row), output, ValueRange{currentM, outputN});
        }
      };
      auto columns = [&](Value m, int64_t rows) {
        Value width = index(b, loc, config.vectorWidth);
        Value full = multiply(b, loc, b.create<arith::DivSIOp>(loc, nCount, width), width);
        loop(b, loc, zero, full, config.vectorWidth, [&](Value n) {
          micro(m, n, rows, config.vectorWidth);
        });
        loop(b, loc, full, nCount, 1, [&](Value n) { micro(m, n, rows, 1); });
      };
      Value mr = index(b, loc, config.microM);
      Value mCount = b.create<arith::SubIOp>(loc, mEnd, mBegin);
      Value fullM = add(b, loc, mBegin,
          multiply(b, loc, b.create<arith::DivSIOp>(loc, mCount, mr), mr));
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
  function.walk([&](linalg::GenericOp op) { contractions.push_back(op); });
  for (linalg::GenericOp operation : contractions)
    if (failed(block(operation, configuration))) return failure();
  return success();
}

}

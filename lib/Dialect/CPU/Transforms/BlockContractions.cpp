#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"

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
  if (!matchPattern(initial, m_PosZeroFloat()))
    return operation.emitError("CPU contraction blocking requires the closed zero-initialized contraction; splitting a nonzero fused accumulator is not implemented");
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
    Value empty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, kSize, zero);
    auto initializeEmpty = b.create<scf::IfOp>(loc, empty, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(initializeEmpty.thenBlock());
      b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{outputTile});
    }
    auto kBlock = [&](Value kBegin, bool first) {
      Value depth = b.create<arith::MinSIOp>(loc,
          b.create<arith::SubIOp>(loc, kSize, kBegin), bk);
      auto micro = [&](Value packed, Value m, Value n, int64_t rows, int64_t columns, int64_t width) {
        Value left = subview(b, loc, lhs, {m, kBegin}, {b.getIndexAttr(rows), depth});
        Value right = subview(b, loc, packed,
            {b.getIndexAttr(0), b.getIndexAttr(0)}, {depth, b.getIndexAttr(columns)});
        Value out = subview(b, loc, output,
            {m, add(b, loc, nBegin, n)}, {b.getIndexAttr(rows), b.getIndexAttr(columns)});
        auto partial = b.create<memref::AllocaOp>(loc,
            MemRefType::get({rows, columns}, b.getF32Type()));
        partial.setAlignment(width * 4);
        b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{partial});
        auto contract = b.create<linalg::GenericOp>(loc, ValueRange{left, right},
            ValueRange{partial}, operation.getIndexingMapsArray(), operation.getIteratorTypesArray(),
            [](OpBuilder &nested, Location loc, ValueRange arguments) {
              Value value = nested.create<math::FmaOp>(loc, arguments[0], arguments[1], arguments[2]);
              nested.create<linalg::YieldOp>(loc, value);
            });
        contract->setAttr("intent_cpu.microtile", MicrotileAttr::get(b.getContext(), rows, columns, width));
        // K blocking selects adjacent partials inside the original closed
        // contraction. Their merge is explicit; a provider must not split a
        // nonzero FMA accumulator after this numerical boundary has formed.
        auto identity = b.getMultiDimIdentityMap(2);
        SmallVector<Value> inputs{partial};
        if (!first) inputs.insert(inputs.begin(), out);
        b.create<linalg::GenericOp>(loc, inputs, ValueRange{out},
            SmallVector<AffineMap>(inputs.size() + 1, identity),
            SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
            [first](OpBuilder &nested, Location loc, ValueRange arguments) {
              Value value = arguments[0];
              if (!first) value = nested.create<arith::AddFOp>(loc, value, arguments[1]);
              nested.create<linalg::YieldOp>(loc, value);
            });
      };
      struct RowRegion { int64_t rows; Value begin, end; };
      SmallVector<RowRegion> rowRegions;
      Value rowBegin = mBegin;
      for (int64_t rows : {config.microM, int64_t{4}, int64_t{2}, int64_t{1}}) {
        if (!rowRegions.empty() && rows >= rowRegions.back().rows) continue;
        Value end = mEnd;
        if (rows != 1) {
          Value count = b.create<arith::SubIOp>(loc, mEnd, rowBegin);
          Value step = index(b, loc, rows);
          end = add(b, loc, rowBegin, multiply(b, loc, b.create<arith::DivSIOp>(loc, count, step), step));
        }
        rowRegions.push_back({rows, rowBegin, end});
        rowBegin = end;
      }
      auto rows = [&](Value n, int64_t columns, int64_t width) {
        auto packed = b.create<memref::AllocaOp>(loc,
            MemRefType::get({config.tileK, columns}, b.getF32Type()));
        packed.setAlignment(width * 4);
        Value source = subview(b, loc, rhs, {kBegin, add(b, loc, nBegin, n)},
            {depth, b.getIndexAttr(columns)});
        Value destination = subview(b, loc, packed,
            {b.getIndexAttr(0), b.getIndexAttr(0)}, {depth, b.getIndexAttr(columns)});
        b.create<memref::CopyOp>(loc, source, destination);
        for (auto region : rowRegions)
          loop(b, loc, region.begin, region.end, region.rows,
              [&](Value m) { micro(packed, m, n, region.rows, columns, width); });
      };
      // Keep one packed B micro-panel live across the free M traversal.
      // Only disjoint output tiles are interchanged; K partials stay ordered.
      Value columnBegin = zero;
      int64_t previousVectors = config.microN + 1;
      for (int64_t vectors : {config.microN, int64_t{2}, int64_t{1}}) {
        if (vectors >= previousVectors) continue;
        int64_t columns = config.vectorWidth * vectors;
        Value step = index(b, loc, columns);
        Value remaining = b.create<arith::SubIOp>(loc, nCount, columnBegin);
        Value end = add(b, loc, columnBegin,
            multiply(b, loc, b.create<arith::DivSIOp>(loc, remaining, step), step));
        loop(b, loc, columnBegin, end, columns,
            [&](Value n) { rows(n, columns, config.vectorWidth); });
        columnBegin = end;
        previousVectors = vectors;
      }
      loop(b, loc, columnBegin, nCount, 1, [&](Value n) { rows(n, 1, 1); });
    };
    Value firstEnd = b.create<arith::MinSIOp>(loc, kSize, bk);
    loop(b, loc, zero, firstEnd, config.tileK, [&](Value k) { kBlock(k, true); });
    loop(b, loc, firstEnd, kSize, config.tileK, [&](Value k) { kBlock(k, false); });
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

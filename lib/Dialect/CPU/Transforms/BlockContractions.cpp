#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
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

LogicalResult block(linalg::GenericOp operation, const Configuration &config,
                    const ImplementationRegistry &implementations) {
  auto implementation = implementations.lookup(operation);
  if (failed(implementation)) return failure();
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (!(*implementation)->formTile)
    return operation.emitError("selected contraction implementation has no tile expansion");
  auto shared = operation->getParentOfType<func::FuncOp>()->getAttrOfType<ConfigurationAttr>(
      "intent_cpu.configuration");
  Value lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
  Value output = operation.getOutputs()[0];
  linalg::FillOp initialization;
  for (Operation *user : output.getUsers()) {
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (fill->getBlock() == operation->getBlock() && fill->isBeforeInBlock(operation) &&
          (!initialization || initialization->isBeforeInBlock(fill)))
        initialization = fill;
    }
  }
  if (!initialization)
    return operation.emitError("CPU contraction accumulator initialization is missing");
  PhysicalProgramAnalysis analysis(operation->getParentOfType<func::FuncOp>());
  Value root = analysis.storageRoot(output);
  for (Operation *between = initialization->getNextNode(); between != operation;
       between = between->getNextNode())
    for (auto access : analysis.accesses(between))
      if (analysis.storageRoot(access.memory) == root)
        return operation.emitError("CPU contraction initialization has an intervening memory access");
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
  LogicalResult status = success();
  auto tile = [&](Value m, Value n) {
    Value mBegin = multiply(b, loc, m, bm);
    Value nBegin = multiply(b, loc, n, bn);
    Value mCount = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, mSize, mBegin), bm);
    Value nCount = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, nSize, nBegin), bn);
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
      if (failed((*implementation)->formTile(b, operation,
              {lhs, rhs, output, initial, mBegin, mCount, nBegin, nCount, kBegin, depth, first},
              shared, binding))) status = failure();
    };
    Value firstEnd = b.create<arith::MinSIOp>(loc, kSize, bk);
    loop(b, loc, zero, firstEnd, config.tileK, [&](Value k) { kBlock(k, true); });
    loop(b, loc, firstEnd, kSize, config.tileK, [&](Value k) { kBlock(k, false); });
  };
  if (operation->getParentOfType<scf::ForOp>() || operation->getParentOfType<scf::ParallelOp>()) {
    loop(b, loc, zero, mTasks, 1, [&](Value m) {
      loop(b, loc, zero, nTasks, 1, [&](Value n) { tile(m, n); });
    });
  } else {
    auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero, zero},
        ValueRange{mTasks, nTasks}, ValueRange{one, one});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(parallel.getBody());
    tile(parallel.getInductionVars()[0], parallel.getInductionVars()[1]);
  }
  if (failed(status)) return failure();
  initialization.erase();
  operation.erase();
  return success();
}

}

LogicalResult blockContractions(func::FuncOp function, const Configuration &configuration,
                                const ImplementationRegistry &implementations) {
  SmallVector<linalg::GenericOp> contractions;
  function.walk([&](linalg::GenericOp operation) {
    if (isMatrixContraction(operation) && !operation->hasAttr("intent_cpu.microtile"))
      contractions.push_back(operation);
  });
  for (linalg::GenericOp operation : contractions)
    if (failed(block(operation, configuration, implementations))) return failure();
  return success();
}

}

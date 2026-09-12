#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "ImplementationInputs.h"
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
                    const ImplementationRegistry &implementations, ImplementationInputs &inputs) {
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
  bool keepInitialization = (*implementation)->contraction.completePrivateInitialization &&
      isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(root.getDefiningOp());
  for (Operation *between = initialization->getNextNode(); between != operation;
       between = between->getNextNode())
    for (auto access : analysis.accesses(between))
      if (analysis.storageRoot(access.memory) == root)
        return operation.emitError("CPU contraction initialization has an intervening memory access");
  Value initial = initialization.getInputs()[0];
  if (!(initial.getType().isF32() ? matchPattern(initial, m_PosZeroFloat()) : matchPattern(initial, m_Zero())))
    return operation.emitError("CPU contraction blocking requires the closed zero-initialized contraction; splitting a nonzero fused accumulator is not implemented");
  auto requirements = (*implementation)->inputs ? (*implementation)->inputs(operation, shared, binding)
      : SmallVector<InputRequirement>{};
  auto supplies = inputs.prepare(operation, requirements);
  if (failed(supplies)) return failure();
  int64_t groupM = 0, groupN = 0;
  for (auto requirement : requirements) {
    if (requirement.reuse != InputReuse::Group) continue;
    auto map = operation.getIndexingMapsArray()[requirement.operand];
    auto axis = dyn_cast<AffineDimExpr>(map.getResult(requirement.panelAxis));
    if (!axis || axis.getPosition() > 1)
      return operation.emitError("input supply grouping requires a free contraction axis");
    auto &group = axis.getPosition() == 0 ? groupM : groupN;
    group = group ? std::min(group, requirement.panelSize) : requirement.panelSize;
  }
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
  bool staticParallel = (*implementation)->contraction.staticParallelExtent;
  Value fullM, fullN;
  if (staticParallel) {
    fullM = b.create<arith::DivSIOp>(loc, mSize, bm);
    fullN = b.create<arith::DivSIOp>(loc, nSize, bn);
    mTasks = add(b, loc, fullM, b.create<arith::RemSIOp>(loc, mSize, bm));
    nTasks = add(b, loc, fullN, b.create<arith::RemSIOp>(loc, nSize, bn));
  }
  LogicalResult status = success();
  auto tile = [&](Value mBegin, Value nBegin, Value mCount, Value nCount) {
    Value outputTile = subview(b, loc, output, {mBegin, nBegin}, {mCount, nCount});
    Value empty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, kSize, zero);
    if (!keepInitialization) {
      auto initializeEmpty = b.create<scf::IfOp>(loc, empty, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(initializeEmpty.thenBlock());
      b.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{outputTile});
    }
    auto kBlock = [&](Value kBegin, Value depth, bool first) {
      auto group = [&](Value begin, Value extent, int64_t size,
                       const std::function<void(Value, Value)> &body) {
        if (!size) { body(begin, extent); return; }
        Value step = index(b, loc, size);
        Value full = multiply(b, loc, b.create<arith::DivSIOp>(loc, extent, step), step);
        loop(b, loc, zero, full, size, [&](Value offset) { body(add(b, loc, begin, offset), step); });
        Value tail = b.create<arith::SubIOp>(loc, extent, full);
        auto branch = b.create<scf::IfOp>(loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, tail, zero), false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.thenBlock());
        body(add(b, loc, begin, full), tail);
      };
      group(mBegin, mCount, groupM, [&](Value m, Value rows) {
        group(nBegin, nCount, groupN, [&](Value n, Value columns) {
          ContractionTile tile{lhs, rhs, output, initial, m, rows, n, columns, kBegin, depth, first, {}};
          auto local = inputs.prepareGroup(b, operation, tile, shared, requirements);
          if (failed(local)) { status = failure(); return; }
          llvm::append_range(*local, *supplies);
          tile.inputs = *local;
          if (failed((*implementation)->formTile(b, operation, tile, shared, binding))) status = failure();
        });
      });
    };
    if ((*implementation)->contraction.staticReductionExtent) {
      Value fullEnd = b.create<arith::SubIOp>(loc, kSize, b.create<arith::RemSIOp>(loc, kSize, bk));
      Value firstEnd = b.create<arith::MinSIOp>(loc, fullEnd, bk);
      loop(b, loc, zero, firstEnd, config.tileK, [&](Value k) { kBlock(k, bk, true); });
      loop(b, loc, bk, fullEnd, config.tileK, [&](Value k) { kBlock(k, bk, false); });
      Value noFullTile = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, fullEnd, zero);
      Value firstTailEnd = b.create<arith::SelectOp>(loc, noFullTile,
          b.create<arith::MinSIOp>(loc, kSize, one), zero);
      loop(b, loc, zero, firstTailEnd, 1, [&](Value k) { kBlock(k, one, true); });
      Value tailBegin = b.create<arith::MaxSIOp>(loc, fullEnd, one);
      loop(b, loc, tailBegin, kSize, 1, [&](Value k) { kBlock(k, one, false); });
    } else {
      Value firstEnd = b.create<arith::MinSIOp>(loc, kSize, bk);
      auto dynamicBlock = [&](Value k, bool first) {
        Value depth = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, kSize, k), bk);
        kBlock(k, depth, first);
      };
      loop(b, loc, zero, firstEnd, config.tileK, [&](Value k) { dynamicBlock(k, true); });
      loop(b, loc, firstEnd, kSize, config.tileK, [&](Value k) { dynamicBlock(k, false); });
    }
  };
  auto emitTile = [&](Value m, Value n) {
    if (!staticParallel) {
      Value mBegin = multiply(b, loc, m, bm), nBegin = multiply(b, loc, n, bn);
      Value mCount = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, mSize, mBegin), bm);
      Value nCount = b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, nSize, nBegin), bn);
      tile(mBegin, nBegin, mCount, nCount);
      return;
    }
    auto bounded = [&](Value ordinal, Value full, Value block,
                       const std::function<void(Value, Value)> &body) {
      Value complete = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ordinal, full);
      auto branch = b.create<scf::IfOp>(loc, complete, true);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      body(multiply(b, loc, ordinal, block), block);
      b.setInsertionPointToStart(branch.elseBlock());
      Value begin = add(b, loc, multiply(b, loc, full, block), b.create<arith::SubIOp>(loc, ordinal, full));
      body(begin, one);
    };
    bounded(m, fullM, bm, [&](Value mBegin, Value mCount) {
      bounded(n, fullN, bn, [&](Value nBegin, Value nCount) { tile(mBegin, nBegin, mCount, nCount); });
    });
  };
  if (operation->getParentOfType<scf::ForOp>() || operation->getParentOfType<scf::ParallelOp>()) {
    loop(b, loc, zero, mTasks, 1, [&](Value m) {
      loop(b, loc, zero, nTasks, 1, [&](Value n) { emitTile(m, n); });
    });
  } else {
    auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero, zero},
        ValueRange{mTasks, nTasks}, ValueRange{one, one});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(parallel.getBody());
    emitTile(parallel.getInductionVars()[0], parallel.getInductionVars()[1]);
  }
  if (failed(status)) return failure();
  if (!keepInitialization) initialization.erase();
  operation.erase();
  return success();
}

}

LogicalResult blockContractions(func::FuncOp function, const Configuration &configuration,
                                const ImplementationRegistry &implementations) {
  SmallVector<linalg::GenericOp> contractions;
  ImplementationInputs inputs(function);
  function.walk([&](linalg::GenericOp operation) {
    if (isMatrixContraction(operation) && !operation->hasAttr("intent_cpu.microtile"))
      contractions.push_back(operation);
  });
  for (linalg::GenericOp operation : contractions)
    if (failed(block(operation, configuration, implementations, inputs))) return failure();
  return success();
}

}

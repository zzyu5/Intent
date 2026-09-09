#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

LogicalResult partition(scf::ParallelOp root, int64_t grain) {
  SmallVector<scf::ParallelOp> nest;
  scf::ParallelOp current = root;
  while (current) {
    if (current.getNumResults()) return current.emitError("CPU task reduction is not implemented");
    nest.push_back(current);
    scf::ParallelOp child;
    for (Operation &operation : current.getBody()->without_terminator()) {
      if (auto parallel = dyn_cast<scf::ParallelOp>(&operation)) {
        if (child) return root.emitError("CPU task flattening requires one nested workset");
        child = parallel;
      }
    }
    if (child) {
      for (Operation &operation : current.getBody()->without_terminator())
        if (&operation != child && (!isMemoryEffectFree(&operation) || operation.getNumRegions()))
          return root.emitError("CPU task flattening cannot duplicate effects around a nested workset");
    }
    current = child;
  }
  SmallVector<Value> extents, coordinates;
  DominanceInfo dominance(root->getParentOfType<func::FuncOp>());
  for (scf::ParallelOp parallel : nest) {
    for (auto [begin, end, step, iv] : llvm::zip(parallel.getLowerBound(), parallel.getUpperBound(),
                                               parallel.getStep(), parallel.getInductionVars())) {
      if (!matchPattern(begin, m_Zero()) || !matchPattern(step, m_One()) ||
          !dominance.dominates(end, root))
        return parallel.emitError("CPU tasks require a rectangular zero-based unit-step workset");
      extents.push_back(end);
      coordinates.push_back(iv);
    }
  }
  OpBuilder b(root);
  Location loc = root.getLoc();
  Value zero = index(b, loc, 0), one = index(b, loc, 1);
  Value size = one;
  for (Value extent : extents) size = multiply(b, loc, size, extent);
  Value chunk = index(b, loc, grain);
  Value taskCount = b.create<arith::CeilDivSIOp>(loc, size, chunk);
  auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{taskCount}, ValueRange{one});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(tasks.getBody());
    Value begin = multiply(b, loc, tasks.getInductionVars()[0], chunk);
    Value end = b.create<arith::MinSIOp>(loc, add(b, loc, begin, chunk), size);
    loop(b, loc, begin, end, 1, [&](Value linear) {
      IRMapping mapping;
      Value remainder = linear;
      for (int64_t axis = extents.size() - 1; axis >= 0; --axis) {
        Value coordinate = axis == 0 ? remainder
            : Value(b.create<arith::RemSIOp>(loc, remainder, extents[axis]));
        mapping.map(coordinates[axis], coordinate);
        if (axis) remainder = b.create<arith::DivSIOp>(loc, remainder, extents[axis]);
      }
      std::function<void(scf::ParallelOp)> cloneBody = [&](scf::ParallelOp parallel) {
        for (Operation &operation : parallel.getBody()->without_terminator()) {
          if (auto child = dyn_cast<scf::ParallelOp>(&operation)) cloneBody(child);
          else b.clone(operation, mapping);
        }
      };
      cloneBody(root);
    });
  }
  root.erase();
  return success();
}

}

LogicalResult partitionTasks(func::FuncOp function, int64_t grain) {
  SmallVector<scf::ParallelOp> roots;
  function.walk([&](scf::ParallelOp operation) {
    if (!operation->getParentOfType<scf::ParallelOp>()) roots.push_back(operation);
  });
  for (scf::ParallelOp root : roots)
    if (failed(partition(root, grain))) return failure();
  return success();
}

LogicalResult isolateTasks(func::FuncOp function) {
  SmallVector<scf::ParallelOp> operations;
  function.walk([&](scf::ParallelOp operation) { operations.push_back(operation); });
  for (auto parallel : operations) {
    if (parallel.getNumLoops() != 1 || parallel.getNumResults() ||
        !matchPattern(parallel.getLowerBound()[0], m_Zero()) ||
        !matchPattern(parallel.getStep()[0], m_One()))
      return parallel.emitError("task isolation requires the partitioned one-dimensional workset");
    llvm::SetVector<Value> used;
    getUsedValuesDefinedAbove(parallel.getRegion(), used);
    SmallVector<Value> captures;
    for (Value value : used)
      if (!value.getDefiningOp<arith::ConstantOp>()) captures.push_back(value);
    OpBuilder b(parallel);
    auto tasks = b.create<TasksOp>(parallel.getLoc(), parallel.getUpperBound()[0], captures);
    Block *body = new Block;
    tasks.getBody().push_back(body);
    body->addArgument(b.getIndexType(), parallel.getLoc());
    IRMapping mapping;
    mapping.map(parallel.getInductionVars()[0], body->getArgument(0));
    for (Value capture : captures)
      mapping.map(capture, body->addArgument(capture.getType(), capture.getLoc()));
    b.setInsertionPointToStart(body);
    for (Value value : used)
      if (auto constant = value.getDefiningOp<arith::ConstantOp>()) b.clone(*constant, mapping);
    for (Operation &operation : parallel.getBody()->without_terminator()) b.clone(operation, mapping);
    b.create<TaskYieldOp>(parallel.getLoc());
    parallel.erase();
  }
  return success();
}

LogicalResult materializeTaskLoops(func::FuncOp function) {
  SmallVector<TasksOp> operations;
  function.walk([&](TasksOp tasks) { operations.push_back(tasks); });
  for (auto tasks : operations) {
    OpBuilder b(tasks);
    Location loc = tasks.getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{tasks.getCount()}, ValueRange{one});
    b.setInsertionPointToStart(parallel.getBody());
    Block &body = tasks.getBody().front();
    IRMapping mapping;
    mapping.map(body.getArgument(0), parallel.getInductionVars()[0]);
    for (auto [argument, capture] : llvm::zip(body.getArguments().drop_front(), tasks.getCaptures()))
      mapping.map(argument, capture);
    for (Operation &operation : body.without_terminator()) b.clone(operation, mapping);
    tasks.erase();
  }
  return success();
}

}

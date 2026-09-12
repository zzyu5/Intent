#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

void exposeStructuredWorksets(func::FuncOp function) {
  AliasAnalysis aliases(function);
  PhysicalProgramAnalysis analysis(function);
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
  auto disjoint = [&](Value left, Value right) {
    if (aliases.alias(left, right).isNo()) return true;
    auto lhs = analysis.externalView(left), rhs = analysis.externalView(right);
    return lhs && rhs && analysis.storageRoot(left) != analysis.storageRoot(right) &&
        interface.getDisjointOutputs() && (lhs.getAccess() != 0 || rhs.getAccess() != 0);
  };
  SmallVector<linalg::GenericOp> computations(function.front().getOps<linalg::GenericOp>());
  for (auto operation : computations) {
    if (!operation.getNumLoops() || operation.getNumResults() || operation.getOutputs().empty() ||
        llvm::any_of(operation.getIteratorTypesArray(), [](utils::IteratorType type) {
          return type != utils::IteratorType::parallel;
        })) continue;
    auto maps = operation.getIndexingMapsArray();
    if (llvm::any_of(maps, [](AffineMap map) {
          return map.getNumSymbols() || llvm::any_of(map.getResults(), [](AffineExpr expression) {
            return !isa<AffineDimExpr, AffineConstantExpr>(expression);
          });
        }) || llvm::any_of(ArrayRef<AffineMap>(maps).drop_front(operation.getNumDpsInputs()),
                           [](AffineMap map) { return !map.isIdentity(); })) continue;
    bool independent = true;
    for (auto [number, destination] : llvm::enumerate(operation.getOutputs())) {
      for (Value other : operation.getOutputs().drop_front(number + 1))
        independent &= disjoint(destination, other);
      for (auto [inputNumber, input] : llvm::enumerate(operation.getInputs())) {
        if (!isa<MemRefType>(input.getType())) continue;
        independent &= disjoint(input, destination) ||
            (input == destination && maps[inputNumber] == maps[operation.getNumDpsInputs() + number]);
      }
    }
    for (Operation &body : operation.getRegion().front().without_terminator()) {
      if (isMemoryEffectFree(&body)) continue;
      auto load = dyn_cast<memref::LoadOp>(body);
      if (!load) { independent = false; break; }
      for (Value output : operation.getOutputs())
        independent &= disjoint(load.getMemref(), output);
    }
    if (!independent) continue;
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    Value extent = b.create<memref::DimOp>(loc, operation.getOutputs()[0], 0);
    auto workset = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{extent}, ValueRange{one});
    b.setInsertionPointToStart(workset.getBody());
    Value row = workset.getInductionVars()[0];
    SmallVector<Value> inputs, outputs;
    for (auto [number, operand] : llvm::enumerate(operation->getOperands())) {
      Value selected = operand;
      if (auto type = dyn_cast<MemRefType>(operand.getType())) {
        SmallVector<OpFoldResult> offsets, sizes, strides(type.getRank(), b.getIndexAttr(1));
        for (auto [axis, expression] : llvm::enumerate(maps[number].getResults())) {
          auto dimension = dyn_cast<AffineDimExpr>(expression);
          bool leading = dimension && dimension.getPosition() == 0;
          offsets.push_back(leading ? OpFoldResult(row) : OpFoldResult(b.getIndexAttr(0)));
          sizes.push_back(leading ? OpFoldResult(b.getIndexAttr(1))
              : type.isDynamicDim(axis) ? OpFoldResult(b.create<memref::DimOp>(loc, operand, axis).getResult())
                                        : OpFoldResult(b.getIndexAttr(type.getDimSize(axis))));
        }
        selected = b.create<memref::SubViewOp>(loc, operand, offsets, sizes, strides);
      }
      (number < static_cast<size_t>(operation.getNumDpsInputs()) ? inputs : outputs).push_back(selected);
    }
    auto tile = b.create<linalg::GenericOp>(loc, inputs, outputs, maps, operation.getIteratorTypesArray(),
        [&](OpBuilder &nested, Location location, ValueRange arguments) {
          IRMapping mapping;
          Block &body = operation.getRegion().front();
          mapping.map(body.getArguments(), arguments);
          for (Operation &instruction : body.without_terminator()) {
            if (auto coordinate = dyn_cast<linalg::IndexOp>(instruction); coordinate && coordinate.getDim() == 0)
              mapping.map(coordinate.getResult(), row);
            else nested.clone(instruction, mapping);
          }
          SmallVector<Value> results;
          for (Value value : body.getTerminator()->getOperands()) results.push_back(mapping.lookupOrDefault(value));
          nested.create<linalg::YieldOp>(location, results);
        });
    tile->setAttrs(operation->getAttrs());
    operation.erase();
  }
}

LogicalResult partition(scf::ParallelOp root, int64_t grain) {
  auto function = root->getParentOfType<func::FuncOp>();
  DominanceInfo dominance(function);
  PhysicalProgramAnalysis analysis(function);
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
  auto rectangular = [&](scf::ParallelOp parallel) {
    for (auto [begin, end, step] : llvm::zip(parallel.getLowerBound(), parallel.getUpperBound(), parallel.getStep()))
      if (!matchPattern(begin, m_Zero()) || !matchPattern(step, m_One()) || !dominance.dominates(end, root))
        return false;
    return true;
  };
  auto duplicable = [&](Operation *operation) {
    if (operation->getNumRegions()) return false;
    if (isMemoryEffectFree(operation)) return true;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      auto view = analysis.externalView(load.getMemref());
      return view && view.getAccess() == 0 && interface.getDisjointOutputs();
    }
    return false;
  };
  SmallVector<scf::ParallelOp> nest;
  scf::ParallelOp current = root;
  while (current) {
    if (current.getNumResults()) return current.emitError("CPU task reduction is not implemented");
    nest.push_back(current);
    scf::ParallelOp child;
    bool flatten = true;
    for (Operation &operation : current.getBody()->without_terminator()) {
      if (auto parallel = dyn_cast<scf::ParallelOp>(&operation)) {
        if (child) { flatten = false; break; }
        child = parallel;
      }
    }
    if (child) {
      for (Operation &operation : current.getBody()->without_terminator())
        if (&operation != child && !duplicable(&operation)) flatten = false;
    }
    current = child && flatten && rectangular(child) ? child : scf::ParallelOp();
  }
  SmallVector<Value> extents, coordinates;
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
          if (auto child = dyn_cast<scf::ParallelOp>(&operation); child && llvm::is_contained(nest, child)) cloneBody(child);
          else b.clone(operation, mapping);
        }
      };
      cloneBody(root);
    });
  }
  SmallVector<scf::ParallelOp> nested;
  tasks.walk<WalkOrder::PostOrder>([&](scf::ParallelOp parallel) {
    if (parallel != tasks) nested.push_back(parallel);
  });
  for (scf::ParallelOp parallel : nested) {
    if (parallel.getNumResults()) return parallel.emitError("CPU task reduction is not implemented");
    OpBuilder nestedBuilder(parallel);
    IRMapping mapping;
    std::function<void(unsigned)> traverse = [&](unsigned axis) {
      if (axis == parallel.getNumLoops()) {
        for (Operation &operation : parallel.getBody()->without_terminator())
          nestedBuilder.clone(operation, mapping);
        return;
      }
      auto loop = nestedBuilder.create<scf::ForOp>(parallel.getLoc(), parallel.getLowerBound()[axis],
          parallel.getUpperBound()[axis], parallel.getStep()[axis]);
      OpBuilder::InsertionGuard guard(nestedBuilder);
      nestedBuilder.setInsertionPointToStart(loop.getBody());
      mapping.map(parallel.getInductionVars()[axis], loop.getInductionVar());
      traverse(axis + 1);
    };
    traverse(0);
    parallel.erase();
  }
  root.erase();
  return success();
}

}

LogicalResult partitionTasks(func::FuncOp function, int64_t grain) {
  exposeStructuredWorksets(function);
  SmallVector<scf::ParallelOp> roots;
  function.walk([&](scf::ParallelOp operation) {
    if (!operation->getParentOfType<scf::ParallelOp>()) roots.push_back(operation);
  });
  for (scf::ParallelOp root : roots)
    if (failed(partition(root, grain))) return failure();
  return success();
}

LogicalResult isolateTasks(func::FuncOp function) {
  SmallVector<QuantizeOp> preparations;
  function.walk([&](QuantizeOp operation) {
    if (!operation->getParentOfType<scf::ParallelOp>()) preparations.push_back(operation);
  });
  for (auto preparation : preparations) {
    OpBuilder b(preparation);
    auto loc = preparation.getLoc();
    auto task = b.create<scf::ParallelOp>(loc, ValueRange{index(b, loc, 0)},
        ValueRange{index(b, loc, 1)}, ValueRange{index(b, loc, 1)});
    preparation->moveBefore(task.getBody()->getTerminator());
  }
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

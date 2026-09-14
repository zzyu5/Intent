#include "Intent/Dialect/CPU/Transforms/Implementation.h"
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

void partitionScalarSums(func::FuncOp function, int64_t grain) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  if (capabilities.getWorkers() <= 1) return;
  int64_t limit = std::min(grain, capabilities.getPrivateBytes() / 4 / capabilities.getWorkers()) * capabilities.getWorkers();
  if (!limit) return;
  SmallVector<ReduceOp> reductions(function.front().getOps<ReduceOp>());
  for (ReduceOp operation : reductions) {
    if (!operation.getResult().getType().isF32() || !operation.getOrder().getAdjacentReassociation() ||
        !matchPattern(operation.getInitial(), m_PosZeroFloat())) continue;
    Block &body = operation.getCombine().front();
    auto combine = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!combine || !body.getArgument(0).hasOneUse()) continue;
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    Value count = b.create<arith::CeilDivSIOp>(loc, operation.getExtent(), index(b, loc, 4096));
    count = b.create<arith::MinSIOp>(loc, index(b, loc, limit),
        b.create<arith::MaxSIOp>(loc, one, count));
    Value width = b.create<arith::DivSIOp>(loc, operation.getExtent(), count);
    Value remainder = b.create<arith::RemSIOp>(loc, operation.getExtent(), count);
    Value partials = b.create<memref::AllocOp>(loc, MemRefType::get({limit}, b.getF32Type()));
    auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{count}, ValueRange{one});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(tasks.getBody());
      Value group = tasks.getInductionVars()[0];
      Value begin = add(b, loc, multiply(b, loc, group, width), b.create<arith::MinSIOp>(loc, group, remainder));
      Value extra = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, group, remainder);
      Value size = add(b, loc, width, b.create<arith::SelectOp>(loc, extra, one, zero));
      SmallVector<Value> inputs;
      for (auto [input, attr] : llvm::zip(operation.getInputs(), operation.getIndexingMaps())) {
        auto type = dyn_cast<MemRefType>(input.getType());
        if (!type) { inputs.push_back(input); continue; }
        SmallVector<OpFoldResult> offsets, sizes, strides(type.getRank(), b.getIndexAttr(1));
        for (AffineExpr expression : cast<AffineMapAttr>(attr).getValue().getResults()) {
          bool varying = isa<AffineDimExpr>(expression);
          offsets.push_back(varying ? OpFoldResult(begin) : OpFoldResult(b.getIndexAttr(0)));
          sizes.push_back(varying ? OpFoldResult(size) : OpFoldResult(b.getIndexAttr(1)));
        }
        inputs.push_back(b.create<memref::SubViewOp>(loc, input, offsets, sizes, strides));
      }
      auto partial = b.create<ReduceOp>(loc, operation.getResult().getType(), size,
          operation.getInitial(), inputs, operation.getIndexingMaps(), operation.getOrder());
      partial->setAttr("intent_cpu.implementation", operation->getAttr("intent_cpu.implementation"));
      IRMapping mapping;
      operation.getCombine().cloneInto(&partial.getCombine(), mapping);
      b.create<memref::StoreOp>(loc, partial.getResult(), partials, ValueRange{group});
    }
    auto result = b.create<ReduceOp>(loc, b.getF32Type(), count, operation.getInitial(), ValueRange{partials},
        b.getArrayAttr({AffineMapAttr::get(b.getMultiDimIdentityMap(1))}), operation.getOrder());
    result->setAttr("intent_cpu.implementation", operation->getAttr("intent_cpu.implementation"));
    {
      OpBuilder::InsertionGuard guard(b);
      Block *merge = &result.getCombine().emplaceBlock();
      merge->addArguments(TypeRange{b.getF32Type(), b.getF32Type()}, {loc, loc});
      b.setInsertionPointToStart(merge);
      b.create<YieldOp>(loc, b.create<arith::AddFOp>(loc, merge->getArgument(0), merge->getArgument(1)).getResult());
    }
    b.create<memref::DeallocOp>(loc, partials);
    operation.replaceAllUsesWith(result.getResult());
    operation.erase();
  }
}

LogicalResult exposeStructuredWorksets(func::FuncOp function, const ImplementationRegistry &implementations) {
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
        operation.getIteratorTypesArray()[0] != utils::IteratorType::parallel) continue;
    auto maps = operation.getIndexingMapsArray();
    if (llvm::any_of(maps, [](AffineMap map) {
          return map.getNumSymbols() || llvm::any_of(map.getResults(), [](AffineExpr expression) {
            return !isa<AffineDimExpr, AffineConstantExpr>(expression);
          });
        }) || llvm::any_of(ArrayRef<AffineMap>(maps).drop_front(operation.getNumDpsInputs()),
                           [](AffineMap map) {
                             if (!map.getNumResults() || map.getResult(0) != getAffineDimExpr(0, map.getContext())) return true;
                             return llvm::any_of(map.getResults().drop_front(), [](AffineExpr expression) {
                               auto dimension = dyn_cast<AffineDimExpr>(expression);
                               return dimension && dimension.getPosition() == 0;
                             });
                           })) continue;
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
    int64_t rows = 1;
    if (auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation")) {
      auto implementation = implementations.lookup(operation);
      if (failed(implementation)) return failure();
      if ((*implementation)->worksetRows) rows = (*implementation)->worksetRows(operation, binding);
      if (rows <= 0) return operation.emitError("implementation requires a positive leading parallel workset");
    }
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    Value extent = b.create<memref::DimOp>(loc, operation.getOutputs()[0], 0);
    Value window = index(b, loc, rows);
    Value count = rows == 1 ? extent : b.create<arith::CeilDivSIOp>(loc, extent, window);
    auto workset = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{count}, ValueRange{one});
    b.setInsertionPointToStart(workset.getBody());
    Value row = multiply(b, loc, workset.getInductionVars()[0], window);
    OpFoldResult rowCount = b.getIndexAttr(1);
    if (rows != 1)
      rowCount = b.create<arith::MinSIOp>(loc, window, b.create<arith::SubIOp>(loc, extent, row)).getResult();
    SmallVector<Value> inputs, outputs;
    for (auto [number, operand] : llvm::enumerate(operation->getOperands())) {
      Value selected = operand;
      if (auto type = dyn_cast<MemRefType>(operand.getType())) {
        SmallVector<OpFoldResult> offsets, sizes, strides(type.getRank(), b.getIndexAttr(1));
        for (auto [axis, expression] : llvm::enumerate(maps[number].getResults())) {
          auto dimension = dyn_cast<AffineDimExpr>(expression);
          bool leading = dimension && dimension.getPosition() == 0;
          offsets.push_back(leading ? OpFoldResult(row) : OpFoldResult(b.getIndexAttr(0)));
          sizes.push_back(leading ? rowCount
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
            if (auto coordinate = dyn_cast<linalg::IndexOp>(instruction); coordinate && coordinate.getDim() == 0) {
              Value coordinateRow = row;
              if (rows != 1)
                coordinateRow = add(nested, location, row, nested.create<linalg::IndexOp>(location, 0));
              mapping.map(coordinate.getResult(), coordinateRow);
            } else nested.clone(instruction, mapping);
          }
          SmallVector<Value> results;
          for (Value value : body.getTerminator()->getOperands()) results.push_back(mapping.lookupOrDefault(value));
          nested.create<linalg::YieldOp>(location, results);
        });
    tile->setAttrs(operation->getAttrs());
    operation.erase();
  }
  return success();
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
    auto clonePoint = [&](ValueRange point) {
      IRMapping mapping;
      mapping.map(ValueRange(coordinates), point);
      std::function<void(scf::ParallelOp)> cloneBody = [&](scf::ParallelOp parallel) {
        for (Operation &operation : parallel.getBody()->without_terminator()) {
          if (auto child = dyn_cast<scf::ParallelOp>(&operation); child && llvm::is_contained(nest, child)) cloneBody(child);
          else b.clone(operation, mapping);
        }
      };
      cloneBody(root);
    };
    if (extents.size() == 1) {
      loop(b, loc, begin, end, 1, [&](Value linear) { clonePoint(ValueRange{linear}); });
    } else {
      Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, begin, end);
      auto active = b.create<scf::IfOp>(loc, nonempty, false);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(active.thenBlock());
      auto decode = [&](Value ordinal) {
        SmallVector<Value> point(extents.size());
        for (int64_t axis = extents.size() - 1; axis >= 0; --axis) {
          point[axis] = axis == 0 ? ordinal
              : Value(b.create<arith::RemSIOp>(loc, ordinal, extents[axis]));
          if (axis) ordinal = b.create<arith::DivSIOp>(loc, ordinal, extents[axis]);
        }
        return point;
      };
      auto first = decode(begin);
      auto last = decode(b.create<arith::SubIOp>(loc, end, one));
      SmallVector<Value> lastEnds, point(extents.size());
      for (Value coordinate : last) lastEnds.push_back(add(b, loc, coordinate, one));
      Value boundary = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));
      std::function<void(unsigned, Value, Value)> traverse = [&](unsigned axis, Value firstPrefix, Value lastPrefix) {
        Value lower = b.createOrFold<arith::SelectOp>(loc, firstPrefix, first[axis], zero);
        Value upper = b.createOrFold<arith::SelectOp>(loc, lastPrefix, lastEnds[axis], extents[axis]);
        loop(b, loc, lower, upper, 1, [&](Value coordinate) {
          point[axis] = coordinate;
          if (axis + 1 == extents.size()) { clonePoint(point); return; }
          Value atFirst = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coordinate, first[axis]);
          Value atLast = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coordinate, last[axis]);
          traverse(axis + 1, b.createOrFold<arith::AndIOp>(loc, firstPrefix, atFirst),
              b.createOrFold<arith::AndIOp>(loc, lastPrefix, atLast));
        });
      };
      traverse(0, boundary, boundary);
    }
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

LogicalResult partitionTasks(func::FuncOp function, int64_t grain, const ImplementationRegistry &implementations) {
  partitionScalarSums(function, grain);
  if (failed(exposeStructuredWorksets(function, implementations))) return failure();
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

LogicalResult materializeTaskDispatches(func::FuncOp function) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  SmallVector<TasksOp> operations;
  function.walk([&](TasksOp tasks) { operations.push_back(tasks); });
  for (auto tasks : operations) {
    OpBuilder b(tasks);
    Location loc = tasks.getLoc();
    Value workers = b.create<arith::MaxSIOp>(loc, index(b, loc, 1),
        b.create<arith::MinSIOp>(loc, tasks.getCount(), index(b, loc, capabilities.getWorkers())));
    Value nonempty = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, tasks.getCount(), index(b, loc, 0));
    auto active = b.create<scf::IfOp>(loc, nonempty, false);
    b.setInsertionPointToStart(active.thenBlock());
    Value width = b.create<arith::DivSIOp>(loc, tasks.getCount(), workers);
    Value remainder = b.create<arith::RemSIOp>(loc, tasks.getCount(), workers);
    auto dispatch = b.create<TaskDispatchOp>(loc, workers, workers);
    Block *body = &dispatch.getBody().emplaceBlock();
    body->addArgument(b.getIndexType(), loc);
    IRMapping mapping;
    Block &original = tasks.getBody().front();
    for (auto [argument, capture] : llvm::zip(original.getArguments().drop_front(), tasks.getCaptures()))
      mapping.map(argument, capture);
    b.setInsertionPointToStart(body);
    Value ordinal = body->getArgument(0);
    Value extraBefore = b.create<arith::MinSIOp>(loc, ordinal, remainder);
    Value begin = add(b, loc, multiply(b, loc, ordinal, width), extraBefore);
    Value extra = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ordinal, remainder);
    Value count = add(b, loc, width, b.create<arith::SelectOp>(loc, extra, index(b, loc, 1), index(b, loc, 0)));
    loop(b, loc, begin, add(b, loc, begin, count), 1, [&](Value task) {
      mapping.map(original.getArgument(0), task);
      for (Operation &operation : original.without_terminator()) b.clone(operation, mapping);
    });
    b.create<TaskYieldOp>(loc);
    tasks.erase();
  }
  return success();
}

}

#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace intent::cpu {
namespace {

SmallVector<Value> coordinates(OpBuilder &b, Location loc, AffineMap map,
                               ValueRange domain) {
  SmallVector<Value> result;
  for (AffineExpr expression : map.getResults()) {
    if (auto axis = dyn_cast<AffineDimExpr>(expression)) result.push_back(domain[axis.getPosition()]);
    else result.push_back(index(b, loc, cast<AffineConstantExpr>(expression).getValue()));
  }
  return result;
}

bool supportedMap(AffineMap map) {
  return !map.getNumSymbols() && llvm::all_of(map.getResults(), [](AffineExpr expression) {
    return isa<AffineDimExpr, AffineConstantExpr>(expression);
  });
}

LogicalResult materialize(linalg::GenericOp operation) {
  if (operation.getOutputs().empty() || operation.getNumResults())
    return operation.emitError("CPU structured materialization requires destination buffers");
  auto maps = operation.getIndexingMapsArray();
  if (!llvm::all_of(maps, supportedMap))
    return operation.emitError("CPU pointwise coordinate map has no implemented scalar projection");
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  SmallVector<Value> sizes(operation.getNumLoops()), position;
  for (auto [memory, map] : llvm::zip(operation->getOperands(), maps)) {
    if (!isa<MemRefType>(memory.getType())) continue;
    for (auto [axis, expression] : llvm::enumerate(map.getResults()))
      if (auto dimension = dyn_cast<AffineDimExpr>(expression))
        if (!sizes[dimension.getPosition()])
          sizes[dimension.getPosition()] = b.create<memref::DimOp>(loc, memory, axis);
  }
  if (llvm::any_of(sizes, [](Value size) { return !size; }))
    return operation.emitError("CPU structured traversal has an unbound loop extent");
  std::function<void(unsigned)> visit = [&](unsigned axis) {
    if (axis != sizes.size()) {
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value i) {
        position.push_back(i); visit(axis + 1); position.pop_back();
      });
      return;
    }
    IRMapping mapping;
    Block &body = operation.getRegion().front();
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      Value value = input;
      if (isa<MemRefType>(input.getType()))
        value = b.create<memref::LoadOp>(loc, input, coordinates(b, loc, maps[number], position));
      mapping.map(body.getArgument(number), value);
    }
    for (auto [number, output] : llvm::enumerate(operation.getOutputs())) {
      unsigned argument = operation.getInputs().size() + number;
      if (!body.getArgument(argument).use_empty())
        mapping.map(body.getArgument(argument), b.create<memref::LoadOp>(loc, output, coordinates(b, loc, maps[argument], position)));
    }
    for (Operation &nested : body.without_terminator()) {
      if (auto index = dyn_cast<linalg::IndexOp>(nested)) mapping.map(index.getResult(), position[index.getDim()]);
      else b.clone(nested, mapping);
    }
    for (auto [number, output] : llvm::enumerate(operation.getOutputs()))
      b.create<memref::StoreOp>(loc, mapping.lookupOrDefault(body.getTerminator()->getOperand(number)), output,
          coordinates(b, loc, maps[operation.getInputs().size() + number], position));
  };
  visit(0);
  operation.erase();
  return success();
}

LogicalResult materialize(ReduceOp operation) {
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  auto reduction = b.create<scf::ForOp>(loc, index(b, loc, 0), operation.getExtent(),
      index(b, loc, 1), ValueRange{operation.getInitial()});
  reduction->setAttr("intent_cpu.reduction_order", operation.getOrder());
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(reduction.getBody());
    IRMapping mapping;
    Block &body = operation.getCombine().front();
    mapping.map(body.getArgument(0), reduction.getRegionIterArgs()[0]);
    for (auto [number, input] : llvm::enumerate(operation.getInputs())) {
      Value value = input;
      if (isa<MemRefType>(input.getType()))
        value = b.create<memref::LoadOp>(loc, input,
            coordinates(b, loc, cast<AffineMapAttr>(operation.getIndexingMaps()[number]).getValue(),
                        ValueRange{reduction.getInductionVar()}));
      mapping.map(body.getArgument(number + 1), value);
    }
    for (Operation &nested : body.without_terminator()) b.clone(nested, mapping);
    b.create<scf::YieldOp>(loc, mapping.lookupOrDefault(cast<YieldOp>(body.getTerminator()).getValue()));
  }
  operation.getResult().replaceAllUsesWith(reduction.getResult(0));
  operation.erase();
  return success();
}

LogicalResult materialize(ScanOp operation) {
  OpBuilder b(operation);
  Location loc = operation.getLoc();
  auto type = cast<MemRefType>(operation.getSources()[0].getType());
  int64_t scanAxis = operation.getAxis();
  SmallVector<Value> sizes, position(type.getRank());
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    sizes.push_back(b.create<memref::DimOp>(loc, operation.getSources()[0], axis));
  int64_t width = 1;
  auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
  if (binding)
    if (auto parameter = dyn_cast_or_null<IntegerAttr>(binding.getParameters().get("scan_width")))
      width = parameter.getInt();
  if (width > 1 && !isElementwiseContiguousScan(operation))
    return operation.emitError("selected vector scan requires a contiguous last axis and an elementwise combine");
  auto combine = [&](ValueRange left, ValueRange right, bool vectorized) {
    Block &body = operation.getCombine().front();
    IRMapping mapping;
    unsigned component = 0;
    for (Value value : left) mapping.map(body.getArgument(component++), value);
    for (Value value : right) mapping.map(body.getArgument(component++), value);
    for (Value capture : operation.getCaptures()) {
      Value value = vectorized ? Value(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, capture.getType()), capture)) : capture;
      mapping.map(body.getArgument(component++), value);
    }
    for (Operation &instruction : body.without_terminator()) {
      if (vectorized && isa<arith::ConstantOp>(instruction)) {
        Value scalar = b.clone(instruction)->getResult(0);
        mapping.map(instruction.getResult(0), b.create<vector::BroadcastOp>(loc, VectorType::get({width}, scalar.getType()), scalar));
      } else {
        Operation *cloned = b.clone(instruction, mapping);
        if (vectorized) cloned->getResult(0).setType(VectorType::get({width}, instruction.getResult(0).getType()));
      }
    }
    SmallVector<Value> result;
    for (Value value : body.getTerminator()->getOperands()) result.push_back(mapping.lookupOrDefault(value));
    return result;
  };
  auto broadcast = [&](ValueRange scalars) {
    SmallVector<Value> result;
    for (Value scalar : scalars) result.push_back(b.create<vector::BroadcastOp>(loc, VectorType::get({width}, scalar.getType()), scalar));
    return result;
  };
  auto shift = [&](ValueRange vectors, ValueRange leading, int64_t offset) {
    SmallVector<int64_t> lanes;
    for (int64_t lane = 0; lane < width; ++lane) lanes.push_back(lane < offset ? lane : width + lane - offset);
    SmallVector<Value> result;
    for (auto [value, initial] : llvm::zip(vectors, leading))
      result.push_back(b.create<vector::ShuffleOp>(loc, initial, value, lanes));
    return result;
  };
  std::function<void(int64_t)> traverse = [&](int64_t axis) {
    if (axis != type.getRank()) {
      if (axis == scanAxis) { traverse(axis + 1); return; }
      loop(b, loc, index(b, loc, 0), sizes[axis], 1, [&](Value coordinate) {
        position[axis] = coordinate;
        traverse(axis + 1);
      });
      return;
    }
    Value extent = sizes[operation.getAxis()];
    Value begin = index(b, loc, 0);
    SmallVector<Value> initials(operation.getInitials());
    if (width > 1) {
      Value step = index(b, loc, width);
      Value completeEnd = b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, step));
      auto blocks = b.create<scf::ForOp>(loc, begin, completeEnd, step, initials);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(blocks.getBody());
        Value coordinate = blocks.getInductionVar();
        if (operation.getReverse()) coordinate = b.create<arith::SubIOp>(loc,
            b.create<arith::SubIOp>(loc, extent, step), coordinate);
        position[scanAxis] = coordinate;
        SmallVector<int64_t> reverseLanes;
        for (int64_t lane = width - 1; lane >= 0; --lane) reverseLanes.push_back(lane);
        SmallVector<Value> prefix;
        for (Value source : operation.getSources()) {
          auto vectorType = VectorType::get({width}, cast<MemRefType>(source.getType()).getElementType());
          Value values = b.create<vector::LoadOp>(loc, vectorType, source, position);
          if (operation.getReverse()) values = b.create<vector::ShuffleOp>(loc, values, values, reverseLanes);
          prefix.push_back(values);
        }
        auto identity = broadcast(operation.getInitials());
        // Each stage combines adjacent ranges in the selected traversal order.
        for (int64_t offset = 1; offset < width; offset *= 2)
          prefix = combine(shift(prefix, identity, offset), prefix, true);
        auto incoming = broadcast(blocks.getRegionIterArgs());
        prefix = combine(incoming, prefix, true);
        auto output = operation.getInclusive() ? prefix : shift(prefix, incoming, 1);
        for (auto [value, destination] : llvm::zip(output, operation.getOutputs())) {
          if (operation.getReverse()) value = b.create<vector::ShuffleOp>(loc, value, value, reverseLanes);
          b.create<vector::StoreOp>(loc, value, destination, position);
        }
        SmallVector<Value> next;
        for (Value value : prefix)
          next.push_back(b.create<vector::ExtractElementOp>(loc, value, index(b, loc, width - 1)));
        b.create<scf::YieldOp>(loc, next);
      }
      begin = completeEnd;
      initials.assign(blocks.getResults().begin(), blocks.getResults().end());
    }
    auto scan = b.create<scf::ForOp>(loc, begin, extent, index(b, loc, 1), initials);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(scan.getBody());
    Value coordinate = scan.getInductionVar();
    if (operation.getReverse()) coordinate = b.create<arith::SubIOp>(loc,
        b.create<arith::SubIOp>(loc, extent, index(b, loc, 1)), coordinate);
    position[operation.getAxis()] = coordinate;
    SmallVector<Value> inputs;
    for (Value source : operation.getSources())
      inputs.push_back(b.create<memref::LoadOp>(loc, source, position));
    auto next = combine(scan.getRegionIterArgs(), inputs, false);
    ValueRange output = operation.getInclusive() ? ValueRange(next) : ValueRange(scan.getRegionIterArgs());
    for (auto [value, destination] : llvm::zip(output, operation.getOutputs()))
      b.create<memref::StoreOp>(loc, value, destination, position);
    b.create<scf::YieldOp>(loc, next);
  };
  traverse(0);
  operation.erase();
  return success();
}

}

LogicalResult realizeHistograms(func::FuncOp function) {
  auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>("intent_cpu.capabilities");
  auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
  SmallVector<HistogramOp> histograms;
  function.walk([&](HistogramOp operation) { histograms.push_back(operation); });
  for (HistogramOp operation : histograms) {
    OpBuilder b(operation);
    Location loc = operation.getLoc();
    auto inputType = cast<MemRefType>(operation.getValues().getType());
    auto outputType = cast<MemRefType>(operation.getOutput().getType());
    int64_t groups = operation->getParentOfType<scf::ParallelOp>() ? 1
        : capabilities.getWorkers() * configuration.getTaskGrain();
    Value zero = index(b, loc, 0), one = index(b, loc, 1), groupCount = index(b, loc, groups);
    Value size = one;
    SmallVector<Value> extents;
    for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
      extents.push_back(b.create<memref::DimOp>(loc, operation.getValues(), axis));
      size = multiply(b, loc, size, extents.back());
    }
    Value bins = b.create<memref::DimOp>(loc, operation.getOutput(), 0);
    SmallVector<Value> dynamicSizes;
    if (outputType.isDynamicDim(0)) dynamicSizes.push_back(bins);
    Value partials = b.create<memref::AllocOp>(loc,
        MemRefType::get({groups, outputType.getDimSize(0)}, outputType.getElementType()), dynamicSizes);
    Value countZero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outputType.getElementType(), 0));
    Value countOne = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(outputType.getElementType(), 1));
    b.create<linalg::FillOp>(loc, ValueRange{countZero}, ValueRange{partials});
    Value chunk = b.create<arith::CeilDivSIOp>(loc, size, groupCount);
    auto tasks = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{groupCount}, ValueRange{one});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(tasks.getBody());
      Value group = tasks.getInductionVars()[0];
      Value begin = multiply(b, loc, group, chunk);
      Value end = b.create<arith::MinSIOp>(loc, add(b, loc, begin, chunk), size);
      loop(b, loc, begin, end, 1, [&](Value ordinal) {
        SmallVector<Value> coordinates(inputType.getRank());
        Value remaining = ordinal;
        for (int64_t axis = inputType.getRank() - 1; axis >= 0; --axis) {
          coordinates[axis] = axis ? Value(b.create<arith::RemSIOp>(loc, remaining, extents[axis])) : remaining;
          if (axis) remaining = b.create<arith::DivSIOp>(loc, remaining, extents[axis]);
        }
        Value active = b.create<memref::LoadOp>(loc, operation.getValid(), coordinates);
        auto conditional = b.create<scf::IfOp>(loc, active, false);
        OpBuilder::InsertionGuard predicateGuard(b);
        b.setInsertionPointToStart(conditional.thenBlock());
        Value value = b.create<memref::LoadOp>(loc, operation.getValues(), coordinates);
        if (!value.getType().isIndex()) {
          if (operation.getUnsignedValues()) value = b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), value);
          else value = b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
        }
        SmallVector<Value> bin{group, value};
        Value previous = b.create<memref::LoadOp>(loc, partials, bin);
        Value next = b.create<arith::AddIOp>(loc, previous, countOne);
        b.create<memref::StoreOp>(loc, next, partials, bin);
      });
    }
    b.create<linalg::FillOp>(loc, ValueRange{countZero}, ValueRange{operation.getOutput()});
    auto bin = b.getAffineDimExpr(0), group = b.getAffineDimExpr(1);
    b.create<linalg::GenericOp>(loc, ValueRange{partials}, ValueRange{operation.getOutput()},
        SmallVector<AffineMap>{AffineMap::get(2, 0, {group, bin}, b.getContext()),
                              AffineMap::get(2, 0, {bin}, b.getContext())},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::reduction},
        [](OpBuilder &nested, Location location, ValueRange arguments) {
          nested.create<linalg::YieldOp>(location, nested.create<arith::AddIOp>(location, arguments[0], arguments[1]).getResult());
        });
    b.create<memref::DeallocOp>(loc, partials);
    operation.erase();
  }
  return success();
}

LogicalResult materializeStructuredComputations(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<linalg::GenericOp, linalg::FillOp, ReduceOp, ScanOp>(operation)) operations.push_back(operation);
  });
  for (Operation *operation : operations) {
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      if (fill.getOutputs().size() != 1 || fill.getNumResults())
        return fill.emitError("CPU fill requires one physical buffer");
      OpBuilder b(fill);
      int64_t rank = cast<MemRefType>(fill.getOutputs()[0].getType()).getRank();
      auto generic = b.create<linalg::GenericOp>(fill.getLoc(), fill.getInputs(), fill.getOutputs(),
          SmallVector<AffineMap>{AffineMap::get(rank, 0, {}, b.getContext()), b.getMultiDimIdentityMap(rank)},
          SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
          [](OpBuilder &nested, Location loc, ValueRange arguments) {
            nested.create<linalg::YieldOp>(loc, arguments[0]);
          });
      fill.erase();
      if (failed(materialize(generic))) return failure();
    } else if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      if (failed(materialize(generic))) return failure();
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      if (failed(materialize(scan))) return failure();
    } else if (failed(materialize(cast<ReduceOp>(operation)))) return failure();
  }
  return success();
}

}

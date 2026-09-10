#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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

}

LogicalResult materializeStructuredComputations(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<linalg::GenericOp, linalg::FillOp, ReduceOp>(operation)) operations.push_back(operation);
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
    } else if (failed(materialize(cast<ReduceOp>(operation)))) return failure();
  }
  return success();
}

}

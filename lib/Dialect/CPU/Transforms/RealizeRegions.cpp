#include "Intent/Dialect/CPU/IR/RegionProgram.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
namespace intent::cpu {
namespace {

void copy(OpBuilder &b, Location loc, Value source, Value target) {
  if (isa<MemRefType>(source.getType())) b.create<memref::CopyOp>(loc, source, target);
  else b.create<memref::StoreOp>(loc, source, target, ValueRange{});
}

SmallVector<Value> scratch(OpBuilder &b, Location loc, ValueRange prototypes) {
  SmallVector<Value> result;
  for (Value value : prototypes) {
    auto memory = dyn_cast<MemRefType>(value.getType());
    auto type = memory ? MemRefType::get(memory.getShape(), memory.getElementType()) : MemRefType::get({}, value.getType());
    SmallVector<Value> sizes;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      if (type.isDynamicDim(axis)) sizes.push_back(b.create<memref::DimOp>(loc, value, axis));
    result.push_back(b.create<memref::AllocOp>(loc, type, sizes));
  }
  return result;
}

LogicalResult instantiate(OpBuilder &b, Region &helper, ValueRange arguments) {
  Block &body = helper.front();
  if (arguments.size() != body.getNumArguments()) return helper.getParentOp()->emitError("region helper binding is incomplete");
  IRMapping mapping;
  for (auto [source, value] : llvm::zip(body.getArguments(), arguments)) {
    if (source.getType() != value.getType()) {
      if (!isa<MemRefType>(source.getType()) || !isa<MemRefType>(value.getType()))
        return helper.getParentOp()->emitError("region helper scalar binding type mismatch");
      value = b.create<memref::CastOp>(helper.getLoc(), source.getType(), value);
    }
    mapping.map(source, value);
  }
  for (Operation &operation : body.without_terminator()) b.clone(operation, mapping);
  return success();
}

SmallVector<Value> slices(OpBuilder &b, Location loc, ValueRange sources,
                          unsigned axis, Value begin, OpFoldResult extent) {
  SmallVector<Value> result;
  for (Value source : sources) {
    auto type = cast<MemRefType>(source.getType());
    SmallVector<OpFoldResult> offsets, sizes, steps;
    for (unsigned index = 0; index < type.getRank(); ++index) {
      offsets.push_back(index == axis ? OpFoldResult(begin) : b.getIndexAttr(0));
      sizes.push_back(index == axis ? extent :
          type.isDynamicDim(index) ? OpFoldResult(b.create<memref::DimOp>(loc, source, index).getResult()) : b.getIndexAttr(type.getDimSize(index)));
      steps.push_back(b.getIndexAttr(1));
    }
    result.push_back(b.create<memref::SubViewOp>(loc, source, offsets, sizes, steps));
  }
  return result;
}

LogicalResult realize(Operation *operation, int64_t segmentSize) {
  RegionProgram program(operation);
  OpBuilder b(operation);
  Location loc = operation->getLoc();
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value count = b.create<memref::DimOp>(loc, program.sources().front(), program.count("axis"));
  auto step = b.create<arith::ConstantIndexOp>(loc, segmentSize);
  operation->setAttr("segment_size", b.getI64IntegerAttr(segmentSize));
  ValueRange state = program.isScan() ? program.outputs().take_back(program.count("state_count")) : program.outputs();
  ValueRange initial = program.isScan() ? program.initialState() : program.identities();
  for (auto [input, output] : llvm::zip(initial, state)) copy(b, loc, input, output);
  auto summary = scratch(b, loc, program.identities());
  auto next = scratch(b, loc, state);
  Value fullEnd = b.create<arith::SubIOp>(loc, count, b.create<arith::RemSIOp>(loc, count, step));
  auto visit = [&](Value lower, Value upper, int64_t width) -> LogicalResult {
    auto loop = b.create<scf::ForOp>(loc, lower, upper, b.create<arith::ConstantIndexOp>(loc, width));
    loop->setAttr("intent_cpu.region_axis", b.getI64IntegerAttr(program.count("axis")));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value begin = loop.getInductionVar();
    OpFoldResult extent = b.getIndexAttr(width);
    auto inputs = slices(b, loc, program.sources(), program.count("axis"), begin, extent);
    SmallVector<Value> arguments(inputs);
    llvm::append_range(arguments, program.captures()); llvm::append_range(arguments, summary);
    if (failed(instantiate(b, program.summarize(), arguments))) return failure();
    if (program.isScan()) {
      arguments = inputs; llvm::append_range(arguments, state); llvm::append_range(arguments, program.captures());
      auto outputAxes = operation->getAttrOfType<DenseI64ArrayAttr>("output_axes").asArrayRef();
      for (auto [output, axis] : llvm::zip(program.outputs().take_front(program.count("output_count")), outputAxes))
        llvm::append_range(arguments, slices(b, loc, ValueRange{output}, axis, begin, extent));
      if (failed(instantiate(b, program.emit(), arguments))) return failure();
      arguments = summary; llvm::append_range(arguments, state); llvm::append_range(arguments, next);
      if (failed(instantiate(b, program.apply(), arguments))) return failure();
    } else {
      arguments.assign(state.begin(), state.end()); llvm::append_range(arguments, summary); llvm::append_range(arguments, next);
      if (failed(instantiate(b, program.combine(), arguments))) return failure();
    }
    for (auto [source, target] : llvm::zip(next, state)) copy(b, loc, source, target);
    return success();
  };
  if (failed(visit(zero, fullEnd, segmentSize)) || failed(visit(fullEnd, count, 1))) return failure();
  for (Value value : summary) b.create<memref::DeallocOp>(loc, value);
  for (Value value : next) b.create<memref::DeallocOp>(loc, value);
  operation->erase();
  return success();
}

}

LogicalResult realizeRegions(func::FuncOp function, int64_t segmentSize) {
  SmallVector<Operation *> regions;
  function.walk([&](Operation *operation) {
    if (isa<RegionFoldOp, RegionScanOp>(operation)) regions.push_back(operation);
  });
  for (Operation *operation : regions)
    if (failed(realize(operation, segmentSize))) return failure();
  return success();
}

}

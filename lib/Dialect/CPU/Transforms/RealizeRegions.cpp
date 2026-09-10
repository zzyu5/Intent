#include "Intent/Dialect/CPU/IR/RegionProgram.h"
#include "Intent/Dialect/CPU/Analysis/RegionPartition.h"
#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

LogicalResult instantiate(OpBuilder &b, Region &helper, ValueRange arguments,
                          const RegionPartition *partition, Value begin, OpFoldResult width) {
  Block &body = helper.front();
  if (arguments.size() != body.getNumArguments()) return helper.getParentOp()->emitError("region helper binding is incomplete");
  IRMapping mapping;
  for (auto [source, value] : llvm::zip(body.getArguments(), arguments)) {
    if (!partition && source.getType() != value.getType()) {
      if (!isa<MemRefType>(source.getType()) || !isa<MemRefType>(value.getType()))
        return helper.getParentOp()->emitError("region helper scalar binding type mismatch");
      value = b.create<memref::CastOp>(helper.getLoc(), source.getType(), value);
    }
    mapping.map(source, value);
  }
  auto tiledType = [&](Value value) {
    auto type = cast<MemRefType>(value.getType());
    SmallVector<int64_t> shape(type.getShape());
    auto found = partition->axes.find(value);
    if (found != partition->axes.end()) shape[found->second] = getConstantIntValue(width).value_or(ShapedType::kDynamic);
    return MemRefType::get(shape, type.getElementType(), type.getLayout(), type.getMemorySpace());
  };
  for (Operation &operation : body.without_terminator()) {
    if (partition) {
      if (auto allocation = dyn_cast<memref::AllocOp>(operation)) {
        auto type = tiledType(allocation.getResult());
        SmallVector<Value> sizes;
        auto selected = partition->axes.find(allocation.getResult());
        for (int64_t axis = 0; axis < allocation.getType().getRank(); ++axis) {
          if (!type.isDynamicDim(axis)) continue;
          if (selected != partition->axes.end() && selected->second == axis)
            sizes.push_back(getValueOrCreateConstantIndexOp(b, allocation.getLoc(), width));
          else sizes.push_back(mapping.lookupOrDefault(
              allocation.getDynamicSizes()[allocation.getType().getDynamicDimIndex(axis)]));
        }
        auto replacement = b.create<memref::AllocOp>(allocation.getLoc(), type, sizes,
            allocation.getAlignmentAttr());
        mapping.map(allocation.getResult(), replacement.getResult());
        continue;
      }
      if (auto dimension = dyn_cast<memref::DimOp>(operation)) {
        auto found = partition->axes.find(dimension.getSource());
        if (found != partition->axes.end() && dimension.getConstantIndex() == found->second) {
          mapping.map(dimension.getResult(), getValueOrCreateConstantIndexOp(b, dimension.getLoc(), width));
          continue;
        }
      }
      if (auto cast = dyn_cast<memref::CastOp>(operation)) {
        mapping.map(cast.getResult(), b.create<memref::CastOp>(cast.getLoc(), tiledType(cast.getResult()),
            mapping.lookupOrDefault(cast.getSource())));
        continue;
      }
      if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
        auto mapped = [&](ArrayRef<OpFoldResult> values) {
          SmallVector<OpFoldResult> result;
          for (OpFoldResult value : values)
            result.push_back(isa<Value>(value) ? OpFoldResult(mapping.lookupOrDefault(cast<Value>(value))) : value);
          return result;
        };
        auto offsets = mapped(view.getMixedOffsets()), sizes = mapped(view.getMixedSizes()), strides = mapped(view.getMixedStrides());
        auto found = partition->axes.find(view.getSource());
        if (found != partition->axes.end()) sizes[found->second] = width;
        Value source = mapping.lookupOrDefault(view.getSource());
        auto type = memref::SubViewOp::inferRankReducedResultType(tiledType(view.getResult()).getShape(),
            cast<MemRefType>(source.getType()), offsets, sizes, strides);
        mapping.map(view.getResult(), b.create<memref::SubViewOp>(view.getLoc(), cast<MemRefType>(type), source, offsets, sizes, strides));
        continue;
      }
    }
    Operation *cloned = b.clone(operation, mapping);
    if (partition) {
      auto found = partition->loops.find(&operation);
      if (found == partition->loops.end()) continue;
      SmallVector<linalg::IndexOp> indices;
      cloned->walk([&](linalg::IndexOp index) { if (index.getDim() == found->second) indices.push_back(index); });
      for (auto index : indices) {
        OpBuilder builder(index);
        builder.setInsertionPointAfter(index);
        auto absolute = builder.create<arith::AddIOp>(index.getLoc(), index, begin);
        index.getResult().replaceAllUsesExcept(absolute, absolute);
      }
    }
  }
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

LogicalResult realize(Operation *operation, const Configuration &configuration,
                      const ImplementationRegistry &implementations) {
  RegionProgram program(operation);
  OpBuilder b(operation);
  Location loc = operation->getLoc();
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value count = b.create<memref::DimOp>(loc, program.sources().front(), program.count("axis"));
  int64_t segmentSize = configuration.regionSize;
  auto step = b.create<arith::ConstantIndexOp>(loc, segmentSize);
  operation->setAttr("segment_size", b.getI64IntegerAttr(segmentSize));
  ValueRange state = program.isScan() ? program.outputs().take_back(program.count("state_count")) : program.outputs();
  ValueRange initial = program.isScan() ? program.initialState() : program.identities();
  for (auto [input, output] : llvm::zip(initial, state)) copy(b, loc, input, output);
  Value fullEnd = b.create<arith::SubIOp>(loc, count, b.create<arith::RemSIOp>(loc, count, step));
  auto partition = analyzeRegionPartition(program, configuration.tileM, configuration.tileN);
  bool staticPanel = false;
  if (partition)
    for (auto [consumer, axis] : partition->loops) {
      auto generic = dyn_cast<linalg::GenericOp>(consumer);
      if (!generic || !isMatrixContraction(generic)) continue;
      auto implementation = implementations.lookup(consumer);
      if (failed(implementation)) return failure();
      staticPanel |= (*implementation)->contraction.staticParallelExtent;
    }
  auto realizePanel = [&](Value beginPanel, OpFoldResult panelWidth) -> LogicalResult {
  auto project = [&](ValueRange values) {
    SmallVector<Value> projected;
    for (Value value : values) {
      if (partition) {
        auto found = partition->axes.find(value);
        if (found != partition->axes.end()) {
          projected.push_back(slices(b, loc, ValueRange{value}, found->second, beginPanel, panelWidth).front());
          continue;
        }
      }
      projected.push_back(value);
    }
    return projected;
  };
  auto sources = project(program.sources()), identities = project(program.identities());
  auto captures = project(program.captures()), outputs = project(program.outputs());
  auto panelState = ArrayRef(outputs).take_back(state.size());
  auto summary = scratch(b, loc, identities);
  auto next = scratch(b, loc, panelState);
  auto expand = [&](Region &helper, ValueRange arguments) {
    return instantiate(b, helper, arguments, partition ? &*partition : nullptr, beginPanel, panelWidth);
  };
  auto visit = [&](Value lower, Value upper, int64_t width) -> LogicalResult {
    auto loop = b.create<scf::ForOp>(loc, lower, upper, b.create<arith::ConstantIndexOp>(loc, width));
    loop->setAttr("intent_cpu.region_axis", b.getI64IntegerAttr(program.count("axis")));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value begin = loop.getInductionVar();
    OpFoldResult extent = b.getIndexAttr(width);
    auto inputs = slices(b, loc, sources, program.count("axis"), begin, extent);
    SmallVector<Value> arguments(inputs);
    llvm::append_range(arguments, captures); llvm::append_range(arguments, summary);
    if (failed(expand(program.summarize(), arguments))) return failure();
    if (program.isScan()) {
      arguments = inputs; llvm::append_range(arguments, panelState); llvm::append_range(arguments, captures);
      auto outputAxes = operation->getAttrOfType<DenseI64ArrayAttr>("output_axes").asArrayRef();
      for (auto [output, axis] : llvm::zip(ArrayRef(outputs).take_front(program.count("output_count")), outputAxes))
        llvm::append_range(arguments, slices(b, loc, ValueRange{output}, axis, begin, extent));
      if (failed(expand(program.emit(), arguments))) return failure();
      arguments = summary; llvm::append_range(arguments, panelState); llvm::append_range(arguments, next);
      if (failed(expand(program.apply(), arguments))) return failure();
    } else {
      arguments.assign(panelState.begin(), panelState.end()); llvm::append_range(arguments, summary); llvm::append_range(arguments, next);
      if (failed(expand(program.combine(), arguments))) return failure();
    }
    for (auto [source, target] : llvm::zip(next, panelState)) copy(b, loc, source, target);
    return success();
  };
  if (failed(visit(zero, fullEnd, segmentSize)) || failed(visit(fullEnd, count, 1))) return failure();
  for (Value value : summary) b.create<memref::DeallocOp>(loc, value);
  for (Value value : next) b.create<memref::DeallocOp>(loc, value);
  return success();
  };
  if (partition) {
    Value output = program.outputs().front();
    Value extent = b.create<memref::DimOp>(loc, output, partition->axes.lookup(output));
    Value width = b.create<arith::ConstantIndexOp>(loc, partition->width);
    Value end = staticPanel ? Value(b.create<arith::SubIOp>(loc, extent, b.create<arith::RemSIOp>(loc, extent, width))) : extent;
    auto panelLoop = [&](Value lower, Value upper, int64_t size) {
      auto loop = b.create<scf::ForOp>(loc, lower, upper, b.create<arith::ConstantIndexOp>(loc, size));
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      OpFoldResult active = b.getIndexAttr(size);
      if (!staticPanel)
        active = b.create<arith::MinSIOp>(loc, width,
            b.create<arith::SubIOp>(loc, extent, loop.getInductionVar())).getResult();
      return realizePanel(loop.getInductionVar(), active);
    };
    if (failed(panelLoop(zero, end, partition->width)) ||
        (staticPanel && failed(panelLoop(end, extent, 1)))) return failure();
  } else if (failed(realizePanel({}, {}))) return failure();
  operation->erase();
  return success();
}

}

LogicalResult realizeRegions(func::FuncOp function, const Configuration &configuration,
                              const ImplementationRegistry &implementations) {
  SmallVector<Operation *> regions;
  function.walk([&](Operation *operation) {
    if (isa<RegionFoldOp, RegionScanOp>(operation)) regions.push_back(operation);
  });
  for (Operation *operation : regions)
    if (failed(realize(operation, configuration, implementations))) return failure();
  return success();
}

}

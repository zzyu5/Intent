#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

linalg::GenericOp pointwiseProducer(Value buffer, Operation *consumer,
                                    PhysicalProgramAnalysis &analysis) {
  if (!buffer.getDefiningOp<memref::AllocOp>()) return {};
  linalg::GenericOp producer;
  for (Operation *user : buffer.getUsers()) {
    if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
      if (!llvm::is_contained(generic.getOutputs(), buffer)) continue;
      if (producer) return {};
      producer = generic;
    } else if (!isa<memref::DimOp, memref::DeallocOp, ReduceOp>(user)) return {};
  }
  if (!producer || producer == consumer || producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer) || producer.getOutputs().size() != 1 ||
      producer.getNumResults() || producer.getNumReductionLoops() ||
      !producer.getIndexingMapsArray().back().isIdentity()) return {};
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return {};
  for (Operation &operation : body.without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return {};
  for (Value input : producer.getInputs())
    if (isa<MemRefType>(input.getType()) && !analysis.mayReadAt(input, producer, consumer))
      return {};
  return producer;
}

void eraseUnusedProducer(linalg::GenericOp producer) {
  Value buffer = producer.getOutputs()[0];
  auto allocation = buffer.getDefiningOp<memref::AllocOp>();
  if (!allocation) return;
  for (Operation *user : buffer.getUsers())
    if (user != producer && !isa<memref::DimOp, memref::DeallocOp>(user)) return;
  SmallVector<Operation *> uses(buffer.getUsers());
  for (Operation *user : uses) {
    if (auto dimension = dyn_cast<memref::DimOp>(user)) {
      auto axis = dimension.getConstantIndex();
      if (!axis) return;
      OpBuilder b(dimension);
      Value extent;
      if (allocation.getType().isDynamicDim(*axis)) {
        unsigned position = 0;
        for (int64_t i = 0; i < *axis; ++i)
          position += allocation.getType().isDynamicDim(i);
        extent = allocation.getDynamicSizes()[position];
      } else {
        extent = b.create<arith::ConstantIndexOp>(dimension.getLoc(),
            allocation.getType().getDimSize(*axis));
      }
      dimension.getResult().replaceAllUsesWith(extent);
      dimension.erase();
    }
  }
  producer.erase();
  SmallVector<Operation *> remaining(buffer.getUsers());
  for (Operation *user : remaining) cast<memref::DeallocOp>(user).erase();
  allocation.erase();
}

bool fuseOne(Operation *consumer, unsigned inputNumber,
             linalg::GenericOp producer) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  auto reduce = dyn_cast<ReduceOp>(consumer);
  ValueRange oldInputs = generic ? ValueRange(generic.getInputs()) : ValueRange(reduce.getInputs());
  SmallVector<AffineMap> oldMaps = generic ? generic.getIndexingMapsArray()
      : llvm::to_vector(llvm::map_range(reduce.getIndexingMaps(), [](Attribute a) {
          return cast<AffineMapAttr>(a).getValue();
        }));
  auto producerMaps = producer.getIndexingMapsArray();
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  for (auto [i, input] : llvm::enumerate(oldInputs)) {
    if (i == inputNumber) {
      for (auto [j, operand] : llvm::enumerate(producer.getInputs())) {
        inputs.push_back(operand);
        maps.push_back(producerMaps[j].compose(oldMaps[i]));
      }
    } else {
      inputs.push_back(input);
      maps.push_back(oldMaps[i]);
    }
  }
  auto populate = [&](OpBuilder &b, Block &target, Block &oldBody, unsigned prefix) {
    IRMapping mapping;
    unsigned current = prefix;
    if (prefix) mapping.map(oldBody.getArgument(0), target.getArgument(0));
    for (unsigned i = 0; i < oldInputs.size(); ++i) {
      if (i != inputNumber) {
        mapping.map(oldBody.getArgument(prefix + i), target.getArgument(current++));
        continue;
      }
      IRMapping producerMapping;
      Block &body = producer.getRegion().front();
      for (unsigned j = 0; j < producer.getInputs().size(); ++j)
        producerMapping.map(body.getArgument(j), target.getArgument(current++));
      for (Operation &operation : body.without_terminator()) b.clone(operation, producerMapping);
      mapping.map(oldBody.getArgument(prefix + i),
          producerMapping.lookupOrDefault(body.getTerminator()->getOperand(0)));
    }
    if (generic)
      mapping.map(oldBody.getArguments().back(), target.getArguments().back());
    for (Operation &operation : oldBody.without_terminator()) b.clone(operation, mapping);
    return mapping.lookupOrDefault(oldBody.getTerminator()->getOperand(0));
  };
  OpBuilder builder(consumer);
  if (generic) {
    maps.push_back(oldMaps.back());
    builder.create<linalg::GenericOp>(generic.getLoc(), inputs,
        generic.getOutputs(), maps, generic.getIteratorTypesArray(),
        [&](OpBuilder &b, Location loc, ValueRange) {
          Value value = populate(b, *b.getInsertionBlock(), generic.getRegion().front(), 0);
          b.create<linalg::YieldOp>(loc, value);
        });
    generic.erase();
  } else {
    SmallVector<Attribute> attributes;
    for (AffineMap map : maps) attributes.push_back(AffineMapAttr::get(map));
    auto replacement = builder.create<ReduceOp>(reduce.getLoc(), reduce.getResult().getType(),
        reduce.getExtent(), reduce.getInitial(), inputs, builder.getArrayAttr(attributes),
        reduce.getOrder());
    Block &body = replacement.getCombine().emplaceBlock();
    body.addArgument(reduce.getInitial().getType(), reduce.getLoc());
    for (Value input : inputs) {
      auto memory = dyn_cast<MemRefType>(input.getType());
      body.addArgument(memory ? memory.getElementType() : input.getType(), reduce.getLoc());
    }
    builder.setInsertionPointToStart(&body);
    Value value = populate(builder, body, reduce.getCombine().front(), 1);
    builder.create<YieldOp>(reduce.getLoc(), value);
    reduce.getResult().replaceAllUsesWith(replacement.getResult());
    reduce.erase();
  }
  eraseUnusedProducer(producer);
  return true;
}

}

LogicalResult fuseStructuredComputations(func::FuncOp function) {
  forwardCPUOutputs(function);
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> consumers;
    function.walk([&](Operation *operation) {
      if (isa<ReduceOp>(operation)) consumers.push_back(operation);
      else if (auto generic = dyn_cast<linalg::GenericOp>(operation);
               generic && !generic.getNumReductionLoops() && generic.getOutputs().size() == 1)
        consumers.push_back(operation);
    });
    for (Operation *consumer : llvm::reverse(consumers)) {
      PhysicalProgramAnalysis analysis(function);
      auto generic = dyn_cast<linalg::GenericOp>(consumer);
      ValueRange inputs = generic ? ValueRange(generic.getInputs())
                                 : ValueRange(cast<ReduceOp>(consumer).getInputs());
      for (auto [number, input] : llvm::enumerate(inputs)) {
        auto producer = pointwiseProducer(input, consumer, analysis);
        if (!producer) continue;
        changed = fuseOne(consumer, number, producer);
        break;
      }
      if (changed) break;
    }
  } while (changed);
  return success();
}

}

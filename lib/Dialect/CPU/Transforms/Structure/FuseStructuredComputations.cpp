#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/Computations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "IntegerSources.h"
#include "ContiguousAccesses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {

namespace {

struct PointwiseProducer {
  linalg::GenericOp operation;
  ProducerReplay payload;
};

std::optional<PointwiseProducer> pointwiseProducer(Value buffer, Operation *consumer,
                                                  StorageAnalysis &storage) {
  if (!buffer.getDefiningOp<memref::AllocOp>()) return std::nullopt;
  linalg::GenericOp producer;
  for (Operation *user : buffer.getUsers()) {
    if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
      if (!llvm::is_contained(generic.getOutputs(), buffer)) {
        if (user != consumer) return std::nullopt;
        continue;
      }
      if (producer) return std::nullopt;
      producer = generic;
    } else if (isa<ReduceOp>(user)) {
      if (user != consumer) return std::nullopt;
    } else if (!isa<memref::DimOp, memref::DeallocOp>(user)) return std::nullopt;
  }
  if (!producer || producer == consumer || producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer) || producer.getOutputs().size() != 1 ||
      producer.getNumResults() || producer.getNumReductionLoops() ||
      !producer.getIndexingMapsArray().back().isIdentity()) return std::nullopt;
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return std::nullopt;
  // CPU Reduce's scalar combine has a deliberately narrower contract than a
  // linalg payload. Do not import regions or explicit reads into that ABI.
  if (isa<ReduceOp>(consumer))
    for (Operation &operation : body.without_terminator())
      if (operation.getNumRegions() || !isMemoryEffectFree(&operation))
        return std::nullopt;
  auto payload = analyzeLinalgProducer(producer, storage);
  if (failed(payload)) return std::nullopt;
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  ValueRange outputs = generic ? ValueRange(generic.getOutputs()) : ValueRange{};
  if (!canReplayProducerAt(*payload, producer, consumer, storage, outputs))
    return std::nullopt;
  return PointwiseProducer{producer, std::move(*payload)};
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
             const PointwiseProducer &source) {
  auto producer = source.operation;
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  auto reduce = dyn_cast<ReduceOp>(consumer);
  ValueRange oldInputs = generic ? ValueRange(generic.getInputs()) : ValueRange(reduce.getInputs());
  SmallVector<AffineMap> oldMaps = generic ? generic.getIndexingMapsArray()
      : llvm::to_vector(llvm::map_range(reduce.getIndexingMaps(), [](Attribute a) {
          return cast<AffineMapAttr>(a).getValue();
        }));
  auto producerMaps = producer.getIndexingMapsArray();
  SmallVector<linalg::IndexOp> indices;
  producer.getRegion().walk([&](linalg::IndexOp index) { indices.push_back(index); });
  if (reduce && !indices.empty()) return false;
  for (auto index : indices)
    if (!isa<AffineDimExpr, AffineConstantExpr>(oldMaps[inputNumber].getResult(index.getDim())))
      return false;
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
  if (generic) {
    // Removing an input must not remove the only shape binding for a loop.
    llvm::SmallBitVector bound(generic.getNumLoops());
    for (AffineMap map : maps)
      for (AffineExpr expression : map.getResults())
        if (auto dimension = dyn_cast<AffineDimExpr>(expression)) bound.set(dimension.getPosition());
    for (AffineExpr expression : oldMaps.back().getResults())
      if (auto dimension = dyn_cast<AffineDimExpr>(expression)) bound.set(dimension.getPosition());
    if (!bound.all()) return false;
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
      for (auto index : indices) {
        AffineExpr coordinate = oldMaps[inputNumber].getResult(index.getDim());
        Value value;
        if (auto dimension = dyn_cast<AffineDimExpr>(coordinate))
          value = b.create<linalg::IndexOp>(index.getLoc(), dimension.getPosition());
        else value = b.create<arith::ConstantIndexOp>(index.getLoc(), cast<AffineConstantExpr>(coordinate).getValue());
        producerMapping.map(index.getResult(), value);
      }
      LogicalResult cloned = cloneProducerPayload(source.payload, b, producerMapping);
      assert(succeeded(cloned) && "structured replay must bind the complete payload frontier");
      (void)cloned;
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
    auto replacement = builder.create<linalg::GenericOp>(generic.getLoc(), inputs,
        generic.getOutputs(), maps, generic.getIteratorTypesArray(),
        [](OpBuilder &, Location, ValueRange) {});
    Block &body = replacement.getRegion().front();
    builder.setInsertionPointToStart(&body);
    Value value = populate(builder, body, generic.getRegion().front(), 0);
    builder.create<linalg::YieldOp>(generic.getLoc(), value);
    replacement->setDiscardableAttrs(llvm::to_vector(generic->getDiscardableAttrs()));
    generic.erase();
  } else {
    SmallVector<Attribute> attributes;
    for (AffineMap map : maps) attributes.push_back(AffineMapAttr::get(map));
    auto replacement = builder.create<ReduceOp>(reduce.getLoc(), reduce.getResult().getType(),
        reduce.getExtent(), reduce.getInitial(), inputs, builder.getArrayAttr(attributes),
        reduce.getOrder());
    replacement->setDiscardableAttrs(llvm::to_vector(reduce->getDiscardableAttrs()));
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

bool reuseOutput(linalg::GenericOp consumer, Value buffer,
                 StorageAnalysis &storage) {
  auto allocation = buffer.getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != consumer->getBlock() ||
      consumer.getNumReductionLoops() || consumer.getOutputs().size() != 1 ||
      !consumer.getRegion().front().getArguments().back().use_empty()) return false;
  auto maps = consumer.getIndexingMapsArray();
  if (!maps.back().isIdentity()) return false;
  for (auto [input, map] : llvm::zip(consumer.getInputs(), maps))
    if (input == buffer && !map.isIdentity()) return false;
  Value output = consumer.getOutputs()[0];
  auto external = storage.externalView(output);
  auto view = output.getDefiningOp<memref::SubViewOp>();
  if (!external || external.getAccess() != 1 || !view ||
      view.getType().getRank() != allocation.getType().getRank() ||
      view.getType().getElementType() != allocation.getType().getElementType()) return false;
  SmallVector<OpFoldResult> extents;
  auto dropped = view.getDroppedDims();
  for (auto [axis, size] : llvm::enumerate(view.getMixedSizes()))
    if (!dropped.test(axis)) extents.push_back(size);
  unsigned dynamic = 0;
  for (auto [axis, extent] : llvm::enumerate(extents)) {
    if (allocation.getType().isDynamicDim(axis)) {
      Value size = allocation.getDynamicSizes()[dynamic++];
      if (extent == OpFoldResult(size)) continue;
      auto left = getConstantIntValue(extent), right = getConstantIntValue(size);
      if (!left || !right || *left != *right) return false;
    } else if (getConstantIntValue(extent) != allocation.getType().getDimSize(axis)) return false;
  }
  auto aliases = storage.aliases(output);
  if (!aliases.complete || llvm::any_of(aliases.users, [&](Operation *user) {
        return user != consumer && !isa<memref::DimOp>(user) && !isStorageAliasOperation(user);
      })) return false;
  linalg::GenericOp producer;
  llvm::SmallPtrSet<Operation *, 4> readers;
  SmallVector<memref::DeallocOp> deallocations;
  for (Operation *user : buffer.getUsers()) {
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocations.push_back(dealloc);
      continue;
    }
    if (isa<memref::DimOp>(user)) continue;
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (generic && llvm::is_contained(generic.getOutputs(), buffer)) {
      if (producer || generic.getOutputs().size() != 1 || generic.getNumReductionLoops() ||
          !generic.getIndexingMapsArray().back().isIdentity() ||
          !generic.getRegion().front().getArguments().back().use_empty() ||
          llvm::is_contained(generic.getInputs(), buffer)) return false;
      producer = generic;
    } else if (generic || isa<ReduceOp>(user)) readers.insert(user);
    else return false;
    if (user->getBlock() != consumer->getBlock() ||
        (user != consumer && !user->isBeforeInBlock(consumer))) return false;
  }
  if (!producer || readers.size() < 2) return false;
  for (Operation *reader : readers)
    if (!producer->isBeforeInBlock(reader)) return false;
  for (auto operation : {producer, consumer})
    for (Operation &nested : operation.getRegion().front().without_terminator())
      if (nested.getNumRegions() || !isMemoryEffectFree(&nested)) return false;
  DominanceInfo dominance(consumer->getParentOfType<func::FuncOp>());
  if (!dominance.dominates(output, allocation)) {
    if (view->getBlock() != allocation->getBlock() ||
        llvm::any_of(view->getOperands(), [&](Value value) {
          return !dominance.dominates(value, allocation);
        })) return false;
    view->moveBefore(allocation);
  }
  for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
  allocation.getResult().replaceAllUsesWith(output);
  allocation.erase();
  return true;
}

void forwardPointwiseCopies(func::FuncOp function) {
  SmallVector<memref::AllocOp> allocations;
  function.walk([&](memref::AllocOp allocation) { allocations.push_back(allocation); });
  for (auto allocation : allocations) {
    Value buffer = allocation.getResult();
    llvm::DenseMap<Block *, linalg::GenericOp> writers;
    llvm::DenseMap<Block *, memref::CopyOp> readers;
    bool closed = true;
    for (Operation *user : buffer.getUsers()) {
      if (isa<memref::DimOp, memref::DeallocOp>(user)) continue;
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getSource() != buffer || copy.getTarget() == buffer ||
            !readers.try_emplace(copy->getBlock(), copy).second) closed = false;
        continue;
      }
      auto writer = dyn_cast<linalg::GenericOp>(user);
      if (!writer || writer.getNumResults() || writer.getNumReductionLoops() ||
          writer.getOutputs().size() != 1 || writer.getOutputs()[0] != buffer ||
          llvm::is_contained(writer.getInputs(), buffer) ||
          !writer.getIndexingMapsArray().back().isIdentity() ||
          !writer.getRegion().front().getArguments().back().use_empty() ||
          llvm::any_of(writer.getRegion().front().without_terminator(), [](Operation &operation) {
            return operation.getNumRegions() || !isMemoryEffectFree(&operation);
          }) || !writers.try_emplace(writer->getBlock(), writer).second) closed = false;
    }
    if (!closed || readers.empty() || readers.size() != writers.size()) continue;
    // Every read has its own complete same-block definition, including distinct
    // full/tail loop bodies. No other user may observe this scratch afterwards.
    if (llvm::any_of(readers, [&](auto &entry) {
          auto writer = writers.lookup(entry.first);
          return !writer || !writer->isBeforeInBlock(entry.second);
        })) continue;
    for (auto &entry : readers) {
      auto copy = entry.second;
      auto writer = writers.lookup(entry.first);
      Value target = copy.getTarget();
      StorageAnalysis storage(function);
      Value targetRoot = storage.uniqueOrigin(target);
      // Retain a private owner. Forwarding into caller storage also needs the
      // provider's native ABI to preserve the disjointness used for load reuse.
      if (!targetRoot ||
          !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(targetRoot.getDefiningOp())) continue;
      DominanceInfo dominance(function);
      if (!dominance.dominates(target, writer)) continue;
      bool legal = true;
      auto strip = [](Value value) {
        while (auto cast = value.getDefiningOp<memref::CastOp>()) value = cast.getSource();
        return value;
      };
      auto maps = writer.getIndexingMapsArray();
      for (auto [number, input] : llvm::enumerate(writer.getInputs())) {
        if (!isa<MemRefType>(input.getType()) || storage.disjoint(input, target)) continue;
        if (strip(input) != strip(target) || maps[number] != maps.back()) legal = false;
      }
      // Only dst[i] = f(dst[i], ...) is safe in-place. Other observations of the
      // old destination between the definition and its copy keep the snapshot.
      for (Operation *between = writer->getNextNode(); legal && between != copy;
           between = between->getNextNode()) {
        auto effects = storage.effects(between);
        if (!effects.complete || effects.ordered) { legal = false; break; }
        for (const StorageEffect &entry : effects.entries) {
          const auto &effect = entry.effect;
          if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
          if (!effect.getValue() || !storage.disjoint(effect.getValue(), target)) legal = false;
        }
      }
      if (!legal) continue;
      writer.getDpsInitsMutable()[0].set(target);
      copy.erase();
    }
  }
}

}

LogicalResult fuseStructuredComputations(func::FuncOp function) {
  foldIntegerSources(function);
  foldContiguousAccesses(function);
  forwardCPUOutputs(function);
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> consumers;
    StorageAnalysis payloadStorage(function);
    function.walk([&](Operation *operation) {
      if (isa<ReduceOp>(operation)) consumers.push_back(operation);
      else if (auto generic = dyn_cast<linalg::GenericOp>(operation);
               generic && !isMatrixContraction(generic) && !generic.getNumResults() &&
               generic.getOutputs().size() == 1) {
        if (failed(analyzeLinalgProducer(generic, payloadStorage))) return;
        if (generic.getNumReductionLoops()) {
          llvm::SmallBitVector outputAxes(generic.getNumLoops());
          for (AffineExpr expression : generic.getIndexingMapsArray().back().getResults()) {
            auto axis = dyn_cast<AffineDimExpr>(expression);
            if (!axis || outputAxes.test(axis.getPosition())) return;
            outputAxes.set(axis.getPosition());
          }
          for (auto [axis, kind] : llvm::enumerate(generic.getIteratorTypesArray()))
            if (outputAxes.test(axis) != (kind == utils::IteratorType::parallel)) return;
        }
        consumers.push_back(operation);
      }
    });
    for (Operation *consumer : llvm::reverse(consumers)) {
      StorageAnalysis analysis(function);
      auto generic = dyn_cast<linalg::GenericOp>(consumer);
      ValueRange inputs = generic ? ValueRange(generic.getInputs())
                                 : ValueRange(cast<ReduceOp>(consumer).getInputs());
      for (auto [number, input] : llvm::enumerate(inputs)) {
        auto producer = pointwiseProducer(input, consumer, analysis);
        if (!producer) continue;
        changed = fuseOne(consumer, number, *producer);
        break;
      }
      if (changed) break;
    }
  } while (changed);
  SmallVector<linalg::GenericOp> outputs;
  function.walk([&](linalg::GenericOp operation) { outputs.push_back(operation); });
  for (auto operation : llvm::reverse(outputs)) {
    SmallVector<Value> inputs(operation.getInputs());
    for (Value input : inputs) {
      StorageAnalysis analysis(function);
      if (reuseOutput(operation, input, analysis)) break;
    }
  }
  forwardPointwiseCopies(function);
  eraseDeadPrivateBuffers(function);
  return success();
}


}

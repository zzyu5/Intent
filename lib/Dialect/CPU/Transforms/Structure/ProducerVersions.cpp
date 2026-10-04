#include "ProducerVersions.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool completeWrite(Operation *operation, Value buffer) {
  auto writer = dyn_cast<linalg::LinalgOp>(operation);
  if (!writer || operation->getNumResults() || writer.getNumDpsInits() != 1 ||
      writer.getDpsInits().front() != buffer || writer.getNumReductionLoops() ||
      failed(fullOutputProjection(buffer, writer.getIndexingMapsArray().back())))
    return false;
  // A bijective map is not enough: the actual loop bounds must cover every
  // destination element, including dynamic dimensions and omitted unit axes.
  AffineMap bounds = writer.getShapesToLoopsMap();
  if (!bounds || bounds.getNumSymbols()) return false;
  SmallVector<std::pair<Value, int64_t>> dimensions;
  for (OpOperand &operand : operation->getOpOperands())
    for (int64_t axis = 0, rank = writer.getShape(&operand).size(); axis < rank; ++axis)
      dimensions.emplace_back(operand.get(), axis);
  for (auto [axis, expression] : llvm::enumerate(writer.getIndexingMapsArray().back().getResults())) {
    auto loop = dyn_cast<AffineDimExpr>(expression);
    if (!loop) continue;
    auto dimension = dyn_cast<AffineDimExpr>(bounds.getResult(loop.getPosition()));
    if (!dimension || dimension.getPosition() >= dimensions.size()) return false;
    auto [value, sourceAxis] = dimensions[dimension.getPosition()];
    if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(buffer, axis),
                          ValueBoundsConstraintSet::Variable(value, sourceAxis)))
      return false;
  }
  return true;
}

bool disjointEffect(Value memory, Value buffer, StorageAnalysis &storage) {
  return memory && (!isa<BaseMemRefType>(memory.getType()) ||
                    storage.disjoint(memory, buffer));
}

std::optional<unsigned> readingInput(Operation *operation, Value buffer,
                                     StorageAnalysis &storage) {
  auto generic = dyn_cast<linalg::GenericOp>(operation);
  auto reduce = dyn_cast<ReduceOp>(operation);
  if ((!generic && !reduce) || (generic &&
      (generic.getNumResults() || generic.getOutputs().size() != 1)))
    return std::nullopt;
  ValueRange inputs = generic ? ValueRange(generic.getInputs())
                             : ValueRange(reduce.getInputs());
  SmallVector<AffineMap> maps = generic ? generic.getIndexingMapsArray()
      : llvm::to_vector(llvm::map_range(reduce.getIndexingMaps(), [](Attribute attr) {
          return cast<AffineMapAttr>(attr).getValue();
        }));
  std::optional<unsigned> selected;
  for (auto [number, input] : llvm::enumerate(inputs)) {
    if (!isa<BaseMemRefType>(input.getType()) || storage.disjoint(input, buffer)) continue;
    if (input != buffer || (selected && maps[*selected] != maps[number]))
      return std::nullopt;
    selected = number;
  }
  if (!selected) return std::nullopt;
  Block &body = generic ? generic.getRegion().front() : reduce.getCombine().front();
  if (generic && !storage.disjoint(generic.getOutputs()[0], buffer) &&
      !body.getArguments().back().use_empty()) return std::nullopt;
  // Implicit input slots above are the complete observation. Explicit reads or
  // writes through another descriptor cannot be transported by that mapping.
  for (Operation &nested : body.without_terminator()) {
    auto effects = storage.effects(&nested);
    if (!effects.complete || effects.ordered) return std::nullopt;
    for (const StorageEffect &entry : effects.entries)
      if (!disjointEffect(entry.effect.getValue(), buffer, storage))
        return std::nullopt;
  }
  return selected;
}

} // namespace

FailureOr<ProducerVersion> queryProducerVersion(linalg::GenericOp producer,
                                               StorageAnalysis &storage) {
  if (producer.getOutputs().size() != 1) return failure();
  Value buffer = producer.getOutputs()[0];
  if (!completeWrite(producer, buffer) || !isContiguousDescriptor(buffer) ||
      !producer.getRegion().front().getArguments().back().use_empty())
    return failure();
  Value origin = storage.uniqueOrigin(buffer);
  if (!origin || !storage.aliases(origin).complete) return failure();
  auto produced = storage.effects(producer);
  if (!produced.complete || produced.ordered) return failure();
  for (const StorageEffect &entry : produced.entries)
    if (isa<MemoryEffects::Read>(entry.effect.getEffect()) &&
        !disjointEffect(entry.effect.getValue(), buffer, storage))
      return failure();

  ProducerVersion result;
  for (Operation *operation = producer->getNextNode(); operation;
       operation = operation->getNextNode()) {
    auto effects = storage.effects(operation);
    if (!effects.complete || effects.ordered) return failure();
    bool reads = false, writes = false, frees = false;
    for (const StorageEffect &entry : effects.entries) {
      const auto &effect = entry.effect;
      if (disjointEffect(effect.getValue(), buffer, storage)) continue;
      if (isa<MemoryEffects::Read>(effect.getEffect())) reads = true;
      else if (isa<MemoryEffects::Write>(effect.getEffect())) writes = true;
      else if (isa<MemoryEffects::Free>(effect.getEffect())) frees = true;
      else return failure();
    }
    if (reads) {
      auto input = readingInput(operation, buffer, storage);
      if (!input) return failure();
      result.uses.push_back({operation, *input});
    }
    if (writes) {
      if (!completeWrite(operation, buffer)) return failure();
      return result;
    }
    if (frees) {
      auto release = dyn_cast<memref::DeallocOp>(operation);
      if (!release || release.getMemref() != buffer ||
          !buffer.getDefiningOp<memref::AllocOp>()) return failure();
      return result;
    }
  }
  // A surviving output is an observation too. Without a complete overwrite or
  // release, this query cannot authorize deleting the defining write.
  return failure();
}

bool canReplayProducerVersionAt(const ProducerReplay &payload,
    linalg::GenericOp producer, Operation *consumer, AffineMap coordinates,
    StorageAnalysis &storage) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer);
  ValueRange outputs = generic ? ValueRange(generic.getOutputs()) : ValueRange{};
  if (canReplayProducerAt(payload, producer, consumer, storage, outputs)) return true;
  if (!generic || generic.getNumResults() || generic.getOutputs().size() != 1 ||
      generic.getNumReductionLoops() ||
      !generic.getRegion().front().getArguments().back().use_empty()) return false;
  Value output = generic.getOutputs()[0];
  if (!completeWrite(generic, output) || !isContiguousDescriptor(output)) return false;
  for (auto operation : {producer, generic})
    for (Operation &nested : operation.getRegion().front().without_terminator())
      if (nested.getNumRegions() || !isMemoryEffectFree(&nested)) return false;
  AffineMap outputMap = generic.getIndexingMapsArray().back();
  bool sourceFound = false;
  for (auto [input, map] : llvm::zip(producer.getInputs(), producer.getIndexingMapsArray())) {
    if (!isa<BaseMemRefType>(input.getType()) || storage.disjoint(input, output)) continue;
    if (input != output || map.compose(coordinates) != outputMap) return false;
    sourceFound = true;
  }
  if (!sourceFound || !storage.preserves(producer, output) ||
      !storage.unchangedBetween(output, producer, consumer)) return false;
  ProducerReplay remaining = payload;
  llvm::erase_if(remaining.reads, [&](Value memory) { return memory == output; });
  // Only the exact, injective endpoint read above has a different movement
  // proof. Captures, lifetime and every other read retain the common contract.
  return canReplayProducerAt(remaining, producer, consumer, storage, outputs);
}

} // namespace intent::cpu

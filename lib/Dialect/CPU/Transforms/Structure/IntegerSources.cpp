#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "IntegerSources.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool integer(Type type) { return isa<IndexType, IntegerType>(type); }

bool projectedCoordinates(AffineMap map) {
  return !map.getNumSymbols() && llvm::all_of(map.getResults(), [](AffineExpr expr) {
    return isa<AffineDimExpr, AffineConstantExpr>(expr);
  });
}

struct IntegerProducer {
  linalg::GenericOp operation;
  ProducerReplay payload;
};

std::optional<IntegerProducer> integerProducer(memref::LoadOp load,
                                               StorageAnalysis &storage) {
  if (!integer(load.getType())) return std::nullopt;
  auto allocation = load.getMemref().getDefiningOp<memref::AllocOp>();
  if (!allocation) return std::nullopt;
  auto lifetime = storage.lifetime(allocation);
  if (!lifetime || !lifetime->aliases.complete) return std::nullopt;
  linalg::GenericOp producer;
  for (Operation *user : lifetime->aliases.users) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || !llvm::is_contained(generic.getOutputs(), allocation.getResult())) continue;
    if (producer) return std::nullopt;
    producer = generic;
  }
  if (!producer || producer.getNumResults() || producer.getOutputs().size() != 1 ||
      producer.getNumReductionLoops() || producer->getBlock() != allocation->getBlock() ||
      !producer.getIndexingMapsArray().back().isPermutation()) return std::nullopt;
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return std::nullopt;
  if (llvm::any_of(producer.getIndexingMapsArray(), [](AffineMap map) {
        return !projectedCoordinates(map);
      })) return std::nullopt;
  for (Operation *user : lifetime->aliases.users)
    if (user != producer && user != lifetime->end &&
        !storage.preserves(user, allocation)) return std::nullopt;

  Operation *consumer = producer->getBlock()->findAncestorOpInBlock(*load);
  if (!consumer || consumer == producer || !producer->isBeforeInBlock(consumer)) return std::nullopt;
  for (auto [number, input] : llvm::enumerate(producer.getInputs())) {
    if (body.getArgument(number).use_empty()) continue;
    if (auto type = dyn_cast<MemRefType>(input.getType())) {
      if (!integer(type.getElementType())) return std::nullopt;
    } else if (!integer(input.getType())) return std::nullopt;
  }
  auto payload = analyzeLinalgProducer(producer, storage);
  if (failed(payload) ||
      !canReplayProducerAt(*payload, producer, load, storage, ValueRange{allocation}))
    return std::nullopt;
  // Integer-only rematerialization is a policy of this consumer, independent
  // of the common control/effect/read-stability contract.
  for (Operation *operation : payload->nodes)
    if (!llvm::all_of(operation->getResultTypes(), integer) ||
        (!isa<memref::LoadOp, linalg::IndexOp>(operation) &&
         !llvm::all_of(operation->getOperandTypes(), integer))) return std::nullopt;
  return IntegerProducer{producer, std::move(*payload)};
}

void replay(memref::LoadOp load, const IntegerProducer &source) {
  auto producer = source.operation;
  OpBuilder builder(load);
  Location loc = load.getLoc();
  auto maps = producer.getIndexingMapsArray();
  SmallVector<Value> coordinates(producer.getNumLoops());
  for (auto [axis, expr] : llvm::enumerate(maps.back().getResults()))
    coordinates[cast<AffineDimExpr>(expr).getPosition()] = load.getIndices()[axis];
  IRMapping mapping;
  Block &body = producer.getRegion().front();
  for (auto [number, input] : llvm::enumerate(producer.getInputs())) {
    if (body.getArgument(number).use_empty()) continue;
    Value value = input;
    if (isa<MemRefType>(input.getType())) {
      SmallVector<Value> indices;
      for (AffineExpr expr : maps[number].getResults()) {
        if (auto dim = dyn_cast<AffineDimExpr>(expr)) indices.push_back(coordinates[dim.getPosition()]);
        else indices.push_back(builder.create<arith::ConstantIndexOp>(loc, cast<AffineConstantExpr>(expr).getValue()));
      }
      value = builder.create<memref::LoadOp>(loc, input, indices);
    }
    mapping.map(body.getArgument(number), value);
  }
  body.walk([&](linalg::IndexOp index) {
    mapping.map(index.getResult(), coordinates[index.getDim()]);
  });
  LogicalResult cloned = cloneProducerPayload(source.payload, builder, mapping);
  assert(succeeded(cloned) && "integer replay must bind the complete payload frontier");
  (void)cloned;
  load.replaceAllUsesWith(mapping.lookupOrDefault(body.getTerminator()->getOperand(0)));
  load.erase();
}

} // namespace

void foldIntegerSources(func::FuncOp function) {
  bool changed;
  do {
    changed = false;
    SmallVector<memref::LoadOp> loads;
    function.walk([&](memref::LoadOp load) { if (integer(load.getType())) loads.push_back(load); });
    for (auto load : loads) {
      // Every successful replay changes both uses and read placement. Queries
      // deliberately do not survive that rewrite or a dead-buffer deletion.
      StorageAnalysis storage(function);
      if (auto producer = integerProducer(load, storage)) {
        replay(load, *producer);
        changed = true;
      }
    }
    if (changed) eraseDeadPrivateBuffers(function);
  } while (changed);
}

} // namespace intent::cpu

#include "PassDetail.h"
#include "Intent/Dialect/DSA/Transforms/LocalSupplyRelations.h"
#include "llvm/ADT/DenseSet.h"

namespace intent::bangc {
namespace {

// Follow completed snapshots at their original read points. A later value of
// the source, or a partial writer hiding an older value, proves nothing here.
class ZeroInitialization {
public:
  explicit ZeroInitialization(dsa::StorageAnalysis &storage) : storage(storage) {}

  bool before(Value memory, Operation *reader) {
    auto type = dyn_cast<MemRefType>(memory.getType());
    Value origin = storage.uniqueOrigin(memory);
    if (!type || !isa<FloatType>(type.getElementType()) || !origin ||
        !origin.getDefiningOp<memref::AllocaOp>() ||
        !dsa::isCompleteStorageViewOf(memory, origin) ||
        !storage.aliases(origin).complete || !visiting.insert({memory, reader}).second)
      return false;
    Operation *writer = storage.lastWriterBefore(memory, reader);
    bool result = false;
    auto covers = [&](Value output) {
      return storage.uniqueOrigin(output) == origin &&
             dsa::isCompleteStorageViewOf(output, origin);
    };
    if (auto fill = dyn_cast_or_null<dsa::FillOp>(writer)) {
      FloatAttr value;
      result = covers(fill.getOutput()) &&
          matchPattern(fill.getValue(), m_Constant(&value)) &&
          value.getValue().isZero() && !value.getValue().isNegative();
    } else if (auto load = dyn_cast_or_null<dsa::LoadTileOp>(writer)) {
      result = !load.getAsynchronous() && covers(load.getOutput()) &&
          storage.disjoint(load.getSource(), load.getOutput()) &&
          before(load.getSource(), load);
    } else if (auto copy = dyn_cast_or_null<memref::CopyOp>(writer)) {
      result = covers(copy.getTarget()) &&
          storage.disjoint(copy.getSource(), copy.getTarget()) &&
          before(copy.getSource(), copy);
    }
    visiting.erase({memory, reader});
    return result;
  }

private:
  dsa::StorageAnalysis &storage;
  llvm::DenseSet<std::pair<Value, Operation *>> visiting;
};

bool hasFirstIteration(scf::ForOp loop, dsa::LocalSupplyRelations &relations) {
  if (!loop.getInductionVar().getType().isIndex()) return false;
  auto lower = relations.interval(loop.getLowerBound());
  auto upper = relations.interval(loop.getUpperBound());
  auto step = relations.interval(loop.getStep());
  int64_t next;
  if (!lower || !upper || !step || step->first <= 0 ||
      llvm::AddOverflow(lower->second, step->second, next))
    return false;
  if (lower->second < upper->first) return true;
  auto difference = dyn_cast<AffineConstantExpr>(
      relations.difference(loop.getUpperBound(), loop.getLowerBound()));
  return difference && difference.getValue() > 0;
}

bool firstWrite(dsa::MatrixTileOp matrix, scf::ForOp loop,
                dsa::StorageAnalysis &storage, DominanceInfo &dominance,
                ZeroInitialization &zeros) {
  Value accumulator = matrix.getAccumulator();
  if (!loop.getInitArgs().empty() || matrix->getBlock() != loop.getBody() ||
      !dominance.properlyDominates(accumulator, loop)) return false;
  // No earlier operation of this iteration may have changed the seed. Reads
  // are retained; ordinary dead-write elimination decides whether Fill dies.
  for (Operation &operation : *loop.getBody()) {
    if (&operation == matrix.getOperation()) break;
    if (!storage.preservesContents(&operation, accumulator)) return false;
  }
  return zeros.before(accumulator, loop);
}

void initializeFirstIteration(scf::ForOp loop, dsa::MatrixTileOp matrix) {
  OpBuilder builder(loop);
  Value next = builder.createOrFold<arith::AddIOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getStep());
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), loop.getLowerBound());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    Operation *cloned = builder.clone(operation, mapping);
    if (&operation == matrix.getOperation())
      cast<dsa::MatrixTileOp>(cloned).setAccumulate(false);
  }
  loop.setLowerBound(next);
}

} // namespace

bool initializeMatrixStorage(func::FuncOp function) {
  dsa::StorageAnalysis storage(function);
  DominanceInfo dominance(function);
  dsa::LocalSupplyRelations relations(function);
  ZeroInitialization zeros(storage);
  SmallVector<dsa::MatrixTileOp> matrices;
  function.walk([&](dsa::MatrixTileOp matrix) { matrices.push_back(matrix); });
  for (dsa::MatrixTileOp matrix : matrices) {
    if (!matrix.getAccumulate()) continue;
    if (zeros.before(matrix.getAccumulator(), matrix)) {
      matrix.setAccumulate(false);
      return true;
    }
    auto loop = dyn_cast<scf::ForOp>(matrix->getParentOp());
    if (!loop || !hasFirstIteration(loop, relations)) continue;
    auto effects = storage.effects(loop);
    if (!effects.complete || effects.ordered ||
        !firstWrite(matrix, loop, storage, dominance, zeros)) continue;
    initializeFirstIteration(loop, matrix);
    return true;
  }
  return false;
}

} // namespace intent::bangc

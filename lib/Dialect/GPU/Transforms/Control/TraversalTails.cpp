#include "TraversalTails.h"

#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool hasMixedTilePredicate(scf::ForOp loop, PhysicalProgramAnalysis &analysis,
                           IndexRelations &relations) {
  auto width = queryLaunchExpression(loop.getStep());
  if (!width) return false;
  SmallVector<MakeRangeOp> ranges;
  for (MakeRangeOp range : loop.getBody()->getOps<MakeRangeOp>()) {
    auto type = range.getResult().getType();
    if (type.getShape().size() == 1 && type.getElementType().isIndex() &&
        type.getShape()[0] == width && isUnitStepRange(range) &&
        relations.same(range.getStart(), loop.getInductionVar()) &&
        relations.same(range.getExtent(), loop.getStep()) &&
        relations.same(range.getLogicalStart(), loop.getLowerBound()) &&
        relations.same(range.getLogicalStop(), loop.getUpperBound()))
      ranges.push_back(range);
  }
  if (ranges.empty()) return false;

  bool found = false;
  loop.walk([&](CompareOp compare) {
    if (compare.getPredicate() != ComparePredicate::Lt ||
        compare.getResult().use_empty()) return WalkResult::advance();
    Value upper = uniformScalarSource(compare.getRhs());
    if (!upper || !relations.same(upper, loop.getUpperBound()))
      return WalkResult::advance();
    for (MakeRangeOp range : ranges) {
      std::pair<MakeRangeOp, Value> tail{range, loop.getUpperBound()};
      if (analysis.isTailPredicate(compare.getResult(),
              ArrayRef<std::pair<MakeRangeOp, Value>>{tail})) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

bool peel(scf::ForOp loop, func::FuncOp kernel) {
  IndexRelations relations;
  if (!loop.getInductionVar().getType().isIndex() ||
      relations.constant(loop.getLowerBound()) != 0 ||
      !relations.nonnegative(loop.getUpperBound()) ||
      !relations.positive(loop.getStep()) ||
      relations.alignedBound(loop.getUpperBound(), loop.getStep()) ||
      relations.atMost(loop.getUpperBound(), loop.getStep()))
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  if (!hasMixedTilePredicate(loop, analysis, relations)) return false;

  // For U >= 0 and W > 0, floor(U/W)*W is representable, aligned, and in [0,U].
  // The two loops partition precisely the original induction values. In
  // particular U=0 reads nothing, and the suffix has zero or one iteration.
  // Its final increment is the original loop's final increment; no new
  // endpoint addition (which could wrap) is needed for the split itself.
  OpBuilder builder(loop);
  Value completeTiles = builder.create<BinaryOp>(loop.getLoc(), builder.getIndexType(),
      loop.getUpperBound(), loop.getStep(), BinaryOperator::FloorDivide);
  Value fullEnd = builder.create<BinaryOp>(loop.getLoc(), builder.getIndexType(),
      completeTiles, loop.getStep(), BinaryOperator::Multiply);
  IRMapping mapping;
  auto complete = cast<scf::ForOp>(builder.clone(*loop, mapping));
  complete.getUpperBoundMutable().assign(fullEnd);
  loop.getLowerBoundMutable().assign(fullEnd);
  loop.getInitArgsMutable().assign(complete.getResults());
  // Both attached loops already have their complete original bodies/yields.
  // No source read, write, allocation, callback or carry component is moved
  // across an original iteration; only its enclosing interval changes.
  return true;
}

} // namespace

bool peelTraversalTails(func::FuncOp kernel) {
  SmallVector<scf::ForOp> loops;
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (scf::ForOp loop : loops) changed |= peel(loop, kernel);
  return changed;
}

} // namespace intent::gpu

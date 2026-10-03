#include "Intent/Dialect/GPU/Analysis/IndexPredicates.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Value withoutProjection(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp>(operation))
      break;
    value = operation->getOperand(0);
  }
  return value;
}

std::optional<int64_t> integer(Value value) {
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(value));
  return constant && constant.getType().isIndex()
             ? std::optional<int64_t>(constant.getInt()) : std::nullopt;
}

Value completeTileLimit(MakeRangeOp range, IndexRelations &relations) {
  auto type = cast<FragmentType>(range.getResult().getType());
  if (!type.getElementType().isIndex() || type.getShape().size() != 1 ||
      integer(range.getStep()) != 1)
    return {};
  for (Operation *owner = range->getParentOp(); owner; owner = owner->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop || !relations.same(range.getStart(), loop.getInductionVar()) ||
        integer(loop.getLowerBound()) != 0 || !relations.positive(loop.getStep()))
      continue;
    auto width = queryLaunchExpression(loop.getStep());
    if (!width || type.getShape()[0] != width ||
        !relations.same(range.getExtent(), loop.getStep()))
      continue;
    Value limit = relations.alignedBound(loop.getUpperBound(), loop.getStep());
    // For 0 <= iv < limit and an aligned limit, iv + [0, step) stays below
    // limit without signed overflow. No fact about a partial final tile is used.
    if (limit && relations.nonnegative(limit))
      return limit;
  }
  return {};
}

} // namespace

Value queryCompleteTileLimit(MakeRangeOp range) {
  IndexRelations relations;
  return completeTileLimit(range, relations);
}

std::optional<bool> proveRangeComparison(CompareOp comparison) {
  if (comparison.getLhs().getType().isIndex() &&
      comparison.getPredicate() == ComparePredicate::Ge &&
      integer(comparison.getRhs()) == 0 &&
      queryNonNegativeIndexUpperBound(comparison.getLhs()))
    return true;
  Value source = withoutProjection(comparison.getLhs());
  Value bound = withoutProjection(comparison.getRhs());
  auto range = source.getDefiningOp<MakeRangeOp>();
  if (!range || !bound.getType().isIndex())
    return std::nullopt;
  IndexRelations relations;
  Value limit = queryCompleteTileLimit(range);
  if (!limit)
    return std::nullopt;
  if (comparison.getPredicate() == ComparePredicate::Lt &&
      relations.atMost(limit, bound))
    return true;
  if (comparison.getPredicate() == ComparePredicate::Ge) {
    if (relations.atMost(limit, bound))
      return false;
    if (integer(bound) == 0)
      return true;
  }
  return std::nullopt;
}

std::optional<IndexComparisonBound> queryIndexComparisonBound(CompareOp comparison) {
  if (comparison.getPredicate() != ComparePredicate::Lt ||
      !comparison.getLhs().getType().isIndex() ||
      !comparison.getRhs().getType().isIndex() ||
      queryLaunchExpression(comparison.getLhs()))
    return std::nullopt;
  auto upper = queryNonNegativeIndexUpperBound(comparison.getLhs());
  auto limit = queryLaunchExpression(comparison.getRhs());
  if (!upper || !limit || !isShapeBound(upper) || !isShapeBound(limit))
    return std::nullopt;
  // A loop-local bound is not automatically safe to evaluate in the host's
  // specialization of an empty loop. Prove the entire checked expression.
  auto kernel = comparison->getParentOfType<func::FuncOp>();
  if (!queryPhysicalExpressionRange(upper, kernel) ||
      !queryPhysicalExpressionRange(limit, kernel))
    return std::nullopt;
  for (Operation *user : comparison.getResult().getUsers()) {
    auto logicalOr = dyn_cast<BinaryOp>(user);
    if (!logicalOr || logicalOr.getOperatorKind() != BinaryOperator::LogicalOr)
      continue;
    Value other = logicalOr.getLhs() == comparison.getResult()
                      ? logicalOr.getRhs() : logicalOr.getLhs();
    auto proof = other.getDefiningOp<CompareOp>();
    if (!proof || proof.getPredicate() != ComparePredicate::Lt) continue;
    auto left = queryLaunchExpression(proof.getLhs());
    auto right = queryLaunchExpression(proof.getRhs());
    if (left && right && left == upper && right == limit)
      return std::nullopt;
  }
  return IndexComparisonBound{upper, limit};
}

} // namespace intent::gpu

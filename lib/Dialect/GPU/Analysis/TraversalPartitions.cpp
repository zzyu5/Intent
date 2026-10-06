#include "Intent/Dialect/GPU/Analysis/TraversalPartitions.h"
#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool sameIndex(Value lhs, Value rhs, const IndexRelations &relations) {
  return relations.same(lhs, rhs) ||
      (lhs.getType().isIndex() && rhs.getType().isIndex() &&
       relations.atMost(lhs, rhs) && relations.atMost(rhs, lhs));
}

std::optional<bool> constantPredicate(Value value, const IndexRelations &relations) {
  if (auto constant = relations.constant(value)) return *constant != 0;
  auto compare = value.getDefiningOp<CompareOp>();
  if (!compare || !compare.getLhs().getType().isIndex() ||
      !compare.getRhs().getType().isIndex()) return std::nullopt;
  if (compare.getPredicate() == ComparePredicate::Lt) {
    if (relations.lessThan(compare.getLhs(), compare.getRhs())) return true;
    if (relations.atMost(compare.getRhs(), compare.getLhs())) return false;
  }
  if (compare.getPredicate() == ComparePredicate::Ge) {
    if (relations.atMost(compare.getRhs(), compare.getLhs())) return true;
    if (relations.lessThan(compare.getLhs(), compare.getRhs())) return false;
  }
  return std::nullopt;
}

bool tailCondition(Value condition, Value end, Value upper,
                   const IndexRelations &relations) {
  if (auto compare = condition.getDefiningOp<CompareOp>())
    return compare.getPredicate() == ComparePredicate::Lt &&
        sameIndex(compare.getLhs(), end, relations) &&
        sameIndex(compare.getRhs(), upper, relations);
  auto binary = condition.getDefiningOp<BinaryOp>();
  if (!binary || !condition.getType().isInteger(1)) return false;
  bool conjunction = binary.getOperatorKind() == BinaryOperator::LogicalAnd;
  bool disjunction = binary.getOperatorKind() == BinaryOperator::LogicalOr;
  if (!conjunction && !disjunction) return false;
  for (auto [constant, remaining] : {std::pair{binary.getLhs(), binary.getRhs()},
                                   std::pair{binary.getRhs(), binary.getLhs()}}) {
    auto truth = constantPredicate(constant, relations);
    if (truth && *truth == conjunction && tailCondition(remaining, end, upper, relations))
      return true;
  }
  return false;
}

class CoordinateMatch {
public:
  CoordinateMatch(scf::ForOp prefix, IndexRelations &relations) : relations(relations) {
    bindings[prefix.getInductionVar()] = prefix.getUpperBound();
    for (auto [argument, result] : llvm::zip(prefix.getRegionIterArgs(), prefix.getResults()))
      bindings[argument] = result;
  }

  bool same(Value lhs, Value rhs) {
    if (auto found = bindings.find(lhs); found != bindings.end())
      return sameIndex(found->second, rhs, relations);
    if (lhs == rhs) return true;
    if (lhs.getType() != rhs.getType()) return false;
    if (sameIndex(lhs, rhs, relations)) return true;
    auto left = dyn_cast<OpResult>(lhs), right = dyn_cast<OpResult>(rhs);
    if (!left || !right || left.getResultNumber() != right.getResultNumber()) return false;
    Operation *a = left.getOwner(), *b = right.getOwner();
    auto coordinate = [](Operation *operation) {
      return isa<MakeRangeOp>(operation) ||
          isPhysicalReplayNode(operation, PhysicalReplayScope::Coordinate, false);
    };
    if (a->getNumRegions() || b->getNumRegions() || !coordinate(a) || !coordinate(b)) return false;
    if (!OperationEquivalence::isEquivalentTo(a, b,
          [&](Value x, Value y) { return success(same(x, y)); }, nullptr,
          OperationEquivalence::IgnoreLocations)) return false;
    bindings[lhs] = rhs;
    return true;
  }

private:
  IndexRelations &relations;
  llvm::DenseMap<Value, Value> bindings;
};

bool matchingControl(Operation *first, Operation *second, scf::ForOp prefix,
                     scf::IfOp suffix, CoordinateMatch &match) {
  SmallVector<std::pair<scf::IfOp, bool>> left, right;
  auto collect = [](Operation *effect, Region *boundary,
                    SmallVectorImpl<std::pair<scf::IfOp, bool>> &path) {
    for (Region *region = effect->getParentRegion(); region != boundary;) {
      auto branch = region ? dyn_cast_or_null<scf::IfOp>(region->getParentOp()) : scf::IfOp();
      // No unmatched loop, While, execution group or other region may repeat
      // the effect inside the supposedly single suffix iteration.
      if (!branch) return false;
      path.emplace_back(branch, region == &branch.getThenRegion());
      region = branch->getParentRegion();
    }
    return true;
  };
  if (!collect(first, &prefix.getRegion(), left) ||
      !collect(second, &suffix.getThenRegion(), right) || left.size() != right.size()) return false;
  for (auto [a, b] : llvm::zip(llvm::reverse(left), llvm::reverse(right)))
    if (a.second != b.second || !match.same(a.first.getCondition(), b.first.getCondition())) return false;
  return true;
}

bool confinesTail(Value validity, MakeRangeOp range, PhysicalProgramAnalysis &analysis,
                  const IndexRelations &relations) {
  if (relations.coordinateLessThan(range.getResult(), range.getLogicalStop())) return true;
  if (!validity) return false;
  if (auto broadcast = validity.getDefiningOp<BroadcastOp>())
    return confinesTail(broadcast.getValue(), range, analysis, relations);
  if (auto splat = validity.getDefiningOp<SplatOp>())
    return confinesTail(splat.getValue(), range, analysis, relations);
  if (auto reshape = validity.getDefiningOp<ReshapeOp>())
    return confinesTail(reshape.getValue(), range, analysis, relations);
  if (auto transpose = validity.getDefiningOp<TransposeOp>())
    return confinesTail(transpose.getValue(), range, analysis, relations);
  if (auto binary = validity.getDefiningOp<BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::LogicalAnd)
      return confinesTail(binary.getLhs(), range, analysis, relations) ||
             confinesTail(binary.getRhs(), range, analysis, relations);
    if (binary.getOperatorKind() == BinaryOperator::LogicalOr)
      return confinesTail(binary.getLhs(), range, analysis, relations) &&
             confinesTail(binary.getRhs(), range, analysis, relations);
    return false;
  }
  auto compare = validity.getDefiningOp<CompareOp>();
  if (!compare || compare.getPredicate() != ComparePredicate::Lt) return false;
  Value limit = uniformScalarSource(compare.getRhs());
  return limit && sameIndex(limit, range.getLogicalStop(), relations) &&
      analysis.isTailPredicate(validity, {{range, range.getLogicalStop()}});
}

bool matchingStores(Operation *first, Operation *second, scf::ForOp prefix,
                    scf::IfOp suffix, Value upper, PhysicalProgramAnalysis &analysis,
                    IndexRelations &relations) {
  auto a = analysis.footprint(first), b = analysis.footprint(second);
  if (a.state != PhysicalFactState::Exact || b.state != PhysicalFactState::Exact ||
      a.rangeState != PhysicalFactState::Exact || b.rangeState != PhysicalFactState::Exact ||
      a.resource != b.resource || a.sourceAxes != b.sourceAxes ||
      a.coordinates.size() != b.coordinates.size()) return false;
  // Exact native boundary validity has no unrelated data-dependent filter. It
  // preserves all in-bounds members of these actual coordinate ranges, including
  // their original logical tail. Bounds remain checked by the ordinary verifier.
  if (analysis.boundaryValidity(first).state != PhysicalFactState::Exact)
    return false;
  if (analysis.boundaryValidity(second).state != PhysicalFactState::Exact)
    return false;
  if (!analysis.accessBounds(first).isExact() || !analysis.accessBounds(second).isExact())
    return false;
  CoordinateMatch match(prefix, relations);
  if (!matchingControl(first, second, prefix, suffix, match)) return false;
  bool anchored = false;
  for (auto [left, right] : llvm::zip(a.coordinates, b.coordinates)) {
    auto x = left.getDefiningOp<MakeRangeOp>(), y = right.getDefiningOp<MakeRangeOp>();
    if (x && y && sameIndex(x.getStart(), prefix.getInductionVar(), relations)) {
      auto type = x.getResult().getType();
      if (type != y.getResult().getType() || type.getShape().size() != 1 ||
          !(sourceAxisIdentity(x) == sourceAxisIdentity(y)) ||
          !isUnitStepRange(x) || !isUnitStepRange(y) ||
          !sameIndex(y.getStart(), prefix.getUpperBound(), relations) ||
          !sameIndex(x.getExtent(), prefix.getStep(), relations) ||
          !sameIndex(y.getExtent(), prefix.getStep(), relations) ||
          relations.constant(x.getLogicalStart()) != 0 ||
          relations.constant(y.getLogicalStart()) != 0 ||
          !sameIndex(x.getLogicalStop(), upper, relations) ||
          !sameIndex(y.getLogicalStop(), upper, relations) ||
          queryLaunchExpression(prefix.getStep()) != type.getShape()[0] ||
          !confinesTail(b.validity, y, analysis, relations)) return false;
      anchored = true;
      continue;
    }
    if (!match.same(left, right)) return false;
  }
  return anchored;
}

bool prefixAndSuffix(Operation *first, Operation *second,
                     PhysicalProgramAnalysis &analysis) {
  if (!isa<StoreOp>(first) || !isa<StoreOp>(second)) return false;
  IndexRelations relations;
  for (Operation *owner = first->getParentOp(); owner;
       owner = owner->getParentOp()) {
    auto prefix = dyn_cast<scf::ForOp>(owner);
    if (!prefix) continue;
    if (!prefix.getInductionVar().getType().isIndex() ||
        relations.constant(prefix.getLowerBound()) != 0 ||
        !relations.positive(prefix.getStep()))
      continue;
    auto rounded = prefix.getUpperBound().getDefiningOp<BinaryOp>();
    if (!rounded || rounded.getOperatorKind() != BinaryOperator::Multiply)
      continue;
    Value quotient;
    if (relations.same(rounded.getRhs(), prefix.getStep()))
      quotient = rounded.getLhs();
    else if (relations.same(rounded.getLhs(), prefix.getStep()))
      quotient = rounded.getRhs();
    auto division = quotient ? quotient.getDefiningOp<BinaryOp>() : BinaryOp();
    if (!division || division.getOperatorKind() != BinaryOperator::FloorDivide ||
        !relations.same(division.getRhs(), prefix.getStep()) ||
        !relations.nonnegative(division.getLhs()))
      continue;
    for (Operation *other = second->getParentOp(); other;
         other = other->getParentOp()) {
      auto suffix = dyn_cast<scf::IfOp>(other);
      if (!suffix || suffix.getThenRegion().empty() ||
          !suffix.getThenRegion().isAncestor(second->getParentRegion()) ||
          prefix->getBlock() != suffix->getBlock() ||
          !prefix->isBeforeInBlock(suffix))
        continue;
      if (!tailCondition(suffix.getCondition(), prefix.getUpperBound(), division.getLhs(), relations))
        continue;
      if (matchingStores(first, second, prefix, suffix, division.getLhs(), analysis, relations))
        return true;
    }
  }
  return false;
}

} // namespace

bool areDisjointTraversalPartitions(Operation *first, Operation *second,
                                   PhysicalProgramAnalysis &analysis) {
  return prefixAndSuffix(first, second, analysis) || prefixAndSuffix(second, first, analysis);
}

} // namespace intent::gpu

#ifndef INTENT_DIALECT_GPU_ANALYSIS_INDEXPREDICATES_H
#define INTENT_DIALECT_GPU_ANALYSIS_INDEXPREDICATES_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include <optional>

namespace intent::gpu {

// Prove the comparison for every physical lane from current ranges and loops.
// An absent result leaves the original predicate and all its users intact.
std::optional<bool> proveRangeComparison(CompareOp comparison);

// Exclusive bound for every lane of a complete, nonnegative physical tile.
// Recompute from the current range and enclosing loop; absent means unproven.
mlir::Value queryCompleteTileLimit(MakeRangeOp range);

// Inclusive upper bound and comparison limit, both specialization expressions.
// This is an implication guard: upper < limit implies the original comparison;
// it does not assert that arbitrary runtime shapes satisfy that relationship.
struct IndexComparisonBound {
  PhysicalExprAttr upper;
  PhysicalExprAttr limit;
};
std::optional<IndexComparisonBound> queryIndexComparisonBound(CompareOp comparison);

} // namespace intent::gpu

#endif

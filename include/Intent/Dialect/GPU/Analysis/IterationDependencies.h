#ifndef INTENT_DIALECT_GPU_ANALYSIS_ITERATIONDEPENDENCIES_H
#define INTENT_DIALECT_GPU_ANALYSIS_ITERATIONDEPENDENCIES_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace intent::gpu {

// Conditional independence of a current unit-step, carry-free iteration.
// The caller must preserve each iteration's internal order and materialize
// injective-layout guards for guardedViews and non-overlap guards for every
// disjointViews pair before reordering iterations. These are actual resources,
// not an execution plan; unknown effects or coordinate relations fail the query.
struct IndependentIterationAccesses {
  llvm::SmallVector<mlir::Value> guardedViews;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> disjointViews;
};

mlir::FailureOr<IndependentIterationAccesses>
queryIndependentIterationAccesses(mlir::scf::ForOp loop);

} // namespace intent::gpu

#endif

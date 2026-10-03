#ifndef INTENT_DIALECT_GPU_ANALYSIS_MEMORYEFFECTS_H
#define INTENT_DIALECT_GPU_ANALYSIS_MEMORYEFFECTS_H

#include "mlir/IR/Operation.h"

namespace intent::gpu {

class ResourceAliasAnalysis;

// Recursive effect contracts must be complete. Atomic observations and effects
// other than ordinary reads prevent cloning a region as a read-only producer;
// in particular this does not authorize duplicating control-local assumptions.
bool hasOnlyReadEffects(mlir::Operation *operation);

// Whether an intervening operation preserves every ordinary read of `reads`.
// Distinct effect resources cannot clobber one another. Within one resource,
// writes/frees need an explicit value with proven-disjoint allocation identity.
// Unknown effects and atomic ordering remain barriers.
bool preservesMemoryReads(mlir::Operation *reads, mlir::Operation *intervening,
                          ResourceAliasAnalysis &aliases);

} // namespace intent::gpu

#endif

#ifndef INTENT_DIALECT_GPU_TRANSFORMS_VALUERELATIONS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_VALUERELATIONS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

using ValueTypeChangeCallback =
    llvm::function_ref<void(mlir::Value, mlir::Type)>;

// Intermediate rewrites sometimes need only the relations they have made
// authoritative. Complete transformation entries close every relation.
enum class ValueRelationScope {
  Complete, Pointwise, Contracts, AccessResults, ReductionInputs
};

// Relation maintenance belongs inside a complete transformation, before verify.
void retargetSourceExtent(mlir::Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent,
                          std::optional<int64_t> dimension = std::nullopt,
                          ValueTypeChangeCallback changed = {});
void retargetDimensionExtent(mlir::Value root, int64_t dimensionId,
                             PhysicalExprAttr extent,
                             ValueTypeChangeCallback changed = {});
mlir::LogicalResult closeValueRelations(
    mlir::func::FuncOp kernel,
    ValueRelationScope scope = ValueRelationScope::Complete);
void eraseDeadPhysicalValues(mlir::func::FuncOp kernel);

} // namespace intent::gpu

#endif

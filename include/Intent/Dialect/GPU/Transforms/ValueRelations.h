#ifndef INTENT_DIALECT_GPU_TRANSFORMS_VALUERELATIONS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_VALUERELATIONS_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// Relation maintenance belongs inside a complete transformation, before verify.
void retargetSourceExtent(mlir::Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent,
                          std::optional<int64_t> dimension = std::nullopt);
void retargetDimensionExtent(mlir::Value root, int64_t dimensionId,
                             PhysicalExprAttr extent);
mlir::LogicalResult alignStructuredCaptureRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignPointwiseValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignAccessResultRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignAggregateValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignContractValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignAccessValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult refreshReshapeRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignReductionResultRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignReductionIdentityRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult alignReductionYieldRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult closeValueAccessRelations(mlir::func::FuncOp kernel);
void eraseDeadPhysicalValues(mlir::func::FuncOp kernel);

} // namespace intent::gpu

#endif

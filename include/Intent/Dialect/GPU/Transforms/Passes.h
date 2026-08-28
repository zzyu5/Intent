#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

struct PhysicalSourceAxis;

enum class GPUProgramStage {
  /// Construction has emitted a complete typed shell; structured operations
  /// still own their explicit semantic regions but all effects and mappings
  /// are already represented in GPU IR.
  StructuredConstruction,
  /// Every structured shell has been materialized into executable control,
  /// carry, access and value flow.
  Executable,
};

mlir::FailureOr<mlir::func::FuncOp>
getPhysicalKernel(mlir::ModuleOp module);
mlir::FailureOr<uint64_t> blockedDimension(mlir::Attribute attribute);
bool hasBlockedDimension(mlir::func::FuncOp kernel, uint64_t dimension);
void retargetSourceExtent(mlir::Value root, uint64_t sourceId,
                          PhysicalExprAttr extent);
void retargetDimensionExtent(mlir::Value root, int64_t dimensionId,
                             PhysicalExprAttr extent);
mlir::LogicalResult
alignStructuredCaptureRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignPointwiseValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignAggregateValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult bindFullCoverageDimension(mlir::func::FuncOp kernel,
                                              uint64_t dimension,
                                              mlir::Value physicalExtent);
mlir::FailureOr<mlir::Value>
resolveLogicalRangeEnd(mlir::func::FuncOp kernel, MakeRangeOp range);
mlir::FailureOr<mlir::Value>
materializeScalarConstant(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Attribute value, mlir::Type resultType);
mlir::FailureOr<mlir::Value>
materializeBroadcastToFragment(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value value,
                               FragmentType target);
mlir::FailureOr<mlir::Value>
materializeZeroFragment(mlir::OpBuilder &builder, mlir::Location location,
                        FragmentType target);
mlir::FailureOr<mlir::Value>
projectPredicateToFragment(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate, FragmentType target,
                           PhysicalSourceAxis source);
mlir::FailureOr<mlir::Value>
projectPredicateToFragment(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate, FragmentType target,
                           int64_t dimensionId);
mlir::FailureOr<mlir::Value>
materializeValidityConjunction(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value lhs,
                               mlir::Value rhs, FragmentType valueType);
mlir::LogicalResult verifyGPUProgram(mlir::ModuleOp module);
mlir::LogicalResult verifyGPUProgramStage(mlir::ModuleOp module,
                                          GPUProgramStage stage);
void eraseDeadPhysicalValues(mlir::func::FuncOp kernel);
void eraseUnusedPhysicalParameters(mlir::func::FuncOp kernel);
mlir::LogicalResult realizeAccessComposition(mlir::ModuleOp module);
mlir::LogicalResult realizeRegionFolds(mlir::ModuleOp module);
mlir::LogicalResult realizeRegionScans(mlir::ModuleOp module);
mlir::LogicalResult realizeContractionBlocking(mlir::ModuleOp module);
mlir::LogicalResult decomposeMultiAxisReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeReductionBlocking(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseOwnership(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseBlocking(mlir::ModuleOp module);
mlir::LogicalResult refineProgramMapping(mlir::ModuleOp module);
mlir::LogicalResult completeGPUProgramConstruction(mlir::ModuleOp module);
mlir::LogicalResult runSharedGPUPasses(mlir::ModuleOp module);

} // namespace intent::gpu

#endif

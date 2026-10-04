#ifndef INTENT_GPU_TRANSFORMS_REDUCTIONREALIZATION_H
#define INTENT_GPU_TRANSFORMS_REDUCTIONREALIZATION_H

#include "ReductionAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::gpu::reduction {

// Private realization entry points. The public pass driver owns their order
// and refreshes worklists after rewrites; these are not independently run passes.

// Preserve unstable source snapshots before creating a new chunk traversal.
mlir::LogicalResult prepareReductionReads(
    llvm::ArrayRef<RootAccess> accesses, ReduceOp reduce,
    mlir::func::FuncOp kernel);

mlir::FailureOr<mlir::Value> materializeReductionRead(
    mlir::OpBuilder &builder, mlir::Location location, RootAccess access,
    FragmentType resultType, mlir::ValueRange coordinates,
    mlir::Value valid, mlir::Value fill, mlir::Value traversalCoordinate,
    ReduceOp reduce);

mlir::FailureOr<bool> realizeStaticPaddingReduce(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::LogicalResult neutralizeReductionTails(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::FailureOr<bool> realizeFullCoverageReduce(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::LogicalResult decomposeMultiAxisReduce(
    ReduceOp reduce, mlir::func::FuncOp kernel);

mlir::LogicalResult realizeRuntimeReduce(
    ReduceOp reduce, llvm::ArrayRef<SourcePlan> sourcePlans,
    mlir::func::FuncOp kernel, bool tileProducerFreeAxis);

bool hoistNestedReduction(mlir::scf::ForOp outer, mlir::func::FuncOp kernel);

bool sinkReductionIntoSourceIf(ReduceOp reduce, mlir::func::FuncOp kernel);

// Collapse sole-use ordinary reduction chains over complete physical members.
bool normalizeCompletedReductions(mlir::func::FuncOp kernel);

} // namespace intent::gpu::reduction
#endif

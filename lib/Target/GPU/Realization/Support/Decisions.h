#ifndef INTENT_LIB_TARGET_GPU_REALIZATION_SUPPORT_DECISIONS_H
#define INTENT_LIB_TARGET_GPU_REALIZATION_SUPPORT_DECISIONS_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Support/Model.h"

namespace intent::gpu::realization {

struct PhysicalDecisions {
  intent::plan::LaunchOp program;
  llvm::SmallVector<intent::plan::BlockExtentOp> blockExtents;
  llvm::SmallVector<intent::plan::AxisOp> axes;
  llvm::SmallVector<intent::plan::RangeOp> ranges;
  llvm::SmallVector<intent::plan::RegionBindingOp> regionBindings;
  llvm::SmallVector<intent::plan::StreamBindingOp> streamBindings;
  llvm::SmallVector<intent::plan::StageOp> stages;
  llvm::SmallVector<intent::plan::StageAxisOp> stageAxes;
};

mlir::FailureOr<PhysicalDecisions>
emitPhysicalDecisions(mlir::OpBuilder &builder, const KernelFacts &facts);

mlir::LogicalResult
emitSearchSpace(mlir::ModuleOp module, const KernelFacts &facts,
                const PhysicalDecisions &decisions);

} // namespace intent::gpu::realization

#endif

#ifndef INTENT_TARGET_GPU_TRANSFORMS_PASSES_H
#define INTENT_TARGET_GPU_TRANSFORMS_PASSES_H

#include "Intent/Target/GPU/Config/Device.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace intent::gpu {

std::unique_ptr<mlir::Pass>
createConstructPhysicalProgramPass(const DeviceCapabilities &device);
std::unique_ptr<mlir::Pass> createFormAutomaticBlockingPass();
std::unique_ptr<mlir::Pass> createReconcileAccessRangesPass();
std::unique_ptr<mlir::Pass> createRefineTransferRealizationPass();
std::unique_ptr<mlir::Pass> createFormStagesPass();
std::unique_ptr<mlir::Pass>
createRefinePrivateBufferResidencyPass(const DeviceCapabilities &device);
std::unique_ptr<mlir::Pass> createRefineContractionRealizationPass();
std::unique_ptr<mlir::Pass> createRefineScanRealizationPass();
std::unique_ptr<mlir::Pass> createRefineValueRealizationPass();
std::unique_ptr<mlir::Pass> createRefinePersistentTraversalPass();
std::unique_ptr<mlir::Pass> createRefineBoundaryNeutralizationPass();
std::unique_ptr<mlir::Pass> createMaterializeSearchSpacePass();
std::unique_ptr<mlir::Pass> createVerifyPhysicalProgramPass();

mlir::LogicalResult runPhysicalProgramPipeline(
    mlir::ModuleOp module, const DeviceCapabilities &device);

} // namespace intent::gpu

#endif

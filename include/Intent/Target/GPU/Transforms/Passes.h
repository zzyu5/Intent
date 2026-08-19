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
std::unique_ptr<mlir::Pass> createVerifyPhysicalProgramPass();

mlir::LogicalResult runPhysicalProgramPipeline(
    mlir::ModuleOp module, const DeviceCapabilities &device);

} // namespace intent::gpu

#endif

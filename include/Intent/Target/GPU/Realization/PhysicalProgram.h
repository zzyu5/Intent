#ifndef INTENT_TARGET_GPU_REALIZATION_PHYSICALPROGRAM_H
#define INTENT_TARGET_GPU_REALIZATION_PHYSICALPROGRAM_H

#include "Intent/Target/GPU/Config/Device.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

mlir::LogicalResult constructPhysicalProgram(
    mlir::ModuleOp module, const DeviceCapabilities &device);

} // namespace intent::gpu

#endif

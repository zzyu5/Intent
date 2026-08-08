#ifndef INTENT_TARGET_GPU_REALIZATION_REALIZE_H
#define INTENT_TARGET_GPU_REALIZATION_REALIZE_H

#include "Intent/Target/GPU/Config/Device.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

mlir::LogicalResult realizeKernel(mlir::ModuleOp module,
                                  const DeviceCapabilities &device);

} // namespace intent::gpu

#endif

#ifndef INTENT_LIB_TARGET_GPU_REALIZATION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_GPU_REALIZATION_SUPPORT_MODEL_H

#include "Intent/Target/Common/Realization/KernelFacts.h"
#include "Intent/Target/GPU/Config/Device.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu::realization {

using KernelFacts = intent::target::KernelFacts;

mlir::LogicalResult buildPhysicalProgram(mlir::ModuleOp module,
                                         const DeviceCapabilities &device,
                                         const KernelFacts &facts);

} // namespace intent::gpu::realization

#endif

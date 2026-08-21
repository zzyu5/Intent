#ifndef INTENT_TARGET_GPU_REALIZATION_PHYSICALPROGRAM_H
#define INTENT_TARGET_GPU_REALIZATION_PHYSICALPROGRAM_H

#include "Intent/Target/GPU/Config/Device.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Realization/KernelFacts.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

mlir::LogicalResult constructPhysicalProgram(
    mlir::ModuleOp module, const DeviceCapabilities &device);

mlir::LogicalResult materializeSearchSpace(
    mlir::ModuleOp module, const intent::target::KernelFacts &facts,
    intent::plan::ProgramOp program);
mlir::LogicalResult formAutomaticBlocking(
    intent::plan::ProgramOp program,
    const intent::target::KernelFacts &facts);
mlir::LogicalResult reconcileAccessRanges(
    intent::plan::ProgramOp program,
    const intent::target::KernelFacts &facts);
mlir::LogicalResult reconcileStages(
    intent::plan::ProgramOp program,
    const intent::target::KernelFacts &facts);

} // namespace intent::gpu

#endif

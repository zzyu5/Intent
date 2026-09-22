#ifndef INTENT_TARGET_TRITON_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TRITON_TRANSFORMS_PASSES_H

#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
namespace intent::triton {

mlir::LogicalResult legalizeProgramGrid(mlir::ModuleOp module);
mlir::LogicalResult legalizeGPUProgram(mlir::ModuleOp module,
                                     const gpu::TuningProfiles &profiles);
mlir::LogicalResult verifyTritonProgram(mlir::ModuleOp module);

} // namespace intent::triton

#endif

#ifndef INTENT_TARGET_TILELANG_TRANSFORMS_PASSDETAIL_H
#define INTENT_TARGET_TILELANG_TRANSFORMS_PASSDETAIL_H

#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::tilelang {

mlir::LogicalResult materializeLaunchConfiguration(
    mlir::func::FuncOp kernel, const gpu::TuningProfiles &profiles);
mlir::LogicalResult bufferizeGPUProgram(mlir::func::FuncOp kernel);
mlir::LogicalResult formPipelines(mlir::func::FuncOp kernel,
                                const gpu::TuningProfiles &profiles);
mlir::LogicalResult verifyTileLangKernel(mlir::func::FuncOp kernel);

} // namespace intent::tilelang

#endif

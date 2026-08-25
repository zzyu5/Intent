#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

mlir::FailureOr<mlir::func::FuncOp>
getPhysicalKernel(mlir::ModuleOp module);
mlir::LogicalResult verifyGPUProgram(mlir::ModuleOp module);
void eraseDeadPhysicalValues(mlir::func::FuncOp kernel);
mlir::LogicalResult realizeAccessComposition(mlir::ModuleOp module);
mlir::LogicalResult realizeContractionBlocking(mlir::ModuleOp module);
mlir::LogicalResult realizeReductionBlocking(mlir::ModuleOp module);
mlir::LogicalResult runSharedGPUPasses(mlir::ModuleOp module);

} // namespace intent::gpu

#endif

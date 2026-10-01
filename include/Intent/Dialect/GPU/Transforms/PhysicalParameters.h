#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PHYSICALPARAMETERS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PHYSICALPARAMETERS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

mlir::FailureOr<ParameterRefAttr> declareParameter(
    mlir::func::FuncOp kernel, ParameterAttr declaration);
mlir::FailureOr<ParameterRefAttr> getOrCreatePhysicalParameter(
    mlir::func::FuncOp kernel, llvm::StringRef name, ParameterRole role,
    ParameterCategory category, uint32_t elementBitWidth,
    llvm::ArrayRef<int64_t> candidates, ParameterBindingAttr binding = {});
ParameterOp materializeParameter(mlir::OpBuilder &builder, mlir::Location location,
                                ParameterRefAttr reference);
// Declaration changes invalidate candidate bindings. Reference changes also
// rewrite operation attributes, result types and region argument types.
mlir::LogicalResult updateParameter(mlir::func::FuncOp kernel,
                                    ParameterAttr declaration);
mlir::LogicalResult replaceParameter(mlir::func::FuncOp kernel,
                                     ParameterRefAttr previous,
                                     ParameterRefAttr replacement);
mlir::LogicalResult renameParameters(
    mlir::func::FuncOp kernel,
    llvm::function_ref<mlir::StringAttr(mlir::StringAttr)> rename);
void eraseUnusedParameters(mlir::func::FuncOp kernel);
mlir::LogicalResult materializeSharedConfigTuples(mlir::func::FuncOp kernel);
mlir::LogicalResult verifySharedConfigTuples(mlir::func::FuncOp kernel);
mlir::LogicalResult writeConfigurations(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::DictionaryAttr> rows,
    ConfigurationStage stage);

} // namespace intent::gpu

#endif

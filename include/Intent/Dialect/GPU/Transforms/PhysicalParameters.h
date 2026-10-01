#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PHYSICALPARAMETERS_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PHYSICALPARAMETERS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

ParameterOp getOrCreatePhysicalParameter(
    mlir::func::FuncOp kernel, llvm::StringRef name, ParameterRole role,
    ParameterCategory category, uint32_t elementBitWidth,
    llvm::ArrayRef<int64_t> candidates);
void eraseUnusedPhysicalParameters(mlir::func::FuncOp kernel);
mlir::LogicalResult replacePhysicalParameter(mlir::func::FuncOp kernel,
                                             ParameterOp previous,
                                             ParameterOp replacement);
mlir::LogicalResult materializeSharedConfigTuples(mlir::func::FuncOp kernel);
mlir::LogicalResult verifySharedConfigTuples(mlir::func::FuncOp kernel);
mlir::LogicalResult writeConfigurations(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::DictionaryAttr> rows,
    ConfigurationStage stage);

} // namespace intent::gpu

#endif

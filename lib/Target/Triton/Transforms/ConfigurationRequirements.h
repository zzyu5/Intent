#ifndef INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONREQUIREMENTS_H
#define INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONREQUIREMENTS_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::triton {

// Read current native operations and declarations at each complete boundary.
mlir::FailureOr<llvm::SmallVector<gpu::ConfigurationRequirementAttr>>
collectConfigurationRequirements(mlir::func::FuncOp kernel);

mlir::LogicalResult
finalizeConfigurationRequirements(mlir::func::FuncOp kernel);

mlir::LogicalResult
verifyConfigurationRequirements(mlir::func::FuncOp kernel);

} // namespace intent::triton
#endif

#ifndef INTENT_TARGET_GPU_REALIZATION_SUPPORT_OPERATIONS_H
#define INTENT_TARGET_GPU_REALIZATION_SUPPORT_OPERATIONS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu::realization {

mlir::LogicalResult materializePhysicalOperations(mlir::func::FuncOp entry);

} // namespace intent::gpu::realization

#endif

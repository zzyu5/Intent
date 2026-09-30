#ifndef INTENT_DIALECT_GPU_TRANSFORMS_UTILITIES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_UTILITIES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

/// Closes access-result, pointwise, access-value, reshape and contraction
/// relations in dependency order inside a complete transformation group.
mlir::LogicalResult closeValueAccessRelations(mlir::func::FuncOp kernel);

} // namespace intent::gpu

#endif

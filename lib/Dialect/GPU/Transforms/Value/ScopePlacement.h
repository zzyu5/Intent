#ifndef INTENT_GPU_TRANSFORMS_VALUE_SCOPEPLACEMENT_H
#define INTENT_GPU_TRANSFORMS_VALUE_SCOPEPLACEMENT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {
class ResourceAliasAnalysis;

namespace placement {

// These queries concern one current rewrite, not a cached schedule. Unknown
// effects and ordered accesses cannot establish independent execution.
bool independentMemoryEffects(mlir::Operation *first, mlir::Operation *second,
                              ResourceAliasAnalysis &aliases);

bool isMovableValueOperation(mlir::Operation *operation);
bool canMoveBefore(mlir::Operation *operation, mlir::Operation *before);

// Move only the intervening definitions required by second's operands and
// captures. No operation moves until the complete dependency slice is legal.
bool moveInputsBefore(mlir::Operation *first, mlir::Operation *second,
                      mlir::func::FuncOp kernel);

} // namespace placement
} // namespace intent::gpu

#endif

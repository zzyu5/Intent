#ifndef INTENT_DIALECT_GPU_TRANSFORMS_RESOURCES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_RESOURCES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

// Bounds that depend on ABI specialization remain explicit executable asserts.
// Candidate-static bounds are checked before serialization by the provider.
void materializeDeferredReductionBounds(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::ValueRange> sourceGroups);

} // namespace intent::gpu

#endif

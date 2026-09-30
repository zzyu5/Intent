#ifndef INTENT_DIALECT_GPU_TRANSFORMS_STORAGE_H
#define INTENT_DIALECT_GPU_TRANSFORMS_STORAGE_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace intent::gpu {

mlir::Value createInvocationWorkspace(mlir::func::FuncOp kernel,
                                      mlir::Location location,
                                      mlir::Type elementType,
                                      mlir::ArrayAttr shape, uint64_t owner);
mlir::FailureOr<mlir::Value> materializeRetainedSlice(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    unsigned axis, PhysicalExprAttr blockedExtent, mlir::Value coordinates,
    mlir::Operation *insertionAnchor);

} // namespace intent::gpu

#endif

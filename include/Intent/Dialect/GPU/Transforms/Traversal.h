#ifndef INTENT_DIALECT_GPU_TRANSFORMS_TRAVERSAL_H
#define INTENT_DIALECT_GPU_TRANSFORMS_TRAVERSAL_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

PhysicalExprAttr boundedTraversalChunk(ParameterAttr chunk, MakeRangeOp range);
mlir::LogicalResult bindFullCoverageDimension(mlir::func::FuncOp kernel,
                                              uint64_t dimension,
                                              mlir::Value physicalExtent);
mlir::LogicalResult realizeFullCoverageDimension(mlir::func::FuncOp kernel,
                                                 mlir::Value source,
                                                 unsigned fragmentAxis);

} // namespace intent::gpu

#endif

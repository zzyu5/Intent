#ifndef INTENT_DIALECT_GPU_TRANSFORMS_TRAVERSAL_H
#define INTENT_DIALECT_GPU_TRANSFORMS_TRAVERSAL_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

// Attach the loop to current IR before building a body that queries dominance,
// resource snapshots or enclosing program facts. The callback owns its yield.
mlir::scf::ForOp createTraversalLoop(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value lower,
    mlir::Value upper, mlir::Value step, mlir::ValueRange initialValues,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::Value,
                            mlir::ValueRange)> buildBody);

mlir::FailureOr<PhysicalExprAttr> boundedTraversalChunk(ParameterAttr chunk,
                                                       MakeRangeOp range);
mlir::LogicalResult bindFullCoverageDimension(mlir::func::FuncOp kernel,
                                              uint64_t dimension,
                                              mlir::Value physicalExtent);
mlir::LogicalResult realizeFullCoverageDimension(mlir::func::FuncOp kernel,
                                                 mlir::Value source,
                                                 unsigned fragmentAxis);

} // namespace intent::gpu

#endif

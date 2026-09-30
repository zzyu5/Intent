#ifndef INTENT_DIALECT_GPU_ANALYSIS_VALUESCHEMA_H
#define INTENT_DIALECT_GPU_ANALYSIS_VALUESCHEMA_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::gpu {

// These queries inspect current IR and return facts/types without changing it.
bool isShapeBound(PhysicalExprAttr bound);
mlir::Type scalarCallbackType(mlir::Type type);
bool canPredicateValueOperation(mlir::Operation *operation);
bool variesWithIteration(mlir::Value value, mlir::scf::ForOp loop,
                         llvm::DenseMap<mlir::Value, bool> &known);
std::pair<uint64_t, int64_t>
nextPhysicalAxisIdentities(mlir::func::FuncOp kernel);
PhysicalExprAttr multiplyExtent(PhysicalExprAttr lhs, PhysicalExprAttr rhs);
PhysicalExprAttr productExtent(mlir::MLIRContext *context,
                               llvm::ArrayRef<mlir::Attribute> shape,
                               llvm::ArrayRef<int64_t> axes, unsigned prefix);
bool isIntroducedReshapeUnitAxis(mlir::Value value, unsigned fragmentAxis);
std::optional<unsigned> reshapeInputAxis(ReshapeOp reshape, unsigned resultAxis);
mlir::Value stripAdditiveProjection(mlir::Value value, bool singleUse = false);
bool isLiteralZeroProjection(mlir::Value value);
mlir::FailureOr<FragmentType> queryValueSchema(
    mlir::func::FuncOp kernel, FragmentType target,
    mlir::ValueRange contributors);
mlir::FailureOr<FragmentType> queryAccessResultSchema(
    mlir::func::FuncOp kernel, FragmentType target,
    mlir::ValueRange coordinates);
mlir::FailureOr<uint64_t> blockedDimension(mlir::Attribute attribute);
bool hasBlockedDimension(mlir::func::FuncOp kernel, uint64_t dimension);

} // namespace intent::gpu

#endif

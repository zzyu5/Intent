#ifndef INTENT_DIALECT_CPU_TRANSFORMS_STORAGE_STORAGE_H
#define INTENT_DIALECT_CPU_TRANSFORMS_STORAGE_STORAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

void eraseDeadPrivateBuffers(mlir::func::FuncOp function);
mlir::LogicalResult reusePrivateStorage(mlir::func::FuncOp function);
enum class ScratchRepresentation { PreserveDescriptors, LinearCapacity };
mlir::LogicalResult reuseScratchStorage(mlir::func::FuncOp function,
                                       ScratchRepresentation representation);
void forwardCPUOutputs(mlir::func::FuncOp function);
mlir::LogicalResult fuseIntermediateBuffers(mlir::func::FuncOp function);

} // namespace intent::cpu
#endif

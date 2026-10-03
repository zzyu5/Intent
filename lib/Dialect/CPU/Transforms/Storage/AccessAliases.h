#ifndef INTENT_CPU_TRANSFORMS_STORAGE_ACCESSALIASES_H
#define INTENT_CPU_TRANSFORMS_STORAGE_ACCESSALIASES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

// Expose scalar accesses to private storage in the allocation's coordinates so
// the existing producer replay policies can inspect the actual indexed value.
mlir::LogicalResult foldPrivateAccessAliases(mlir::func::FuncOp function);

} // namespace intent::cpu

#endif

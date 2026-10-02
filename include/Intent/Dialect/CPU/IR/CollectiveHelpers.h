#ifndef INTENT_DIALECT_CPU_IR_COLLECTIVEHELPERS_H
#define INTENT_DIALECT_CPU_IR_COLLECTIVEHELPERS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"

namespace intent::cpu {

// The scratch representation of one immutable scalar or shaped state.
mlir::MemRefType collectiveStateType(mlir::Type type);
// Drop member axes while retaining the source descriptor's arbitrary layout.
mlir::MemRefType collectiveMemberType(mlir::MemRefType source,
                                     llvm::ArrayRef<int64_t> axes);

// Helper inputs are read-only during invocation, including opaque incoming
// states. This does not imply that the formal owns a fresh allocation.
bool isCollectiveArgument(mlir::BlockArgument argument);
bool isReadOnlyCollectiveArgument(mlir::BlockArgument argument);

// Read-only explicit inputs and caller-owned destinations. The body may own
// temporary allocations and nested pure structured computations.
mlir::LogicalResult verifyCollectiveHelper(
    mlir::Operation *owner, mlir::Region &region, unsigned destinations);
mlir::LogicalResult verifyCollectiveHelperEffects(
    mlir::Operation *owner, mlir::Region &region, unsigned destinations);

} // namespace intent::cpu
#endif

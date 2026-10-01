#ifndef INTENT_DIALECT_GPU_IR_ACCESSOPINTERFACE_H
#define INTENT_DIALECT_GPU_IR_ACCESSOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include <optional>

namespace intent::gpu {

// These kinds describe the operation's existing semantics. They do not grant
// permission to predicate, reorder, duplicate, or alias an access.
// Payload order for compare-exchange is expected, desired; its predicate follows
// that payload schema rather than the {old_value, success} result record.
enum class AccessKind {
  Load, Gather, Store, ScatterReduce,
  AtomicLoad, AtomicStore, AtomicRMW, AtomicCompareExchange
};

class AccessOpInterface;
mlir::LogicalResult verifyAccessSchema(AccessOpInterface access);

} // namespace intent::gpu

#include "Intent/Dialect/GPU/IR/AccessOpInterface.h.inc"

#endif

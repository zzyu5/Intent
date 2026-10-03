#ifndef INTENT_DIALECT_DSA_IR_VIEWS_H
#define INTENT_DIALECT_DSA_IR_VIEWS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace intent::dsa {

// A complete, zero-offset contiguous local view rooted in an allocation or a
// borrowed collective argument. Partial or differently strided aliases fail.
bool isCompleteLocalStorageView(mlir::Value value);

// Re-expose that storage at its selected capacity rank, reusing an existing
// matching view or allocation whenever possible. Ownership and elements do not
// change, and this builder never creates a partial view.
mlir::FailureOr<mlir::Value>
materializeCollectiveView(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value source, mlir::MemRefType requested);

} // namespace intent::dsa

#endif

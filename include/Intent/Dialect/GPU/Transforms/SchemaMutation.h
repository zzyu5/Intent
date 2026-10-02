#ifndef INTENT_DIALECT_GPU_TRANSFORMS_SCHEMAMUTATION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_SCHEMAMUTATION_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {

using ValueTypeChangeCallback = llvm::function_ref<void(mlir::Value, mlir::Type)>;

void setPhysicalValueType(mlir::Value value, mlir::Type type,
                          ValueTypeChangeCallback changed = {});

// source remains unchanged. Snapshot its operation-owned coordinate relations
// before changing clone results/formals; transport those relations using the
// actual remapped clone operands. transform receives the unchanged source Value
// and returns its clone's selected type. A null type is unsupported.
mlir::LogicalResult rewriteClonedPhysicalTypes(
    mlir::Operation *source, mlir::Operation *clone,
    llvm::function_ref<mlir::Type(mlir::Value)> transform,
    ValueTypeChangeCallback changed = {});

bool hasSchemaBoundary(mlir::Operation *operation);

// Project incoming operand slots at their own boundary. Shared SSA seeds are
// never rewritten as if distinct carried components were one schema identity.
mlir::LogicalResult closeSchemaBoundary(
    mlir::Operation *operation, ValueTypeChangeCallback changed = {},
    mlir::OpBuilder::Listener *listener = nullptr);

// Apply an explicit extent decision to the selected control component's
// incoming slots. This never propagates that decision into a shared seed SSA.
mlir::LogicalResult projectSchemaBoundary(
    mlir::Value target, ValueTypeChangeCallback changed = {},
    mlir::OpBuilder::Listener *listener = nullptr);

} // namespace intent::gpu
#endif

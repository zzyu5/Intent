#ifndef INTENT_DIALECT_GPU_TRANSFORMS_EXECUTIONSCHEMA_H
#define INTENT_DIALECT_GPU_TRANSFORMS_EXECUTIONSCHEMA_H

#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::gpu {

struct LiftedFragmentSchema {
  FragmentType type;
  llvm::SmallVector<unsigned> oldToNew;
  llvm::SmallVector<unsigned> executionToNew;
};

// A short-lived selection of independent execution axes, never a new policy or
// an execution plan. Original axis occurrences remain distinct and ordered.
class ExecutionSchema {
public:
  explicit ExecutionSchema(FragmentType selectedAxes) : selectedAxes(selectedAxes) {}
  mlir::FailureOr<mlir::Type> lift(mlir::Type original) const;
  mlir::FailureOr<LiftedFragmentSchema> project(FragmentType original) const;

private:
  FragmentType selectedAxes;
};

// Infer only the elementwise/product schema established by mapped operands.
// Operations whose result axes are a decision require explicit selected types.
mlir::FailureOr<llvm::SmallVector<mlir::Type>>
inferElementwiseSchema(mlir::Operation *source, mlir::ValueRange mappedOperands);

// Rebind axis-bearing attributes against original types saved before mutation.
// Equal-rank transport retains axis positions; rank expansion embeds existing
// occurrences. This is not a query for an arbitrary axis permutation.
mlir::LogicalResult remapSchemaAxes(mlir::Operation *operation,
                                  mlir::TypeRange originalOperands,
                                  mlir::TypeRange originalResults);

// Clone attributes/properties/regions with MLIR and project the operation's
// operands/results. Callers that transform a region own its formal/yield and
// control-flow rewrite. Empty selectedResults requests elementwise inference.
// A projection may reuse a value instead of creating an operation.
mlir::FailureOr<llvm::SmallVector<mlir::Value>> cloneWithSchema(
    mlir::OpBuilder &builder, mlir::Operation *source, mlir::IRMapping &mapping,
    mlir::TypeRange selectedResults = {});

} // namespace intent::gpu
#endif

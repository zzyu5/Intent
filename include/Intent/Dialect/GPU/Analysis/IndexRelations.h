#ifndef INTENT_DIALECT_GPU_ANALYSIS_INDEXRELATIONS_H
#define INTENT_DIALECT_GPU_ANALYSIS_INDEXRELATIONS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu {

// Read-only relations over the current typed index graph. Unknown arithmetic,
// invalid divisors and unrepresentable shape expressions do not prove a fact.
// These queries neither insert guards nor choose a provider access form.
// Provider-rebound ResidentWorkers candidates are not a fixed analysis domain;
// only their positive-capacity contract and domain-independent proofs are used.
class IndexRelations {
public:
  std::optional<int64_t> constant(mlir::Value value) const;
  std::optional<int64_t> constant(PhysicalExprAttr expression,
                                  mlir::func::FuncOp kernel) const;
  bool same(mlir::Value lhs, mlir::Value rhs) const;
  bool nonnegative(mlir::Value value) const;
  bool positive(mlir::Value value) const;
  bool atMost(mlir::Value lhs, mlir::Value rhs) const;
  bool powerOfTwo(mlir::Value value) const;
  bool multipleOf(mlir::Value value, mlir::Value divisor) const;

  // An optional leaf proof is valid only within the caller's already guarded
  // specialization. It is consulted for typed parameter declarations, never
  // persisted as a fact about the unrestricted parameter domain.
  bool multipleOf(mlir::Value value, int64_t divisor,
                  llvm::function_ref<bool(ParameterOp)> alignedParameter = {}) const;

  // Return an equal aligned endpoint, including a selected min arm whose
  // ordering is proven. This does not prove that an entire loop is nonempty.
  mlir::Value alignedBound(mlir::Value value, mlir::Value step) const;

private:
  bool nonnegative(mlir::Value value, unsigned depth) const;
  bool atMost(mlir::Value lhs, mlir::Value rhs, unsigned depth) const;
};

} // namespace intent::gpu
#endif

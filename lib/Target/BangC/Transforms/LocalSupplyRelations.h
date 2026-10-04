#ifndef INTENT_TARGET_BANGC_TRANSFORMS_LOCALSUPPLYRELATIONS_H
#define INTENT_TARGET_BANGC_TRANSFORMS_LOCALSUPPLYRELATIONS_H

#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "mlir/IR/AffineExpr.h"

namespace intent::bangc {

// Current typed scalar coordinates of a local supply. Discard after mutation.
class LocalSupplyRelations {
public:
  explicit LocalSupplyRelations(mlir::func::FuncOp function);
  dsa::SignedInterval interval(mlir::Value value);
  bool equal(mlir::Value value, int64_t expected);

private:
  std::optional<int64_t> constant(mlir::Value value);
  mlir::AffineExpr expression(mlir::Value value);
  mlir::func::FuncOp function;
  intent::IntegerRangeAnalysis ranges;
  llvm::DenseMap<mlir::Value, mlir::AffineExpr> expressions;
  unsigned symbols = 0;
};

} // namespace intent::bangc
#endif

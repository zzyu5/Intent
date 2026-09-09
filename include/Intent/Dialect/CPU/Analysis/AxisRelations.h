#ifndef INTENT_DIALECT_CPU_ANALYSIS_AXISRELATIONS_H
#define INTENT_DIALECT_CPU_ANALYSIS_AXISRELATIONS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace intent::cpu {

// Coordinate equivalence in the current shaped program, not extent equality.
// Construct again after a rewrite. Positional maps remain the IR authority.
class AxisRelations {
public:
  explicit AxisRelations(mlir::func::FuncOp function);
  llvm::SmallVector<int64_t> axes(mlir::Value memory);
  unsigned dynamicAxisCount() const { return dynamicAxes; }

private:
  unsigned root(unsigned item);
  void unite(unsigned lhs, unsigned rhs);
  void equate(mlir::Value lhs, mlir::Value rhs);
  llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned>> positions;
  llvm::SmallVector<unsigned> parents;
  llvm::DenseMap<unsigned, int64_t> identities;
  unsigned dynamicAxes = 0;
};

}
#endif

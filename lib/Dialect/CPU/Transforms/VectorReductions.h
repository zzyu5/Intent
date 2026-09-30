#pragma once

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::cpu {

// Reduce adjacent lanes of a nonempty tuple of fixed rank-one vectors with
// equal power-of-two widths. The callback combines corresponding components;
// source permutation and accumulator initialization remain caller decisions.
llvm::SmallVector<mlir::Value> horizontalReduce(
    mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(
        mlir::ValueRange, mlir::ValueRange, int64_t)> combine);

} // namespace intent::cpu

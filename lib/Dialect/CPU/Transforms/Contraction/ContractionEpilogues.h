#pragma once

#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace intent::cpu {

// A pointwise observation of a complete private accumulator. The proof belongs
// to the current IR snapshot; move the recorded pure dependencies before any
// contraction rewriting, then emit only after a tile's complete K traversal.
struct ContractionEpilogue {
  mlir::linalg::GenericOp consumer;
  unsigned accumulatorInput;
  llvm::SmallVector<mlir::Operation *> dependencies;
};

std::optional<ContractionEpilogue> queryContractionEpilogue(
    mlir::linalg::GenericOp contraction, mlir::Operation *initialization,
    StorageAnalysis &storage);

mlir::LogicalResult emitContractionEpilogue(mlir::OpBuilder &builder,
    ContractionEpilogue &epilogue, mlir::ValueRange offsets,
    mlir::ValueRange sizes);

} // namespace intent::cpu

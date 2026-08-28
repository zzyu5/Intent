#ifndef INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace intent::tilelang {

bool isLegalMmaWarpPartition(int64_t m, int64_t n, int64_t threads);
mlir::LogicalResult legalizeGPUProgram(mlir::ModuleOp module);
mlir::LogicalResult verifyTileLangProgram(mlir::ModuleOp module);

} // namespace intent::tilelang

#endif

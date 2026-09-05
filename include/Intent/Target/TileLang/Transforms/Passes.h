#ifndef INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H

#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace intent::tilelang {

inline constexpr llvm::StringLiteral lowerPredicatedLoadStoreAttr =
    "intent_tilelang.lower_predicated_load_store";

bool isLegalMmaWarpPartition(int64_t m, int64_t n, int64_t threads);
mlir::LogicalResult legalizeGPUProgram(mlir::ModuleOp module,
                                     const gpu::TuningProfiles &profiles);
mlir::LogicalResult verifyTileLangProgram(mlir::ModuleOp module);

} // namespace intent::tilelang

#endif

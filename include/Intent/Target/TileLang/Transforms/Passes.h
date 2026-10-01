#ifndef INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir { class OpPassManager; }
namespace intent::gpu { struct TuningProfileSchema; }

namespace intent::tilelang {

#define GEN_PASS_DECL
#include "Intent/Target/TileLang/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Intent/Target/TileLang/Transforms/Passes.h.inc"

inline constexpr llvm::StringLiteral lowerPredicatedLoadStoreAttr =
    "intent_tilelang.lower_predicated_load_store";

bool isLegalMmaWarpPartition(int64_t m, int64_t n, int64_t threads);
const gpu::TuningProfileSchema &tuningProfileSchema();
void buildTileLangPipeline(mlir::OpPassManager &manager);
void registerTileLangPipelines();
mlir::LogicalResult verifyTileLangProgram(mlir::ModuleOp module);

} // namespace intent::tilelang

#endif

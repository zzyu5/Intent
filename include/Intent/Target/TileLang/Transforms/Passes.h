#ifndef INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::tilelang {

mlir::LogicalResult legalizeGPUProgram(mlir::ModuleOp module);
mlir::LogicalResult verifyTileLangProgram(mlir::ModuleOp module);

} // namespace intent::tilelang

#endif

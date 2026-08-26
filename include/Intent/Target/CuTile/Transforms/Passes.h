#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_PASSES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::cutile {

mlir::LogicalResult legalizeGPUProgram(mlir::ModuleOp module);
mlir::LogicalResult verifyCuTileProgram(mlir::ModuleOp module);

} // namespace intent::cutile

#endif

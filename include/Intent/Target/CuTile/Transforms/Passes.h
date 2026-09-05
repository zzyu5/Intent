#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_PASSES_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_PASSES_H

#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::cutile {

mlir::LogicalResult legalizeGPUProgram(mlir::ModuleOp module,
                                     const gpu::TuningProfiles &profiles);
mlir::LogicalResult verifyCuTileProgram(mlir::ModuleOp module);

} // namespace intent::cutile

#endif

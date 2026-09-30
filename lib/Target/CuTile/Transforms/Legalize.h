#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_LEGALIZE_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_LEGALIZE_H
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "mlir/IR/BuiltinOps.h"
namespace intent::cutile {
// Shared GPU input is verified and its buffers/workspaces are executable.
mlir::LogicalResult prepareProgram(mlir::ModuleOp module);
// Consume all current-GPU facts before committing native replacements.
mlir::LogicalResult formNativeProgram(mlir::ModuleOp module,
                                      const gpu::TuningProfiles &profiles);
// Finish native loop/config/index forms and verify the complete provider IR.
mlir::LogicalResult finalizeProgram(mlir::ModuleOp module);
}
#endif

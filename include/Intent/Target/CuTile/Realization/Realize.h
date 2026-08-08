#ifndef INTENT_TARGET_CUTILE_REALIZATION_REALIZE_H
#define INTENT_TARGET_CUTILE_REALIZATION_REALIZE_H

#include "Intent/Target/CuTile/Config/Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::cutile {

mlir::LogicalResult realizeKernel(mlir::ModuleOp module,
                                  const TargetOptions &target);

} // namespace intent::cutile

#endif

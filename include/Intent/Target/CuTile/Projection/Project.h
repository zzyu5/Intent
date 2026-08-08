#ifndef INTENT_TARGET_CUTILE_PROJECTION_PROJECT_H
#define INTENT_TARGET_CUTILE_PROJECTION_PROJECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::cutile {

mlir::LogicalResult projectSurface(mlir::ModuleOp module);

} // namespace intent::cutile

#endif

#ifndef INTENT_TARGET_TRITON_PROJECTION_PROJECT_H
#define INTENT_TARGET_TRITON_PROJECTION_PROJECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::triton {

mlir::LogicalResult projectSurface(mlir::ModuleOp module);

} // namespace intent::triton

#endif

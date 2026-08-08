#ifndef INTENT_TARGET_TILELANG_PROJECTION_PROJECT_H
#define INTENT_TARGET_TILELANG_PROJECTION_PROJECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::tilelang {

mlir::LogicalResult projectSurface(mlir::ModuleOp module);

} // namespace intent::tilelang

#endif

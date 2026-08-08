#ifndef INTENT_TARGET_TILELANG_REALIZATION_REALIZE_H
#define INTENT_TARGET_TILELANG_REALIZATION_REALIZE_H

#include "Intent/Target/TileLang/Config/Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::tilelang {

mlir::LogicalResult realizeKernel(mlir::ModuleOp module,
                                  const TargetOptions &target);

} // namespace intent::tilelang

#endif

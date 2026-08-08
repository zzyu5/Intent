#ifndef INTENT_TARGET_TRITON_REALIZATION_REALIZE_H
#define INTENT_TARGET_TRITON_REALIZATION_REALIZE_H

#include "Intent/Target/Triton/Config/Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::triton {

mlir::LogicalResult realizeKernel(mlir::ModuleOp module,
                                  const TargetOptions &target);

} // namespace intent::triton

#endif

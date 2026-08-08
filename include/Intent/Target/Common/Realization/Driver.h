#ifndef INTENT_TARGET_COMMON_REALIZATION_DRIVER_H
#define INTENT_TARGET_COMMON_REALIZATION_DRIVER_H

#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::target {

using TargetRealization =
    llvm::function_ref<mlir::LogicalResult(KernelModel &kernel)>;

mlir::LogicalResult realizeTarget(mlir::ModuleOp module,
                                  TargetRealization realize);

} // namespace intent::target

#endif

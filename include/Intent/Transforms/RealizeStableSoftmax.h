#ifndef INTENT_TRANSFORMS_REALIZESTABLESOFTMAX_H
#define INTENT_TRANSFORMS_REALIZESTABLESOFTMAX_H

#include "Intent/Target/Triton/Target.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent {

mlir::LogicalResult
realizeStableSoftmax(mlir::ModuleOp module,
                     const triton::TargetOptions &target);

} // namespace intent

#endif

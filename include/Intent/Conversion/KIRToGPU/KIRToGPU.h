#ifndef INTENT_CONVERSION_KIRTOGPU_KIRTOGPU_H
#define INTENT_CONVERSION_KIRTOGPU_KIRTOGPU_H

#include "mlir/IR/BuiltinOps.h"

namespace intent {

/// The typed boundary to the shared executable GPU Program. Round two stops
/// here deliberately: no legacy Plan or provider path may consume canonical
/// KIR while the new physical program is absent.
mlir::LogicalResult lowerCanonicalKIRToGPU(mlir::ModuleOp module);

} // namespace intent

#endif

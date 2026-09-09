#ifndef INTENT_CONVERSION_KIRTOCPU_KIRTOCPU_H
#define INTENT_CONVERSION_KIRTOCPU_KIRTOCPU_H

#include "mlir/IR/BuiltinOps.h"

namespace intent {
mlir::LogicalResult lowerCanonicalKIRToCPU(mlir::ModuleOp module);
}

#endif

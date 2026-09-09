#ifndef INTENT_TARGET_WEFT_TRANSFORMS_PASSES_H
#define INTENT_TARGET_WEFT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace intent::weft_provider {
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
legalizeProgram(mlir::ModuleOp cpuProgram, std::string &metadata);
}
#endif

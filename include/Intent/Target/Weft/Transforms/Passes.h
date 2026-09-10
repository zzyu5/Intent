#ifndef INTENT_TARGET_WEFT_TRANSFORMS_PASSES_H
#define INTENT_TARGET_WEFT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include <string>

namespace intent::weft_provider {
cpu::ImplementationRegistry implementations();
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
legalizeProgram(mlir::ModuleOp cpuProgram, std::string &metadata);
}
#endif

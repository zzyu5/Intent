#ifndef INTENT_TARGET_MOJO_TRANSFORMS_PASSES_H
#define INTENT_TARGET_MOJO_TRANSFORMS_PASSES_H
#include "mlir/IR/BuiltinOps.h"
namespace intent::mojo {
mlir::LogicalResult legalizeProgram(mlir::ModuleOp module);
}
#endif

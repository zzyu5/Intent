#ifndef INTENT_TARGET_MOJO_TRANSFORMS_PASSES_H
#define INTENT_TARGET_MOJO_TRANSFORMS_PASSES_H
#include "mlir/IR/BuiltinOps.h"
#include "Intent/Dialect/CPU/Transforms/Implementation.h"
namespace intent::mojo {
cpu::ImplementationRegistry implementations();
mlir::LogicalResult materializeRegisterContractions(mlir::func::FuncOp function);
mlir::LogicalResult legalizeProgram(mlir::ModuleOp module);
}
#endif

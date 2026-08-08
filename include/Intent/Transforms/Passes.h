#ifndef INTENT_TRANSFORMS_PASSES_H
#define INTENT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace intent {

mlir::LogicalResult verifyKernelModule(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass> createVerifyKernelIRPass();
void registerIntentPasses();

} // namespace intent

#endif

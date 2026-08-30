#ifndef INTENT_TARGET_TRITON_TRANSFORMS_PASSES_H
#define INTENT_TARGET_TRITON_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace intent::triton {

mlir::LogicalResult legalizeProgramGrid(mlir::ModuleOp module);
mlir::LogicalResult
legalizeGPUProgram(mlir::ModuleOp module,
                   llvm::ArrayRef<llvm::StringRef> completeConfigs = {});
mlir::LogicalResult verifyTritonProgram(mlir::ModuleOp module);

} // namespace intent::triton

#endif

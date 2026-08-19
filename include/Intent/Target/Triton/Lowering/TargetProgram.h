#ifndef INTENT_TARGET_TRITON_LOWERING_TARGETPROGRAM_H
#define INTENT_TARGET_TRITON_LOWERING_TARGETPROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::triton {

mlir::LogicalResult materializeTritonProgram(mlir::ModuleOp module);
mlir::LogicalResult translateTritonProgram(mlir::ModuleOp module,
                                          llvm::raw_ostream &output);

} // namespace intent::triton

#endif

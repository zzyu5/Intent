#ifndef INTENT_TARGET_TRITON_EMISSION_TRANSLATE_H
#define INTENT_TARGET_TRITON_EMISSION_TRANSLATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::triton {

mlir::LogicalResult emitTritonSource(mlir::ModuleOp module,
                                    llvm::raw_ostream &output);

} // namespace intent::triton

#endif

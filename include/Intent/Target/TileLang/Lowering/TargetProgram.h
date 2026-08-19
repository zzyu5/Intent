#ifndef INTENT_TARGET_TILELANG_LOWERING_TARGETPROGRAM_H
#define INTENT_TARGET_TILELANG_LOWERING_TARGETPROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::tilelang {

mlir::LogicalResult materializeTileLangProgram(mlir::ModuleOp module);
mlir::LogicalResult translateTileLangProgram(mlir::ModuleOp module,
                                            llvm::raw_ostream &output);

} // namespace intent::tilelang

#endif

#ifndef INTENT_TARGET_CUTILE_LOWERING_TARGETPROGRAM_H
#define INTENT_TARGET_CUTILE_LOWERING_TARGETPROGRAM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::cutile {

mlir::LogicalResult materializeCuTileProgram(mlir::ModuleOp module);
mlir::LogicalResult translateCuTileProgram(mlir::ModuleOp module,
                                          llvm::raw_ostream &output);

} // namespace intent::cutile

#endif

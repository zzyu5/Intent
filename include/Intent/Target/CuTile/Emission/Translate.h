#ifndef INTENT_TARGET_CUTILE_EMISSION_TRANSLATE_H
#define INTENT_TARGET_CUTILE_EMISSION_TRANSLATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::cutile {

mlir::LogicalResult emitCuTileSource(mlir::ModuleOp module,
                                    llvm::raw_ostream &output);

} // namespace intent::cutile

#endif

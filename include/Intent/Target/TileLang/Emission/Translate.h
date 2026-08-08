#ifndef INTENT_TARGET_TILELANG_EMISSION_TRANSLATE_H
#define INTENT_TARGET_TILELANG_EMISSION_TRANSLATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::tilelang {

mlir::LogicalResult emitTileLangSource(mlir::ModuleOp module,
                                      llvm::raw_ostream &output);

} // namespace intent::tilelang

#endif

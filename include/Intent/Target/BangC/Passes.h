#ifndef INTENT_TARGET_BANGC_PASSES_H
#define INTENT_TARGET_BANGC_PASSES_H
#include "mlir/IR/BuiltinOps.h"
#include <string>
namespace intent::bangc {
mlir::LogicalResult legalizeProgram(mlir::ModuleOp module, llvm::StringRef architecture);
mlir::LogicalResult serializeProgram(mlir::ModuleOp module, std::string &source, std::string &metadata);
}
#endif

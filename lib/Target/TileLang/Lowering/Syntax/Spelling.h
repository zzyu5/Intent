#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace intent::tilelang::lowering::syntax {

std::string dimension(llvm::StringRef symbol);
mlir::FailureOr<std::string> tile(mlir::Operation *operation,
                                  llvm::StringRef role);
mlir::FailureOr<std::string> parameter(mlir::Operation *operation,
                                       llvm::StringRef role);
mlir::FailureOr<llvm::StringRef> pointwise(mlir::Operation *operation,
                                           llvm::StringRef role,
                                           llvm::StringRef materialization);
llvm::StringRef bufferSpace(llvm::StringRef space);
llvm::StringRef reduction(llvm::StringRef role);
llvm::StringRef scan();
llvm::StringRef contraction();
llvm::StringRef scaledContraction();
std::string cast(llvm::StringRef value, llvm::StringRef targetType,
                 bool decodeE8M0, bool resultIsF32);

} // namespace intent::tilelang::lowering::syntax

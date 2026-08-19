#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace intent::cutile::lowering::syntax {

mlir::FailureOr<std::string> tile(mlir::Operation *operation,
                                  llvm::StringRef role);
mlir::FailureOr<std::string> parameter(mlir::Operation *operation,
                                       llvm::StringRef role);
mlir::FailureOr<llvm::StringRef> pointwise(mlir::Operation *operation,
                                           llvm::StringRef role,
                                           llvm::StringRef resultSpace);
llvm::StringRef reduction(llvm::StringRef role);
llvm::StringRef scan(llvm::StringRef role);
llvm::StringRef contraction();
llvm::StringRef scaledContraction();

std::string gather(llvm::StringRef array, llvm::StringRef indices,
                   llvm::StringRef padding, llvm::StringRef mask = {});
std::string cast(llvm::StringRef lowering, llvm::StringRef value,
                 llvm::StringRef targetType);

} // namespace intent::cutile::lowering::syntax

#ifndef INTENT_TARGET_COMMON_TRAVERSAL_OPERATIONREGISTRY_H
#define INTENT_TARGET_COMMON_TRAVERSAL_OPERATIONREGISTRY_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <functional>
#include <string>

namespace intent::target {

using OperationCallback =
    std::function<mlir::LogicalResult(mlir::Operation &)>;

struct OperationHandler {
  OperationCallback enter;
  OperationCallback leave;
};

class OperationHandlerRegistry {
public:
  mlir::LogicalResult add(llvm::StringRef operationName,
                          OperationHandler handler);
  const OperationHandler *lookup(llvm::StringRef operationName) const;
  mlir::LogicalResult dispatch(mlir::Operation &operation,
                               llvm::StringRef stage) const;

private:
  llvm::StringMap<OperationHandler> handlers;
};

mlir::LogicalResult traverseKernel(mlir::func::FuncOp entry,
                                   const OperationHandlerRegistry &registry,
                                   llvm::StringRef stage);

} // namespace intent::target

#endif

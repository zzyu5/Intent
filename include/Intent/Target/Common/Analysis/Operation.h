#ifndef INTENT_TARGET_COMMON_ANALYSIS_OPERATION_H
#define INTENT_TARGET_COMMON_ANALYSIS_OPERATION_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"

namespace intent::target {

inline llvm::StringRef semanticOperationName(mlir::Operation &operation) {
  if (auto source = operation.getAttrOfType<mlir::StringAttr>(
          "intent_plan.source_op"))
    return source.getValue();
  return operation.getName().getStringRef();
}

inline bool isSemanticOperation(mlir::Operation &operation,
                                llvm::StringRef name) {
  return semanticOperationName(operation) == name;
}

inline bool isPhysicalExecutableOperation(mlir::Operation &operation) {
  return operation.getName().getStringRef().starts_with("intent_plan.exec_");
}

} // namespace intent::target

#endif

#ifndef INTENT_TARGET_COMMON_PROJECTION_CAPABILITIES_H
#define INTENT_TARGET_COMMON_PROJECTION_CAPABILITIES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::target {

struct CapabilityProfile {
  llvm::ArrayRef<llvm::StringRef> intentDecided;
  llvm::ArrayRef<llvm::StringRef> delegated;
  llvm::ArrayRef<llvm::StringRef> absent;
};

mlir::LogicalResult verifyCapabilityPartition(
    mlir::Operation *operation, mlir::ArrayAttr intentDecided,
    mlir::ArrayAttr delegated, mlir::ArrayAttr absent);

mlir::LogicalResult verifyCapabilityProfile(
    mlir::Operation *operation, mlir::ArrayAttr intentDecided,
    mlir::ArrayAttr delegated, mlir::ArrayAttr absent,
    const CapabilityProfile &profile);

mlir::LogicalResult verifyParameterMap(mlir::Operation *operation,
                                       mlir::DictionaryAttr parameterMap);

} // namespace intent::target

#endif

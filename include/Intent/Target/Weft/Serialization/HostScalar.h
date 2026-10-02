#ifndef INTENT_TARGET_WEFT_SERIALIZATION_HOSTSCALAR_H
#define INTENT_TARGET_WEFT_SERIALIZATION_HOSTSCALAR_H
#include "Intent/Serialization/Scalar.h"

namespace intent::weft_provider {
std::optional<CScalarType> hostScalarType(mlir::Type type);
mlir::FailureOr<std::string> emitHostScalar(mlir::Operation *operation,
                                          llvm::ArrayRef<std::string> operands);
mlir::LogicalResult verifyHostScalar(mlir::Operation *operation);
} // namespace intent::weft_provider
#endif

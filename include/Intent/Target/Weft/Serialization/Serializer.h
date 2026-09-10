#ifndef INTENT_TARGET_WEFT_SERIALIZATION_SERIALIZER_H
#define INTENT_TARGET_WEFT_SERIALIZATION_SERIALIZER_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace intent::weft_provider {
mlir::LogicalResult serializeProgram(mlir::ModuleOp program, std::string &source);
mlir::LogicalResult serializeHostProgram(mlir::ModuleOp program, std::string &source);
}
#endif

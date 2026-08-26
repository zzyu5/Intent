#ifndef INTENT_TARGET_CUTILE_SERIALIZATION_SERIALIZER_H
#define INTENT_TARGET_CUTILE_SERIALIZATION_SERIALIZER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace intent::cutile {

mlir::LogicalResult serializeProgram(mlir::ModuleOp module,
                                     std::string &source);

} // namespace intent::cutile

#endif

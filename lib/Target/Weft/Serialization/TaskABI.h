#pragma once

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/JSON.h"

namespace intent::weft_provider {
mlir::FailureOr<llvm::json::Object> serializeTaskABI(::weft::kernel::KernelOp kernel);
}

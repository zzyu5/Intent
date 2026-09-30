#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

class ImplementationRegistry;

// Region instantiation exposes concrete storage relationships. Fold only
// input descriptors here, preserving each selected matrix implementation.
mlir::LogicalResult foldContractionInputs(
    mlir::func::FuncOp function, const ImplementationRegistry &implementations);

} // namespace intent::cpu

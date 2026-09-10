#ifndef INTENT_TARGET_WEFT_TRANSFORMS_QUANTIZATION_H
#define INTENT_TARGET_WEFT_TRANSFORMS_QUANTIZATION_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

namespace intent::weft_provider {
::weft::kernel::EncodingType quantEncoding(mlir::OpBuilder &builder, intent::QuantFormat format);
void declareQuantEncodings(mlir::ModuleOp module);
mlir::LogicalResult expandQuantize(mlir::OpBuilder &builder, cpu::QuantizeOp operation,
    mlir::Value input, mlir::Value output, int64_t &nextAxis);
mlir::FailureOr<mlir::Value> expandQuantizedDot(mlir::OpBuilder &builder,
    cpu::QuantizedDotOp operation, mlir::Value lhs, mlir::Value rhs, int64_t &nextAxis);
}
#endif

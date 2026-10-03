#ifndef INTENT_CPU_TRANSFORMS_INPUTWINDOWS_H
#define INTENT_CPU_TRANSFORMS_INPUTWINDOWS_H

#include "Intent/Dialect/CPU/Transforms/Implementation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace intent::cpu {

struct ConsumerWindow {
  mlir::memref::SubViewOp view;
  unsigned axis;
  bool transposed;
};

std::optional<ConsumerWindow> consumerWindow(
    mlir::Value source, const InputRequirement &requirement,
    mlir::Operation *consumer);
bool hasIndependentWindowCoordinates(mlir::memref::SubViewOp window,
                                     mlir::Operation *loop,
                                     mlir::Value groupCoordinate = {});

} // namespace intent::cpu
#endif

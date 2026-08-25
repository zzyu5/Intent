#ifndef INTENT_CONVERSION_KIRTOGPU_KIRTOGPU_H
#define INTENT_CONVERSION_KIRTOGPU_KIRTOGPU_H

#include "mlir/IR/BuiltinOps.h"

namespace intent {

struct GPUCapabilities {
  int64_t computeUnits;
  int64_t sharedMemoryPerUnit;
  int64_t registersPerUnit;
  bool matrixUnits;
  bool dynamicVectorWidth;
};

/// Consumes canonical KIR and replaces it with one complete conservative
/// provider-neutral executable GPU program.  The result never references KIR.
mlir::LogicalResult lowerCanonicalKIRToGPU(mlir::ModuleOp module,
                                           const GPUCapabilities &capabilities);

} // namespace intent

#endif

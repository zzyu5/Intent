#ifndef INTENT_LIB_TARGET_TRITON_EMISSION_EMITTER_H
#define INTENT_LIB_TARGET_TRITON_EMISSION_EMITTER_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace intent::triton {

mlir::LogicalResult emitRealizedKernelSource(
    mlir::ModuleOp module, intent::plan::RealizationOp realization,
    llvm::raw_ostream &output);

} // namespace intent::triton

#endif

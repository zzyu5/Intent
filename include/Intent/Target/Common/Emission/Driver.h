#ifndef INTENT_TARGET_COMMON_EMISSION_DRIVER_H
#define INTENT_TARGET_COMMON_EMISSION_DRIVER_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::target {

using KernelSourceEmission = mlir::LogicalResult (*)(
    KernelModel, intent::plan::ProgramOp, intent::plan::SearchSpaceOp,
    llvm::raw_ostream &);

struct EmissionTarget {
  llvm::StringRef displayName;
  KernelSourceEmission emitKernelSource;
};

mlir::LogicalResult emitTargetSource(mlir::ModuleOp module,
                                     llvm::raw_ostream &output,
                                     const EmissionTarget &target);

} // namespace intent::target

#endif

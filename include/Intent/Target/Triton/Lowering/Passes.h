#ifndef INTENT_TARGET_TRITON_LOWERING_PASSES_H
#define INTENT_TARGET_TRITON_LOWERING_PASSES_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::triton::lowering {

inline constexpr llvm::StringLiteral rowLaunchAttr =
    "intent_plan.triton.row_launch";

void addProviderPasses(mlir::PassManager &manager);
mlir::LogicalResult verifyProviderProgram(
    const intent::target::KernelModel &kernel,
    intent::plan::ProgramOp program,
    intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::triton::lowering

#endif

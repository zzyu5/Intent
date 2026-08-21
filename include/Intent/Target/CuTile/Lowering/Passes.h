#ifndef INTENT_TARGET_CUTILE_LOWERING_PASSES_H
#define INTENT_TARGET_CUTILE_LOWERING_PASSES_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::cutile::lowering {

inline constexpr llvm::StringLiteral accessAttr =
    "intent_plan.cutile.access";
inline constexpr llvm::StringLiteral boundsAttr =
    "intent_plan.cutile.bounds";
inline constexpr llvm::StringLiteral rowOccupancyAttr =
    "intent_plan.cutile.row_occupancy";
inline constexpr llvm::StringLiteral gatherSpellingRole =
    "cutile_gather_spelling";

void addProviderPasses(mlir::PassManager &manager);
mlir::LogicalResult verifyProviderProgram(
    const intent::target::KernelModel &kernel,
    intent::plan::ProgramOp program,
    intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::cutile::lowering

#endif

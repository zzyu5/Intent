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
inline constexpr llvm::StringLiteral transferAttr =
    "intent_plan.cutile.transfer";
inline constexpr llvm::StringLiteral loadShapeNodeAttr =
    "intent_plan.cutile.load_shape_node";
inline constexpr llvm::StringLiteral rowOccupancyAttr =
    "intent_plan.cutile.row_occupancy";
inline constexpr llvm::StringLiteral contractLoweringAttr =
    "intent_plan.cutile.contract_lowering";
inline constexpr llvm::StringLiteral contractOrientationAttr =
    "intent_plan.cutile.contract_orientation";
inline constexpr llvm::StringLiteral contractBatchedAttr =
    "intent_plan.cutile.contract_batched";
inline constexpr llvm::StringLiteral contractMmaFormAttr =
    "intent_plan.cutile.contract_mma_form";
inline constexpr llvm::StringLiteral scaledContractLayoutAttr =
    "intent_plan.cutile.scaled_contract_layout";
inline constexpr llvm::StringLiteral reductionLoweringAttr =
    "intent_plan.cutile.reduction_lowering";
inline constexpr llvm::StringLiteral reductionAxisAttr =
    "intent_plan.cutile.reduction_axis";
inline constexpr llvm::StringLiteral scanLoweringAttr =
    "intent_plan.cutile.scan_lowering";
inline constexpr llvm::StringLiteral pointwiseLoweringAttr =
    "intent_plan.cutile.pointwise_lowering";
inline constexpr llvm::StringLiteral gatherFormAttr =
    "intent_plan.cutile.gather_form";
inline constexpr llvm::StringLiteral streamTileAttr =
    "intent_plan.cutile.stream_tile";
inline constexpr llvm::StringLiteral raggedRouteAttr =
    "intent_plan.cutile.ragged_route";
inline constexpr llvm::StringLiteral gatherSpellingRole =
    "gather_spelling";

void addProviderPasses(mlir::PassManager &manager);
mlir::LogicalResult verifyProviderProgram(
    const intent::target::KernelModel &kernel,
    intent::plan::ProgramOp program,
    intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::cutile::lowering

#endif

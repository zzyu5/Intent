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
inline constexpr llvm::StringLiteral contractLoweringAttr =
    "intent_plan.triton.contract_lowering";
inline constexpr llvm::StringLiteral contractOrientationAttr =
    "intent_plan.triton.contract_orientation";
inline constexpr llvm::StringLiteral contractBatchedAttr =
    "intent_plan.triton.contract_batched";
inline constexpr llvm::StringLiteral scaledContractLayoutAttr =
    "intent_plan.triton.scaled_contract_layout";
inline constexpr llvm::StringLiteral reductionLoweringAttr =
    "intent_plan.triton.reduction_lowering";
inline constexpr llvm::StringLiteral reductionAxisAttr =
    "intent_plan.triton.reduction_axis";
inline constexpr llvm::StringLiteral scanLoweringAttr =
    "intent_plan.triton.scan_lowering";
inline constexpr llvm::StringLiteral pointwiseLoweringAttr =
    "intent_plan.triton.pointwise_lowering";
inline constexpr llvm::StringLiteral gatherFormAttr =
    "intent_plan.triton.gather_form";
inline constexpr llvm::StringLiteral transferAccessAttr =
    "intent_plan.triton.transfer_access";
inline constexpr llvm::StringLiteral transferFormAttr =
    "intent_plan.triton.transfer_form";
inline constexpr llvm::StringLiteral streamTileAttr =
    "intent_plan.triton.stream_tile";
inline constexpr llvm::StringLiteral streamFormAttr =
    "intent_plan.triton.stream_form";
inline constexpr llvm::StringLiteral streamBoundaryAxisAttr =
    "intent_plan.triton.stream_boundary_axis";
inline constexpr llvm::StringLiteral streamNeutralMasksAttr =
    "intent_plan.triton.stream_neutral_masks";
inline constexpr llvm::StringLiteral raggedRouteAttr =
    "intent_plan.triton.ragged_route";
void addProviderPasses(mlir::PassManager &manager);
mlir::LogicalResult verifyProviderProgram(
    const intent::target::KernelModel &kernel,
    intent::plan::ProgramOp program,
    intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::triton::lowering

#endif

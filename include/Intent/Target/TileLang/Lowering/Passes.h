#ifndef INTENT_TARGET_TILELANG_LOWERING_PASSES_H
#define INTENT_TARGET_TILELANG_LOWERING_PASSES_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::tilelang::lowering {

inline constexpr llvm::StringLiteral accessAttr =
    "intent_plan.tilelang.access";
inline constexpr llvm::StringLiteral transferAttr =
    "intent_plan.tilelang.transfer";
inline constexpr llvm::StringLiteral boundsAttr =
    "intent_plan.tilelang.bounds";
inline constexpr llvm::StringLiteral deferredAttr =
    "intent_plan.tilelang.deferred";
inline constexpr llvm::StringLiteral pointwiseFormAttr =
    "intent_plan.tilelang.pointwise_form";
inline constexpr llvm::StringLiteral rowLaunchAttr =
    "intent_plan.tilelang.row_launch";
inline constexpr llvm::StringLiteral equalProgramTilesAttr =
    "intent_plan.tilelang.equal_program_tiles";
inline constexpr llvm::StringLiteral isolateLhsAttr =
    "intent_plan.tilelang.isolate_lhs";
inline constexpr llvm::StringLiteral isolateRhsAttr =
    "intent_plan.tilelang.isolate_rhs";
inline constexpr llvm::StringLiteral gemmWarpPolicyAttr =
    "intent_plan.tilelang.gemm_warp_policy";

void addProviderPasses(mlir::PassManager &manager);
mlir::LogicalResult verifyProviderProgram(
    const intent::target::KernelModel &kernel,
    intent::plan::ProgramOp program,
    intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::tilelang::lowering

#endif

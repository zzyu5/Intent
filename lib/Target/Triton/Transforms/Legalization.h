#ifndef INTENT_TARGET_TRITON_TRANSFORMS_LEGALIZATION_H
#define INTENT_TARGET_TRITON_TRANSFORMS_LEGALIZATION_H

#include "Configurations.h"
#include "Intent/Target/Triton/Transforms/Passes.h"

namespace intent::triton {

// Complete provider phases consume only current IR; local form candidates are
// closed within one phase.
mlir::LogicalResult prepareTritonMemory(mlir::ModuleOp module);
mlir::LogicalResult formTritonProgram(mlir::ModuleOp module);
mlir::LogicalResult finalizeTritonProgram(mlir::ModuleOp module);

namespace detail {
inline constexpr llvm::StringLiteral contractFormAttr =
    "intent_gpu.triton.contract_form";
inline constexpr llvm::StringLiteral legalizedAttr = "intent_gpu.triton.legalized";

// AccessForms owns source representations of already-decided GPU accesses.
mlir::FailureOr<TensorDescriptorChoiceOp> materializeTensorDescriptorForms(
    mlir::func::FuncOp kernel, llvm::ArrayRef<TritonLocalOptions> localOptions);
mlir::LogicalResult materializeBlockPointerForms(mlir::func::FuncOp kernel);
void orientPointerLoads(mlir::func::FuncOp kernel);

// Collectives owns local gather/scatter and native callback legalization.
void foldIntegerScanTails(mlir::func::FuncOp kernel);
mlir::LogicalResult materializeOversizedGathers(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeLargeScalarGathers(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeMaskedGather(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeExpandingGathers(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeScatterAdd(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeSplitGatherPairs(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeCollectiveCallbacks(mlir::func::FuncOp kernel);

// Supply owns ordered access dependencies and provider load-loop policy.
bool hasOrderedViewDependencies(mlir::func::FuncOp kernel);
mlir::LogicalResult legalizeOrderedViewDependencies(mlir::func::FuncOp kernel);
llvm::SmallVector<mlir::scf::ForOp> findLoadPipelineLoops(mlir::func::FuncOp kernel);
void selectOrderedLoadUnrolling(mlir::func::FuncOp kernel);

// Values owns provider compute forms and value-graph representation.
gpu::ConfigurationRequirementAttr
contractionExpansionRequirement(gpu::ContractOp contract);
void selectContractForms(mlir::func::FuncOp kernel);
void canonicalizeBroadcastProjections(mlir::func::FuncOp kernel);
void sinkSelectProducers(mlir::func::FuncOp kernel);

} // namespace detail
} // namespace intent::triton

#endif

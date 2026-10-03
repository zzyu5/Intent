#ifndef INTENT_DIALECT_GPU_TRANSFORMS_RESOURCES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_RESOURCES_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::gpu {

enum class ReductionRequirementScope { AllCandidates, InvocationDependent };

// Collect once from the current program, then use the same requirements for
// candidate filtering and invocation-dependent specialization.
llvm::SmallVector<ConfigurationRequirementAttr> collectReductionRequirements(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::ValueRange> sourceGroups,
    ReductionRequirementScope scope);

mlir::FailureOr<llvm::SmallVector<mlir::DictionaryAttr>> filterConfigurationRequirements(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::DictionaryAttr> rows,
    llvm::ArrayRef<ConfigurationRequirementAttr> requirements);

// Verify the final current-IR condition set and rows without changing either.
// Conditions have set semantics; candidate row order remains significant.
mlir::LogicalResult verifyConfigurationRequirements(
    mlir::func::FuncOp kernel,
    llvm::ArrayRef<ConfigurationRequirementAttr> expected);

} // namespace intent::gpu

#endif

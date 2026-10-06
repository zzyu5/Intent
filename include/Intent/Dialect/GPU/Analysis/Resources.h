#ifndef INTENT_DIALECT_GPU_ANALYSIS_RESOURCES_H
#define INTENT_DIALECT_GPU_ANALYSIS_RESOURCES_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu {

enum class ReductionRequirementScope { AllCandidates, InvocationDependent };

// The same tunable traversal domain is budgeted during row construction and
// after deferred coverage has been bound for an invocation.
bool isPointwiseTraversalParameter(ParameterAttr parameter);

// Collect from current IR for candidate filtering and invocation specialization.
llvm::SmallVector<ConfigurationRequirementAttr> collectReductionRequirements(
    mlir::func::FuncOp kernel, llvm::ArrayRef<mlir::Operation *> reductions,
    ReductionRequirementScope scope);

// Verify the current condition set and rows without changing either. Conditions
// have set semantics; candidate row order remains significant.
mlir::LogicalResult verifyConfigurationRequirements(
    mlir::func::FuncOp kernel,
    llvm::ArrayRef<ConfigurationRequirementAttr> expected);

PhysicalExprAttr fragmentElementCount(FragmentType fragment);
PhysicalExprAttr fragmentRegisterFootprint(FragmentType fragment);

enum class FragmentFootprintScope { PhysicalShape, FullScalarSeedCapacity };

// Minimum nominal 32-bit words over the current extent domains, saturated at
// limit + 1. This is neither live-register allocation nor a machine limit.
// Only constant/direct-parameter extents establish a domain minimum; arbitrary
// expressions and unproved scalar-seed capacities remain unknown.
std::optional<int64_t> minimumFragmentRegisterFootprint(
    mlir::func::FuncOp kernel, mlir::Value value, int64_t limit,
    FragmentFootprintScope scope);

// A structural estimate, not a machine-register allocation or occupancy model.
// The provider still owns layout reuse, scheduling, spills and final legality.
enum class FootprintBound { Unknown, Within, Exceeds, Invalid };
enum class RequirementStatus { Unknown, Satisfied, Violated, Invalid, Inactive };
struct RequirementEvaluation {
  RequirementStatus status;
  std::optional<int64_t> usage;
  std::optional<int64_t> limit;
};

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf,
    llvm::function_ref<std::optional<int64_t>(ParameterRefAttr)>
        resolveActivation = {});
RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, mlir::DictionaryAttr bindings);
// A missing deferred binding remains unknown unless its declared domain proves
// that a positive additive/multiplicative nominal footprint already exceeds the
// budget. A domain lower bound never establishes satisfaction.
RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, mlir::DictionaryAttr bindings,
    mlir::func::FuncOp kernel);
// Unknown includes expressions the checked evaluator cannot establish (missing
// leaves or arithmetic failure); Invalid denotes a proven nonpositive extent.
FootprintBound checkFragmentFootprint(
    FragmentType fragment, int64_t limit, int64_t wordsPerElement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf);

// Read once at a stable program boundary; discard after IR/type mutations.
// Shape-only views participate in valueTypes (surface legality), but do not
// introduce independent materialized payloads for shared resource estimates.
class FragmentResourceAnalysis {
public:
  explicit FragmentResourceAnalysis(mlir::func::FuncOp kernel);
  llvm::ArrayRef<FragmentType> valueTypes() const { return values; }
  llvm::ArrayRef<FragmentType> materializedTypesUsing(mlir::StringAttr name) const;
  llvm::ArrayRef<ConfigurationRequirementAttr> collectiveRequirements() const {
    return collectives;
  }

private:
  llvm::SmallVector<FragmentType> values;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<FragmentType>> payloads;
  llvm::SmallVector<ConfigurationRequirementAttr> collectives;
};

llvm::SmallVector<ConfigurationRequirementAttr> collectPointwiseRequirements(
    mlir::func::FuncOp kernel, const FragmentResourceAnalysis &resources);

} // namespace intent::gpu

#endif

#ifndef INTENT_DIALECT_GPU_ANALYSIS_RESOURCES_H
#define INTENT_DIALECT_GPU_ANALYSIS_RESOURCES_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu {

PhysicalExprAttr fragmentElementCount(FragmentType fragment);
PhysicalExprAttr fragmentRegisterFootprint(FragmentType fragment);
PhysicalExprAttr reductionRegisterFootprint(mlir::ValueRange sources,
                                          mlir::func::FuncOp kernel);

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
struct RequirementEvaluation {
  FootprintBound bound;
  std::optional<int64_t> usage;
  std::optional<int64_t> limit;
};

RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf);
RequirementEvaluation evaluateConfigurationRequirement(
    ConfigurationRequirementAttr requirement, mlir::DictionaryAttr bindings);
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

private:
  llvm::SmallVector<FragmentType> values;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<FragmentType>> payloads;
};

} // namespace intent::gpu

#endif

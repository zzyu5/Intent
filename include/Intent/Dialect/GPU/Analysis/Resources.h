#ifndef INTENT_DIALECT_GPU_ANALYSIS_RESOURCES_H
#define INTENT_DIALECT_GPU_ANALYSIS_RESOURCES_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu {

PhysicalExprAttr fragmentRegisterFootprint(FragmentType fragment);
PhysicalExprAttr reductionRegisterFootprint(mlir::ValueRange sources,
                                          mlir::func::FuncOp kernel);

// A structural estimate, not a machine-register allocation or occupancy model.
// The provider still owns layout reuse, scheduling, spills and final legality.
enum class FootprintBound { Unknown, Within, Exceeds, Invalid };
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

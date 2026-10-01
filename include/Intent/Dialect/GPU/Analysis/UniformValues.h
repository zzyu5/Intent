#ifndef INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Analysis/UniformValues.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu {
class PhysicalExprAttr;
// Resolve only symbolic leaves. Arithmetic has one checked interpretation for
// shared analysis and provider candidate instantiation; an absent binding,
// division by zero or unrepresentable result is not a proven constant. A caller
// with a narrower expression contract may reject a composed operation after its
// operands are evaluated, without supplying another arithmetic implementation.
std::optional<int64_t> evaluatePhysicalExpression(
    PhysicalExprAttr expression,
    llvm::function_ref<std::optional<int64_t>(PhysicalExprAttr)> resolveLeaf,
    llvm::function_ref<bool(PhysicalExprAttr, llvm::ArrayRef<int64_t>)>
        supportsOperation = {});
std::optional<int64_t> constantPhysicalExpression(PhysicalExprAttr expression);
UniformExpression describeUniformValue(mlir::Value value);
mlir::Type uniformElementType(mlir::Type type);
}
#endif

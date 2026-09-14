#ifndef INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Analysis/UniformValues.h"

namespace intent::gpu {
class PhysicalExprAttr;
std::optional<int64_t> constantPhysicalExpression(PhysicalExprAttr expression);
UniformExpression describeUniformValue(mlir::Value value);
mlir::Type uniformElementType(mlir::Type type);
}
#endif

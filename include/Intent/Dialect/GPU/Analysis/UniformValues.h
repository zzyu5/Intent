#ifndef INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H
#define INTENT_DIALECT_GPU_ANALYSIS_UNIFORMVALUES_H

#include "Intent/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"

namespace intent::gpu {
UniformExpression describeUniformValue(mlir::Value value);
mlir::Type uniformElementType(mlir::Type type);
}
#endif

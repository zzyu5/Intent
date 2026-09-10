#ifndef INTENT_DIALECT_GPU_IR_GPUATTRS_H
#define INTENT_DIALECT_GPU_IR_GPUATTRS_H

#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Intent/Dialect/GPU/IR/GPUAttrs.h.inc"

namespace intent::gpu {
bool isCompileTimePhysicalExpr(PhysicalExprAttr expression);
}

#endif

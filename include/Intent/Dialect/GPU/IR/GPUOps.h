#ifndef INTENT_DIALECT_GPU_IR_GPUOPS_H
#define INTENT_DIALECT_GPU_IR_GPUOPS_H

#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/Intent/IR/IntentAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Intent/Dialect/GPU/IR/GPUOps.h.inc"

namespace intent::gpu {

mlir::FailureOr<mlir::ArrayAttr>
inferReshapeReassociation(FragmentType source, FragmentType result,
                          unsigned sourcePrefix = 0,
                          unsigned resultPrefix = 0);

} // namespace intent::gpu

#endif

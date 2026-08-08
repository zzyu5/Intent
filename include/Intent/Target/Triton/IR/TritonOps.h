#ifndef INTENT_TARGET_TRITON_IR_TRITONOPS_H
#define INTENT_TARGET_TRITON_IR_TRITONOPS_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Projection/Capabilities.h"
#include "Intent/Target/Triton/IR/TritonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Intent/Target/Triton/IR/TritonOps.h.inc"

namespace intent::triton::plan {

const intent::target::CapabilityProfile &getCapabilityProfile();

mlir::LogicalResult
verifyTritonRealization(intent::plan::RealizationOp realization);

mlir::LogicalResult
verifyTritonSearchSpace(intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::triton::plan

#endif

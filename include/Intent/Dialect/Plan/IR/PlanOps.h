#ifndef INTENT_DIALECT_PLAN_IR_PLANOPS_H
#define INTENT_DIALECT_PLAN_IR_PLANOPS_H

#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Intent/Dialect/Plan/IR/PlanOps.h.inc"

namespace intent::plan {

mlir::LogicalResult verifyGpuRealization(RealizationOp realization);
mlir::LogicalResult verifyGpuSearchSpace(SearchSpaceOp searchSpace);

} // namespace intent::plan

#endif

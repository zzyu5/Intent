#ifndef INTENT_DIALECT_INTENT_IR_INTENTOPS_H
#define INTENT_DIALECT_INTENT_IR_INTENTOPS_H

#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Intent/Dialect/Intent/IR/IntentOps.h.inc"

#endif

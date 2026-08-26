#ifndef INTENT_TARGET_CUTILE_IR_CUTILEOPS_H
#define INTENT_TARGET_CUTILE_IR_CUTILEOPS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Intent/Target/CuTile/IR/CuTileOps.h.inc"

#endif

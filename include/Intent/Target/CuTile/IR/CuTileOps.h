#ifndef INTENT_TARGET_CUTILE_IR_CUTILEOPS_H
#define INTENT_TARGET_CUTILE_IR_CUTILEOPS_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Projection/Capabilities.h"
#include "Intent/Target/CuTile/IR/CuTileDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Intent/Target/CuTile/IR/CuTileOps.h.inc"

namespace intent::cutile::plan {

const intent::target::CapabilityProfile &getCapabilityProfile();

mlir::LogicalResult
verifyCuTileRealization(intent::plan::RealizationOp realization);

mlir::LogicalResult
verifyCuTileSearchSpace(intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::cutile::plan

#endif

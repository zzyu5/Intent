#ifndef INTENT_TARGET_TILELANG_IR_TILELANGOPS_H
#define INTENT_TARGET_TILELANG_IR_TILELANGOPS_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Projection/Capabilities.h"
#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Intent/Target/TileLang/IR/TileLangOps.h.inc"

namespace intent::tilelang::plan {

const intent::target::CapabilityProfile &getCapabilityProfile();

mlir::LogicalResult
verifyTileLangRealization(intent::plan::RealizationOp realization);

mlir::LogicalResult
verifyTileLangSearchSpace(intent::plan::SearchSpaceOp searchSpace);

} // namespace intent::tilelang::plan

#endif

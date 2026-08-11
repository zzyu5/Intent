#ifndef INTENT_TARGET_COMMON_ANALYSIS_STRUCTUREDCONTROL_H
#define INTENT_TARGET_COMMON_ANALYSIS_STRUCTUREDCONTROL_H

#include "mlir/IR/Operation.h"

namespace intent::target {

inline mlir::Operation *whileConditionOwner(mlir::Operation &operation) {
  mlir::Operation *owner = operation.getParentOp();
  if (!owner || owner->getName().getStringRef() != "intent.while" ||
      owner->getNumRegions() != 2 ||
      operation.getParentRegion() != &owner->getRegion(0))
    return nullptr;
  return owner;
}

} // namespace intent::target

#endif

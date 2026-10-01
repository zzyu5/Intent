#include "Intent/Dialect/GPU/IR/AccessOpInterface.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

#include "Intent/Dialect/GPU/IR/AccessOpInterface.cpp.inc"

namespace intent::gpu {

bool AccessOpInterface::writesMemory() {
  switch (getAccessKind()) {
  case AccessKind::Store:
  case AccessKind::ScatterReduce:
  case AccessKind::AtomicStore:
  case AccessKind::AtomicRMW:
  case AccessKind::AtomicCompareExchange: return true;
  case AccessKind::Load:
  case AccessKind::Gather:
  case AccessKind::AtomicLoad: return false;
  }
  llvm_unreachable("unknown physical access kind");
}

LogicalResult AccessOpInterface::updateAccessOperands(
    ValueRange coordinates, ValueRange payloads, Value valid, Value fill) {
  if (coordinates.size() != getAccessSourceAxes().size())
    return emitOpError("access update must preserve its explicit coordinate-axis mapping");
  if (payloads.size() != getAccessPayloads().size())
    return emitOpError("access update must preserve its payload arity");
  bool hasFillGroup = getAccessFillMutable().has_value();
  if ((hasFillGroup && bool(valid) != bool(fill)) || (!hasFillGroup && fill))
    return emitOpError("access update violates the operation's validity/fill contract");
  SmallVector<Value> copiedCoordinates(coordinates), copiedPayloads(payloads);
  getAccessCoordinatesMutable().assign(copiedCoordinates);
  getAccessPayloadsMutable().assign(copiedPayloads);
  getAccessValidityMutable().assign(valid ? ValueRange{valid} : ValueRange{});
  if (hasFillGroup)
    getAccessFillMutable()->assign(fill ? ValueRange{fill} : ValueRange{});
  return success();
}

} // namespace intent::gpu

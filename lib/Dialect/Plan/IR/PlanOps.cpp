#include "Intent/Dialect/Plan/IR/PlanOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace intent::plan;

namespace {

LogicalResult verifyEnvelope(Operation *operation, FlatSymbolRefAttr entry,
                             StringRef target, Region &body) {
  if (target.empty())
    return operation->emitOpError("requires a non-empty target name");
  auto module = operation->getParentOfType<ModuleOp>();
  if (!module || operation->getParentOp() != module)
    return operation->emitOpError("must be nested directly in an MLIR module");
  auto function = module.lookupSymbol<func::FuncOp>(entry.getValue());
  auto kind = function ? function->getAttrOfType<StringAttr>("intent.kind")
                       : StringAttr();
  if (!function || !kind || kind.getValue() != "kernel")
    return operation->emitOpError("entry must reference an Intent kernel function");
  if (!llvm::hasSingleElement(body))
    return operation->emitOpError("requires exactly one body block");
  for (Operation &nested : body.front()) {
    if (isa<YieldOp>(nested))
      continue;
    if (nested.getName().getDialectNamespace() == "intent_plan")
      return nested.emitOpError(
          "target-neutral Plan operations cannot encode concrete choices");
  }
  return success();
}

} // namespace

LogicalResult RealizationOp::verify() {
  return verifyEnvelope(*this, getEntryAttr(), getTarget(), getBody());
}

LogicalResult SearchSpaceOp::verify() {
  return verifyEnvelope(*this, getEntryAttr(), getTarget(), getBody());
}

#define GET_OP_CLASSES
#include "Intent/Dialect/Plan/IR/PlanOps.cpp.inc"

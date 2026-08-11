#ifndef INTENT_TARGET_COMMON_ANALYSIS_RECORD_H
#define INTENT_TARGET_COMMON_ANALYSIS_RECORD_H

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include <optional>

namespace intent::target {

inline mlir::FailureOr<mlir::Value>
resolveRecordField(mlir::Operation &extract) {
  auto key = extract.getAttrOfType<mlir::StringAttr>("intent.key");
  mlir::Operation *record =
      extract.getNumOperands() == 1
          ? extract.getOperand(0).getDefiningOp()
          : nullptr;
  auto fields = record ? record->getAttrOfType<mlir::ArrayAttr>("intent.fields")
                       : mlir::ArrayAttr();
  if (!key || !record ||
      record->getName().getStringRef() != "intent.make_record" ||
      record->getNumResults() != 1 || !fields ||
      fields.size() != record->getNumOperands() || extract.getNumResults() != 1)
    return extract.emitOpError("has no canonical record-field source");
  std::optional<unsigned> position;
  for (auto [index, attribute] : llvm::enumerate(fields)) {
    auto field = mlir::dyn_cast<mlir::StringAttr>(attribute);
    if (!field)
      return extract.emitOpError("references a record with a non-string field");
    if (field.getValue() != key.getValue())
      continue;
    if (position)
      return extract.emitOpError("references an ambiguous record field");
    position = index;
  }
  if (!position)
    return extract.emitOpError("references a missing record field");
  mlir::Value value = record->getOperand(*position);
  if (value.getType() != extract.getResult(0).getType())
    return extract.emitOpError("record field type does not match its extraction");
  return value;
}

} // namespace intent::target

#endif

#include "Intent/Target/Common/Analysis/IndexRelation.h"

using namespace mlir;

namespace intent::target {

FailureOr<llvm::SmallVector<IndexTerm>>
parseIndexRelation(Operation &operation) {
  auto relation = operation.getAttrOfType<ArrayAttr>("intent.index");
  if (!relation) {
    operation.emitOpError("requires an intent.index relation for target lowering");
    return failure();
  }
  llvm::SmallVector<IndexTerm> terms;
  for (Attribute attribute : relation) {
    auto dictionary = dyn_cast<DictionaryAttr>(attribute);
    auto kind = dictionary ? dictionary.getAs<StringAttr>("kind") : StringAttr();
    auto operands =
        dictionary ? dictionary.getAs<ArrayAttr>("operands") : ArrayAttr();
    auto staticValues =
        dictionary ? dictionary.getAs<ArrayAttr>("static") : ArrayAttr();
    if (!dictionary || !kind || !operands || !staticValues) {
      operation.emitOpError("contains a malformed intent.index term");
      return failure();
    }
    IndexTerm term{kind.getValue().str(), {}, {}};
    for (Attribute operand : operands) {
      if (isa<UnitAttr>(operand)) {
        term.operands.push_back(std::nullopt);
        continue;
      }
      auto position = dyn_cast<IntegerAttr>(operand);
      if (!position || position.getInt() < 0 ||
          static_cast<unsigned>(position.getInt()) >= operation.getNumOperands()) {
        operation.emitOpError("contains an invalid intent.index operand position");
        return failure();
      }
      term.operands.push_back(static_cast<unsigned>(position.getInt()));
    }
    for (Attribute value : staticValues) {
      if (isa<UnitAttr>(value)) {
        term.staticValues.push_back(std::nullopt);
        continue;
      }
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer) {
        operation.emitOpError("contains a non-integer static index value");
        return failure();
      }
      term.staticValues.push_back(integer.getInt());
    }
    terms.push_back(std::move(term));
  }
  return terms;
}

} // namespace intent::target

#ifndef INTENT_TARGET_COMMON_LOWERING_LITERAL_H
#define INTENT_TARGET_COMMON_LOWERING_LITERAL_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <string>

namespace intent::target::lowering {

inline std::string spellFiniteFloatLiteral(mlir::FloatAttr value) {
  llvm::SmallString<32> spelling;
  value.getValue().toString(spelling);
  llvm::StringRef text(spelling);
  if (!text.contains('.') && !text.contains('E') && !text.contains('e'))
    spelling.append(".0");
  return spelling.str().str();
}

} // namespace intent::target::lowering

#endif

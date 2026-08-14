#ifndef INTENT_TARGET_COMMON_EMISSION_LITERAL_H
#define INTENT_TARGET_COMMON_EMISSION_LITERAL_H

#include "llvm/ADT/SmallString.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <string>

namespace intent::target::emission {

inline std::string spellFiniteFloatLiteral(mlir::FloatAttr value) {
  llvm::SmallString<32> spelling;
  value.getValue().toString(spelling);
  return spelling.str().str();
}

} // namespace intent::target::emission

#endif

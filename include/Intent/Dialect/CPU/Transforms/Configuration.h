#ifndef INTENT_DIALECT_CPU_TRANSFORMS_CONFIGURATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_CONFIGURATION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace intent::cpu {

// One profile row. Shared fields bind the task/block program; local fields are
// validated by the provider's family schema and consumed by implementations.
struct Configuration {
  int64_t taskGrain;
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
  int64_t regionSize;
  mlir::DictionaryAttr local;

  int64_t parameter(llvm::StringRef name) const {
    return mlir::cast<mlir::IntegerAttr>(local.get(name)).getInt();
  }
};

} // namespace intent::cpu
#endif

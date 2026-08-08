#ifndef INTENT_TARGET_COMMON_ANALYSIS_INDEXRELATION_H
#define INTENT_TARGET_COMMON_ANALYSIS_INDEXRELATION_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"

#include <optional>
#include <string>

namespace intent::target {

struct IndexTerm {
  std::string kind;
  llvm::SmallVector<std::optional<unsigned>> operands;
  llvm::SmallVector<std::optional<int64_t>> staticValues;
};

mlir::FailureOr<llvm::SmallVector<IndexTerm>>
parseIndexRelation(mlir::Operation &operation);

} // namespace intent::target

#endif

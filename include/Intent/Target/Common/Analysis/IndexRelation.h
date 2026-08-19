#ifndef INTENT_TARGET_COMMON_ANALYSIS_INDEXRELATION_H
#define INTENT_TARGET_COMMON_ANALYSIS_INDEXRELATION_H

#include "llvm/ADT/ArrayRef.h"
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

struct ScalarIndexSource {
  mlir::Operation *domain = nullptr;
  llvm::SmallVector<mlir::Operation *> domains;
  bool opaque = false;
  bool transformed = false;

  bool isStatic() const { return domains.empty() && !opaque; }
  bool hasDomain() const { return !domains.empty(); }
};

struct TensorIndexGroup {
  unsigned count = 0;
  unsigned rank = 0;

  bool requiresBroadcastProjection() const { return count > 1 || rank > 1; }
};

mlir::FailureOr<llvm::SmallVector<IndexTerm>>
parseIndexRelation(mlir::Operation &operation);

TensorIndexGroup tensorIndexGroup(mlir::Operation &operation,
                                  llvm::ArrayRef<IndexTerm> relation);

mlir::FailureOr<ScalarIndexSource>
traceScalarIndexSource(mlir::Value value, mlir::Operation &consumer);

mlir::FailureOr<llvm::SmallVector<mlir::Operation *>>
expandDomainSource(mlir::Value source, mlir::Operation &consumer);

bool hasInBoundsPrecondition(mlir::Value index, mlir::Value view, unsigned axis,
                             mlir::Operation &access);

mlir::FailureOr<bool> hasDerivedScalarIndex(mlir::Operation &operation);

mlir::FailureOr<bool> isWholeViewAccess(mlir::Operation &operation);

} // namespace intent::target

#endif

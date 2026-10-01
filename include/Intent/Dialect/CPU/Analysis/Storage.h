#ifndef INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H
#define INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::cpu {

// A closed upper bound for a current descriptor dimension. This query does not
// use observed runtime shapes or interpret wrapping index arithmetic as affine.
std::optional<int64_t> constantDimensionUpperBound(mlir::Value memory, unsigned axis);

// Current-IR aliases of the supplied origin, not an execution or allocation plan.
// Unknown memref-producing users remain visible and make the closure incomplete.
struct StorageAliasFacts {
  mlir::Value root;
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Operation *> users;
  bool complete = true;
};

bool isStorageAliasOperation(mlir::Operation *operation);
StorageAliasFacts queryStorageAliases(mlir::Value root);

struct StorageLifetime {
  mlir::memref::AllocOp allocation;
  mlir::memref::DeallocOp end;
  StorageAliasFacts aliases;

  bool contains(mlir::Operation *operation) const;
};

// Requires one lexical lifetime end covering every known alias use. Consumers
// that change storage must additionally require aliases.complete and their own
// access/effect/representation legality. Recompute after any relevant rewrite.
std::optional<StorageLifetime>
queryStorageLifetime(mlir::memref::AllocOp allocation);

// Prove that all effects in scope preserve the observed storage. Distinct SSA
// roots alone are not disjoint: use alias analysis or the explicit CPU ABI.
bool preservesStorage(mlir::Operation *scope, mlir::Value memory);
bool areDisjointStorage(mlir::Value first, mlir::Value second,
                         mlir::Operation *scope);

// A read performed in 'from' may be replayed at 'to' only while its allocation
// remains live and no intervening or enclosing-consumer effect can change it.
bool isStorageReadStable(mlir::Value memory, mlir::Operation *from,
                         mlir::Operation *to);

} // namespace intent::cpu
#endif

#ifndef INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H
#define INTENT_DIALECT_CPU_ANALYSIS_STORAGE_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::cpu {

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

} // namespace intent::cpu
#endif

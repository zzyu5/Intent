#ifndef INTENT_DSA_TRANSFORMS_STORAGE_PATTERNS_H
#define INTENT_DSA_TRANSFORMS_STORAGE_PATTERNS_H

#include "Intent/Dialect/DSA/Analysis/Storage.h"
#include "Intent/Dialect/DSA/IR/Views.h"

namespace intent::dsa::detail {

// These rewrites substitute coordinates directly. A common allocation alone
// does not preserve their indexing: both descriptors must expose the complete
// storage with the same element, shape and layout.
inline bool sameCompleteView(StorageAnalysis &storage, mlir::Value first,
                             mlir::Value second) {
  if (first == second)
    return true;
  if (first.getType() != second.getType())
    return false;
  mlir::Value origin = storage.uniqueOrigin(first);
  return origin && storage.uniqueOrigin(second) == origin &&
         isCompleteStorageViewOf(first, origin) &&
         isCompleteStorageViewOf(second, origin);
}

} // namespace intent::dsa::detail

#endif

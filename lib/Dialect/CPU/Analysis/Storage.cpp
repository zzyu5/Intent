#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cpu {

bool isStorageAliasOperation(Operation *operation) {
  return isa<ViewLikeOpInterface, memref::CastOp,
             memref::ExtractStridedMetadataOp>(operation);
}

StorageAliasFacts queryStorageAliases(Value root) {
  StorageAliasFacts result;
  result.root = root;
  result.values.push_back(root);
  llvm::DenseSet<Value> seenValues;
  llvm::DenseSet<Operation *> seenUsers;
  for (unsigned index = 0; index < result.values.size(); ++index) {
    Value alias = result.values[index];
    if (!seenValues.insert(alias).second)
      continue;
    for (Operation *user : alias.getUsers()) {
      if (seenUsers.insert(user).second)
        result.users.push_back(user);
      if (isStorageAliasOperation(user)) {
        // A view-like operation can also read another memref (for example the
        // shape operand of memref.reshape). Only its view source forwards data.
        if (auto view = dyn_cast<ViewLikeOpInterface>(user);
            view && view.getViewSource() != alias) {
          result.complete = false;
          continue;
        }
        for (Value value : user->getResults())
          if (isa<BaseMemRefType>(value.getType()) && !seenValues.contains(value))
            result.values.push_back(value);
      } else if (auto tasks = dyn_cast<TasksOp>(user)) {
        for (auto [position, capture] : llvm::enumerate(tasks.getCaptures()))
          if (capture == alias)
            result.values.push_back(tasks.getBody().front().getArgument(position + 1));
      } else if (llvm::any_of(user->getResultTypes(), [](Type type) {
                   return isa<BaseMemRefType>(type);
                 })) {
        result.complete = false;
      }
    }
  }
  return result;
}

bool StorageLifetime::contains(Operation *operation) const {
  auto owner = allocation;
  auto release = end;
  Operation *ancestor = owner->getBlock()->findAncestorOpInBlock(*operation);
  return ancestor && owner->isBeforeInBlock(ancestor) &&
         ancestor->isBeforeInBlock(release);
}

std::optional<StorageLifetime>
queryStorageLifetime(memref::AllocOp allocation) {
  StorageLifetime lifetime{allocation, {}, queryStorageAliases(allocation)};
  for (Operation *user : lifetime.aliases.users) {
    auto end = dyn_cast<memref::DeallocOp>(user);
    if (!end)
      continue;
    if (lifetime.end || end.getMemref() != allocation.getResult() ||
        end->getBlock() != allocation->getBlock() ||
        !allocation->isBeforeInBlock(end))
      return std::nullopt;
    lifetime.end = end;
  }
  if (!lifetime.end)
    return std::nullopt;
  for (Operation *user : lifetime.aliases.users)
    if (user != lifetime.end && !lifetime.contains(user))
      return std::nullopt;
  return lifetime;
}

} // namespace intent::cpu

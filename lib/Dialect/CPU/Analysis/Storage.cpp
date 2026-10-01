#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cpu {

std::optional<int64_t> constantDimensionUpperBound(Value memory, unsigned axis) {
  auto type = dyn_cast<MemRefType>(memory.getType());
  if (!type || axis >= static_cast<unsigned>(type.getRank())) return std::nullopt;
  auto stop = [](Value value, std::optional<int64_t> dimension, ValueBoundsConstraintSet &) {
    if (dimension) return false;
    // Descriptor dimensions and constants are exact size facts. MLIR's affine
    // models for arbitrary index arithmetic do not prove absence of wrapping;
    // leave those scalar leaves unconstrained instead of shrinking from them.
    return !isa_and_nonnull<arith::ConstantOp, memref::DimOp, memref::RankOp>(value.getDefiningOp());
  };
  auto bound = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::UB, ValueBoundsConstraintSet::Variable(memory, axis), stop,
      /*closedUB=*/true);
  if (failed(bound) || *bound < 0) return std::nullopt;
  return *bound;
}

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

namespace {
bool disjoint(Value first, Value second, PhysicalProgramAnalysis &physical,
              AliasAnalysis &aliases, EntryRequirementsAttr interface) {
  Value root = physical.storageRoot(first), other = physical.storageRoot(second);
  if (root == other) return false;
  if (aliases.alias(root, other).isNo()) return true;
  auto source = physical.externalView(root), destination = physical.externalView(other);
  if (!source || !destination) return false;
  return source.getConstraints().getNoalias() || destination.getConstraints().getNoalias() ||
      (interface && interface.getDisjointOutputs() &&
       (source.getAccess() != 0 || destination.getAccess() != 0));
}
} // namespace

bool areDisjointStorage(Value first, Value second, Operation *scope) {
  auto function = dyn_cast<func::FuncOp>(scope);
  if (!function) function = scope->getParentOfType<func::FuncOp>();
  if (!function) return false;
  PhysicalProgramAnalysis physical(function);
  AliasAnalysis aliases(function);
  return disjoint(first, second, physical, aliases,
      function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr));
}

bool preservesStorage(Operation *scope, Value memory) {
  auto function = dyn_cast<func::FuncOp>(scope);
  if (!function) function = scope->getParentOfType<func::FuncOp>();
  if (!function) return false;
  auto effects = getEffectsRecursively(scope);
  if (!effects) return false;
  PhysicalProgramAnalysis physical(function);
  AliasAnalysis aliases(function);
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  return llvm::all_of(*effects, [&](const MemoryEffects::EffectInstance &effect) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) return true;
    Value affected = effect.getValue();
    if (!affected || !isa<BaseMemRefType>(affected.getType())) return false;
    return disjoint(memory, affected, physical, aliases, interface);
  });
}

bool isStorageReadStable(Value memory, Operation *from, Operation *to) {
  auto function = from->getParentOfType<func::FuncOp>();
  if (!function || function != to->getParentOfType<func::FuncOp>() ||
      !DominanceInfo(function).dominates(memory, to)) return false;
  Operation *consumer = from->getBlock()->findAncestorOpInBlock(*to);
  if (!consumer || consumer == from || !from->isBeforeInBlock(consumer)) return false;
  PhysicalProgramAnalysis physical(function);
  Value root = physical.storageRoot(memory);
  if (auto allocation = root.getDefiningOp<memref::AllocOp>()) {
    auto lifetime = queryStorageLifetime(allocation);
    if (!lifetime || !lifetime->aliases.complete || !lifetime->contains(to)) return false;
  }
  for (Operation *operation = from; ; operation = operation->getNextNode()) {
    if (!preservesStorage(operation, memory)) return false;
    if (operation == consumer) break;
  }
  return true;
}

} // namespace intent::cpu

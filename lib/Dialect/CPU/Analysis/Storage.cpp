#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CollectiveHelpers.h"
#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cpu {
namespace {

struct CacheHintResource : SideEffects::Resource::Base<CacheHintResource> {
  StringRef getName() final { return "IntentCPUCacheHint"; }
};

struct PrefetchEffects
    : MemoryEffectOpInterface::ExternalModel<PrefetchEffects, memref::PrefetchOp> {
  void getEffects(Operation *operation,
                  SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    // MLIR 20 leaves prefetch effects unspecified. Both cache intentions observe
    // the descriptor's storage without modifying its contents; isWrite is a
    // cache hint, not a store. Keep the op's original speculation contract.
    effects.emplace_back(MemoryEffects::Read::get(),
                         &operation->getOpOperand(0));
    // Preserve the hint through ordinary MLIR dead-op elimination even though
    // it produces no SSA result. Cache state is separate from buffer contents.
    effects.emplace_back(MemoryEffects::Write::get(), CacheHintResource::get());
  }
};

bool isBuffer(Value value) {
  return value && isa<BaseMemRefType>(value.getType());
}

bool isAllocation(Value value) {
  Operation *owner = value.getDefiningOp();
  return owner && hasEffect<MemoryEffects::Allocate>(owner, value);
}

bool isEntryArgument(Value value, func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &function.front();
}

bool hasBufferResultOrArgument(Operation *operation) {
  if (llvm::any_of(operation->getResults(), isBuffer)) return true;
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      if (llvm::any_of(block.getArguments(), isBuffer)) return true;
  return false;
}

bool describesBufferFlow(Operation *operation) {
  return isa<bufferization::BufferViewFlowOpInterface, ViewLikeOpInterface,
             BranchOpInterface, RegionBranchOpInterface,
             RegionBranchTerminatorOpInterface>(operation);
}

bool hasAtomicOrdering(Operation *operation) {
  return isa<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp,
             memref::AtomicRMWOp, memref::GenericAtomicRMWOp>(operation);
}

func::FuncOp enclosingFunction(Operation *scope) {
  if (auto function = dyn_cast<func::FuncOp>(scope)) return function;
  return scope->getParentOfType<func::FuncOp>();
}

} // namespace

void registerStorageInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, memref::MemRefDialect *) {
    memref::PrefetchOp::attachInterface<PrefetchEffects>(*context);
  });
}

std::optional<int64_t> constantDimensionUpperBound(Value memory, unsigned axis) {
  auto type = dyn_cast<MemRefType>(memory.getType());
  if (!type || axis >= static_cast<unsigned>(type.getRank())) return std::nullopt;
  return constantExtentUpperBound(ValueBoundsConstraintSet::Variable(memory, axis));
}

Value StorageOriginFacts::uniqueOrigin() const {
  return complete && values.size() == 1 ? values.front() : Value{};
}

StorageAnalysis::StorageAnalysis(func::FuncOp function)
    : function(function), flow(function), bufferOrigins(function),
      aliasAnalysis(function), dominance(function) {}

StorageOriginFacts StorageAnalysis::origins(Value memory) const {
  StorageOriginFacts result;
  if (!isBuffer(memory)) return result;
  result.complete = true;
  for (Value value : flow.resolveReverse(memory)) {
    if (!isBuffer(value) || !flow.mayBeTerminalBuffer(value)) continue;
    result.values.push_back(value);
    // A declared helper formal has identity within its invocation, but is not a
    // fresh allocation. Unknown operation terminals have no such contract.
    auto formal = dyn_cast<BlockArgument>(value);
    if (!isAllocation(value) && !isEntryArgument(value, function) &&
        !(formal && isCollectiveArgument(formal)))
      result.complete = false;
  }
  if (result.values.empty()) result.complete = false;
  return result;
}

Value StorageAnalysis::uniqueOrigin(Value memory) const {
  return origins(memory).uniqueOrigin();
}

intent::ViewType StorageAnalysis::externalView(Value memory) const {
  auto argument = dyn_cast_or_null<BlockArgument>(uniqueOrigin(memory));
  auto currentFunction = function;
  if (!argument || argument.getOwner() != &currentFunction.front()) return {};
  auto interface = getPublicInterface(currentFunction);
  return interface ? getPublicView(interface, argument.getArgNumber())
                   : intent::ViewType{};
}

bool StorageAnalysis::isReadOnly(Value memory) const {
  if (auto formal = dyn_cast_or_null<BlockArgument>(memory);
      formal && isReadOnlyCollectiveArgument(formal)) return true;
  auto source = origins(memory);
  if (source.values.empty()) return false;
  return llvm::all_of(source.values, [&](Value value) {
    if (auto view = externalView(value)) return view.getAccess() == 0;
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && isReadOnlyCollectiveArgument(argument);
  });
}

bool isStorageAliasOperation(Operation *operation) {
  return isa<ViewLikeOpInterface, memref::CastOp,
             memref::ExtractStridedMetadataOp>(operation);
}

StorageAliasFacts StorageAnalysis::aliases(Value root) const {
  StorageAliasFacts result;
  result.root = root;
  if (!isBuffer(root)) {
    result.complete = false;
    return result;
  }
  llvm::DenseSet<Operation *> seenUsers;
  for (Value value : flow.resolve(root)) {
    if (!isBuffer(value)) continue;
    result.values.push_back(value);
    for (Operation *user : value.getUsers()) {
      if (seenUsers.insert(user).second) result.users.push_back(user);
      // Calls and returns may escape storage. Unknown buffer-producing users
      // cannot be silently treated as ordinary reads just because their effects
      // are known. Native forwarding interfaces own their complete flow.
      if (isa<CallOpInterface, func::ReturnOp,
              memref::ExtractAlignedPointerAsIndexOp>(user) ||
          (!describesBufferFlow(user) && hasBufferResultOrArgument(user)) ||
          !effects(user).complete)
        result.complete = false;
    }
  }
  return result;
}

bool StorageLifetime::contains(Operation *operation) const {
  Operation *ancestor = allocation->getBlock()->findAncestorOpInBlock(*operation);
  return ancestor && allocation->isBeforeInBlock(ancestor) &&
         ancestor->isBeforeInBlock(end);
}

std::optional<StorageLifetime>
StorageAnalysis::lifetime(memref::AllocOp allocation) const {
  StorageLifetime result{allocation, {}, aliases(allocation)};
  if (!result.aliases.complete) return std::nullopt;
  for (Operation *user : result.aliases.users) {
    auto release = dyn_cast<memref::DeallocOp>(user);
    if (!release) continue;
    if (result.end || release.getMemref() != allocation.getResult() ||
        release->getBlock() != allocation->getBlock() ||
        !allocation->isBeforeInBlock(release))
      return std::nullopt;
    result.end = release;
  }
  if (!result.end) return std::nullopt;
  for (Operation *user : result.aliases.users)
    if (user != result.end && !result.contains(user)) return std::nullopt;
  return result;
}

StorageEffects StorageAnalysis::effects(Operation *scope) const {
  StorageEffects result;
  SmallVector<Operation *> pending;
  if (isa<func::FuncOp>(scope)) {
    for (Region &region : scope->getRegions())
      for (Block &block : region)
        for (Operation &operation : block) pending.push_back(&operation);
  } else pending.push_back(scope);
  while (!pending.empty()) {
    Operation *operation = pending.pop_back_val();
    bool recursive = operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (recursive)
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (Operation &nested : block) pending.push_back(&nested);
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      SmallVector<MemoryEffects::EffectInstance> current;
      interface.getEffects(current);
      for (auto effect : current)
        if (effect.getResource() != CacheHintResource::get())
          result.entries.push_back({operation, effect});
    } else if (!recursive) result.complete = false;
    result.ordered |= hasAtomicOrdering(operation);
  }
  return result;
}

bool StorageAnalysis::disjoint(Value first, Value second) {
  if (!isBuffer(first) || !isBuffer(second) || first == second) return false;
  auto left = origins(first), right = origins(second);
  // Native local AA does not consume external buffer-flow models. In particular
  // an isolated task formal can carry an existing allocation. Resolve the
  // registered flow before asking AA about terminal storage identities.
  for (Value lhs : left.values)
    if (llvm::is_contained(right.values, lhs)) return false;
  auto allocatedAfter = [&](const StorageOriginFacts &sources, Value existing) {
    return sources.complete && llvm::all_of(sources.values, [&](Value source) {
      return isAllocation(source) &&
          dominance.properlyDominates(existing, source.getDefiningOp());
    });
  };
  // A fresh allocation cannot alias a descriptor already available before it.
  // This also applies inside a helper with opaque state formals: it proves only
  // the local allocation's freshness, never disjointness between those formals.
  if (allocatedAfter(left, second) || allocatedAfter(right, first)) return true;
  if (!left.values.empty() && !right.values.empty())
    if (auto same = bufferOrigins.isSameAllocation(first, second); same && !*same)
      return true;
  if (!left.complete || !right.complete) return false;
  auto interface = getPublicInterface(function);
  auto requirements = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  for (Value lhs : left.values)
    for (Value rhs : right.values) {
      if (lhs == rhs) return false;
      if (aliasAnalysis.alias(lhs, rhs).isNo()) continue;
      if (!isEntryArgument(lhs, function) || !isEntryArgument(rhs, function) || !interface)
        return false;
      auto source = getPublicView(interface, cast<BlockArgument>(lhs).getArgNumber());
      auto target = getPublicView(interface, cast<BlockArgument>(rhs).getArgNumber());
      if (!source || !target) return false;
      if (source.getConstraints().getNoalias() || target.getConstraints().getNoalias()) continue;
      if (requirements && requirements.getDisjointOutputs() &&
          (source.getAccess() != 0 || target.getAccess() != 0)) continue;
      return false;
    }
  return true;
}

bool StorageAnalysis::preserves(Operation *scope, Value memory) {
  auto summary = effects(scope);
  if (!summary.complete || summary.ordered) return false;
  for (const auto &entry : summary.entries) {
    const auto &effect = entry.effect;
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
    if (!disjoint(memory, effect.getValue())) return false;
  }
  return true;
}

bool StorageAnalysis::isLiveAt(Value memory, Operation *operation) const {
  auto origin = origins(memory);
  if (origin.values.empty()) return false;
  for (Value value : origin.values) {
    if (auto allocation = value.getDefiningOp<memref::AllocOp>()) {
      auto owner = lifetime(allocation);
      if (!owner || !owner->contains(operation)) return false;
    } else if (auto allocation = value.getDefiningOp<memref::AllocaOp>()) {
      Operation *owner = allocation->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
      if (!owner || !owner->isAncestor(operation)) return false;
    } else if (!isEntryArgument(value, function)) {
      // A modeled region's opaque formal is valid for that region invocation.
      // This is a lifetime fact, not a fresh-allocation or no-alias assertion.
      auto formal = dyn_cast<BlockArgument>(value);
      if (!formal || !isCollectiveArgument(formal) ||
          !formal.getOwner()->getParent()->isAncestor(operation->getParentRegion()))
        return false;
    }
  }
  return true;
}

bool StorageAnalysis::unchangedBetween(Value memory, Operation *from, Operation *to) {
  if (!isBuffer(memory) || enclosingFunction(from) != function ||
      enclosingFunction(to) != function || from->getBlock() != to->getBlock() ||
      !from->isBeforeInBlock(to) || !dominance.dominates(memory, to) ||
      !isLiveAt(memory, to)) return false;
  for (Operation *operation = from->getNextNode(); operation != to;
       operation = operation->getNextNode())
    if (!preserves(operation, memory)) return false;
  return true;
}

bool StorageAnalysis::readStable(Value memory, Operation *from, Operation *to) {
  if (!isBuffer(memory) || enclosingFunction(from) != function ||
      enclosingFunction(to) != function || !dominance.dominates(memory, to) ||
      !isLiveAt(memory, to)) return false;
  Operation *consumer = from->getBlock()->findAncestorOpInBlock(*to);
  if (!consumer || consumer == from || !from->isBeforeInBlock(consumer)) return false;
  for (Operation *operation = from; ; operation = operation->getNextNode()) {
    if (!preserves(operation, memory)) return false;
    if (operation == consumer) break;
  }
  return true;
}

} // namespace intent::cpu

#include "TraversalFusion.h"
#include "Intent/Dialect/CPU/Analysis/ViewRelations.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::cpu::detail {

bool sameTraversalValue(Value first, Value second, const IRMapping &mapping) {
  if (mapping.lookupOrDefault(first) == second) return true;
  auto lhs = getConstantIntValue(first), rhs = getConstantIntValue(second);
  if (lhs && rhs) return first.getType() == second.getType() && *lhs == *rhs;
  auto firstResult = dyn_cast<OpResult>(first), secondResult = dyn_cast<OpResult>(second);
  if (!firstResult || !secondResult || firstResult.getResultNumber() != secondResult.getResultNumber())
    return false;
  Operation *a = firstResult.getOwner(), *b = secondResult.getOwner();
  if (a->getNumRegions() || b->getNumRegions() ||
      !isMemoryEffectFree(a) || !isMemoryEffectFree(b)) return false;
  return OperationEquivalence::isEquivalentTo(a, b,
      [&](Value left, Value right) { return success(sameTraversalValue(left, right, mapping)); },
      [](Value, Value) {}, OperationEquivalence::IgnoreLocations);
}

bool sameTraversalAddress(const TraversalAccess &first, const TraversalAccess &second,
                          const IRMapping &mapping) {
  if (first.indexed != second.indexed || first.indices.size() != second.indices.size() ||
      !sameTraversalValue(first.memory, second.memory, mapping)) return false;
  if (!first.indexed && (!first.indexingMap || !second.indexingMap ||
                         first.indexingMap != second.indexingMap)) return false;
  return llvm::all_of(llvm::zip(first.indices, second.indices), [&](auto pair) {
    return sameTraversalValue(std::get<0>(pair), std::get<1>(pair), mapping);
  });
}

FailureOr<SmallVector<TraversalAccess>> traversalAccesses(
    const Traversal &traversal, StorageAnalysis &storage) {
  auto effects = storage.effects(traversal.operation);
  if (!effects.complete || effects.ordered) return failure();
  SmallVector<TraversalAccess> accesses;
  for (const StorageEffect &entry : effects.entries) {
    const auto &effect = entry.effect;
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
    Value memory = effect.getValue();
    if (!memory || !isa<MemRefType>(memory.getType())) return failure();
    auto origins = storage.origins(memory);
    bool local = origins.complete && !origins.values.empty() &&
        llvm::all_of(origins.values, [&](Value origin) {
          Operation *owner = origin.getDefiningOp();
          if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) ||
              !traversal.operation->isAncestor(owner)) return false;
          auto aliases = storage.aliases(origin);
          return aliases.complete && llvm::all_of(aliases.users, [&](Operation *user) {
            return traversal.operation->isAncestor(user);
          });
        });
    if (local) continue;
    if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) return failure();
    TraversalAccess access{entry.operation, memory, {}, false,
                            isa<MemoryEffects::Write>(effect.getEffect())};
    if (auto load = dyn_cast<memref::LoadOp>(entry.operation)) {
      access.indices.assign(load.getIndices().begin(), load.getIndices().end());
      access.indexed = true;
    } else if (auto store = dyn_cast<memref::StoreOp>(entry.operation)) {
      access.indices.assign(store.getIndices().begin(), store.getIndices().end());
      access.indexed = true;
    } else if (auto generic = dyn_cast<linalg::LinalgOp>(entry.operation)) {
      for (OpOperand &operand : generic->getOpOperands()) {
        if (operand.get() != memory ||
            (access.write ? !generic.isDpsInit(&operand)
                          : !generic.payloadUsesValueFromOperand(&operand))) continue;
        access.indexingMap = generic.getMatchingIndexingMap(&operand);
        accesses.push_back(access);
      }
      continue;
    }
    accesses.push_back(std::move(access));
  }
  return accesses;
}

bool containedSubview(memref::SubViewOp view) {
  using Bounds = ValueBoundsConstraintSet;
  using Variable = Bounds::Variable;
  MLIRContext *context = view.getContext();
  auto zero = IntegerAttr::get(IndexType::get(context), 0);
  if (!llvm::all_of(view.getMixedStrides(), [](OpFoldResult stride) {
        return getConstantIntValue(stride) == 1;
      })) return false;
  AffineMap addition = AffineMap::get(2, 0,
      getAffineDimExpr(0, context) + getAffineDimExpr(1, context));
  auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
  for (auto [axis, pair] : llvm::enumerate(llvm::zip(offsets, sizes))) {
    Variable offset(std::get<0>(pair)), size(std::get<1>(pair));
    Variable end(addition, ArrayRef<Variable>{offset, size});
    if (!Bounds::compare(offset, Bounds::GE, Variable(zero)) ||
        !Bounds::compare(size, Bounds::GE, Variable(zero)) ||
        !Bounds::compare(end, Bounds::LE, Variable(view.getSource(), axis)))
      return false;
  }
  return true;
}

bool separatesIterations(const TraversalAccess &access, const Traversal &traversal,
                         DominanceInfo &dominance) {
  // A descriptor outside the traversal is fixed for all of its iterations.
  // Contiguity plus explicit distinct coordinates establishes disjoint cells.
  if (access.indexed && dominance.dominates(access.memory, traversal.operation) &&
      isContiguousDescriptor(access.memory) &&
      llvm::all_of(traversal.coordinates, [&](Value coordinate) {
        return llvm::is_contained(access.indices, coordinate);
      })) return true;
  Value memory = access.memory;
  while (Operation *owner = memory.getDefiningOp()) {
    if (auto cast = dyn_cast<memref::CastOp>(owner)) {
      memory = cast.getSource();
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(owner);
    if (!view || !containedSubview(view)) return false;
    if (dominance.dominates(view.getSource(), traversal.operation) &&
        isContiguousDescriptor(view.getSource())) {
      auto offsets = view.getMixedOffsets(), sizes = view.getMixedSizes();
      return llvm::all_of(traversal.coordinates, [&](Value coordinate) {
        for (auto [offset, size] : llvm::zip(offsets, sizes))
          if (offset == OpFoldResult(coordinate) && getConstantIntValue(size) == 1)
            return true;
        return false;
      });
    }
    memory = view.getSource();
  }
  return false;
}

namespace {

struct AvailableRead {
  TraversalAccess access;
  Value value;
};

// A block keeps only definitely executed values. Branch-local values are never
// published to the parent scope, and an aliasing write invalidates cached reads.
bool forwardOne(Block &block, SmallVector<AvailableRead> available,
                StorageAnalysis &storage) {
  IRMapping identity;
  for (Operation &operation : block) {
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      TraversalAccess access{load, load.getMemref(), llvm::to_vector(load.getIndices()), true, false};
      auto found = llvm::find_if(llvm::reverse(available), [&](const AvailableRead &previous) {
        return previous.value.getType() == load.getType() &&
               sameTraversalAddress(previous.access, access, identity);
      });
      if (found != llvm::reverse(available).end()) {
        load.getResult().replaceAllUsesWith(found->value);
        load.erase();
        return true;
      }
      available.push_back({std::move(access), load.getResult()});
      continue;
    }
    auto effects = storage.effects(&operation);
    if (!effects.complete || effects.ordered) {
      available.clear();
      continue;
    }
    for (Region &region : operation.getRegions()) {
      for (Block &nested : region) {
        SmallVector<AvailableRead> inherited;
        // For loops, a write in a previous iteration can invalidate the incoming
        // value. Branches execute at most once and retain their enclosing guard.
        if (isa<scf::IfOp>(operation)) inherited = available;
        else
          for (const auto &entry : available)
            if (storage.preserves(&operation, entry.access.memory)) inherited.push_back(entry);
        if (forwardOne(nested, std::move(inherited), storage)) return true;
      }
    }
    for (const StorageEffect &entry : effects.entries) {
      if (!isa<MemoryEffects::Write, MemoryEffects::Free>(entry.effect.getEffect())) continue;
      Value memory = entry.effect.getValue();
      llvm::erase_if(available, [&](const AvailableRead &previous) {
        return !memory || !storage.disjoint(memory, previous.access.memory);
      });
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation))
      available.push_back({{store, store.getMemref(), llvm::to_vector(store.getIndices()), true, true},
                           store.getValue()});
  }
  return false;
}

} // namespace

bool reuseTraversalReads(func::FuncOp function) {
  StorageAnalysis storage(function);
  for (Block &block : function.getBody())
    if (forwardOne(block, {}, storage)) return true;
  return false;
}

} // namespace intent::cpu::detail

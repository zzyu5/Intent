#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;
namespace intent::cpu {
namespace {

struct PreparedInput {
  linalg::LinalgOp producer;
  memref::AllocOp allocation;
  memref::DeallocOp end;
  ImplementationAttr consumer;
};

std::optional<PreparedInput> scopedInput(linalg::LinalgOp producer, Operation *loop,
    PhysicalProgramAnalysis &physical, ArrayRef<MemoryEffects::EffectInstance> effects,
    ArrayRef<MemoryAccess> accesses, const llvm::SmallDenseSet<Value> &groupInputs) {
  if (producer->getNumResults() || producer.getNumDpsInits() != 1 || producer.getNumReductionLoops() ||
      !producer.getIndexingMapsArray().back().isIdentity() ||
      !producer->getRegion(0).front().getArguments().back().use_empty()) return std::nullopt;
  auto allocation = producer.getDpsInits()[0].getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != producer->getBlock() ||
      !allocation->isBeforeInBlock(producer)) return std::nullopt;
  auto stable = [&](Value memory) {
    if (physical.isReadOnly(memory)) return true;
    Value root = physical.storageRoot(memory);
    if (groupInputs.contains(root)) return true;
    Operation *owner = root.getDefiningOp();
    if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(owner) || loop->isAncestor(owner)) return false;
    if (llvm::any_of(accesses, [&](const MemoryAccess &access) {
          return access.write && physical.storageRoot(access.memory) == root;
        })) return false;
    return llvm::all_of(effects, [&](const MemoryEffects::EffectInstance &effect) {
      return isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect()) ||
          (effect.getValue() && physical.storageRoot(effect.getValue()) != root);
    });
  };
  for (Value input : producer.getDpsInputs())
    if (isa<MemRefType>(input.getType()) && !stable(input)) return std::nullopt;
  for (Operation &operation : producer->getRegion(0).front().without_terminator()) {
    if (operation.getNumRegions()) return std::nullopt;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      if (!stable(load.getMemref())) return std::nullopt;
    } else if (!isMemoryEffectFree(&operation)) return std::nullopt;
  }

  PreparedInput result{producer, allocation, {}, {}};
  SmallVector<Value> aliases{allocation.getResult()};
  llvm::SmallDenseSet<Value> seen;
  SmallVector<Operation *> reads;
  for (unsigned i = 0; i < aliases.size(); ++i) {
    Value value = aliases[i];
    if (!seen.insert(value).second) continue;
    for (Operation *user : value.getUsers()) {
      if (user == producer) continue;
      if (!loop->isAncestor(user)) return std::nullopt;
      if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
      else if (auto cast = dyn_cast<memref::CastOp>(user)) aliases.push_back(cast.getResult());
      else if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) aliases.push_back(view.getResult());
      else if (isa<memref::DimOp>(user)) continue;
      else if (auto end = dyn_cast<memref::DeallocOp>(user)) {
        if (value != allocation.getResult() || result.end || end->getBlock() != producer->getBlock())
          return std::nullopt;
        result.end = end;
      } else {
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          if (physical.storageRoot(copy.getTarget()) == allocation.getResult()) return std::nullopt;
        } else if (auto generic = dyn_cast<linalg::LinalgOp>(user)) {
          for (Value output : generic.getDpsInits())
            if (physical.storageRoot(output) == allocation.getResult()) return std::nullopt;
        } else if (!isa<memref::LoadOp>(user)) return std::nullopt;
        reads.push_back(user);
      }
    }
  }
  if (!result.end || reads.empty() || !producer->isBeforeInBlock(result.end)) return std::nullopt;
  for (Operation *read : reads) {
    Operation *ancestor = producer->getBlock()->findAncestorOpInBlock(*read);
    if (!ancestor || !producer->isBeforeInBlock(ancestor) || !ancestor->isBeforeInBlock(result.end))
      return std::nullopt;
  }
  return result;
}

void groupScopedInputs(func::FuncOp function) {
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
  if (!interface || !interface.getDisjointOutputs()) return;
  SmallVector<Operation *> loops;
  function.walk([&](Operation *operation) {
    if (isa<scf::ForOp, scf::ParallelOp>(operation)) loops.push_back(operation);
  });
  for (Operation *loop : llvm::reverse(loops)) {
    Value lower, upper, step, induction;
    SmallVector<Value> outerCoordinates;
    Block *body;
    if (auto serial = dyn_cast<scf::ForOp>(loop)) {
      if (serial.getNumResults()) continue;
      lower = serial.getLowerBound(); upper = serial.getUpperBound(); step = serial.getStep();
      induction = serial.getInductionVar(); body = serial.getBody();
    } else {
      auto parallel = cast<scf::ParallelOp>(loop);
      if (!parallel.getNumLoops() || parallel.getNumResults()) continue;
      lower = parallel.getLowerBound().back(); upper = parallel.getUpperBound().back(); step = parallel.getStep().back();
      induction = parallel.getInductionVars().back(); body = parallel.getBody();
      llvm::append_range(outerCoordinates, ValueRange(parallel.getInductionVars()).drop_back());
    }
    if (getConstantIntValue(lower) != 0 || getConstantIntValue(step) != 1) continue;
    auto effects = getEffectsRecursively(loop);
    if (!effects) continue;
    bool knownEffects = true;
    loop->walk([&](Operation *operation) {
      if (!isa<MemoryEffectOpInterface>(operation) &&
          !operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) knownEffects = false;
    });
    if (!knownEffects) continue;
    PhysicalProgramAnalysis physical(function);
    auto accesses = physical.accesses(loop);
    for (auto quotient : body->getOps<arith::FloorDivSIOp>()) {
      auto divisor = getConstantIntValue(quotient.getRhs());
      if (quotient.getLhs() != induction || !divisor || *divisor <= 1) continue;
      SmallVector<PreparedInput> preparations;
      llvm::SetVector<Operation *> dependencies;
      llvm::SmallDenseSet<Value> groupInputs;
      bool groupDependent = false;
      for (Operation &operation : body->without_terminator()) {
        auto producer = dyn_cast<linalg::LinalgOp>(operation);
        if (!producer) continue;
        auto prepared = scopedInput(producer, loop, physical, *effects, accesses, groupInputs);
        if (!prepared) continue;
        llvm::SetVector<Operation *> needed;
        bool usesQuotient = false;
        std::function<bool(Value)> invariant = [&](Value value) {
          if (value == quotient.getResult()) { usesQuotient = true; return true; }
          if (groupInputs.contains(value) || llvm::is_contained(outerCoordinates, value)) return true;
          Operation *scope = value.getParentRegion()->getParentOp();
          if (scope != loop && !loop->isAncestor(scope)) return true;
          Operation *definition = value.getDefiningOp();
          if (!definition || definition->getBlock() != body || definition->getNumRegions() ||
              !isMemoryEffectFree(definition)) return false;
          if (needed.contains(definition)) return true;
          if (!llvm::all_of(definition->getOperands(), invariant)) return false;
          needed.insert(definition);
          return true;
        };
        llvm::SetVector<Value> inputs;
        getUsedValuesDefinedAbove(producer->getRegion(0), inputs);
        auto operands = producer.getDpsInputs();
        inputs.insert(operands.begin(), operands.end());
        inputs.insert(prepared->allocation->operand_begin(), prepared->allocation->operand_end());
        if (!llvm::all_of(inputs, invariant)) continue;
        dependencies.insert(needed.begin(), needed.end());
        preparations.push_back(*prepared);
        groupInputs.insert(prepared->allocation.getResult());
        groupDependent |= usesQuotient;
      }
      if (!groupDependent) continue;
      OpBuilder builder(loop);
      Location loc = loop->getLoc();
      Value size = index(builder, loc, *divisor);
      Value count = builder.create<arith::CeilDivSIOp>(loc, upper, size);
      Value ordinal;
      Block *group;
      scf::ParallelOp parallel = dyn_cast<scf::ParallelOp>(loop), outerParallel;
      if (parallel) {
        SmallVector<Value> ends(parallel.getUpperBound());
        ends.back() = count;
        outerParallel = builder.create<scf::ParallelOp>(loc, parallel.getLowerBound(), ends, parallel.getStep());
        group = outerParallel.getBody(); ordinal = outerParallel.getInductionVars().back();
      } else {
        auto outer = builder.create<scf::ForOp>(loc, lower, count, step);
        group = outer.getBody(); ordinal = outer.getInductionVar();
      }
      builder.setInsertionPointToStart(group);
      Value begin = multiply(builder, loc, ordinal, size);
      Value remaining = builder.create<arith::SubIOp>(loc, upper, begin);
      Value extent = builder.create<arith::MinSIOp>(loc, size, remaining);
      Value end = add(builder, loc, begin, extent);
      if (parallel) {
        auto inner = builder.create<scf::ForOp>(loc, begin, end, step);
        auto replacements = outerParallel.getInductionVars();
        for (auto [original, replacement] : llvm::zip(outerCoordinates, ArrayRef(replacements).drop_back()))
          original.replaceAllUsesWith(replacement);
        induction.replaceAllUsesWith(inner.getInductionVar());
        for (Operation &operation : llvm::make_early_inc_range(body->without_terminator()))
          operation.moveBefore(inner.getBody()->getTerminator());
        parallel.erase();
        loop = inner;
        body = inner.getBody();
      } else {
        auto serial = cast<scf::ForOp>(loop);
        serial->moveBefore(group->getTerminator());
        serial.setLowerBound(begin); serial.setUpperBound(end);
      }
      quotient.getResult().replaceAllUsesWith(ordinal);
      quotient.erase();
      for (auto preparation : preparations) {
        dependencies.insert(preparation.allocation);
        dependencies.insert(preparation.producer);
      }
      SmallVector<Operation *> ordered;
      for (Operation &operation : body->without_terminator())
        if (dependencies.contains(&operation)) ordered.push_back(&operation);
      // Preserve the preparation chain's original order, including allocations
      // used by later views and producers; all consumers finish before release.
      for (Operation *operation : ordered) operation->moveBefore(loop);
      for (auto preparation : preparations) preparation.end->moveAfter(loop);
      break;
    }
  }
}

std::optional<PreparedInput> preparedInput(linalg::GenericOp producer) {
  if (producer.getNumResults() || producer.getOutputs().size() != 1 ||
      producer.getNumReductionLoops() || !producer.getIndexingMapsArray().back().isPermutation() ||
      !producer.getRegion().front().getArguments().back().use_empty()) return std::nullopt;
  for (Operation &operation : producer.getRegion().front().without_terminator())
    if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return std::nullopt;
  Value output = producer.getOutputs()[0];
  auto allocation = output.getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != producer->getBlock() ||
      !allocation->isBeforeInBlock(producer)) return std::nullopt;
  PreparedInput result{cast<linalg::LinalgOp>(producer.getOperation()), allocation, {}, {}};
  SmallVector<Value> aliases{output};
  llvm::SmallDenseSet<Value> seen;
  SmallVector<Operation *> users;
  for (unsigned i = 0; i < aliases.size(); ++i) {
    Value value = aliases[i];
    if (!seen.insert(value).second) continue;
    for (Operation *user : value.getUsers()) {
      if (user == producer) continue;
      if (auto cast = dyn_cast<memref::CastOp>(user)) aliases.push_back(cast.getResult());
      else if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
      else if (auto end = dyn_cast<memref::DeallocOp>(user)) {
        if (value != output || result.end || end->getBlock() != producer->getBlock()) return std::nullopt;
        result.end = end;
        continue;
      } else if (auto generic = dyn_cast<linalg::GenericOp>(user)) {
        if (llvm::is_contained(generic.getOutputs(), value)) return std::nullopt;
        for (Operation &operation : generic.getRegion().front().without_terminator())
          if (operation.getNumRegions() || !isMemoryEffectFree(&operation)) return std::nullopt;
        if (isMatrixContraction(generic)) {
          auto binding = generic->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
          if (!binding || (result.consumer && result.consumer != binding)) return std::nullopt;
          result.consumer = binding;
        }
      } else if (!isa<memref::DimOp, memref::LoadOp>(user)) return std::nullopt;
      users.push_back(user);
    }
  }
  if (!result.end || !result.consumer) return std::nullopt;
  for (Operation *user : users) {
    Operation *ancestor = producer->getBlock()->findAncestorOpInBlock(*user);
    if (!ancestor || !producer->isBeforeInBlock(ancestor) || !ancestor->isBeforeInBlock(result.end)) return std::nullopt;
  }
  return result;
}

bool equivalent(PreparedInput &lhs, PreparedInput &rhs, PhysicalProgramAnalysis &physical) {
  if (lhs.producer->getBlock() != rhs.producer->getBlock() ||
      !lhs.producer->isBeforeInBlock(rhs.producer) || lhs.consumer != rhs.consumer ||
      lhs.allocation.getType() != rhs.allocation.getType() ||
      lhs.allocation->getAttrs() != rhs.allocation->getAttrs()) return false;
  auto maps = lhs.producer.getIndexingMapsArray();
  for (int64_t axis = 0; axis < lhs.allocation.getType().getRank(); ++axis) {
    if (!lhs.allocation.getType().isDynamicDim(axis)) continue;
    unsigned dynamic = lhs.allocation.getType().getDynamicDimIndex(axis);
    if (lhs.allocation.getDynamicSizes()[dynamic] == rhs.allocation.getDynamicSizes()[dynamic]) continue;
    AffineExpr loop = maps.back().getResult(axis);
    if (llvm::none_of(ArrayRef(maps).drop_back(), [&](AffineMap map) {
          return llvm::is_contained(map.getResults(), loop);
        })) return false;
  }
  for (Value input : lhs.producer.getDpsInputs())
    if (isa<MemRefType>(input.getType()) && !physical.mayReadAt(input, lhs.producer, rhs.producer)) return false;
  llvm::DenseMap<Value, Value> pairs;
  pairs[lhs.allocation] = rhs.allocation;
  return OperationEquivalence::isEquivalentTo(lhs.producer, rhs.producer,
      [&](Value left, Value right) { return success(left == right || pairs.lookup(left) == right); },
      [&](Value left, Value right) { pairs[left] = right; }, OperationEquivalence::IgnoreLocations);
}

}

LogicalResult reusePreparedInputs(func::FuncOp function) {
  groupScopedInputs(function);
  bool changed;
  do {
    changed = false;
    SmallVector<PreparedInput> supplies;
    function.walk([&](linalg::GenericOp operation) {
      if (auto supply = preparedInput(operation)) supplies.push_back(*supply);
    });
    PhysicalProgramAnalysis physical(function);
    for (unsigned i = 0; i < supplies.size() && !changed; ++i) {
      for (unsigned j = i + 1; j < supplies.size(); ++j) {
        auto &first = supplies[i], &second = supplies[j];
        if (!equivalent(first, second, physical)) continue;
        // The shared representation remains owned by this lexical scope until
        // both consumer groups finish, before either implementation expands.
        if (first.end->isBeforeInBlock(second.end)) first.end->moveAfter(second.end);
        second.producer.erase();
        second.end.erase();
        second.allocation.getResult().replaceAllUsesWith(first.allocation.getResult());
        second.allocation.erase();
        changed = true;
        break;
      }
    }
  } while (changed);
  return success();
}

}

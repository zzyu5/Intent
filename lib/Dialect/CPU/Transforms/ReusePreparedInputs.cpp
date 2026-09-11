#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
namespace intent::cpu {
namespace {

struct PreparedInput {
  linalg::GenericOp producer;
  memref::AllocOp allocation;
  memref::DeallocOp end;
  ImplementationAttr consumer;
};

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
  PreparedInput result{producer, allocation, {}, {}};
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
  for (Value input : lhs.producer.getInputs())
    if (isa<MemRefType>(input.getType()) && !physical.mayReadAt(input, lhs.producer, rhs.producer)) return false;
  llvm::DenseMap<Value, Value> pairs;
  pairs[lhs.allocation] = rhs.allocation;
  return OperationEquivalence::isEquivalentTo(lhs.producer, rhs.producer,
      [&](Value left, Value right) { return success(left == right || pairs.lookup(left) == right); },
      [&](Value left, Value right) { pairs[left] = right; }, OperationEquivalence::IgnoreLocations);
}

}

LogicalResult reusePreparedInputs(func::FuncOp function) {
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

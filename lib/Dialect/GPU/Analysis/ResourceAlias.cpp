#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isResource(Value value) {
  return value && isa<ViewType, BufferType>(value.getType());
}

bool isAllocationRoot(Value value) {
  if (value.getDefiningOp<BufferOp>())
    return true;
  auto argument = dyn_cast<BlockArgument>(value);
  auto function = argument
                      ? dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp())
                      : func::FuncOp();
  if (!function || argument.getOwner() != &function.front())
    return false;
  auto binding = getArgumentBinding(argument);
  return binding && (binding.getKind() == ArgumentKind::Workspace ||
                     (binding.getKind() == ArgumentKind::Public &&
                      bool(getPublicView(argument))));
}

// RegionBranch describes the actual forwarded operands, including zero-trip
// loop results and both while regions. Produced arguments have no incoming
// resource and therefore cannot establish an allocation identity.
bool regionPredecessors(RegionBranchOpInterface owner, Value value,
                        SmallVectorImpl<Value> &incoming) {
  auto targetsValue = [&](RegionSuccessor successor) {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return successor.getSuccessor() == argument.getOwner()->getParent();
    return successor.isParent();
  };
  auto append = [&](RegionSuccessor successor, ValueRange operands) {
    if (!targetsValue(successor))
      return true;
    auto inputs = successor.getSuccessorInputs();
    auto found = llvm::find(inputs, value);
    if (found == inputs.end() || inputs.size() != operands.size())
      return false;
    incoming.push_back(operands[found - inputs.begin()]);
    return true;
  };
  SmallVector<RegionSuccessor> successors;
  owner.getSuccessorRegions(RegionBranchPoint::parent(), successors);
  for (RegionSuccessor successor : successors)
    if (!append(successor, owner.getEntrySuccessorOperands(successor)))
      return false;
  for (Region &region : owner->getRegions()) {
    successors.clear();
    owner.getSuccessorRegions(&region, successors);
    if (!llvm::any_of(successors, targetsValue))
      continue;
    for (Block &block : region) {
      if (block.empty())
        return false;
      Operation *terminator = block.getTerminator();
      if (terminator->getNumSuccessors())
        continue;
      auto branch = dyn_cast<RegionBranchTerminatorOpInterface>(terminator);
      if (!branch)
        return false;
      SmallVector<RegionSuccessor> outgoing;
      SmallVector<Attribute> unknownOperands(terminator->getNumOperands());
      branch.getSuccessorRegions(unknownOperands, outgoing);
      for (RegionSuccessor successor : outgoing)
        if (!append(successor, branch.getSuccessorOperands(successor)))
          return false;
    }
  }
  return !incoming.empty();
}

bool predecessors(Value value, SmallVectorImpl<Value> &incoming) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Block *block = argument.getOwner();
    bool complete = true;
    if (block->isEntryBlock()) {
      auto owner = dyn_cast<RegionBranchOpInterface>(block->getParentOp());
      complete = owner && regionPredecessors(owner, value, incoming);
    }
    for (Block *predecessor : block->getPredecessors()) {
      auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        return false;
      for (unsigned index = 0; index < branch->getNumSuccessors(); ++index) {
        if (branch->getSuccessor(index) != block)
          continue;
        auto operands = branch.getSuccessorOperands(index);
        unsigned position = argument.getArgNumber();
        if (position >= operands.size() || operands.isOperandProduced(position))
          return false;
        incoming.push_back(operands[position]);
      }
    }
    return complete && !incoming.empty();
  }
  Operation *definition = value.getDefiningOp();
  if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(definition)) {
    incoming.push_back(view.getViewSource());
    return true;
  }
  if (auto select = dyn_cast_or_null<SelectLikeOpInterface>(definition)) {
    incoming.append({select.getTrueValue(), select.getFalseValue()});
    return true;
  }
  auto owner = dyn_cast_or_null<RegionBranchOpInterface>(definition);
  return owner && regionPredecessors(owner, value, incoming);
}

bool disjointRoots(Value lhs, Value rhs) {
  if (lhs == rhs)
    return false;
  // Every BufferOp and hidden workspace owns a fresh allocation. A different
  // SSA handle alone is insufficient; both inputs here are proven roots.
  if (lhs.getDefiningOp<BufferOp>() || rhs.getDefiningOp<BufferOp>() ||
      isInvocationWorkspace(lhs) || isInvocationWorkspace(rhs))
    return true;
  auto left = getPublicView(lhs);
  auto right = getPublicView(rhs);
  return left && right && (left.getConstraints().getNoalias() ||
                           right.getConstraints().getNoalias());
}

} // namespace

const ResourceAliasAnalysis::Roots &ResourceAliasAnalysis::roots(Value value) {
  auto found = cache.find(value);
  if (found != cache.end())
    return found->second;
  Roots result;
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (!isResource(current)) {
      result.complete = false;
      continue;
    }
    if (isAllocationRoot(current)) {
      result.allocations.push_back(current);
      continue;
    }
    SmallVector<Value> incoming;
    if (!predecessors(current, incoming))
      result.complete = false;
    llvm::append_range(pending, incoming);
  }
  result.complete &= !result.allocations.empty();
  return cache.try_emplace(value, std::move(result)).first->second;
}

AliasResult ResourceAliasAnalysis::alias(Value lhs, Value rhs) {
  if (!isResource(lhs) || !isResource(rhs))
    return AliasResult::MayAlias;
  if (lhs == rhs)
    return AliasResult::MustAlias;
  // Querying the second value may grow the DenseMap; do not keep references
  // into it across insertion.
  roots(lhs);
  roots(rhs);
  const Roots &left = cache.find(lhs)->second;
  const Roots &right = cache.find(rhs)->second;
  if (!left.complete || !right.complete)
    return AliasResult::MayAlias;
  for (Value a : left.allocations)
    for (Value b : right.allocations)
      if (!disjointRoots(a, b))
        return AliasResult::MayAlias;
  return AliasResult::NoAlias;
}

} // namespace intent::gpu

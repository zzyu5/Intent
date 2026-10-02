#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Analysis/ControlFlow.h"
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

bool predecessors(Value value, SmallVectorImpl<Value> &incoming) {
  Operation *definition = value.getDefiningOp();
  if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(definition)) {
    incoming.push_back(view.getViewSource());
    return true;
  }
  if (auto select = dyn_cast_or_null<SelectLikeOpInterface>(definition)) {
    incoming.append({select.getTrueValue(), select.getFalseValue()});
    return true;
  }
  auto edges = queryControlFlowIncoming(value);
  for (const ControlFlowEdge &edge : edges.edges)
    if (edge.operand) incoming.push_back(edge.operand->get());
  return edges.complete && !incoming.empty();
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

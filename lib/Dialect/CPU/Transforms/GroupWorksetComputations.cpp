#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Utilities.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool sameExtent(Value first, Value second) {
  return haveEqualExtents(ValueBoundsConstraintSet::Variable(first),
                          ValueBoundsConstraintSet::Variable(second));
}

bool workset(scf::ParallelOp loop) {
  return loop && loop.getNumLoops() == 1 && !loop.getNumResults() &&
      matchPattern(loop.getLowerBound()[0], m_Zero()) && matchPattern(loop.getStep()[0], m_One());
}

bool leadingView(memref::SubViewOp view) {
  return view.getSourceType().getRank() && !view.getDroppedDims()[0] &&
      getConstantIntValue(view.getMixedOffsets()[0]) == 0 &&
      getConstantIntValue(view.getMixedStrides()[0]) == 1;
}

bool injectiveSlice(Value memory) {
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) return injectiveSlice(cast.getSource());
  if (auto view = memory.getDefiningOp<memref::SubViewOp>())
    return llvm::all_of(view.getMixedStrides(), [](OpFoldResult stride) {
      return getConstantIntValue(stride) == 1;
    }) && injectiveSlice(view.getSource());
  return cast<MemRefType>(memory.getType()).getLayout().isIdentity();
}

// Preserve the original leading coordinate through views before a row slice.
Value leadingRoot(Value memory) {
  while (true) {
    if (auto cast = memory.getDefiningOp<memref::CastOp>()) memory = cast.getSource();
    else if (auto view = memory.getDefiningOp<memref::SubViewOp>(); view && leadingView(view)) memory = view.getSource();
    else return memory;
  }
}

bool owned(Value memory, Value coordinate, Value root) {
  if (auto cast = memory.getDefiningOp<memref::CastOp>()) return owned(cast.getSource(), coordinate, root);
  auto view = memory.getDefiningOp<memref::SubViewOp>();
  if (!view) return false;
  if (owned(view.getSource(), coordinate, root)) return true;
  return view.getSourceType().getRank() && leadingRoot(view.getSource()) == root &&
      view.getMixedOffsets()[0] == OpFoldResult(coordinate) &&
      getConstantIntValue(view.getMixedSizes()[0]) == 1 &&
      getConstantIntValue(view.getMixedStrides()[0]) == 1;
}

struct Access {
  Value root;
  bool write, row;
};

FailureOr<SmallVector<Access>> accesses(scf::ParallelOp loop, StorageAnalysis &storage) {
  SmallVector<Access> result;
  Value coordinate = loop.getInductionVars()[0];
  auto effects = storage.effects(loop);
  if (!effects.complete || effects.ordered) return failure();
  for (const StorageEffect &entry : effects.entries) {
    const auto &effect = entry.effect;
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
    Value memory = effect.getValue();
    if (!memory || !isa<MemRefType>(memory.getType())) return failure();
    Value root = storage.uniqueOrigin(memory);
    if (!root) return failure();
    if (isa<MemoryEffects::Free>(effect.getEffect())) {
      auto allocation = root.getDefiningOp<memref::AllocOp>();
      if (!allocation || !loop->isAncestor(allocation)) return failure();
      continue;
    }
    if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
      return failure();
    ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(entry.operation))
      indices = load.getIndices();
    else if (auto store = dyn_cast<memref::StoreOp>(entry.operation))
      indices = store.getIndices();
    bool row = owned(memory, coordinate, root) ||
        (!indices.empty() && indices[0] == coordinate && leadingRoot(memory) == root);
    result.push_back({root, isa<MemoryEffects::Write>(effect.getEffect()), row});
  }
  return result;
}

bool fuse(scf::ParallelOp first, scf::ParallelOp second, ArrayRef<Operation *> between) {
  if (!workset(second) || !sameExtent(first.getUpperBound()[0], second.getUpperBound()[0])) return false;
  auto function = first->getParentOfType<func::FuncOp>();
  StorageAnalysis storage(function);
  auto lhs = accesses(first, storage), rhs = accesses(second, storage);
  if (failed(lhs) || failed(rhs)) return false;
  for (const auto &left : *lhs)
    for (const auto &right : *rhs) {
      if (!left.write && !right.write) continue;
      if (left.root == right.root) {
        auto type = dyn_cast<MemRefType>(left.root.getType());
        if (!left.row || !right.row || !type || !type.getLayout().isIdentity()) return false;
        continue;
      }
      if (!storage.disjoint(left.root, right.root)) return false;
    }
  DominanceInfo dominance(function);
  llvm::SmallPtrSet<Operation *, 16> movable;
  for (Operation *operation : between) {
    if (isa<memref::DeallocOp>(operation)) continue;
    if (operation->getNumRegions() ||
        (!isa<memref::AllocOp>(operation) && !isMemoryEffectFree(operation))) return false;
    if (llvm::any_of(operation->getOperands(), [&](Value value) {
          return !dominance.dominates(value, first) && !movable.contains(value.getDefiningOp());
        })) return false;
    movable.insert(operation);
  }
  for (Operation *operation : between)
    if (!isa<memref::DeallocOp>(operation)) operation->moveBefore(first);
  OpBuilder b(first.getBody()->getTerminator());
  IRMapping mapping;
  mapping.map(second.getInductionVars()[0], first.getInductionVars()[0]);
  for (Operation &operation : second.getBody()->without_terminator()) b.clone(operation, mapping);
  second.erase();
  return true;
}

void localize(memref::AllocOp allocation, scf::ParallelOp loop) {
  auto type = allocation.getType();
  if (!type.getRank() || !type.getLayout().isIdentity()) return;
  auto size = queryExtentValue(allocation, 0);
  if (!size) return;
  if (auto value = dyn_cast<Value>(*size)) {
    if (!sameExtent(value, loop.getUpperBound()[0])) return;
  } else if (getConstantIntValue(*size) != getConstantIntValue(loop.getUpperBound()[0])) return;
  Value coordinate = loop.getInductionVars()[0];
  DominanceInfo dominance(loop->getParentOfType<func::FuncOp>());
  SmallVector<Value> memories{allocation.getResult()};
  SmallVector<Operation *> views, direct;
  SmallVector<memref::SubViewOp> rows;
  SmallVector<std::pair<memref::DimOp, OpFoldResult>> dimensions;
  memref::DeallocOp deallocation;
  for (unsigned number = 0; number < memories.size(); ++number) {
    Value memory = memories[number];
    for (Operation *user : memory.getUsers()) {
      if (auto dim = dyn_cast<memref::DimOp>(user)) {
        auto axis = getConstantIntValue(dim.getIndex());
        auto value = axis ? queryExtentValue(memory, *axis) : std::nullopt;
        if (!value) return;
        if (auto size = dyn_cast<Value>(*value))
          if (!dominance.dominates(size, dim)) return;
        dimensions.emplace_back(dim, *value);
        continue;
      }
      if (auto free = dyn_cast<memref::DeallocOp>(user)) {
        if (deallocation || memory != allocation || free->getBlock() != loop->getBlock() ||
            !loop->isBeforeInBlock(free)) return;
        deallocation = free;
        continue;
      }
      if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        if (view.getMixedOffsets()[0] == OpFoldResult(coordinate) &&
            getConstantIntValue(view.getMixedSizes()[0]) == 1 &&
            getConstantIntValue(view.getMixedStrides()[0]) == 1 && loop->isAncestor(view)) {
          rows.push_back(view);
          continue;
        }
        if (!leadingView(view)) return;
      } else if (!isa<memref::CastOp>(user)) {
        ValueRange indices;
        if (auto load = dyn_cast<memref::LoadOp>(user)) indices = load.getIndices();
        else if (auto store = dyn_cast<memref::StoreOp>(user)) indices = store.getIndices();
        else return;
        if (!loop->isAncestor(user) || indices.empty() || indices[0] != coordinate) return;
        direct.push_back(user);
        continue;
      }
      if (llvm::any_of(user->getOperands().drop_front(), [&](Value value) {
            return !dominance.dominates(value, loop);
          })) return;
      views.push_back(user);
      memories.push_back(user->getResult(0));
    }
  }
  if (!deallocation || (rows.empty() && direct.empty())) return;
  // Preserve logical dimensions queried outside the new physical slice.
  IRMapping resolved;
  for (auto [dim, value] : dimensions) {
    OpBuilder b(dim);
    Value replacement = getValueOrCreateConstantIndexOp(b, dim.getLoc(), value);
    resolved.map(dim.getResult(), replacement);
  }
  for (auto [dim, value] : dimensions) {
    Value replacement = resolved.lookup(dim.getResult());
    while (resolved.contains(replacement)) replacement = resolved.lookup(replacement);
    dim.getResult().replaceAllUsesWith(replacement);
  }
  for (auto [dim, value] : dimensions) dim.erase();
  OpBuilder b(loop.getBody(), loop.getBody()->begin());
  Location loc = allocation.getLoc();
  SmallVector<int64_t> shape(type.getShape());
  shape[0] = 1;
  SmallVector<Value> dynamic;
  unsigned number = 0;
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    if (type.isDynamicDim(axis)) {
      Value value = allocation.getDynamicSizes()[number++];
      if (axis) dynamic.push_back(value);
    }
  auto local = b.create<memref::AllocOp>(loc,
      MemRefType::get(shape, type.getElementType(), type.getLayout(), type.getMemorySpace()), dynamic);
  local->setDiscardableAttrs(llvm::to_vector(allocation->getDiscardableAttrs()));
  if (auto alignment = allocation.getAlignmentAttr()) local.setAlignmentAttr(alignment);
  IRMapping mapping;
  mapping.map(allocation.getResult(), local.getResult());
  for (Operation *operation : views) {
    Value source = mapping.lookup(operation->getOperand(0));
    Value replacement;
    if (auto view = dyn_cast<memref::SubViewOp>(operation)) {
      auto sizes = view.getMixedSizes();
      sizes[0] = b.getIndexAttr(1);
      SmallVector<int64_t> shape(view.getType().getShape());
      shape[0] = 1;
      auto result = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          shape, cast<MemRefType>(source.getType()), view.getMixedOffsets(), sizes, view.getMixedStrides()));
      replacement = b.create<memref::SubViewOp>(loc, result, source, view.getMixedOffsets(), sizes, view.getMixedStrides());
    } else {
      auto old = cast<MemRefType>(operation->getResult(0).getType());
      SmallVector<int64_t> shape(old.getShape());
      shape[0] = 1;
      replacement = b.create<memref::CastOp>(loc,
          MemRefType::get(shape, old.getElementType(), old.getLayout(), old.getMemorySpace()), source);
    }
    mapping.map(operation->getResult(0), replacement);
  }
  Value zero = index(b, loc, 0);
  for (auto view : rows) {
    OpBuilder at(view);
    auto offsets = view.getMixedOffsets();
    offsets[0] = at.getIndexAttr(0);
    Value source = mapping.lookup(view.getSource());
    auto result = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        view.getType().getShape(), cast<MemRefType>(source.getType()), offsets,
        view.getMixedSizes(), view.getMixedStrides()));
    Value replacement = at.create<memref::SubViewOp>(view.getLoc(), result, source,
        offsets, view.getMixedSizes(), view.getMixedStrides());
    if (replacement.getType() != view.getType()) replacement = at.create<memref::CastOp>(view.getLoc(), view.getType(), replacement);
    view.getResult().replaceAllUsesWith(replacement);
    view.erase();
  }
  for (Operation *operation : direct) {
    bool load = isa<memref::LoadOp>(operation);
    unsigned operand = load ? 0 : 1;
    operation->setOperand(operand, mapping.lookup(operation->getOperand(operand)));
    operation->setOperand(operand + 1, zero);
  }
  for (Operation *operation : llvm::reverse(views)) operation->erase();
  b.setInsertionPoint(loop.getBody()->getTerminator());
  b.create<memref::DeallocOp>(loc, local);
  deallocation.erase();
  allocation.erase();
}

}

LogicalResult groupWorksetComputations(func::FuncOp function, const ImplementationRegistry &implementations) {
  SmallVector<Value> extents;
  for (auto loop : function.front().getOps<scf::ParallelOp>())
    if (workset(loop)) extents.push_back(loop.getUpperBound()[0]);
  if (extents.empty()) return success();
  if (failed(exposeStructuredWorksets(function, implementations, extents))) return failure();
  SmallVector<Operation *> transfers;
  for (Operation &operation : function.front())
    if (isa<linalg::FillOp, memref::CopyOp>(operation)) transfers.push_back(&operation);
  for (Operation *operation : transfers) {
    StorageAnalysis storage(function);
    Value output, input;
    if (auto fill = dyn_cast<linalg::FillOp>(operation)) {
      if (fill.getOutputs().size() != 1 || fill.getNumResults()) continue;
      output = fill.getOutputs()[0];
    } else {
      auto copy = cast<memref::CopyOp>(operation);
      input = copy.getSource(); output = copy.getTarget();
      if (!storage.disjoint(input, output)) continue;
    }
    auto type = cast<MemRefType>(output.getType());
    if (!type.getRank() || !injectiveSlice(output)) continue;
    OpBuilder b(operation);
    Location loc = operation->getLoc();
    Value extent = b.createOrFold<memref::DimOp>(loc, output, 0);
    if (!llvm::any_of(extents, [&](Value candidate) { return sameExtent(extent, candidate); })) continue;
    if (input && !sameExtent(extent, b.createOrFold<memref::DimOp>(loc, input, 0))) continue;
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    auto loop = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{extent}, ValueRange{one});
    b.setInsertionPointToStart(loop.getBody());
    auto slice = [&](Value memory) -> Value {
      auto type = cast<MemRefType>(memory.getType());
      SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)), sizes;
      offsets[0] = loop.getInductionVars()[0];
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        sizes.push_back(axis ? OpFoldResult(b.createOrFold<memref::DimOp>(loc, memory, axis)) : OpFoldResult(b.getIndexAttr(1)));
      return b.create<memref::SubViewOp>(loc, memory, offsets, sizes,
          SmallVector<OpFoldResult>(type.getRank(), b.getIndexAttr(1)));
    };
    if (input) b.create<memref::CopyOp>(loc, slice(input), slice(output));
    else b.create<linalg::FillOp>(loc, cast<linalg::FillOp>(operation).getInputs(), ValueRange{slice(output)});
    operation->erase();
  }
  if (failed(applyPatternsGreedily(function, RewritePatternSet(function.getContext())))) return failure();
  SmallVector<scf::ParallelOp> groups;
  for (Operation *operation = &function.front().front(); operation; ) {
    auto first = dyn_cast<scf::ParallelOp>(operation);
    if (!workset(first)) { operation = operation->getNextNode(); continue; }
    bool changed = false;
    while (true) {
      SmallVector<Operation *> between;
      Operation *next = first->getNextNode();
      while (next && (isa<memref::AllocOp, memref::DeallocOp>(next) ||
                      (!next->getNumRegions() && isMemoryEffectFree(next)))) {
        between.push_back(next);
        next = next->getNextNode();
      }
      auto second = dyn_cast_or_null<scf::ParallelOp>(next);
      if (!second || !fuse(first, second, between)) break;
      changed = true;
    }
    if (changed) groups.push_back(first);
    operation = first->getNextNode();
  }
  for (auto loop : groups) {
    SmallVector<memref::AllocOp> allocations(function.front().getOps<memref::AllocOp>());
    for (auto allocation : allocations) localize(allocation, loop);
  }
  return success();
}

}

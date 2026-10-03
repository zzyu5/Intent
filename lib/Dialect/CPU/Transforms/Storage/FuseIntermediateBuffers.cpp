#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Structure/LoopBuilders.h"
#include "Intent/Dialect/CPU/Transforms/Structure/ProducerReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool sameStaticLayout(MemRefType source, MemRefType destination) {
  if (source.getShape() != destination.getShape() ||
      source.getElementType() != destination.getElementType() ||
      source.getMemorySpace() != destination.getMemorySpace()) return false;
  SmallVector<int64_t> sourceStrides, destinationStrides;
  int64_t sourceOffset, destinationOffset;
  if (failed(source.getStridesAndOffset(sourceStrides, sourceOffset)) ||
      failed(destination.getStridesAndOffset(destinationStrides, destinationOffset)) ||
      ShapedType::isDynamic(sourceOffset) || ShapedType::isDynamic(destinationOffset) ||
      sourceOffset != destinationOffset || sourceStrides != destinationStrides ||
      llvm::any_of(sourceStrides, ShapedType::isDynamic)) return false;
  return memref::CastOp::areCastCompatible(TypeRange{source}, TypeRange{destination});
}

void forwardDestinations(func::FuncOp function) {
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    Value source = copy.getSource();
    Operation *reshape = source.getDefiningOp();
    if (unitReshapeAxes(reshape)) {
      if (!source.hasOneUse() || reshape->getBlock() != copy->getBlock()) continue;
      source = reshape->getOperand(0);
    } else reshape = nullptr;
    auto allocation = source.getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != copy->getBlock()) continue;
    Value target = copy.getTarget();
    MemRefType replacementType = cast<MemRefType>(copy.getTarget().getType());
    SmallVector<ReassociationIndices> reassociation;
    if (auto collapse = dyn_cast_or_null<memref::CollapseShapeOp>(reshape)) {
      reassociation = collapse.getReassociationIndices();
      auto expanded = memref::ExpandShapeOp::computeExpandedType(
          replacementType, allocation.getType().getShape(), reassociation);
      if (failed(expanded)) continue;
      replacementType = *expanded;
    } else if (auto expand = dyn_cast_or_null<memref::ExpandShapeOp>(reshape)) {
      reassociation = expand.getReassociationIndices();
      if (!memref::CollapseShapeOp::isGuaranteedCollapsible(replacementType, reassociation)) continue;
      replacementType = memref::CollapseShapeOp::computeCollapsedType(replacementType, reassociation);
    }
    if (allocation.getType().getRank() != replacementType.getRank()) continue;
    if (reshape && (replacementType.getShape() != allocation.getType().getShape() ||
                    replacementType.getMemorySpace() != allocation.getType().getMemorySpace())) continue;
    bool castReplacement = replacementType != allocation.getType() &&
        sameStaticLayout(replacementType, allocation.getType());
    StorageAnalysis storage(function);
    Value targetRoot = storage.uniqueOrigin(target);
    if (!targetRoot) continue;
    auto external = storage.externalView(target);
    if (!targetRoot.getDefiningOp<memref::AllocOp>() &&
        (!external || external.getAccess() != 1)) continue;
    if (targetRoot == allocation.getResult()) continue;
    auto targetOp = target.getDefiningOp();
    DominanceInfo dominance(function);
    if (targetOp && !dominance.dominates(targetOp, allocation)) {
      if (targetOp->getBlock() != allocation->getBlock() ||
          (!isMemoryEffectFree(targetOp) && !isa<memref::AllocOp>(targetOp)) ||
          llvm::any_of(targetOp->getOperands(), [&](Value value) {
            return !dominance.dominates(value, allocation);
          })) continue;
    }
    auto lifetime = storage.lifetime(allocation);
    if (!lifetime || !lifetime->aliases.complete) continue;
    bool legal = true;
    Operation *lastUse = copy;
    for (Operation *user : lifetime->aliases.users) {
      if (user == reshape || user == lifetime->end) continue;
      bool view = isStorageAliasOperation(user);
      // Retained views encode the private allocation's representation, including
      // reassociation and extracted metadata. Preserve their input type with a
      // proven static-layout cast; unknown strides cannot establish equivalence.
      // The copy-source reshape is removed separately.
      if (view && replacementType != allocation.getType() && !castReplacement) {
        legal = false;
        break;
      }
      if (!view && !isa<memref::CopyOp, memref::DimOp, memref::LoadOp, memref::StoreOp,
                       linalg::LinalgOp, ReduceOp, ScanOp, HistogramOp, QuantizedDotOp>(user)) {
        legal = false;
        break;
      }
      Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor) { legal = false; break; }
      if (ancestor != copy && copy->isBeforeInBlock(ancestor)) {
        if (!view && !isa<memref::LoadOp, memref::DimOp>(user)) { legal = false; break; }
        if (lastUse->isBeforeInBlock(ancestor)) lastUse = ancestor;
      }
    }
    // Forwarding moves the destination writes earlier. Its previous contents
    // must remain unobserved before the copy and stable through all later reads.
    for (Operation *between = allocation->getNextNode(); legal && between != lastUse->getNextNode();
         between = between->getNextNode()) {
      if (between == copy) continue;
      auto effects = storage.effects(between);
      if (!effects.complete || effects.ordered) {
        legal = false;
        break;
      }
      for (const StorageEffect &entry : effects.entries) {
        const auto &effect = entry.effect;
        if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
        if (!effect.getValue() || !storage.disjoint(effect.getValue(), target))
          legal = false;
      }
    }
    if (!legal) continue;
    if (targetOp && !dominance.dominates(targetOp, allocation)) targetOp->moveBefore(allocation);
    if (reshape) {
      OpBuilder builder(allocation);
      if (isa<memref::CollapseShapeOp>(reshape)) {
        SmallVector<OpFoldResult> shape;
        unsigned dynamicAxis = 0;
        for (int64_t size : allocation.getType().getShape())
          shape.push_back(ShapedType::isDynamic(size)
              ? OpFoldResult(allocation.getDynamicSizes()[dynamicAxis++])
              : OpFoldResult(builder.getIndexAttr(size)));
        target = builder.create<memref::ExpandShapeOp>(allocation.getLoc(), replacementType,
            target, reassociation, shape);
      } else {
        target = builder.create<memref::CollapseShapeOp>(allocation.getLoc(), replacementType,
            target, reassociation);
      }
    }
    if (castReplacement) {
      OpBuilder builder(allocation);
      target = builder.create<memref::CastOp>(allocation.getLoc(), allocation.getType(), target);
    }
    lifetime->end.erase();
    copy.erase();
    if (reshape) reshape->erase();
    allocation.getResult().replaceAllUsesWith(target);
    allocation.erase();
  }
}

bool dependsOn(Value value, Value coordinate, Operation *root) {
  if (value == coordinate) return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || !root->isAncestor(operation)) return false;
  // Captures and yields of an intact control region contribute to dependence,
  // even when its condition is independent of the vectorized coordinate.
  return operation->walk([&](Operation *nested) -> WalkResult {
    for (Value operand : nested->getOperands())
      if (dependsOn(operand, coordinate, root)) return WalkResult::interrupt();
    return WalkResult::advance();
  }).wasInterrupted();
}

bool canReplayVector(Value value, Value coordinate, Operation *root) {
  if (value == coordinate || !dependsOn(value, coordinate, root)) return true;
  Operation *operation = value.getDefiningOp();
  if (auto load = dyn_cast<memref::LoadOp>(operation)) {
    auto type = load.getMemRefType();
    SmallVector<int64_t> strides;
    int64_t offset;
    return type.getRank() && !type.getElementType().isInteger(1) &&
        succeeded(type.getStridesAndOffset(strides, offset)) && strides.back() == 1 &&
        !dependsOn(load.getMemref(), coordinate, root) &&
        load.getIndices().back() == coordinate &&
        llvm::none_of(load.getIndices().drop_back(), [&](Value index) {
          return dependsOn(index, coordinate, root);
        });
  }
  return operation && !operation->getNumRegions() && operation->getNumResults() == 1 &&
      operation->hasTrait<OpTrait::Elementwise>() &&
      llvm::all_of(operation->getOperandTypes(), [](Type type) {
        return isa<FloatType, IntegerType, IndexType>(type);
      }) && llvm::all_of(operation->getOperands(), [&](Value operand) {
        return canReplayVector(operand, coordinate, root);
      });
}

Value replayVector(const ProducerReplay &payload, Value value, Value coordinate,
                   Operation *root, int64_t width,
                   OpBuilder &builder, IRMapping &scalars, IRMapping &vectors) {
  if (vectors.contains(value)) return vectors.lookup(value);
  auto type = VectorType::get({width}, value.getType());
  Location loc = root->getLoc();
  auto scalar = [&](Value value) {
    auto replayed = materializeProducerValue(payload, value, builder, scalars);
    assert(succeeded(replayed) && "vector adapter requires a proven scalar payload");
    return *replayed;
  };
  Value result;
  if (value == coordinate) {
    Value start = builder.create<vector::BroadcastOp>(loc, type, scalars.lookup(value));
    result = builder.create<arith::AddIOp>(loc, start, builder.create<vector::StepOp>(loc, type));
  } else if (!dependsOn(value, coordinate, root)) {
    result = builder.create<vector::BroadcastOp>(loc, type, scalar(value));
  } else if (auto load = value.getDefiningOp<memref::LoadOp>()) {
    SmallVector<Value> indices;
    for (Value index : load.getIndices()) indices.push_back(scalar(index));
    result = builder.create<vector::LoadOp>(loc, type,
        scalar(load.getMemref()), indices);
  } else {
    Operation *operation = value.getDefiningOp();
    IRMapping mapping;
    for (Value operand : operation->getOperands())
      mapping.map(operand, replayVector(payload, operand, coordinate, root, width, builder, scalars, vectors));
    Operation *cloned = builder.clone(*operation, mapping);
    cloned->getResult(0).setType(type);
    result = cloned->getResult(0);
  }
  vectors.map(value, result);
  return result;
}

bool fullVectorRead(vector::LoadOp load, memref::AllocOp allocation) {
  auto type = allocation.getType();
  auto vector = load.getVectorType();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!type.getRank() || !type.getLayout().isIdentity() || vector.getRank() != 1 || vector.isScalable() ||
      failed(type.getStridesAndOffset(strides, offset)) || strides.back() != 1) return false;
  unsigned dynamicAxis = 0;
  for (int64_t axis = 0; axis + 1 < type.getRank(); ++axis) {
    int64_t size = type.getDimSize(axis);
    Value extent = type.isDynamicDim(axis) ? allocation.getDynamicSizes()[dynamicAxis++] : Value{};
    Value position = load.getIndices()[axis];
    if (auto constant = getConstantIntValue(position)) {
      if (extent || *constant < 0 || *constant >= size) return false;
      continue;
    }
    auto coordinate = dyn_cast<BlockArgument>(position);
    auto traversal = coordinate ? dyn_cast<scf::ForOp>(coordinate.getOwner()->getParentOp()) : scf::ForOp{};
    auto step = traversal ? getConstantIntValue(traversal.getStep()) : std::nullopt;
    auto begin = traversal ? getConstantIntValue(traversal.getLowerBound()) : std::nullopt;
    if (!traversal || position != traversal.getInductionVar() || !step || *step <= 0 ||
        !begin || *begin < 0) return false;
    auto end = getConstantIntValue(traversal.getUpperBound());
    if (extent ? traversal.getUpperBound() != extent : !end || *end > size) return false;
  }
  int64_t width = vector.getDimSize(0);
  int64_t staticExtent = type.getShape().back();
  Value extent;
  if (ShapedType::isDynamic(staticExtent)) extent = allocation.getDynamicSizes().back();
  auto isExtent = [&](Value value) {
    auto constant = getConstantIntValue(value);
    return extent ? value == extent : constant && *constant == staticExtent;
  };
  Value begin = load.getIndices().back();
  if (auto constant = getConstantIntValue(begin))
    return !extent && *constant >= 0 && *constant <= staticExtent && width <= staticExtent - *constant;
  auto coordinate = dyn_cast<BlockArgument>(begin);
  auto traversal = coordinate ? dyn_cast<scf::ForOp>(coordinate.getOwner()->getParentOp()) : scf::ForOp{};
  auto step = traversal ? getConstantIntValue(traversal.getStep()) : std::nullopt;
  if (!traversal || begin != traversal.getInductionVar() || !step || *step != width ||
      !matchPattern(traversal.getLowerBound(), m_Zero())) return false;
  Value end = traversal.getUpperBound();
  if (auto constant = getConstantIntValue(end))
    return !extent && *constant >= 0 && *constant <= staticExtent && *constant % width == 0;
  // Full blocks use either n - n % width or (n / width) * width.
  // A plain vector.load does not itself promise that its entire slice is in bounds.
  if (auto subtract = end.getDefiningOp<arith::SubIOp>()) {
    auto remainder = subtract.getRhs().getDefiningOp<arith::RemSIOp>();
    return isExtent(subtract.getLhs()) && remainder && isExtent(remainder.getLhs()) &&
        getConstantIntValue(remainder.getRhs()) == width;
  }
  if (auto product = end.getDefiningOp<arith::MulIOp>()) {
    Value quotient;
    if (getConstantIntValue(product.getLhs()) == width) quotient = product.getRhs();
    else if (getConstantIntValue(product.getRhs()) == width) quotient = product.getLhs();
    auto division = quotient ? quotient.getDefiningOp<arith::DivSIOp>() : arith::DivSIOp{};
    return division && isExtent(division.getLhs()) && getConstantIntValue(division.getRhs()) == width;
  }
  return false;
}

bool fuse(memref::AllocOp allocation) {
  memref::StoreOp store;
  SmallVector<Operation *> loads;
  SmallVector<memref::DeallocOp> deallocations;
  for (Operation *user : allocation->getUsers()) {
    if (auto write = dyn_cast<memref::StoreOp>(user)) {
      if (store) return false;
      store = write;
    } else if (auto read = dyn_cast<memref::LoadOp>(user)) {
      loads.push_back(read);
    } else if (auto read = dyn_cast<vector::LoadOp>(user)) {
      if (!fullVectorRead(read, allocation)) return false;
      loads.push_back(read);
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocations.push_back(dealloc);
    } else return false;
  }
  if (!store || loads.empty()) return false;
  llvm::SmallPtrSet<Operation *, 4> consumers;
  for (Operation *load : loads) {
    Operation *stage = allocation->getBlock()->findAncestorOpInBlock(*load);
    if (!stage) return false;
    consumers.insert(stage);
  }
  SmallVector<scf::ForOp> loops;
  Operation *root = store;
  while (root->getBlock() != allocation->getBlock()) {
    auto parent = dyn_cast<scf::ForOp>(root->getParentOp());
    if (!parent || parent.getNumResults() || !matchPattern(parent.getLowerBound(), m_Zero()) ||
        !matchPattern(parent.getStep(), m_One())) return false;
    loops.push_back(parent);
    root = parent;
  }
  std::reverse(loops.begin(), loops.end());
  if (root->getBlock() != allocation->getBlock() ||
      loops.size() != store.getIndices().size()) return false;
  unsigned dynamicAxis = 0;
  for (auto [axis, loop] : llvm::enumerate(loops)) {
    if (store.getIndices()[axis] != loop.getInductionVar()) return false;
    if (allocation.getType().isDynamicDim(axis)) {
      if (loop.getUpperBound() != allocation.getDynamicSizes()[dynamicAxis++]) return false;
    } else {
      auto extent = getConstantIntValue(loop.getUpperBound());
      if (!extent || *extent != allocation.getType().getDimSize(axis)) return false;
    }
  }
  auto function = allocation->getParentOfType<func::FuncOp>();
  StorageAnalysis storage(function);
  // Erasing the traversal also erases operations outside the replayed value
  // slice. Their effects must be known reads, apart from this exact store.
  auto effects = storage.effects(root);
  if (!effects.complete || effects.ordered) return false;
  for (const StorageEffect &entry : effects.entries)
    if (entry.operation != store && !isa<MemoryEffects::Read>(entry.effect.getEffect()))
      return false;
  SmallVector<Value> frontier;
  for (auto loop : loops) frontier.push_back(loop.getInductionVar());
  auto payload = analyzeProducerValue(store.getValue(), root, frontier, storage);
  if (failed(payload)) return false;
  for (Operation *load : loads)
    if (isa<vector::LoadOp>(load) && (loops.empty() ||
        !canReplayVector(store.getValue(), loops.back().getInductionVar(), root))) return false;
  if (consumers.size() != 1) {
    auto integer = [](Type type) { return isa<IndexType, IntegerType>(type); };
    if (!integer(store.getValue().getType()) || llvm::any_of(payload->nodes, [&](Operation *operation) {
          return !llvm::all_of(operation->getResultTypes(), integer);
        })) return false;
  }
  for (Operation *load : loads)
    if (!canReplayProducerAt(*payload, root, load, storage)) return false;
  for (Operation *load : loads) {
    IRMapping mapping;
    ValueRange indices = isa<memref::LoadOp>(load) ? cast<memref::LoadOp>(load).getIndices()
                                                 : cast<vector::LoadOp>(load).getIndices();
    for (auto [loop, coordinate] : llvm::zip(loops, indices))
      mapping.map(loop.getInductionVar(), coordinate);
    OpBuilder builder(load);
    Value replacement;
    if (auto vector = dyn_cast<vector::LoadOp>(load)) {
      IRMapping vectors;
      replacement = replayVector(*payload, store.getValue(), loops.back().getInductionVar(), root,
          vector.getVectorType().getDimSize(0), builder, mapping, vectors);
    } else {
      auto replayed = materializeProducerValue(*payload, store.getValue(), builder, mapping);
      assert(succeeded(replayed) && "buffer replay must bind the complete coordinate frontier");
      replacement = *replayed;
    }
    load->getResult(0).replaceAllUsesWith(replacement);
    load->erase();
  }
  root->erase();
  for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
  allocation.erase();
  return true;
}

}

LogicalResult fuseIntermediateBuffers(func::FuncOp function) {
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!interface || !interface.getDisjointOutputs())
    return function.emitError("CPU buffer fusion requires established external alias legality");
  forwardDestinations(function);
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    OpBuilder builder(copy);
    Location loc = copy.getLoc();
    SmallVector<Value> indices;
    std::function<void(int64_t)> materialize = [&](int64_t axis) {
      if (axis == copy.getSource().getType().getRank()) {
        Value value = builder.create<memref::LoadOp>(loc, copy.getSource(), indices);
        builder.create<memref::StoreOp>(loc, value, copy.getTarget(), indices);
        return;
      }
      Value extent = builder.create<memref::DimOp>(loc, copy.getSource(), axis);
      loop(builder, loc, index(builder, loc, 0), extent, 1, [&](Value coordinate) {
        indices.push_back(coordinate);
        materialize(axis + 1);
        indices.pop_back();
      });
    };
    materialize(0);
    copy.erase();
  }
  bool changed;
  do {
    changed = false;
    SmallVector<memref::AllocOp> allocations;
    function.walk([&](memref::AllocOp op) { allocations.push_back(op); });
    for (memref::AllocOp allocation : llvm::reverse(allocations))
      changed |= fuse(allocation);
  } while (changed);
  return success();
}

void forwardCPUOutputs(func::FuncOp function) { forwardDestinations(function); }

}

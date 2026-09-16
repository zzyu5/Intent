#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Analysis/AliasAnalysis.h"
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

void forwardDestinations(func::FuncOp function) {
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    auto allocation = copy.getSource().getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != copy->getBlock()) continue;
    Value target = copy.getTarget();
    if (allocation.getType().getRank() != copy.getTarget().getType().getRank()) continue;
    PhysicalProgramAnalysis analysis(function);
    AliasAnalysis aliasAnalysis(function);
    Value targetRoot = analysis.storageRoot(target);
    auto external = analysis.externalView(target);
    if (!targetRoot.getDefiningOp<memref::AllocOp>() &&
        (!external || external.getAccess() != 1)) continue;
    if (targetRoot == allocation.getResult()) continue;
    auto disjoint = [&](Value memory) {
      Value root = analysis.storageRoot(memory);
      if (root == targetRoot) return false;
      if (aliasAnalysis.alias(root, targetRoot).isNo()) return true;
      return external && analysis.externalView(root);
    };
    auto targetOp = target.getDefiningOp();
    DominanceInfo dominance(function);
    if (targetOp && !dominance.dominates(targetOp, allocation)) {
      if (targetOp->getBlock() != allocation->getBlock() ||
          (!isMemoryEffectFree(targetOp) && !isa<memref::AllocOp>(targetOp)) ||
          llvm::any_of(targetOp->getOperands(), [&](Value value) {
            return !dominance.dominates(value, allocation);
          })) continue;
    }
    bool legal = true;
    Operation *lastUse = copy;
    SmallVector<memref::DeallocOp> deallocations;
    SmallVector<Value> aliases{allocation.getResult()};
    for (unsigned i = 0; i < aliases.size(); ++i) {
      for (Operation *user : aliases[i].getUsers()) {
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
          if (aliases[i] != allocation.getResult()) legal = false;
          else deallocations.push_back(dealloc);
          continue;
        }
        if (auto view = dyn_cast<memref::SubViewOp>(user)) aliases.push_back(view.getResult());
        else if (auto cast = dyn_cast<memref::CastOp>(user)) aliases.push_back(cast.getResult());
        else if (!isa<memref::CopyOp, memref::DimOp, memref::LoadOp, memref::StoreOp,
                      linalg::LinalgOp, ReduceOp, ScanOp, HistogramOp, QuantizedDotOp>(user)) legal = false;
        Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
        if (!ancestor) { legal = false; continue; }
        if (ancestor != copy && copy->isBeforeInBlock(ancestor)) {
          if (!isa<memref::LoadOp, memref::DimOp, memref::SubViewOp, memref::CastOp>(user)) legal = false;
          if (lastUse->isBeforeInBlock(ancestor)) lastUse = ancestor;
        }
      }
    }
    // Forwarding moves the destination writes earlier. Its previous contents
    // must remain unobserved before the copy and stable through all later reads.
    for (Operation *between = allocation->getNextNode(); legal && between != lastUse->getNextNode();
         between = between->getNextNode()) {
      if (between == copy) continue;
      between->walk([&](Operation *operation) {
        if (isMemoryEffectFree(operation)) return;
        auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
        if (!effects) {
          if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) legal = false;
          return;
        }
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (auto &effect : instances) {
          if (isa<MemoryEffects::Allocate>(effect.getEffect())) continue;
          if (!effect.getValue() || !disjoint(effect.getValue()))
            legal = false;
        }
      });
    }
    if (!legal) continue;
    if (targetOp && !dominance.dominates(targetOp, allocation)) targetOp->moveBefore(allocation);
    for (memref::DeallocOp dealloc : deallocations) dealloc.erase();
    allocation.getResult().replaceAllUsesExcept(target, copy);
    copy.erase();
    allocation.erase();
  }
}

bool canReplay(Value value, Operation *root, llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *operation = value.getDefiningOp();
  if (!operation || !root->isAncestor(operation)) return true;
  if (!seen.insert(operation).second) return true;
  if (operation->getNumRegions() || operation->getNumResults() != 1 ||
      (!isMemoryEffectFree(operation) && !isa<memref::LoadOp>(operation))) return false;
  return llvm::all_of(operation->getOperands(), [&](Value input) { return canReplay(input, root, seen); });
}

bool stableRead(memref::LoadOp load, Operation *producer, func::FuncOp function,
                ArrayRef<Operation *> consumers) {
  Value base = load.getMemref();
  while (true) {
    if (auto view = base.getDefiningOp<memref::SubViewOp>()) base = view.getSource();
    else if (auto cast = base.getDefiningOp<memref::CastOp>()) base = cast.getSource();
    else break;
  }
  if (auto argument = dyn_cast<BlockArgument>(base)) {
    if (argument.getOwner() != &function.front()) return false;
    auto abi = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
    auto field = dyn_cast<ViewArgumentAttr>(abi.getArguments()[argument.getArgNumber()]);
    if (!field) return false;
    if (field.getAccess() == 0) return true;
    SmallVector<Value> aliases{base};
    for (unsigned i = 0; i < aliases.size(); ++i)
      for (Operation *user : aliases[i].getUsers()) {
        if (isa<memref::LoadOp, memref::DimOp>(user)) continue;
        if (auto view = dyn_cast<memref::SubViewOp>(user)) { aliases.push_back(view.getResult()); continue; }
        if (auto cast = dyn_cast<memref::CastOp>(user)) { aliases.push_back(cast.getResult()); continue; }
        if (!isa<memref::StoreOp>(user)) return false;
        Operation *write = producer->getBlock()->findAncestorOpInBlock(*user);
        if (!write || write == producer || !write->isBeforeInBlock(producer)) return false;
      }
    return true;
  }
  auto allocation = base.getDefiningOp<memref::AllocOp>();
  if (!allocation) return false;
  Block *owner = allocation->getBlock();
  Operation *preparation = owner->findAncestorOpInBlock(*producer);
  if (!preparation) return false;
  memref::DeallocOp end;
  SmallVector<Operation *> overwrites;
  SmallVector<Value> aliases{base};
  for (unsigned i = 0; i < aliases.size(); ++i)
    for (Operation *user : aliases[i].getUsers()) {
      if (isa<memref::LoadOp, vector::LoadOp, memref::DimOp>(user)) continue;
      if (auto view = dyn_cast<memref::SubViewOp>(user)) { aliases.push_back(view.getResult()); continue; }
      if (auto cast = dyn_cast<memref::CastOp>(user)) { aliases.push_back(cast.getResult()); continue; }
      if (auto release = dyn_cast<memref::DeallocOp>(user)) {
        if (release.getMemref() != base || end || release->getBlock() != owner) return false;
        end = release;
        continue;
      }
      if (!isa<memref::StoreOp, vector::StoreOp>(user)) return false;
      Operation *write = owner->findAncestorOpInBlock(*user);
      if (!write || write == preparation) return false;
      // A later overwrite cannot change a replayed read that has already
      // completed. Same-loop writes stay excluded by the owner-block order.
      if (write->isBeforeInBlock(preparation)) continue;
      if (!llvm::all_of(consumers, [&](Operation *consumer) {
            Operation *use = owner->findAncestorOpInBlock(*consumer);
            return use && use->isBeforeInBlock(write);
          })) return false;
      overwrites.push_back(write);
    }
  // Replaying a read also extends its use of the backing storage. A snapshot
  // may outlive its source, in which case it must keep its own materialization.
  if (!end) return false;
  for (Operation *consumer : consumers) {
    Operation *use = owner->findAncestorOpInBlock(*consumer);
    if (!use || !use->isBeforeInBlock(end)) return false;
  }
  return llvm::all_of(overwrites, [&](Operation *write) { return write->isBeforeInBlock(end); });
}

Value replay(Value value, Operation *root, OpBuilder &builder, IRMapping &mapping) {
  if (mapping.contains(value)) return mapping.lookup(value);
  Operation *operation = value.getDefiningOp();
  if (!operation || !root->isAncestor(operation)) return value;
  for (Value input : operation->getOperands())
    mapping.map(input, replay(input, root, builder, mapping));
  builder.clone(*operation, mapping);
  return mapping.lookup(value);
}

bool dependsOn(Value value, Value coordinate, Operation *root) {
  if (value == coordinate) return true;
  Operation *operation = value.getDefiningOp();
  return operation && root->isAncestor(operation) &&
      llvm::any_of(operation->getOperands(), [&](Value operand) {
        return dependsOn(operand, coordinate, root);
      });
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
  return operation->hasTrait<OpTrait::Elementwise>() &&
      llvm::all_of(operation->getOperandTypes(), [](Type type) {
        return isa<FloatType, IntegerType, IndexType>(type);
      }) && llvm::all_of(operation->getOperands(), [&](Value operand) {
        return canReplayVector(operand, coordinate, root);
      });
}

Value replayVector(Value value, Value coordinate, Operation *root, int64_t width,
                   OpBuilder &builder, IRMapping &scalars, IRMapping &vectors) {
  if (vectors.contains(value)) return vectors.lookup(value);
  auto type = VectorType::get({width}, value.getType());
  Location loc = root->getLoc();
  Value result;
  if (value == coordinate) {
    Value start = builder.create<vector::BroadcastOp>(loc, type, scalars.lookup(value));
    result = builder.create<arith::AddIOp>(loc, start, builder.create<vector::StepOp>(loc, type));
  } else if (!dependsOn(value, coordinate, root)) {
    result = builder.create<vector::BroadcastOp>(loc, type, replay(value, root, builder, scalars));
  } else if (auto load = value.getDefiningOp<memref::LoadOp>()) {
    SmallVector<Value> indices;
    for (Value index : load.getIndices()) indices.push_back(replay(index, root, builder, scalars));
    result = builder.create<vector::LoadOp>(loc, type,
        replay(load.getMemref(), root, builder, scalars), indices);
  } else {
    Operation *operation = value.getDefiningOp();
    IRMapping mapping;
    for (Value operand : operation->getOperands())
      mapping.map(operand, replayVector(operand, coordinate, root, width, builder, scalars, vectors));
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
  bool otherEffect = false;
  root->walk([&](Operation *operation) {
    if (operation == store || isa<scf::ForOp, scf::YieldOp, memref::LoadOp>(operation)) return;
    if (!isMemoryEffectFree(operation)) otherEffect = true;
  });
  if (otherEffect) return false;
  llvm::SmallPtrSet<Operation *, 16> seen;
  if (!canReplay(store.getValue(), root, seen)) return false;
  for (Operation *load : loads)
    if (isa<vector::LoadOp>(load) && (loops.empty() ||
        !canReplayVector(store.getValue(), loops.back().getInductionVar(), root))) return false;
  if (consumers.size() != 1) {
    auto integer = [](Type type) { return isa<IndexType, IntegerType>(type); };
    if (!integer(store.getValue().getType()) || llvm::any_of(seen, [&](Operation *operation) {
          return (!isMemoryEffectFree(operation) && !isa<memref::LoadOp>(operation)) ||
              !llvm::all_of(operation->getResultTypes(), integer);
        })) return false;
  }
  auto function = allocation->getParentOfType<func::FuncOp>();
  for (Operation *operation : seen)
    if (auto read = dyn_cast<memref::LoadOp>(operation); read && !stableRead(read, root, function, loads)) return false;
  DominanceInfo dominance(function);
  for (Operation *load : loads)
    if (root->isAncestor(load) || !dominance.dominates(root, load)) return false;
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
      replacement = replayVector(store.getValue(), loops.back().getInductionVar(), root,
          vector.getVectorType().getDimSize(0), builder, mapping, vectors);
    } else replacement = replay(store.getValue(), root, builder, mapping);
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
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
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

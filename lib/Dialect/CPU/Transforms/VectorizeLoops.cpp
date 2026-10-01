#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Intent/Analysis/IntegerRelations.h"
#include "Utilities.h"
#include "VectorReductions.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

bool vectorElement(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64() ||
      isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
      type.isIndex() || type.isSignlessInteger(1) || type.isSignlessInteger(8) ||
      type.isSignlessInteger(16) || type.isSignlessInteger(32) || type.isSignlessInteger(64);
}

std::optional<int64_t> coefficient(Value value, scf::ForOp loop) {
  if (value == loop.getInductionVar()) return 1;
  Operation *op = value.getDefiningOp();
  if (!op || !loop->isAncestor(op)) return 0;
  // Intent CPU logical coordinates use the DSL's signed 64-bit index contract;
  // the shared query does not assume a width for arbitrary MLIR index values.
  auto folded = foldIntegerDifference(describeScalarValue(value),
      [&](Value input) { return coefficient(input, loop); },
      [](Value input) { return getConstantIntValue(input); }, /*indexBitWidth=*/64);
  if (folded) return folded;
  if (llvm::all_of(op->getOperands(), [&](Value input) {
        auto c = coefficient(input, loop); return c && *c == 0;
      })) return 0;
  return std::nullopt;
}

bool contiguous(Value memory, ValueRange indices, scf::ForOp loop, bool allowInvariant,
                SmallVectorImpl<Value> &guardedMemories) {
  auto type = cast<MemRefType>(memory.getType());
  if (!vectorElement(type.getElementType()) || type.getElementType().isInteger(1)) return false;
  if (auto owner = memory.getDefiningOp(); owner && loop->isAncestor(owner)) return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) return false;
  bool invariant = true;
  for (auto [axis, index] : llvm::enumerate(indices)) {
    auto c = coefficient(index, loop);
    if (!c) return false;
    if (*c != 0) {
      invariant = false;
      if (*c != 1 || axis + 1 != indices.size()) return false;
      if (ShapedType::isDynamic(strides[axis])) {
        if (!llvm::is_contained(guardedMemories, memory)) guardedMemories.push_back(memory);
      } else if (strides[axis] != 1) return false;
    }
  }
  return !invariant || allowInvariant;
}

class VectorBody {
public:
  VectorBody(scf::ForOp original, OpBuilder &builder, OpBuilder &invariants, Value iv, int64_t width)
      : original(original), b(builder), invariants(invariants), width(width) {
    mapping.map(original.getInductionVar(), iv);
  }

  static bool invariant(Value value, scf::ForOp loop) {
    if (value == loop.getInductionVar() || llvm::is_contained(loop.getRegionIterArgs(), value)) return false;
    Operation *op = value.getDefiningOp();
    return !op || !loop->isAncestor(op) || llvm::all_of(op->getOperands(), [&](Value input) {
      return invariant(input, loop);
    });
  }

  Value scalar(Value value) {
    if (mapping.contains(value)) return mapping.lookup(value);
    Operation *op = value.getDefiningOp();
    if (!op || !original->isAncestor(op)) return value;
    for (Value input : op->getOperands()) mapping.map(input, scalar(input));
    (invariant(value, original) ? invariants : b).clone(*op, mapping);
    return mapping.lookup(value);
  }

  SmallVector<Value> indices(ValueRange inputs) {
    SmallVector<Value> result;
    for (Value input : inputs) result.push_back(scalar(input));
    return result;
  }

  Value vector(Value value) {
    if (vectors.count(value)) return vectors.lookup(value);
    Location loc = original.getLoc();
    Operation *op = value.getDefiningOp();
    auto type = VectorType::get({width}, value.getType());
    Value result;
    if (value == original.getInductionVar()) {
      Value start = b.create<vector::BroadcastOp>(loc, type, scalar(value));
      Value lanes = b.create<vector::StepOp>(loc, type);
      result = b.create<arith::AddIOp>(loc, start, lanes);
    } else if (!op || !original->isAncestor(op) || coefficient(value, original) == 0) {
      result = b.create<vector::BroadcastOp>(loc, type, scalar(value));
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      bool invariant = llvm::all_of(load.getIndices(), [&](Value input) {
        return coefficient(input, original) == 0;
      });
      result = invariant
          ? Value(b.create<vector::BroadcastOp>(loc, type, scalar(value)))
          : Value(b.create<vector::LoadOp>(loc, type, load.getMemref(), indices(load.getIndices())));
    } else {
      IRMapping operationMapping;
      for (Value input : op->getOperands())
        operationMapping.map(input, vector(input));
      Operation *cloned = b.clone(*op, operationMapping);
      cloned->getResult(0).setType(type);
      result = cloned->getResult(0);
    }
    vectors[value] = result;
    return result;
  }

private:
  scf::ForOp original;
  OpBuilder &b;
  OpBuilder &invariants;
  int64_t width;
  IRMapping mapping;
  llvm::DenseMap<Value, Value> vectors;
};

bool dependsOnCarry(Value value, scf::ForOp loop) {
  if (llvm::is_contained(loop.getRegionIterArgs(), value)) return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || !loop->isAncestor(operation)) return false;
  return llvm::any_of(operation->getOperands(), [&](Value input) {
    return dependsOnCarry(input, loop);
  });
}

void vectorize(scf::ForOp original, int64_t width, int64_t replicas, bool nonempty = false) {
  if (!matchPattern(original.getStep(), m_One())) return;
  SmallVector<Value> reductionInputs;
  SmallVector<Operation *> combines;
  if (original.getNumResults()) {
    auto order = original->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
    if (!order || !order.getAdjacentReassociation()) return;
    for (auto [carry, yielded] : llvm::zip(original.getRegionIterArgs(),
             original.getBody()->getTerminator()->getOperands())) {
      if (!carry.getType().isF32() || !carry.hasOneUse()) return;
      Operation *combine = yielded.getDefiningOp();
      if (!combine || !isa<arith::AddFOp, arith::MaxNumFOp, arith::MaximumFOp>(combine)) return;
      if (combine->getOperand(0) == carry) reductionInputs.push_back(combine->getOperand(1));
      else if (combine->getOperand(1) == carry) reductionInputs.push_back(combine->getOperand(0));
      else return;
      if (dependsOnCarry(reductionInputs.back(), original)) return;
      combines.push_back(combine);
    }
  }
  SmallVector<memref::StoreOp> stores;
  SmallVector<memref::LoadOp> loads;
  SmallVector<Value> guardedMemories;
  for (Operation &operation : original.getBody()->without_terminator()) {
    if (operation.getNumRegions()) return;
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      if (!contiguous(load.getMemref(), load.getIndices(), original, true, guardedMemories)) return;
      loads.push_back(load);
    } else if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
      if (dependsOnCarry(store.getValue(), original) ||
          !contiguous(store.getMemref(), store.getIndices(), original, false, guardedMemories)) return;
      stores.push_back(store);
    } else if (!isMemoryEffectFree(&operation) || operation.getNumResults() != 1 ||
               !vectorElement(operation.getResult(0).getType()) ||
               (!operation.hasTrait<OpTrait::Elementwise>() &&
                coefficient(operation.getResult(0), original) != 0)) return;
  }
  if (reductionInputs.empty() && stores.empty()) return;
  auto function = original->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis physical(function);
  AliasAnalysis aliases(function);
  auto independent = [&](Value lhs, ValueRange lhsIndices, Value rhs, ValueRange rhsIndices) {
    if (lhs == rhs && lhsIndices == rhsIndices) return true;
    if (aliases.alias(lhs, rhs).isNo()) return true;
    auto left = physical.externalView(lhs), right = physical.externalView(rhs);
    auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
    return left && right && physical.storageRoot(lhs) != physical.storageRoot(rhs) &&
        interface.getDisjointOutputs() && (left.getAccess() != 0 || right.getAccess() != 0);
  };
  for (auto [number, store] : llvm::enumerate(stores)) {
    for (auto load : loads)
      if (!independent(load.getMemref(), load.getIndices(), store.getMemref(), store.getIndices())) return;
    for (auto other : ArrayRef(stores).drop_front(number + 1))
      if (!independent(other.getMemref(), other.getIndices(), store.getMemref(), store.getIndices())) return;
  }
  OpBuilder b(original);
  Location loc = original.getLoc();
  bool needsInvariantGuard = llvm::any_of(original.getBody()->without_terminator(), [&](Operation &op) {
    return op.getNumResults() == 1 && !isSpeculatable(&op) &&
        VectorBody::invariant(op.getResult(0), original);
  });
  if (!guardedMemories.empty() || (!nonempty && needsInvariantGuard)) {
    Value one = index(b, loc, 1), condition;
    SmallVector<memref::ExtractStridedMetadataOp> descriptors;
    for (Value memory : guardedMemories) {
      auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, memory);
      descriptors.push_back(metadata);
      Value unit = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, metadata.getStrides().back(), one);
      condition = condition ? Value(b.create<arith::AndIOp>(loc, condition, unit)) : unit;
    }
    if (!nonempty && needsInvariantGuard) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
          original.getUpperBound(), original.getLowerBound());
      condition = condition ? Value(b.create<arith::AndIOp>(loc, condition, active)) : active;
    }
    auto dispatch = b.create<scf::IfOp>(loc, original.getResultTypes(), condition, true);
    original.replaceAllUsesWith(dispatch.getResults());
    b.setInsertionPointToStart(&dispatch.getThenRegion().front());
    IRMapping mapping;
    for (auto [memory, metadata] : llvm::zip(guardedMemories, descriptors)) {
      auto type = cast<MemRefType>(memory.getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      (void)type.getStridesAndOffset(strides, offset);
      strides.back() = 1;
      auto contiguousType = MemRefType::get(type.getShape(), type.getElementType(),
          StridedLayoutAttr::get(b.getContext(), offset, strides), type.getMemorySpace());
      SmallVector<OpFoldResult> sizes, physicalStrides;
      for (int64_t axis = 0; axis < type.getRank(); ++axis) {
        sizes.push_back(type.isDynamicDim(axis) ? OpFoldResult(metadata.getSizes()[axis])
                                               : OpFoldResult(b.getIndexAttr(type.getDimSize(axis))));
        physicalStrides.push_back(ShapedType::isDynamic(strides[axis]) ? OpFoldResult(metadata.getStrides()[axis])
                                                                     : OpFoldResult(b.getIndexAttr(strides[axis])));
      }
      OpFoldResult physicalOffset = ShapedType::isDynamic(offset) ? OpFoldResult(metadata.getOffset())
                                                                 : OpFoldResult(b.getIndexAttr(offset));
      // A descriptor reconstruction retains the proved stride through vector
      // canonicalization, which can strip memref.cast from vector loads.
      mapping.map(memory, b.create<memref::ReinterpretCastOp>(loc, contiguousType,
          metadata.getBaseBuffer(), physicalOffset, sizes, physicalStrides));
    }
    auto contiguousLoop = cast<scf::ForOp>(b.clone(*original, mapping));
    vectorize(contiguousLoop, width, replicas, nonempty || needsInvariantGuard);
    if (original.getNumResults()) {
      b.setInsertionPointToEnd(&dispatch.getThenRegion().front());
      b.create<scf::YieldOp>(loc, contiguousLoop.getResults());
    }
    original->moveBefore(&dispatch.getElseRegion().front(), dispatch.getElseRegion().front().begin());
    if (original.getNumResults()) {
      b.setInsertionPointToEnd(&dispatch.getElseRegion().front());
      b.create<scf::YieldOp>(loc, original.getResults());
    }
    return;
  }
  int64_t logicalWidth = replicas * width;
  Value step = index(b, loc, logicalWidth);
  Value length = b.create<arith::SubIOp>(loc, original.getUpperBound(), original.getLowerBound());
  auto merge = [&](Operation *combine, Value lhs, Value rhs) {
    IRMapping mapping;
    mapping.map(combine->getOperand(0), lhs);
    mapping.map(combine->getOperand(1), rhs);
    Operation *result = b.clone(*combine, mapping);
    result->getResult(0).setType(lhs.getType());
    return result->getResult(0);
  };
  auto combineValues = [&](ValueRange lhs, ValueRange rhs, int64_t) {
    SmallVector<Value> results;
    for (auto [number, combine] : llvm::enumerate(combines))
      results.push_back(merge(combine, lhs[number], rhs[number]));
    return results;
  };
  auto vectorBody = [&](Value coordinate, OpBuilder &invariants) {
    VectorBody body(original, b, invariants, coordinate, logicalWidth);
    for (Operation &operation : original.getBody()->without_terminator()) {
      if (auto load = dyn_cast<memref::LoadOp>(&operation)) body.vector(load.getResult());
      else if (auto store = dyn_cast<memref::StoreOp>(&operation))
        b.create<vector::StoreOp>(loc, body.vector(store.getValue()), store.getMemref(), body.indices(store.getIndices()));
    }
    SmallVector<Value> inputs;
    for (Value input : reductionInputs) inputs.push_back(body.vector(input));
    return inputs;
  };
  auto order = original->getAttrOfType<ReductionOrderAttr>("intent_cpu.reduction_order");
  if (!combines.empty() && order.getElementPermutation()) {
    Value full = add(b, loc, original.getLowerBound(),
        multiply(b, loc, b.create<arith::DivSIOp>(loc, length, step), step));
    Value hasBlock = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          full, original.getLowerBound());
    auto active = b.create<scf::IfOp>(loc, original.getResultTypes(), hasBlock, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(active.thenBlock());
      Value begin = add(b, loc, original.getLowerBound(), step);
      OpBuilder invariants(begin.getDefiningOp());
      // Seed each lane from an actual member. The original accumulator may be
      // a non-identity partial from an enclosing block and is included once.
      auto seeds = vectorBody(original.getLowerBound(), invariants);
      auto blocks = b.create<scf::ForOp>(loc, begin, full, step, seeds);
      {
        OpBuilder::InsertionGuard blockGuard(b);
        b.setInsertionPointToStart(blocks.getBody());
        auto values = vectorBody(blocks.getInductionVar(), invariants);
        b.create<scf::YieldOp>(loc, combineValues(blocks.getRegionIterArgs(), values, logicalWidth));
      }
      auto partial = horizontalReduce(b, loc, blocks.getResults(), combineValues);
      b.create<scf::YieldOp>(loc, combineValues(original.getInitArgs(), partial, 0));
      b.setInsertionPointToStart(active.elseBlock());
      b.create<scf::YieldOp>(loc, original.getInitArgs());
    }
    original.setLowerBound(full);
    original.getInitArgsMutable().assign(active.getResults());
    if (replicas > 1) vectorize(original, width, 1);
    original->removeAttr("intent_cpu.reduction_order");
    return;
  }
  // Adjacent-only reductions keep every chunk and partial in coordinate order.
  // Register replicas widen a local tree without introducing striped carries.
  // Shorten the scalar carry chains with contiguous sums, then combine their
  // results in coordinate order. Splitting requires the closed +0 identity.
  int64_t partitions = stores.empty() && !combines.empty() &&
      llvm::all_of(combines, [](Operation *op) { return isa<arith::AddFOp>(op); }) &&
      llvm::all_of(original.getInitArgs(), [](Value value) { return matchPattern(value, m_PosZeroFloat()); })
      ? std::min<int64_t>(replicas, 4) : 1;
  Value groupStep = index(b, loc, logicalWidth * partitions);
  Value full = add(b, loc, original.getLowerBound(),
      multiply(b, loc, b.create<arith::DivSIOp>(loc, length, groupStep), groupStep));
  Value span = b.create<arith::DivSIOp>(loc,
      b.create<arith::SubIOp>(loc, full, original.getLowerBound()), index(b, loc, partitions));
  SmallVector<Value> initial;
  for (int64_t part = 0; part < partitions; ++part) llvm::append_range(initial, original.getInitArgs());
  auto vectorLoop = b.create<scf::ForOp>(loc, original.getLowerBound(),
      add(b, loc, original.getLowerBound(), span), step, initial);
  OpBuilder invariantBuilder(vectorLoop);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(vectorLoop.getBody());
    SmallVector<Value> results;
    for (int64_t part = 0; part < partitions; ++part) {
      Value coordinate = add(b, loc, vectorLoop.getInductionVar(), multiply(b, loc, span, index(b, loc, part)));
      auto inputs = vectorBody(coordinate, invariantBuilder);
      if (!inputs.empty()) {
        auto partial = horizontalReduce(b, loc, inputs, combineValues);
        auto accumulated = combineValues(
            vectorLoop.getRegionIterArgs().slice(part * combines.size(), combines.size()), partial, 0);
        llvm::append_range(results, accumulated);
      }
    }
    if (!results.empty()) b.create<scf::YieldOp>(loc, results);
  }
  SmallVector<Value> combined;
  for (auto [number, combine] : llvm::enumerate(combines)) {
    SmallVector<Value> leaves;
    for (int64_t part = 0; part < partitions; ++part)
      leaves.push_back(vectorLoop.getResult(part * combines.size() + number));
    while (leaves.size() > 1) {
      SmallVector<Value> level;
      for (unsigned part = 0; part < leaves.size(); part += 2)
        level.push_back(merge(combine, leaves[part], leaves[part + 1]));
      leaves = std::move(level);
    }
    combined.push_back(leaves.front());
  }
  original.setLowerBound(full);
  if (!reductionInputs.empty()) original.getInitArgsMutable().assign(combined);
  if (partitions > 1) vectorize(original, width, replicas);
  else if (replicas > 1) vectorize(original, width, 1);
  original->removeAttr("intent_cpu.reduction_order");
}

}

LogicalResult vectorizeLoops(func::FuncOp function, int64_t width, int64_t replicas,
                            int64_t reductionReplicas) {
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp op) { loops.push_back(op); });
  for (scf::ForOp loop : loops)
    vectorize(loop, width, loop.getNumResults() ? reductionReplicas : replicas);
  return success();
}

}

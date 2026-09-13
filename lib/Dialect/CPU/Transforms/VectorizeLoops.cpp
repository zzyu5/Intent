#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
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
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    auto lhs = coefficient(add.getLhs(), loop), rhs = coefficient(add.getRhs(), loop);
    if (lhs && rhs) return *lhs + *rhs;
  } else if (auto sub = dyn_cast<arith::SubIOp>(op)) {
    auto lhs = coefficient(sub.getLhs(), loop), rhs = coefficient(sub.getRhs(), loop);
    if (lhs && rhs) return *lhs - *rhs;
  } else if (auto mul = dyn_cast<arith::MulIOp>(op)) {
    if (auto lhs = getConstantIntValue(mul.getLhs())) {
      if (auto rhs = coefficient(mul.getRhs(), loop)) return *lhs * *rhs;
    }
    if (auto rhs = getConstantIntValue(mul.getRhs())) {
      if (auto lhs = coefficient(mul.getLhs(), loop)) return *lhs * *rhs;
    }
  }
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
  VectorBody(scf::ForOp original, OpBuilder &builder, Value iv, int64_t width)
      : original(original), b(builder), width(width) {
    mapping.map(original.getInductionVar(), iv);
  }

  Value scalar(Value value) {
    if (mapping.contains(value)) return mapping.lookup(value);
    Operation *op = value.getDefiningOp();
    if (!op || !original->isAncestor(op)) return value;
    for (Value input : op->getOperands()) mapping.map(input, scalar(input));
    b.clone(*op, mapping);
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

void vectorize(scf::ForOp original, int64_t width, int64_t replicas) {
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
      if (!combine || !isa<arith::AddFOp, arith::MaxNumFOp>(combine)) return;
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
    auto interface = function->getAttrOfType<InterfaceAttr>("intent_cpu.interface");
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
  if (!guardedMemories.empty()) {
    Value one = index(b, loc, 1), condition;
    SmallVector<memref::ExtractStridedMetadataOp> descriptors;
    for (Value memory : guardedMemories) {
      auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, memory);
      descriptors.push_back(metadata);
      Value unit = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, metadata.getStrides().back(), one);
      condition = condition ? Value(b.create<arith::AndIOp>(loc, condition, unit)) : unit;
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
    vectorize(contiguousLoop, width, replicas);
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
  // One local tree spans adjacent register replicas. Keeping the leaves in
  // coordinate order permits reassociation without striped accumulators, and
  // amortizes the narrow horizontal stages across the bound register replicas.
  int64_t logicalWidth = replicas * width;
  Value step = index(b, loc, logicalWidth);
  Value length = b.create<arith::SubIOp>(loc, original.getUpperBound(), original.getLowerBound());
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
  auto merge = [&](Operation *combine, Value lhs, Value rhs) {
    IRMapping mapping;
    mapping.map(combine->getOperand(0), lhs);
    mapping.map(combine->getOperand(1), rhs);
    Operation *result = b.clone(*combine, mapping);
    result->getResult(0).setType(lhs.getType());
    return result->getResult(0);
  };
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(vectorLoop.getBody());
    SmallVector<Value> results;
    for (int64_t part = 0; part < partitions; ++part) {
      Value coordinate = add(b, loc, vectorLoop.getInductionVar(), multiply(b, loc, span, index(b, loc, part)));
      VectorBody body(original, b, coordinate, logicalWidth);
      for (Operation &operation : original.getBody()->without_terminator()) {
        if (auto load = dyn_cast<memref::LoadOp>(&operation)) body.vector(load.getResult());
        else if (auto store = dyn_cast<memref::StoreOp>(&operation))
          b.create<vector::StoreOp>(loc, body.vector(store.getValue()), store.getMemref(), body.indices(store.getIndices()));
      }
      for (auto [number, reductionInput] : llvm::enumerate(reductionInputs)) {
        Operation *combine = combines[number];
        Value value = body.vector(reductionInput);
        for (int64_t count = logicalWidth; count > 1; count /= 2) {
          SmallVector<int64_t> even, odd;
          for (int64_t lane = 0; lane < count; lane += 2) {
            even.push_back(lane);
            odd.push_back(lane + 1);
          }
          Value lhs = b.create<vector::ShuffleOp>(loc, value, value, even);
          Value rhs = b.create<vector::ShuffleOp>(loc, value, value, odd);
          value = merge(combine, lhs, rhs);
        }
        Value sum = b.create<vector::ExtractElementOp>(loc, value, index(b, loc, 0));
        results.push_back(merge(combine,
            vectorLoop.getRegionIterArgs()[part * reductionInputs.size() + number], sum));
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

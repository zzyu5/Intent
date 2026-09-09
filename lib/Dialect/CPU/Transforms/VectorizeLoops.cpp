#include "Intent/Dialect/CPU/Transforms/Passes.h"
#include "Utilities.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::cpu {
namespace {

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

bool contiguous(Value memory, ValueRange indices, scf::ForOp loop, bool allowInvariant) {
  auto type = cast<MemRefType>(memory.getType());
  if (!type.getElementType().isF32()) return false;
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
      if (*c != 1 || axis + 1 != indices.size() || strides[axis] != 1) return false;
    }
  }
  return !invariant || allowInvariant;
}

class VectorBody {
public:
  VectorBody(scf::ForOp original, OpBuilder &builder, Value iv, int64_t width)
      : original(original), b(builder), type(VectorType::get({width}, b.getF32Type())) {
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
    Value result;
    if (!op || !original->isAncestor(op) || isa<arith::ConstantOp>(op)) {
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
        operationMapping.map(input, input.getType().isF32() ? vector(input) : scalar(input));
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
  VectorType type;
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
  for (Operation &operation : original.getBody()->without_terminator()) {
    if (operation.getNumRegions()) return;
    if (auto load = dyn_cast<memref::LoadOp>(&operation)) {
      if (!contiguous(load.getMemref(), load.getIndices(), original, true)) return;
    } else if (auto store = dyn_cast<memref::StoreOp>(&operation)) {
      if (dependsOnCarry(store.getValue(), original) ||
          !contiguous(store.getMemref(), store.getIndices(), original, false)) return;
      stores.push_back(store);
    } else if (!isMemoryEffectFree(&operation) ||
               llvm::any_of(operation.getResultTypes(), [](Type type) {
                 return !type.isF32() && !type.isIndex() && !type.isInteger(64);
               })) return;
  }
  if (reductionInputs.empty() && stores.empty()) return;
  OpBuilder b(original);
  Location loc = original.getLoc();
  // One local tree spans adjacent register replicas. Keeping the leaves in
  // coordinate order permits reassociation without striped accumulators, and
  // amortizes the narrow horizontal stages across the bound register replicas.
  int64_t logicalWidth = replicas * width;
  Value step = index(b, loc, logicalWidth);
  Value length = b.create<arith::SubIOp>(loc, original.getUpperBound(), original.getLowerBound());
  Value full = add(b, loc, original.getLowerBound(),
      multiply(b, loc, b.create<arith::DivSIOp>(loc, length, step), step));
  auto vectorLoop = b.create<scf::ForOp>(loc, original.getLowerBound(), full, step, original.getInitArgs());
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(vectorLoop.getBody());
    VectorBody body(original, b, vectorLoop.getInductionVar(), logicalWidth);
    for (memref::StoreOp store : stores)
      b.create<vector::StoreOp>(loc, body.vector(store.getValue()), store.getMemref(), body.indices(store.getIndices()));
    if (!reductionInputs.empty()) {
      SmallVector<Value> results;
      for (auto [number, reductionInput] : llvm::enumerate(reductionInputs)) {
        Operation *combine = combines[number];
        auto merge = [&](Value lhs, Value rhs) {
          IRMapping mapping;
          mapping.map(combine->getOperand(0), lhs);
          mapping.map(combine->getOperand(1), rhs);
          Operation *result = b.clone(*combine, mapping);
          result->getResult(0).setType(lhs.getType());
          return result->getResult(0);
        };
        Value value = body.vector(reductionInput);
        for (int64_t count = logicalWidth; count > 1; count /= 2) {
          SmallVector<int64_t> even, odd;
          for (int64_t lane = 0; lane < count; lane += 2) {
            even.push_back(lane);
            odd.push_back(lane + 1);
          }
          Value lhs = b.create<vector::ShuffleOp>(loc, value, value, even);
          Value rhs = b.create<vector::ShuffleOp>(loc, value, value, odd);
          value = merge(lhs, rhs);
        }
        Value sum = b.create<vector::ExtractElementOp>(loc, value, index(b, loc, 0));
        results.push_back(merge(vectorLoop.getRegionIterArgs()[number], sum));
      }
      b.create<scf::YieldOp>(loc, results);
    }
  }
  original.setLowerBound(full);
  if (!reductionInputs.empty()) original.getInitArgsMutable().assign(vectorLoop.getResults());
  if (replicas > 1) vectorize(original, width, 1);
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

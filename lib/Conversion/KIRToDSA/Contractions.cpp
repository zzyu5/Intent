#include "Construction.h"

namespace intent::kir_to_dsa {

std::optional<ContractionAxes> Construction::contractionAxes(ContractOp matrix) {
  SmallVector<int64_t> lhsReduction, rhsReduction, lhsBatch, rhsBatch;
  auto pairs = [](ArrayAttr attributes, SmallVectorImpl<int64_t> &lhs,
                  SmallVectorImpl<int64_t> &rhs) {
    for (Attribute attribute : attributes) {
      auto pair = cast<ArrayAttr>(attribute);
      lhs.push_back(cast<IntegerAttr>(pair[0]).getInt());
      rhs.push_back(cast<IntegerAttr>(pair[1]).getInt());
    }
  };
  pairs(matrix.getReduce(), lhsReduction, rhsReduction);
  pairs(matrix.getBatch(), lhsBatch, rhsBatch);
  return ContractionAxes::get(cast<RankedTensorType>(matrix.getLhs().getType()).getRank(),
      cast<RankedTensorType>(matrix.getRhs().getType()).getRank(), lhsReduction,
      rhsReduction, lhsBatch, rhsBatch);
}

SmallVector<ContractOp> Construction::sharedInputProducts(ContractOp matrix) {
  SmallVector<ContractOp> candidates{matrix};
  for (Operation *user : matrix.getLhs().getUsers()) {
    auto other = dyn_cast<ContractOp>(user);
    if (!other || other == matrix || other->getBlock() != matrix->getBlock() ||
        other.getLhs() != matrix.getLhs() ||
        other.getRhs().getType() != matrix.getRhs().getType() ||
        other.getResult().getType() != matrix.getResult().getType() ||
        other.getReduce() != matrix.getReduce() || other.getBatch() != matrix.getBatch() ||
        values.lookupOrNull(other.getResult())) continue;
    bool sameWindow = true;
    auto resultType = cast<RankedTensorType>(matrix.getResult().getType());
    for (unsigned axis = 0; axis < resultType.getRank(); ++axis) {
      auto lhs = selectedAxis(matrix.getResult(), axis);
      auto rhs = selectedAxis(other.getResult(), axis);
      sameWindow &= bool(lhs) == bool(rhs);
      if (lhs && rhs)
        sameWindow &= lhs->capacity == rhs->capacity &&
            sameIndex(lhs->extent, rhs->extent) && sameIndex(lhs->begin, rhs->begin) &&
            sameIndex(lhs->count, rhs->count);
    }
    if (!sameWindow) continue;
    Operation *first = matrix->isBeforeInBlock(other) ? matrix.getOperation() : other.getOperation();
    Operation *last = first == matrix.getOperation() ? other.getOperation() : matrix.getOperation();
    bool independent = true;
    for (Operation *op = first->getNextNode(); op != last; op = op->getNextNode()) {
      if (auto load = dyn_cast<ViewLoadOp>(op)) {
        independent &= cast<ViewType>(load.getSource().getType()).getAccess() == 0;
      } else independent &= isMemoryEffectFree(op) || isa<AssumeInBoundsOp>(op);
    }
    if (independent) candidates.push_back(other);
  }
  if (candidates.size() == 1) return candidates;
  DenseSet<Value> results;
  for (ContractOp product : candidates) results.insert(product.getResult());
  for (ContractOp product : candidates) {
    SmallVector<Value> pending{product.getRhs()};
    DenseSet<Value> visited;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!visited.insert(value).second) continue;
      if (results.contains(value)) return {matrix};
      if (Operation *op = value.getDefiningOp())
        llvm::append_range(pending, op->getOperands());
    }
  }
  llvm::stable_sort(candidates, [](ContractOp lhs, ContractOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return candidates;
}

LogicalResult Construction::localMatMul(ContractOp matrix, const LocalShape &shape, Value output, bool stream) {
  Location loc = matrix.getLoc();
  auto leftType = cast<RankedTensorType>(matrix.getLhs().getType());
  auto rightType = cast<RankedTensorType>(matrix.getRhs().getType());
  auto axes = contractionAxes(matrix);
  if (leftType.getRank() < 1 || leftType.getRank() > 2 || rightType.getRank() < 1 || rightType.getRank() > 2 || !axes || axes->reduction.size() != 1 ||
      !axes->batch.empty() ||
      !cast<RankedTensorType>(matrix.getResult().getType()).getElementType().isF32())
    return matrix.emitError("DSA local matrix/vector multiplication requires one paired axis and f32 accumulation");
  int64_t leftReduce = axes->reduction.front().lhs, rightReduce = axes->reduction.front().rhs;
  if (stream && shape.size() == 2 && leftType.getRank() == 2 && rightType.getRank() == 2) {
    SmallVector<ContractOp> group = sharedInputProducts(matrix);
    SmallVector<std::pair<Value, unsigned>> roots{{matrix.getLhs(), leftReduce}};
    for (ContractOp product : group) roots.emplace_back(product.getRhs(), rightReduce);
    auto plan = planExecutionSlices(*matrix->getBlock(), 0, roots);
    if (!plan && group.size() > 1) {
      group = {matrix};
      roots = {{matrix.getLhs(), leftReduce}, {matrix.getRhs(), rightReduce}};
      plan = planExecutionSlices(*matrix->getBlock(), 0, roots);
    }
    auto domain = plan ? sliceDomain(*plan, config.getTileK(), !distributedTiles) : std::nullopt;
    APInt extent;
    bool needsSlicing = domain && (distributedTiles ||
        !matchPattern(domain->extent, m_ConstantInt(&extent)) || extent.getSExtValue() > domain->capacity);
    if (domain && needsSlicing) {
      SmallVector<Value> accumulators;
      SmallVector<LocalShape> resultShapes;
      for (ContractOp product : group) {
        auto selected = localShape(product.getResult(), loc);
        if (failed(selected)) return failure();
        resultShapes.push_back(*selected);
        accumulators.push_back(allocateTensor(loc, b.getF32Type(), *selected));
      }
      auto savedValues = values; auto savedProducts = products; auto savedSlices = valueSlices;
      LogicalResult status = loop(loc, index(loc, 0), domain->extent, index(loc, domain->capacity), [&](Value begin) {
        Value count = b.create<arith::MinSIOp>(loc, sub(loc, domain->extent, begin), index(loc, domain->capacity));
        bindExecutionSlice(*plan, *domain, begin, count);
        for (auto [product, resultShape, accumulator] : llvm::zip_equal(group, resultShapes, accumulators))
          if (failed(localMatMul(product, resultShape, accumulator, false))) return failure();
        return success();
      });
      values = std::move(savedValues); products = std::move(savedProducts); valueSlices = std::move(savedSlices);
      if (failed(status)) return failure();
      for (auto [product, accumulator] : llvm::zip_equal(group, accumulators))
        values.map(product.getResult(), accumulator);
      return success();
    }
  }
  Value rightOperand = matrix.getRhs();
  while (auto transpose = rightOperand.getDefiningOp<TransposeOp>()) {
    auto permutation = transpose.getPermutation();
    if (rightType.getRank() != 2 || permutation.size() != 2 ||
        cast<IntegerAttr>(permutation[0]).getInt() != 1 || cast<IntegerAttr>(permutation[1]).getInt() != 0) break;
    rightOperand = transpose.getInput();
    rightReduce = 1 - rightReduce;
  }
  Value lhs = get(matrix.getLhs()), rhs = get(rightOperand);
  if (!lhs || !rhs || !localShapes.count(lhs) || !localShapes.count(rhs)) return matrix.emitError("DSA local MatMul inputs are unavailable");
  auto normalize = [&](Value input, bool left, int64_t reduction) -> Value {
    LocalShape original = localShapes.lookup(input);
    bool transpose = original.size() == 2 && reduction == (left ? 0 : 1);
    bool columnVector = original.size() == 1 && !left;
    if (!transpose && !columnVector) return input;
    LocalAxis unit{index(loc, 1), index(loc, 0), index(loc, 1), 1};
    LocalShape normalized = transpose ? LocalShape{original[1], original[0]} : LocalShape{original[0], unit};
    Value output = allocateTensor(loc, cast<MemRefType>(input.getType()).getElementType(), normalized);
    if (transpose) {
      b.create<dsa::TransposeOp>(loc, input, output, original[0].count, original[1].count);
      return output;
    }
    if (failed(eachElement(loc, normalized, [&](ValueRange coordinates) {
      SmallVector<Value> from = transpose ? SmallVector<Value>{coordinates[1], coordinates[0]} : SmallVector<Value>{coordinates[0]};
      storeLocal(loc, loadLocal(loc, input, from), output, coordinates); return success();
    }))) return {};
    return output;
  };
  LocalShape leftShape = localShapes.lookup(lhs), rightShape = localShapes.lookup(rhs);
  bool rhsTransposed = rightShape.size() == 2 && rightReduce == 1;
  lhs = normalize(lhs, true, leftReduce);
  if (!rhsTransposed) rhs = normalize(rhs, false, rightReduce);
  if (!lhs || !rhs) return failure();
  if (!output) output = allocateTensor(loc, b.getF32Type(), shape);
  int64_t rows = cast<MemRefType>(lhs.getType()).getDimSize(0);
  int64_t columns = cast<MemRefType>(rhs.getType()).getDimSize(rhsTransposed ? 0 : 1);
  Value rowCount = leftShape.size() == 1 ? index(loc, 1) : leftShape[1 - leftReduce].count;
  Value columnCount = rightShape.size() == 1 ? index(loc, 1) : rightShape[1 - rightReduce].count;
  Value accumulator = output;
  if (cast<MemRefType>(output.getType()).getShape() != ArrayRef<int64_t>({rows, columns})) {
    accumulator = allocate(loc, b.getF32Type(), rows, columns);
    b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
  }
  b.create<dsa::MatMulOp>(loc, lhs, rhs, accumulator, rowCount, leftShape[leftReduce].count, columnCount,
      b.getBoolAttr(rhsTransposed));
  if (accumulator != output) {
    if (shape.size() != 1 || rightShape.size() != 1) return matrix.emitError("DSA matrix result projection is unavailable");
    if (failed(eachElement(loc, shape, [&](ValueRange coordinates) {
      storeLocal(loc, loadLocal(loc, accumulator, ValueRange{coordinates[0], index(loc, 0)}), output, coordinates); return success();
    }))) return failure();
  }
  values.map(matrix.getResult(), output); return success();
}

} // namespace intent::kir_to_dsa

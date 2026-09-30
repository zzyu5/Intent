#include "PassDetail.h"

using namespace mlir;
namespace intent::bangc {
bool paddedInput(Operation *consumer, Value input, Value rows, Value columns) {
  if (!input.getDefiningOp<memref::AllocaOp>()) return false;
  // This proof concerns one owned buffer. An escaping view or control operand
  // requires the general sub-tile path, which materializes independent inputs.
  for (Operation *user : input.getUsers())
    if (user->getNumRegions() || llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); }) ||
        (!isa<MemoryEffectOpInterface>(user) && !isMemoryEffectFree(user)))
      return false;
  auto sameCount = [](Value lhs, Value rhs) {
    if (lhs == rhs) return true;
    APInt left, right;
    return matchPattern(lhs, m_ConstantInt(&left)) && matchPattern(rhs, m_ConstantInt(&right)) && left == right;
  };
  for (Operation *previous = consumer->getPrevNode(); previous; previous = previous->getPrevNode()) {
    if (auto load = dyn_cast<dsa::LoadTileOp>(previous); load && load.getOutput() == input)
      return sameCount(load.getRows(), rows) && sameCount(load.getColumns(), columns);
    if (auto gather = dyn_cast<dsa::GatherRowsOp>(previous); gather && gather.getOutput() == input)
      return sameCount(gather.getRows(), rows) && sameCount(gather.getColumns(), columns);
    if (auto gather = dyn_cast<dsa::GroupGatherRowsOp>(previous); gather && gather.getOutput() == input) {
      APInt count;
      return sameCount(gather.getRows(), rows) && matchPattern(columns, m_ConstantInt(&count)) &&
          count.getSExtValue() == cast<MemRefType>(input.getType()).getDimSize(1);
    }
    if (auto conversion = dyn_cast<dsa::CastOp>(previous); conversion && conversion.getOutput() == input) {
      auto source = cast<MemRefType>(conversion.getInput().getType());
      auto destination = cast<MemRefType>(input.getType());
      // Element conversion preserves the zero padding introduced by LoadTile.
      // Prove it before the conversion, independently of later source reuse.
      return conversion.getInput() != input && source.getShape() == destination.getShape() &&
          paddedInput(previous, conversion.getInput(), rows, columns);
    }
    bool overwritten = false;
    previous->walk([&](Operation *operation) {
      // The allocation does not escape and has no aliases. Work on other
      // buffers, including nested scalar loops, cannot invalidate its padding.
      if (!llvm::is_contained(operation->getOperands(), input)) return;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == input)) overwritten = true;
      } else if (!isMemoryEffectFree(operation)) overwritten = true;
    });
    if (overwritten) return false;
  }
  return false;
}

bool consumesMatrixRhs(dsa::MatMulOp matrix) {
  Value input = matrix.getRhs();
  if (input == matrix.getLhs() || input == matrix.getAccumulator() ||
      !input.getDefiningOp<memref::AllocaOp>()) return false;
  return llvm::all_of(input.getUsers(), [&](Operation *user) {
    if (user == matrix.getOperation()) return true;
    if (auto load = dyn_cast<dsa::LoadTileOp>(user))
      return load.getOutput() == input && load.getSource() != input;
    if (auto fill = dyn_cast<dsa::FillOp>(user)) return fill.getOutput() == input;
    return false;
  });
}

Value matrixReshapeWorkspace(OpBuilder &b, Location loc, Value input, Value transposed, bool consumeInput) {
  auto type = cast<MemRefType>(transposed.getType());
  if (type.getDimSize(0) == 64) return {};
  if (!consumeInput) return allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
  // Packing has already copied the row-major input into an independent
  // transposed tile. Its dead storage can now hold the reshaped filter.
  return b.create<memref::ReinterpretCastOp>(loc, type, input, int64_t(0),
      type.getShape(), ArrayRef<int64_t>{type.getDimSize(1), 1});
}

LogicalResult realizeMatMul(dsa::MatMulOp matrix, dsa::ConfigurationAttr config) {
  Location loc = matrix.getLoc();
  OpBuilder b(matrix);
  auto lhsType = cast<MemRefType>(matrix.getLhs().getType());
  auto rhsType = cast<MemRefType>(matrix.getRhs().getType());
  auto accType = cast<MemRefType>(matrix.getAccumulator().getType());
  bool rhsTransposed = matrix.getRhsTransposed();
  Type element = lhsType.getElementType();
  if (!element.isF16() && !element.isBF16() && !element.isF32())
    return matrix.emitError("MLU370 matrix profile requires f16/bf16/f32 input tiles");
  int64_t bm = config.getTileM(), bn = config.getTileN(), bk = config.getTileK();
  if (bn % 64)
    return matrix.emitError("MLU370 matrix N block must be a multiple of 64");
  if ((bk * element.getIntOrFloatBitWidth() / 8) % 64)
    return matrix.emitError("MLU370 matrix K block must be aligned to 64 bytes");
  bm = std::min(bm, lhsType.getDimSize(0));
  bn = std::min<int64_t>(bn, llvm::alignTo(rhsType.getDimSize(rhsTransposed ? 0 : 1), int64_t(64)));
  bk = std::min<int64_t>(bk, llvm::alignTo(lhsType.getDimSize(1), int64_t(512 / element.getIntOrFloatBitWidth())));
  auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
  auto offset = [&](Value row, Value column, int64_t stride) -> Value {
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, row, index(stride)), column);
  };
  auto count = [&](Value extent, Value begin, int64_t tile) -> Value {
    return b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, extent, begin), index(tile));
  };
  auto function = matrix->getParentOfType<func::FuncOp>();
  auto full = [&](Value value, int64_t expected) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == expected && interval->second == expected;
  };
  // A full narrow output can use C^T += B^T A^T. The original M=64 axis
  // then fills the matrix engine's N lanes, and A already supplies the packed
  // transposed RHS form. Preserve the initial accumulator through a transpose.
  if (!rhsTransposed && lhsType.getDimSize(0) == 64 && rhsType.getDimSize(1) < 64 &&
      rhsType.getDimSize(1) <= config.getTileM() && lhsType.getDimSize(1) == bk &&
      full(matrix.getRows(), 64) && full(matrix.getColumns(), rhsType.getDimSize(1)) &&
      full(matrix.getDepth(), lhsType.getDimSize(1)) &&
      matrix.getAccumulator() != matrix.getLhs() && matrix.getAccumulator() != matrix.getRhs()) {
    int64_t n = rhsType.getDimSize(1), k = lhsType.getDimSize(1);
    Operation *previous = matrix->getPrevNode();
    Value left = allocate(b, loc, element, {n, k}, dsa::nramSpace);
    Value accumulator = allocate(b, loc, b.getF32Type(), {n, 64}, dsa::nramSpace);
    Value packed = allocate(b, loc, element, {k, 64}, dsa::matrixSpace);
    b.create<dsa::TransposeOp>(loc, matrix.getRhs(), left, index(k), index(n));
    b.create<dsa::TransposeOp>(loc, matrix.getAccumulator(), accumulator, index(64), index(n));
    b.create<dsa::PrepareMatrixOp>(loc, matrix.getLhs(), packed, Value{}, Value{}, b.getBoolAttr(true));
    b.create<dsa::MatrixTileOp>(loc, left, packed, accumulator);
    b.create<dsa::TransposeOp>(loc, accumulator, matrix.getAccumulator(), index(n), index(64));
    int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
    if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) {
      matrix.erase(); return success();
    }
    while (matrix->getPrevNode() != previous) matrix->getPrevNode()->erase();
  }
  if (lhsType.getShape() == ArrayRef<int64_t>({bm, bk}) &&
      rhsType.getShape() == (rhsTransposed ? ArrayRef<int64_t>({bn, bk}) : ArrayRef<int64_t>({bk, bn})) &&
      accType.getShape() == ArrayRef<int64_t>({bm, bn}) &&
      matrix.getAccumulator().getDefiningOp<memref::AllocaOp>() &&
      matrix.getAccumulator() != matrix.getLhs() && matrix.getAccumulator() != matrix.getRhs() &&
      paddedInput(matrix, matrix.getLhs(), matrix.getRows(), matrix.getDepth()) &&
      paddedInput(matrix, matrix.getRhs(), rhsTransposed ? matrix.getColumns() : matrix.getDepth(),
                  rhsTransposed ? matrix.getDepth() : matrix.getColumns())) {
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, matrix.getDepth(), index(0));
    active = b.create<arith::AndIOp>(loc, active,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, matrix.getRows(), index(0)));
    active = b.create<arith::AndIOp>(loc, active,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, matrix.getColumns(), index(0)));
    auto guard = b.create<scf::IfOp>(loc, active, false);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    Value transpose = rhsTransposed ? Value{} : allocate(b, loc, element, {bn, bk}, dsa::nramSpace);
    Value reshaped = rhsTransposed ? (bn == 64 ? Value{} : allocate(b, loc, element, {bn, bk}, dsa::nramSpace))
                                  : matrixReshapeWorkspace(b, loc, matrix.getRhs(), transpose, consumesMatrixRhs(matrix));
    Value packed = allocate(b, loc, element, {bk, bn}, dsa::matrixSpace);
    b.create<dsa::PrepareMatrixOp>(loc, matrix.getRhs(), packed, transpose, reshaped, b.getBoolAttr(rhsTransposed));
    b.create<dsa::MatrixTileOp>(loc, matrix.getLhs(), packed, matrix.getAccumulator());
    matrix.erase();
    return success();
  }
  auto rows = b.create<scf::ForOp>(loc, index(0), matrix.getRows(), index(bm));
  b.setInsertionPointToStart(rows.getBody());
  Value m = rows.getInductionVar(), mCount = count(matrix.getRows(), m, bm);
  auto columns = b.create<scf::ForOp>(loc, index(0), matrix.getColumns(), index(bn));
  b.setInsertionPointToStart(columns.getBody());
  Value n = columns.getInductionVar(), nCount = count(matrix.getColumns(), n, bn);
  Value lhs = allocate(b, loc, element, {bm, bk}, dsa::nramSpace);
  Value rhs = allocate(b, loc, element, rhsTransposed ? ArrayRef<int64_t>({bn, bk}) : ArrayRef<int64_t>({bk, bn}), dsa::nramSpace);
  Value accumulator = allocate(b, loc, b.getF32Type(), {bm, bn}, dsa::nramSpace);
  Value transpose = rhsTransposed ? Value{} : allocate(b, loc, element, {bn, bk}, dsa::nramSpace);
  Value reshaped = rhsTransposed ? (bn == 64 ? Value{} : allocate(b, loc, element, {bn, bk}, dsa::nramSpace))
                                : matrixReshapeWorkspace(b, loc, rhs, transpose, true);
  Value packed = allocate(b, loc, element, {bk, bn}, dsa::matrixSpace);
  b.create<dsa::LoadTileOp>(loc, matrix.getAccumulator(), accumulator,
      offset(m, n, accType.getDimSize(1)), index(accType.getDimSize(1)), index(1), mCount, nCount);
  auto reduction = b.create<scf::ForOp>(loc, index(0), matrix.getDepth(), index(bk));
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(reduction.getBody());
    Value k = reduction.getInductionVar(), kCount = count(matrix.getDepth(), k, bk);
    b.create<dsa::LoadTileOp>(loc, matrix.getLhs(), lhs, offset(m, k, lhsType.getDimSize(1)),
        index(lhsType.getDimSize(1)), index(1), mCount, kCount);
    b.create<dsa::LoadTileOp>(loc, matrix.getRhs(), rhs,
        offset(rhsTransposed ? n : k, rhsTransposed ? k : n, rhsType.getDimSize(1)),
        index(rhsType.getDimSize(1)), index(1), rhsTransposed ? nCount : kCount, rhsTransposed ? kCount : nCount);
    b.create<dsa::PrepareMatrixOp>(loc, rhs, packed, transpose, reshaped, b.getBoolAttr(rhsTransposed));
    b.create<dsa::MatrixTileOp>(loc, lhs, packed, accumulator);
  }
  b.create<dsa::StoreTileOp>(loc, accumulator, matrix.getAccumulator(), offset(m, n, accType.getDimSize(1)),
      index(accType.getDimSize(1)), index(1), mCount, nCount);
  matrix.erase();
  return success();
}
} // namespace intent::bangc

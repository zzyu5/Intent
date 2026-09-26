#include "Intent/Target/BangC/Passes.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>
using namespace mlir;
namespace intent::bangc {
namespace {
bool supportedUnary(UnaryOperator kind) {
  return kind == UnaryOperator::Exp || kind == UnaryOperator::Exp2 || kind == UnaryOperator::Log || kind == UnaryOperator::Sqrt ||
      kind == UnaryOperator::Rsqrt || kind == UnaryOperator::Tanh || kind == UnaryOperator::Abs || kind == UnaryOperator::Negate ||
      kind == UnaryOperator::Sin || kind == UnaryOperator::Cos || kind == UnaryOperator::Floor;
}
bool supportedBinary(BinaryOperator kind) {
  return kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply ||
      kind == BinaryOperator::TrueDivide || kind == BinaryOperator::Maximum || kind == BinaryOperator::Minimum ||
      kind == BinaryOperator::MaximumNum || kind == BinaryOperator::MinimumNum;
}
bool supportedScalarBinary(BinaryOperator kind) {
  return kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply;
}
Value allocate(OpBuilder &b, Location loc, Type element, ArrayRef<int64_t> shape, int64_t space) {
  auto type = MemRefType::get(shape, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
  auto result = b.create<memref::AllocaOp>(loc, type);
  result.setAlignment(128);
  return result;
}
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
void bindStorage(func::FuncOp function, int64_t &nram, int64_t &wram);
std::optional<std::pair<int64_t, int64_t>> integerInterval(Value value, func::FuncOp function);
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
    packed.getDefiningOp()->setAttr("bangc.layout", b.getStringAttr("matrix_filter_interleaved64"));
    b.create<dsa::TransposeOp>(loc, matrix.getRhs(), left, index(k), index(n));
    b.create<dsa::TransposeOp>(loc, matrix.getAccumulator(), accumulator, index(64), index(n));
    b.create<dsa::PrepareMatrixOp>(loc, matrix.getLhs(), packed, Value{}, Value{}, b.getBoolAttr(true));
    auto tile = b.create<dsa::MatrixTileOp>(loc, left, packed, accumulator);
    tile->setAttr("bangc.implementation", b.getStringAttr("matmul_local_f32_accumulator"));
    b.create<dsa::TransposeOp>(loc, accumulator, matrix.getAccumulator(), index(n), index(64));
    int64_t nram = 0, wram = 0; bindStorage(function, nram, wram);
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
    packed.getDefiningOp()->setAttr("bangc.layout", b.getStringAttr("matrix_filter_interleaved64"));
    b.create<dsa::PrepareMatrixOp>(loc, matrix.getRhs(), packed, transpose, reshaped, b.getBoolAttr(rhsTransposed));
    auto tile = b.create<dsa::MatrixTileOp>(loc, matrix.getLhs(), packed, matrix.getAccumulator());
    tile->setAttr("bangc.implementation", b.getStringAttr("matmul_local_f32_accumulator"));
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
  packed.getDefiningOp()->setAttr("bangc.layout", b.getStringAttr("matrix_filter_interleaved64"));
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
    auto tile = b.create<dsa::MatrixTileOp>(loc, lhs, packed, accumulator);
    tile->setAttr("bangc.implementation", b.getStringAttr("matmul_local_f32_accumulator"));
  }
  b.create<dsa::StoreTileOp>(loc, accumulator, matrix.getAccumulator(), offset(m, n, accType.getDimSize(1)),
      index(accType.getDimSize(1)), index(1), mCount, nCount);
  matrix.erase();
  return success();
}
LogicalResult realizeSigmoid(dsa::UnaryOp sigmoid) {
  if (sigmoid.getApproximate() || sigmoid.getFlushToZero())
    return sigmoid.emitError("sigmoid numerical mode has no selected BANG C implementation");
  Location loc = sigmoid.getLoc();
  OpBuilder b(sigmoid);
  auto type = cast<MemRefType>(sigmoid.getInput().getType());
  Value input = sigmoid.getInput(), output = sigmoid.getOutput();
  if (!type.getElementType().isF32()) {
    input = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
    output = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
    b.create<dsa::CastOp>(loc, sigmoid.getInput(), input);
  }
  Value absolute = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
  Value negative = allocate(b, loc, b.getF32Type(), type.getShape(), dsa::nramSpace);
  auto unary = [&](Value from, Value to, UnaryOperator kind) {
    b.create<dsa::UnaryOp>(loc, from, to, UnaryOperatorAttr::get(b.getContext(), kind),
        b.getBoolAttr(false), b.getBoolAttr(false), Value());
  };
  unary(input, absolute, UnaryOperator::Abs);
  unary(absolute, negative, UnaryOperator::Negate);
  unary(negative, absolute, UnaryOperator::Exp);
  auto index = [&](int64_t n) -> Value { return b.create<arith::ConstantIndexOp>(loc, n); };
  Value one = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1));
  Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
  auto rows = b.create<scf::ForOp>(loc, index(0), index(type.getDimSize(0)), index(1));
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(rows.getBody());
    auto columns = b.create<scf::ForOp>(loc, index(0), index(type.getDimSize(1)), index(1));
    b.setInsertionPointToStart(columns.getBody());
    SmallVector<Value> coordinates{rows.getInductionVar(), columns.getInductionVar()};
    Value x = b.create<memref::LoadOp>(loc, input, coordinates);
    Value e = b.create<memref::LoadOp>(loc, absolute, coordinates);
    Value denominator = b.create<arith::AddFOp>(loc, one, e);
    Value belowZero = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, x, zero);
    Value numerator = b.create<arith::SelectOp>(loc, belowZero, e, one);
    Value result = b.create<arith::DivFOp>(loc, numerator, denominator);
    b.create<memref::StoreOp>(loc, result, output, coordinates);
  }
  if (output != sigmoid.getOutput()) b.create<dsa::CastOp>(loc, output, sigmoid.getOutput());
  sigmoid.erase();
  return success();
}
void realizeRowSumChannels(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::ReduceOp> operations;
  function.walk([&](dsa::ReduceOp reduce) {
    auto input = cast<MemRefType>(reduce.getInput().getType());
    if (reduce.getKind() == BinaryOperator::Add && reduce.getAxis() == 1 &&
        input.getDimSize(0) >= 32 && input.getDimSize(0) % 32 == 0 && input.getDimSize(1) >= 32)
      operations.push_back(reduce);
  });
  for (auto reduce : operations) {
    auto input = cast<MemRefType>(reduce.getInput().getType());
    OpBuilder b(reduce);
    Location loc = reduce.getLoc();
    Value storage = allocate(b, loc, input.getElementType(), {input.getDimSize(1), input.getDimSize(0)}, dsa::nramSpace);
    Value rows = b.create<arith::ConstantIndexOp>(loc, input.getDimSize(0));
    auto transpose = b.create<dsa::TransposeOp>(loc, reduce.getInput(), storage, rows, reduce.getCount());
    auto replacement = b.create<dsa::ReduceOp>(loc, storage, reduce.getOutput(), reduce.getScratch(),
        reduce.getCount(), reduce.getIdentity(), reduce.getKindAttr(), b.getI64IntegerAttr(0));
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024) {
      replacement.erase(); transpose.erase(); storage.getDefiningOp()->erase();
    } else reduce.erase();
  }
}
LogicalResult realizeNumericExtremaWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> operations;
  function.walk([&](dsa::BinaryOp binary) {
    if ((binary.getKind() == BinaryOperator::MaximumNum || binary.getKind() == BinaryOperator::MinimumNum) &&
        !binary.getScratch()) operations.push_back(binary);
  });
  for (auto binary : operations) {
    auto type = cast<MemRefType>(binary.getOutput().getType());
    if ((!type.getElementType().isF16() && !type.getElementType().isF32()) ||
        !isa<MemRefType>(binary.getRhs().getType()))
      return binary.emitError("numeric extrema requires a supported floating tile");
    for (int64_t width = std::min<int64_t>(8192, type.getNumElements()); width > 0; width /= 2) {
      OpBuilder b(binary);
      Value scratch = allocate(b, binary.getLoc(), type.getElementType(), {1, width}, dsa::nramSpace);
      binary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024) break;
      binary.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
    if (!binary.getScratch()) return binary.emitError("numeric extrema has no room for its destructive-operand workspace");
  }
  return success();
}
void realizeCompareWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::CompareOp> comparisons;
  function.walk([&](dsa::CompareOp compare) { if (!compare.getScratch()) comparisons.push_back(compare); });
  for (auto compare : comparisons) {
    auto type = cast<MemRefType>(compare.getLhs().getType());
    if (!type.getElementType().isInteger(64) || type.getNumElements() < 256) continue;
    for (int64_t width = int64_t(1) << llvm::Log2_64(std::min<int64_t>(8192, type.getNumElements())); width >= 256; width /= 2) {
      OpBuilder b(compare);
      Value scratch = allocate(b, compare.getLoc(), b.getI32Type(), {6, width}, dsa::nramSpace);
      compare.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024) break;
      compare.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
  }
}
void realizeGatherWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::GatherRowsOp> gathers;
  function.walk([&](dsa::GatherRowsOp gather) { if (!gather.getPlan()) gathers.push_back(gather); });
  DenseMap<int64_t, Value> indicesByRows;
  for (auto gather : gathers) {
    int64_t rows = cast<MemRefType>(gather.getOutput().getType()).getDimSize(0);
    if (rows < 64 || rows % 64 || rows > 65536) continue;
    OpBuilder b(gather);
    Location loc = gather.getLoc();
    // Adjacent supplies with identical offsets consume the same immutable
    // address snapshot. Only pure operations and fresh allocations may lie
    // between them; writes and control boundaries terminate this reuse.
    Operation *previous = gather->getPrevNode();
    while (previous && (isa<memref::AllocaOp>(previous) || isMemoryEffectFree(previous)))
      previous = previous->getPrevNode();
    if (auto supply = dyn_cast_or_null<dsa::GatherRowsOp>(previous);
        supply && supply.getPlan() && supply.getRowOffsets() == gather.getRowOffsets() &&
        supply.getRows() == gather.getRows() &&
        cast<MemRefType>(supply.getOutput().getType()).getDimSize(0) == rows) {
      gather.getPlanMutable().assign(supply.getPlan());
      continue;
    }
    Value scratch = allocate(b, loc, b.getI64Type(), {3, rows}, dsa::nramSpace);
    Value indices = indicesByRows.lookup(rows);
    dsa::IotaOp initialize;
    if (!indices) {
      OpBuilder init(function.getContext());
      init.setInsertionPointToStart(&function.front());
      indices = allocate(init, loc, init.getF32Type(), {1, rows}, dsa::nramSpace);
      initialize = init.create<dsa::IotaOp>(loc, indices);
    }
    auto prepare = b.create<dsa::GatherPlanOp>(loc, gather.getRowOffsets(), gather.getRows(), indices, scratch);
    prepare->setAttr("bangc.internal_nram_bytes", b.getI64IntegerAttr(512));
    gather.getPlanMutable().assign(scratch);
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram + 512 > config.getLocalBytes() || nram + 512 > 768 * 1024) {
      gather.getPlanMutable().clear();
      prepare.erase();
      scratch.getDefiningOp()->erase();
      if (initialize) { initialize.erase(); indices.getDefiningOp()->erase(); }
      continue;
    }
    indicesByRows[rows] = indices;
  }
  // Submit consecutive independent destinations before the common completion
  // fence. The plan is read-only throughout the group, and remains live until
  // every transfer has been issued.
  auto owner = [](Value value) {
    while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
    return value;
  };
  for (auto gather : gathers) {
    if (!gather.getPlan() || gather.getAsynchronous()) continue;
    SmallVector<dsa::GatherRowsOp> group{gather};
    SmallVector<Value> outputs{owner(gather.getOutput())};
    Operation *next = gather->getNextNode();
    while (next) {
      if (isa<memref::AllocaOp>(next) || isMemoryEffectFree(next)) { next = next->getNextNode(); continue; }
      auto supply = dyn_cast<dsa::GatherRowsOp>(next);
      if (!supply || supply.getPlan() != gather.getPlan() ||
          llvm::is_contained(outputs, owner(supply.getOutput()))) break;
      outputs.push_back(owner(supply.getOutput()));
      group.push_back(supply);
      next = next->getNextNode();
    }
    if (group.size() < 2) continue;
    OpBuilder b(gather);
    b.create<dsa::SynchronizeOp>(gather.getLoc());
    for (auto supply : group) supply.setAsynchronous(true);
    b.setInsertionPointAfter(group.back());
    b.create<dsa::SynchronizeOp>(gather.getLoc());
  }
}
bool realizeRowBroadcasts(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::LoadTileOp> loads;
  function.walk([&](dsa::LoadTileOp load) { loads.push_back(load); });
  bool changed = false;
  for (auto load : loads) {
    auto source = cast<MemRefType>(load.getSource().getType());
    auto output = cast<MemRefType>(load.getOutput().getType());
    if (load.getAsynchronous() || source.getMemorySpaceAsInt() != dsa::nramSpace ||
        !source.getLayout().isIdentity() || !output.getElementType().isF32() ||
        output.getDimSize(0) < 16 || output.getDimSize(1) < 16 ||
        !matchPattern(load.getRowStride(), m_One()) || !matchPattern(load.getColumnStride(), m_Zero())) continue;
    OpBuilder b(load);
    Value scratch = allocate(b, load.getLoc(), output.getElementType(),
        {output.getDimSize(1), output.getDimSize(0)}, dsa::nramSpace);
    auto broadcast = b.create<dsa::BroadcastRowsOp>(load.getLoc(), load.getSource(), load.getOutput(), scratch,
        load.getOffset(), load.getRows(), load.getColumns());
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024) {
      broadcast.erase(); scratch.getDefiningOp()->erase(); continue;
    }
    load.erase(); changed = true;
  }
  return changed;
}
void coalesceTileLoads(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
  auto eligible = [&](dsa::LoadTileOp load) {
    if (!load || load.getAsynchronous()) return false;
    auto argument = dyn_cast<BlockArgument>(load.getSource());
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == 0;
  };
  auto owner = [](Value value) {
    while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
    return value;
  };
  SmallVector<dsa::LoadTileOp> loads;
  function.walk([&](dsa::LoadTileOp load) { if (eligible(load)) loads.push_back(load); });
  for (auto load : loads) {
    if (load.getAsynchronous()) continue;
    SmallVector<dsa::LoadTileOp> group{load};
    SmallVector<Value> destinations{owner(load.getOutput())};
    Operation *next = load->getNextNode();
    while (next) {
      if (isa<memref::AllocaOp>(next) || isMemoryEffectFree(next)) { next = next->getNextNode(); continue; }
      auto supply = dyn_cast<dsa::LoadTileOp>(next);
      if (!eligible(supply) || llvm::is_contained(destinations, owner(supply.getOutput()))) break;
      group.push_back(supply); destinations.push_back(owner(supply.getOutput()));
      next = next->getNextNode();
    }
    if (group.size() < 2) continue;
    OpBuilder b(load);
    auto before = b.create<dsa::SynchronizeOp>(load.getLoc());
    for (auto supply : group) supply.setAsynchronous(true);
    b.setInsertionPointAfter(group.back());
    auto after = b.create<dsa::SynchronizeOp>(load.getLoc());
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      for (auto supply : group) supply.setAsynchronous(false);
      before.erase(); after.erase();
    }
  }
}
// Every finite value is an integer multiple of 2^k. RN addition preserves
// the coarser operand grid; RN multiplication preserves the product grid,
// bounded below by the destination's smallest subnormal. Nonfinite values and
// zero impose no grid restriction. This is a type/dataflow fact, not a bound
// inferred from a benchmark's input samples.
int floatGrid(Value value, Operation *read, unsigned depth = 0) {
  Type type = value.getType();
  if (auto buffer = dyn_cast<MemRefType>(type)) type = buffer.getElementType();
  if (type.isF16()) return -24;
  if (type.isBF16()) return -133;
  if (!type.isF32() || depth > 32) return -149;
  auto grid = [&](Value input, Operation *before) { return floatGrid(input, before, depth + 1); };
  auto binary = [&](Value lhs, Value rhs, Operation *before, bool multiply) {
    int a = grid(lhs, before), b = grid(rhs, before);
    return multiply ? std::clamp(a + b, -149, 1024) : std::min(a, b);
  };
  FloatAttr constant;
  if (matchPattern(value, m_Constant(&constant))) {
    uint32_t bits = constant.getValue().bitcastToAPInt().getZExtValue();
    uint32_t exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 255 || (!exponent && !fraction)) return 1024;
    uint32_t significand = fraction | (exponent ? 0x800000 : 0);
    return (exponent ? int(exponent) - 150 : -149) + llvm::countr_zero(significand);
  }
  if (auto load = value.getDefiningOp<memref::LoadOp>()) return grid(load.getMemref(), load);
  if (auto cast = value.getDefiningOp<arith::ExtFOp>()) return grid(cast.getIn(), cast);
  if (auto add = value.getDefiningOp<arith::AddFOp>()) return binary(add.getLhs(), add.getRhs(), add, false);
  if (auto sub = value.getDefiningOp<arith::SubFOp>()) return binary(sub.getLhs(), sub.getRhs(), sub, false);
  if (auto mul = value.getDefiningOp<arith::MulFOp>()) return binary(mul.getLhs(), mul.getRhs(), mul, true);
  if (!value.getDefiningOp<memref::AllocaOp>()) return -149;
  // Owned, unaliased storage permits an exact last-writer query. Other views
  // and opaque uses retain the general f32 grid.
  for (Operation *user : value.getUsers())
    if (llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); }) ||
        (!isa<MemoryEffectOpInterface>(user) && !isMemoryEffectFree(user))) return -149;
  for (Operation *previous = read->getPrevNode(); previous; previous = previous->getPrevNode()) {
    if (auto fill = dyn_cast<dsa::FillOp>(previous); fill && fill.getOutput() == value) return grid(fill.getValue(), fill);
    if (auto cast = dyn_cast<dsa::CastOp>(previous); cast && cast.getOutput() == value) return grid(cast.getInput(), cast);
    if (auto copy = dyn_cast<memref::CopyOp>(previous); copy && copy.getTarget() == value) return grid(copy.getSource(), copy);
    if (auto load = dyn_cast<dsa::LoadTileOp>(previous); load && load.getOutput() == value) return grid(load.getSource(), load);
    if (auto operation = dyn_cast<dsa::BinaryOp>(previous); operation && operation.getOutput() == value) {
      auto kind = operation.getKind();
      if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract || kind == BinaryOperator::Multiply ||
          kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum ||
          kind == BinaryOperator::Minimum || kind == BinaryOperator::MinimumNum)
        return binary(operation.getLhs(), operation.getRhs(), operation, kind == BinaryOperator::Multiply);
      return -149;
    }
    if (auto reduce = dyn_cast<dsa::ReduceOp>(previous); reduce && reduce.getOutput() == value) {
      auto kind = reduce.getKind();
      if (kind == BinaryOperator::Add || kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum ||
          kind == BinaryOperator::Minimum || kind == BinaryOperator::MinimumNum)
        return std::min(grid(reduce.getInput(), reduce), grid(reduce.getIdentity(), reduce));
      return -149;
    }
    bool overwritten = false;
    previous->walk([&](Operation *operation) {
      if (!llvm::is_contained(operation->getOperands(), value)) return;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == value)) overwritten = true;
      } else if (!isMemoryEffectFree(operation)) overwritten = true;
    });
    if (overwritten) return -149;
  }
  return -149;
}
void realizeExponentialWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::UnaryOp> operations;
  function.walk([&](dsa::UnaryOp unary) {
    if (!unary.getScratch() &&
        ((unary.getKind() == UnaryOperator::Exp && !unary.getApproximate() && !unary.getFlushToZero()) ||
         (unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() && unary.getFlushToZero())))
      operations.push_back(unary);
  });
  for (auto unary : operations) {
    auto type = cast<MemRefType>(unary.getOutput().getType());
    if (unary.getKind() == UnaryOperator::Exp2) {
      if (floatGrid(unary.getInput(), unary) >= -126)
        unary->setAttr("bangc.input_non_subnormal", UnitAttr::get(function.getContext()));
      continue;
    }
    if (!type.getElementType().isF32() || type.getNumElements() < 1024 || unary.getInput() == unary.getOutput() ||
        !unary.getInput().getDefiningOp<memref::AllocaOp>() || !unary.getOutput().getDefiningOp<memref::AllocaOp>()) continue;
    for (int64_t width = std::min<int64_t>(8192, type.getNumElements()); width >= 256; width /= 2) {
      OpBuilder b(unary);
      Value scratch = allocate(b, unary.getLoc(), b.getF32Type(), {4, width}, dsa::nramSpace);
      unary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) break;
      unary.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
  }
}
void realizeFlushWorkspace(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::UnaryOp> operations;
  function.walk([&](dsa::UnaryOp unary) {
    if (unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() && unary.getFlushToZero())
      operations.push_back(unary);
  });
  for (auto unary : operations) {
    auto output = cast<MemRefType>(unary.getOutput().getType());
    if (unary.getScratch() || !output.getElementType().isF32() || output.getNumElements() < 64) continue;
    for (int64_t width = std::min<int64_t>(8192, output.getNumElements()); width >= 64; width /= 2) {
      OpBuilder builder(unary);
      Value scratch = allocate(builder, unary.getLoc(), builder.getI32Type(), {1, width}, dsa::nramSpace);
      unary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) break;
      unary.getScratchMutable().clear(); scratch.getDefiningOp()->erase();
    }
  }
}
void realizeSelections(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::SelectOp> selections;
  function.walk([&](dsa::SelectOp select) { selections.push_back(select); });
  for (auto select : selections) {
    auto output = cast<MemRefType>(select.getOutput().getType());
    if (select.getScratch() || !isa<MemRefType>(select.getCondition().getType()) ||
        !output.getElementType().isF32() || output.getNumElements() < 64) continue;
    for (int64_t width = int64_t(1) << llvm::Log2_64(std::min<int64_t>(8192, output.getNumElements())); width >= 64; width /= 2) {
      OpBuilder builder(select);
      int64_t rows = isa<MemRefType>(select.getFalseValue().getType()) ? 2 : 1;
      Value scratch = allocate(builder, select.getLoc(), builder.getI32Type(), {rows, width}, dsa::nramSpace);
      select.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) break;
      select.getScratchMutable().clear();
      scratch.getDefiningOp()->erase();
    }
  }
}
dsa::FillOp uniformFillBefore(Value input, Operation *read) {
  while (auto view = input.getDefiningOp<memref::ReinterpretCastOp>()) input = view.getSource();
  if (!input.getDefiningOp<memref::AllocaOp>()) return {};
  SmallVector<Value> aliases{input};
  for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) aliases.push_back(view.getResult());
    else if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); }) ||
             (!isa<MemoryEffectOpInterface>(user) && !isMemoryEffectFree(user))) return {};
  }
  for (Operation *previous = read->getPrevNode(); previous; previous = previous->getPrevNode()) {
    if (auto fill = dyn_cast<dsa::FillOp>(previous); fill && llvm::is_contained(aliases, fill.getOutput())) return fill;
    bool changed = false;
    previous->walk([&](Operation *operation) {
      if (!llvm::any_of(operation->getOperands(), [&](Value value) { return llvm::is_contained(aliases, value); })) return;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || llvm::is_contained(aliases, effect.getValue()))) changed = true;
      } else if (!isMemoryEffectFree(operation)) changed = true;
    });
    if (changed) return {};
  }
  return {};
}
void realizeApproximateReciprocals(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> operations;
  function.walk([&](dsa::BinaryOp binary) {
    if (binary.getKind() == BinaryOperator::TrueDivide && binary.getApproximate() &&
        binary.getFlushToZero() && !binary.getScratch()) operations.push_back(binary);
  });
  for (auto binary : operations) {
    auto type = cast<MemRefType>(binary.getOutput().getType());
    if (!type.getElementType().isF32() ||
        !isa<MemRefType>(binary.getRhs().getType()) || binary.getOutput() == binary.getRhs()) continue;
    auto fill = uniformFillBefore(binary.getLhs(), binary);
    FloatAttr value;
    if (!fill || !matchPattern(fill.getValue(), m_Constant(&value)) || !value.getValue().isExactlyValue(1.0)) continue;
    for (int64_t width = std::min<int64_t>(8192, type.getNumElements()); width >= 1; width /= 2) {
      OpBuilder b(binary);
      Value scratch = allocate(b, binary.getLoc(), b.getI32Type(), {2, width}, dsa::nramSpace);
      binary.getScratchMutable().assign(scratch);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram <= config.getLocalBytes() && nram <= 768 * 1024 && wram <= 1024 * 1024) {
        binary->setAttr("bangc.implementation", b.getStringAttr("reciprocal_f32_ftz"));
        break;
      }
      binary.getScratchMutable().clear(); scratch.getDefiningOp()->erase();
    }
  }
}
bool specializeZeroMatrixTiles(func::FuncOp function) {
  SmallVector<dsa::TransposeOp> transposes;
  function.walk([&](dsa::TransposeOp transpose) { transposes.push_back(transpose); });
  bool changed = false;
  for (auto transpose : transposes) {
    auto fill = uniformFillBefore(transpose.getInput(), transpose);
    if (!fill) continue;
    auto shape = cast<MemRefType>(transpose.getInput().getType());
    auto rows = integerInterval(transpose.getRows(), function), columns = integerInterval(transpose.getColumns(), function);
    if (!rows || !columns || rows->first != shape.getDimSize(0) || rows->second != rows->first ||
        columns->first != shape.getDimSize(1) || columns->second != columns->first) continue;
    OpBuilder b(transpose);
    b.create<dsa::FillOp>(transpose.getLoc(), transpose.getOutput(), fill.getValue());
    transpose.erase(); changed = true;
  }
  SmallVector<dsa::MatrixTileOp> matrices;
  function.walk([&](dsa::MatrixTileOp matrix) { matrices.push_back(matrix); });
  for (auto matrix : matrices) {
    auto fill = uniformFillBefore(matrix.getAccumulator(), matrix);
    FloatAttr zero;
    if (!matrix.getAccumulate() || !fill || !matchPattern(fill.getValue(), m_Constant(&zero)) ||
        !zero.getValue().isZero() || zero.getValue().isNegative()) continue;
    matrix.setAccumulate(false);
    auto owner = [](Value value) {
      while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
      return value;
    };
    Value buffer = owner(matrix.getAccumulator());
    bool observed = false;
    for (Operation *op = fill->getNextNode(); op && op != matrix; op = op->getNextNode()) {
      op->walk([&](Operation *nested) {
        if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
          SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
          for (const auto &effect : instances)
            if (isa<MemoryEffects::Read>(effect.getEffect()) && (!effect.getValue() || owner(effect.getValue()) == buffer)) observed = true;
        } else if (!isMemoryEffectFree(nested) &&
                   llvm::any_of(nested->getOperands(), [&](Value value) { return owner(value) == buffer; })) observed = true;
      });
    }
    if (!observed) fill.erase();
    changed = true;
  }
  return changed;
}
bool retainNarrowExtremaInputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::ReduceOp> reductions;
  function.walk([&](dsa::ReduceOp reduce) {
    if (reduce.getAxis() == 1 && (reduce.getKind() == BinaryOperator::MaximumNum ||
        reduce.getKind() == BinaryOperator::MinimumNum)) reductions.push_back(reduce);
  });
  auto unaliased = [](Value value) {
    return value.getDefiningOp<memref::AllocaOp>() && llvm::all_of(value.getUsers(), [](Operation *user) {
      return !llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); }) &&
          (isa<MemoryEffectOpInterface>(user) || isMemoryEffectFree(user));
    });
  };
  auto writes = [](Operation *op, Value value) {
    bool written = false;
    op->walk([&](Operation *nested) {
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == value)) written = true;
      }
    });
    return written;
  };
  bool changed = false;
  for (auto reduce : reductions) {
    Value oldInput = reduce.getInput(), oldScratch = reduce.getScratch();
    auto type = cast<MemRefType>(oldInput.getType());
    if (!type.getElementType().isF32() || type.getDimSize(0) >= 32 || type.getDimSize(1) < 4 || !unaliased(oldInput)) continue;
    dsa::CastOp conversion;
    for (Operation *op = reduce->getPrevNode(); op; op = op->getPrevNode()) {
      if (auto cast = dyn_cast<dsa::CastOp>(op); cast && cast.getOutput() == oldInput) { conversion = cast; break; }
      if (writes(op, oldInput)) break;
    }
    if (!conversion || !unaliased(conversion.getInput()) ||
        !cast<MemRefType>(conversion.getInput().getType()).getElementType().isF16()) continue;
    bool unchanged = true;
    for (Operation *op = conversion->getNextNode(); op && op != reduce; op = op->getNextNode())
      if (writes(op, conversion.getInput())) { unchanged = false; break; }
    if (!unchanged) continue;
    OpBuilder b(reduce);
    Value scratch = allocate(b, reduce.getLoc(), b.getF16Type(), {1, type.getDimSize(1)}, dsa::nramSpace);
    reduce.getInputMutable().assign(conversion.getInput()); reduce.getScratchMutable().assign(scratch);
    int64_t nram = 0, wram = 0; bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      reduce.getInputMutable().assign(oldInput); reduce.getScratchMutable().assign(oldScratch);
      scratch.getDefiningOp()->erase(); continue;
    }
    if (oldScratch.use_empty()) oldScratch.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}
void bindUniformOperands(func::FuncOp function) {
  SmallVector<dsa::BinaryOp> binaries;
  function.walk([&](dsa::BinaryOp binary) { binaries.push_back(binary); });
  for (auto binary : binaries) {
    if (!supportedScalarBinary(binary.getKind()) ||
        binary.getApproximate() || binary.getFlushToZero()) continue;
    Type element = cast<MemRefType>(binary.getLhs().getType()).getElementType();
    if (!element.isF16() && !element.isF32()) continue;
    Value previous = binary.getRhs();
    auto fill = uniformFillBefore(previous, binary);
    if (!fill) continue;
    binary.getRhsMutable().assign(fill.getValue());
    if (llvm::all_of(previous.getUsers(), [&](Operation *user) {
          auto unused = dyn_cast<dsa::FillOp>(user);
          return unused && unused.getOutput() == previous;
        })) {
      SmallVector<Operation *> fills(previous.getUsers());
      for (Operation *unused : fills) unused->erase();
      previous.getDefiningOp()->erase();
    }
  }
  function.walk([&](dsa::SelectOp select) {
    if (!isa<MemRefType>(select.getFalseValue().getType())) return;
    auto fill = uniformFillBefore(select.getFalseValue(), select);
    FloatAttr constant;
    if (fill && matchPattern(fill.getValue(), m_Constant(&constant))) select.getFalseValueMutable().assign(fill.getValue());
  });
}
void eliminateOverwrittenFills(func::FuncOp function) {
  SmallVector<dsa::FillOp> fills;
  function.walk([&](dsa::FillOp fill) { fills.push_back(fill); });
  for (auto fill : fills) {
    Value output = fill.getOutput();
    if (!output.getDefiningOp<memref::AllocaOp>()) continue;
    SmallVector<Value> aliases{output};
    bool escaped = false;
    for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
      if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) {
        if (!llvm::is_contained(aliases, view.getResult())) aliases.push_back(view.getResult());
      } else if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) escaped = true;
    }
    if (escaped) continue;
    for (Operation *next = fill->getNextNode(); next; next = next->getNextNode()) {
      if (next->getNumRegions()) {
        bool independent = true;
        next->walk([&](Operation *nested) {
          if (llvm::any_of(nested->getOperands(), [&](Value operand) { return llvm::is_contained(aliases, operand); }))
            independent = false;
          if (nested->getNumRegions()) {
            if (!isa<scf::ForOp, scf::IfOp>(nested)) independent = false;
          } else if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
            SmallVector<MemoryEffects::EffectInstance> instances;
            effects.getEffects(instances);
            if (llvm::any_of(instances, [](const auto &effect) { return !effect.getValue(); })) independent = false;
          } else if (!isMemoryEffectFree(nested)) independent = false;
        });
        if (independent) continue;
        break;
      }
      auto effects = dyn_cast<MemoryEffectOpInterface>(next);
      if (!effects) {
        if (isMemoryEffectFree(next)) continue;
        break;
      }
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      bool read = false, written = false;
      for (const auto &effect : instances) {
        if (effect.getValue() && !llvm::is_contained(aliases, effect.getValue())) continue;
        written |= isa<MemoryEffects::Write>(effect.getEffect());
        read |= !isa<MemoryEffects::Write, MemoryEffects::Allocate>(effect.getEffect());
        if (!effect.getValue()) read = true;
      }
      if (auto select = dyn_cast<dsa::SelectOp>(next)) read |= llvm::is_contained(aliases, select.getCondition());
      if (read) break;
      if (!written) continue;
      bool complete = isa<dsa::FillOp, dsa::LoadTileOp, dsa::GatherRowsOp, dsa::GroupGatherRowsOp, dsa::TransposeOp, dsa::UnaryOp, dsa::BinaryOp, dsa::CompareOp,
                          dsa::CastOp, dsa::SelectOp, memref::CopyOp>(next);
      if (auto divide = dyn_cast<dsa::DivideCastOp>(next)) complete = divide.getOutput() == output;
      // A complete write to an alias may cover only a subview of the owner.
      if (complete) {
        auto original = cast<MemRefType>(output.getType());
        for (const auto &effect : instances) if (isa<MemoryEffects::Write>(effect.getEffect()) &&
            effect.getValue() && effect.getValue() != output && llvm::is_contained(aliases, effect.getValue())) {
          auto type = cast<MemRefType>(effect.getValue().getType());
          complete &= type.getNumElements() == original.getNumElements() && type.getLayout().isIdentity();
          Value alias = effect.getValue();
          while (alias != output) {
            auto view = alias.getDefiningOp<memref::ReinterpretCastOp>();
            if (!view || view.getStaticOffsets().front() != 0) { complete = false; break; }
            alias = view.getSource();
          }
        }
      }
      if (complete) fill.erase();
      break;
    }
  }
}
bool eliminateUnreadLocalWrites(func::FuncOp function) {
  SmallVector<memref::AllocaOp> allocations;
  function.walk([&](memref::AllocaOp allocation) {
    if (allocation.getType().getMemorySpaceAsInt() == dsa::nramSpace) allocations.push_back(allocation);
  });
  bool changed = false;
  for (auto allocation : allocations) {
    SmallVector<Value> aliases{allocation};
    for (unsigned i = 0; i < aliases.size(); ++i)
      for (Operation *user : aliases[i].getUsers())
        if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) aliases.push_back(view.getResult());
    SmallVector<Operation *> writes;
    bool unread = true;
    for (Value alias : aliases) for (Operation *user : alias.getUsers()) {
      if (isa<memref::ReinterpretCastOp>(user)) continue;
      bool write = false;
      if (auto fill = dyn_cast<dsa::FillOp>(user)) write = fill.getOutput() == alias;
      if (auto store = dyn_cast<memref::StoreOp>(user)) write = store.getMemref() == alias;
      if (auto copy = dyn_cast<memref::CopyOp>(user))
        write = copy.getTarget() == alias && !llvm::is_contained(aliases, copy.getSource());
      if (auto load = dyn_cast<dsa::LoadTileOp>(user))
        write = !load.getAsynchronous() && load.getOutput() == alias && !llvm::is_contained(aliases, load.getSource());
      if (auto cast = dyn_cast<dsa::CastOp>(user))
        write = cast.getOutput() == alias && !llvm::is_contained(aliases, cast.getInput());
      if (!write) unread = false;
      else if (!llvm::is_contained(writes, user)) writes.push_back(user);
    }
    if (!unread || writes.empty()) continue;
    for (Operation *write : writes) write->erase();
    for (Value alias : llvm::reverse(aliases)) if (alias.use_empty()) alias.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}
bool forwardFullLocalCopies(func::FuncOp function) {
  SmallVector<Operation *> copies;
  function.walk([&](Operation *op) {
    if (isa<memref::CopyOp, dsa::LoadTileOp>(op)) copies.push_back(op);
  });
  auto owner = [](Value value) {
    while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
    return value;
  };
  auto exact = [&](Value value, int64_t expected) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == expected && interval->second == expected;
  };
  auto aliases = [](Value allocation) {
    SmallVector<Value> values{allocation};
    for (unsigned i = 0; i < values.size(); ++i)
      for (Operation *user : values[i].getUsers())
        if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) values.push_back(view);
    return values;
  };
  bool changed = false;
  for (Operation *copy : copies) {
    Value source, destination;
    if (auto local = dyn_cast<memref::CopyOp>(copy)) {
      source = local.getSource(); destination = local.getTarget();
    } else {
      auto load = cast<dsa::LoadTileOp>(copy);
      if (load.getAsynchronous()) continue;
      source = load.getSource(); destination = load.getOutput();
      auto shape = cast<MemRefType>(destination.getType());
      if (!exact(load.getOffset(), 0) || !exact(load.getRows(), shape.getDimSize(0)) ||
          !exact(load.getColumns(), shape.getDimSize(1)) ||
          (shape.getDimSize(0) > 1 && !exact(load.getRowStride(), shape.getDimSize(1))) ||
          (shape.getDimSize(1) > 1 && !exact(load.getColumnStride(), 1))) continue;
    }
    auto input = dyn_cast<MemRefType>(source.getType());
    auto output = cast<MemRefType>(destination.getType());
    Value sourceOwner = owner(source);
    if (!input || !input.hasStaticShape() || !output.hasStaticShape() ||
        input.getMemorySpaceAsInt() != dsa::nramSpace || output.getMemorySpaceAsInt() != dsa::nramSpace ||
        !input.getLayout().isIdentity() || !output.getLayout().isIdentity() ||
        input.getElementType() != output.getElementType() || input.getNumElements() != output.getNumElements() ||
        !sourceOwner.getDefiningOp<memref::AllocaOp>() || !destination.getDefiningOp<memref::AllocaOp>() ||
        sourceOwner == destination) continue;
    auto destinationAliases = aliases(destination), sourceAliases = aliases(sourceOwner);
    Operation *last = copy;
    bool eligible = true;
    // A copied snapshot can be forwarded only to read-only consumers. Views
    // retain the entire owned allocation, as required by the DSA verifier.
    for (Value alias : destinationAliases) for (Operation *user : alias.getUsers()) {
      if (user == copy) continue;
      if (user->getBlock() != copy->getBlock() || !copy->isBeforeInBlock(user)) { eligible = false; break; }
      if (last->isBeforeInBlock(user)) last = user;
      if (isa<memref::ReinterpretCastOp>(user)) continue;
      if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) { eligible = false; break; }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || llvm::is_contained(destinationAliases, effect.getValue()))) eligible = false;
      } else if (!isMemoryEffectFree(user)) eligible = false;
    }
    if (!eligible || last == copy) continue;
    for (Value alias : sourceAliases) for (Operation *user : alias.getUsers()) {
      if (isa<memref::ReinterpretCastOp>(user)) continue;
      if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); }) ||
          (!isa<MemoryEffectOpInterface>(user) && !isMemoryEffectFree(user))) eligible = false;
    }
    // Preserve the source snapshot through its final redirected read, including
    // writes nested in intervening control flow and writes by that last user.
    for (Operation *op = copy->getNextNode(); eligible && op; op = op->getNextNode()) {
      op->walk([&](Operation *nested) {
        if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
          SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
          for (const auto &effect : instances)
            if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
                (!effect.getValue() || llvm::is_contained(sourceAliases, effect.getValue()))) eligible = false;
        }
      });
      if (op == last) break;
    }
    if (!eligible) continue;
    OpBuilder builder(copy);
    Value replacement = source;
    if (input != output)
      replacement = builder.create<memref::ReinterpretCastOp>(copy->getLoc(), output, source, int64_t(0),
          output.getShape(), ArrayRef<int64_t>{output.getDimSize(1), 1});
    destination.replaceAllUsesExcept(replacement, copy);
    copy->erase();
    if (destination.use_empty()) destination.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}
bool forwardUniformScalarLoads(func::FuncOp function) {
  DominanceInfo dominance(function);
  SmallVector<memref::AllocaOp> allocations;
  function.walk([&](memref::AllocaOp allocation) { allocations.push_back(allocation); });
  bool changed = false;
  for (auto allocation : allocations) {
    Value buffer = allocation.getResult();
    SmallVector<Value> aliases{buffer};
    auto owner = [](Value value) {
      while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
      return value;
    };
    dsa::FillOp initialization;
    SmallVector<memref::LoadOp> loads;
    bool immutable = true;
    for (unsigned i = 0; i < aliases.size() && immutable; ++i) for (Operation *user : aliases[i].getUsers()) {
      if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) {
        // DSA verification restricts these to same-dtype, zero-offset views
        // covering the entire owned allocation.
        aliases.push_back(view.getResult());
        continue;
      }
      if (auto fill = dyn_cast<dsa::FillOp>(user)) {
        if (initialization || fill.getOutput() != aliases[i]) { immutable = false; break; }
        initialization = fill;
        continue;
      }
      if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) {
        immutable = false;
        break;
      }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || owner(effect.getValue()) == buffer)) immutable = false;
      } else if (!isMemoryEffectFree(user)) immutable = false;
      if (!immutable) break;
      if (auto load = dyn_cast<memref::LoadOp>(user)) loads.push_back(load);
    }
    if (!immutable || !initialization || loads.empty() ||
        llvm::any_of(loads, [&](memref::LoadOp load) { return !dominance.dominates(initialization, load); })) continue;
    for (auto load : loads) {
      load.getResult().replaceAllUsesWith(initialization.getValue());
      load.erase();
    }
    if (llvm::all_of(aliases, [&](Value alias) {
          return llvm::all_of(alias.getUsers(), [&](Operation *user) {
            return user == initialization || isa<memref::ReinterpretCastOp>(user);
          });
        })) {
      initialization.erase();
      for (Value alias : llvm::reverse(aliases)) alias.getDefiningOp()->erase();
    }
    changed = true;
  }
  return changed;
}
bool forwardIndexExpressions(func::FuncOp function) {
  SmallVector<memref::StoreOp> stores;
  function.walk([&](memref::StoreOp store) { stores.push_back(store); });
  auto integer = [](Type type) { return isa<IndexType, IntegerType>(type); };
  bool changed = false;
  for (auto store : stores) {
    Value buffer = store.getMemref();
    auto allocation = buffer.getDefiningOp<memref::AllocaOp>();
    auto producer = dyn_cast<scf::ForOp>(store->getParentOp());
    auto type = store.getMemRefType();
    if (!allocation || !producer || allocation->getBlock() != producer->getBlock() ||
        !producer.getInitArgs().empty() || type.getRank() != 2 || type.getDimSize(0) != 1 ||
        !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
        (!type.getElementType().isInteger(32) && !type.getElementType().isInteger(64)) ||
        !matchPattern(producer.getLowerBound(), m_Zero()) || !matchPattern(producer.getStep(), m_One()) ||
        store.getIndices().size() != 2 || !matchPattern(store.getIndices()[0], m_Zero()) ||
        store.getIndices()[1] != producer.getInductionVar()) continue;
    // Replaying only integer arithmetic and immutable view strides avoids
    // moving data reads across intervening writes or duplicating FP work.
    bool pure = llvm::all_of(*producer.getBody(), [&](Operation &operation) {
      if (&operation == store.getOperation() || isa<scf::YieldOp>(operation)) return true;
      if (isa<dsa::StrideOp>(operation)) return true;
      return operation.getName().getDialectNamespace() == "arith" && !operation.getNumRegions() &&
          isMemoryEffectFree(&operation) && llvm::all_of(operation.getOperandTypes(), integer) &&
          llvm::all_of(operation.getResultTypes(), integer);
    });
    if (!pure) continue;
    SmallVector<memref::LoadOp> loads;
    bool uniqueWrite = true;
    for (Operation *user : buffer.getUsers()) {
      if (user == store.getOperation()) continue;
      if (auto fill = dyn_cast<dsa::FillOp>(user)) {
        if (fill->getBlock() != producer->getBlock() || !fill->isBeforeInBlock(producer)) uniqueWrite = false;
        continue;
      }
      if (auto load = dyn_cast<memref::LoadOp>(user)) { loads.push_back(load); continue; }
      if (llvm::any_of(user->getResultTypes(), [](Type result) { return isa<MemRefType>(result); })) {
        uniqueWrite = false; break;
      }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == buffer)) uniqueWrite = false;
      } else if (!isMemoryEffectFree(user)) uniqueWrite = false;
      if (!uniqueWrite) break;
    }
    if (!uniqueWrite || loads.empty()) continue;
    for (auto load : loads) {
      auto consumer = dyn_cast<scf::ForOp>(load->getParentOp());
      if (!consumer || consumer->getBlock() != producer->getBlock() || !producer->isBeforeInBlock(consumer) ||
          !consumer.getInitArgs().empty() || consumer.getUpperBound() != producer.getUpperBound() ||
          !matchPattern(consumer.getLowerBound(), m_Zero()) || !matchPattern(consumer.getStep(), m_One()) ||
          load.getIndices().size() != 2 || !matchPattern(load.getIndices()[0], m_Zero()) ||
          load.getIndices()[1] != consumer.getInductionVar()) continue;
      IRMapping mapping;
      mapping.map(producer.getInductionVar(), consumer.getInductionVar());
      OpBuilder builder(load);
      for (Operation &operation : *producer.getBody())
        if (&operation != store.getOperation() && !isa<scf::YieldOp>(operation)) builder.clone(operation, mapping);
      load.getResult().replaceAllUsesWith(mapping.lookupOrDefault(store.getValue()));
      load.erase();
      changed = true;
    }
    if (llvm::all_of(buffer.getUsers(), [&](Operation *user) {
          return user == store.getOperation() || isa<dsa::FillOp>(user);
        })) {
      producer.erase();
      SmallVector<Operation *> fills(buffer.getUsers());
      for (Operation *fill : fills) fill->erase();
      allocation.erase();
    }
  }
  return changed;
}
bool realizeRangeComparisons(func::FuncOp function) {
  DominanceInfo dominance(function);
  auto integer = [](Value value) -> std::optional<int64_t> {
    APInt bits;
    if (matchPattern(value, m_ConstantInt(&bits))) return bits.getSExtValue();
    return std::nullopt;
  };
  auto uniqueCopy = [&](Value buffer, Operation *consumer) -> Operation * {
    if (!buffer.getDefiningOp<memref::AllocaOp>()) return {};
    Operation *copy = nullptr;
    for (Operation *user : buffer.getUsers()) {
      if (llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); })) return {};
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances) {
          if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) ||
              (effect.getValue() && effect.getValue() != buffer)) continue;
          auto load = dyn_cast<dsa::LoadTileOp>(user);
          auto transfer = dyn_cast<memref::CopyOp>(user);
          bool known = (load && load.getOutput() == buffer && load.getSource() != buffer) ||
              (transfer && transfer.getTarget() == buffer && transfer.getSource() != buffer);
          if (!known || copy || !dominance.dominates(user, consumer)) return {};
          copy = user;
        }
      } else if (!isMemoryEffectFree(user)) return {};
    }
    return copy;
  };
  auto rampBase = [&](Value buffer, int64_t columns, Operation *consumer) -> std::optional<int64_t> {
    // Peel complete local copies; the remaining owned row must be initialized
    // by exactly one unit-step integer ramp, with no subsequent writers.
    while (Operation *copy = uniqueCopy(buffer, consumer)) {
      Value input;
      if (auto load = dyn_cast<dsa::LoadTileOp>(copy)) {
        if (!matchPattern(load.getOffset(), m_Zero()) || !matchPattern(load.getColumnStride(), m_One()) ||
            !matchPattern(load.getRows(), m_One()) || integer(load.getColumns()) != columns) return std::nullopt;
        input = load.getSource();
      } else input = cast<memref::CopyOp>(copy).getSource();
      auto source = cast<MemRefType>(input.getType());
      if (source.getMemorySpaceAsInt() != dsa::nramSpace || !source.getLayout().isIdentity() ||
          source.getShape() != ArrayRef<int64_t>({1, columns})) return std::nullopt;
      buffer = input;
      consumer = copy;
    }
    if (!buffer.getDefiningOp<memref::AllocaOp>()) return std::nullopt;
    memref::StoreOp store;
    SmallVector<dsa::FillOp> fills;
    for (Operation *user : buffer.getUsers()) {
      if (llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); })) return std::nullopt;
      if (auto write = dyn_cast<memref::StoreOp>(user); write && write.getMemref() == buffer) {
        if (store) return std::nullopt;
        store = write; continue;
      }
      if (auto fill = dyn_cast<dsa::FillOp>(user); fill && fill.getOutput() == buffer) { fills.push_back(fill); continue; }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == buffer)) return std::nullopt;
      } else if (!isMemoryEffectFree(user)) return std::nullopt;
    }
    auto loop = store ? dyn_cast<scf::ForOp>(store->getParentOp()) : scf::ForOp();
    if (!loop || !loop.getInitArgs().empty() || !dominance.dominates(loop, consumer) ||
        !matchPattern(loop.getLowerBound(), m_Zero()) || !matchPattern(loop.getStep(), m_One()) ||
        integer(loop.getUpperBound()) != columns || store.getIndices().size() != 2 ||
        !matchPattern(store.getIndices()[0], m_Zero()) || store.getIndices()[1] != loop.getInductionVar() ||
        llvm::any_of(fills, [&](dsa::FillOp fill) { return !dominance.dominates(fill, loop); })) return std::nullopt;
    Value value = store.getValue();
    while (auto cast = value.getDefiningOp<arith::IndexCastOp>()) value = cast.getIn();
    if (value == loop.getInductionVar()) return 0;
    if (auto sum = value.getDefiningOp<arith::AddIOp>()) {
      if (sum.getLhs() == loop.getInductionVar()) return integer(sum.getRhs());
      if (sum.getRhs() == loop.getInductionVar()) return integer(sum.getLhs());
    }
    return std::nullopt;
  };
  SmallVector<dsa::CompareOp> comparisons;
  function.walk([&](dsa::CompareOp compare) { comparisons.push_back(compare); });
  bool changed = false;
  for (auto compare : comparisons) {
    if (compare.getPredicate() != ComparePredicate::Ge) continue;
    auto type = cast<MemRefType>(compare.getLhs().getType());
    if (!type.getElementType().isInteger(64)) continue;
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    auto lhs = dyn_cast_or_null<dsa::LoadTileOp>(uniqueCopy(compare.getLhs(), compare));
    auto rhs = dyn_cast_or_null<dsa::LoadTileOp>(uniqueCopy(compare.getRhs(), compare));
    if (!lhs || !rhs || !matchPattern(lhs.getOffset(), m_Zero()) || !matchPattern(rhs.getOffset(), m_Zero()) ||
        !matchPattern(lhs.getRowStride(), m_One()) || !matchPattern(lhs.getColumnStride(), m_Zero()) ||
        !matchPattern(rhs.getRowStride(), m_Zero()) || !matchPattern(rhs.getColumnStride(), m_One()) ||
        lhs.getRows() != rhs.getRows() || integer(lhs.getColumns()) != columns || integer(rhs.getColumns()) != columns ||
        cast<MemRefType>(lhs.getSource().getType()).getShape() != ArrayRef<int64_t>({rows, 1}) ||
        cast<MemRefType>(rhs.getSource().getType()).getShape() != ArrayRef<int64_t>({1, columns})) continue;
    if (!uniqueCopy(lhs.getSource(), lhs)) continue;
    auto base = rampBase(rhs.getSource(), columns, rhs);
    if (!base || columns <= 0 || *base > std::numeric_limits<int64_t>::max() - (columns - 1)) continue;
    OpBuilder b(compare);
    Location loc = compare.getLoc();
    // Both broadcasts pad with zero: GE is true in their inactive rows too.
    b.create<dsa::CompareRangeOp>(loc, lhs.getSource(), compare.getOutput(), lhs.getRows(), b.getI64IntegerAttr(*base));
    compare.erase();
    for (auto copy : {lhs, rhs})
      if (llvm::all_of(copy.getOutput().getUsers(), [&](Operation *user) { return user == copy; })) copy.erase();
    changed = true;
  }
  return changed;
}
using SignedInterval = std::optional<std::pair<int64_t, int64_t>>;
SignedInterval integerInterval(Value value, func::FuncOp function) {
  DenseMap<Value, SignedInterval> cache;
  DenseSet<Value> active;
  auto checked = [](const APInt &low, const APInt &high, Type type) -> SignedInterval {
    unsigned width = type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
    if (width > 64 || !low.isSignedIntN(width) || !high.isSignedIntN(width) || low.sgt(high)) return std::nullopt;
    return std::make_pair(low.getSExtValue(), high.getSExtValue());
  };
  std::function<SignedInterval(Value)> analyze = [&](Value current) -> SignedInterval {
    if (auto known = cache.find(current); known != cache.end()) return known->second;
    if (!active.insert(current).second) return std::nullopt;
    auto infer = [&]() -> SignedInterval {
      APInt constant;
      if (matchPattern(current, m_ConstantInt(&constant)) && constant.isSignedIntN(64))
        return std::make_pair(constant.getSExtValue(), constant.getSExtValue());
      if (auto argument = dyn_cast<BlockArgument>(current)) {
        auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
        if (!loop || argument != loop.getInductionVar()) return std::nullopt;
        auto low = analyze(loop.getLowerBound()), high = analyze(loop.getUpperBound()), step = analyze(loop.getStep());
        if (!low || !high || !step || step->first <= 0 || high->second <= low->first) return std::nullopt;
        return std::make_pair(low->first, high->second - 1);
      }
      Operation *op = current.getDefiningOp();
      if (!op) return std::nullopt;
      auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
      if (isa<dsa::TaskIdOp>(op) && !function->hasAttr("intent_dsa.group_width"))
        return std::make_pair(int64_t(0), config.getTasks() - 1);
      if (isa<dsa::TaskCountOp>(op) && !function->hasAttr("intent_dsa.group_width"))
        return std::make_pair(config.getTasks(), config.getTasks());
      if (isa<arith::IndexCastOp, arith::ExtSIOp>(op)) {
        auto source = analyze(op->getOperand(0));
        return source ? checked(APInt(128, source->first, true), APInt(128, source->second, true), current.getType()) : std::nullopt;
      }
      if (op->getNumOperands() != 2) return std::nullopt;
      auto a = analyze(op->getOperand(0)), c = analyze(op->getOperand(1));
      if (!a || !c) return std::nullopt;
      APInt al(128, a->first, true), ah(128, a->second, true), cl(128, c->first, true), ch(128, c->second, true);
      if (isa<arith::AddIOp>(op)) return checked(al + cl, ah + ch, current.getType());
      if (isa<arith::SubIOp>(op)) return checked(al - ch, ah - cl, current.getType());
      if (isa<arith::MulIOp>(op)) {
        SmallVector<APInt> products{al * cl, al * ch, ah * cl, ah * ch};
        auto bounds = std::minmax_element(products.begin(), products.end(), [](const APInt &x, const APInt &y) { return x.slt(y); });
        return checked(*bounds.first, *bounds.second, current.getType());
      }
      if (isa<arith::MinSIOp>(op)) return std::make_pair(std::min(a->first, c->first), std::min(a->second, c->second));
      if (isa<arith::MaxSIOp>(op)) return std::make_pair(std::max(a->first, c->first), std::max(a->second, c->second));
      if (a->first >= 0 && c->first > 0 && c->first == c->second) {
        if (isa<arith::DivSIOp, arith::FloorDivSIOp>(op)) return std::make_pair(a->first / c->first, a->second / c->first);
        if (isa<arith::RemSIOp>(op)) return std::make_pair(int64_t(0), std::min(a->second, c->first - 1));
      }
      return std::nullopt;
    };
    SignedInterval result = infer(); active.erase(current); cache[current] = result; return result;
  };
  return analyze(value);
}
bool foldPresentRange(dsa::ReduceOp reduce, dsa::CompareRangeOp range, dsa::LoadTileOp copy,
                      func::FuncOp function, DominanceInfo &dominance) {
  auto exact = [&](Value value, int64_t expected) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == expected && interval->second == expected;
  };
  int64_t rows = cast<MemRefType>(range.getOutput().getType()).getDimSize(0);
  if (!copy || !exact(range.getRows(), rows) || !exact(copy.getRows(), rows) ||
      !exact(copy.getColumns(), 1) || !exact(copy.getOffset(), 0) ||
      !exact(copy.getRowStride(), 1) || !exact(copy.getColumnStride(), 1)) return false;
  Value source = copy.getSource();
  while (auto view = source.getDefiningOp<memref::ReinterpretCastOp>()) source = view.getSource();
  if (!source.getDefiningOp<memref::AllocaOp>() || cast<MemRefType>(source.getType()).getNumElements() != rows) return false;
  SmallVector<Value> aliases{source};
  SmallVector<dsa::FillOp> fills;
  memref::StoreOp initialization;
  for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) { aliases.push_back(view.getResult()); continue; }
    if (auto store = dyn_cast<memref::StoreOp>(user); store && llvm::is_contained(aliases, store.getMemref())) {
      if (initialization) return false;
      initialization = store; continue;
    }
    if (auto fill = dyn_cast<dsa::FillOp>(user); fill && llvm::is_contained(aliases, fill.getOutput())) { fills.push_back(fill); continue; }
    if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) return false;
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
      SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
      for (const auto &effect : instances)
        if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
            (!effect.getValue() || llvm::is_contained(aliases, effect.getValue()))) return false;
    } else if (!isMemoryEffectFree(user)) return false;
  }
  auto ramp = initialization ? dyn_cast<scf::ForOp>(initialization->getParentOp()) : scf::ForOp();
  if (!ramp || !ramp.getInitArgs().empty() || ramp->isProperAncestor(copy) ||
      !dominance.dominates(ramp, copy) || !exact(ramp.getLowerBound(), 0) ||
      !exact(ramp.getUpperBound(), rows) || !exact(ramp.getStep(), 1) ||
      llvm::any_of(fills, [&](dsa::FillOp fill) { return !dominance.dominates(fill, ramp); })) return false;
  auto indices = initialization.getIndices();
  auto initialized = initialization.getMemRefType();
  bool complete = indices.size() == 2 &&
      ((initialized.getShape() == ArrayRef<int64_t>({1, rows}) && exact(indices[0], 0) && indices[1] == ramp.getInductionVar()) ||
       (initialized.getShape() == ArrayRef<int64_t>({rows, 1}) && indices[0] == ramp.getInductionVar() && exact(indices[1], 0)));
  auto coordinates = integerInterval(initialization.getValue(), function);
  if (!complete || !coordinates || coordinates->first < range.getBaseAttr().getInt()) return false;
  memref::LoadOp count;
  for (Operation *user : reduce.getOutput().getUsers()) {
    if (user == reduce) continue;
    auto load = dyn_cast<memref::LoadOp>(user);
    if (!load || load.getMemref() != reduce.getOutput() || count) return false;
    count = load;
  }
  if (!count || !count.getResult().hasOneUse()) return false;
  auto compare = dyn_cast<arith::CmpFOp>(*count.getResult().getUsers().begin());
  if (!compare || !compare.getResult().hasOneUse() ||
      (compare.getPredicate() != arith::CmpFPredicate::UNE && compare.getPredicate() != arith::CmpFPredicate::ONE)) return false;
  Value other = compare.getLhs() == count.getResult() ? compare.getRhs() : compare.getLhs();
  FloatAttr zero;
  if (!matchPattern(other, m_Constant(&zero)) || !zero.getValue().isZero()) return false;
  auto store = dyn_cast<memref::StoreOp>(*compare.getResult().getUsers().begin());
  auto loop = store ? dyn_cast<scf::ForOp>(store->getParentOp()) : scf::ForOp();
  if (!loop || !loop.getInitArgs().empty() || count->getBlock() != loop.getBody() || compare->getBlock() != loop.getBody() ||
      !exact(loop.getLowerBound(), 0) || !exact(loop.getUpperBound(), rows) || !exact(loop.getStep(), 1) ||
      store.getMemRefType().getShape() != ArrayRef<int64_t>({1, rows}) || !store.getMemRefType().getElementType().isInteger(1) ||
      store.getMemRefType().getMemorySpaceAsInt() != dsa::nramSpace || !store.getMemRefType().getLayout().isIdentity() ||
      store.getIndices().size() != 2 || !exact(store.getIndices()[0], 0) || store.getIndices()[1] != loop.getInductionVar() ||
      count.getIndices().size() != 2 || !exact(count.getIndices()[0], 0) || count.getIndices()[1] != loop.getInductionVar()) return false;
  for (Operation &operation : loop.getBody()->without_terminator())
    if (&operation != count && &operation != store && (operation.getNumRegions() || !isMemoryEffectFree(&operation))) return false;
  OpBuilder b(loop);
  b.create<dsa::FillOp>(loop.getLoc(), store.getMemref(), b.create<arith::ConstantIntOp>(loop.getLoc(), 1, 1));
  loop.erase();
  return true;
}
bool foldUniformBooleanTiles(func::FuncOp function) {
  SmallVector<Operation *> operations;
  function.walk([&](Operation *operation) {
    if (isa<dsa::LoadTileOp, memref::CopyOp, dsa::SelectOp>(operation)) operations.push_back(operation);
  });
  bool changed = false;
  for (Operation *operation : operations) {
    OpBuilder b(operation);
    Location loc = operation->getLoc();
    if (auto select = dyn_cast<dsa::SelectOp>(operation)) {
      Value condition = select.getCondition();
      if (isa<MemRefType>(condition.getType())) {
        auto fill = uniformFillBefore(condition, select);
        if (!fill) continue;
        condition = fill.getValue();
      }
      APInt value;
      if (!matchPattern(condition, m_ConstantInt(&value))) continue;
      Value source = value.isZero() ? select.getFalseValue() : select.getTrueValue();
      if (!isa<MemRefType>(source.getType())) b.create<dsa::FillOp>(loc, select.getOutput(), source);
      else if (source != select.getOutput()) b.create<memref::CopyOp>(loc, source, select.getOutput());
      select.erase(); changed = true; continue;
    }
    Value input, output;
    if (auto copy = dyn_cast<memref::CopyOp>(operation)) { input = copy.getSource(); output = copy.getTarget(); }
    else { auto load = cast<dsa::LoadTileOp>(operation); input = load.getSource(); output = load.getOutput(); }
    auto type = cast<MemRefType>(output.getType());
    if (!type.getElementType().isInteger(1) || type.getRank() != 2 || !type.hasStaticShape() ||
        type.getNumElements() <= 0 || type.getMemorySpaceAsInt() != dsa::nramSpace || !type.getLayout().isIdentity()) continue;
    auto fill = uniformFillBefore(input, operation);
    if (!fill) continue;
    if (auto load = dyn_cast<dsa::LoadTileOp>(operation)) {
      auto rows = integerInterval(load.getRows(), function), columns = integerInterval(load.getColumns(), function);
      if (!rows || !columns || rows->first != type.getDimSize(0) || rows->second != type.getDimSize(0) ||
          columns->first != type.getDimSize(1) || columns->second != type.getDimSize(1)) continue;
    }
    b.create<dsa::FillOp>(loc, output, fill.getValue());
    operation->erase(); changed = true;
  }
  return changed;
}
bool realizeAffineRanges(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::CompareRangeOp> ranges;
  function.walk([&](dsa::CompareRangeOp range) { ranges.push_back(range); });
  DominanceInfo dominance(function);
  auto exact = [&](Value value, int64_t number) {
    auto interval = integerInterval(value, function);
    return interval && interval->first == number && interval->second == number;
  };
  bool changed = false;
  for (auto range : ranges) {
    auto type = cast<MemRefType>(range.getOutput().getType());
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    if (columns < 64 || !llvm::isPowerOf2_64(columns) || type.getNumElements() > 65536 || !exact(range.getRows(), rows)) continue;
    dsa::LoadTileOp copy;
    bool valid = true;
    for (Operation *user : range.getRowCoordinates().getUsers()) {
      if (user == range) continue;
      if (auto load = dyn_cast<dsa::LoadTileOp>(user); load && load.getOutput() == range.getRowCoordinates() && !copy) copy = load;
      else valid = false;
    }
    if (!valid || !copy || !exact(copy.getRows(), rows) || !exact(copy.getColumns(), 1) || !exact(copy.getOffset(), 0) ||
        !exact(copy.getRowStride(), 1) || !exact(copy.getColumnStride(), 1)) continue;
    Value source = copy.getSource();
    while (auto view = source.getDefiningOp<memref::ReinterpretCastOp>()) source = view.getSource();
    if (!source.getDefiningOp<memref::AllocaOp>() || cast<MemRefType>(source.getType()).getNumElements() != rows) continue;
    SmallVector<Value> aliases{source};
    SmallVector<dsa::FillOp> fills;
    memref::StoreOp initialization;
    for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
      if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) { aliases.push_back(view.getResult()); continue; }
      if (auto store = dyn_cast<memref::StoreOp>(user); store && llvm::is_contained(aliases, store.getMemref())) {
        if (initialization) valid = false;
        initialization = store; continue;
      }
      if (auto fill = dyn_cast<dsa::FillOp>(user); fill && llvm::is_contained(aliases, fill.getOutput())) { fills.push_back(fill); continue; }
      if (llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); })) valid = false;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || llvm::is_contained(aliases, effect.getValue()))) valid = false;
      } else if (!isMemoryEffectFree(user)) valid = false;
    }
    auto ramp = initialization ? dyn_cast<scf::ForOp>(initialization->getParentOp()) : scf::ForOp();
    if (!valid || !ramp || ramp->isProperAncestor(copy) || !dominance.dominates(ramp, copy) ||
        !exact(ramp.getLowerBound(), 0) || !exact(ramp.getUpperBound(), rows) || !exact(ramp.getStep(), 1) ||
        llvm::any_of(fills, [&](dsa::FillOp fill) { return !dominance.dominates(fill, ramp); })) continue;
    auto indices = initialization.getIndices();
    auto initialized = initialization.getMemRefType();
    bool complete = indices.size() == 2 &&
        ((initialized.getShape() == ArrayRef<int64_t>({1, rows}) && exact(indices[0], 0) && indices[1] == ramp.getInductionVar()) ||
         (initialized.getShape() == ArrayRef<int64_t>({rows, 1}) && indices[0] == ramp.getInductionVar() && exact(indices[1], 0)));
    if (!complete || !integerInterval(initialization.getValue(), function)) continue;
    std::function<std::optional<int64_t>(Value)> coefficient = [&](Value value) -> std::optional<int64_t> {
      if (value == ramp.getInductionVar()) return 1;
      auto *op = value.getDefiningOp();
      if (!op || !ramp->isProperAncestor(op)) return dominance.dominates(value, range) ? std::optional<int64_t>(0) : std::nullopt;
      if (auto cast = dyn_cast<arith::IndexCastOp>(op)) return coefficient(cast.getIn());
      if (auto cast = dyn_cast<arith::ExtSIOp>(op)) return coefficient(cast.getIn());
      if (!isa<arith::AddIOp, arith::SubIOp>(op)) return std::nullopt;
      auto a = coefficient(op->getOperand(0)), b = coefficient(op->getOperand(1));
      if (!a || !b) return std::nullopt;
      int64_t result = isa<arith::AddIOp>(op) ? *a + *b : *a - *b;
      return result >= -1 && result <= 1 ? std::optional<int64_t>(result) : std::nullopt;
    };
    auto slope = coefficient(initialization.getValue());
    if (!slope || *slope != 1) continue;
    OpBuilder b(range); Location loc = range.getLoc();
    IRMapping mapping;
    mapping.map(ramp.getInductionVar(), b.create<arith::ConstantIndexOp>(loc, 0));
    std::function<Value(Value)> atZero = [&](Value value) -> Value {
      if (mapping.contains(value)) return mapping.lookup(value);
      auto *op = value.getDefiningOp();
      if (!op || !ramp->isProperAncestor(op)) return value;
      for (Value operand : op->getOperands()) mapping.map(operand, atZero(operand));
      b.clone(*op, mapping);
      return mapping.lookup(value);
    };
    Value begin = atZero(initialization.getValue());
    if (!begin.getType().isIndex()) begin = b.create<arith::IndexCastOp>(loc, b.getIndexType(), begin);
    Value scratch = allocate(b, loc, b.getF32Type(), {1, type.getNumElements()}, dsa::nramSpace);
    auto replacement = b.create<dsa::CompareRampOp>(loc, begin, range.getOutput(), scratch, range.getBaseAttr());
    int64_t nram = 0, wram = 0; bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      replacement.erase(); scratch.getDefiningOp()->erase(); continue;
    }
    range.erase(); changed = true;
  }
  return changed;
}
bool foldRangeCounts(func::FuncOp function) {
  DominanceInfo dominance(function);
  auto soleWriter = [&](Value buffer, Operation *consumer) -> Operation * {
    if (!buffer.getDefiningOp<memref::AllocaOp>()) return nullptr;
    Operation *writer = nullptr;
    for (Operation *user : buffer.getUsers()) {
      if (user->getNumRegions() || llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); }))
        return nullptr;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances) {
          if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect())) continue;
          if (!effect.getValue()) return nullptr;
          if (effect.getValue() != buffer) continue;
          if (writer || !isa<MemoryEffects::Write>(effect.getEffect()) || !dominance.dominates(user, consumer)) return nullptr;
          writer = user;
        }
      } else if (!isMemoryEffectFree(user)) return nullptr;
    }
    return writer;
  };
  SmallVector<dsa::ReduceOp> reductions;
  function.walk([&](dsa::ReduceOp reduce) { reductions.push_back(reduce); });
  bool changed = false;
  for (auto reduce : reductions) {
    FloatAttr identity;
    if (reduce.getKind() != BinaryOperator::Add || !matchPattern(reduce.getIdentity(), m_Constant(&identity)) ||
        !identity.getValue().isZero() ||
        llvm::any_of(reduce.getScratch().getUsers(), [&](Operation *user) { return user != reduce; })) continue;
    Value converted = reduce.getInput();
    Operation *consumer = reduce;
    dsa::TransposeOp transpose;
    if (reduce.getAxis() == 0) {
      transpose = dyn_cast_or_null<dsa::TransposeOp>(soleWriter(converted, consumer));
      if (!transpose) continue;
      converted = transpose.getInput(); consumer = transpose;
    } else if (reduce.getAxis() != 1) continue;
    auto conversion = dyn_cast_or_null<dsa::CastOp>(soleWriter(converted, consumer));
    if (!conversion) continue;
    auto mask = cast<MemRefType>(conversion.getInput().getType());
    auto type = cast<MemRefType>(converted.getType());
    if (!mask.getElementType().isInteger(1) || !type.getElementType().isF32() || mask.getShape() != type.getShape()) continue;
    auto range = dyn_cast_or_null<dsa::CompareRangeOp>(soleWriter(conversion.getInput(), conversion));
    if (!range || !soleWriter(range.getRowCoordinates(), range)) continue;
    int64_t rows = mask.getDimSize(0), columns = mask.getDimSize(1);
    auto constant = [](Value value, int64_t expected) {
      auto op = value.getDefiningOp<arith::ConstantIndexOp>();
      return op && op.value() == expected;
    };
    if (columns > (1 << 24) || !constant(reduce.getCount(), columns) ||
        (transpose && (!constant(transpose.getRows(), rows) || !constant(transpose.getColumns(), columns)))) continue;
    auto output = cast<MemRefType>(reduce.getOutput().getType());
    if (!output.getLayout().isIdentity() || output.getNumElements() != rows ||
        (output.getDimSize(0) != 1 && output.getDimSize(1) != 1)) continue;
    if (foldPresentRange(reduce, range, dyn_cast_or_null<dsa::LoadTileOp>(soleWriter(range.getRowCoordinates(), range)), function, dominance)) {
      reduce.erase();
      if (transpose && llvm::all_of(transpose.getOutput().getUsers(), [&](Operation *user) { return user == transpose; })) transpose.erase();
      if (llvm::all_of(conversion.getOutput().getUsers(), [&](Operation *user) { return user == conversion; })) conversion.erase();
      changed = true;
      continue;
    }
    OpBuilder b(reduce);
    Location loc = reduce.getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0), one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value extent = b.create<arith::ConstantIndexOp>(loc, rows);
    Value base = b.create<arith::ConstantIntOp>(loc, range.getBase(), 64);
    Value last = b.create<arith::ConstantIntOp>(loc, range.getBase() + columns - 1, 64);
    Value width = b.create<arith::ConstantIntOp>(loc, columns, 64);
    Value empty = b.create<arith::ConstantIntOp>(loc, 0, 64);
    Value unit = b.create<arith::ConstantIntOp>(loc, 1, 64);
    b.create<scf::ForOp>(loc, zero, extent, one, ValueRange{}, [&](OpBuilder &nested, Location loc, Value row, ValueRange) {
      Value query = nested.create<memref::LoadOp>(loc, range.getRowCoordinates(), ValueRange{row, zero});
      Value below = nested.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, query, base);
      Value full = nested.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, query, last);
      Value count = nested.create<arith::SubIOp>(loc, query, base);
      count = nested.create<arith::AddIOp>(loc, count, unit);
      count = nested.create<arith::SelectOp>(loc, full, width, count);
      count = nested.create<arith::SelectOp>(loc, below, empty, count);
      Value active = nested.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, range.getRows());
      count = nested.create<arith::SelectOp>(loc, active, count, width);
      Value result = nested.create<arith::SIToFPOp>(loc, nested.getF32Type(), count);
      SmallVector<Value> indices = output.getDimSize(0) == 1 ? SmallVector<Value>{zero, row} : SmallVector<Value>{row, zero};
      nested.create<memref::StoreOp>(loc, result, reduce.getOutput(), indices);
      nested.create<scf::YieldOp>(loc);
    });
    reduce.erase();
    auto eraseDeadWriter = [](Operation *op, Value buffer) {
      if (llvm::all_of(buffer.getUsers(), [&](Operation *user) { return user == op; })) op->erase();
    };
    if (transpose) eraseDeadWriter(transpose, transpose.getOutput());
    eraseDeadWriter(conversion, conversion.getOutput());
    changed = true;
  }
  return changed;
}
bool bindRowScalarOperands(func::FuncOp function) {
  SmallVector<dsa::BinaryOp> binaries;
  function.walk([&](dsa::BinaryOp binary) { binaries.push_back(binary); });
  auto exact = [](Value value, int64_t expected) {
    APInt bits; return matchPattern(value, m_ConstantInt(&bits)) && bits.getSExtValue() == expected;
  };
  bool changed = false;
  for (auto binary : binaries) {
    auto type = cast<MemRefType>(binary.getOutput().getType());
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    if (rows <= 1 || rows >= 16 || columns < 1024 || binary.getApproximate() || binary.getFlushToZero() ||
        binary.getScratch() || !supportedScalarBinary(binary.getKind()) ||
        (!type.getElementType().isF32() && !type.getElementType().isF16())) continue;
    Value previous = binary.getRhs();
    if (!previous.getDefiningOp<memref::AllocaOp>()) continue;
    dsa::LoadTileOp broadcast;
    bool eligible = true;
    for (Operation *user : previous.getUsers()) {
      if (user == binary) continue;
      if (auto load = dyn_cast<dsa::LoadTileOp>(user); load && load.getOutput() == previous && !broadcast) broadcast = load;
      else eligible = false;
    }
    if (!eligible || !broadcast || broadcast->getBlock() != binary->getBlock() || !broadcast->isBeforeInBlock(binary) ||
        broadcast.getAsynchronous() || !exact(broadcast.getRows(), rows) || !exact(broadcast.getColumns(), columns) ||
        !exact(broadcast.getOffset(), 0) || !exact(broadcast.getRowStride(), 1) || !exact(broadcast.getColumnStride(), 0)) continue;
    Value source = broadcast.getSource();
    auto allocation = source.getDefiningOp<memref::AllocaOp>();
    if (!allocation || allocation.getType().getShape() != ArrayRef<int64_t>({1, rows}) ||
        allocation.getType().getElementType() != type.getElementType() || !allocation.getType().getLayout().isIdentity()) continue;
    for (Operation *user : source.getUsers())
      if (llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); }) ||
          (!isa<MemoryEffectOpInterface>(user) && !isMemoryEffectFree(user))) eligible = false;
    // Keep the captured row scalars unchanged between broadcast and use.
    for (Operation *op = broadcast->getNextNode(); op && op != binary; op = op->getNextNode()) {
      if (op->getNumRegions()) { eligible = false; break; }
      if (!llvm::is_contained(op->getOperands(), source)) continue;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == source)) eligible = false;
      } else if (!isMemoryEffectFree(op)) eligible = false;
    }
    if (!eligible) continue;
    binary.getRhsMutable().assign(source);
    binary->setAttr("bangc.implementation", StringAttr::get(function.getContext(), "row_scalar"));
    broadcast.erase();
    if (previous.use_empty()) previous.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}
bool realizeFullWidthMasks(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::CompareRampOp> comparisons;
  function.walk([&](dsa::CompareRampOp compare) {
    if (cast<MemRefType>(compare.getOutput().getType()).getElementType().isInteger(1)) comparisons.push_back(compare);
  });
  bool changed = false;
  for (auto compare : comparisons) {
    Value previous = compare.getOutput();
    if (!previous.getDefiningOp<memref::AllocaOp>()) continue;
    SmallVector<dsa::SelectOp> selections;
    bool eligible = true;
    for (Operation *user : previous.getUsers()) {
      if (user == compare) continue;
      auto select = dyn_cast<dsa::SelectOp>(user);
      if (!select || select.getCondition() != previous || !select.getFalseValue().getType().isF32() ||
          !cast<MemRefType>(select.getOutput().getType()).getElementType().isF32() ||
          user->getBlock() != compare->getBlock() || !compare->isBeforeInBlock(user)) { eligible = false; break; }
      selections.push_back(select);
    }
    if (!eligible || selections.empty()) continue;
    OpBuilder b(compare); Location loc = compare.getLoc();
    Value mask = allocate(b, loc, b.getI32Type(), cast<MemRefType>(previous.getType()).getShape(), dsa::nramSpace);
    auto packed = b.create<dsa::CompareRampOp>(loc, compare.getRowBegin(), mask, Value{}, compare.getBaseAttr());
    SmallVector<dsa::MaskedFillOp> replacements;
    for (auto select : selections) {
      b.setInsertionPoint(select);
      replacements.push_back(b.create<dsa::MaskedFillOp>(select.getLoc(), mask,
          select.getTrueValue(), select.getFalseValue(), select.getOutput()));
    }
    // Preflight with both representations present. Removing the old boolean
    // producers and conversions can only reduce their storage requirements.
    int64_t nram = 0, wram = 0; bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      for (auto replacement : replacements) replacement.erase();
      packed.erase(); mask.getDefiningOp()->erase(); continue;
    }
    SmallVector<Value> unused{previous, compare.getScratch()};
    for (auto select : selections) {
      if (select.getScratch() && !llvm::is_contained(unused, select.getScratch())) unused.push_back(select.getScratch());
      select.erase();
    }
    compare.erase();
    for (Value buffer : unused) if (buffer && buffer.use_empty()) buffer.getDefiningOp()->erase();
    changed = true;
  }
  return changed;
}
bool reuseGatherOffsets(func::FuncOp function) {
  DominanceInfo dominance(function);
  struct Candidate { scf::ForOp loop; memref::StoreOp store; };
  SmallVector<scf::ForOp> loops;
  SmallVector<Candidate> candidates;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getInitArgs().empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One())) continue;
    memref::StoreOp store;
    SmallVector<Value> sources;
    bool eligible = true;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto write = dyn_cast<memref::StoreOp>(op)) {
        if (store) { eligible = false; break; }
        store = write;
      } else if (auto read = dyn_cast<memref::LoadOp>(op)) sources.push_back(read.getMemref());
      else if (op.getNumRegions() || !isMemoryEffectFree(&op) ||
          (op.getName().getDialectNamespace() != "arith" && !isa<dsa::StrideOp>(op))) {
        eligible = false; break;
      }
    }
    if (!eligible || !store) continue;
    Value output = store.getMemref();
    auto type = store.getMemRefType();
    auto allocation = output.getDefiningOp<memref::AllocaOp>();
    if (!allocation || type.getRank() != 2 || type.getDimSize(0) != 1 || !type.getElementType().isInteger(64) ||
        type.getMemorySpaceAsInt() != dsa::nramSpace || store.getIndices().size() != 2 ||
        !matchPattern(store.getIndices()[0], m_Zero()) || store.getIndices()[1] != loop.getInductionVar()) continue;
    SmallVector<OpOperand *> consumers;
    SmallVector<dsa::FillOp> fills;
    for (Operation *user : output.getUsers()) {
      if (user == store.getOperation()) continue;
      if (auto fill = dyn_cast<dsa::FillOp>(user); fill && dominance.dominates(fill, loop)) {
        fills.push_back(fill); continue;
      }
      if (auto gather = dyn_cast<dsa::GatherRowsOp>(user); gather && gather.getRowOffsets() == output &&
          gather.getRows() == loop.getUpperBound() && dominance.dominates(loop, gather)) {
        consumers.push_back(&gather->getOpOperand(1)); continue;
      }
      if (auto gather = dyn_cast<dsa::GroupGatherRowsOp>(user); gather && gather.getRowOffsets() == output &&
          gather.getRows() == loop.getUpperBound() && dominance.dominates(loop, gather)) {
        consumers.push_back(&gather->getOpOperand(1)); continue;
      }
      if (auto plan = dyn_cast<dsa::GatherPlanOp>(user); plan && plan.getRowOffsets() == output &&
          plan.getRows() == loop.getUpperBound() && dominance.dominates(loop, plan)) {
        consumers.push_back(&plan->getOpOperand(0)); continue;
      }
      eligible = false; break;
    }
    if (!eligible || consumers.empty()) continue;
    bool reused = false;
    for (auto previous : candidates) {
      if (previous.loop->getBlock() != loop->getBlock() || previous.store.getMemref().getType() != type ||
          previous.loop.getUpperBound() != loop.getUpperBound()) continue;
      // A local coordinate tensor may be read by both expressions, but no
      // intervening mutation or escaping alias may change its contents.
      bool stable = llvm::all_of(sources, [&](Value source) {
        if (!source.getDefiningOp<memref::AllocaOp>()) return false;
        for (Operation *user : source.getUsers()) {
          if (llvm::any_of(user->getResultTypes(), [](Type t) { return isa<MemRefType>(t); })) return false;
          if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
            SmallVector<MemoryEffects::EffectInstance> instances;
            effects.getEffects(instances);
            for (const auto &effect : instances)
              if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
                  (!effect.getValue() || effect.getValue() == source) && !dominance.dominates(user, previous.loop))
                return false;
          } else if (!isMemoryEffectFree(user)) return false;
        }
        return true;
      });
      if (!stable || previous.loop.getBody()->getOperations().size() != loop.getBody()->getOperations().size()) continue;
      IRMapping mapping;
      mapping.map(previous.loop.getInductionVar(), loop.getInductionVar());
      mapping.map(previous.store.getMemref(), output);
      bool equivalent = true;
      for (auto [a, c] : llvm::zip(*previous.loop.getBody(), *loop.getBody())) {
        if (a.getName() != c.getName() || a.getAttrs() != c.getAttrs() || a.getResultTypes() != c.getResultTypes() ||
            a.getNumOperands() != c.getNumOperands() || a.getNumRegions() || c.getNumRegions()) {
          equivalent = false; break;
        }
        for (auto [lhs, rhs] : llvm::zip(a.getOperands(), c.getOperands()))
          if (mapping.lookupOrDefault(lhs) != rhs) { equivalent = false; break; }
        if (!equivalent) break;
        mapping.map(a.getResults(), c.getResults());
      }
      if (!equivalent) continue;
      for (OpOperand *consumer : consumers) consumer->set(previous.store.getMemref());
      loop.erase();
      for (auto fill : fills) fill.erase();
      allocation.erase();
      reused = changed = true;
      break;
    }
    if (!reused) candidates.push_back({loop, store});
  }
  return changed;
}
bool vectorizeIndexLoops(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  DenseMap<int64_t, Value> ramps;
  bool changed = false;
  auto integer64 = [](Type type) { return type.isIndex() || type.isInteger(64); };
  auto constant = [](Value value) -> std::optional<int64_t> {
    APInt bits;
    if (matchPattern(value, m_ConstantInt(&bits))) return bits.getSExtValue();
    return std::nullopt;
  };
  for (auto loop : loops) {
    if (!loop.getInitArgs().empty() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One())) continue;
    memref::StoreOp store;
    SmallVector<memref::LoadOp> loads;
    bool eligible = true;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto write = dyn_cast<memref::StoreOp>(op)) {
        if (store) { eligible = false; break; }
        store = write;
      } else if (auto read = dyn_cast<memref::LoadOp>(op)) loads.push_back(read);
      else if (isa<dsa::StrideOp>(op)) continue;
      else if (auto divide = dyn_cast<arith::FloorDivSIOp>(op)) {
        auto divisor = constant(divide.getRhs());
        if (!integer64(divide.getType()) || !divisor || *divisor <= 0 || !llvm::isPowerOf2_64(*divisor)) {
          eligible = false; break;
        }
      } else if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::IndexCastOp, arith::ConstantOp>(op) ||
                 !llvm::all_of(op.getOperandTypes(), integer64) || !llvm::all_of(op.getResultTypes(), integer64)) {
        eligible = false; break;
      }
    }
    if (!eligible || !store || !store.getValue().getType().isInteger(64)) continue;
    auto type = store.getMemRefType();
    if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 || type.getDimSize(1) < 256 ||
        !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
        !store.getMemref().getDefiningOp<memref::AllocaOp>() || store.getIndices().size() != 2 ||
        !matchPattern(store.getIndices()[0], m_Zero()) || store.getIndices()[1] != loop.getInductionVar()) continue;
    int64_t width = type.getDimSize(1);
    std::function<bool(Value)> bounded = [&](Value value) {
      if (auto number = constant(value)) return *number <= width;
      if (auto minimum = value.getDefiningOp<arith::MinSIOp>())
        return bounded(minimum.getLhs()) || bounded(minimum.getRhs());
      return false;
    };
    if (!bounded(loop.getUpperBound()) || llvm::any_of(loads, [&](memref::LoadOp load) {
          return load.getMemRefType() != type || !load.getMemref().getDefiningOp<memref::AllocaOp>() ||
              load.getMemref() == store.getMemref() || load.getIndices().size() != 2 ||
              !matchPattern(load.getIndices()[0], m_Zero()) || load.getIndices()[1] != loop.getInductionVar();
        })) continue;
    Operation *previous = loop->getPrevNode();
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, loop.getUpperBound(), zero);
    auto guard = b.create<scf::IfOp>(loc, active, false);
    b.setInsertionPointToStart(&guard.getThenRegion().front());
    Value ramp = ramps.lookup(width);
    dsa::IotaOp initialization;
    dsa::IndexLayoutOp rampLayout;
    Value interleavedRamp;
    if (!ramp) {
      OpBuilder init(function.getContext());
      init.setInsertionPointToStart(&function.front());
      interleavedRamp = allocate(init, loc, init.getI64Type(), {1, width}, dsa::nramSpace);
      initialization = init.create<dsa::IotaOp>(loc, interleavedRamp);
      ramp = allocate(init, loc, init.getI32Type(), {2, width}, dsa::nramSpace);
      rampLayout = init.create<dsa::IndexLayoutOp>(loc, interleavedRamp, ramp);
    }
    DenseMap<Value, Value> mapped;
    mapped[loop.getInductionVar()] = ramp;
    DenseMap<Value, Value> copiedInputs;
    auto tile = [&]() { return allocate(b, loc, b.getI32Type(), {2, width}, dsa::nramSpace); };
    auto materialize = [&](Value scalar) {
      Value result = tile();
      b.create<dsa::IndexLayoutOp>(loc, scalar, result);
      return result;
    };
    std::function<Value(Value)> emit = [&](Value value) -> Value {
      if (Value known = mapped.lookup(value)) return known;
      if (auto number = constant(value)) return mapped[value] = b.create<arith::ConstantIntOp>(loc, *number, 64);
      if (loop.isDefinedOutsideOfLoop(value)) {
        Value result = value.getType().isIndex() ? Value(b.create<arith::IndexCastOp>(loc, b.getI64Type(), value)) : value;
        return mapped[value] = result;
      }
      Operation *definition = value.getDefiningOp();
      if (auto load = dyn_cast<memref::LoadOp>(definition)) {
        Value source = load.getMemref();
        Value copied = copiedInputs.lookup(source);
        if (!copied) {
          Value interleaved = allocate(b, loc, b.getI64Type(), {1, width}, dsa::nramSpace);
          b.create<dsa::LoadTileOp>(loc, source, interleaved, zero, zero, one, one, loop.getUpperBound());
          copied = tile();
          b.create<dsa::IndexLayoutOp>(loc, interleaved, copied);
          copiedInputs[source] = copied;
        }
        return mapped[value] = copied;
      }
      if (auto cast = dyn_cast<arith::IndexCastOp>(definition)) return mapped[value] = emit(cast.getIn());
      if (auto stride = dyn_cast<dsa::StrideOp>(definition)) {
        Value scalar = b.create<dsa::StrideOp>(loc, b.getIndexType(), stride.getSource(), stride.getAxis());
        return mapped[value] = b.create<arith::IndexCastOp>(loc, b.getI64Type(), scalar);
      }
      Value lhs = emit(definition->getOperand(0)), rhs = emit(definition->getOperand(1));
      bool leftTile = isa<MemRefType>(lhs.getType()), rightTile = isa<MemRefType>(rhs.getType());
      if (!leftTile && !rightTile) {
        OperationState state(loc, definition->getName());
        state.addOperands({lhs, rhs}); state.addTypes(b.getI64Type()); state.addAttributes(definition->getAttrs());
        return mapped[value] = b.create(state)->getResult(0);
      }
      BinaryOperator kind = isa<arith::AddIOp>(definition) ? BinaryOperator::Add :
          isa<arith::SubIOp>(definition) ? BinaryOperator::Subtract :
          isa<arith::MulIOp>(definition) ? BinaryOperator::Multiply : BinaryOperator::RightShift;
      if (kind == BinaryOperator::RightShift)
        rhs = b.create<arith::ConstantIntOp>(loc, llvm::Log2_64(*constant(definition->getOperand(1))), 64);
      if (!leftTile && (kind == BinaryOperator::Add || kind == BinaryOperator::Multiply)) {
        std::swap(lhs, rhs); leftTile = true;
      }
      if (!leftTile) lhs = materialize(lhs);
      // Scalars are split into full-width low/high words by the selected
      // implementation; no signed-48-bit immediate restriction is needed.
      bool scalar = !isa<MemRefType>(rhs.getType());
      auto number = scalar ? constant(rhs) : std::optional<int64_t>{};
      if (kind == BinaryOperator::Multiply && number && *number > 0 && llvm::isPowerOf2_64(*number)) {
        kind = BinaryOperator::LeftShift;
        number = llvm::Log2_64(*number);
        rhs = b.create<arith::ConstantIntOp>(loc, *number, 64);
      }
      Value result = tile();
      b.create<dsa::IndexBinaryOp>(loc, lhs, rhs, result, BinaryOperatorAttr::get(b.getContext(), kind));
      return mapped[value] = result;
    };
    Value result = emit(store.getValue());
    if (!isa<MemRefType>(result.getType())) result = materialize(result);
    Value interleaved = allocate(b, loc, b.getI64Type(), {1, width}, dsa::nramSpace);
    b.create<dsa::IndexLayoutOp>(loc, result, interleaved);
    b.create<dsa::StoreTileOp>(loc, interleaved, store.getMemref(), zero, zero, one, one, loop.getUpperBound());
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram + 512 > config.getLocalBytes() || nram + 512 > 768 * 1024 || wram > 1024 * 1024) {
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
      if (initialization) {
        rampLayout.erase(); initialization.erase();
        ramp.getDefiningOp()->erase(); interleavedRamp.getDefiningOp()->erase();
      }
      continue;
    }
    ramps[width] = ramp;
    loop.erase();
    changed = true;
  }
  return changed;
}
bool normalizeLinearIndices(func::FuncOp function) {
  bool changed = false;
  function.walk([&](memref::StoreOp store) {
    Value root = store.getValue();
    if (auto cast = root.getDefiningOp<arith::IndexCastOp>()) root = cast.getIn();
    Type type = root.getType();
    if (!type.isIndex() && !type.isInteger(64)) return;
    auto definition = root.getDefiningOp();
    if (!definition || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(definition)) return;
    // Integer add/sub/mul are modular. Divisions and reads remain opaque SSA
    // leaves; cancellation never depends on overflow or runtime index bounds.
    SmallVector<std::pair<Value, APInt>> terms;
    APInt constant(64, 0);
    llvm::SmallPtrSet<Operation *, 16> arithmetic;
    std::function<void(Value, APInt)> collect = [&](Value value, APInt factor) {
      if (factor.isZero()) return;
      IntegerAttr literal;
      if (matchPattern(value, m_Constant(&literal))) {
        constant += literal.getValue().sextOrTrunc(64) * factor;
        return;
      }
      Operation *op = value.getDefiningOp();
      if (op && value.getType() == type) {
        if (auto add = dyn_cast<arith::AddIOp>(op)) {
          arithmetic.insert(op); collect(add.getLhs(), factor); collect(add.getRhs(), factor); return;
        }
        if (auto sub = dyn_cast<arith::SubIOp>(op)) {
          arithmetic.insert(op); collect(sub.getLhs(), factor); collect(sub.getRhs(), -factor); return;
        }
        if (auto mul = dyn_cast<arith::MulIOp>(op)) {
          if (matchPattern(mul.getRhs(), m_Constant(&literal))) {
            arithmetic.insert(op); collect(mul.getLhs(), factor * literal.getValue().sextOrTrunc(64)); return;
          }
          if (matchPattern(mul.getLhs(), m_Constant(&literal))) {
            arithmetic.insert(op); collect(mul.getRhs(), factor * literal.getValue().sextOrTrunc(64)); return;
          }
        }
      }
      for (auto &term : terms)
        if (term.first == value) { term.second += factor; return; }
      terms.emplace_back(value, factor);
    };
    collect(root, APInt(64, 1));
    int64_t count = constant.isZero() ? 0 : 1, multiplies = 0;
    for (const auto &term : terms) {
      if (term.second.isZero()) continue;
      ++count;
      multiplies += !term.second.isOne();
    }
    if (std::max<int64_t>(0, count - 1) + multiplies >= int64_t(arithmetic.size())) return;
    OpBuilder b(definition);
    auto literal = [&](const APInt &value) -> Value {
      return b.create<arith::ConstantOp>(store.getLoc(), type, b.getIntegerAttr(type, value));
    };
    Value result;
    for (const auto &term : terms) {
      if (term.second.isZero()) continue;
      Value value = term.first;
      if (!term.second.isOne()) value = b.create<arith::MulIOp>(store.getLoc(), value, literal(term.second));
      result = result ? b.create<arith::AddIOp>(store.getLoc(), result, value) : value;
    }
    if (!constant.isZero()) result = result ? b.create<arith::AddIOp>(store.getLoc(), result, literal(constant)) : literal(constant);
    if (!result) result = literal(APInt(64, 0));
    root.replaceAllUsesWith(result);
    changed = true;
  });
  return changed;
}
bool reuseConsumedBinaryInputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> operations;
  function.walk([&](dsa::BinaryOp operation) { operations.push_back(operation); });
  int64_t currentNram = 0, currentWram = 0;
  bindStorage(function, currentNram, currentWram);
  bool changed = false;
  for (auto operation : operations) {
    // These same-dtype native operations support exact output/input aliasing.
    // Division and operations with additional workspace retain separate storage.
    if (operation.getApproximate() || operation.getFlushToZero()) continue;
    if (!isa<MemRefType>(operation.getRhs().getType()) && !supportedScalarBinary(operation.getKind())) continue;
    switch (operation.getKind()) {
    case BinaryOperator::Add: case BinaryOperator::Subtract: case BinaryOperator::Multiply:
    case BinaryOperator::Maximum: case BinaryOperator::Minimum:
    case BinaryOperator::MaximumNum: case BinaryOperator::MinimumNum: break;
    default: continue;
    }
    Value input = operation.getLhs(), output = operation.getOutput();
    auto source = input.getDefiningOp<memref::AllocaOp>();
    auto destination = output.getDefiningOp<memref::AllocaOp>();
    if (!source || !destination || input == output || output == operation.getRhs() ||
        source->getBlock() != operation->getBlock() || destination->getBlock() != operation->getBlock() ||
        source.getType() != destination.getType() || !source.getType().getLayout().isIdentity() ||
        source.getType().getMemorySpaceAsInt() != dsa::nramSpace ||
        source.getAlignment().value_or(0) < destination.getAlignment().value_or(0)) continue;
    Type element = source.getType().getElementType();
    if (!element.isF16() && !element.isF32()) continue;
    if (auto loop = dyn_cast<scf::ForOp>(operation->getParentOp()))
      if (auto uniform = uniformFillBefore(input, operation);
          uniform && loop.isDefinedOutsideOfLoop(uniform.getValue())) continue;
    if (llvm::any_of(input.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !scope->isBeforeInBlock(operation) || user->getNumRegions() ||
              user->hasTrait<OpTrait::IsTerminator>() ||
              llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); });
        }) || llvm::any_of(output.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !operation->isBeforeInBlock(scope) || user->hasTrait<OpTrait::IsTerminator>();
        })) continue;
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : output.getUses()) uses.push_back(&use);
    for (OpOperand *use : uses) use->set(input);
    Operation *next = destination->getNextNode();
    destination->remove();
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > currentNram || wram > currentWram || nram > config.getLocalBytes() ||
        nram > 768 * 1024 || wram > 1024 * 1024) {
      OpBuilder restore(next);
      restore.insert(destination.getOperation());
      for (OpOperand *use : uses) use->set(output);
    } else {
      destination->destroy();
      currentNram = nram;
      currentWram = wram;
      changed = true;
    }
  }
  return changed;
}
bool reuseConsumedExp2Inputs(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::UnaryOp> operations;
  function.walk([&](dsa::UnaryOp op) {
    if (op.getKind() == UnaryOperator::Exp2 && op.getApproximate() && op.getFlushToZero()) operations.push_back(op);
  });
  bool changed = false;
  for (auto operation : operations) {
    Value input = operation.getInput(), output = operation.getOutput();
    auto source = input.getDefiningOp<memref::AllocaOp>(), destination = output.getDefiningOp<memref::AllocaOp>();
    if (!source || !destination || input == output || source.getType() != destination.getType() ||
        source->getBlock() != operation->getBlock() || destination->getBlock() != operation->getBlock() ||
        !source.getType().getElementType().isF32() || !source.getType().getLayout().isIdentity() ||
        source.getType().getMemorySpaceAsInt() != dsa::nramSpace ||
        source.getAlignment().value_or(0) < destination.getAlignment().value_or(0) ||
        (operation.getScratch() && !cast<MemRefType>(operation.getScratch().getType()).getElementType().isInteger(32))) continue;
    if (llvm::any_of(input.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !scope->isBeforeInBlock(operation) || user->getNumRegions() ||
              user->hasTrait<OpTrait::IsTerminator>() ||
              llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); });
        }) || llvm::any_of(output.getUsers(), [&](Operation *user) {
          if (user == operation.getOperation()) return false;
          Operation *scope = operation->getBlock()->findAncestorOpInBlock(*user);
          return !scope || !operation->isBeforeInBlock(scope) || user->hasTrait<OpTrait::IsTerminator>();
        })) continue;
    // compute_30 exp2 is one pow2.nram instruction. Its FTZ implementation
    // already supports exact input/output aliasing; the old input is dead here.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : output.getUses()) uses.push_back(&use);
    for (OpOperand *use : uses) use->set(input);
    Operation *next = destination->getNextNode();
    destination->remove();
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      OpBuilder restore(next); restore.insert(destination.getOperation());
      for (OpOperand *use : uses) use->set(output);
    } else {
      destination->destroy(); changed = true;
    }
  }
  return changed;
}
bool batchPointwiseTasks(func::FuncOp function) {
  auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  auto argumentWithAccess = [&](Value value, unsigned access) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == access;
  };
  auto constant = [](Value value) -> int64_t {
    auto index = value.getDefiningOp<arith::ConstantIndexOp>();
    return index ? index.value() : -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() ||
        !loop.getStep().getDefiningOp<dsa::TaskCountOp>() || !loop.getInitArgs().empty()) continue;
    Block *body = loop.getBody();
    Value induction = loop.getInductionVar();
    DenseMap<Value, bool> dependent;
    std::function<bool(Value)> dependsOnRow = [&](Value value) {
      if (value == induction) return true;
      auto known = dependent.find(value);
      if (known != dependent.end()) return known->second;
      Operation *definition = value.getDefiningOp();
      return dependent[value] = definition && definition->getBlock() == body &&
          llvm::any_of(definition->getOperands(), dependsOnRow);
    };
    SmallVector<memref::AllocaOp> allocations;
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    int64_t capacity = 0;
    bool eligible = true;
    for (Operation &operation : *body) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(operation)) {
        auto type = allocation.getType();
        if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 ||
            !type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != dsa::nramSpace ||
            (capacity && capacity != type.getDimSize(1)) ||
            llvm::any_of(allocation.getResult().getUsers(), [&](Operation *user) { return user->getBlock() != body; })) {
          eligible = false; break;
        }
        capacity = type.getDimSize(1);
        allocations.push_back(allocation);
      } else if (auto transfer = dyn_cast<dsa::LoadTileOp>(operation)) {
        if (load || !argumentWithAccess(transfer.getSource(), 0)) { eligible = false; break; }
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(operation)) {
        if (store || !argumentWithAccess(transfer.getDestination(), 1)) { eligible = false; break; }
        store = transfer;
      } else if (isa<dsa::FillOp, dsa::UnaryOp, dsa::BinaryOp, dsa::CastOp, dsa::CompareOp, dsa::SelectOp>(operation)) {
        if (auto unary = dyn_cast<dsa::UnaryOp>(operation); unary && unary.getScratch()) eligible = false;
        if (auto select = dyn_cast<dsa::SelectOp>(operation); select && select.getScratch()) eligible = false;
        for (Value operand : operation.getOperands()) {
          if (isa<MemRefType>(operand.getType())) {
            auto allocation = operand.getDefiningOp<memref::AllocaOp>();
            if (!allocation || allocation->getBlock() != body) eligible = false;
          } else if (dependsOnRow(operand)) eligible = false;
        }
        if (!eligible) break;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(operation)) {
        auto argument = dyn_cast<BlockArgument>(stride.getSource());
        if (!argument || argument.getOwner() != &function.front()) { eligible = false; break; }
      } else if (!isa<scf::YieldOp>(operation) &&
                 (operation.getNumRegions() || !isMemoryEffectFree(&operation) ||
                  operation.getName().getDialectNamespace() != "arith")) {
        eligible = false; break;
      }
    }
    if (!eligible || !load || !store || allocations.empty()) continue;
    int64_t columns = constant(load.getColumns());
    if (columns <= 0 || constant(store.getColumns()) != columns ||
        constant(load.getRows()) != 1 || constant(store.getRows()) != 1 ||
        capacity <= columns || capacity % columns) continue;
    auto matchesStride = [&](Value value, Value resource, int64_t axis) {
      if (auto stride = value.getDefiningOp<dsa::StrideOp>())
        return stride.getSource() == resource && stride.getAxis() == uint64_t(axis);
      auto argument = dyn_cast<BlockArgument>(resource);
      if (!argument || argument.getOwner() != &function.front()) return false;
      auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
      if (!view || !view.getConstraints().getHasStrides()) return false;
      auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]);
      APInt bits;
      return fixed && matchPattern(value, m_ConstantInt(&bits)) && bits.getSExtValue() == fixed.getInt();
    };
    auto rowCoefficient = [&](Value view, Value offset, Value columnStride) -> Value {
      auto type = cast<MemRefType>(view.getType());
      if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(1) != columns ||
          type.getDimSize(0) != constant(loop.getUpperBound())) return {};
      if (!matchesStride(columnStride, view, 1)) return {};
      auto product = offset.getDefiningOp<arith::MulIOp>();
      if (!product) return {};
      Value coefficient = product.getLhs() == induction ? product.getRhs() :
          product.getRhs() == induction ? product.getLhs() : Value{};
      return coefficient && matchesStride(coefficient, view, 0) ? coefficient : Value{};
    };
    Value inputPitch = rowCoefficient(load.getSource(), load.getOffset(), load.getColumnStride());
    Value outputPitch = rowCoefficient(store.getDestination(), store.getOffset(), store.getColumnStride());
    auto owned = [&](Value value) {
      auto allocation = value.getDefiningOp<memref::AllocaOp>();
      return allocation && llvm::is_contained(allocations, allocation);
    };
    if (!inputPitch || !outputPitch || !owned(load.getOutput()) || !owned(store.getInput())) continue;
    // The runtime rejects overlapping writable view arguments. With read-only
    // input, write-only output and no carried/local scalar state, these rows
    // can share one supply/compute/store while retaining each task's row order.
    int64_t rows = capacity / columns;
    int64_t tasks = config.getTasks(), upper = constant(loop.getUpperBound());
    if (tasks <= 0 || rows > (std::numeric_limits<int64_t>::max() - upper) / tasks) continue;
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    Value factor = builder.create<arith::ConstantIndexOp>(loc, rows);
    Value oldStep = loop.getStep();
    loop.getStepMutable().assign(builder.create<arith::MulIOp>(loc, oldStep, factor));
    builder.setInsertionPointToStart(body);
    Value remaining = builder.create<arith::SubIOp>(loc, loop.getUpperBound(), induction);
    Value count = builder.create<arith::CeilDivSIOp>(loc, remaining, oldStep);
    count = builder.create<arith::MinSIOp>(loc, count, factor);
    for (auto allocation : allocations) {
      auto type = allocation.getType();
      allocation.getResult().setType(MemRefType::get({rows, columns}, type.getElementType(),
          MemRefLayoutAttrInterface{}, type.getMemorySpace()));
    }
    builder.setInsertionPoint(load);
    load.getRowStrideMutable().assign(builder.create<arith::MulIOp>(loc, inputPitch, oldStep));
    load.getRowsMutable().assign(count);
    builder.setInsertionPoint(store);
    store.getRowStrideMutable().assign(builder.create<arith::MulIOp>(loc, outputPitch, oldStep));
    store.getRowsMutable().assign(count);
    changed = true;
  }
  return changed;
}
bool batchIndependentRowPrograms(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
  auto viewArgument = [&](Value value, unsigned access) -> dsa::ViewArgumentAttr {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return {};
    auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == access ? view : dsa::ViewArgumentAttr{};
  };
  auto constant = [](Value value) -> int64_t {
    APInt bits;
    return matchPattern(value, m_ConstantInt(&bits)) ? bits.getSExtValue() : -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  bool changed = false;
  for (auto loop : loops) {
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() ||
        !loop.getStep().getDefiningOp<dsa::TaskCountOp>() || !loop.getInitArgs().empty()) continue;
    auto *body = loop.getBody();
    Value induction = loop.getInductionVar();
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    SmallVector<memref::AllocaOp> allocations;
    SmallVector<dsa::ReduceOp> reductions;
    SmallVector<memref::LoadOp> scalarLoads;
    bool eligible = true;
    for (Operation &op : body->without_terminator()) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(op)) allocations.push_back(allocation);
      else if (auto transfer = dyn_cast<dsa::LoadTileOp>(op)) {
        if (load || transfer.getAsynchronous() || !viewArgument(transfer.getSource(), 0)) eligible = false;
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(op)) {
        if (store || !viewArgument(transfer.getDestination(), 1)) eligible = false;
        store = transfer;
      } else if (auto reduction = dyn_cast<dsa::ReduceOp>(op)) {
        auto kind = reduction.getKind();
        if (reduction.getAxis() != 1 || (kind != BinaryOperator::Add && kind != BinaryOperator::Maximum &&
            kind != BinaryOperator::MaximumNum && kind != BinaryOperator::Minimum && kind != BinaryOperator::MinimumNum)) eligible = false;
        reductions.push_back(reduction);
      } else if (auto read = dyn_cast<memref::LoadOp>(op)) {
        if (!read.getType().isF32() || read.getMemRefType().getShape() != ArrayRef<int64_t>({1, 1}) ||
            llvm::any_of(read.getIndices(), [&](Value index) { return constant(index) != 0; })) eligible = false;
        for (Operation *user : read.getResult().getUsers()) {
          if (auto fill = dyn_cast<dsa::FillOp>(user); fill && fill.getValue() == read.getResult()) continue;
          if (auto binary = dyn_cast<dsa::BinaryOp>(user); binary && binary.getRhs() == read.getResult()) continue;
          eligible = false;
        }
        scalarLoads.push_back(read);
      } else if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
        if (unary.getScratch()) eligible = false;
      } else if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
        if (binary.getScratch()) eligible = false;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(op)) {
        if (!isa<BlockArgument>(stride.getSource())) eligible = false;
      } else if (!isa<dsa::FillOp, dsa::CastOp, memref::CopyOp>(op) &&
                 (op.getNumRegions() || op.getName().getDialectNamespace() != "arith" || !isMemoryEffectFree(&op))) eligible = false;
    }
    if (!eligible || !load || !store || reductions.empty()) continue;
    int64_t columns = constant(load.getColumns()), upper = constant(loop.getUpperBound());
    if (columns < 1024 || columns % 32 || columns != constant(store.getColumns()) ||
        constant(load.getRows()) != 1 || constant(store.getRows()) != 1 || upper <= 0) continue;
    auto owned = [&](Value value) {
      auto allocation = value.getDefiningOp<memref::AllocaOp>();
      return allocation && llvm::is_contained(allocations, allocation);
    };
    for (auto allocation : allocations) {
      auto type = allocation.getType();
      if (type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != 1 ||
          (type.getDimSize(1) != 1 && type.getDimSize(1) != columns) || !type.getLayout().isIdentity() ||
          type.getMemorySpaceAsInt() != dsa::nramSpace ||
          llvm::any_of(allocation.getResult().getUsers(), [&](Operation *user) { return user->getBlock() != body; })) eligible = false;
    }
    for (Operation &op : body->without_terminator()) {
      if (isa<dsa::LoadTileOp, dsa::StoreTileOp, dsa::StrideOp>(op)) continue;
      for (Value operand : op.getOperands()) if (isa<MemRefType>(operand.getType()) && !owned(operand)) eligible = false;
    }
    for (auto reduction : reductions) {
      if (cast<MemRefType>(reduction.getInput().getType()).getShape() != ArrayRef<int64_t>({1, columns}) ||
          !cast<MemRefType>(reduction.getInput().getType()).getElementType().isF32() || constant(reduction.getCount()) != columns ||
          !reduction.getScratch().hasOneUse() || reduction.getScratch() == reduction.getInput() ||
          reduction.getScratch() == reduction.getOutput()) eligible = false;
    }
    DenseMap<Value, bool> varying;
    std::function<bool(Value)> depends = [&](Value value) {
      if (value == induction || value.getDefiningOp<dsa::TaskIdOp>()) return true;
      if (auto found = varying.find(value); found != varying.end()) return found->second;
      auto *definition = value.getDefiningOp();
      return varying[value] = definition && llvm::any_of(definition->getOperands(), depends);
    };
    for (Operation &op : body->without_terminator()) if (isa<dsa::FillOp, dsa::UnaryOp, dsa::BinaryOp, dsa::ReduceOp>(op))
      for (Value operand : op.getOperands()) if (!isa<MemRefType>(operand.getType()) && depends(operand)) eligible = false;
    auto pitch = [&](Value resource, Value offset, Value columnStride, unsigned access) -> Value {
      auto type = cast<MemRefType>(resource.getType());
      auto view = viewArgument(resource, access);
      if (!view || type.getRank() != 2 || !type.hasStaticShape() || type.getDimSize(0) != upper || type.getDimSize(1) != columns) return {};
      auto matches = [&](Value value, int axis) {
        if (auto stride = value.getDefiningOp<dsa::StrideOp>()) return stride.getSource() == resource && stride.getAxis() == uint64_t(axis);
        if (!view.getConstraints().getHasStrides()) return false;
        auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]);
        return fixed && constant(value) == fixed.getInt();
      };
      auto product = offset.getDefiningOp<arith::MulIOp>();
      if (!product || !matches(columnStride, 1)) return {};
      Value coefficient = product.getLhs() == induction ? product.getRhs() : product.getRhs() == induction ? product.getLhs() : Value{};
      return coefficient && !depends(coefficient) && matches(coefficient, 0) ? coefficient : Value{};
    };
    Value inputPitch = pitch(load.getSource(), load.getOffset(), load.getColumnStride(), 0);
    Value outputPitch = pitch(store.getDestination(), store.getOffset(), store.getColumnStride(), 1);
    if (!eligible || !inputPitch || !outputPitch || !owned(load.getOutput()) || !owned(store.getInput())) continue;
    for (int64_t rows : {8, 4, 2}) {
      if (config.getTasks() <= 0 || config.getTasks() > upper / rows ||
          upper % (rows * config.getTasks()) || columns > 65536 / rows) continue;
      OpBuilder b(loop); Location loc = loop.getLoc();
      Value rowCount = b.create<arith::ConstantIndexOp>(loc, rows);
      Value batches = b.create<arith::ConstantIndexOp>(loc, upper / rows);
      auto batch = b.create<scf::ForOp>(loc, loop.getLowerBound(), batches, loop.getStep());
      b.setInsertionPointToStart(batch.getBody());
      // Independent logical rows are partitioned into adjacent row blocks.
      // A task owns whole blocks, so each transfer has the original row pitch.
      Value firstRow = b.create<arith::MulIOp>(loc, batch.getInductionVar(), rowCount);
      IRMapping mapping; mapping.map(induction, firstRow);
      auto mapped = [&](Value value) { return mapping.lookupOrDefault(value); };
      auto broadcast = [&](Value source, Value destination) {
        auto type = cast<MemRefType>(destination.getType());
        if (source.getType() == destination.getType()) b.create<memref::CopyOp>(loc, source, destination);
        else {
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0), one = b.create<arith::ConstantIndexOp>(loc, 1);
          Value width = b.create<arith::ConstantIndexOp>(loc, type.getDimSize(1));
          b.create<dsa::LoadTileOp>(loc, source, destination, zero, one, zero, rowCount, width, b.getBoolAttr(false));
        }
      };
      for (Operation &op : body->without_terminator()) {
        if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
          auto type = allocation.getType();
          bool scratch = llvm::any_of(reductions, [&](dsa::ReduceOp reduce) { return reduce.getScratch() == allocation; });
          SmallVector<int64_t> shape = scratch ? SmallVector<int64_t>{1, columns} :
              type.getDimSize(1) == 1 ? SmallVector<int64_t>{1, rows} : SmallVector<int64_t>{rows, columns};
          mapping.map(allocation, allocate(b, loc, type.getElementType(), shape, dsa::nramSpace));
        } else if (auto read = dyn_cast<memref::LoadOp>(op)) mapping.map(read.getResult(), mapped(read.getMemref()));
        else if (auto fill = dyn_cast<dsa::FillOp>(op); fill && isa<MemRefType>(mapped(fill.getValue()).getType()))
          broadcast(mapped(fill.getValue()), mapped(fill.getOutput()));
        else if (auto binary = dyn_cast<dsa::BinaryOp>(op); binary && !isa<MemRefType>(binary.getRhs().getType()) && isa<MemRefType>(mapped(binary.getRhs()).getType())) {
          Value rhs = mapped(binary.getRhs()), output = mapped(binary.getOutput());
          if (rhs.getType() != output.getType()) {
            auto type = cast<MemRefType>(output.getType());
            Value expanded = allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
            broadcast(rhs, expanded); rhs = expanded;
          }
          b.create<dsa::BinaryOp>(loc, mapped(binary.getLhs()), rhs, output, binary.getKindAttr(),
              binary.getApproximateAttr(), binary.getFlushToZeroAttr(), Value{});
        } else {
          Operation *cloned = b.clone(op, mapping);
          if (&op == load.getOperation()) {
            auto transfer = cast<dsa::LoadTileOp>(cloned);
            transfer.getRowsMutable().assign(rowCount);
            transfer.getRowStrideMutable().assign(mapped(inputPitch));
          } else if (&op == store.getOperation()) {
            auto transfer = cast<dsa::StoreTileOp>(cloned);
            transfer.getRowsMutable().assign(rowCount);
            transfer.getRowStrideMutable().assign(mapped(outputPitch));
          }
        }
      }
      int64_t nram = 0, wram = 0; bindStorage(function, nram, wram);
      if (nram + 32768 > config.getLocalBytes() || nram + 32768 > 768 * 1024 || wram > 1024 * 1024) {
        batch.erase(); batches.getDefiningOp()->erase(); rowCount.getDefiningOp()->erase(); continue;
      }
      loop.erase(); changed = true; break;
    }
  }
  return changed;
}
void hoistInvariantFills(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<memref::AllocaOp> allocations;
  function.walk([&](memref::AllocaOp allocation) { allocations.push_back(allocation); });
  DominanceInfo dominance(function);
  int64_t currentNram = 0, currentWram = 0;
  bindStorage(function, currentNram, currentWram);
  for (auto allocation : allocations) {
    auto loop = dyn_cast<scf::ForOp>(allocation->getParentOp());
    if (!loop || allocation->getBlock() != loop.getBody()) continue;
    Value buffer = allocation.getResult();
    dsa::FillOp initialization;
    bool immutable = true;
    for (Operation *user : buffer.getUsers()) {
      if (auto fill = dyn_cast<dsa::FillOp>(user)) {
        if (initialization || fill->getBlock() != loop.getBody()) { immutable = false; break; }
        initialization = fill;
        continue;
      }
      if (llvm::any_of(user->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) {
        immutable = false; break;
      }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(user)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances)
          if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()) &&
              (!effect.getValue() || effect.getValue() == buffer)) immutable = false;
      } else if (!isMemoryEffectFree(user)) immutable = false;
      if (!immutable) break;
    }
    if (!immutable || !initialization || buffer.hasOneUse() || !loop.isDefinedOutsideOfLoop(initialization.getValue()) ||
        llvm::any_of(buffer.getUsers(), [&](Operation *user) {
          return user != initialization && !dominance.dominates(initialization, user);
        })) continue;
    Operation *allocationNext = allocation->getNextNode(), *initializationNext = initialization->getNextNode();
    allocation->moveBefore(loop);
    initialization->moveBefore(loop);
    // Sharing the immutable tile across iterations extends its live range.
    // Retain a move only when it does not increase the program's peak storage.
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > currentNram || wram > currentWram ||
        nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      initialization->moveBefore(initializationNext);
      allocation->moveBefore(allocationNext);
    } else { currentNram = nram; currentWram = wram; }
  }
}
void realizeRoundedDivisions(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> divisions;
  DenseMap<int64_t, Value> indicesByWidth;
  function.walk([&](dsa::BinaryOp operation) {
    if (operation.getKind() == BinaryOperator::TrueDivide && !operation.getApproximate() && !operation.getFlushToZero())
      divisions.push_back(operation);
  });
  auto owner = [](Value value) {
    while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
    return value.getDefiningOp<memref::AllocaOp>() ? value : Value{};
  };
  for (auto division : divisions) {
    auto type = cast<MemRefType>(division.getOutput().getType());
    if (!type.getElementType().isF32() || type.getNumElements() < 256 || type.getNumElements() % 32 ||
        !owner(division.getOutput()) || !owner(division.getLhs()) || !owner(division.getRhs()) ||
        owner(division.getOutput()) == owner(division.getLhs()) || owner(division.getOutput()) == owner(division.getRhs())) continue;
    auto uniform = uniformFillBefore(division.getRhs(), division);
    Value divisor = uniform ? uniform.getValue() : division.getRhs();
    for (int64_t width = std::min<int64_t>(4096, type.getNumElements()); width >= 64; width /= 2) {
      OpBuilder builder(division);
      Location loc = division.getLoc();
      Value scratch = allocate(builder, loc, builder.getI32Type(), {12, width}, dsa::nramSpace);
      Value indices = indicesByWidth.lookup(width);
      dsa::IotaOp initialize;
      if (!indices) {
        OpBuilder initializer(function.getContext());
        initializer.setInsertionPointToStart(&function.front());
        indices = allocate(initializer, loc, initializer.getF32Type(), {1, width}, dsa::nramSpace);
        initialize = initializer.create<dsa::IotaOp>(loc, indices);
      }
      auto replacement = builder.create<dsa::DivideRNOp>(loc, division.getLhs(), divisor, division.getOutput(), scratch, indices);
      int64_t nram = 0, wram = 0;
      bindStorage(function, nram, wram);
      if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
        replacement.erase(); scratch.getDefiningOp()->erase();
        if (initialize) { initialize.erase(); indices.getDefiningOp()->erase(); }
        continue;
      }
      indicesByWidth[width] = indices;
      Value previous = division.getRhs();
      division.erase();
      if (uniform && llvm::all_of(previous.getUsers(), [&](Operation *user) {
            auto fill = dyn_cast<dsa::FillOp>(user);
            return fill && fill.getOutput() == previous;
          })) {
        SmallVector<Operation *> fills(previous.getUsers());
        for (Operation *fill : fills) fill->erase();
        previous.getDefiningOp()->erase();
      }
      break;
    }
  }
}
void fuseNarrowDivisions(func::FuncOp function, dsa::ConfigurationAttr config) {
  SmallVector<dsa::BinaryOp> divisions;
  DenseMap<int64_t, Value> laneIndicesBySize;
  function.walk([&](dsa::BinaryOp op) {
    if (op.getKind() == BinaryOperator::TrueDivide && !op.getApproximate() && !op.getFlushToZero())
      divisions.push_back(op);
  });
  for (auto division : divisions) {
    auto owner = [](Value value) {
      while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
      return value.getDefiningOp<memref::AllocaOp>() ? value : Value{};
    };
    Value quotient = division.getOutput();
    auto type = cast<MemRefType>(quotient.getType());
    if (!type.getElementType().isF32() || type.getNumElements() < 128 || type.getNumElements() % 32 ||
        type.getNumElements() > (1 << 24) ||
        !quotient.getDefiningOp<memref::AllocaOp>() || !owner(division.getLhs()) || !owner(division.getRhs()) ||
        quotient == owner(division.getLhs()) || quotient == owner(division.getRhs()))
      continue;
    dsa::CastOp narrow;
    bool unique = true;
    for (Operation *user : quotient.getUsers()) {
      if (user == division) continue;
      auto cast = dyn_cast<dsa::CastOp>(user);
      if (!cast || cast.getInput() != quotient || narrow) { unique = false; break; }
      narrow = cast;
    }
    if (!unique || !narrow || narrow->getBlock() != division->getBlock() ||
        !cast<MemRefType>(narrow.getOutput().getType()).getElementType().isF16()) continue;
    Value outputOwner = owner(narrow.getOutput());
    if (!outputOwner || outputOwner == owner(division.getLhs()) || outputOwner == owner(division.getRhs())) continue;
    // No quotient consumer or intervening effect may observe the internal
    // approximation. The fused implementation certifies the final f16 value.
    Operation *next = division->getNextNode();
    for (; next && next != narrow.getOperation(); next = next->getNextNode())
      if (!isa<memref::AllocaOp>(next) && !isMemoryEffectFree(next)) break;
    if (next != narrow.getOperation()) continue;
    OpBuilder builder(narrow);
    Location loc = narrow.getLoc();
    Value bounds = allocate(builder, loc, builder.getF32Type(), type.getShape(), dsa::nramSpace);
    Value accepted = allocate(builder, loc, builder.getF32Type(), type.getShape(), dsa::nramSpace);
    Value narrowBounds = allocate(builder, loc, builder.getF16Type(), type.getShape(), dsa::nramSpace);
    int64_t lanes = type.getNumElements();
    Value laneIndices = laneIndicesBySize.lookup(lanes);
    dsa::IotaOp initializeIndices;
    if (!laneIndices) {
      OpBuilder initializer(function.getContext());
      initializer.setInsertionPointToStart(&function.front());
      laneIndices = allocate(initializer, loc, initializer.getF32Type(), {1, lanes}, dsa::nramSpace);
      initializeIndices = initializer.create<dsa::IotaOp>(loc, laneIndices);
    }
    auto uniform = uniformFillBefore(division.getRhs(), division);
    Value divisor = uniform ? uniform.getValue() : division.getRhs();
    auto fused = builder.create<dsa::DivideCastOp>(loc, division.getLhs(), divisor, narrow.getOutput(),
        quotient, bounds, accepted, narrowBounds, laneIndices);
    // Include the extended input lifetimes in the same allocation calculation
    // used for the final program. Retain ordinary divide/cast when the selected
    // local implementation would exceed the task's storage budget.
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      fused.erase();
      narrowBounds.getDefiningOp()->erase();
      accepted.getDefiningOp()->erase();
      bounds.getDefiningOp()->erase();
      if (initializeIndices) {
        initializeIndices.erase();
        laneIndices.getDefiningOp()->erase();
      }
      continue;
    }
    laneIndicesBySize[lanes] = laneIndices;
    Value previousDivisor = division.getRhs();
    narrow.erase();
    division.erase();
    if (uniform && llvm::all_of(previousDivisor.getUsers(), [&](Operation *user) {
          auto fill = dyn_cast<dsa::FillOp>(user);
          return fill && fill.getOutput() == previousDivisor;
        })) {
      SmallVector<Operation *> unusedFills(previousDivisor.getUsers());
      for (Operation *fill : unusedFills) fill->erase();
      previousDivisor.getDefiningOp()->erase();
    }
  }
}
// Contiguous views retain the allocation's ownership. Payload lifetime begins
// at its first use; pointer declarations and single-entry conditionals do not
// extend it. Enclosing loops retain the complete backedge lifetime.
void bindStorage(func::FuncOp function, int64_t &nram, int64_t &wram) {
  DenseMap<Operation *, uint64_t> begin, end;
  uint64_t clock = 0;
  std::function<void(Operation *)> number = [&](Operation *op) {
    begin[op] = clock++;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block) number(&nested);
    end[op] = clock++;
  };
  number(function);
  struct Range { uint64_t finish; int64_t offset, bytes, space; };
  struct AllocationRange { memref::AllocaOp allocation; uint64_t start, finish; };
  SmallVector<AllocationRange> allocations;
  SmallVector<Range> live;
  int64_t shared = 0;
  function.walk<WalkOrder::PreOrder>([&](memref::AllocaOp allocation) {
    uint64_t start = std::numeric_limits<uint64_t>::max(), finish = end[allocation];
    auto scope = [&](Operation *operation) {
      // An asynchronous use can complete after the conditional that owns its
      // buffer. That completion is already outside this lexical allocation.
      if (!allocation->getParentRegion()->isAncestor(operation->getParentRegion())) return operation;
      Operation *lifetime = operation;
      while (operation->getBlock() != allocation->getBlock()) {
        operation = operation->getParentOp();
        if (!isa<scf::IfOp>(operation)) lifetime = operation;
      }
      return lifetime;
    };
    SmallVector<Value> aliases{allocation.getResult()};
    for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
      if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) { aliases.push_back(view.getResult()); continue; }
      start = std::min(start, begin[scope(user)]);
      Operation *last = user;
      bool asynchronous = false;
      if (auto load = dyn_cast<dsa::LoadTileOp>(user)) asynchronous = load.getAsynchronous();
      if (auto gather = dyn_cast<dsa::GatherRowsOp>(user)) asynchronous = gather.getAsynchronous();
      if (asynchronous) {
        Operation *cursor = user;
        while (true) {
          Operation *completion = cursor->getNextNode();
          auto completesIO = [](Operation *op) {
            if (auto sync = dyn_cast<dsa::SynchronizeOp>(op)) return !sync.getLocalOnly();
            return isa<dsa::GroupSynchronizeOp, dsa::GroupGatherRowsOp>(op);
          };
          while (completion && !completesIO(completion)) completion = completion->getNextNode();
          if (completion) { last = completion; break; }
          Operation *parent = cursor->getParentOp();
          if (isa<scf::IfOp>(parent)) { cursor = parent; continue; }
          // Do not cross a loop backedge looking for a later fence. Keep the
          // payload live through the entire loop if this iteration has none.
          last = isa<func::FuncOp>(parent) ? cursor->getBlock()->getTerminator() : parent;
          break;
        }
      }
      finish = std::max(finish, end[scope(last)]);
    }
    if (start == std::numeric_limits<uint64_t>::max()) start = begin[allocation];
    allocations.push_back({allocation, start, finish});
  });
  llvm::stable_sort(allocations, [](const AllocationRange &a, const AllocationRange &b) { return a.start < b.start; });
  for (const auto &range : allocations) {
    auto allocation = range.allocation;
    uint64_t start = range.start, finish = range.finish;
    auto type = allocation.getType();
    int64_t space = type.getMemorySpaceAsInt();
    bool matrix = space == dsa::matrixSpace;
    int64_t payload = type.getNumElements() * llvm::divideCeil(type.getElementTypeBitWidth(), 8u);
    // The selected walign(16) filter is striped over 16 LT banks. WRAM offsets name
    // the position within each LT; its capacity is the sum over all banks.
    // Using aggregate byte offsets for a second filter addresses outside WRAM.
    int64_t bytes = matrix ? llvm::alignTo(llvm::divideCeil(payload, int64_t(16)), int64_t(64))
                           : llvm::alignTo(payload, int64_t(128));
    llvm::erase_if(live, [&](const Range &range) { return range.finish < start; });
    llvm::sort(live, [](const Range &a, const Range &b) { return a.offset < b.offset; });
    int64_t offset = 0;
    for (const Range &range : live) {
      if (range.space != space) continue;
      if (offset + bytes <= range.offset) break;
      offset = std::max(offset, range.offset + range.bytes);
    }
    live.push_back({finish, offset, bytes, space});
    int64_t &peak = matrix ? wram : space == dsa::sharedSpace ? shared : nram;
    peak = std::max(peak, (offset + bytes) * (matrix ? 16 : 1));
    Builder builder(function.getContext());
    allocation->setAttr("bangc.offset", builder.getI64IntegerAttr(offset));
    allocation->setAttr("bangc.allocation_bytes", builder.getI64IntegerAttr(bytes * (matrix ? 16 : 1)));
  }
  function->setAttr("bangc.sram_bytes", IntegerAttr::get(IntegerType::get(function.getContext(), 64), shared));
}
void pipelinePointwiseLoads(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
  auto access = [&](Value value, unsigned kind) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == kind;
  };
  std::function<int64_t(Value)> stepSize = [&](Value value) -> int64_t {
    if (value.getDefiningOp<dsa::TaskCountOp>()) return config.getTasks();
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) return constant.value();
    if (auto product = value.getDefiningOp<arith::MulIOp>()) {
      int64_t a = stepSize(product.getLhs()), b = stepSize(product.getRhs());
      if (a > 0 && b > 0 && a <= std::numeric_limits<int64_t>::max() / b) return a * b;
    }
    return -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (auto loop : loops) {
    auto upper = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    int64_t step = stepSize(loop.getStep());
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() || !loop.getInitArgs().empty() || !upper ||
        step <= 0 || step > std::numeric_limits<int64_t>::max() / 2 ||
        upper.value() > std::numeric_limits<int64_t>::max() - 2 * step || upper.value() <= step) continue;
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    SmallVector<memref::AllocaOp> allocations;
    bool eligible = true, compute = false;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (auto allocation = dyn_cast<memref::AllocaOp>(op)) allocations.push_back(allocation);
      else if (auto transfer = dyn_cast<dsa::LoadTileOp>(op)) {
        if (load || store || transfer.getAsynchronous() || !access(transfer.getSource(), 0)) { eligible = false; break; }
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(op)) {
        if (store || !load || !compute || transfer.getInput() != load.getOutput() || !access(transfer.getDestination(), 1)) {
          eligible = false; break;
        }
        store = transfer;
      } else if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
        if (!load || store || binary.getLhs() != load.getOutput() || binary.getOutput() != load.getOutput() ||
            (binary.getKind() != BinaryOperator::Add && binary.getKind() != BinaryOperator::Subtract &&
             binary.getKind() != BinaryOperator::Multiply && binary.getKind() != BinaryOperator::Maximum)) {
          eligible = false; break;
        }
        if (Operation *rhs = binary.getRhs().getDefiningOp();
            isa<MemRefType>(binary.getRhs().getType()) && binary.getRhs() != load.getOutput() && (!rhs || loop->isAncestor(rhs))) {
          eligible = false; break;
        }
        compute = true;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(op)) {
        if (!isa<BlockArgument>(stride.getSource())) { eligible = false; break; }
      } else if (!isa<dsa::SynchronizeOp>(op) &&
                 (op.getNumRegions() || op.getName().getDialectNamespace() != "arith" || !isMemoryEffectFree(&op))) {
        eligible = false; break;
      }
    }
    if (!eligible || !load || !store || allocations.size() != 1 || allocations.front().getResult() != load.getOutput() ||
        llvm::any_of(load.getOutput().getUsers(), [&](Operation *user) { return user->getBlock() != loop.getBody(); })) continue;
    auto type = cast<MemRefType>(load.getOutput().getType());
    if (!type.getLayout().isIdentity()) continue;
    Operation *previous = loop->getPrevNode();
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    Value firstSlot = allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
    Value secondSlot = allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
    auto activeTask = b.create<scf::IfOp>(loc, active, false);
    b.setInsertionPointToStart(&activeTask.getThenRegion().front());
    auto emit = [&](Value coordinate, Value slot, bool supply) {
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), coordinate);
      mapping.map(load.getOutput(), slot);
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (isa<memref::AllocaOp, dsa::SynchronizeOp>(op)) continue;
        if (isa<dsa::LoadTileOp>(op)) {
          if (supply) b.clone(op, mapping)->setAttr("asynchronous", b.getBoolAttr(true));
          continue;
        }
        if (supply && !isa<dsa::StrideOp>(op) && op.getName().getDialectNamespace() != "arith") continue;
        if (isa<dsa::StoreTileOp>(op)) b.create<dsa::SynchronizeOp>(loc);
        b.clone(op, mapping);
      }
      if (!supply) b.create<dsa::SynchronizeOp>(loc);
    };
    emit(loop.getLowerBound(), firstSlot, true);
    b.create<dsa::SynchronizeOp>(loc);
    Value doubled = b.create<arith::AddIOp>(loc, loop.getStep(), loop.getStep());
    auto pipeline = b.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), doubled);
    b.setInsertionPointToStart(pipeline.getBody());
    Value current = pipeline.getInductionVar();
    Value next = b.create<arith::AddIOp>(loc, current, loop.getStep());
    Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
    auto prefetch = b.create<scf::IfOp>(loc, hasNext, false);
    { OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&prefetch.getThenRegion().front()); emit(next, secondSlot, true); }
    emit(current, firstSlot, false);
    auto second = b.create<scf::IfOp>(loc, hasNext, false);
    {
      OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&second.getThenRegion().front());
      Value following = b.create<arith::AddIOp>(loc, next, loop.getStep());
      Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, loop.getUpperBound());
      auto future = b.create<scf::IfOp>(loc, hasFollowing, false);
      { OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&future.getThenRegion().front()); emit(following, firstSlot, true); }
      emit(next, secondSlot, false);
    }
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
    } else loop.erase();
  }
}
void pipelineRowLoads(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
  auto access = [&](Value value, unsigned kind) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == kind;
  };
  std::function<int64_t(Value)> stepSize = [&](Value value) -> int64_t {
    if (value.getDefiningOp<dsa::TaskCountOp>()) return config.getTasks();
    APInt bits; if (matchPattern(value, m_ConstantInt(&bits))) return bits.getSExtValue();
    if (auto product = value.getDefiningOp<arith::MulIOp>()) {
      int64_t a = stepSize(product.getLhs()), b = stepSize(product.getRhs());
      if (a > 0 && b > 0 && a <= INT64_MAX / b) return a * b;
    }
    return -1;
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (auto loop : loops) {
    auto upper = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    int64_t step = stepSize(loop.getStep());
    if (!loop.getLowerBound().getDefiningOp<dsa::TaskIdOp>() || !loop.getInitArgs().empty() || !upper ||
        step <= 0 || step > INT64_MAX / 2 || upper.value() > INT64_MAX - 2 * step || upper.value() <= step) continue;
    auto *body = loop.getBody();
    dsa::LoadTileOp load;
    dsa::StoreTileOp store;
    bool eligible = true, reduction = false;
    for (Operation &op : body->without_terminator()) {
      if (auto transfer = dyn_cast<dsa::LoadTileOp>(op)) {
        if (load || store || transfer.getAsynchronous() || !access(transfer.getSource(), 0)) eligible = false;
        load = transfer;
      } else if (auto transfer = dyn_cast<dsa::StoreTileOp>(op)) {
        if (store || !load || !access(transfer.getDestination(), 1)) eligible = false;
        store = transfer;
      } else if (auto reduce = dyn_cast<dsa::ReduceOp>(op)) {
        auto input = cast<MemRefType>(reduce.getInput().getType());
        auto scratch = cast<MemRefType>(reduce.getScratch().getType());
        if (!load || store || reduce.getAxis() != 1 || input.getDimSize(0) >= 32 || input.getDimSize(1) < 1024 ||
            scratch.getShape() != ArrayRef<int64_t>({1, input.getDimSize(1)})) eligible = false;
        reduction = true;
      } else if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
        if (!load || store || unary.getKind() != UnaryOperator::Exp2 || !unary.getApproximate() || !unary.getFlushToZero() ||
            !unary.getScratch() || !cast<MemRefType>(unary.getScratch().getType()).getElementType().isInteger(32)) eligible = false;
      } else if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
        auto implementation = binary->getAttrOfType<StringAttr>("bangc.implementation");
        bool reciprocal = implementation && implementation.getValue() == "reciprocal_f32_ftz";
        if (!load || store || (!reciprocal && (binary.getApproximate() || binary.getFlushToZero() ||
            !supportedScalarBinary(binary.getKind())))) eligible = false;
      } else if (auto conversion = dyn_cast<dsa::CastOp>(op)) {
        Type a = cast<MemRefType>(conversion.getInput().getType()).getElementType();
        Type b = cast<MemRefType>(conversion.getOutput().getType()).getElementType();
        if (!load || store || !((a.isF16() && b.isF32()) || (a.isF32() && b.isF16()))) eligible = false;
      } else if (auto fill = dyn_cast<dsa::FillOp>(op)) {
        if (!load || store || (!fill.getValue().getType().isF16() && !fill.getValue().getType().isF32())) eligible = false;
      } else if (auto stride = dyn_cast<dsa::StrideOp>(op)) {
        if (!isa<BlockArgument>(stride.getSource())) eligible = false;
      } else if (!isa<memref::AllocaOp, memref::LoadOp, memref::StoreOp, memref::CopyOp, dsa::SynchronizeOp>(op) &&
                 (op.getNumRegions() || op.getName().getDialectNamespace() != "arith" || !isMemoryEffectFree(&op))) eligible = false;
      if (store && isa<memref::LoadOp, memref::StoreOp, memref::CopyOp>(op)) eligible = false;
      if (isa<dsa::LoadTileOp, dsa::StoreTileOp, dsa::StrideOp, dsa::SynchronizeOp>(op)) continue;
      for (Value operand : op.getOperands()) if (auto type = dyn_cast<MemRefType>(operand.getType()))
        if (type.getMemorySpaceAsInt() != dsa::nramSpace) eligible = false;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (const auto &effect : instances) if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect())) {
          Operation *owner = effect.getValue() ? effect.getValue().getDefiningOp() : nullptr;
          if (!owner || !loop->isProperAncestor(owner)) eligible = false;
        }
      }
    }
    if (!eligible || !reduction || !load || !store) continue;
    auto input = load.getOutput().getDefiningOp<memref::AllocaOp>();
    if (!input || input->getBlock() != body || !input.getType().getLayout().isIdentity() ||
        llvm::any_of(input.getResult().getUsers(), [&](Operation *user) { return user->getBlock() != body; })) continue;
    std::function<bool(Value)> pureSupply = [&](Value value) {
      if (value == loop.getInductionVar()) return true;
      auto *op = value.getDefiningOp();
      if (!op || !loop->isProperAncestor(op)) return true;
      return !op->getNumRegions() && (isa<dsa::StrideOp>(op) ||
          (op->getName().getDialectNamespace() == "arith" && isMemoryEffectFree(op))) &&
          llvm::all_of(op->getOperands(), pureSupply);
    };
    for (Value operand : load->getOperands()) if (operand != load.getOutput()) eligible &= pureSupply(operand);
    if (!eligible) continue;
    Operation *previous = loop->getPrevNode();
    OpBuilder b(loop); Location loc = loop.getLoc();
    auto type = input.getType();
    Value firstSlot = allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
    Value secondSlot = allocate(b, loc, type.getElementType(), type.getShape(), dsa::nramSpace);
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
    auto activeTask = b.create<scf::IfOp>(loc, active, false);
    b.setInsertionPointToStart(&activeTask.getThenRegion().front());
    auto supply = [&](Value coordinate, Value slot) {
      IRMapping mapping; mapping.map(loop.getInductionVar(), coordinate); mapping.map(load.getOutput(), slot);
      std::function<Value(Value)> project = [&](Value value) -> Value {
        if (mapping.contains(value)) return mapping.lookup(value);
        auto *op = value.getDefiningOp();
        if (!op || !loop->isProperAncestor(op)) return value;
        for (Value operand : op->getOperands()) mapping.map(operand, project(operand));
        b.clone(*op, mapping); return mapping.lookup(value);
      };
      for (Value operand : load->getOperands()) if (operand != load.getOutput()) mapping.map(operand, project(operand));
      auto transfer = cast<dsa::LoadTileOp>(b.clone(*load, mapping)); transfer.setAsynchronous(true);
    };
    auto compute = [&](Value coordinate, Value slot) {
      IRMapping mapping; mapping.map(loop.getInductionVar(), coordinate); mapping.map(load.getOutput(), slot);
      bool stored = false;
      for (Operation &op : body->without_terminator()) {
        if (&op == input.getOperation() || &op == load.getOperation()) continue;
        if (auto sync = dyn_cast<dsa::SynchronizeOp>(op)) {
          if (!stored) b.create<dsa::SynchronizeOp>(loc, b.getBoolAttr(true));
          continue;
        }
        b.clone(op, mapping);
        if (&op == store.getOperation()) { b.create<dsa::SynchronizeOp>(loc); stored = true; }
      }
    };
    supply(loop.getLowerBound(), firstSlot); b.create<dsa::SynchronizeOp>(loc);
    Value doubled = b.create<arith::AddIOp>(loc, loop.getStep(), loop.getStep());
    auto pipeline = b.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), doubled);
    b.setInsertionPointToStart(pipeline.getBody());
    Value current = pipeline.getInductionVar(), next = b.create<arith::AddIOp>(loc, current, loop.getStep());
    Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
    auto prefetch = b.create<scf::IfOp>(loc, hasNext, false);
    { OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&prefetch.getThenRegion().front()); supply(next, secondSlot); }
    compute(current, firstSlot);
    auto second = b.create<scf::IfOp>(loc, hasNext, false);
    {
      OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&second.getThenRegion().front());
      Value following = b.create<arith::AddIOp>(loc, next, loop.getStep());
      Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, loop.getUpperBound());
      auto future = b.create<scf::IfOp>(loc, hasFollowing, false);
      { OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(&future.getThenRegion().front()); supply(following, firstSlot); }
      compute(next, secondSlot);
    }
    int64_t nram = 0, wram = 0; bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024)
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
    else loop.erase();
  }
}
void pipelineMatrixLoads(func::FuncOp function, dsa::ConfigurationAttr config) {
  auto interface = function->getAttrOfType<dsa::InterfaceAttr>("intent_dsa.interface");
  auto readOnlyArgument = [&](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = dyn_cast<dsa::ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    return view && view.getAccess() == 0;
  };
  auto constant = [](Value value) -> int64_t {
    auto op = value.getDefiningOp<arith::ConstantIndexOp>();
    return op ? op.value() : -1;
  };
  std::function<int64_t(Value)> upperLimit = [&](Value value) -> int64_t {
    int64_t fixed = constant(value);
    if (fixed >= 0) return fixed;
    if (auto minimum = value.getDefiningOp<arith::MinSIOp>()) {
      int64_t lhs = upperLimit(minimum.getLhs()), rhs = upperLimit(minimum.getRhs());
      return lhs >= 0 && rhs >= 0 ? std::min(lhs, rhs) : std::max(lhs, rhs);
    }
    return -1;
  };
  auto owner = [](Value value) {
    while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
    return value;
  };
  SmallVector<scf::ForOp> loops;
  function.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (auto loop : loops) {
    int64_t lower = constant(loop.getLowerBound()), upper = upperLimit(loop.getUpperBound());
    int64_t step = constant(loop.getStep());
    if (!loop.getInitArgs().empty() || lower < 0 || step <= 0 || upper <= lower ||
        step > std::numeric_limits<int64_t>::max() / 2 ||
        upper > std::numeric_limits<int64_t>::max() - 2 * step || upper - lower <= step) continue;
    SmallVector<dsa::LoadTileOp> loads;
    dsa::PrepareMatrixOp prepare;
    dsa::MatrixTileOp matrix;
    auto readOnlySource = [&](Value source) {
      if (readOnlyArgument(source)) return true;
      auto allocation = source.getDefiningOp<memref::AllocaOp>();
      if (!allocation || allocation.getType().getMemorySpaceAsInt() != dsa::sharedSpace || loop->isAncestor(allocation)) return false;
      return llvm::all_of(source.getUsers(), [&](Operation *user) {
        if (!loop->isAncestor(user)) return true;
        auto load = dyn_cast<dsa::LoadTileOp>(user);
        return load && load.getSource() == source;
      });
    };
    auto eligible = loop.walk([&](Operation *op) {
      if (op == loop.getOperation()) return WalkResult::advance();
      if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
        if (load->getBlock() != loop.getBody() || load.getAsynchronous() || !readOnlySource(load.getSource()))
          return WalkResult::interrupt();
        loads.push_back(load);
        return WalkResult::advance();
      }
      if (auto packing = dyn_cast<dsa::PrepareMatrixOp>(op)) {
        if (prepare) return WalkResult::interrupt();
        prepare = packing;
        return WalkResult::advance();
      }
      if (auto compute = dyn_cast<dsa::MatrixTileOp>(op)) {
        if (matrix) return WalkResult::interrupt();
        matrix = compute;
        return WalkResult::advance();
      }
      if (auto branch = dyn_cast<scf::IfOp>(op))
        return branch.getNumResults() || !branch.getElseRegion().empty()
            ? WalkResult::interrupt() : WalkResult::advance();
      if (auto stride = dyn_cast<dsa::StrideOp>(op))
        return readOnlyArgument(stride.getSource()) ? WalkResult::advance() : WalkResult::interrupt();
      if (isa<dsa::SynchronizeOp, memref::AllocaOp, memref::ReinterpretCastOp, scf::YieldOp>(op))
        return WalkResult::advance();
      if (op->getName().getDialectNamespace() == "arith" && isMemoryEffectFree(op) && !op->getNumRegions())
        return WalkResult::advance();
      return WalkResult::interrupt();
    });
    if (eligible.wasInterrupted() || loads.size() != 2 || !prepare || !matrix ||
        prepare->getBlock() != matrix->getBlock() || !prepare->isBeforeInBlock(matrix) ||
        matrix.getRhs() != prepare.getOutput()) continue;
    dsa::LoadTileOp lhsLoad, rhsLoad;
    for (auto load : loads) {
      if (load.getOutput() == matrix.getLhs()) lhsLoad = load;
      if (load.getOutput() == prepare.getInput()) rhsLoad = load;
    }
    if (!lhsLoad || !rhsLoad || lhsLoad == rhsLoad) continue;
    Operation *computeScope = loop.getBody()->findAncestorOpInBlock(*prepare.getOperation());
    if (!computeScope || !lhsLoad->isBeforeInBlock(computeScope) || !rhsLoad->isBeforeInBlock(computeScope)) continue;
    Value lhs = lhsLoad.getOutput(), rhs = rhsLoad.getOutput(), accumulator = matrix.getAccumulator();
    auto lhsAllocation = lhs.getDefiningOp<memref::AllocaOp>();
    auto rhsAllocation = rhs.getDefiningOp<memref::AllocaOp>();
    auto accAllocation = accumulator.getDefiningOp<memref::AllocaOp>();
    if (!lhsAllocation || !rhsAllocation || !accAllocation || lhs == rhs || lhs == accumulator || rhs == accumulator ||
        lhsAllocation->getBlock() != loop->getBlock() || rhsAllocation->getBlock() != loop->getBlock() ||
        loop->isAncestor(accAllocation)) continue;
    bool privateInputs = true;
    for (Value input : {lhs, rhs}) {
      SmallVector<Value> aliases{input};
      for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
        if (!loop->isAncestor(user)) privateInputs = false;
        if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) aliases.push_back(view.getResult());
      }
    }
    for (Value scratch : {prepare.getOutput(), prepare.getScratch(), prepare.getReshaped()}) {
      if (!scratch) continue;
      Value allocation = owner(scratch);
      if (allocation == rhs) continue;
      if (!allocation.getDefiningOp<memref::AllocaOp>() || !loop->isAncestor(allocation.getDefiningOp()))
        privateInputs = false;
    }
    if (!privateInputs) continue;
    auto lhsType = cast<MemRefType>(lhs.getType()), rhsType = cast<MemRefType>(rhs.getType());
    if (!lhsType.getLayout().isIdentity() || !rhsType.getLayout().isIdentity()) continue;
    // Keep storage ownership explicit. Both input slots live across the whole
    // loop; each stage finishes before the shared packing workspace is reused.
    Operation *previous = loop->getPrevNode();
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    Value nextLhs = allocate(b, loc, lhsType.getElementType(), lhsType.getShape(), dsa::nramSpace);
    Value nextRhs = allocate(b, loc, rhsType.getElementType(), rhsType.getShape(), dsa::nramSpace);
    Value doubledStep = b.create<arith::ConstantIndexOp>(loc, 2 * step);
    if (constant(loop.getUpperBound()) < 0) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
      auto nonempty = b.create<scf::IfOp>(loc, active, false);
      b.setInsertionPointToStart(nonempty.thenBlock());
    }
    auto emitLoads = [&](Value coordinate, Value a, Value rawB) {
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), coordinate);
      mapping.map(lhs, a);
      mapping.map(rhs, rawB);
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (isa<dsa::LoadTileOp>(op)) {
          Operation *copy = b.clone(op, mapping);
          copy->setAttr("asynchronous", b.getBoolAttr(true));
        } else if (isa<dsa::StrideOp>(op) || op.getName().getDialectNamespace() == "arith") {
          b.clone(op, mapping);
        }
      }
    };
    auto emitCompute = [&](Value coordinate, Value a, Value rawB) {
      IRMapping mapping;
      mapping.map(loop.getInductionVar(), coordinate);
      mapping.map(lhs, a);
      mapping.map(rhs, rawB);
      for (Operation &op : loop.getBody()->without_terminator()) {
        if (isa<dsa::LoadTileOp, dsa::SynchronizeOp>(op)) continue;
        Operation *copy = b.clone(op, mapping);
        SmallVector<dsa::SynchronizeOp> fences;
        copy->walk([&](dsa::SynchronizeOp fence) { fences.push_back(fence); });
        for (auto fence : fences) fence.erase();
      }
      b.create<dsa::SynchronizeOp>(loc);
    };
    emitLoads(loop.getLowerBound(), lhs, rhs);
    b.create<dsa::SynchronizeOp>(loc);
    auto pipeline = b.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), doubledStep);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(pipeline.getBody());
      Value k = pipeline.getInductionVar();
      Value next = b.create<arith::AddIOp>(loc, k, loop.getStep());
      Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, loop.getUpperBound());
      auto prefetch = b.create<scf::IfOp>(loc, hasNext, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&prefetch.getThenRegion().front());
        emitLoads(next, nextLhs, nextRhs);
      }
      emitCompute(k, lhs, rhs);
      auto second = b.create<scf::IfOp>(loc, hasNext, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&second.getThenRegion().front());
        Value following = b.create<arith::AddIOp>(loc, next, loop.getStep());
        Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, loop.getUpperBound());
        auto prefetch = b.create<scf::IfOp>(loc, hasFollowing, false);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&prefetch.getThenRegion().front());
          emitLoads(following, lhs, rhs);
        }
        emitCompute(next, nextLhs, nextRhs);
      }
    }
    // Probe with the original loop still attached so its operand uses remain
    // well formed. The two loops are sequential and share the original slots.
    int64_t nram = 0, wram = 0;
    bindStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      while (loop->getPrevNode() != previous) loop->getPrevNode()->erase();
      continue;
    }
    loop.erase();
  }
}
}
LogicalResult realizeGroupParticipants(func::FuncOp function) {
  bool collective = false;
  function.walk([&](dsa::GroupGatherRowsOp) { collective = true; });
  if (!collective) return success();
  OpBuilder b(function.getContext());
  b.setInsertionPointToStart(&function.front());
  Location loc = function.getLoc();
  Value memory = b.create<dsa::IsMemoryCoreOp>(loc, b.getI1Type());
  Value compute = b.create<arith::XOrIOp>(loc, memory, b.create<arith::ConstantIntOp>(loc, 1, 1));
  auto containsCollective = [](Operation *operation) {
    bool found = false;
    operation->walk([&](dsa::GroupGatherRowsOp) { found = true; });
    return found;
  };
  std::function<LogicalResult(Block &)> partition = [&](Block &block) -> LogicalResult {
    SmallVector<Operation *> original;
    for (Operation &operation : block) original.push_back(&operation);
    SmallVector<Operation *> pending;
    DenseSet<Operation *> privateOperations;
    auto flush = [&](Operation *before) -> LogicalResult {
      if (pending.empty()) return success();
      for (Operation *operation : pending) for (Value result : operation->getResults())
        for (Operation *user : result.getUsers()) {
          while (user && user->getBlock() != &block) user = user->getParentOp();
          if (!user || !privateOperations.contains(user))
            return operation->emitError("compute-local scalar escapes a collective supply interval");
        }
      b.setInsertionPoint(before);
      auto guard = b.create<scf::IfOp>(before->getLoc(), compute, false);
      for (Operation *operation : pending) operation->moveBefore(guard.thenBlock()->getTerminator());
      pending.clear(); privateOperations.clear();
      return success();
    };
    for (Operation *operation : original) {
      if (containsCollective(operation)) {
        if (failed(flush(operation))) return failure();
        if (!isa<dsa::GroupGatherRowsOp>(operation))
          for (Region &region : operation->getRegions()) for (Block &body : region)
            if (failed(partition(body))) return failure();
        continue;
      }
      if (operation->hasTrait<OpTrait::IsTerminator>()) {
        if (failed(flush(operation))) return failure();
        continue;
      }
      bool available = llvm::all_of(operation->getOperands(), [&](Value value) {
        return !privateOperations.contains(value.getDefiningOp());
      });
      bool scalarLoad = isa<dsa::LoadScalarOp>(operation) && available;
      bool movable = available && !operation->getNumRegions() &&
          (isa<memref::AllocaOp, memref::ReinterpretCastOp>(operation) || isMemoryEffectFree(operation));
      if (scalarLoad) {
        if (failed(flush(operation))) return failure();
      } else if (movable) {
        // Static storage views and scalar expressions can precede the local
        // interval. Their operands are already outside that interval.
        if (!pending.empty()) operation->moveBefore(pending.front());
      } else {
        pending.push_back(operation); privateOperations.insert(operation);
      }
    }
    return success();
  };
  return partition(function.front());
}
LogicalResult legalizeProgram(ModuleOp module, StringRef architecture) {
  if (architecture != "mtp_372") return module.emitError("BANG C currently has a bound implementation profile for mtp_372");
  if (failed(dsa::verifyProgram(module))) return failure();
  auto function = *module.getOps<func::FuncOp>().begin();
  auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
  module->setAttr("bangc.architecture", StringAttr::get(module.getContext(), architecture));
  SmallVector<dsa::MatMulOp> matrices;
  function.walk([&](dsa::MatMulOp op) { matrices.push_back(op); });
  for (auto matrix : matrices) if (failed(realizeMatMul(matrix, config))) return failure();
  // Operations without a full-domain native storage-dtype implementation use
  // f32 local arithmetic and an explicit cast back to the declared dtype.
  SmallVector<Operation *> promotedPointwise;
  function.walk([&](Operation *op) {
    Value output;
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) output = unary.getOutput();
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) output = binary.getOutput();
    if (!output) return;
    Type element = cast<MemRefType>(output.getType()).getElementType();
    if (element.isBF16()) promotedPointwise.push_back(op);
  });
  for (Operation *op : promotedPointwise) {
    OpBuilder builder(op);
    auto convert = [&](Value source) -> Value {
      if (source.getType().isBF16())
        return builder.create<arith::ExtFOp>(op->getLoc(), builder.getF32Type(), source);
      auto type = cast<MemRefType>(source.getType());
      Value result = allocate(builder, op->getLoc(), builder.getF32Type(), type.getShape(), dsa::nramSpace);
      builder.create<dsa::CastOp>(op->getLoc(), source, result);
      return result;
    };
    Value destination;
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      destination = unary.getOutput();
      Value input = convert(unary.getInput());
      Value output = allocate(builder, op->getLoc(), builder.getF32Type(), cast<MemRefType>(input.getType()).getShape(), dsa::nramSpace);
      unary.getInputMutable().assign(input); unary.getOutputMutable().assign(output);
      builder.setInsertionPointAfter(op);
      builder.create<dsa::CastOp>(op->getLoc(), output, destination);
    } else {
      auto binary = cast<dsa::BinaryOp>(op);
      destination = binary.getOutput();
      Value lhs = convert(binary.getLhs()), rhs = convert(binary.getRhs());
      Value output = allocate(builder, op->getLoc(), builder.getF32Type(), cast<MemRefType>(lhs.getType()).getShape(), dsa::nramSpace);
      binary.getLhsMutable().assign(lhs); binary.getRhsMutable().assign(rhs); binary.getOutputMutable().assign(output);
      builder.setInsertionPointAfter(op);
      builder.create<dsa::CastOp>(op->getLoc(), output, destination);
    }
  }
  // The hardware activation approximations truncate part of sigmoid's domain.
  // Keep exp(-abs(x)) and a sign-selected numerator explicit instead.
  SmallVector<dsa::UnaryOp> sigmoids;
  function.walk([&](dsa::UnaryOp op) { if (op.getKind() == UnaryOperator::Sigmoid) sigmoids.push_back(op); });
  for (auto sigmoid : sigmoids) if (failed(realizeSigmoid(sigmoid))) return failure();
  SmallVector<Operation *> bf16Scalar;
  function.walk([&](Operation *op) {
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::NegFOp,
            arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp,
            math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp, math::RsqrtOp, math::TanhOp, math::AbsFOp,
            math::SinOp, math::CosOp, math::FloorOp>(op) &&
        llvm::any_of(op->getOperandTypes(), [](Type type) { return type.isBF16(); })) bf16Scalar.push_back(op);
  });
  for (Operation *op : bf16Scalar) {
    OpBuilder builder(op);
    SmallVector<Value> arguments;
    for (Value input : op->getOperands())
      arguments.push_back(input.getType().isBF16() ? Value(builder.create<arith::ExtFOp>(op->getLoc(), builder.getF32Type(), input)) : input);
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(arguments); state.addAttributes(op->getAttrs());
    Type original = op->getResult(0).getType();
    state.addTypes(original.isBF16() ? builder.getF32Type() : original);
    Value result = builder.create(state)->getResult(0);
    if (original.isBF16()) result = builder.create<arith::TruncFOp>(op->getLoc(), original, result);
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
  }
  bindUniformOperands(function);
  {
    SmallVector<Operation *> arithmetic;
    function.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "arith" && op->getNumResults() == 1 &&
          !isa<arith::ConstantOp>(op) &&
          (op->getResult(0).getType().isIndex() || op->getResult(0).getType().isInteger(64))) arithmetic.push_back(op);
    });
    for (Operation *op : arithmetic) {
      auto interval = integerInterval(op->getResult(0), function);
      if (!interval || interval->first != interval->second) continue;
      OpBuilder b(op);
      Value value = b.create<arith::ConstantOp>(op->getLoc(), b.getIntegerAttr(op->getResult(0).getType(), interval->first));
      op->getResult(0).replaceAllUsesWith(value); op->erase();
    }
    PassManager cleanup(module.getContext());
    cleanup.addPass(createCanonicalizerPass()); cleanup.addPass(createCSEPass());
    if (failed(cleanup.run(module))) return failure();
  }
  reuseConsumedBinaryInputs(function, config);
  reuseConsumedExp2Inputs(function, config);
  if (batchIndependentRowPrograms(function, config)) {
    PassManager cleanup(module.getContext());
    cleanup.addPass(createCanonicalizerPass()); cleanup.addPass(createCSEPass());
    if (failed(cleanup.run(module))) return failure();
  }
  fuseNarrowDivisions(function, config);
  realizeRoundedDivisions(function, config);
  realizeApproximateReciprocals(function, config);
  bindUniformOperands(function);
  realizeSelections(function, config);
  realizeRowSumChannels(function, config);
  if (failed(realizeNumericExtremaWorkspace(function, config))) return failure();
  realizeCompareWorkspace(function, config);
  realizeExponentialWorkspace(function, config);
  realizeFlushWorkspace(function, config);
  auto walk = function.walk([&](Operation *op) {
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      bool approximateExp2 = unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() &&
          cast<MemRefType>(unary.getInput().getType()).getElementType().isF32();
      if (approximateExp2) op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), "exp2_f32"));
      else if (!supportedUnary(unary.getKind()) || unary.getApproximate() || unary.getFlushToZero()) {
        op->emitError("unary numerical mode has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      Type element = cast<MemRefType>(unary.getInput().getType()).getElementType();
      if (!unary.getApproximate() && (element.isF32() || element.isF16())) {
        StringRef callee;
        switch (unary.getKind()) {
        case UnaryOperator::Exp: callee = "__cn_vector_exp"; break;
        case UnaryOperator::Exp2: callee = "__cn_vector_exp2"; break;
        case UnaryOperator::Log: callee = "__cn_vector_log"; break;
        case UnaryOperator::Tanh: callee = "__cn_vector_tanh"; break;
        case UnaryOperator::Sqrt: callee = "__cn_vector_sqrt"; break;
        case UnaryOperator::Rsqrt: callee = "__cn_vector_rsqrt"; break;
        case UnaryOperator::Sin: callee = "__cn_vector_sin"; break;
        case UnaryOperator::Cos: callee = "__cn_vector_cos"; break;
        case UnaryOperator::Floor: callee = "__cn_vector_floor"; break;
        default: break;
        }
        if (!callee.empty()) op->setAttr("bangc.callee", StringAttr::get(module.getContext(), callee.str() + (element.isF16() ? "_f16" : "_f32")));
      }
    }
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      Type element = cast<MemRefType>(binary.getLhs().getType()).getElementType();
      if (!isa<MemRefType>(binary.getRhs().getType()) &&
          (!supportedScalarBinary(binary.getKind()) || (!element.isF16() && !element.isF32() && !element.isInteger(32)))) {
        op->emitError("scalar binary kind or dtype has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      bool division = binary.getKind() == BinaryOperator::TrueDivide &&
          cast<MemRefType>(binary.getLhs().getType()).getElementType().isF32();
      if (!supportedBinary(binary.getKind()) || ((binary.getApproximate() || binary.getFlushToZero()) && !division)) {
        op->emitError("binary numerical mode has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      if (binary.getKind() == BinaryOperator::TrueDivide && !op->hasAttr("bangc.implementation"))
        op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), division ? "divide_f32" : "scalar_divide"));
      if (binary.getKind() == BinaryOperator::Maximum && (element.isF16() || element.isF32()) &&
          isa<MemRefType>(binary.getRhs().getType()) && binary.getOutput() != binary.getRhs()) {
        if (auto fill = uniformFillBefore(binary.getRhs(), op)) {
          auto constant = fill.getValue().getDefiningOp<arith::ConstantOp>();
          auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
          // On MTP372 the SDK's NaN-propagating maximum first injects RHS NaNs
          // into the LHS, then issues maxequal. A non-NaN RHS makes that prefix
          // an exact bitwise identity, including for special LHS values.
          if (value && !value.getValue().isNaN())
            op->setAttr("bangc.callee", StringAttr::get(module.getContext(), "__bang_maxequal"));
        }
      }
    }
    if (auto reduction = dyn_cast<dsa::ReduceOp>(op)) {
      auto input = cast<MemRefType>(reduction.getInput().getType());
      bool rowBatch = reduction.getAxis() == 1 && input.getDimSize(0) > 1;
      if (!input.getElementType().isF32() ||
          (reduction.getAxis() == 0 && reduction.getKind() != BinaryOperator::Add) ||
          (rowBatch && input.getDimSize(1) >= 65536) ||
          (reduction.getKind() != BinaryOperator::Add && reduction.getKind() != BinaryOperator::Maximum &&
           reduction.getKind() != BinaryOperator::MaximumNum && reduction.getKind() != BinaryOperator::Minimum &&
           reduction.getKind() != BinaryOperator::MinimumNum)) {
        op->emitError("reduction has no selected BANG C f32 local implementation"); return WalkResult::interrupt();
      }
    }
    if (auto compare = dyn_cast<dsa::CompareOp>(op)) {
      if (!cast<MemRefType>(compare.getLhs().getType()).getElementType().isInteger(64)) {
        op->emitError("comparison has no selected BANG C tile implementation"); return WalkResult::interrupt();
      }
      static constexpr const char *predicates[] = {"eq", "ne", "lt", "le", "gt", "ge"};
      op->setAttr("bangc.callee", StringAttr::get(module.getContext(),
          std::string("__cn_vector_") + predicates[static_cast<unsigned>(compare.getPredicate())] + "_s64"));
    }
    if (auto conversion = dyn_cast<dsa::CastOp>(op)) {
      auto source = cast<MemRefType>(conversion.getInput().getType());
      Type destination = cast<MemRefType>(conversion.getOutput().getType()).getElementType();
      if (source.getNumElements() >= 64) {
        StringRef callee;
        if (source.getElementType().isSignlessInteger(32) && destination.isSignlessInteger(64))
          callee = "__cn_vector_cast_s32_to_s64";
        if (source.getElementType().isSignlessInteger(64) && destination.isSignlessInteger(32))
          callee = "__cn_vector_cast_s64_to_s32";
        if (!callee.empty()) op->setAttr("bangc.callee", StringAttr::get(module.getContext(), callee));
      }
    }
    if (!isa<dsa::SynchronizeOp, dsa::GroupSynchronizeOp, dsa::GroupIdOp, dsa::GroupCountOp, dsa::LocalIdOp,
             dsa::IsMemoryCoreOp, dsa::StageTileOp, dsa::TaskIdOp, dsa::TaskCountOp, dsa::StrideOp, dsa::LoadScalarOp, dsa::StoreScalarOp,
             dsa::LoadTileOp, dsa::GatherPlanOp, dsa::GatherRowsOp, dsa::GroupGatherRowsOp, dsa::StoreTileOp, dsa::FillOp, dsa::IotaOp,
             dsa::IndexLayoutOp, dsa::IndexBinaryOp, dsa::BroadcastRowsOp, dsa::TransposeOp, dsa::SelectOp, dsa::MaskedFillOp, dsa::UnaryOp, dsa::BinaryOp,
             dsa::CastOp, dsa::CompareOp, dsa::CompareRangeOp, dsa::CompareRampOp, dsa::DivideCastOp, dsa::DivideRNOp, dsa::ReduceOp,
             dsa::PrepareMatrixOp, dsa::PrepareMatrixViewOp, dsa::MatrixTileOp,
             arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::DivSIOp, arith::RemSIOp, arith::CeilDivSIOp, arith::FloorDivSIOp, arith::MinSIOp, arith::MaxSIOp,
             arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::ExtFOp, arith::TruncFOp, arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
             arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
             arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp,
             arith::NegFOp, math::ExpOp, math::Exp2Op, math::LogOp, math::SqrtOp, math::RsqrtOp, math::TanhOp, math::AbsFOp,
             math::SinOp, math::CosOp, math::FloorOp,
             memref::DimOp, memref::AllocaOp, memref::ReinterpretCastOp, memref::LoadOp, memref::StoreOp, memref::CopyOp,
             scf::ForOp, scf::WhileOp, scf::IfOp, scf::ConditionOp, scf::YieldOp, func::FuncOp, func::ReturnOp>(op)) {
      op->emitError("operation is outside the bound BANG C surface"); return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) return failure();
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) return failure();
  eliminateOverwrittenFills(function);
  if (realizeRangeComparisons(function) && failed(pm.run(module))) return failure();
  if (foldRangeCounts(function) && failed(pm.run(module))) return failure();
  while (foldUniformBooleanTiles(function)) if (failed(pm.run(module))) return failure();
  if (realizeAffineRanges(function, config) && failed(pm.run(module))) return failure();
  if (realizeFullWidthMasks(function, config) && failed(pm.run(module))) return failure();
  while (eliminateUnreadLocalWrites(function)) if (failed(pm.run(module))) return failure();
  if (realizeRowBroadcasts(function, config) && failed(pm.run(module))) return failure();
  if (bindRowScalarOperands(function) && failed(pm.run(module))) return failure();
  if (forwardUniformScalarLoads(function) && failed(pm.run(module))) return failure();
  while (forwardFullLocalCopies(function)) if (failed(pm.run(module))) return failure();
  if (specializeZeroMatrixTiles(function) && failed(pm.run(module))) return failure();
  if (retainNarrowExtremaInputs(function, config) && failed(pm.run(module))) return failure();
  eliminateOverwrittenFills(function);
  if (forwardIndexExpressions(function) && failed(pm.run(module))) return failure();
  if (normalizeLinearIndices(function) && failed(pm.run(module))) return failure();
  if (reuseGatherOffsets(function) && failed(pm.run(module))) return failure();
  if (vectorizeIndexLoops(function, config) && failed(pm.run(module))) return failure();
  realizeGatherWorkspace(function, config);
  if (batchPointwiseTasks(function) && failed(pm.run(module))) return failure();
  if (reuseConsumedBinaryInputs(function, config) && failed(pm.run(module))) return failure();
  hoistInvariantFills(function, config);
  coalesceTileLoads(function, config);
  // Scalar NRAM accesses preserve program order within a task. Complete them
  // before entering a bulk operation, and complete bulk operations before any
  // scalar consumer or allocation reuse. A fence inside every scalar store or
  // scalar carry copy would serialize each element of broadcast/reduce loops.
  SmallVector<Operation *> localEffects;
  function.walk([&](Operation *op) {
    if (isa<dsa::LoadTileOp, dsa::GatherPlanOp, dsa::GatherRowsOp, dsa::StoreTileOp, dsa::FillOp, dsa::IotaOp,
            dsa::IndexLayoutOp, dsa::IndexBinaryOp, dsa::BroadcastRowsOp, dsa::TransposeOp, dsa::SelectOp, dsa::MaskedFillOp, dsa::UnaryOp,
            dsa::BinaryOp, dsa::CastOp, dsa::CompareOp, dsa::CompareRangeOp, dsa::CompareRampOp, dsa::DivideCastOp, dsa::DivideRNOp, dsa::ReduceOp,
            dsa::PrepareMatrixOp, dsa::PrepareMatrixViewOp, dsa::MatrixTileOp>(op))
      localEffects.push_back(op);
    if (auto copy = dyn_cast<memref::CopyOp>(op))
      if (cast<MemRefType>(copy.getSource().getType()).getNumElements() > 1)
        localEffects.push_back(op);
  });
  DominanceInfo transferDominance(function);
  auto canOverlapPrefetch = [&](Operation *compute) {
    if (!isa<dsa::MatrixTileOp>(compute) || !compute->getPrevNode()) return false;
    Operation *previous = compute->getPrevNode();
    SmallVector<Value> pending;
    bool eligible = true;
    previous->walk([&](Operation *op) {
      if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
        if (!load.getAsynchronous()) eligible = false;
        pending.push_back(load.getOutput());
      } else if (op->getNumRegions()) {
        if (!isa<scf::IfOp>(op)) eligible = false;
      } else if (!isMemoryEffectFree(op)) eligible = false;
    });
    if (!eligible || pending.empty()) return false;
    auto owner = [](Value value) {
      while (auto view = value.getDefiningOp<memref::ReinterpretCastOp>()) value = view.getSource();
      return value;
    };
    for (Value input : compute->getOperands()) {
      Value allocation = owner(input);
      if (!allocation.getDefiningOp<memref::AllocaOp>() ||
          !transferDominance.dominates(allocation.getDefiningOp(), previous)) return false;
      for (Value output : pending) if (owner(output) == allocation) return false;
    }
    return true;
  };
  for (Operation *op : localEffects) {
    // An explicitly asynchronous transfer is consumed at the next explicit
    // work-unit/group fence. Independent current matrix inputs may execute
    // while the alternate input buffer is being supplied.
    if (auto load = dyn_cast<dsa::LoadTileOp>(op); load && load.getAsynchronous()) continue;
    if (auto gather = dyn_cast<dsa::GatherRowsOp>(op); gather && gather.getAsynchronous()) continue;
    OpBuilder builder(op);
    if ((!op->getPrevNode() || !isa<dsa::SynchronizeOp>(op->getPrevNode())) && !canOverlapPrefetch(op))
      builder.create<dsa::SynchronizeOp>(op->getLoc());
    builder.setInsertionPointAfter(op);
    builder.create<dsa::SynchronizeOp>(op->getLoc());
  }
  // Consecutive native Compute operations obey stream dependencies, including
  // storage reuse. Keep fences at scalar access, transfer and helper boundaries.
  auto nativeCompute = [](Operation *op) {
    if (isa<dsa::IndexBinaryOp, dsa::MaskedFillOp>(op)) return true;
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      auto type = cast<MemRefType>(binary.getOutput().getType());
      return !binary.getApproximate() && !binary.getFlushToZero() && type.getNumElements() >= 64 &&
          (type.getElementType().isF16() || type.getElementType().isF32()) &&
          (binary.getKind() == BinaryOperator::Add || binary.getKind() == BinaryOperator::Subtract ||
           binary.getKind() == BinaryOperator::Multiply);
    }
    if (auto conversion = dyn_cast<dsa::CastOp>(op)) {
      auto input = cast<MemRefType>(conversion.getInput().getType());
      auto output = cast<MemRefType>(conversion.getOutput().getType());
      return input.getNumElements() >= 64 &&
          ((input.getElementType().isF16() && output.getElementType().isF32()) ||
           (input.getElementType().isF32() && output.getElementType().isF16()));
    }
    if (auto unary = dyn_cast<dsa::UnaryOp>(op))
      return unary.getKind() == UnaryOperator::Exp2 && unary.getApproximate() && unary.getFlushToZero() &&
          unary.getScratch() && cast<MemRefType>(unary.getScratch().getType()).getElementType().isInteger(32);
    return false;
  };
  function.walk([&](Operation *compute) {
    if (!nativeCompute(compute)) return;
    SmallVector<Operation *> fences;
    Operation *previous = compute->getPrevNode();
    while (previous) {
      if (isa<dsa::SynchronizeOp>(previous)) fences.push_back(previous);
      else if (!isa<memref::AllocaOp>(previous) && !isMemoryEffectFree(previous)) break;
      previous = previous->getPrevNode();
    }
    if (previous && nativeCompute(previous))
      for (Operation *fence : fences) fence->erase();
  });
  pipelinePointwiseLoads(function, config);
  pipelineRowLoads(function, config);
  pipelineMatrixLoads(function, config);
  SmallVector<dsa::SynchronizeOp> fences;
  function.walk([&](dsa::SynchronizeOp fence) { fences.push_back(fence); });
  for (auto fence : fences) {
    Operation *previous = fence->getPrevNode();
    while (previous && (isa<memref::AllocaOp>(previous) || isMemoryEffectFree(previous)))
      previous = previous->getPrevNode();
    auto earlier = dyn_cast_or_null<dsa::SynchronizeOp>(previous);
    if (!earlier) continue;
    // No command was issued between these waits. A full wait subsumes a local
    // wait; IO completion remains explicit at every asynchronous slot boundary.
    if (!earlier.getLocalOnly() || fence.getLocalOnly()) fence.erase();
    else earlier.erase();
  }
  if (failed(realizeGroupParticipants(function))) return failure();
  int64_t nram = 0, wram = 0;
  bindStorage(function, nram, wram);
  int64_t shared = function->getAttrOfType<IntegerAttr>("bangc.sram_bytes").getInt();
  if (shared > 3968 * 1024) return function.emitError("selected shared supply exceeds the MLU370 SRAM budget");
  int64_t internalNram = 0;
  function.walk([&](Operation *op) {
    if (auto bytes = op->getAttrOfType<IntegerAttr>("bangc.internal_nram_bytes"))
      internalNram = std::max(internalNram, bytes.getInt());
  });
  if (nram + internalNram > config.getLocalBytes() || nram + internalNram > 768 * 1024 || wram > 1024 * 1024)
    return function.emitError("selected local implementation exceeds the MLU370 per-task storage budget")
        << "; NRAM=" << nram << ", WRAM=" << wram;
  Builder builder(module.getContext());
  function->setAttr("bangc.nram_bytes", builder.getI64IntegerAttr(nram));
  function->setAttr("bangc.wram_bytes", builder.getI64IntegerAttr(wram));
  function->setAttr("bangc.wram_align", builder.getI64IntegerAttr(16));
  return dsa::verifyProgram(module, true);
}
}

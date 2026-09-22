#include "Intent/Target/BangC/Passes.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
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
Value allocate(OpBuilder &b, Location loc, Type element, ArrayRef<int64_t> shape, int64_t space) {
  auto type = MemRefType::get(shape, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(space));
  auto result = b.create<memref::AllocaOp>(loc, type);
  result.setAlignment(128);
  return result;
}
LogicalResult realizeMatMul(dsa::MatMulOp matrix, dsa::ConfigurationAttr config) {
  Location loc = matrix.getLoc();
  OpBuilder b(matrix);
  auto lhsType = cast<MemRefType>(matrix.getLhs().getType());
  auto rhsType = cast<MemRefType>(matrix.getRhs().getType());
  auto accType = cast<MemRefType>(matrix.getAccumulator().getType());
  Type element = lhsType.getElementType();
  if (!element.isF16() && !element.isBF16() && !element.isF32())
    return matrix.emitError("MLU370 matrix profile requires f16/bf16/f32 input tiles");
  int64_t bm = config.getTileM(), bn = 64, bk = config.getTileK();
  if ((bk * element.getIntOrFloatBitWidth() / 8) % 64)
    return matrix.emitError("MLU370 matrix K block must be aligned to 64 bytes");
  auto index = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
  auto offset = [&](Value row, Value column, int64_t stride) -> Value {
    return b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, row, index(stride)), column);
  };
  auto count = [&](Value extent, Value begin, int64_t tile) -> Value {
    return b.create<arith::MinSIOp>(loc, b.create<arith::SubIOp>(loc, extent, begin), index(tile));
  };
  auto rows = b.create<scf::ForOp>(loc, index(0), matrix.getRows(), index(bm));
  b.setInsertionPointToStart(rows.getBody());
  Value m = rows.getInductionVar(), mCount = count(matrix.getRows(), m, bm);
  auto columns = b.create<scf::ForOp>(loc, index(0), matrix.getColumns(), index(bn));
  b.setInsertionPointToStart(columns.getBody());
  Value n = columns.getInductionVar(), nCount = count(matrix.getColumns(), n, bn);
  Value lhs = allocate(b, loc, element, {bm, bk}, dsa::nramSpace);
  Value rhs = allocate(b, loc, element, {bk, bn}, dsa::nramSpace);
  Value accumulator = allocate(b, loc, b.getF32Type(), {bm, bn}, dsa::nramSpace);
  Value partial = allocate(b, loc, b.getF32Type(), {bm, bn}, dsa::nramSpace);
  Value transpose = allocate(b, loc, element, {bn, bk}, dsa::nramSpace);
  Value packed = allocate(b, loc, element, {bk, bn}, dsa::matrixSpace);
  packed.getDefiningOp()->setAttr("bangc.layout", b.getStringAttr("matrix_transposed64"));
  b.create<dsa::LoadTileOp>(loc, matrix.getAccumulator(), accumulator,
      offset(m, n, accType.getDimSize(1)), index(accType.getDimSize(1)), index(1), mCount, nCount);
  auto reduction = b.create<scf::ForOp>(loc, index(0), matrix.getDepth(), index(bk));
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(reduction.getBody());
    Value k = reduction.getInductionVar(), kCount = count(matrix.getDepth(), k, bk);
    b.create<dsa::LoadTileOp>(loc, matrix.getLhs(), lhs, offset(m, k, lhsType.getDimSize(1)),
        index(lhsType.getDimSize(1)), index(1), mCount, kCount);
    b.create<dsa::LoadTileOp>(loc, matrix.getRhs(), rhs, offset(k, n, rhsType.getDimSize(1)),
        index(rhsType.getDimSize(1)), index(1), kCount, nCount);
    b.create<dsa::PrepareMatrixOp>(loc, rhs, packed, transpose);
    auto tile = b.create<dsa::MatrixTileOp>(loc, lhs, packed, accumulator, partial);
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
        b.getBoolAttr(false), b.getBoolAttr(false));
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
// Contiguous views retain the allocation's ownership. Follow their uses and
// lift nested uses to the allocation's block, retaining loop backedge liveness.
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
  struct Range { uint64_t finish; int64_t offset, bytes; bool matrix; };
  SmallVector<Range> live;
  function.walk<WalkOrder::PreOrder>([&](memref::AllocaOp allocation) {
    uint64_t start = begin[allocation], finish = end[allocation];
    SmallVector<Value> aliases{allocation.getResult()};
    for (unsigned i = 0; i < aliases.size(); ++i) for (Operation *user : aliases[i].getUsers()) {
      if (auto view = dyn_cast<memref::ReinterpretCastOp>(user)) aliases.push_back(view.getResult());
      Operation *scope = user;
      while (scope->getBlock() != allocation->getBlock()) scope = scope->getParentOp();
      finish = std::max(finish, end[scope]);
    }
    auto type = allocation.getType();
    bool matrix = type.getMemorySpaceAsInt() == dsa::matrixSpace;
    int64_t bytes = llvm::alignTo(type.getNumElements() * llvm::divideCeil(type.getElementTypeBitWidth(), 8u), int64_t(128));
    llvm::erase_if(live, [&](const Range &range) { return range.finish < start; });
    llvm::sort(live, [](const Range &a, const Range &b) { return a.offset < b.offset; });
    int64_t offset = 0;
    for (const Range &range : live) {
      if (range.matrix != matrix) continue;
      if (offset + bytes <= range.offset) break;
      offset = std::max(offset, range.offset + range.bytes);
    }
    live.push_back({finish, offset, bytes, matrix});
    int64_t &peak = matrix ? wram : nram;
    peak = std::max(peak, offset + bytes);
    Builder builder(function.getContext());
    allocation->setAttr("bangc.offset", builder.getI64IntegerAttr(offset));
    allocation->setAttr("bangc.allocation_bytes", builder.getI64IntegerAttr(bytes));
  });
}
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
    auto convert = [&](Value source) {
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
      bool division = binary.getKind() == BinaryOperator::TrueDivide &&
          cast<MemRefType>(binary.getLhs().getType()).getElementType().isF32();
      if (!supportedBinary(binary.getKind()) || ((binary.getApproximate() || binary.getFlushToZero()) && !division)) {
        op->emitError("binary numerical mode has no selected BANG C implementation"); return WalkResult::interrupt();
      }
      if (binary.getKind() == BinaryOperator::TrueDivide)
        op->setAttr("bangc.implementation", StringAttr::get(module.getContext(), division ? "divide_f32" : "scalar_divide"));
    }
    if (auto reduction = dyn_cast<dsa::ReduceOp>(op)) {
      if (!cast<MemRefType>(reduction.getInput().getType()).getElementType().isF32() ||
          (reduction.getKind() != BinaryOperator::Add && reduction.getKind() != BinaryOperator::Maximum &&
           reduction.getKind() != BinaryOperator::MaximumNum && reduction.getKind() != BinaryOperator::Minimum &&
           reduction.getKind() != BinaryOperator::MinimumNum)) {
        op->emitError("reduction has no selected BANG C f32 local implementation"); return WalkResult::interrupt();
      }
    }
    if (!isa<dsa::SynchronizeOp, dsa::TaskIdOp, dsa::TaskCountOp, dsa::StrideOp, dsa::LoadScalarOp, dsa::StoreScalarOp,
             dsa::LoadTileOp, dsa::StoreTileOp, dsa::FillOp, dsa::SelectOp, dsa::UnaryOp, dsa::BinaryOp,
             dsa::CastOp, dsa::ReduceOp, dsa::PrepareMatrixOp, dsa::MatrixTileOp,
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
  // Scalar NRAM accesses preserve program order within a task. Complete them
  // before entering a bulk operation, and complete bulk operations before any
  // scalar consumer or allocation reuse. A fence inside every scalar store or
  // scalar carry copy would serialize each element of broadcast/reduce loops.
  SmallVector<Operation *> localEffects;
  function.walk([&](Operation *op) {
    if (isa<dsa::LoadTileOp, dsa::StoreTileOp, dsa::FillOp, dsa::SelectOp, dsa::UnaryOp,
            dsa::BinaryOp, dsa::CastOp, dsa::ReduceOp, dsa::PrepareMatrixOp, dsa::MatrixTileOp>(op))
      localEffects.push_back(op);
    if (auto copy = dyn_cast<memref::CopyOp>(op))
      if (cast<MemRefType>(copy.getSource().getType()).getNumElements() > 1)
        localEffects.push_back(op);
  });
  for (Operation *op : localEffects) {
    OpBuilder builder(op);
    if (!op->getPrevNode() || !isa<dsa::SynchronizeOp>(op->getPrevNode()))
      builder.create<dsa::SynchronizeOp>(op->getLoc());
    builder.setInsertionPointAfter(op);
    builder.create<dsa::SynchronizeOp>(op->getLoc());
  }
  int64_t nram = 0, wram = 0;
  bindStorage(function, nram, wram);
  if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024)
    return function.emitError("selected local implementation exceeds the MLU370 per-task storage budget")
        << "; NRAM=" << nram << ", WRAM=" << wram;
  Builder builder(module.getContext());
  function->setAttr("bangc.nram_bytes", builder.getI64IntegerAttr(nram));
  function->setAttr("bangc.wram_bytes", builder.getI64IntegerAttr(wram));
  return dsa::verifyProgram(module, true);
}
}

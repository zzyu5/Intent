#include "PassDetail.h"

using namespace mlir;
namespace intent::bangc {
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
    dsa::StorageAnalysis storage(function);
    auto copy = dyn_cast_or_null<dsa::LoadTileOp>(storage.lastWriterBefore(range.getRowCoordinates(), range));
    if (!copy || copy.getOutput() != range.getRowCoordinates() ||
        !exact(copy.getRows(), rows) || !exact(copy.getColumns(), 1) || !exact(copy.getOffset(), 0) ||
        !exact(copy.getRowStride(), 1) || !exact(copy.getColumnStride(), 1)) continue;
    Value source = storage.uniqueOrigin(copy.getSource());
    if (!source || !source.getDefiningOp<memref::AllocaOp>() ||
        !dsa::isCompleteStorageViewOf(copy.getSource(), source) ||
        cast<MemRefType>(source.getType()).getNumElements() != rows) continue;
    auto writers = storage.writers(source);
    if (failed(writers)) continue;
    SmallVector<dsa::FillOp> fills;
    memref::StoreOp initialization;
    bool valid = true;
    for (Operation *user : *writers) {
      if (auto store = dyn_cast<memref::StoreOp>(user); store && dsa::isCompleteStorageViewOf(store.getMemref(), source)) {
        if (initialization) valid = false;
        initialization = store; continue;
      }
      if (auto fill = dyn_cast<dsa::FillOp>(user); fill && dsa::isCompleteStorageViewOf(fill.getOutput(), source)) {
        fills.push_back(fill); continue;
      }
      valid = false;
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
    int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
    if (nram > config.getLocalBytes() || nram > 768 * 1024 || wram > 1024 * 1024) {
      replacement.erase(); scratch.getDefiningOp()->erase(); continue;
    }
    range.erase(); changed = true;
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
    int64_t nram = 0, wram = 0; measureStorage(function, nram, wram);
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
    measureStorage(function, nram, wram);
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
} // namespace intent::bangc

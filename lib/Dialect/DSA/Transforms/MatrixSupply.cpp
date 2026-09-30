#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <functional>
#include <optional>

using namespace mlir;
namespace intent::dsa {
namespace {
struct Slice {
  Value extent, begin;
};
std::optional<Slice> sliceBounds(Value count, int64_t width) {
  auto minimum = count.getDefiningOp<arith::MinSIOp>();
  if (!minimum) return std::nullopt;
  for (unsigned side = 0; side < 2; ++side) {
    APInt bound;
    if (!matchPattern(minimum->getOperand(side), m_ConstantInt(&bound)) || bound.getSExtValue() != width) continue;
    auto difference = minimum->getOperand(1 - side).getDefiningOp<arith::SubIOp>();
    if (difference) return Slice{difference.getLhs(), difference.getRhs()};
  }
  return std::nullopt;
}

// This match refers only to the existing tiled program, including its epilogue.
// It does not retain a KIR clone or authorize an alternative algorithm.
struct MatrixSupplyMatch {
  scf::ForOp work, reduction;
  MatMulOp matrix;
  LoadTileOp lhs, rhs;
  FillOp initial;
  Value m, n, rows, columns, M, N, K;
};
std::optional<MatrixSupplyMatch> matchMatrixSupply(scf::ForOp work, ConfigurationAttr config) {
  if (!work.getLowerBound().getDefiningOp<TaskIdOp>() ||
      !work.getStep().getDefiningOp<TaskCountOp>() || !work.getInitArgs().empty()) return std::nullopt;
  SmallVector<MatMulOp> matrices;
  work.walk([&](MatMulOp matrix) { matrices.push_back(matrix); });
  if (matrices.size() != 1) return std::nullopt;
  MatMulOp matrix = matrices.front();
  auto reduction = dyn_cast<scf::ForOp>(matrix->getParentOp());
  APInt step;
  if (!reduction || reduction->getBlock() != work.getBody() || !reduction.getInitArgs().empty() ||
      !matchPattern(reduction.getLowerBound(), m_Zero()) ||
      !matchPattern(reduction.getStep(), m_ConstantInt(&step)) || step.getSExtValue() != config.getTileK() ||
      matrix.getRhsTransposed()) return std::nullopt;
  auto row = sliceBounds(matrix.getRows(), config.getTileM());
  auto column = sliceBounds(matrix.getColumns(), config.getTileN());
  auto depth = sliceBounds(matrix.getDepth(), config.getTileK());
  if (!row || !column || !depth || depth->begin != reduction.getInductionVar() ||
      depth->extent != reduction.getUpperBound()) return std::nullopt;
  auto tileCoordinate = [](Value begin, int64_t width) -> Value {
    auto product = begin.getDefiningOp<arith::MulIOp>();
    if (!product) return {};
    for (unsigned side = 0; side < 2; ++side) {
      APInt constant;
      if (matchPattern(product->getOperand(side), m_ConstantInt(&constant)) && constant.getSExtValue() == width)
        return product->getOperand(1 - side);
    }
    return {};
  };
  Value rowCoordinate = tileCoordinate(row->begin, config.getTileM());
  Value columnCoordinate = tileCoordinate(column->begin, config.getTileN());
  if (!rowCoordinate || !columnCoordinate) return std::nullopt;
  auto mi = rowCoordinate.getDefiningOp<arith::DivSIOp>();
  auto ni = columnCoordinate.getDefiningOp<arith::RemSIOp>();
  auto grid = work.getUpperBound().getDefiningOp<arith::MulIOp>();
  if (!mi || !ni || !grid || mi.getLhs() != work.getInductionVar() ||
      ni.getLhs() != work.getInductionVar() || mi.getRhs() != ni.getRhs()) return std::nullopt;
  auto tileCount = [](Value value, Value extent, int64_t width) {
    auto count = value.getDefiningOp<arith::CeilDivSIOp>();
    APInt constant;
    return count && count.getLhs() == extent && matchPattern(count.getRhs(), m_ConstantInt(&constant)) &&
        constant.getSExtValue() == width;
  };
  Value gridN = mi.getRhs();
  Value gridM = grid.getLhs() == gridN ? grid.getRhs() : grid.getRhs() == gridN ? grid.getLhs() : Value{};
  if (!gridM || !tileCount(gridM, row->extent, config.getTileM()) ||
      !tileCount(gridN, column->extent, config.getTileN())) return std::nullopt;
  LoadTileOp lhs, rhs;
  for (Operation &operation : reduction.getBody()->without_terminator()) {
    if (&operation == matrix.getOperation()) continue;
    if (auto load = dyn_cast<LoadTileOp>(operation)) {
      if (load.getAsynchronous() || !isa<BlockArgument>(load.getSource())) return std::nullopt;
      if (load.getOutput() == matrix.getLhs() && !lhs) lhs = load;
      else if (load.getOutput() == matrix.getRhs() && !rhs) rhs = load;
      else return std::nullopt;
    } else if (!isMemoryEffectFree(&operation)) return std::nullopt;
  }
  if (!lhs || !rhs || lhs.getRows() != matrix.getRows() || lhs.getColumns() != matrix.getDepth() ||
      rhs.getRows() != matrix.getDepth() || rhs.getColumns() != matrix.getColumns() ||
      !lhs->isBeforeInBlock(matrix) || !rhs->isBeforeInBlock(matrix)) return std::nullopt;
  if (!isSumOfIntegerProducts(lhs.getOffset(), {{row->begin, lhs.getRowStride()}, {depth->begin, lhs.getColumnStride()}}) ||
      !isSumOfIntegerProducts(rhs.getOffset(), {{depth->begin, rhs.getRowStride()}, {column->begin, rhs.getColumnStride()}}))
    return std::nullopt;
  auto external = [](Value source) {
    auto type = dyn_cast<MemRefType>(source.getType());
    return type && type.getRank() == 2 && type.getMemorySpaceAsInt() == 0;
  };
  if (!external(lhs.getSource()) || !external(rhs.getSource())) return std::nullopt;
  auto function = work->getParentOfType<func::FuncOp>();
  auto interface = function->getAttrOfType<InterfaceAttr>("intent_dsa.interface");
  auto viewStride = [&](Value stride, Value source, unsigned axis) {
    auto argument = dyn_cast<BlockArgument>(source);
    if (!argument || argument.getOwner() != &function.front()) return false;
    auto view = dyn_cast<ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    if (!view || view.getAccess() != 0) return false;
    if (auto query = stride.getDefiningOp<StrideOp>())
      return query.getSource() == source && query.getAxis() == axis;
    if (!view.getConstraints().getHasStrides()) return false;
    auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis]);
    APInt value;
    return fixed && matchPattern(stride, m_ConstantInt(&value)) && value.getSExtValue() == fixed.getInt();
  };
  if (!viewStride(lhs.getRowStride(), lhs.getSource(), 0) ||
      !viewStride(lhs.getColumnStride(), lhs.getSource(), 1) ||
      !viewStride(rhs.getRowStride(), rhs.getSource(), 0) ||
      !viewStride(rhs.getColumnStride(), rhs.getSource(), 1)) return std::nullopt;
  auto fullDimension = [](Value extent, Value source, unsigned axis) {
    if (auto dim = extent.getDefiningOp<memref::DimOp>())
      return dim.getSource() == source && dim.getConstantIndex() == axis;
    APInt constant;
    auto type = cast<MemRefType>(source.getType());
    return !type.isDynamicDim(axis) && matchPattern(extent, m_ConstantInt(&constant)) &&
        constant.getSExtValue() == type.getDimSize(axis);
  };
  if (!fullDimension(row->extent, lhs.getSource(), 0) || !fullDimension(depth->extent, lhs.getSource(), 1) ||
      !fullDimension(column->extent, rhs.getSource(), 1)) return std::nullopt;
  auto lhsSourceType = cast<MemRefType>(lhs.getSource().getType());
  auto rhsSourceType = cast<MemRefType>(rhs.getSource().getType());
  Type element = lhsSourceType.getElementType();
  if (!element.isF16() && !element.isBF16() && !element.isF32()) return std::nullopt;
  if (!lhsSourceType.isDynamicDim(1) && !rhsSourceType.isDynamicDim(0)) {
    if (lhsSourceType.getDimSize(1) != rhsSourceType.getDimSize(0)) return std::nullopt;
  } else {
    auto leftView = cast<ViewArgumentAttr>(interface.getArguments()[cast<BlockArgument>(lhs.getSource()).getArgNumber()]);
    auto rightView = cast<ViewArgumentAttr>(interface.getArguments()[cast<BlockArgument>(rhs.getSource()).getArgNumber()]);
    int64_t reduction = leftView.getDimensions()[1];
    if (reduction <= 0 || reduction != rightView.getDimensions()[0]) return std::nullopt;
  }
  if (matrix.getLhs() == matrix.getRhs() || matrix.getAccumulator() == matrix.getLhs() ||
      matrix.getAccumulator() == matrix.getRhs()) return std::nullopt;
  SmallVector<Value> buffers{matrix.getLhs(), matrix.getRhs(), matrix.getAccumulator()};
  SmallVector<SmallVector<int64_t>> shapes{{config.getTileM(), config.getTileK()},
      {config.getTileK(), config.getTileN()}, {config.getTileM(), config.getTileN()}};
  for (auto [buffer, shape] : llvm::zip(buffers, shapes)) {
    auto allocation = buffer.getDefiningOp<memref::AllocaOp>();
    if (!allocation || allocation->getBlock() != work.getBody()) return std::nullopt;
    auto type = allocation.getType();
    if (type.getShape() != ArrayRef<int64_t>(shape) || !type.getLayout().isIdentity() ||
        type.getMemorySpaceAsInt() != nramSpace) return std::nullopt;
  }
  // Only the accumulator persists through the K traversal. All other work in
  // the outer body must be its pointwise epilogue or pure coordinate setup.
  FillOp initial;
  for (Operation &operation : work.getBody()->without_terminator()) {
    if (&operation == reduction.getOperation()) break;
    if (auto fill = dyn_cast<FillOp>(operation)) {
      FloatAttr zero;
      if (initial || fill.getOutput() != matrix.getAccumulator() ||
          !matchPattern(fill.getValue(), m_Constant(&zero)) || !zero.getValue().isZero()) return std::nullopt;
      initial = fill;
    } else if (!isa<memref::AllocaOp>(operation) && !isMemoryEffectFree(&operation)) return std::nullopt;
  }
  if (!initial) return std::nullopt;
  DenseSet<Value> available{matrix.getAccumulator(), row->begin, column->begin,
                            matrix.getRows(), matrix.getColumns()};
  DenseSet<Value> visiting;
  std::function<bool(Value)> canCapture = [&](Value value) {
    if (available.contains(value)) return true;
    if (auto argument = dyn_cast<BlockArgument>(value))
      return !work->isAncestor(argument.getOwner()->getParentOp()) &&
          argument.getOwner() != work.getBody();
    Operation *definition = value.getDefiningOp();
    if (!definition || !work->isAncestor(definition)) return true;
    if (!visiting.insert(value).second || definition->getNumRegions() || !isMemoryEffectFree(definition)) return false;
    bool legal = llvm::all_of(definition->getOperands(), canCapture);
    visiting.erase(value);
    if (legal) available.insert(value);
    return legal;
  };
  if (!canCapture(row->extent) || !canCapture(column->extent) || !canCapture(depth->extent)) return std::nullopt;
  for (Operation *operation = reduction->getNextNode(); operation && !operation->hasTrait<OpTrait::IsTerminator>();
       operation = operation->getNextNode()) {
    if (operation->getNumRegions() ||
        (!isa<memref::AllocaOp, StoreTileOp, UnaryOp, BinaryOp, CastOp, SelectOp, FillOp>(operation) &&
         !isMemoryEffectFree(operation))) return std::nullopt;
    if (!llvm::all_of(operation->getOperands(), canCapture)) return std::nullopt;
    available.insert(operation->result_begin(), operation->result_end());
  }
  return MatrixSupplyMatch{work, reduction, matrix, lhs, rhs, initial, row->begin, column->begin,
      matrix.getRows(), matrix.getColumns(), row->extent, column->extent, depth->extent};
}

class MatrixSupplyRewrite {
public:
  explicit MatrixSupplyRewrite(MatrixSupplyMatch match, ConfigurationAttr config)
      : match(match), config(config), function(match.work->getParentOfType<func::FuncOp>()), b(match.work) {}
  LogicalResult run(bool resident) {
    auto matrix = match.matrix;
    Location loc = matrix.getLoc();
    Value lhsSource = match.lhs.getSource(), rhsSource = match.rhs.getSource();
    Value M = capture(match.M), N = capture(match.N), K = capture(match.K);
    Value tm = index(loc, config.getTileM()), tn = index(loc, config.getTileN()), tk = index(loc, config.getTileK());
    Value gridM = b.create<arith::CeilDivSIOp>(loc, M, tm), gridN = b.create<arith::CeilDivSIOp>(loc, N, tn);
    auto aType = cast<MemRefType>(lhsSource.getType()), bType = cast<MemRefType>(rhsSource.getType());
    Type matrixElement = aType.getElementType();
    int64_t elementBytes = matrixElement.getIntOrFloatBitWidth() / 8;
    auto emitOutput = [&](Value accumulator, Value m, Value n, Value rows, Value columns) {
      IRMapping mapping;
      mapping.map(match.matrix.getAccumulator(), accumulator);
      mapping.map(match.m, m); mapping.map(match.n, n);
      mapping.map(match.rows, rows); mapping.map(match.columns, columns);
      std::function<Value(Value)> operand = [&](Value value) -> Value {
        if (Value replacement = mapping.lookupOrNull(value)) return replacement;
        Operation *definition = value.getDefiningOp();
        if (!definition || !match.work->isAncestor(definition)) return value;
        for (Value input : definition->getOperands()) mapping.map(input, operand(input));
        b.clone(*definition, mapping);
        return mapping.lookup(value);
      };
      for (Operation *operation = match.reduction->getNextNode(); !operation->hasTrait<OpTrait::IsTerminator>();
           operation = operation->getNextNode()) {
        for (Value input : operation->getOperands()) mapping.map(input, operand(input));
        b.clone(*operation, mapping);
      }
      return success();
    };
    if (resident) {
      // A resident RHS panel is reused across output-row blocks. The group's
      // four column programs consume the same streamed LHS row block.
      function->setAttr("intent_dsa.group_width", b.getI64IntegerAttr(4));
      Value groupId = b.create<dsa::GroupIdOp>(loc, b.getIndexType());
      Value groupCount = b.create<dsa::GroupCountOp>(loc, b.getIndexType());
      Value localId = b.create<dsa::LocalIdOp>(loc, b.getIndexType());
      Value memoryCore = b.create<dsa::IsMemoryCoreOp>(loc, b.getI1Type());
      Value computeCore = b.create<arith::XOrIOp>(loc, memoryCore, b.create<arith::ConstantIntOp>(loc, 1, 1));
      Value groupsN = b.create<arith::CeilDivSIOp>(loc, gridN, index(loc, 4));
      auto sharedType = MemRefType::get({config.getTileM(), config.getTileK()}, matrixElement,
          MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(dsa::sharedSpace));
      auto shared = b.create<memref::AllocaOp>(loc, sharedType);
      shared.setAlignment(128);
      auto alternate = b.create<memref::AllocaOp>(loc, sharedType);
      alternate.setAlignment(128);
      auto when = [&](Value condition, const std::function<LogicalResult()> &body) -> LogicalResult {
        auto branch = b.create<scf::IfOp>(loc, condition, false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.thenBlock());
        return body();
      };
      int64_t sliceN = 256;
      while (config.getTileN() % sliceN) sliceN /= 2;
      int64_t sliceK = 2048;
      while (sliceK > config.getTileK() || config.getTileK() % sliceK ||
             2 * sliceK * sliceN * elementBytes > config.getLocalBytes()) sliceK /= 2;
      return loop(loc, groupId, groupsN, groupCount, [&](Value group) -> LogicalResult {
        Value ni = add(loc, mul(loc, group, index(loc, 4)), localId);
        Value n0 = mul(loc, ni, tn);
        Value cols = b.create<arith::MaxSIOp>(loc, index(loc, 0), b.create<arith::MinSIOp>(loc, sub(loc, N, n0), tn));
        Value active = b.create<arith::AndIOp>(loc, computeCore,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ni, gridN));
        auto packedType = MemRefType::get({config.getTileK(), config.getTileN()}, matrixElement,
            MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(dsa::matrixSpace));
        auto packed = b.create<memref::AllocaOp>(loc, packedType);
        packed.setAlignment(128);
        if (failed(when(active, [&]() -> LogicalResult {
          Value raw = allocate(loc, matrixElement, sliceK, sliceN);
          Value transposed = allocate(loc, matrixElement, sliceN, sliceK);
          Value br = stride(loc, rhsSource, 0), bc = stride(loc, rhsSource, 1);
          b.create<dsa::PrepareMatrixViewOp>(loc, rhsSource, packed, raw, transposed, mul(loc, n0, bc), br, bc, cols);
          return success();
        }))) return failure();
        auto supply = [&](Value m0, Value slot) -> LogicalResult {
          return when(memoryCore, [&]() -> LogicalResult {
            Value rows = b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm);
            Value ar = stride(loc, lhsSource, 0), ac = stride(loc, lhsSource, 1);
            b.create<dsa::StageTileOp>(loc, lhsSource, slot, mul(loc, m0, ar), ar, ac, rows, K);
            return success();
          });
        };
        const bool pipelineLocal = config.getTileM() *
            (2 * config.getTileK() * elementBytes + config.getTileN() * 4) <= config.getLocalBytes();
        Value local0, local1;
        if (pipelineLocal) {
          local0 = allocate(loc, matrixElement, config.getTileM(), config.getTileK());
          local1 = allocate(loc, matrixElement, config.getTileM(), config.getTileK());
        }
        auto compute = [&](Value m0, Value slot, Value nextShared = Value{}, Value nextLocal = Value{}, Value nextRow = Value{}) -> LogicalResult {
          return when(active, [&]() -> LogicalResult {
            Value rows = b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm);
            Value lhs = slot;
            if (!pipelineLocal) {
              lhs = allocate(loc, matrixElement, config.getTileM(), config.getTileK());
              b.create<dsa::LoadTileOp>(loc, slot, lhs, index(loc, 0), tk, index(loc, 1), rows, K);
            }
            Value accumulator = allocate(loc, b.getF32Type(), config.getTileM(), config.getTileN());
            b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
            if (nextShared && failed(when(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, nextRow, M), [&]() {
              Value nextRows = b.create<arith::MinSIOp>(loc, sub(loc, M, nextRow), tm);
              b.create<dsa::LoadTileOp>(loc, nextShared, nextLocal, index(loc, 0), tk, index(loc, 1), nextRows, K, b.getBoolAttr(true));
              return success();
            }))) return failure();
            b.create<dsa::MatrixTileOp>(loc, lhs, packed, accumulator);
            return emitOutput(accumulator, m0, n0, rows, cols);
          });
        };
        if (failed(supply(index(loc, 0), shared))) return failure();
        if (pipelineLocal) {
          Value second = tm;
          if (failed(when(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, second, M),
                          [&]() { return supply(second, alternate); }))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          if (failed(when(active, [&]() {
            Value rows = b.create<arith::MinSIOp>(loc, M, tm);
            b.create<dsa::LoadTileOp>(loc, shared, local0, index(loc, 0), tk, index(loc, 1), rows, K);
            return success();
          }))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          return loop(loc, index(loc, 0), M, index(loc, 2 * config.getTileM()), [&](Value m0) -> LogicalResult {
            Value next = add(loc, m0, tm), following = add(loc, next, tm);
            Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, M);
            Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, M);
            if (failed(when(hasFollowing, [&]() { return supply(following, shared); }))) return failure();
            if (failed(compute(m0, local0, alternate, local1, next))) return failure();
            b.create<dsa::GroupSynchronizeOp>(loc);
            return when(hasNext, [&]() -> LogicalResult {
              Value third = add(loc, following, tm);
              if (failed(when(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, third, M),
                              [&]() { return supply(third, alternate); }))) return failure();
              if (failed(compute(next, local1, shared, local0, following))) return failure();
              b.create<dsa::GroupSynchronizeOp>(loc);
              return success();
            });
          });
        }
        b.create<dsa::GroupSynchronizeOp>(loc);
        return loop(loc, index(loc, 0), M, index(loc, 2 * config.getTileM()), [&](Value m0) -> LogicalResult {
          Value next = add(loc, m0, tm);
          Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, M);
          if (failed(when(hasNext, [&]() { return supply(next, alternate); }))) return failure();
          if (failed(compute(m0, shared))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          return when(hasNext, [&]() -> LogicalResult {
            Value following = add(loc, next, tm);
            Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, M);
            if (failed(when(hasFollowing, [&]() { return supply(following, shared); }))) return failure();
            if (failed(compute(next, alternate))) return failure();
            b.create<dsa::GroupSynchronizeOp>(loc);
            return success();
          });
        });
      });
    }
    {
      // Four independent M tiles reuse one RHS panel supplied by the group's
      // memory participant. Local matrix computation remains explicit below.
      function->setAttr("intent_dsa.group_width", b.getI64IntegerAttr(4));
      Value groupId = b.create<dsa::GroupIdOp>(loc, b.getIndexType());
      Value groupCount = b.create<dsa::GroupCountOp>(loc, b.getIndexType());
      Value localId = b.create<dsa::LocalIdOp>(loc, b.getIndexType());
      Value memoryCore = b.create<dsa::IsMemoryCoreOp>(loc, b.getI1Type());
      Value computeCore = b.create<arith::XOrIOp>(loc, memoryCore, b.create<arith::ConstantIntOp>(loc, 1, 1));
      Value groupsM = b.create<arith::CeilDivSIOp>(loc, gridM, index(loc, 4));
      Type element = aType.getElementType();
      int64_t panelBytes = config.getTileK() * config.getTileN() * (element.getIntOrFloatBitWidth() / 8);
      int64_t stageTiles = std::min<int64_t>(8, (3968 * 1024) / (2 * panelBytes));
      if (!stageTiles) return matrix.emitError("matrix panel exceeds execution-group shared storage");
      int64_t stageRows = stageTiles * config.getTileK();
      auto sharedType = MemRefType::get({stageRows, config.getTileN()}, element,
          MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(dsa::sharedSpace));
      auto shared = b.create<memref::AllocaOp>(loc, sharedType);
      shared.setAlignment(128);
      auto alternate = b.create<memref::AllocaOp>(loc, sharedType);
      alternate.setAlignment(128);
      auto when = [&](Value condition, const std::function<LogicalResult()> &body) -> LogicalResult {
        auto branch = b.create<scf::IfOp>(loc, condition, false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(branch.thenBlock());
        return body();
      };
      return loop(loc, groupId, mul(loc, groupsM, gridN), groupCount, [&](Value group) -> LogicalResult {
        Value mi = add(loc, mul(loc, b.create<arith::DivSIOp>(loc, group, gridN), index(loc, 4)), localId);
        Value ni = b.create<arith::RemSIOp>(loc, group, gridN);
        Value m0 = mul(loc, mi, tm), n0 = mul(loc, ni, tn);
        Value rows = b.create<arith::MaxSIOp>(loc, index(loc, 0),
            b.create<arith::MinSIOp>(loc, sub(loc, M, m0), tm));
        Value cols = b.create<arith::MinSIOp>(loc, sub(loc, N, n0), tn);
        Value active = b.create<arith::AndIOp>(loc, computeCore,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, mi, gridM));
        Value accumulator = allocate(loc, b.getF32Type(), config.getTileM(), config.getTileN());
        if (failed(when(active, [&]() {
          b.create<dsa::FillOp>(loc, accumulator, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0)));
          return success();
        }))) return failure();
        auto supply = [&](Value stage, Value slot) -> LogicalResult {
          return when(memoryCore, [&]() {
            Value stageDepth = b.create<arith::MinSIOp>(loc, sub(loc, K, stage), index(loc, stageRows));
            Value br = stride(loc, rhsSource, 0), bc = stride(loc, rhsSource, 1);
            b.create<dsa::StageTileOp>(loc, rhsSource, slot, add(loc, mul(loc, stage, br), mul(loc, n0, bc)),
                br, bc, stageDepth, cols);
            return success();
          });
        };
        auto compute = [&](Value stage, Value slot) -> LogicalResult {
          return when(active, [&]() {
            Value stageDepth = b.create<arith::MinSIOp>(loc, sub(loc, K, stage), index(loc, stageRows));
            Value lhs = allocate(loc, element, config.getTileM(), config.getTileK());
            Value rhs = allocate(loc, element, config.getTileK(), config.getTileN());
            return loop(loc, index(loc, 0), stageDepth, tk, [&](Value within) -> LogicalResult {
              Value depth = b.create<arith::MaxSIOp>(loc, index(loc, 0),
                  b.create<arith::MinSIOp>(loc, sub(loc, stageDepth, within), tk));
              Value k0 = add(loc, stage, within);
              Value ar = stride(loc, lhsSource, 0), ac = stride(loc, lhsSource, 1);
              b.create<dsa::LoadTileOp>(loc, lhsSource, lhs, add(loc, mul(loc, m0, ar), mul(loc, k0, ac)), ar, ac, rows, depth);
              b.create<dsa::LoadTileOp>(loc, slot, rhs, mul(loc, within, tn), tn, index(loc, 1), depth, cols);
              b.create<dsa::MatMulOp>(loc, lhs, rhs, accumulator, rows, depth, cols);
              return success();
            });
          });
        };
        if (failed(supply(index(loc, 0), shared))) return failure();
        b.create<dsa::GroupSynchronizeOp>(loc);
        if (failed(loop(loc, index(loc, 0), K, index(loc, 2 * stageRows), [&](Value stage) -> LogicalResult {
          Value next = add(loc, stage, index(loc, stageRows));
          Value hasNext = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, next, K);
          if (failed(when(hasNext, [&]() { return supply(next, alternate); }))) return failure();
          if (failed(compute(stage, shared))) return failure();
          b.create<dsa::GroupSynchronizeOp>(loc);
          return when(hasNext, [&]() -> LogicalResult {
            Value following = add(loc, next, index(loc, stageRows));
            Value hasFollowing = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, following, K);
            if (failed(when(hasFollowing, [&]() { return supply(following, shared); }))) return failure();
            if (failed(compute(next, alternate))) return failure();
            // Inactive compute participants also reach both group barriers.
            b.create<dsa::GroupSynchronizeOp>(loc);
            return success();
          });
        }))) return failure();
        return when(active, [&]() { return emitOutput(accumulator, m0, n0, rows, cols); });
      });
    }
  }
private:
  Value capture(Value value) {
    Operation *definition = value.getDefiningOp();
    if (!definition || !match.work->isAncestor(definition)) return value;
    if (Value replacement = captures.lookupOrNull(value)) return replacement;
    for (Value input : definition->getOperands()) captures.map(input, capture(input));
    b.clone(*definition, captures);
    return captures.lookup(value);
  }
  Value index(Location loc, int64_t value) { return b.create<arith::ConstantIndexOp>(loc, value); }
  Value add(Location loc, Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value sub(Location loc, Value a, Value c) { return b.create<arith::SubIOp>(loc, a, c); }
  Value mul(Location loc, Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value stride(Location loc, Value source, unsigned axis) {
    auto function = match.work->getParentOfType<func::FuncOp>();
    auto interface = function->getAttrOfType<InterfaceAttr>("intent_dsa.interface");
    auto argument = cast<BlockArgument>(source);
    auto view = cast<ViewArgumentAttr>(interface.getArguments()[argument.getArgNumber()]);
    if (view.getConstraints().getHasStrides())
      if (auto fixed = dyn_cast<IntegerAttr>(view.getConstraints().getStrides()[axis])) return index(loc, fixed.getInt());
    return b.create<StrideOp>(loc, b.getIndexType(), source, axis);
  }
  Value allocate(Location loc, Type element, int64_t rows, int64_t columns) {
    auto type = MemRefType::get({rows, columns}, element, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(nramSpace));
    auto allocation = b.create<memref::AllocaOp>(loc, type); allocation.setAlignment(128); return allocation;
  }
  LogicalResult loop(Location loc, Value lower, Value upper, Value step,
                     const std::function<LogicalResult(Value)> &body) {
    auto loop = b.create<scf::ForOp>(loc, lower, upper, step);
    OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(loop.getBody());
    return body(loop.getInductionVar());
  }
  MatrixSupplyMatch match;
  ConfigurationAttr config;
  func::FuncOp function;
  OpBuilder b;
  IRMapping captures;
};
} // namespace

LogicalResult realizeMatrixSupply(func::FuncOp function) {
  auto config = function->getAttrOfType<ConfigurationAttr>("intent_dsa.configuration");
  SmallVector<scf::ForOp> worksets;
  function.walk([&](scf::ForOp loop) {
    if (loop.getLowerBound().getDefiningOp<TaskIdOp>()) worksets.push_back(loop);
  });
  for (auto work : worksets) {
    auto match = matchMatrixSupply(work, config);
    if (!match) continue;
    auto a = cast<MemRefType>(match->lhs.getSource().getType());
    auto c = cast<MemRefType>(match->rhs.getSource().getType());
    Type element = a.getElementType();
    int64_t bytes = element.getIntOrFloatBitWidth() / 8;
    bool resident = a.hasStaticShape() && c.hasStaticShape() && a.getDimSize(1) == config.getTileK() &&
        (element.isF16() || element.isBF16() || element.isF32()) && config.getLocalBytes() >= 8192 &&
        config.getTileN() % 64 == 0 && config.getTileK() * bytes % 64 == 0 &&
        config.getTileK() * config.getTileN() * bytes <= 1024 * 1024 &&
        config.getTileM() * (config.getTileK() * bytes + config.getTileN() * 4) <= config.getLocalBytes() &&
        config.getTasks() >= 4 && config.getTasks() % 4 == 0 &&
        c.getDimSize(1) / config.getTileN() >= config.getTasks();
    bool shared = config.getTasks() >= 4 && config.getTasks() % 4 == 0 &&
        a.getDimSize(0) / config.getTileM() >= 4;
    if (!resident && !shared) continue;
    if (failed(MatrixSupplyRewrite(*match, config).run(resident))) return failure();
    work.erase();
  }
  return success();
}
} // namespace intent::dsa

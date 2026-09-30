#include "ContiguousAccesses.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::cpu {
namespace {

struct BoundedCoordinate {
  AffineExpr expression;
  int64_t minimum, maximum;
};

std::optional<BoundedCoordinate> add(BoundedCoordinate lhs, BoundedCoordinate rhs,
                                     bool subtract = false) {
  int64_t minimum, maximum;
  if (subtract) {
    if (llvm::SubOverflow(lhs.minimum, rhs.maximum, minimum) ||
        llvm::SubOverflow(lhs.maximum, rhs.minimum, maximum)) return std::nullopt;
  } else if (llvm::AddOverflow(lhs.minimum, rhs.minimum, minimum) ||
             llvm::AddOverflow(lhs.maximum, rhs.maximum, maximum)) return std::nullopt;
  return BoundedCoordinate{subtract ? lhs.expression - rhs.expression : lhs.expression + rhs.expression,
                           minimum, maximum};
}

std::optional<BoundedCoordinate> multiply(BoundedCoordinate input, int64_t factor) {
  int64_t first, last;
  if (llvm::MulOverflow(input.minimum, factor, first) ||
      llvm::MulOverflow(input.maximum, factor, last)) return std::nullopt;
  return BoundedCoordinate{input.expression * factor, std::min(first, last), std::max(first, last)};
}

int64_t floorDivide(int64_t value, int64_t divisor) {
  // divisor is positive; signed_min / -1 is excluded by construction.
  return value / divisor - (value % divisor < 0);
}

std::optional<SmallVector<int64_t>> rowMajorStrides(MemRefType type) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity()) return std::nullopt;
  SmallVector<int64_t> strides(type.getRank());
  int64_t stride = 1;
  for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
    if (type.getDimSize(axis) <= 0) return std::nullopt;
    strides[axis] = stride;
    if (llvm::MulOverflow(stride, type.getDimSize(axis), stride)) return std::nullopt;
  }
  return strides;
}

// AffineExpr denotes mathematical integers. Only expressions whose actual
// current-IR domain fits the DSL's signed 64-bit index arithmetic enter it.
// No candidate extent or two unknown strides establish a numerical bound.
class Coordinates {
public:
  Coordinates(MLIRContext *context, ArrayRef<int64_t> sizes)
      : context(context), sizes(sizes.begin(), sizes.end()) {}

  void bind(Value value, unsigned axis) {
    values[value] = BoundedCoordinate{getAffineDimExpr(axis, context), 0, sizes[axis] - 1};
  }

  std::optional<BoundedCoordinate> get(Value value) {
    if (!value.getType().isIndex() && !value.getType().isSignlessInteger(64)) return std::nullopt;
    auto found = values.find(value);
    if (found != values.end()) return found->second;
    auto result = compute(value);
    values[value] = result;
    return result;
  }

  ArrayRef<Value> getSymbols() const { return symbols; }

  std::optional<AffineExpr> displacement(Value memory, ValueRange indices,
                                        ArrayRef<int64_t> destinationStrides) {
    auto type = cast<MemRefType>(memory.getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset)) || ShapedType::isDynamic(offset) ||
        llvm::any_of(strides, ShapedType::isDynamic)) return std::nullopt;
    BoundedCoordinate address{getAffineConstantExpr(0, context), 0, 0};
    for (auto [index, stride] : llvm::zip_equal(indices, strides)) {
      auto coordinate = get(index);
      auto term = coordinate ? multiply(*coordinate, stride) : std::nullopt;
      auto sum = term ? add(address, *term) : std::nullopt;
      if (!sum) return std::nullopt;
      address = *sum;
    }
    if (!add(address, {getAffineConstantExpr(offset, context), offset, offset})) return std::nullopt;
    AffineExpr linear = getAffineConstantExpr(0, context);
    for (auto [axis, stride] : llvm::enumerate(destinationStrides))
      linear = linear + getAffineDimExpr(axis, context) * stride;
    AffineExpr difference = simplifyAffineExpr(address.expression - linear, sizes.size(), symbols.size());
    bool memberDependent = false;
    difference.walk([&](AffineExpr expression) { memberDependent |= isa<AffineDimExpr>(expression); });
    if (memberDependent) return std::nullopt;
    // Simplification can reassociate arithmetic. Validate the operations that
    // will actually be materialized, including the source descriptor offset.
    auto bound = materializedBounds(difference);
    if (!bound || !add(*bound, {getAffineConstantExpr(offset, context), offset, offset})) return std::nullopt;
    return difference;
  }

private:
  std::optional<BoundedCoordinate> materializedBounds(AffineExpr expression) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expression))
      return BoundedCoordinate{expression, constant.getValue(), constant.getValue()};
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
      return values.lookup(symbols[symbol.getPosition()]);
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary) return std::nullopt;
    auto lhs = materializedBounds(binary.getLHS()), rhs = materializedBounds(binary.getRHS());
    if (!lhs || !rhs) return std::nullopt;
    if (expression.getKind() == AffineExprKind::Add) return add(*lhs, *rhs);
    if (expression.getKind() == AffineExprKind::Mul) {
      if (auto factor = dyn_cast<AffineConstantExpr>(binary.getRHS())) return multiply(*lhs, factor.getValue());
      if (auto factor = dyn_cast<AffineConstantExpr>(binary.getLHS())) return multiply(*rhs, factor.getValue());
      return std::nullopt;
    }
    auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
    if (!divisor || divisor.getValue() <= 0) return std::nullopt;
    BoundedCoordinate quotient{expression, floorDivide(lhs->minimum, divisor.getValue()),
                                floorDivide(lhs->maximum, divisor.getValue())};
    if (expression.getKind() == AffineExprKind::FloorDiv) return quotient;
    if (expression.getKind() != AffineExprKind::Mod) return std::nullopt;
    auto product = multiply(quotient, divisor.getValue());
    if (!product || !add(*lhs, *product, true)) return std::nullopt;
    return BoundedCoordinate{expression, 0, divisor.getValue() - 1};
  }

  std::optional<BoundedCoordinate> compute(Value value) {
    if (auto constant = getConstantIntValue(value))
      return BoundedCoordinate{getAffineConstantExpr(*constant, context), *constant, *constant};
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) return get(cast.getIn());
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Value lower, upper, step;
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
          loop && argument == loop.getInductionVar()) {
        lower = loop.getLowerBound(); upper = loop.getUpperBound(); step = loop.getStep();
      } else if (auto parallel = dyn_cast<scf::ParallelOp>(argument.getOwner()->getParentOp())) {
        auto position = llvm::find(parallel.getInductionVars(), value);
        if (position == parallel.getInductionVars().end()) return std::nullopt;
        unsigned axis = position - parallel.getInductionVars().begin();
        lower = parallel.getLowerBound()[axis]; upper = parallel.getUpperBound()[axis];
        step = parallel.getStep()[axis];
      } else return std::nullopt;
      auto first = getConstantIntValue(lower), end = getConstantIntValue(upper), increment = getConstantIntValue(step);
      if (!first || !end || !increment || *increment <= 0 || *end <= *first) return std::nullopt;
      auto expression = getAffineSymbolExpr(symbols.size(), context);
      symbols.push_back(value);
      return BoundedCoordinate{expression, *first, *end - 1};
    }
    Operation *operation = value.getDefiningOp();
    if (!operation || operation->getNumOperands() != 2) return std::nullopt;
    Value lhs = operation->getOperand(0), rhs = operation->getOperand(1);
    if (isa<arith::AddIOp, arith::SubIOp>(operation)) {
      auto left = get(lhs), right = get(rhs);
      return left && right ? add(*left, *right, isa<arith::SubIOp>(operation)) : std::nullopt;
    }
    if (isa<arith::MulIOp>(operation)) {
      auto factor = getConstantIntValue(rhs);
      if (!factor) { factor = getConstantIntValue(lhs); std::swap(lhs, rhs); }
      auto input = factor ? get(lhs) : std::nullopt;
      return input ? multiply(*input, *factor) : std::nullopt;
    }
    if (!isa<arith::FloorDivSIOp, arith::DivSIOp, arith::RemSIOp>(operation)) return std::nullopt;
    auto divisor = getConstantIntValue(rhs);
    auto input = get(lhs);
    if (!divisor || *divisor <= 0 || !input) return std::nullopt;
    if (!isa<arith::FloorDivSIOp>(operation) && input->minimum < 0) return std::nullopt;
    if (isa<arith::RemSIOp>(operation))
      return BoundedCoordinate{input->expression % *divisor, 0, std::min(input->maximum, *divisor - 1)};
    return BoundedCoordinate{input->expression.floorDiv(*divisor),
                             floorDivide(input->minimum, *divisor), floorDivide(input->maximum, *divisor)};
  }

  MLIRContext *context;
  SmallVector<int64_t> sizes;
  SmallVector<Value> symbols;
  llvm::DenseMap<Value, std::optional<BoundedCoordinate>> values;
};

Value materialize(AffineExpr expression, ArrayRef<Value> symbols, OpBuilder &builder, Location loc) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return builder.create<arith::ConstantIndexOp>(loc, constant.getValue());
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) return symbols[symbol.getPosition()];
  auto binary = cast<AffineBinaryOpExpr>(expression);
  Value lhs = materialize(binary.getLHS(), symbols, builder, loc);
  Value rhs = materialize(binary.getRHS(), symbols, builder, loc);
  switch (expression.getKind()) {
  case AffineExprKind::Add: return builder.createOrFold<arith::AddIOp>(loc, lhs, rhs);
  case AffineExprKind::Mul: return builder.createOrFold<arith::MulIOp>(loc, lhs, rhs);
  case AffineExprKind::FloorDiv: return builder.createOrFold<arith::FloorDivSIOp>(loc, lhs, rhs);
  case AffineExprKind::Mod: {
    Value quotient = builder.createOrFold<arith::FloorDivSIOp>(loc, lhs, rhs);
    Value product = builder.createOrFold<arith::MulIOp>(loc, quotient, rhs);
    return builder.createOrFold<arith::SubIOp>(loc, lhs, product);
  }
  default: llvm_unreachable("coordinate composition emitted an unsupported affine expression");
  }
}

Value contiguousView(Value source, MemRefType shape, ArrayRef<int64_t> strides,
                     AffineExpr displacement, ArrayRef<Value> symbols, Operation *before) {
  OpBuilder builder(before);
  Location loc = before->getLoc();
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, source);
  Value offset = builder.createOrFold<arith::AddIOp>(loc, metadata.getOffset(),
      materialize(displacement, symbols, builder, loc));
  SmallVector<OpFoldResult> sizes, steps;
  for (int64_t size : shape.getShape()) sizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides) steps.push_back(builder.getIndexAttr(stride));
  int64_t knownOffset = getConstantIntValue(offset).value_or(ShapedType::kDynamic);
  auto type = MemRefType::get(shape.getShape(), shape.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), knownOffset, strides), shape.getMemorySpace());
  return builder.create<memref::ReinterpretCastOp>(loc, type, metadata.getBaseBuffer(),
      OpFoldResult(offset), sizes, steps);
}

bool foldRead(linalg::GenericOp producer, func::FuncOp function) {
  if (producer.getNumResults() || !producer.getInputs().empty() || producer.getOutputs().size() != 1 ||
      producer.getNumReductionLoops() || !producer.getIndexingMapsArray().back().isIdentity()) return false;
  auto allocation = producer.getOutputs()[0].getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation->getBlock() != producer->getBlock()) return false;
  auto strides = rowMajorStrides(allocation.getType());
  auto lifetime = queryStorageLifetime(allocation);
  if (!strides || !lifetime || !lifetime->aliases.complete || lifetime->aliases.values.size() != 1) return false;
  Block &body = producer.getRegion().front();
  if (!body.getArguments().back().use_empty()) return false;
  auto read = body.getTerminator()->getOperand(0).getDefiningOp<memref::LoadOp>();
  if (!read || read->getBlock() != &body) return false;
  Coordinates coordinates(function.getContext(), allocation.getType().getShape());
  for (Operation &operation : body.without_terminator()) {
    if (auto index = dyn_cast<linalg::IndexOp>(operation)) coordinates.bind(index.getResult(), index.getDim());
    else if (&operation != read && (!isMemoryEffectFree(&operation) || operation.getNumRegions())) return false;
  }
  auto sourceType = read.getMemRefType();
  if (sourceType.getElementType() != allocation.getType().getElementType() ||
      sourceType.getMemorySpace() != allocation.getType().getMemorySpace()) return false;
  auto displacement = coordinates.displacement(read.getMemref(), read.getIndices(), *strides);
  if (!displacement) return false;
  DominanceInfo dominance(function);
  if (!dominance.dominates(read.getMemref(), allocation) ||
      llvm::any_of(coordinates.getSymbols(), [&](Value symbol) { return !dominance.dominates(symbol, allocation); })) return false;
  PhysicalProgramAnalysis physical(function);
  Value sourceRoot = physical.storageRoot(read.getMemref());
  if (sourceRoot == allocation.getResult()) return false;
  for (Operation *user : lifetime->aliases.users) {
    if (user == producer || user == lifetime->end || isa<memref::DimOp>(user)) continue;
    // An effect-free operation can still observe the backing pointer or its
    // alignment. Only element-value consumers may replace a snapshot by a view.
    bool elements = isa<memref::LoadOp, memref::CopyOp, linalg::LinalgOp, ReduceOp>(user);
    if (auto scan = dyn_cast<ScanOp>(user))
      elements = !scan.isDestinationPassing() &&
          !llvm::is_contained(scan.getCaptures(), allocation.getResult()) &&
          !llvm::is_contained(scan.getInitials(), allocation.getResult());
    if (!elements) return false;
    if (!preservesStorage(user, allocation) ||
        !isStorageReadStable(read.getMemref(), producer, user)) return false;
  }
  Value view = contiguousView(read.getMemref(), allocation.getType(), *strides,
                              *displacement, coordinates.getSymbols(), allocation);
  producer.erase();
  lifetime->end.erase();
  allocation.replaceAllUsesWith(view);
  allocation.erase();
  return true;
}

bool composeWrite(memref::StoreOp store, func::FuncOp function) {
  auto read = store.getValue().getDefiningOp<memref::LoadOp>();
  auto allocation = read ? read.getMemref().getDefiningOp<memref::AllocOp>() : memref::AllocOp();
  if (!allocation || !read.getResult().hasOneUse() || read->getBlock() != store->getBlock()) return false;
  auto strides = rowMajorStrides(allocation.getType());
  auto lifetime = queryStorageLifetime(allocation);
  if (!strides || !lifetime || !lifetime->aliases.complete || !allocation.getType().getRank()) return false;
  SmallVector<scf::ForOp> loops;
  Operation *root = store;
  while (root->getBlock() != allocation->getBlock()) {
    auto loop = dyn_cast<scf::ForOp>(root->getParentOp());
    if (!loop || loop.getNumResults() || !matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One())) return false;
    loops.push_back(loop);
    root = loop;
  }
  std::reverse(loops.begin(), loops.end());
  if (loops.size() != strides->size()) return false;
  Coordinates coordinates(function.getContext(), allocation.getType().getShape());
  for (auto [axis, loop] : llvm::enumerate(loops)) {
    if (getConstantIntValue(loop.getUpperBound()) != allocation.getType().getDimSize(axis) ||
        read.getIndices()[axis] != loop.getInductionVar()) return false;
    coordinates.bind(loop.getInductionVar(), axis);
  }
  bool valid = true;
  root->walk([&](Operation *operation) {
    if (operation == read || operation == store || isa<scf::ForOp, scf::YieldOp>(operation)) return;
    if (operation->getNumRegions() || !isMemoryEffectFree(operation)) valid = false;
  });
  auto destination = store.getMemRefType();
  if (!valid || destination.getElementType() != allocation.getType().getElementType() ||
      destination.getMemorySpace() != allocation.getType().getMemorySpace()) return false;
  // memref.copy itself must preserve the scalar traversal before the separate
  // output-forwarding transform considers it. Overlapping ranges stay as loops.
  if (!areDisjointStorage(allocation.getResult(), store.getMemref(), function)) return false;
  auto displacement = coordinates.displacement(store.getMemref(), store.getIndices(), *strides);
  if (!displacement) return false;
  DominanceInfo dominance(function);
  if (!dominance.dominates(store.getMemref(), allocation) ||
      llvm::any_of(coordinates.getSymbols(), [&](Value symbol) { return !dominance.dominates(symbol, allocation); })) return false;
  Value view = contiguousView(store.getMemref(), allocation.getType(), *strides,
                              *displacement, coordinates.getSymbols(), allocation);
  OpBuilder(root).create<memref::CopyOp>(root->getLoc(), allocation.getResult(), view);
  root->erase();
  return true;
}

} // namespace

void foldContiguousAccesses(func::FuncOp function) {
  bool changed;
  do {
    changed = false;
    SmallVector<linalg::GenericOp> readers;
    function.walk([&](linalg::GenericOp operation) { readers.push_back(operation); });
    for (auto reader : readers)
      if (foldRead(reader, function)) { changed = true; break; }
  } while (changed);
  do {
    changed = false;
    SmallVector<memref::StoreOp> writers;
    function.walk([&](memref::StoreOp operation) { writers.push_back(operation); });
    for (auto writer : writers)
      if (composeWrite(writer, function)) { changed = true; break; }
  } while (changed);
}

} // namespace intent::cpu

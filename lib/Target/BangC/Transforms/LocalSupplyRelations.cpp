#include "LocalSupplyRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
namespace intent::bangc {
namespace {

IntegerRangePolicy supplyPolicy(func::FuncOp function,
    std::function<std::optional<int64_t>(Value)> constant) {
  IntegerRangePolicy policy;
  policy.infer = [function, constant](Value value, IntegerRangeAnalysis &analysis)
      -> std::optional<ConstantIntRanges> {
    if (value.getDefiningOp<dsa::TaskIdOp>() || value.getDefiningOp<dsa::TaskCountOp>()) {
      if (auto bounds = dsa::integerInterval(value, function))
        return ConstantIntRanges::fromSigned(APInt(64, bounds->first), APInt(64, bounds->second));
      return std::nullopt;
    }
    auto argument = dyn_cast<BlockArgument>(value);
    auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
    if (!loop || loop.getInductionVar() != value || !value.getType().isIndex())
      return std::nullopt;
    auto bound = [&](Value value) {
      auto range = analysis.range(value);
      // The existing no-wrap affine relation can retain correlation lost by
      // interval arithmetic, e.g. a clipped tile end minus its own begin.
      if (auto exact = constant(value))
        range = ConstantIntRanges::constant(APInt(64, *exact, true));
      return range;
    };
    auto lower = bound(loop.getLowerBound());
    auto upper = bound(loop.getUpperBound());
    auto step = bound(loop.getStep());
    if (!lower || !upper || !step || lower->smin() != lower->smax() ||
        step->smin() != step->smax() || step->smin().sle(0) ||
        upper->smax().sle(lower->smin()))
      return std::nullopt;
    // The last actual IV is aligned to the positive step. Widen the arithmetic
    // so deriving this bound does not assume an independently computed add nsw.
    APInt begin = lower->smin().sext(128), end = upper->smax().sext(128);
    APInt stride = step->smin().sext(128);
    APInt last = begin + (end - begin - 1).sdiv(stride) * stride;
    return ConstantIntRanges::fromSigned(lower->smin(), last.trunc(64));
  };
  return policy;
}

} // namespace

LocalSupplyRelations::LocalSupplyRelations(func::FuncOp function)
    : function(function), ranges(supplyPolicy(function,
          [this](Value value) { return constant(value); })) {}

dsa::SignedInterval LocalSupplyRelations::interval(Value value) {
  auto bounds = ranges.range(value);
  if (!bounds || bounds->smin().getBitWidth() > 64) return std::nullopt;
  return std::make_pair(bounds->smin().getSExtValue(), bounds->smax().getSExtValue());
}

std::optional<int64_t> LocalSupplyRelations::constant(Value value) {
  auto folded = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(expression(value), 0, symbols));
  return folded ? std::optional<int64_t>(folded.getValue()) : std::nullopt;
}

bool LocalSupplyRelations::equal(Value value, int64_t expected) {
  return constant(value) == expected;
}

AffineExpr LocalSupplyRelations::expression(Value value) {
  if (auto found = expressions.find(value); found != expressions.end()) return found->second;
  auto infer = [&]() -> AffineExpr {
    auto bounds = interval(value);
    if (bounds && bounds->first == bounds->second)
      return getAffineConstantExpr(bounds->first, function.getContext());
    Operation *op = value.getDefiningOp();
    if (!op || (!value.getType().isIndex() && !value.getType().isInteger(64)) ||
        op->getNumOperands() != 2)
      return getAffineSymbolExpr(symbols++, function.getContext());
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    auto operandBounds = [&](Value operand) {
      auto bounds = interval(operand);
      if (auto exact = constant(operand)) bounds = std::make_pair(*exact, *exact);
      return bounds;
    };
    auto left = operandBounds(lhs), right = operandBounds(rhs);
    if (left && right) {
      if (isa<arith::MinSIOp>(op)) {
        if (left->second <= right->first) return expression(lhs);
        if (right->second <= left->first) return expression(rhs);
      }
      if (isa<arith::MaxSIOp>(op)) {
        if (left->first >= right->second) return expression(lhs);
        if (right->first >= left->second) return expression(rhs);
      }
      std::optional<BinaryOperator> kind;
      if (isa<arith::AddIOp>(op)) kind = BinaryOperator::Add;
      if (isa<arith::SubIOp>(op)) kind = BinaryOperator::Subtract;
      if (isa<arith::MulIOp>(op)) kind = BinaryOperator::Multiply;
      if (kind && provesSignedNoWrap(*kind,
              ConstantIntRanges::fromSigned(APInt(64, left->first), APInt(64, left->second)),
              ConstantIntRanges::fromSigned(APInt(64, right->first), APInt(64, right->second)))) {
        if (*kind == BinaryOperator::Add) return expression(lhs) + expression(rhs);
        if (*kind == BinaryOperator::Subtract) return expression(lhs) - expression(rhs);
        return expression(lhs) * expression(rhs);
      }
    }
    return getAffineSymbolExpr(symbols++, function.getContext());
  };
  AffineExpr result = infer();
  expressions[value] = result;
  return result;
}

} // namespace intent::bangc

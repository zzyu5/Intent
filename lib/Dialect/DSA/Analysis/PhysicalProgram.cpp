#include "Intent/Dialect/DSA/Analysis/PhysicalProgram.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <functional>
#include <limits>

using namespace mlir;
namespace intent::dsa {

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

bool isSumOfIntegerProducts(Value value, ArrayRef<std::pair<Value, Value>> products) {
  auto addressType = [](Type type) { return type.isIndex() || type.isInteger(64); };
  if (!addressType(value.getType()) || llvm::any_of(products, [&](const auto &product) {
        return !addressType(product.first.getType()) || !addressType(product.second.getType());
      })) return false;
  MLIRContext *context = value.getContext();
  DenseMap<Value, AffineExpr> expressions;
  unsigned symbols = 0;
  std::function<AffineExpr(Value)> expression = [&](Value current) -> AffineExpr {
    if (auto known = expressions.find(current); known != expressions.end()) return known->second;
    auto infer = [&]() -> AffineExpr {
      APInt constant;
      if (matchPattern(current, m_ConstantInt(&constant)))
        return getAffineConstantExpr(constant.getSExtValue(), context);
      if (auto add = current.getDefiningOp<arith::AddIOp>()) return expression(add.getLhs()) + expression(add.getRhs());
      if (auto sub = current.getDefiningOp<arith::SubIOp>()) return expression(sub.getLhs()) - expression(sub.getRhs());
      if (auto mul = current.getDefiningOp<arith::MulIOp>()) return expression(mul.getLhs()) * expression(mul.getRhs());
      if (auto cast = current.getDefiningOp<arith::IndexCastOp>(); cast && addressType(cast.getIn().getType()))
        return expression(cast.getIn());
      return getAffineSymbolExpr(symbols++, context);
    };
    auto result = infer(); expressions[current] = result; return result;
  };
  AffineExpr actual = expression(value), expected = getAffineConstantExpr(0, context);
  for (auto [lhs, rhs] : products) expected = expected + expression(lhs) * expression(rhs);
  auto difference = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(actual - expected, 0, symbols));
  return difference && difference.getValue() == 0;
}

} // namespace intent::dsa

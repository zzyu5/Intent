#include "Intent/Dialect/GPU/Analysis/IndexPredicates.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

Value withoutProjection(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<BroadcastOp, SplatOp, ReshapeOp, TransposeOp>(operation))
      break;
    value = operation->getOperand(0);
  }
  return value;
}

std::optional<int64_t> integer(Value value) {
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(value));
  return constant && constant.getType().isIndex()
             ? std::optional<int64_t>(constant.getInt()) : std::nullopt;
}

class IndexRelations {
public:
  bool same(Value lhs, Value rhs) {
    if (samePhysicalScalarExpression(lhs, rhs))
      return true;
    auto left = queryLaunchExpression(lhs), right = queryLaunchExpression(rhs);
    return left && right && left == right;
  }

  bool nonnegative(Value value) {
    return bool(queryNonNegativeIndexUpperBound(value));
  }

  bool positive(Value value) {
    if (auto literal = integer(value))
      return *literal > 0;
    auto expression = queryLaunchExpression(value);
    auto owner = value.getDefiningOp();
    auto kernel = owner ? owner->getParentOfType<func::FuncOp>() : func::FuncOp();
    return expression && kernel && isKnownPositiveExtent(expression, kernel);
  }

  bool atMost(Value lhs, Value rhs) {
    if (!lhs.getType().isIndex() || !rhs.getType().isIndex())
      return false;
    if (same(lhs, rhs))
      return true;
    auto key = std::make_pair(lhs, rhs);
    auto inserted = comparisons.try_emplace(key, false);
    if (!inserted.second)
      return inserted.first->second;
    bool result = proveAtMost(lhs, rhs);
    comparisons[key] = result;
    return result;
  }

  // Return an equal, aligned bound. min(aligned, other) is aligned only when
  // the aligned operand is proven to be the selected minimum.
  Value alignedBound(Value value, Value step) {
    if (auto literal = integer(value); literal && *literal == 0)
      return value;
    if (same(value, step))
      return value;
    auto operation = value.getDefiningOp<BinaryOp>();
    if (!operation)
      return {};
    if (operation.getOperatorKind() == BinaryOperator::Multiply) {
      for (auto [quotient, factor] :
           {std::pair{operation.getLhs(), operation.getRhs()},
            std::pair{operation.getRhs(), operation.getLhs()}}) {
        auto divide = quotient.getDefiningOp<BinaryOp>();
        if (divide && divide.getOperatorKind() == BinaryOperator::FloorDivide &&
            same(factor, step) && same(divide.getRhs(), step) &&
            positive(step) && nonnegative(divide.getLhs()))
          return value;
      }
    }
    if (operation.getOperatorKind() == BinaryOperator::Minimum)
      for (auto [candidate, other] :
           {std::pair{operation.getLhs(), operation.getRhs()},
            std::pair{operation.getRhs(), operation.getLhs()}})
        if (Value aligned = alignedBound(candidate, step);
            aligned && atMost(candidate, other))
          return aligned;
    return {};
  }

private:
  bool proveAtMost(Value lhs, Value rhs) {
    auto left = integer(lhs), right = integer(rhs);
    if (left && right)
      return *left <= *right;
    if (left && *left == 0 && nonnegative(rhs))
      return true;
    if (auto binary = lhs.getDefiningOp<BinaryOp>()) {
      if (binary.getOperatorKind() == BinaryOperator::Minimum)
        return atMost(binary.getLhs(), rhs) || atMost(binary.getRhs(), rhs);
      if (binary.getOperatorKind() == BinaryOperator::Maximum)
        return atMost(binary.getLhs(), rhs) && atMost(binary.getRhs(), rhs);
      if (binary.getOperatorKind() == BinaryOperator::Multiply)
        for (auto [quotient, factor] :
             {std::pair{binary.getLhs(), binary.getRhs()},
              std::pair{binary.getRhs(), binary.getLhs()}}) {
          auto divide = quotient.getDefiningOp<BinaryOp>();
          if (divide && divide.getOperatorKind() == BinaryOperator::FloorDivide &&
              same(divide.getRhs(), factor) && positive(factor) &&
              nonnegative(divide.getLhs()) && atMost(divide.getLhs(), rhs))
            return true;
        }
    }
    if (auto binary = rhs.getDefiningOp<BinaryOp>()) {
      if (binary.getOperatorKind() == BinaryOperator::Maximum)
        return atMost(lhs, binary.getLhs()) || atMost(lhs, binary.getRhs());
      if (binary.getOperatorKind() == BinaryOperator::Minimum)
        return atMost(lhs, binary.getLhs()) && atMost(lhs, binary.getRhs());
    }
    return false;
  }

  DenseMap<std::pair<Value, Value>, bool> comparisons;
};

Value completeTileLimit(MakeRangeOp range, IndexRelations &relations) {
  auto type = cast<FragmentType>(range.getResult().getType());
  if (!type.getElementType().isIndex() || type.getShape().size() != 1 ||
      integer(range.getStep()) != 1)
    return {};
  for (Operation *owner = range->getParentOp(); owner; owner = owner->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop || !relations.same(range.getStart(), loop.getInductionVar()) ||
        integer(loop.getLowerBound()) != 0 || !relations.positive(loop.getStep()))
      continue;
    auto width = queryLaunchExpression(loop.getStep());
    if (!width || type.getShape()[0] != width ||
        !relations.same(range.getExtent(), loop.getStep()))
      continue;
    Value limit = relations.alignedBound(loop.getUpperBound(), loop.getStep());
    // For 0 <= iv < limit and an aligned limit, iv + [0, step) stays below
    // limit without signed overflow. No fact about a partial final tile is used.
    if (limit && relations.nonnegative(limit))
      return limit;
  }
  return {};
}

// A bound valid inside an executing loop need not be safe to evaluate during
// specialization of an empty loop. Check the entire shape expression, not just
// its leaves, before materializing it as an additional constexpr predicate.
class ShapeExpressionBounds {
  using Interval = std::pair<int64_t, int64_t>;
public:
  explicit ShapeExpressionBounds(func::FuncOp kernel) : kernel(kernel) {}
  bool safe(PhysicalExprAttr expression) { return bool(bounds(expression)); }
private:
  std::optional<Interval> bounds(PhysicalExprAttr expression) {
    if (auto found = known.find(expression); found != known.end())
      return found->second;
    auto result = infer(expression);
    known[expression] = result;
    return result;
  }
  std::optional<Interval> infer(PhysicalExprAttr expression) {
    auto kind = static_cast<PhysicalExprKind>(expression.getKind());
    if (kind == PhysicalExprKind::Constant)
      return Interval{expression.getValue(), expression.getValue()};
    if (kind == PhysicalExprKind::Dimension)
      return Interval{0, std::numeric_limits<int64_t>::max()};
    if (kind == PhysicalExprKind::Parameter) {
      auto parameter = queryParameterBySymbol(kernel, expression.getSymbol());
      if (failed(parameter)) return std::nullopt;
      if ((*parameter)->hasAttr(coverageDimensionAttr) ||
          parameter->getParameter().getCategory() ==
              static_cast<uint32_t>(ParameterCategory::Coverage))
        return Interval{0, std::numeric_limits<int64_t>::max()};
      auto domain = parameter->getParameter().getCandidates().asArrayRef();
      if (domain.empty()) return std::nullopt;
      return Interval{*llvm::min_element(domain), *llvm::max_element(domain)};
    }
    if (kind == PhysicalExprKind::ScalarABI)
      return std::nullopt;
    SmallVector<Interval> operands;
    for (Attribute operand : expression.getOperands()) {
      auto value = bounds(cast<PhysicalExprAttr>(operand));
      if (!value) return std::nullopt;
      operands.push_back(*value);
    }
    if (kind == PhysicalExprKind::Select && operands.size() == 3)
      return Interval{std::min(operands[1].first, operands[2].first),
                      std::max(operands[1].second, operands[2].second)};
    if (operands.size() != 2 &&
        !(kind == PhysicalExprKind::NextPowerOfTwo && operands.size() == 1))
      return std::nullopt;
    if ((kind == PhysicalExprKind::FloorDiv || kind == PhysicalExprKind::CeilDiv) &&
        operands[1].first <= 0)
      return std::nullopt;
    if (kind != PhysicalExprKind::Add && kind != PhysicalExprKind::Subtract &&
        kind != PhysicalExprKind::Multiply && kind != PhysicalExprKind::Minimum &&
        kind != PhysicalExprKind::Maximum && kind != PhysicalExprKind::FloorDiv &&
        kind != PhysicalExprKind::CeilDiv && kind != PhysicalExprKind::NextPowerOfTwo)
      return std::nullopt;
    auto context = expression.getContext();
    auto constant = [&](int64_t value) {
      return PhysicalExprAttr::get(context,
          static_cast<uint32_t>(PhysicalExprKind::Constant), value,
          StringAttr::get(context, ""), ArrayAttr::get(context, {}));
    };
    Interval result{std::numeric_limits<int64_t>::max(),
                    std::numeric_limits<int64_t>::min()};
    for (int64_t lhs : {operands[0].first, operands[0].second}) {
      auto rhsBounds = operands.size() == 2 ? operands[1] : Interval{0, 0};
      for (int64_t rhs : {rhsBounds.first, rhsBounds.second}) {
        SmallVector<Attribute> values{constant(lhs)};
        if (operands.size() == 2) values.push_back(constant(rhs));
        auto point = PhysicalExprAttr::get(context, expression.getKind(), 0,
            StringAttr::get(context, ""), ArrayAttr::get(context, values));
        auto value = constantPhysicalExpression(point);
        if (!value) return std::nullopt;
        result.first = std::min(result.first, *value);
        result.second = std::max(result.second, *value);
      }
    }
    return result;
  }
  func::FuncOp kernel;
  DenseMap<PhysicalExprAttr, std::optional<Interval>> known;
};

} // namespace

Value queryCompleteTileLimit(MakeRangeOp range) {
  IndexRelations relations;
  return completeTileLimit(range, relations);
}

std::optional<bool> proveRangeComparison(CompareOp comparison) {
  if (comparison.getLhs().getType().isIndex() &&
      comparison.getPredicate() == ComparePredicate::Ge &&
      integer(comparison.getRhs()) == 0 &&
      queryNonNegativeIndexUpperBound(comparison.getLhs()))
    return true;
  Value source = withoutProjection(comparison.getLhs());
  Value bound = withoutProjection(comparison.getRhs());
  auto range = source.getDefiningOp<MakeRangeOp>();
  if (!range || !bound.getType().isIndex())
    return std::nullopt;
  IndexRelations relations;
  Value limit = queryCompleteTileLimit(range);
  if (!limit)
    return std::nullopt;
  if (comparison.getPredicate() == ComparePredicate::Lt &&
      relations.atMost(limit, bound))
    return true;
  if (comparison.getPredicate() == ComparePredicate::Ge) {
    if (relations.atMost(limit, bound))
      return false;
    if (integer(bound) == 0)
      return true;
  }
  return std::nullopt;
}

std::optional<IndexComparisonBound> queryIndexComparisonBound(CompareOp comparison) {
  if (comparison.getPredicate() != ComparePredicate::Lt ||
      !comparison.getLhs().getType().isIndex() ||
      !comparison.getRhs().getType().isIndex() ||
      queryLaunchExpression(comparison.getLhs()))
    return std::nullopt;
  auto upper = queryNonNegativeIndexUpperBound(comparison.getLhs());
  auto limit = queryLaunchExpression(comparison.getRhs());
  if (!upper || !limit || !isShapeBound(upper) || !isShapeBound(limit))
    return std::nullopt;
  ShapeExpressionBounds bounds(comparison->getParentOfType<func::FuncOp>());
  if (!bounds.safe(upper) || !bounds.safe(limit))
    return std::nullopt;
  for (Operation *user : comparison.getResult().getUsers()) {
    auto logicalOr = dyn_cast<BinaryOp>(user);
    if (!logicalOr || logicalOr.getOperatorKind() != BinaryOperator::LogicalOr)
      continue;
    Value other = logicalOr.getLhs() == comparison.getResult()
                      ? logicalOr.getRhs() : logicalOr.getLhs();
    auto proof = other.getDefiningOp<CompareOp>();
    if (!proof || proof.getPredicate() != ComparePredicate::Lt) continue;
    auto left = queryLaunchExpression(proof.getLhs());
    auto right = queryLaunchExpression(proof.getRhs());
    if (left && right && left == upper && right == limit)
      return std::nullopt;
  }
  return IndexComparisonBound{upper, limit};
}

} // namespace intent::gpu

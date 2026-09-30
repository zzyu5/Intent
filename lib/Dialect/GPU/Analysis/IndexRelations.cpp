#include "Intent/Dialect/GPU/Analysis/IndexRelations.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isCoverageParameter(ParameterOp parameter) {
  return parameter->hasAttr(coverageDimensionAttr) ||
         parameter.getParameter().getCategory() ==
             static_cast<uint32_t>(ParameterCategory::Coverage);
}

bool hasLateBoundDomain(ParameterOp parameter) {
  // Provider configuration formation binds resident capacity after native
  // access formation. Its placeholder candidates prove neither a constant nor
  // common divisibility of the eventual SM/CTA/occupancy-dependent domain.
  return parameter.getParameter().getRole() ==
         static_cast<uint32_t>(ParameterRole::ResidentWorkers);
}

ParameterOp parameterFor(Value value) {
  if (auto parameter = value.getDefiningOp<ParameterOp>())
    return parameter;
  auto expression = value.getDefiningOp<PhysicalExprOp>();
  if (!expression || expression.getExpression().getKind() !=
                         static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return {};
  auto parameter = queryParameterBySymbol(
      expression->getParentOfType<func::FuncOp>(),
      expression.getExpression().getSymbol());
  return succeeded(parameter) ? *parameter : ParameterOp();
}

std::optional<int64_t> evaluateSingletonExpression(PhysicalExprAttr expression,
                                                  func::FuncOp kernel) {
  return evaluatePhysicalExpression(expression, [&](PhysicalExprAttr leaf)
      -> std::optional<int64_t> {
    if (!kernel || leaf.getKind() !=
                       static_cast<uint32_t>(PhysicalExprKind::Parameter))
      return std::nullopt;
    auto parameter = queryParameterBySymbol(kernel, leaf.getSymbol());
    if (failed(parameter) || isCoverageParameter(*parameter) ||
        hasLateBoundDomain(*parameter))
      return std::nullopt;
    auto candidates = parameter->getParameter().getCandidates().asArrayRef();
    return candidates.size() == 1 ? std::optional<int64_t>(candidates.front())
                                  : std::nullopt;
  });
}

} // namespace

std::optional<int64_t> IndexRelations::constant(PhysicalExprAttr expression,
                                                func::FuncOp kernel) const {
  return evaluateSingletonExpression(expression, kernel);
}

std::optional<int64_t> IndexRelations::constant(Value value) const {
  if (!value || !value.getType().isIntOrIndex())
    return std::nullopt;
  if (PhysicalExprAttr expression = queryLaunchExpression(value)) {
    Operation *owner = value.getDefiningOp();
    if (!owner)
      owner = cast<BlockArgument>(value).getOwner()->getParentOp();
    auto kernel = dyn_cast<func::FuncOp>(owner);
    if (!kernel)
      kernel = owner->getParentOfType<func::FuncOp>();
    // Do not retry overflowing launch arithmetic with modular constant folding.
    return evaluateSingletonExpression(expression, kernel);
  }
  auto integer = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(value));
  if (!integer || integer.getValue().getBitWidth() > 64)
    return std::nullopt;
  auto type = dyn_cast<IntegerType>(value.getType());
  if (type && (type.isUnsigned() || type.getWidth() == 1))
    return integer.getValue().getActiveBits() <= 63
               ? std::optional<int64_t>(integer.getValue().getZExtValue())
               : std::nullopt;
  return integer.getInt();
}

bool IndexRelations::same(Value lhs, Value rhs) const {
  if (samePhysicalScalarExpression(lhs, rhs))
    return true;
  auto left = queryLaunchExpression(lhs), right = queryLaunchExpression(rhs);
  return left && right && left == right;
}

bool IndexRelations::nonnegative(Value value) const {
  return nonnegative(value, 0);
}

bool IndexRelations::nonnegative(Value value, unsigned depth) const {
  if (!value || !value.getType().isIndex() || depth >= 32)
    return false;
  if (auto literal = constant(value))
    return *literal >= 0;
  if (queryNonNegativeIndexUpperBound(value))
    return true;
  if (value.getDefiningOp<ProgramIdOp>())
    return true;
  if (auto coordinate = value.getDefiningOp<WorksetCoordinateOp>())
    return nonnegative(coordinate.getCoordinate(), depth + 1);
  if (auto physical = value.getDefiningOp<PhysicalExprOp>())
    return isKnownPositiveExtent(physical.getExpression(),
                                  physical->getParentOfType<func::FuncOp>());
  if (auto binary = value.getDefiningOp<BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Maximum)
      return nonnegative(binary.getLhs(), depth + 1) ||
             nonnegative(binary.getRhs(), depth + 1);
    if (binary.getOperatorKind() == BinaryOperator::Minimum)
      return nonnegative(binary.getLhs(), depth + 1) &&
             nonnegative(binary.getRhs(), depth + 1);
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide ||
        binary.getOperatorKind() == BinaryOperator::Remainder)
      return nonnegative(binary.getLhs(), depth + 1) && positive(binary.getRhs());
  }
  return false;
}

bool IndexRelations::positive(Value value) const {
  if (auto literal = constant(value))
    return *literal > 0;
  if (auto parameter = parameterFor(value)) {
    // Coverage capacity is selected from these candidates at launch. Their
    // common sign/alignment properties apply, but they are not the logical
    // extent and do not make a coverage parameter a compile-time singleton.
    auto candidates = parameter.getParameter().getCandidates().asArrayRef();
    return !candidates.empty() && llvm::all_of(candidates, [](int64_t candidate) {
      return candidate > 0;
    });
  }
  auto expression = queryLaunchExpression(value);
  auto owner = value.getDefiningOp();
  auto kernel = owner ? owner->getParentOfType<func::FuncOp>() : func::FuncOp();
  return expression && kernel && isKnownPositiveExtent(expression, kernel);
}

bool IndexRelations::atMost(Value lhs, Value rhs) const {
  return atMost(lhs, rhs, 0);
}

bool IndexRelations::atMost(Value lhs, Value rhs, unsigned depth) const {
  if (!lhs || !rhs || !lhs.getType().isIndex() || !rhs.getType().isIndex() ||
      depth >= 32)
    return false;
  if (same(lhs, rhs))
    return true;
  auto left = constant(lhs), right = constant(rhs);
  if (left && right)
    return *left <= *right;
  if (left && *left == 0 && nonnegative(rhs))
    return true;
  auto compare = [&](Value a, Value b) { return atMost(a, b, depth + 1); };
  if (auto binary = lhs.getDefiningOp<BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Minimum)
      return compare(binary.getLhs(), rhs) || compare(binary.getRhs(), rhs);
    if (binary.getOperatorKind() == BinaryOperator::Maximum)
      return compare(binary.getLhs(), rhs) && compare(binary.getRhs(), rhs);
    if (binary.getOperatorKind() == BinaryOperator::Subtract &&
        nonnegative(binary.getLhs()) && nonnegative(binary.getRhs()))
      return compare(binary.getLhs(), rhs);
    if (binary.getOperatorKind() == BinaryOperator::Multiply)
      for (auto [quotient, factor] :
           {std::pair{binary.getLhs(), binary.getRhs()},
            std::pair{binary.getRhs(), binary.getLhs()}}) {
        auto divide = quotient.getDefiningOp<BinaryOp>();
        if (divide && divide.getOperatorKind() == BinaryOperator::FloorDivide &&
            same(divide.getRhs(), factor) && positive(factor) &&
            nonnegative(divide.getLhs()) && compare(divide.getLhs(), rhs))
          return true;
      }
  }
  if (auto binary = rhs.getDefiningOp<BinaryOp>()) {
    if (binary.getOperatorKind() == BinaryOperator::Maximum)
      return compare(lhs, binary.getLhs()) || compare(lhs, binary.getRhs());
    if (binary.getOperatorKind() == BinaryOperator::Minimum)
      return compare(lhs, binary.getLhs()) && compare(lhs, binary.getRhs());
  }
  return false;
}

bool IndexRelations::powerOfTwo(Value value) const {
  if (auto parameter = parameterFor(value)) {
    if (hasLateBoundDomain(parameter))
      return false;
    auto candidates = parameter.getParameter().getCandidates().asArrayRef();
    return !candidates.empty() && llvm::all_of(candidates, [](int64_t candidate) {
      return candidate > 0 && llvm::isPowerOf2_64(candidate);
    });
  }
  auto literal = constant(value);
  return literal && *literal > 0 && llvm::isPowerOf2_64(*literal);
}

bool IndexRelations::multipleOf(
    Value value, int64_t divisor,
    llvm::function_ref<bool(ParameterOp)> alignedParameter) const {
  if (!value || !value.getType().isIndex() || divisor <= 0)
    return false;
  // Powers of two divide the index modulus. Their divisibility survives
  // fixed-width add/sub/mul; other divisors require a concrete constant.
  std::function<bool(Value, unsigned)> prove = [&](Value current, unsigned depth) {
    if (!current || !current.getType().isIndex() || depth >= 32)
      return false;
    if (divisor == 1)
      return true;
    if (auto parameter = parameterFor(current)) {
      if (alignedParameter && alignedParameter(parameter))
        return true;
      if (hasLateBoundDomain(parameter))
        return false;
      auto candidates = parameter.getParameter().getCandidates().asArrayRef();
      return !candidates.empty() && llvm::all_of(candidates, [&](int64_t candidate) {
        return candidate % divisor == 0;
      });
    }
    if (auto literal = constant(current))
      return *literal % divisor == 0;
    if (!llvm::isPowerOf2_64(divisor))
      return false;
    auto aligned = [&](Value operand) { return prove(operand, depth + 1); };
    if (auto coordinate = current.getDefiningOp<WorksetCoordinateOp>())
      return aligned(coordinate.getCoordinate());
    if (auto argument = dyn_cast<BlockArgument>(current)) {
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      return loop && argument == loop.getInductionVar() &&
             aligned(loop.getLowerBound()) && aligned(loop.getStep());
    }
    if (auto select = current.getDefiningOp<SelectOp>())
      return aligned(select.getTrueValue()) && aligned(select.getFalseValue());
    if (auto physical = current.getDefiningOp<PhysicalExprOp>()) {
      std::function<bool(PhysicalExprAttr)> expressionAligned =
          [&](PhysicalExprAttr expression) {
        auto kind = static_cast<PhysicalExprKind>(expression.getKind());
        if (kind == PhysicalExprKind::Constant)
          return expression.getValue() % divisor == 0;
        if (kind == PhysicalExprKind::Parameter) {
          auto parameter = queryParameterBySymbol(
              physical->getParentOfType<func::FuncOp>(), expression.getSymbol());
          return succeeded(parameter) && aligned(parameter->getResult());
        }
        if (expression.getOperands().size() != 2)
          return false;
        bool lhs = expressionAligned(cast<PhysicalExprAttr>(expression.getOperands()[0]));
        bool rhs = expressionAligned(cast<PhysicalExprAttr>(expression.getOperands()[1]));
        if (kind == PhysicalExprKind::Multiply)
          return lhs || rhs;
        return (kind == PhysicalExprKind::Add || kind == PhysicalExprKind::Subtract ||
                kind == PhysicalExprKind::Minimum || kind == PhysicalExprKind::Maximum) &&
               lhs && rhs;
      };
      return expressionAligned(physical.getExpression());
    }
    if (auto binary = current.getDefiningOp<BinaryOp>()) {
      auto kind = binary.getOperatorKind();
      if (kind == BinaryOperator::Multiply)
        return aligned(binary.getLhs()) || aligned(binary.getRhs());
      if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract ||
          kind == BinaryOperator::Minimum || kind == BinaryOperator::Maximum)
        return aligned(binary.getLhs()) && aligned(binary.getRhs());
    }
    return false;
  };
  return prove(value, 0);
}

bool IndexRelations::multipleOf(Value value, Value divisor) const {
  if (!value || !divisor || !value.getType().isIndex() ||
      !divisor.getType().isIndex() || !positive(divisor))
    return false;
  if (same(value, divisor) || constant(value) == 0)
    return true;
  if (auto literal = constant(divisor))
    return multipleOf(value, *literal);
  auto parameter = parameterFor(divisor);
  if (!parameter || hasLateBoundDomain(parameter))
    return false;
  auto candidates = parameter.getParameter().getCandidates().asArrayRef();
  if (candidates.empty())
    return false;
  std::function<bool(Value, unsigned)> prove = [&](Value current, unsigned depth) {
    if (!current || !current.getType().isIndex() || depth >= 32)
      return false;
    if (same(current, divisor) || constant(current) == 0)
      return true;
    if (llvm::all_of(candidates, [&](int64_t candidate) {
          return multipleOf(current, candidate);
        }))
      return true;
    if (!powerOfTwo(divisor))
      return false;
    auto aligned = [&](Value operand) { return prove(operand, depth + 1); };
    if (auto coordinate = current.getDefiningOp<WorksetCoordinateOp>())
      return aligned(coordinate.getCoordinate());
    if (auto binary = current.getDefiningOp<BinaryOp>()) {
      auto kind = binary.getOperatorKind();
      if (kind == BinaryOperator::Multiply)
        return aligned(binary.getLhs()) || aligned(binary.getRhs());
      if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract ||
          kind == BinaryOperator::Minimum || kind == BinaryOperator::Maximum)
        return aligned(binary.getLhs()) && aligned(binary.getRhs());
    }
    return false;
  };
  return prove(value, 0);
}

Value IndexRelations::alignedBound(Value value, Value step) const {
  if (constant(value) == 0 || same(value, step))
    return value;
  auto operation = value.getDefiningOp<BinaryOp>();
  if (!operation)
    return {};
  if (operation.getOperatorKind() == BinaryOperator::Multiply)
    for (auto [quotient, factor] :
         {std::pair{operation.getLhs(), operation.getRhs()},
          std::pair{operation.getRhs(), operation.getLhs()}}) {
      auto divide = quotient.getDefiningOp<BinaryOp>();
      if (divide && divide.getOperatorKind() == BinaryOperator::FloorDivide &&
          same(factor, step) && same(divide.getRhs(), step) && positive(step) &&
          nonnegative(divide.getLhs()))
        return value;
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

} // namespace intent::gpu

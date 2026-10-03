#include "Intent/Dialect/GPU/Analysis/PhysicalExpressionBounds.h"
#include "Intent/Analysis/IntegerRanges.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

using Range = ConstantIntRanges;

Range signedRange(int64_t lower, int64_t upper) {
  return Range::fromSigned(APInt(64, lower, true), APInt(64, upper, true));
}

class PhysicalExpressionBounds {
public:
  explicit PhysicalExpressionBounds(func::FuncOp kernel) : kernel(kernel) {}

  std::optional<Range> query(PhysicalExprAttr expression) {
    if (!expression)
      return std::nullopt;
    if (auto found = known.find(expression); found != known.end())
      return found->second;
    auto result = infer(expression);
    known.try_emplace(expression, result);
    return result;
  }

private:
  std::optional<Range> parameterRange(PhysicalExprAttr expression) {
    if (!kernel)
      return std::nullopt;
    auto parameter = lookupParameter(kernel, expression.getParameterReference());
    if (!parameter)
      return std::nullopt;
    // Native configuration formation may replace these placeholder candidates.
    // Only the positive capacity contract survives that later binding.
    if (parameter.getRole() == ParameterRole::ResidentWorkers)
      return signedRange(1, std::numeric_limits<int64_t>::max());
    auto candidates = parameter.getCandidates().asArrayRef();
    if (candidates.empty())
      return std::nullopt;
    int64_t lower = *llvm::min_element(candidates);
    int64_t upper = *llvm::max_element(candidates);
    // Deferred coverage also selects an element of its declared finite domain.
    // A runtime extent that no candidate covers is rejected by the binder.
    if (auto configurations = kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
        configurations && !configurations.getRows().empty()) {
      int64_t selectedLower = std::numeric_limits<int64_t>::max();
      int64_t selectedUpper = std::numeric_limits<int64_t>::min();
      bool complete = true;
      for (Attribute attribute : configurations.getRows()) {
        auto row = dyn_cast<DictionaryAttr>(attribute);
        auto selected = row ? row.getAs<IntegerAttr>(parameter.getName()) : IntegerAttr();
        if (!selected || !llvm::is_contained(candidates, selected.getInt())) {
          complete = false;
          break;
        }
        selectedLower = std::min(selectedLower, selected.getInt());
        selectedUpper = std::max(selectedUpper, selected.getInt());
      }
      if (complete) {
        lower = selectedLower;
        upper = selectedUpper;
      }
    }
    return signedRange(lower, upper);
  }

  std::optional<Range> infer(PhysicalExprAttr expression) {
    auto kind = expression.getKind();
    if (kind == PhysicalExprKind::Constant)
      return Range::constant(APInt(64, expression.getValue(), true));
    if (kind == PhysicalExprKind::Dimension)
      return signedRange(0, std::numeric_limits<int64_t>::max());
    if (kind == PhysicalExprKind::ScalarABI)
      return Range::maxRange(64);
    if (kind == PhysicalExprKind::Parameter)
      return parameterRange(expression);
    SmallVector<Range> operands;
    for (Attribute attribute : expression.getOperands()) {
      auto range = query(cast<PhysicalExprAttr>(attribute));
      if (!range)
        return std::nullopt;
      operands.push_back(*range);
    }
    if (kind == PhysicalExprKind::Select && operands.size() == 3) {
      if (auto constant = operands[0].getConstantValue())
        return constant->isZero() ? operands[2] : operands[1];
      return operands[1].rangeUnion(operands[2]);
    }
    if (kind == PhysicalExprKind::NextPowerOfTwo && operands.size() == 1) {
      int64_t lower = operands[0].smin().getSExtValue();
      int64_t upper = operands[0].smax().getSExtValue();
      if (upper > (int64_t{1} << 62))
        return std::nullopt;
      return signedRange(llvm::PowerOf2Ceil(std::max<int64_t>(1, lower)),
                         llvm::PowerOf2Ceil(std::max<int64_t>(1, upper)));
    }
    if (operands.size() != 2)
      return std::nullopt;
    auto binary = [&](BinaryOperator operation) -> std::optional<Range> {
      if (!provesSignedNoWrap(operation, operands[0], operands[1]))
        return std::nullopt;
      return inferIntegerBinary(operation, IndexType::get(expression.getContext()),
                                operands[0], operands[1]);
    };
    switch (kind) {
    case PhysicalExprKind::Add:
      return binary(BinaryOperator::Add);
    case PhysicalExprKind::Subtract:
      return binary(BinaryOperator::Subtract);
    case PhysicalExprKind::Multiply:
      return binary(BinaryOperator::Multiply);
    case PhysicalExprKind::Minimum:
      return intrange::inferMinS(operands);
    case PhysicalExprKind::Maximum:
      return intrange::inferMaxS(operands);
    case PhysicalExprKind::FloorDiv:
    case PhysicalExprKind::CeilDiv:
      if ((operands[1].smin().isNonPositive() && operands[1].smax().isNonNegative()) ||
          (operands[0].smin().isMinSignedValue() &&
           operands[1].smin().sle(APInt(64, -1, true)) &&
           operands[1].smax().sge(APInt(64, -1, true))))
        return std::nullopt;
      return kind == PhysicalExprKind::FloorDiv ? intrange::inferFloorDivS(operands)
                                                : intrange::inferCeilDivS(operands);
    default:
      return std::nullopt;
    }
  }

  func::FuncOp kernel;
  DenseMap<PhysicalExprAttr, std::optional<Range>> known;
};

} // namespace

std::optional<ConstantIntRanges>
queryPhysicalExpressionRange(PhysicalExprAttr expression, func::FuncOp kernel) {
  return PhysicalExpressionBounds(kernel).query(expression);
}

} // namespace intent::gpu

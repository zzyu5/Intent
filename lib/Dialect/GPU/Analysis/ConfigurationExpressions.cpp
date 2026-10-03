#include "Intent/Dialect/GPU/Analysis/ConfigurationExpressions.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/PhysicalExpressions.h"
#include "mlir/IR/AttrTypeSubElements.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool expressionAtMost(PhysicalExprAttr lhs, PhysicalExprAttr rhs,
                      unsigned depth = 0) {
  if (!lhs || !rhs || depth >= 32) return false;
  if (lhs == rhs) return true;
  auto left = constantPhysicalExpression(lhs);
  auto right = constantPhysicalExpression(rhs);
  if (left && right) return *left <= *right;
  if (lhs.getOperands().size() == 2) {
    auto first = cast<PhysicalExprAttr>(lhs.getOperands()[0]);
    auto second = cast<PhysicalExprAttr>(lhs.getOperands()[1]);
    if (lhs.getKind() == PhysicalExprKind::Minimum)
      return expressionAtMost(first, rhs, depth + 1) ||
             expressionAtMost(second, rhs, depth + 1);
    if (lhs.getKind() == PhysicalExprKind::Maximum)
      return expressionAtMost(first, rhs, depth + 1) &&
             expressionAtMost(second, rhs, depth + 1);
  }
  if (rhs.getOperands().size() == 2) {
    auto first = cast<PhysicalExprAttr>(rhs.getOperands()[0]);
    auto second = cast<PhysicalExprAttr>(rhs.getOperands()[1]);
    if (rhs.getKind() == PhysicalExprKind::Maximum)
      return expressionAtMost(lhs, first, depth + 1) ||
             expressionAtMost(lhs, second, depth + 1);
    if (rhs.getKind() == PhysicalExprKind::Minimum)
      return expressionAtMost(lhs, first, depth + 1) &&
             expressionAtMost(lhs, second, depth + 1);
  }
  return false;
}

} // namespace

FailureOr<PhysicalExprAttr>
instantiateConfigurationExpression(PhysicalExprAttr expression,
                                   DictionaryAttr bindings) {
  if (!expression || !bindings) return failure();
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](PhysicalExprAttr current) -> std::optional<Attribute> {
        bool symbolic = false, invalid = false;
        auto value = evaluatePhysicalExpression(
            current, [&](PhysicalExprAttr leaf) -> std::optional<int64_t> {
              if (leaf.getKind() == PhysicalExprKind::Parameter) {
                if (Attribute binding =
                        bindings.get(leaf.getParameterReference().getName())) {
                  auto integer = dyn_cast<IntegerAttr>(binding);
                  if (!integer || !integer.getType().isSignlessInteger(64)) {
                    invalid = true;
                    return std::nullopt;
                  }
                  return integer.getInt();
                }
              }
              symbolic = true;
              return std::nullopt;
            });
        if (value)
          return PhysicalExprAttr::get(
              current.getContext(), PhysicalExprKind::Constant, *value,
              StringAttr::get(current.getContext(), ""),
              ArrayAttr::get(current.getContext(), {}));
        if (symbolic && !invalid) return std::nullopt;
        return Attribute();
      });
  auto result = dyn_cast_or_null<PhysicalExprAttr>(replacer.replace(expression));
  if (!result) return failure();
  return result;
}

bool configurationExpressionAtMost(func::FuncOp kernel, PhysicalExprAttr lhs,
                                   PhysicalExprAttr rhs) {
  if (!kernel || !lhs || !rhs) return false;
  auto empty = DictionaryAttr::get(kernel.getContext());
  auto symbolicLeft = instantiateConfigurationExpression(lhs, empty);
  auto symbolicRight = instantiateConfigurationExpression(rhs, empty);
  if (failed(symbolicLeft) || failed(symbolicRight)) return false;
  if (expressionAtMost(*symbolicLeft, *symbolicRight)) return true;
  auto configurations =
      kernel->getAttrOfType<ConfigurationSetAttr>(configurationsAttr);
  if (!configurations || configurations.getRows().empty() ||
      !getParameterDeclarations(kernel))
    return false;
  auto space = ParameterSpace::read(kernel);
  if (failed(space)) return false;
  auto rows = space->configurations(configurations.getStage());
  if (failed(rows)) return false;
  auto resident = configurations.getStage() == ConfigurationStage::Shared
                      ? space->find(ParameterRole::ResidentWorkers)
                      : ParameterAttr();
  for (DictionaryAttr row : *rows) {
    if (resident) {
      // Shared analysis also runs before provider preparation. Its initial
      // resident count may still grow, unlike final Complete bindings.
      NamedAttrList stable(row);
      stable.erase(resident.getName());
      row = stable.getDictionary(kernel.getContext());
    }
    auto left = instantiateConfigurationExpression(lhs, row);
    auto right = instantiateConfigurationExpression(rhs, row);
    if (failed(left) || failed(right) || !expressionAtMost(*left, *right))
      return false;
  }
  return true;
}

} // namespace intent::gpu

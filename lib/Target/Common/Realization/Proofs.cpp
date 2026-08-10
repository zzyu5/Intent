#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"

#include "llvm/ADT/DenseMap.h"

#include <cmath>

using namespace mlir;

namespace intent::target {
namespace {

enum class PaddedValue {
  arbitrary,
  zero,
  negativeInfinity,
};

Operation *rootViewLoad(Value value) {
  Operation *definition = value.getDefiningOp();
  while (definition && definition->getName().getStringRef() == "intent.cast" &&
         definition->getNumOperands() == 1) {
    value = definition->getOperand(0);
    definition = value.getDefiningOp();
  }
  if (!definition ||
      definition->getName().getStringRef() != "intent.view_load")
    return nullptr;
  return definition;
}

bool haveAlignedLoadMasks(Value lhs, Value rhs) {
  Operation *lhsLoad = rootViewLoad(lhs);
  Operation *rhsLoad = rootViewLoad(rhs);
  if (!lhsLoad || !rhsLoad ||
      lhsLoad->getAttr("intent.index") != rhsLoad->getAttr("intent.index") ||
      lhsLoad->getAttr("intent.result_shapes") !=
          rhsLoad->getAttr("intent.result_shapes") ||
      lhsLoad->getNumOperands() != rhsLoad->getNumOperands())
    return false;
  for (unsigned index = 1; index < lhsLoad->getNumOperands(); ++index)
    if (lhsLoad->getOperand(index) != rhsLoad->getOperand(index))
      return false;
  return true;
}

bool provePaddedUses(Value value, PaddedValue padded,
                     llvm::DenseMap<Value, PaddedValue> &visited,
                     Operation *ignoredUser = nullptr) {
  auto found = visited.find(value);
  if (found != visited.end())
    return found->second == padded;
  visited[value] = padded;

  for (Operation *user : value.getUsers()) {
    if (user == ignoredUser)
      continue;
    StringRef name = user->getName().getStringRef();
    if (name == "intent.view_store")
      continue;
    if (name == "intent.reduce") {
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      if (!combine ||
          (combine.getValue() == "add" && padded != PaddedValue::zero) ||
          (combine.getValue() == "maximum" &&
           padded != PaddedValue::negativeInfinity) ||
          (combine.getValue() != "add" && combine.getValue() != "maximum"))
        return false;
      continue;
    }
    if (name == "intent.contract" && padded == PaddedValue::zero) {
      auto multiply = user->getAttrOfType<StringAttr>("intent.multiply");
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      if (multiply && multiply.getValue() == "multiply" && combine &&
          combine.getValue() == "add")
        continue;
      return false;
    }

    PaddedValue result = PaddedValue::arbitrary;
    if (name == "intent.cast" || name == "intent.broadcast") {
      result = padded;
    } else if (name == "intent.unary") {
      auto logical = user->getAttrOfType<StringAttr>("intent.operator");
      if (logical &&
          (logical.getValue() == "exp" || logical.getValue() == "exp2") &&
          padded == PaddedValue::negativeInfinity)
        result = PaddedValue::zero;
      else if (logical && logical.getValue() == "negate" &&
               padded == PaddedValue::zero)
        result = PaddedValue::zero;
    } else if (name == "intent.binary") {
      auto logical = user->getAttrOfType<StringAttr>("intent.operator");
      if (logical && logical.getValue() == "multiply" &&
          padded == PaddedValue::zero)
        result = PaddedValue::zero;
      else if (logical && logical.getValue() == "add" &&
               padded == PaddedValue::zero) {
        Value sibling = user->getOperand(0) == value ? user->getOperand(1)
                                                     : user->getOperand(0);
        Operation *siblingLoad = rootViewLoad(sibling);
        llvm::DenseMap<Value, PaddedValue> siblingVisited;
        if (siblingLoad && haveAlignedLoadMasks(value, sibling) &&
            provePaddedUses(siblingLoad->getResult(0), PaddedValue::zero,
                            siblingVisited, user))
          result = PaddedValue::zero;
      } else if (logical && logical.getValue() == "subtract" &&
                 user->getOperand(0) == value &&
                 padded == PaddedValue::negativeInfinity)
        result = PaddedValue::negativeInfinity;
    } else {
      return false;
    }

    if (user->getNumResults() != 1 ||
        !provePaddedUses(user->getResult(0), result, visited, ignoredUser))
      return false;
  }
  return true;
}

bool proveFill(Value loaded, PaddedValue padded) {
  llvm::DenseMap<Value, PaddedValue> visited;
  return provePaddedUses(loaded, padded, visited);
}

std::optional<std::string> literalPadding(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "intent.constant")
    return std::nullopt;
  Attribute literal = definition->getAttr("intent.value");
  if (auto floating = dyn_cast<FloatAttr>(literal)) {
    double number = floating.getValueAsDouble();
    if (number == 0.0)
      return std::string("zero");
    if (std::isinf(number) && number < 0.0)
      return std::string("negative_infinity");
  }
  if (auto integer = dyn_cast<IntegerAttr>(literal))
    if (integer.getValue().isZero())
      return std::string("zero");
  return std::nullopt;
}

std::optional<std::string> inferPadding(
    Value value, const KernelFacts &facts,
    const llvm::DenseMap<Value, std::string> &assumedPadding) {
  auto assumed = assumedPadding.find(value);
  if (assumed != assumedPadding.end())
    return assumed->second;
  if (std::optional<std::string> literal = literalPadding(value))
    return literal;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "intent.view_load") {
    auto fill = facts.boundaryFills.find(definition);
    return fill == facts.boundaryFills.end()
               ? std::nullopt
               : std::optional<std::string>(fill->second);
  }
  if (name == "intent.zeros")
    return std::string("zero");
  if (name == "intent.full" && definition->getNumOperands() == 1)
    return inferPadding(definition->getOperand(0), facts, assumedPadding);
  if ((name == "intent.cast" || name == "intent.broadcast") &&
      definition->getNumOperands() >= 1)
    return inferPadding(definition->getOperand(0), facts, assumedPadding);
  if (name == "intent.gather" && definition->getNumOperands() >= 1) {
    if (!isa<intent::ViewType>(definition->getOperand(0).getType()))
      return inferPadding(definition->getOperand(0), facts, assumedPadding);
    auto fillIndex =
        definition->getAttrOfType<IntegerAttr>("intent.fill_operand_index");
    if (fillIndex && fillIndex.getInt() >= 0 &&
        static_cast<unsigned>(fillIndex.getInt()) < definition->getNumOperands())
      return inferPadding(definition->getOperand(fillIndex.getInt()), facts,
                          assumedPadding);
    return std::nullopt;
  }
  if (name == "intent.unary" && definition->getNumOperands() == 1) {
    std::optional<std::string> operand =
        inferPadding(definition->getOperand(0), facts, assumedPadding);
    auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
    if (operand && *operand == "negative_infinity" && logical &&
        (logical.getValue() == "exp" || logical.getValue() == "exp2"))
      return std::string("zero");
    if (operand && *operand == "zero" && logical &&
        logical.getValue() == "negate")
      return std::string("zero");
  }
  if (name == "intent.binary" && definition->getNumOperands() == 2) {
    std::optional<std::string> lhs =
        inferPadding(definition->getOperand(0), facts, assumedPadding);
    std::optional<std::string> rhs =
        inferPadding(definition->getOperand(1), facts, assumedPadding);
    auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
    if (logical && logical.getValue() == "multiply" && lhs && rhs &&
        *lhs == "zero" && *rhs == "zero")
      return std::string("zero");
    if (logical && logical.getValue() == "subtract" && lhs &&
        *lhs == "negative_infinity")
      return std::string("negative_infinity");
  }
  if (name == "intent.mask" && definition->getNumOperands() == 3) {
    std::optional<std::string> valuePadding =
        inferPadding(definition->getOperand(0), facts, assumedPadding);
    std::optional<std::string> fillPadding =
        inferPadding(definition->getOperand(2), facts, assumedPadding);
    if (valuePadding && fillPadding && *valuePadding == *fillPadding)
      return valuePadding;
  }
  return std::nullopt;
}

} // namespace

std::optional<std::string> inferMaskedLaneFill(Value loaded) {
  if (proveFill(loaded, PaddedValue::zero))
    return std::string("zero");
  if (proveFill(loaded, PaddedValue::negativeInfinity))
    return std::string("negative_infinity");
  return std::nullopt;
}

std::optional<std::string> inferValuePadding(
    Value value, const KernelFacts &facts,
    const llvm::DenseMap<Value, std::string> &assumedPadding) {
  return inferPadding(value, facts, assumedPadding);
}

} // namespace intent::target

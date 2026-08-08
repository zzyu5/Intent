#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::target {
namespace {

enum class PaddedValue {
  unknown,
  zero,
  negativeInfinity,
};

bool provePaddedUses(Value value, PaddedValue padded,
                     llvm::DenseMap<Value, PaddedValue> &visited) {
  auto found = visited.find(value);
  if (found != visited.end())
    return found->second == padded;
  visited[value] = padded;

  for (Operation *user : value.getUsers()) {
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

    PaddedValue result = PaddedValue::unknown;
    if (name == "intent.cast") {
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
      else if (logical && logical.getValue() == "subtract" &&
               user->getOperand(0) == value &&
               padded == PaddedValue::negativeInfinity)
        result = PaddedValue::negativeInfinity;
    } else {
      return false;
    }

    if (user->getNumResults() != 1 ||
        !provePaddedUses(user->getResult(0), result, visited))
      return false;
  }
  return true;
}

bool proveFill(Value loaded, PaddedValue padded) {
  llvm::DenseMap<Value, PaddedValue> visited;
  return provePaddedUses(loaded, padded, visited);
}

} // namespace

std::optional<std::string> inferMaskedLaneFill(Value loaded) {
  if (proveFill(loaded, PaddedValue::zero))
    return std::string("zero");
  if (proveFill(loaded, PaddedValue::negativeInfinity))
    return std::string("negative_infinity");
  return std::nullopt;
}

} // namespace intent::target

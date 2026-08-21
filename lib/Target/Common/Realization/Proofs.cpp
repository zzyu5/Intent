#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/Record.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"

#include <cmath>

using namespace mlir;

namespace intent::target {
namespace {

enum class PaddedValue {
  arbitrary,
  zero,
  negativeInfinity,
  booleanFalse,
  booleanTrue,
};

bool isShapeOnlyGather(Operation &operation) {
  auto relation = operation.getAttrOfType<ArrayAttr>("intent.index");
  if (!relation)
    return false;
  for (Attribute attribute : relation) {
    auto term = dyn_cast<DictionaryAttr>(attribute);
    auto kind = term ? term.getAs<StringAttr>("kind") : StringAttr();
    if (!kind ||
        (kind.getValue() != "full_slice" && kind.getValue() != "new_axis"))
      return false;
  }
  return true;
}

std::optional<std::string> literalPadding(Value value);
std::optional<std::string> semanticLiteralPadding(Value value);

bool isArgmaxLowestReduction(Operation &operation) {
  if (::intent::target::semanticOperationName(operation) != "intent.reduce")
    return false;
  auto builtin =
      operation.getAttrOfType<StringAttr>("intent.combine_builtin");
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  auto captures =
      operation.getAttrOfType<IntegerAttr>("intent.capture_count");
  return builtin && builtin.getValue() == "argmax_lowest" && components &&
         components.getInt() == 2 && captures && captures.getInt() == 0 &&
         operation.getNumOperands() == 4 && operation.getNumResults() == 2;
}

bool provePaddedUses(Value value, PaddedValue padded,
                     llvm::DenseMap<Value, PaddedValue> &visited) {
  auto found = visited.find(value);
  if (found != visited.end())
    return found->second == padded;
  visited[value] = padded;

  for (Operation *user : value.getUsers()) {
    StringRef name = ::intent::target::semanticOperationName(*user);
    if (name == "intent.view_store" || name == "intent.scatter_unique" ||
        name == "intent.scatter_reduce" || name == "intent.atomic_add")
      continue;
    if (name == "intent.mask" && padded == PaddedValue::zero &&
        user->getNumOperands() == 3 && user->getOperand(0) == value &&
        semanticLiteralPadding(user->getOperand(2)) == "zero")
      continue;
    if (name == "intent.indices" && user->getNumOperands() == 1 &&
        user->getOperand(0) == value) {
      auto mode = user->getAttrOfType<StringAttr>("intent.mode");
      if (mode && mode.getValue() == "tensor_axis")
        continue;
      return false;
    }
    if (name == "intent.mask" && user->getNumOperands() == 3 &&
        user->getOperand(0) == value) {
      std::optional<std::string> fill =
          semanticLiteralPadding(user->getOperand(2));
      bool matching =
          (padded == PaddedValue::zero && fill == "zero") ||
          (padded == PaddedValue::negativeInfinity &&
           fill == "negative_infinity") ||
          (padded == PaddedValue::booleanFalse && fill == "false") ||
          (padded == PaddedValue::booleanTrue && fill == "true");
      if (matching && user->getNumResults() == 1 &&
          provePaddedUses(user->getResult(0), padded, visited))
        continue;
      return false;
    }
    if (name == "intent.reduce") {
      if (isArgmaxLowestReduction(*user)) {
        if (user->getOperand(0) != value ||
            padded != PaddedValue::negativeInfinity ||
            semanticLiteralPadding(user->getOperand(2)) !=
                "negative_infinity")
          return false;
        continue;
      }
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      if (!combine ||
          (combine.getValue() == "add" && padded != PaddedValue::zero) ||
          (combine.getValue() == "maximum" &&
           padded != PaddedValue::negativeInfinity) ||
          (combine.getValue() == "logical_or" &&
           padded != PaddedValue::booleanFalse) ||
          (combine.getValue() == "logical_and" &&
           padded != PaddedValue::booleanTrue) ||
          (combine.getValue() != "add" && combine.getValue() != "maximum" &&
           combine.getValue() != "logical_or" &&
           combine.getValue() != "logical_and"))
        return false;
      continue;
    }
    if (name == "intent.scan") {
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      if (!combine || combine.getValue() != "add" ||
          padded != PaddedValue::zero)
        return false;
      continue;
    }
    bool scaledDataOperand =
        name == "intent.scaled_contract" && user->getNumOperands() == 4 &&
        (user->getOperand(0) == value || user->getOperand(1) == value);
    if ((name == "intent.contract" || scaledDataOperand) &&
        padded == PaddedValue::zero) {
      auto multiply = user->getAttrOfType<StringAttr>("intent.multiply");
      auto combine = user->getAttrOfType<StringAttr>("intent.combine");
      if (multiply && multiply.getValue() == "multiply" && combine &&
          combine.getValue() == "add")
        continue;
      return false;
    }

    PaddedValue result = PaddedValue::arbitrary;
    if (name == "intent.cast" || name == "intent.broadcast" ||
        name == "intent.reshape" || name == "intent.transpose") {
      result = padded;
    } else if (name == "intent.gather" && user->getNumOperands() >= 1 &&
               user->getOperand(0) == value && isShapeOnlyGather(*user)) {
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
      else if (logical && logical.getValue() == "bitwise_and" &&
               padded == PaddedValue::zero)
        result = PaddedValue::zero;
      else if (logical &&
               (logical.getValue() == "left_shift" ||
                logical.getValue() == "right_shift") &&
               user->getOperand(0) == value && padded == PaddedValue::zero)
        result = PaddedValue::zero;
      else if (logical && logical.getValue() == "add" &&
               padded == PaddedValue::zero) {
        // This input is neutral; consumers realize the result's own padding.
        continue;
      } else if (logical && logical.getValue() == "subtract" &&
                 user->getNumOperands() == 2 &&
                 user->getOperand(1) == value && padded == PaddedValue::zero) {
        // A zero-padded right operand is neutral for subtraction; the result's
        // invalid-lane value is determined by the left operand.
        continue;
      }
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

std::optional<std::string> literalPadding(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition || ::intent::target::semanticOperationName(*definition) != "intent.constant")
    return std::nullopt;
  Attribute literal = definition->getAttr("intent.value");
  if (value.getType().isInteger(1)) {
    if (auto boolean = dyn_cast<IntegerAttr>(literal))
      return boolean.getValue().isZero() ? std::string("false")
                                         : std::string("true");
  }
  if (auto floating = dyn_cast<FloatAttr>(literal)) {
    const llvm::APFloat &number = floating.getValue();
    if (number.isZero())
      return std::string("zero");
    if (number.isInfinity())
      return number.isNegative() ? std::string("negative_infinity")
                                 : std::string("positive_infinity");
    if (number.isNaN())
      return std::string("nan");
    llvm::SmallString<32> spelling;
    number.toString(spelling);
    return "literal_float:" + spelling.str().str();
  }
  if (auto integer = dyn_cast<IntegerAttr>(literal)) {
    if (integer.getValue().isZero())
      return std::string("zero");
    else
      return "literal_integer:" + std::to_string(integer.getInt());
  }
  return std::nullopt;
}

std::optional<std::string> semanticLiteralPadding(Value value) {
  if (std::optional<std::string> literal = literalPadding(value))
    return literal;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getNumOperands() != 1)
    return std::nullopt;
  StringRef name = ::intent::target::semanticOperationName(*definition);
  if (name != "intent.cast" && name != "intent.broadcast" &&
      name != "intent.reshape")
    return std::nullopt;
  return semanticLiteralPadding(definition->getOperand(0));
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
  StringRef name = ::intent::target::semanticOperationName(*definition);
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
  if (name == "intent.extract") {
    FailureOr<Value> field = resolveRecordField(*definition);
    return succeeded(field)
               ? inferPadding(*field, facts, assumedPadding)
               : std::nullopt;
  }
  if ((name == "intent.cast" || name == "intent.broadcast" ||
       name == "intent.reshape" || name == "intent.transpose") &&
      definition->getNumOperands() >= 1)
    return inferPadding(definition->getOperand(0), facts, assumedPadding);
  if (name == "intent.gather" && definition->getNumOperands() >= 1) {
    if (!isa<intent::ViewType>(definition->getOperand(0).getType()) &&
        isShapeOnlyGather(*definition))
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
    if (logical && logical.getValue() == "bitwise_and" &&
        ((lhs && *lhs == "zero") || (rhs && *rhs == "zero")))
      return std::string("zero");
    if (logical &&
        (logical.getValue() == "left_shift" ||
         logical.getValue() == "right_shift") &&
        lhs && *lhs == "zero")
      return std::string("zero");
  }
  if (name == "intent.mask" && definition->getNumOperands() == 3) {
    std::optional<std::string> valuePadding =
        inferPadding(definition->getOperand(0), facts, assumedPadding);
    std::optional<std::string> fillPadding =
        inferPadding(definition->getOperand(2), facts, assumedPadding);
    if (valuePadding && fillPadding && *valuePadding == *fillPadding)
      return valuePadding;
  }
  if (name == "intent.select" && definition->getNumOperands() == 3) {
    std::optional<std::string> truePadding =
        inferPadding(definition->getOperand(1), facts, assumedPadding);
    std::optional<std::string> falsePadding =
        inferPadding(definition->getOperand(2), facts, assumedPadding);
    if (truePadding && falsePadding && *truePadding == *falsePadding)
      return truePadding;
  }
  if (name == "intent.reduce" && isArgmaxLowestReduction(*definition) &&
      value == definition->getResult(0)) {
    std::optional<std::string> inputPadding =
        inferPadding(definition->getOperand(0), facts, assumedPadding);
    std::optional<std::string> identityPadding =
        inferPadding(definition->getOperand(2), facts, assumedPadding);
    if (inputPadding && identityPadding && *inputPadding == *identityPadding &&
        *inputPadding == "negative_infinity")
      return inputPadding;
  }
  if ((name == "intent.reduce" || name == "intent.scan") &&
      definition->getNumOperands() == 2) {
    std::optional<std::string> inputPadding =
        inferPadding(definition->getOperand(0), facts, assumedPadding);
    std::optional<std::string> identityPadding =
        inferPadding(definition->getOperand(1), facts, assumedPadding);
    auto combine = definition->getAttrOfType<StringAttr>("intent.combine");
    if (inputPadding && identityPadding && *inputPadding == *identityPadding &&
        combine &&
        ((combine.getValue() == "add" && *inputPadding == "zero") ||
         (combine.getValue() == "maximum" &&
          *inputPadding == "negative_infinity") ||
         (combine.getValue() == "logical_or" &&
          *inputPadding == "false") ||
         (combine.getValue() == "logical_and" &&
          *inputPadding == "true")))
      return inputPadding;
  }
  return std::nullopt;
}

} // namespace

std::optional<std::string> inferMaskedLaneFill(Value loaded) {
  if (proveFill(loaded, PaddedValue::zero))
    return std::string("zero");
  if (proveFill(loaded, PaddedValue::negativeInfinity))
    return std::string("negative_infinity");
  if (proveFill(loaded, PaddedValue::booleanFalse))
    return std::string("false");
  if (proveFill(loaded, PaddedValue::booleanTrue))
    return std::string("true");
  return std::nullopt;
}

std::optional<std::string> inferValuePadding(
    Value value, const KernelFacts &facts,
    const llvm::DenseMap<Value, std::string> &assumedPadding) {
  return inferPadding(value, facts, assumedPadding);
}

} // namespace intent::target

#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/Record.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"

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
std::optional<std::string> inferPadding(
    Value value, const KernelFacts &facts,
    const llvm::DenseMap<Value, std::string> &assumedPadding);

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
                     const KernelFacts &facts,
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
          provePaddedUses(user->getResult(0), padded, facts, visited))
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
      } else if (logical && logical.getValue() == "subtract" &&
                 user->getNumOperands() == 2 &&
                 user->getOperand(0) == value && padded == PaddedValue::zero) {
        llvm::DenseMap<Value, std::string> assumedPadding;
        assumedPadding[value] = "zero";
        std::optional<std::string> rhs =
            inferPadding(user->getOperand(1), facts, assumedPadding);
        if (rhs && *rhs == "zero")
          result = PaddedValue::zero;
      }
    } else {
      return false;
    }

    if (user->getNumResults() != 1 ||
        !provePaddedUses(user->getResult(0), result, facts, visited))
      return false;
  }
  return true;
}

bool proveFill(Value loaded, PaddedValue padded, const KernelFacts &facts) {
  llvm::DenseMap<Value, PaddedValue> visited;
  return provePaddedUses(loaded, padded, facts, visited);
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
        (*lhs == "zero" || *rhs == "zero"))
      return std::string("zero");
    if (logical && logical.getValue() == "bitwise_and" &&
        ((lhs && *lhs == "zero") || (rhs && *rhs == "zero")))
      return std::string("zero");
    if (logical &&
        (logical.getValue() == "left_shift" ||
         logical.getValue() == "right_shift") &&
        lhs && *lhs == "zero")
      return std::string("zero");
    if (logical &&
        (logical.getValue() == "add" ||
         logical.getValue() == "subtract") &&
        lhs && rhs && *lhs == "zero" && *rhs == "zero")
      return std::string("zero");
  }
  if ((name == "intent.contract" || name == "intent.scaled_contract") &&
      definition->getNumOperands() >= 2) {
    auto multiply = definition->getAttrOfType<StringAttr>("intent.multiply");
    auto combine = definition->getAttrOfType<StringAttr>("intent.combine");
    if (multiply && multiply.getValue() == "multiply" && combine &&
        combine.getValue() == "add") {
      std::optional<std::string> lhs =
          inferPadding(definition->getOperand(0), facts, assumedPadding);
      std::optional<std::string> rhs =
          inferPadding(definition->getOperand(1), facts, assumedPadding);
      if ((lhs && *lhs == "zero") || (rhs && *rhs == "zero"))
        return std::string("zero");
    }
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

class DomainPaddingProof {
public:
  DomainPaddingProof(const KernelFacts &facts, Operation *candidateLoad,
                     StringRef candidateFill,
                     const MaterializedPaddingQuery &materializedPadding)
      : facts(facts), candidateLoad(candidateLoad),
        candidateFill(candidateFill.str()),
        materializedPadding(materializedPadding) {}

  std::optional<std::string>
  infer(Value value, Operation *domain,
        Value ignoredMaterialization = {}) const {
    if (materializedPadding) {
      std::optional<std::string> materialized =
          materializedPadding(value, domain, ignoredMaterialization);
      if (materialized)
        return materialized;
    }
    Operation *definition = value.getDefiningOp();
    if (!definition)
      return std::nullopt;
    StringRef name = ::intent::target::semanticOperationName(*definition);
    if (name == "intent.view_load") {
      auto bounded = facts.boundaryDomains.find(definition);
      if (bounded == facts.boundaryDomains.end() ||
          !containsDomain(bounded->second, domain))
        return std::nullopt;
      if (definition == candidateLoad)
        return candidateFill;
      auto fill = facts.boundaryFills.find(definition);
      return fill != facts.boundaryFills.end() && fill->second != "none"
                 ? std::optional<std::string>(fill->second)
                 : std::nullopt;
    }
    if (name == "intent.zeros")
      return std::string("zero");
    if (name == "intent.constant") {
      Attribute literal = definition->getAttr("intent.value");
      if (value.getType().isInteger(1)) {
        if (auto boolean = dyn_cast_or_null<IntegerAttr>(literal))
          return boolean.getValue().isZero()
                     ? std::optional<std::string>("false")
                     : std::optional<std::string>("true");
      }
      if (auto integer = dyn_cast_or_null<IntegerAttr>(literal))
        return integer.getValue().isZero()
                   ? std::optional<std::string>("zero")
                   : std::nullopt;
      if (auto floating = dyn_cast_or_null<FloatAttr>(literal)) {
        double number = floating.getValueAsDouble();
        if (number == 0.0)
          return std::string("zero");
        if (std::isinf(number) && number < 0.0)
          return std::string("negative_infinity");
      }
      return std::nullopt;
    }
    if (name == "intent.full" && definition->getNumOperands() == 1)
      return infer(definition->getOperand(0), domain,
                   ignoredMaterialization);
    if ((name == "intent.cast" || name == "intent.broadcast" ||
         name == "intent.reshape" || name == "intent.transpose") &&
        definition->getNumOperands() >= 1)
      return infer(definition->getOperand(0), domain,
                   ignoredMaterialization);
    if (name == "intent.gather" && definition->getNumOperands() >= 1) {
      FailureOr<SmallVector<IndexTerm>> relation =
          parseIndexRelation(*definition);
      auto validIndex =
          definition->getAttrOfType<IntegerAttr>("intent.valid_operand_index");
      Operation *validDefinition =
          validIndex && validIndex.getInt() >= 0 &&
                  static_cast<unsigned>(validIndex.getInt()) <
                      definition->getNumOperands()
              ? definition->getOperand(validIndex.getInt()).getDefiningOp()
              : nullptr;
      auto validLiteral =
          validDefinition &&
                  ::intent::target::semanticOperationName(*validDefinition) ==
                      "intent.constant"
              ? validDefinition->getAttrOfType<IntegerAttr>("intent.value")
              : IntegerAttr();
      bool pureExpansion =
          succeeded(relation) &&
          llvm::all_of(*relation, [](const IndexTerm &term) {
            return term.kind == "full_slice" || term.kind == "new_axis";
          });
      if (pureExpansion && validLiteral &&
          !validLiteral.getValue().isZero())
        return infer(definition->getOperand(0), domain,
                     ignoredMaterialization);
    }
    if (name == "intent.unary" && definition->getNumOperands() == 1) {
      std::optional<std::string> operand =
          infer(definition->getOperand(0), domain, ignoredMaterialization);
      auto logical =
          definition->getAttrOfType<StringAttr>("intent.operator");
      if (operand && *operand == "negative_infinity" && logical &&
          (logical.getValue() == "exp" || logical.getValue() == "exp2"))
        return std::string("zero");
      if (operand && *operand == "zero" && logical &&
          logical.getValue() == "negate")
        return std::string("zero");
      return std::nullopt;
    }
    if (name == "intent.binary" && definition->getNumOperands() == 2) {
      std::optional<std::string> lhs =
          infer(definition->getOperand(0), domain, ignoredMaterialization);
      std::optional<std::string> rhs =
          infer(definition->getOperand(1), domain, ignoredMaterialization);
      auto logical =
          definition->getAttrOfType<StringAttr>("intent.operator");
      if (!logical)
        return std::nullopt;
      if (logical.getValue() == "multiply" &&
          ((lhs && *lhs == "zero") || (rhs && *rhs == "zero")))
        return std::string("zero");
      if (logical.getValue() == "add" && lhs && rhs && *lhs == "zero" &&
          *rhs == "zero")
        return std::string("zero");
      if (logical.getValue() == "subtract" && lhs &&
          *lhs == "negative_infinity")
        return std::string("negative_infinity");
      if (logical.getValue() == "subtract" && lhs && rhs &&
          *lhs == "zero" && *rhs == "zero")
        return std::string("zero");
    }
    if ((name == "intent.contract" || name == "intent.scaled_contract") &&
        definition->getNumOperands() >= 2) {
      auto fact = facts.contractions.find(definition);
      auto multiply =
          definition->getAttrOfType<StringAttr>("intent.multiply");
      auto combine = definition->getAttrOfType<StringAttr>("intent.combine");
      if (fact == facts.contractions.end() || !multiply ||
          multiply.getValue() != "multiply" || !combine ||
          combine.getValue() != "add")
        return std::nullopt;
      for (unsigned operandNumber : {0u, 1u}) {
        Value operand = definition->getOperand(operandNumber);
        std::optional<std::string> padding =
            infer(operand, domain, ignoredMaterialization);
        if (!padding || *padding != "zero")
          continue;
        ArrayRef<LogicalAxis> axes =
            operandNumber == 0 ? ArrayRef(fact->second.lhsAxes)
                               : ArrayRef(fact->second.rhsAxes);
        ArrayRef<unsigned> reduced =
            operandNumber == 0
                ? ArrayRef(fact->second.lhsReductionAxes)
                : ArrayRef(fact->second.rhsReductionAxes);
        bool survivingDomain = false;
        for (auto [axisNumber, axis] : llvm::enumerate(axes))
          if (axis.domain == domain &&
              !llvm::is_contained(reduced, static_cast<unsigned>(axisNumber)))
            survivingDomain = true;
        if (survivingDomain)
          return std::string("zero");
      }
    }
    return std::nullopt;
  }

  bool prove(Operation &load) const {
    auto domains = facts.boundaryDomains.find(&load);
    if (load.getNumResults() != 1 || domains == facts.boundaryDomains.end() ||
        domains->second.empty() || candidateFill.empty() ||
        candidateFill == "none")
      return false;
    SmallVector<Operation *> validityDomains;
    for (Operation *domain : domains->second)
      if (uniqueAxis(load.getResult(0), domain))
        validityDomains.push_back(domain);
    if (validityDomains.empty())
      return false;
    llvm::DenseSet<Value> active;
    return proveUses(load.getResult(0), validityDomains, active);
  }

private:
  const KernelFacts &facts;
  Operation *candidateLoad;
  std::string candidateFill;
  const MaterializedPaddingQuery &materializedPadding;

  static bool containsDomain(ArrayRef<Operation *> domains,
                             Operation *domain) {
    return llvm::is_contained(domains, domain);
  }

  std::optional<unsigned> uniqueAxis(Value value, Operation *domain) const {
    auto axes = facts.valueAxes.find(value);
    if (axes == facts.valueAxes.end())
      return std::nullopt;
    std::optional<unsigned> result;
    for (auto [axisNumber, axis] : llvm::enumerate(axes->second)) {
      if (axis.domain != domain)
        continue;
      if (result)
        return std::nullopt;
      result = axisNumber;
    }
    return result;
  }

  SmallVector<unsigned> matchingAxes(Value value, Operation *domain) const {
    SmallVector<unsigned> result;
    auto axes = facts.valueAxes.find(value);
    if (axes == facts.valueAxes.end())
      return result;
    for (auto [axisNumber, axis] : llvm::enumerate(axes->second))
      if (axis.domain == domain)
        result.push_back(axisNumber);
    return result;
  }

  bool valueCarriesDomains(Value value,
                           ArrayRef<Operation *> domains) const {
    return llvm::all_of(domains, [&](Operation *domain) {
      return !matchingAxes(value, domain).empty();
    });
  }

  bool proveContractUse(Operation &contract, Value value,
                        ArrayRef<Operation *> domains,
                        llvm::DenseSet<Value> &active) const {
    auto fact = facts.contractions.find(&contract);
    if (fact == facts.contractions.end() || contract.getNumOperands() < 2 ||
        contract.getNumResults() != 1)
      return false;
    int64_t operandNumber = -1;
    for (auto [index, operand] : llvm::enumerate(contract.getOperands())) {
      if (operand != value)
        continue;
      if (operandNumber >= 0 || index >= 2)
        return false;
      operandNumber = index;
    }
    if (operandNumber < 0)
      return false;
    ArrayRef<LogicalAxis> axes =
        operandNumber == 0 ? ArrayRef(fact->second.lhsAxes)
                           : ArrayRef(fact->second.rhsAxes);
    auto reductionPairs =
        contract.getAttrOfType<ArrayAttr>("intent.reduce");
    if (!reductionPairs)
      return false;
    SmallVector<Operation *> surviving;
    for (Operation *domain : domains) {
      SmallVector<unsigned> axisNumbers = matchingAxes(value, domain);
      if (axisNumbers.empty())
        return false;
      bool domainSurvives = false;
      for (unsigned axisNumber : axisNumbers) {
        if (axisNumber >= axes.size())
          return false;
        std::optional<unsigned> pairedAxis;
        for (Attribute attribute : reductionPairs) {
          auto pair = dyn_cast<ArrayAttr>(attribute);
          auto lhs = pair && pair.size() == 2
                         ? dyn_cast<IntegerAttr>(pair[0])
                         : IntegerAttr();
          auto rhs = pair && pair.size() == 2
                         ? dyn_cast<IntegerAttr>(pair[1])
                         : IntegerAttr();
          if (!lhs || !rhs)
            return false;
          int64_t current = operandNumber == 0 ? lhs.getInt() : rhs.getInt();
          if (current == static_cast<int64_t>(axisNumber))
            pairedAxis = operandNumber == 0 ? rhs.getInt() : lhs.getInt();
        }
        if (!pairedAxis) {
          domainSurvives = true;
          continue;
        }
        Value other = contract.getOperand(operandNumber == 0 ? 1 : 0);
        ArrayRef<LogicalAxis> otherAxes =
            operandNumber == 0 ? ArrayRef(fact->second.rhsAxes)
                               : ArrayRef(fact->second.lhsAxes);
        if (*pairedAxis >= otherAxes.size() ||
            otherAxes[*pairedAxis].domain != domain)
          return false;
        std::optional<std::string> otherPadding = infer(other, domain);
        if (!otherPadding || *otherPadding != "zero")
          return false;
      }
      if (domainSurvives)
        surviving.push_back(domain);
    }
    return surviving.empty() ||
           proveUses(contract.getResult(0), surviving, active);
  }

  bool proveReductionUse(Operation &reduction, Value value,
                         ArrayRef<Operation *> domains,
                         llvm::DenseSet<Value> &active) const {
    auto components =
        reduction.getAttrOfType<IntegerAttr>("intent.component_count");
    if (!components || components.getInt() <= 0 ||
        static_cast<unsigned>(components.getInt()) !=
            reduction.getNumResults())
      return false;
    bool consumed = false;
    for (auto [operandNumber, operand] :
         llvm::enumerate(reduction.getOperands())) {
      if (operand != value)
        continue;
      if (operandNumber >= static_cast<unsigned>(components.getInt()))
        return false;
      consumed = true;
      Value result = reduction.getResult(operandNumber);
      if (!valueCarriesDomains(result, domains) ||
          !proveUses(result, domains, active))
        return false;
    }
    return consumed;
  }

  bool proveYieldUse(Operation &yield, Value value,
                     ArrayRef<Operation *> domains,
                     llvm::DenseSet<Value> &active) const {
    Operation *owner = yield.getParentOp();
    StringRef ownerName = owner ? ::intent::target::semanticOperationName(*owner)
                               : StringRef();
    if (!owner ||
        (ownerName != "intent.state_stream" && ownerName != "intent.for") ||
        owner->getNumResults() != yield.getNumOperands())
      return false;
    bool consumed = false;
    for (auto [operandNumber, operand] : llvm::enumerate(yield.getOperands())) {
      if (operand != value)
        continue;
      consumed = true;
      Value result = owner->getResult(operandNumber);
      if (!valueCarriesDomains(result, domains) ||
          !proveUses(result, domains, active))
        return false;
    }
    return consumed;
  }

  bool proveUses(Value value, ArrayRef<Operation *> domains,
                 llvm::DenseSet<Value> &active) const {
    SmallVector<Operation *> remaining;
    for (Operation *domain : domains)
      if (!materializedPadding ||
          !materializedPadding(value, domain, {}).has_value())
        remaining.push_back(domain);
    if (remaining.empty())
      return true;
    if (!active.insert(value).second)
      return false;
    auto finish = [&](bool result) {
      active.erase(value);
      return result;
    };
    if (value.use_empty())
      return finish(false);
    for (Operation *user : value.getUsers()) {
      StringRef name = ::intent::target::semanticOperationName(*user);
      if (name == "intent.view_store" || name == "intent.scatter_unique" ||
          name == "intent.scatter_reduce" || name == "intent.atomic_add" ||
          name == "intent.atomic_cas") {
        auto valueIndex =
            user->getAttrOfType<IntegerAttr>("intent.value_operand_index");
        auto bounded = facts.boundaryDomains.find(user);
        if (!valueIndex || valueIndex.getInt() < 0 ||
            static_cast<unsigned>(valueIndex.getInt()) >=
                user->getNumOperands() ||
            user->getOperand(valueIndex.getInt()) != value ||
            bounded == facts.boundaryDomains.end() ||
            !llvm::all_of(remaining, [&](Operation *domain) {
              return containsDomain(bounded->second, domain);
            }))
          return finish(false);
        continue;
      }
      if (name == "intent.contract" || name == "intent.scaled_contract") {
        if (!proveContractUse(*user, value, remaining, active))
          return finish(false);
        continue;
      }
      if (name == "intent.reduce" || name == "intent.scan") {
        if (!proveReductionUse(*user, value, remaining, active))
          return finish(false);
        continue;
      }
      if (name == "intent.yield") {
        if (!proveYieldUse(*user, value, remaining, active))
          return finish(false);
        continue;
      }
      if (name == "intent.gather" && isShapeOnlyGather(*user)) {
        if (user->getNumResults() != 1 ||
            !valueCarriesDomains(user->getResult(0), remaining) ||
            !proveUses(user->getResult(0), remaining, active))
          return finish(false);
        continue;
      }
      if (name != "intent.cast" && name != "intent.broadcast" &&
          name != "intent.reshape" && name != "intent.transpose" &&
          name != "intent.unary" && name != "intent.binary" &&
          name != "intent.compare" && name != "intent.select" &&
          name != "intent.mask")
        return finish(false);
      if (user->getNumResults() != 1 ||
          !valueCarriesDomains(user->getResult(0), remaining) ||
          !proveUses(user->getResult(0), remaining, active))
        return finish(false);
    }
    return finish(true);
  }
};

} // namespace

std::optional<std::string> inferMaskedLaneFill(Value loaded,
                                               const KernelFacts &facts) {
  Operation *load = loaded.getDefiningOp();
  if (proveFill(loaded, PaddedValue::zero, facts) ||
      (load && proveBoundaryNeutralization(*load, "zero", facts)))
    return std::string("zero");
  if (proveFill(loaded, PaddedValue::negativeInfinity, facts) ||
      (load &&
       proveBoundaryNeutralization(*load, "negative_infinity", facts)))
    return std::string("negative_infinity");
  if (proveFill(loaded, PaddedValue::booleanFalse, facts) ||
      (load && proveBoundaryNeutralization(*load, "false", facts)))
    return std::string("false");
  if (proveFill(loaded, PaddedValue::booleanTrue, facts) ||
      (load && proveBoundaryNeutralization(*load, "true", facts)))
    return std::string("true");
  return std::nullopt;
}

std::optional<std::string> inferValuePadding(
    Value value, const KernelFacts &facts,
    const llvm::DenseMap<Value, std::string> &assumedPadding) {
  return inferPadding(value, facts, assumedPadding);
}

std::optional<std::string> inferDomainPadding(
    Value value, Operation *domain, const KernelFacts &facts,
    const MaterializedPaddingQuery &materializedPadding,
    Value ignoredMaterialization) {
  DomainPaddingProof proof(facts, nullptr, "", materializedPadding);
  return proof.infer(value, domain, ignoredMaterialization);
}

bool proveBoundaryNeutralization(
    Operation &load, StringRef fill, const KernelFacts &facts,
    const MaterializedPaddingQuery &materializedPadding) {
  DomainPaddingProof proof(facts, &load, fill, materializedPadding);
  return proof.prove(load);
}

} // namespace intent::target

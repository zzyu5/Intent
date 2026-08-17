#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::target {
namespace {

LogicalResult addHandler(OperationHandlerRegistry &registry, StringRef name,
                         OperationCallback enter) {
  return registry.add(name, OperationHandler{std::move(enter), {}});
}

Operation *nearestParallelOwner(Operation &operation) {
  for (Operation *parent = operation.getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "intent.parallel")
      return parent;
  return nullptr;
}

LogicalResult analyzeBoundary(Operation &operation, KernelFacts &facts,
                              StringRef fill) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Operation *> domains;
  bool hasOpaqueScalarIndex = false;
  bool hasInBoundsIndex = false;
  bool requiresBoundarySource = false;
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice" || term.kind == "static_index" ||
        term.kind == "slice") {
      ++sourceAxis;
      continue;
    }
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "boundary analysis requires one value per dynamic index term");
    requiresBoundarySource = true;
    Value indexed = operation.getOperand(*term.operands.front());
    auto indexedAxes = facts.valueAxes.find(indexed);
    if (term.kind == "value_index") {
      if (hasInBoundsPrecondition(indexed, operation.getOperand(0), sourceAxis,
                                  operation)) {
        hasInBoundsIndex = true;
        ++sourceAxis;
        continue;
      }
      if (indexedAxes != facts.valueAxes.end()) {
        for (const LogicalAxis &axis : indexedAxes->second)
          if (axis.domain && !llvm::is_contained(domains, axis.domain))
            domains.push_back(axis.domain);
        ++sourceAxis;
        continue;
      }
      FailureOr<ScalarIndexSource> source =
          traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      if (source->hasDomain()) {
        for (Operation *domain : source->domains)
          if (!llvm::is_contained(domains, domain))
            domains.push_back(domain);
      } else if (source->opaque) {
        hasOpaqueScalarIndex = true;
      }
      ++sourceAxis;
      continue;
    }
    FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
    if (failed(domain))
      return failure();
    domains.push_back(*domain);
    ++sourceAxis;
  }
  if (requiresBoundarySource && domains.empty() && !hasOpaqueScalarIndex &&
      !hasInBoundsIndex)
    return operation.emitOpError("has no domain-bound index for realization");
  facts.boundaryDomains[&operation] = std::move(domains);
  facts.boundaryFills[&operation] = fill.str();
  return success();
}

FailureOr<Value> backingView(Value tensor, const KernelFacts &facts,
                             Operation &consumer) {
  Operation *load = tensor.getDefiningOp();
  if (!load || !facts.wholeViewLoads.contains(load) ||
      load->getNumOperands() != 1 ||
      !isa<intent::ViewType>(load->getOperand(0).getType())) {
    consumer.emitOpError("ragged metadata is not a canonical whole-view load");
    return failure();
  }
  return load->getOperand(0);
}

FailureOr<LogicalAxis> axisFromView(Value source, unsigned sourceAxis,
                                    const KernelFacts &facts,
                                    Operation &consumer) {
  auto argument = llvm::find_if(facts.kernel.abi.arguments,
                                [&](const ABIArgument &candidate) {
                                  return candidate.value == source;
                                });
  if (argument == facts.kernel.abi.arguments.end()) {
    consumer.emitOpError("axis source is not a canonical ABI view");
    return failure();
  }
  auto shape = argument->metadata.getAs<ArrayAttr>("shape");
  if (!shape || sourceAxis >= shape.size()) {
    consumer.emitOpError("axis exceeds canonical ABI shape metadata");
    return failure();
  }
  if (auto symbol = dyn_cast<StringAttr>(shape[sourceAxis]))
    return LogicalAxis{nullptr, symbol.getValue().str()};
  if (auto integer = dyn_cast<IntegerAttr>(shape[sourceAxis]))
    return LogicalAxis{nullptr, std::to_string(integer.getInt())};
  consumer.emitOpError("axis has an unsupported ABI extent");
  return failure();
}

FailureOr<LogicalAxis> axisFromDomain(Operation &domain,
                                      const KernelFacts &facts,
                                      Operation &consumer) {
  auto staticExtent = facts.staticDomainExtents.find(&domain);
  if (staticExtent != facts.staticDomainExtents.end())
    return LogicalAxis{&domain, std::to_string(staticExtent->second)};
  auto source = facts.domainSources.find(&domain);
  auto sourceAxis = facts.domainSourceAxes.find(&domain);
  if (source == facts.domainSources.end() ||
      sourceAxis == facts.domainSourceAxes.end() || sourceAxis->second < 0) {
    consumer.emitOpError("domain has no canonical ABI axis identity");
    return failure();
  }
  FailureOr<LogicalAxis> axis =
      axisFromView(source->second, sourceAxis->second, facts, consumer);
  if (failed(axis))
    return failure();
  axis->domain = &domain;
  return axis;
}

bool compatibleLogicalAxis(const LogicalAxis &lhs, const LogicalAxis &rhs) {
  if (lhs == rhs)
    return true;
  return lhs.extent == rhs.extent && (!lhs.domain || !rhs.domain);
}

LogicalAxis mergeLogicalAxis(const LogicalAxis &lhs, const LogicalAxis &rhs) {
  return !lhs.domain && rhs.domain ? rhs : lhs;
}

bool mergeLogicalAxes(ArrayRef<LogicalAxis> lhs, ArrayRef<LogicalAxis> rhs,
                      SmallVectorImpl<LogicalAxis> &merged) {
  if (lhs.size() != rhs.size())
    return false;
  merged.clear();
  merged.reserve(lhs.size());
  for (auto [left, right] : llvm::zip(lhs, rhs)) {
    if (!compatibleLogicalAxis(left, right))
      return false;
    merged.push_back(mergeLogicalAxis(left, right));
  }
  return true;
}

struct AffineIndexExpression {
  llvm::DenseMap<Operation *, int64_t> coefficients;
  int64_t constant = 0;
};

std::optional<int64_t> integerConstant(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition ||
      definition->getName().getStringRef() != "intent.constant")
    return std::nullopt;
  if (auto literal = definition->getAttrOfType<BoolAttr>("intent.value"))
    return literal.getValue() ? 1 : 0;
  auto literal = definition->getAttrOfType<IntegerAttr>("intent.value");
  return literal ? std::optional<int64_t>(literal.getInt()) : std::nullopt;
}

std::optional<int64_t> integerSplatConstant(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    StringRef name = definition->getName().getStringRef();
    if ((name != "intent.broadcast" && name != "intent.reshape" &&
         name != "intent.cast") ||
        definition->getNumOperands() != 1)
      break;
    value = definition->getOperand(0);
  }
  return integerConstant(value);
}

std::optional<AffineIndexExpression>
affineIndexExpression(Value value, const KernelFacts &facts,
                      Operation &consumer, llvm::DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return std::nullopt;
  auto finish = [&](std::optional<AffineIndexExpression> expression) {
    active.erase(value);
    return expression;
  };
  if (std::optional<int64_t> literal = integerConstant(value)) {
    AffineIndexExpression expression;
    expression.constant = *literal;
    return finish(std::move(expression));
  }

  Operation *definition = value.getDefiningOp();
  if (!definition) {
    FailureOr<ScalarIndexSource> source =
        traceScalarIndexSource(value, consumer);
    if (failed(source) || !source->domain || source->opaque || source->transformed)
      return finish(std::nullopt);
    AffineIndexExpression expression;
    expression.coefficients[source->domain] = 1;
    return finish(std::move(expression));
  }
  StringRef name = definition->getName().getStringRef();
  if (name == "intent.indices") {
    auto axes = facts.valueAxes.find(value);
    if (axes == facts.valueAxes.end() || axes->second.size() != 1 ||
        !axes->second.front().domain)
      return finish(std::nullopt);
    AffineIndexExpression expression;
    expression.coefficients[axes->second.front().domain] = 1;
    return finish(std::move(expression));
  }

  if (name == "intent.gather") {
    auto valid = definition->getAttrOfType<IntegerAttr>(
        "intent.valid_operand_index");
    if (valid) {
      int64_t operand = valid.getInt();
      if (operand < 0 ||
          static_cast<unsigned>(operand) >= definition->getNumOperands() ||
          integerConstant(definition->getOperand(operand)) !=
              std::optional<int64_t>(1))
        return finish(std::nullopt);
    }
  }
  if (name == "intent.broadcast" || name == "intent.reshape" ||
      name == "intent.transpose" || name == "intent.cast" ||
      name == "intent.gather") {
    if (definition->getNumOperands() == 0)
      return finish(std::nullopt);
    return finish(affineIndexExpression(definition->getOperand(0), facts,
                                        consumer, active));
  }

  if (name == "intent.unary" && definition->getNumOperands() == 1) {
    auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
    std::optional<AffineIndexExpression> operand = affineIndexExpression(
        definition->getOperand(0), facts, consumer, active);
    if (!logical || !operand || logical.getValue() != "negate")
      return finish(std::nullopt);
    operand->constant = -operand->constant;
    for (auto &coefficient : operand->coefficients)
      coefficient.second = -coefficient.second;
    return finish(std::move(operand));
  }

  if (name != "intent.binary" || definition->getNumOperands() != 2)
    return finish(std::nullopt);
  auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
  std::optional<AffineIndexExpression> lhs = affineIndexExpression(
      definition->getOperand(0), facts, consumer, active);
  std::optional<AffineIndexExpression> rhs = affineIndexExpression(
      definition->getOperand(1), facts, consumer, active);
  if (!logical || !lhs || !rhs)
    return finish(std::nullopt);
  auto scale = [](AffineIndexExpression &expression, int64_t factor) {
    expression.constant *= factor;
    for (auto &coefficient : expression.coefficients)
      coefficient.second *= factor;
  };
  if (logical.getValue() == "multiply") {
    if (lhs->coefficients.empty()) {
      int64_t factor = lhs->constant;
      scale(*rhs, factor);
      return finish(std::move(rhs));
    }
    if (rhs->coefficients.empty()) {
      int64_t factor = rhs->constant;
      scale(*lhs, factor);
      return finish(std::move(lhs));
    }
    return finish(std::nullopt);
  }
  int64_t sign = logical.getValue() == "add"
                     ? 1
                     : logical.getValue() == "subtract" ? -1 : 0;
  if (sign == 0)
    return finish(std::nullopt);
  lhs->constant += sign * rhs->constant;
  for (const auto &coefficient : rhs->coefficients)
    lhs->coefficients[coefficient.first] += sign * coefficient.second;
  return finish(std::move(lhs));
}

std::optional<std::pair<int64_t, int64_t>>
staticAffineRange(const AffineIndexExpression &expression,
                  const KernelFacts &facts) {
  int64_t lower = expression.constant;
  int64_t upper = expression.constant;
  for (const auto &[domain, coefficient] : expression.coefficients) {
    auto extent = facts.staticDomainExtents.find(domain);
    if (extent == facts.staticDomainExtents.end() || extent->second <= 0)
      return std::nullopt;
    int64_t domainLower = 0;
    int64_t domainUpper = extent->second - 1;
    auto bounds = facts.staticDomainBounds.find(domain);
    if (bounds != facts.staticDomainBounds.end()) {
      domainLower = bounds->second.first;
      domainUpper = bounds->second.second - 1;
    }
    int64_t first;
    int64_t last;
    if (llvm::MulOverflow(coefficient, domainLower, first) ||
        llvm::MulOverflow(coefficient, domainUpper, last))
      return std::nullopt;
    int64_t nextLower;
    int64_t nextUpper;
    if (llvm::AddOverflow(lower, std::min(first, last), nextLower) ||
        llvm::AddOverflow(upper, std::max(first, last), nextUpper))
      return std::nullopt;
    lower = nextLower;
    upper = nextUpper;
  }
  return std::pair<int64_t, int64_t>{lower, upper};
}

std::optional<int64_t>
affineLowerBound(const AffineIndexExpression &expression,
                 const KernelFacts &facts) {
  int64_t lower = expression.constant;
  for (const auto &[domain, coefficient] : expression.coefficients) {
    std::optional<int64_t> domainLower;
    std::optional<int64_t> domainUpper;
    auto bounds = facts.staticDomainBounds.find(domain);
    if (bounds != facts.staticDomainBounds.end()) {
      domainLower = bounds->second.first;
      domainUpper = bounds->second.second - 1;
    } else {
      auto extent = facts.staticDomainExtents.find(domain);
      if (extent != facts.staticDomainExtents.end() && extent->second > 0) {
        domainLower = 0;
        domainUpper = extent->second - 1;
      } else if (facts.domainSources.count(domain)) {
        domainLower = 0;
      } else if (domain &&
                 domain->getName().getStringRef() == "intent.domain" &&
                 domain->getNumOperands() >= 2) {
        domainLower = integerConstant(domain->getOperand(0));
        std::optional<int64_t> stop = integerConstant(domain->getOperand(1));
        if (stop)
          domainUpper = *stop - 1;
      }
    }
    std::optional<int64_t> endpoint =
        coefficient >= 0 ? domainLower : domainUpper;
    if (!endpoint)
      return std::nullopt;
    int64_t contribution;
    int64_t next;
    if (llvm::MulOverflow(coefficient, *endpoint, contribution) ||
        llvm::AddOverflow(lower, contribution, next))
      return std::nullopt;
    lower = next;
  }
  return lower;
}

bool proveScalarAtLeast(Value value, int64_t minimum,
                        const KernelFacts &facts, Operation &consumer,
                        llvm::DenseSet<Value> &active,
                        llvm::DenseMap<Value, int64_t> &assumed) {
  auto invariant = assumed.find(value);
  if (invariant != assumed.end() && invariant->second >= minimum)
    return true;
  if (!active.insert(value).second)
    return false;
  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  if (std::optional<int64_t> literal = integerConstant(value))
    return finish(*literal >= minimum);

  {
    llvm::DenseSet<Value> affineActive;
    std::optional<AffineIndexExpression> expression =
        affineIndexExpression(value, facts, consumer, affineActive);
    std::optional<int64_t> lower =
        expression ? affineLowerBound(*expression, facts) : std::nullopt;
    if (lower && *lower >= minimum)
      return finish(true);
  }

  auto argument = dyn_cast<BlockArgument>(value);
  if (argument) {
    Operation *loop = argument.getOwner()->getParentOp();
    if (!loop || loop->getName().getStringRef() != "intent.for" ||
        loop->getNumRegions() != 1 || !llvm::hasSingleElement(loop->getRegion(0)) ||
        loop->getNumOperands() != loop->getNumResults() + 1)
      return finish(false);
    FailureOr<SmallVector<Operation *>> domains =
        expandDomainSource(loop->getOperand(0), *loop);
    if (failed(domains) || argument.getArgNumber() < domains->size())
      return finish(false);
    unsigned carried = argument.getArgNumber() - domains->size();
    if (carried >= loop->getNumResults())
      return finish(false);
    Operation &yield = loop->getRegion(0).front().back();
    if (yield.getName().getStringRef() != "intent.yield" ||
        yield.getNumOperands() != loop->getNumResults())
      return finish(false);
    Value initial = loop->getOperand(carried + 1);
    Value next = yield.getOperand(carried);
    assumed[value] = minimum;
    bool result = proveScalarAtLeast(initial, minimum, facts, consumer, active,
                                     assumed) &&
                  proveScalarAtLeast(next, minimum, facts, consumer, active,
                                     assumed);
    assumed.erase(value);
    return finish(result);
  }

  Operation *definition = value.getDefiningOp();
  if (!definition)
    return finish(false);
  StringRef name = definition->getName().getStringRef();
  if (name == "intent.cast" && definition->getNumOperands() == 1 &&
      isa<intent::LogicalIndexType>(definition->getOperand(0).getType())) {
    llvm::DenseSet<Value> sourceActive;
    std::optional<AffineIndexExpression> expression = affineIndexExpression(
        definition->getOperand(0), facts, *definition, sourceActive);
    std::optional<int64_t> lower =
        expression ? affineLowerBound(*expression, facts) : std::nullopt;
    std::optional<std::pair<int64_t, int64_t>> range =
        expression ? staticAffineRange(*expression, facts) : std::nullopt;
    auto resultInteger = dyn_cast<IntegerType>(definition->getResult(0).getType());
    if (lower && *lower >= minimum && range && resultInteger &&
        resultInteger.getWidth() > 1 &&
        (resultInteger.getWidth() >= 64 ||
         range->second <
             (int64_t{1} << (resultInteger.getWidth() - 1))))
      return finish(true);
  }
  auto signPreservingCast = [&]() {
    if (name != "intent.cast" || definition->getNumOperands() == 0)
      return false;
    Type source = definition->getOperand(0).getType();
    Type result = definition->getResult(0).getType();
    auto sourceInteger = dyn_cast<IntegerType>(source);
    auto resultInteger = dyn_cast<IntegerType>(result);
    return sourceInteger && resultInteger &&
           resultInteger.getWidth() >= sourceInteger.getWidth();
  };
  if ((signPreservingCast() || name == "intent.broadcast" ||
       name == "intent.reshape" || name == "intent.transpose") &&
      definition->getNumOperands() > 0)
    return finish(proveScalarAtLeast(definition->getOperand(0), minimum, facts,
                                     consumer, active, assumed));
  if (name != "intent.binary" || definition->getNumOperands() != 2)
    return finish(false);
  auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return finish(false);
  Value lhs = definition->getOperand(0);
  Value rhs = definition->getOperand(1);
  std::optional<int64_t> lhsConstant = integerConstant(lhs);
  std::optional<int64_t> rhsConstant = integerConstant(rhs);
  if (logical.getValue() == "add") {
    if (rhsConstant)
      return finish(proveScalarAtLeast(lhs, minimum - *rhsConstant, facts,
                                       consumer, active, assumed));
    if (lhsConstant)
      return finish(proveScalarAtLeast(rhs, minimum - *lhsConstant, facts,
                                       consumer, active, assumed));
    if (minimum <= 0)
      return finish(proveScalarAtLeast(lhs, 0, facts, consumer, active,
                                       assumed) &&
                    proveScalarAtLeast(rhs, 0, facts, consumer, active,
                                       assumed));
    return finish(false);
  }
  if (logical.getValue() == "multiply") {
    if (rhsConstant && *rhsConstant > 0)
      return finish(proveScalarAtLeast(
          lhs, llvm::divideCeil(minimum, *rhsConstant), facts, consumer, active,
          assumed));
    if (lhsConstant && *lhsConstant > 0)
      return finish(proveScalarAtLeast(
          rhs, llvm::divideCeil(minimum, *lhsConstant), facts, consumer, active,
          assumed));
    if (minimum <= 0)
      return finish(proveScalarAtLeast(lhs, 0, facts, consumer, active,
                                       assumed) &&
                    proveScalarAtLeast(rhs, 0, facts, consumer, active,
                                       assumed));
    return finish(false);
  }
  if (logical.getValue() == "floor_divide" && rhsConstant &&
      *rhsConstant > 0) {
    int64_t required;
    if (llvm::MulOverflow(minimum, *rhsConstant, required))
      return finish(false);
    return finish(
        proveScalarAtLeast(lhs, required, facts, consumer, active, assumed));
  }
  if (minimum <= 0 &&
      (logical.getValue() == "bitwise_and" ||
       logical.getValue() == "bitwise_or" ||
       logical.getValue() == "shift_left" ||
       logical.getValue() == "shift_right"))
    return finish(proveScalarAtLeast(lhs, 0, facts, consumer, active,
                                     assumed) &&
                  proveScalarAtLeast(rhs, 0, facts, consumer, active,
                                     assumed));
  if (minimum <= 0 && logical.getValue() == "remainder") {
    llvm::DenseSet<Value> divisorActive;
    llvm::DenseMap<Value, int64_t> divisorAssumed;
    return finish(proveScalarAtLeast(lhs, 0, facts, consumer, active,
                                     assumed) &&
                  proveScalarAtLeast(rhs, 1, facts, consumer, divisorActive,
                                     divisorAssumed));
  }
  return finish(false);
}

bool hasNonnegativeIntegerOperandsImpl(Operation &operation,
                                       const KernelFacts &facts) {
  if (operation.getName().getStringRef() != "intent.binary" ||
      operation.getNumOperands() != 2)
    return false;
  auto logical = operation.getAttrOfType<StringAttr>("intent.operator");
  if (!logical ||
      (logical.getValue() != "floor_divide" &&
       logical.getValue() != "remainder"))
    return false;
  llvm::DenseSet<Value> divisorActive;
  std::optional<AffineIndexExpression> divisor = affineIndexExpression(
      operation.getOperand(1), facts, operation, divisorActive);
  bool positiveDivisor = divisor && divisor->coefficients.empty() &&
                         divisor->constant > 0;
  if (!positiveDivisor) {
    llvm::DenseSet<Value> proofActive;
    llvm::DenseMap<Value, int64_t> assumed;
    positiveDivisor = proveScalarAtLeast(operation.getOperand(1), 1, facts,
                                         operation, proofActive, assumed);
  }
  if (!positiveDivisor)
    return false;
  llvm::DenseSet<Value> active;
  std::optional<AffineIndexExpression> dividend = affineIndexExpression(
      operation.getOperand(0), facts, operation, active);
  std::optional<int64_t> lower =
      dividend ? affineLowerBound(*dividend, facts) : std::nullopt;
  if (lower && *lower >= 0)
    return true;
  llvm::DenseSet<Value> proofActive;
  llvm::DenseMap<Value, int64_t> assumed;
  return proveScalarAtLeast(operation.getOperand(0), 0, facts, operation,
                            proofActive, assumed);
}

struct CompactQuotientExpression {
  Operation *domain = nullptr;
  int64_t divisor = 1;
  int64_t offset = 0;
};

std::optional<CompactQuotientExpression>
compactQuotientExpression(Value value, const KernelFacts &facts,
                          Operation &consumer) {
  Operation *definition = value.getDefiningOp();
  while (definition &&
         (definition->getName().getStringRef() == "intent.broadcast" ||
          definition->getName().getStringRef() == "intent.reshape" ||
          definition->getName().getStringRef() == "intent.cast") &&
         definition->getNumOperands() == 1) {
    value = definition->getOperand(0);
    definition = value.getDefiningOp();
  }
  if (!definition || definition->getName().getStringRef() != "intent.binary" ||
      definition->getNumOperands() != 2)
    return std::nullopt;
  auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
  std::optional<int64_t> divisor =
      integerSplatConstant(definition->getOperand(1));
  if (!logical || logical.getValue() != "floor_divide" || !divisor ||
      *divisor <= 1 || !hasNonnegativeIntegerOperandsImpl(*definition, facts))
    return std::nullopt;

  llvm::DenseSet<Value> active;
  std::optional<AffineIndexExpression> dividend = affineIndexExpression(
      definition->getOperand(0), facts, consumer, active);
  if (!dividend || dividend->coefficients.size() != 1 ||
      dividend->constant < 0)
    return std::nullopt;
  auto source = dividend->coefficients.begin();
  if (!source->first || source->second != 1)
    return std::nullopt;
  return CompactQuotientExpression{source->first, *divisor,
                                   dividend->constant};
}

std::optional<std::pair<int64_t, int64_t>>
sequentialDomainRange(Operation *domain, const KernelFacts &facts,
                      Operation &consumer) {
  if (!domain)
    return std::nullopt;
  auto bounds = facts.staticDomainBounds.find(domain);
  if (bounds != facts.staticDomainBounds.end())
    return std::pair<int64_t, int64_t>{bounds->second.first,
                                       bounds->second.second - 1};
  if (!facts.runtimeSequentialDomains.contains(domain) ||
      domain->getNumOperands() < 2)
    return std::nullopt;

  llvm::DenseSet<Value> startActive;
  llvm::DenseSet<Value> stopActive;
  std::optional<AffineIndexExpression> start = affineIndexExpression(
      domain->getOperand(0), facts, consumer, startActive);
  std::optional<AffineIndexExpression> stop = affineIndexExpression(
      domain->getOperand(1), facts, consumer, stopActive);
  std::optional<std::pair<int64_t, int64_t>> startRange =
      start ? staticAffineRange(*start, facts) : std::nullopt;
  std::optional<std::pair<int64_t, int64_t>> stopRange =
      stop ? staticAffineRange(*stop, facts) : std::nullopt;
  int64_t upper;
  if (!startRange || !stopRange ||
      llvm::SubOverflow(stopRange->second, int64_t{1}, upper))
    return std::nullopt;
  return std::pair<int64_t, int64_t>{startRange->first, upper};
}

FailureOr<bool> requiresRuntimeBoundary(Operation &operation,
                                        KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice" || term.kind == "static_index") {
      ++sourceAxis;
      continue;
    }
    if (term.kind == "slice")
      return true;
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "boundary classification requires one dynamic index operand");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index") {
      if (hasInBoundsPrecondition(indexed, operation.getOperand(0), sourceAxis,
                                  operation)) {
        ++sourceAxis;
        continue;
      }
      bool tensorIndex = isa<RankedTensorType>(indexed.getType());
      llvm::DenseSet<Value> active;
      std::optional<AffineIndexExpression> expression =
          affineIndexExpression(indexed, facts, operation, active);
      FailureOr<LogicalAxis> staticDestination =
          axisFromView(operation.getOperand(0), sourceAxis, facts, operation);
      int64_t destinationExtent;
      std::optional<std::pair<int64_t, int64_t>> range =
          expression ? staticAffineRange(*expression, facts) : std::nullopt;
      if (succeeded(staticDestination) && range &&
          !StringRef(staticDestination->extent)
               .getAsInteger(10, destinationExtent)) {
        if (range->first < 0 || range->second >= destinationExtent)
          return true;
        ++sourceAxis;
        continue;
      }
      if (tensorIndex)
        return true;
      FailureOr<ScalarIndexSource> source =
          traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      if (source->domain &&
          facts.runtimeSequentialDomains.contains(source->domain)) {
        range = sequentialDomainRange(source->domain, facts, operation);
        if (!range || failed(staticDestination) ||
            StringRef(staticDestination->extent)
                .getAsInteger(10, destinationExtent))
          return true;
        if (range->first < 0 || range->second >= destinationExtent)
          return true;
        ++sourceAxis;
        continue;
      }
      if (source->opaque && !source->hasDomain()) {
        return operation.emitOpError(
            "opaque scalar index requires a preceding I.assume_in_bounds declaration");
      }
      if (source->transformed)
        return true;
      if (!source->hasDomain()) {
        ++sourceAxis;
        continue;
      }
      if (!source->domain) {
        ++sourceAxis;
        continue;
      }
      FailureOr<LogicalAxis> logical =
          axisFromDomain(*source->domain, facts, operation);
      FailureOr<LogicalAxis> destination =
          axisFromView(operation.getOperand(0), sourceAxis++, facts, operation);
      if (failed(logical) || failed(destination))
        return failure();
      if (logical->extent != destination->extent)
        return true;
      continue;
    }
    if (term.kind != "region_index")
      return operation.emitOpError("has an unsupported boundary index relation");
    FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
    FailureOr<LogicalAxis> logical =
        succeeded(domain) ? axisFromDomain(**domain, facts, operation)
                          : FailureOr<LogicalAxis>(failure());
    FailureOr<LogicalAxis> destination =
        axisFromView(operation.getOperand(0), sourceAxis++, facts, operation);
    if (failed(domain) || failed(logical) || failed(destination))
      return failure();
    if (logical->extent != destination->extent)
      return true;
  }
  return false;
}

LogicalResult bindRegionArgumentAxis(Operation &owner, unsigned argumentIndex,
                                     Operation &domain, KernelFacts &facts) {
  if (owner.getNumRegions() == 0 || owner.getRegion(0).empty() ||
      argumentIndex >= owner.getRegion(0).front().getNumArguments())
    return owner.emitOpError("has no canonical region argument");
  Value argument = owner.getRegion(0).front().getArgument(argumentIndex);
  FailureOr<int64_t> node = getValueID(
      argument, facts.kernel, owner, "region argument logical-axis binding");
  FailureOr<LogicalAxis> axis = axisFromDomain(domain, facts, owner);
  if (failed(node) || failed(axis))
    return owner.emitOpError("has no canonical region-axis identity");
  facts.axisLabels["?region_" + std::to_string(*node) + "_0"] = *axis;
  facts.regionArgumentAxes[argument] = *axis;
  return success();
}

FailureOr<SmallVector<StringRef>> resultShapeLabels(Operation &operation,
                                                    unsigned resultIndex) {
  auto shapes = operation.getAttrOfType<ArrayAttr>("intent.result_shapes");
  auto shape = shapes && resultIndex < shapes.size()
                   ? dyn_cast<ArrayAttr>(shapes[resultIndex])
                   : ArrayAttr();
  if (!shape)
    return SmallVector<StringRef>();
  SmallVector<StringRef> labels;
  for (Attribute attribute : shape) {
    auto label = dyn_cast<StringAttr>(attribute);
    if (!label) {
      operation.emitOpError("result shape contains a non-symbolic axis label");
      return failure();
    }
    labels.push_back(label.getValue());
  }
  return labels;
}

FailureOr<LogicalAxis> axisFromLabel(StringRef label, KernelFacts &facts,
                                     Operation &consumer) {
  if (label == "1")
    return LogicalAxis{nullptr, "1"};
  auto known = facts.axisLabels.find(label);
  if (known != facts.axisLabels.end())
    return known->second;
  consumer.emitOpError() << "has no exact logical-axis provenance for shape label '"
                         << label << "'";
  return failure();
}

LogicalResult bindResultAxes(Operation &operation, unsigned resultIndex,
                             SmallVector<LogicalAxis> axes,
                             KernelFacts &facts) {
  if (resultIndex >= operation.getNumResults())
    return operation.emitOpError("axis provenance references a missing result");
  auto tensor = dyn_cast<RankedTensorType>(operation.getResult(resultIndex).getType());
  if (!tensor)
    return axes.empty()
               ? success()
               : operation.emitOpError("scalar result cannot carry tensor axes");
  if (static_cast<size_t>(tensor.getRank()) != axes.size())
    return operation.emitOpError(
        "logical axis provenance does not match the tensor rank");
  FailureOr<SmallVector<StringRef>> labels =
      resultShapeLabels(operation, resultIndex);
  if (failed(labels) || (!labels->empty() && labels->size() != axes.size()))
    return operation.emitOpError("result shape labels do not match logical axes");
  for (auto [label, axis] : llvm::zip(*labels, axes))
    facts.axisLabels[label] = axis;
  facts.valueAxes[operation.getResult(resultIndex)] = std::move(axes);
  return success();
}

FailureOr<SmallVector<LogicalAxis>> axesFromResultShape(Operation &operation,
                                                       unsigned resultIndex,
                                                       KernelFacts &facts) {
  FailureOr<SmallVector<StringRef>> labels =
      resultShapeLabels(operation, resultIndex);
  if (failed(labels))
    return failure();
  SmallVector<LogicalAxis> axes;
  for (StringRef label : *labels) {
    FailureOr<LogicalAxis> axis = axisFromLabel(label, facts, operation);
    if (failed(axis))
      return failure();
    axes.push_back(*axis);
  }
  return axes;
}

LogicalResult recordLoadAxes(Operation &operation, KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned tensorIndexCount = llvm::count_if(*relation, [&](const IndexTerm &term) {
    return term.kind == "value_index" && term.operands.size() == 1 &&
           term.operands.front() &&
           isa<RankedTensorType>(
               operation.getOperand(*term.operands.front()).getType());
  });
  if (tensorIndexCount > 1) {
    FailureOr<SmallVector<LogicalAxis>> axes =
        axesFromResultShape(operation, 0, facts);
    if (failed(axes))
      return failure();
    return bindResultAxes(operation, 0, std::move(*axes), facts);
  }
  SmallVector<LogicalAxis> axes;
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      axes.push_back(LogicalAxis{nullptr, "1"});
      continue;
    }
    if (term.kind == "full_slice" || term.kind == "slice") {
      FailureOr<LogicalAxis> axis =
          axisFromView(operation.getOperand(0), sourceAxis++, facts, operation);
      if (failed(axis))
        return failure();
      axes.push_back(*axis);
      continue;
    }
    if (term.kind == "static_index") {
      ++sourceAxis;
      continue;
    }
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "indexed axis provenance requires one dynamic operand");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "region_index") {
      FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
      if (failed(domain))
        return failure();
      if (!isa<intent::LogicalIndexType, IntegerType, IndexType>(
              indexed.getType())) {
        FailureOr<LogicalAxis> axis = axisFromDomain(**domain, facts, operation);
        if (failed(axis))
          return failure();
        axes.push_back(*axis);
      }
    } else if (term.kind == "value_index") {
      auto indexedAxes = facts.valueAxes.find(indexed);
      if (indexedAxes != facts.valueAxes.end())
        axes.append(indexedAxes->second);
    } else {
      return operation.emitOpError("has an unsupported indexed axis relation");
    }
    ++sourceAxis;
  }
  return bindResultAxes(operation, 0, std::move(axes), facts);
}

FailureOr<SmallVector<LogicalAxis>>
inferIndexedAxes(Operation &operation, Value source, KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned tensorIndexCount = llvm::count_if(
      *relation, [&](const IndexTerm &term) {
        return term.kind == "value_index" && term.operands.size() == 1 &&
               term.operands.front() &&
               isa<RankedTensorType>(
                   operation.getOperand(*term.operands.front()).getType());
      });
  if (tensorIndexCount > 1) {
    SmallVector<LogicalAxis> resultAxes;
    SmallVector<LogicalAxis> advancedAxes;
    std::optional<size_t> advancedPosition;
    auto mergeAdvanced = [&](ArrayRef<LogicalAxis> incoming) -> LogicalResult {
      if (advancedAxes.empty()) {
        advancedAxes.assign(incoming.begin(), incoming.end());
        return success();
      }
      size_t rank = std::max(advancedAxes.size(), incoming.size());
      SmallVector<LogicalAxis> merged(rank, LogicalAxis{nullptr, "1"});
      for (size_t offset = 0; offset < rank; ++offset) {
        std::optional<LogicalAxis> lhs =
            offset < advancedAxes.size()
                ? std::optional<LogicalAxis>(
                      advancedAxes[advancedAxes.size() - 1 - offset])
                : std::nullopt;
        std::optional<LogicalAxis> rhs =
            offset < incoming.size()
                ? std::optional<LogicalAxis>(incoming[incoming.size() - 1 - offset])
                : std::nullopt;
        LogicalAxis selected = lhs ? *lhs : *rhs;
        if (lhs && rhs && *lhs != *rhs) {
          if (lhs->extent == "1")
            selected = *rhs;
          else if (rhs->extent != "1")
            return operation.emitOpError(
                "tensor index axes cannot broadcast to one logical relation");
        }
        merged[rank - 1 - offset] = selected;
      }
      advancedAxes = std::move(merged);
      return success();
    };
    auto sourceAxes = facts.valueAxes.find(source);
    bool sourceIsView = isa<intent::ViewType>(source.getType());
    unsigned sourceAxis = 0;
    for (const IndexTerm &term : *relation) {
      if (term.kind == "new_axis") {
        resultAxes.push_back(LogicalAxis{nullptr, "1"});
        continue;
      }
      if (term.kind == "full_slice" || term.kind == "slice") {
        FailureOr<LogicalAxis> axis =
            sourceIsView
                ? axisFromView(source, sourceAxis, facts, operation)
                : sourceAxes != facts.valueAxes.end() &&
                          sourceAxis < sourceAxes->second.size()
                      ? FailureOr<LogicalAxis>(sourceAxes->second[sourceAxis])
                      : FailureOr<LogicalAxis>(failure());
        if (failed(axis))
          return operation.emitOpError(
              "indexed source axis has no logical provenance");
        resultAxes.push_back(*axis);
        ++sourceAxis;
        continue;
      }
      if (term.kind == "static_index") {
        ++sourceAxis;
        continue;
      }
      if (term.operands.size() != 1 || !term.operands.front())
        return operation.emitOpError("index requires one dynamic operand");
      Value indexed = operation.getOperand(*term.operands.front());
      if (term.kind == "value_index" &&
          isa<RankedTensorType>(indexed.getType())) {
        auto indexedAxes = facts.valueAxes.find(indexed);
        if (indexedAxes == facts.valueAxes.end())
          return operation.emitOpError(
              "tensor index has no logical-axis provenance");
        if (!advancedPosition)
          advancedPosition = resultAxes.size();
        if (failed(mergeAdvanced(indexedAxes->second)))
          return failure();
      } else if (term.kind == "region_index") {
        FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
        if (failed(domain))
          return failure();
        if (!isa<intent::LogicalIndexType, IntegerType, IndexType>(
                indexed.getType())) {
          FailureOr<LogicalAxis> axis =
              axisFromDomain(**domain, facts, operation);
          if (failed(axis))
            return failure();
          resultAxes.push_back(*axis);
        }
      } else if (term.kind != "value_index") {
        return operation.emitOpError("has an unsupported index relation");
      }
      ++sourceAxis;
    }
    if (!advancedPosition)
      return operation.emitOpError(
          "multi-index relation has no tensor index provenance");
    resultAxes.insert(resultAxes.begin() + *advancedPosition,
                      advancedAxes.begin(), advancedAxes.end());
    return resultAxes;
  }
  SmallVector<LogicalAxis> resultAxes;
  auto sourceAxes = facts.valueAxes.find(source);
  bool sourceIsView = isa<intent::ViewType>(source.getType());
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis") {
      resultAxes.push_back(LogicalAxis{nullptr, "1"});
      continue;
    }
    if (term.kind == "full_slice" || term.kind == "slice") {
      if (sourceIsView) {
        FailureOr<LogicalAxis> axis =
            axisFromView(source, sourceAxis, facts, operation);
        if (failed(axis))
          return failure();
        resultAxes.push_back(*axis);
      } else {
        if (sourceAxes == facts.valueAxes.end() ||
            sourceAxis >= sourceAxes->second.size())
          return operation.emitOpError(
              "indexed source axis has no logical provenance");
        resultAxes.push_back(sourceAxes->second[sourceAxis]);
      }
      ++sourceAxis;
      continue;
    }
    if (term.kind == "static_index") {
      ++sourceAxis;
      continue;
    }
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError("index requires one dynamic operand");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "region_index") {
      FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
      if (failed(domain))
        return failure();
      if (!isa<intent::LogicalIndexType, IntegerType, IndexType>(
              indexed.getType())) {
        FailureOr<LogicalAxis> axis =
            axisFromDomain(**domain, facts, operation);
        if (failed(axis))
          return failure();
        resultAxes.push_back(*axis);
      }
    } else if (term.kind == "value_index") {
      auto indexedAxes = facts.valueAxes.find(indexed);
      if (indexedAxes != facts.valueAxes.end())
        resultAxes.append(indexedAxes->second);
    } else {
      return operation.emitOpError("has an unsupported index relation");
    }
    ++sourceAxis;
  }
  return resultAxes;
}

LogicalResult recordAccessRanges(Operation &operation, KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  auto isPartitionedOwnership = [&](Operation *domain) {
    return llvm::any_of(facts.partitionDomains, [&](const auto &entry) {
      return entry.second == domain;
    });
  };
  auto hasSelectedPhysicalRange = [&](Operation *domain) {
    return isPartitionedOwnership(domain) ||
           facts.orderedDomains.contains(domain) ||
           facts.contractionDomains.contains(domain);
  };
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    unsigned currentSourceAxis = sourceAxis++;
    if (term.kind != "value_index" || term.operands.size() != 1 ||
        !term.operands.front())
      continue;
    Value index = operation.getOperand(*term.operands.front());
    if (!isa<RankedTensorType>(index.getType()))
      continue;
    if (std::optional<CompactQuotientExpression> compact =
            compactQuotientExpression(index, facts, operation)) {
      if (hasSelectedPhysicalRange(compact->domain))
        facts.accessRanges.push_back(AccessRangeFact{
            &operation, compact->domain, currentSourceAxis,
            compact->divisor, compact->offset});
      continue;
    }
    llvm::DenseSet<Value> active;
    std::optional<AffineIndexExpression> expression =
        affineIndexExpression(index, facts, operation, active);
    if (!expression) {
      continue;
    }
    if (expression->constant > 0 && expression->coefficients.size() == 1) {
      const auto &source = *expression->coefficients.begin();
      if (source.first && source.second == 1) {
        facts.accessRanges.push_back(AccessRangeFact{
            &operation, source.first, currentSourceAxis, 1,
            expression->constant});
        continue;
      }
    }
    Operation *ownership = nullptr;
    for (const auto &coefficient : expression->coefficients) {
      if (!isPartitionedOwnership(coefficient.first))
        continue;
      if (ownership || coefficient.second != 1) {
        ownership = nullptr;
        break;
      }
      ownership = coefficient.first;
    }
    if (!ownership) {
      continue;
    }
    int64_t lower = expression->constant;
    int64_t upper = expression->constant;
    bool bounded = true;
    for (const auto &coefficient : expression->coefficients) {
      if (coefficient.first == ownership)
        continue;
      auto extent = facts.staticDomainExtents.find(coefficient.first);
      if (extent == facts.staticDomainExtents.end()) {
        bounded = false;
        break;
      }
      int64_t endpoint = coefficient.second * (extent->second - 1);
      lower += std::min<int64_t>(0, endpoint);
      upper += std::max<int64_t>(0, endpoint);
    }
    if (!bounded || (lower == 0 && upper == 0)) {
      continue;
    }
    facts.accessRanges.push_back(
        AccessRangeFact{&operation, ownership, currentSourceAxis, 1, 0});
  }
  return success();
}

struct StructuredTensorIndex {
  bool valid = false;
  bool dependsOnTensorAxis = false;
  bool compact = false;
};

StructuredTensorIndex classifyStructuredIndex(Value value,
                                               const KernelFacts &facts,
                                               llvm::DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return {};
  auto finish = [&](StructuredTensorIndex result) {
    active.erase(value);
    return result;
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *owner = argument.getOwner()->getParentOp();
    StringRef name = owner ? owner->getName().getStringRef() : StringRef();
    bool structural = name == "intent.parallel" || name == "intent.ordered" ||
                      name == "intent.for" || name == "intent.state_stream";
    return finish({structural, false, false});
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return finish({});
  StringRef name = definition->getName().getStringRef();
  if (name == "intent.constant" || name == "intent.dim" ||
      name == "intent.region_end")
    return finish({true, false, false});
  if (name == "intent.indices") {
    auto axes = facts.valueAxes.find(value);
    bool valid = axes != facts.valueAxes.end() &&
                 llvm::count_if(axes->second, [](const LogicalAxis &axis) {
                   return axis.extent != "1" && axis.domain;
                 }) == 1;
    return finish({valid, valid, false});
  }
  if (name == "intent.gather") {
    FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(*definition);
    if (failed(relation) ||
        !llvm::all_of(*relation, [](const IndexTerm &term) {
          return term.kind == "full_slice" || term.kind == "new_axis";
        }) || definition->getNumOperands() == 0)
      return finish({});
    return finish(classifyStructuredIndex(definition->getOperand(0), facts,
                                          active));
  }
  if ((name == "intent.broadcast" || name == "intent.reshape" ||
       name == "intent.cast") &&
      definition->getNumOperands() == 1)
    return finish(classifyStructuredIndex(definition->getOperand(0), facts,
                                          active));
  if (name == "intent.unary" && definition->getNumOperands() == 1) {
    auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
    StructuredTensorIndex operand =
        classifyStructuredIndex(definition->getOperand(0), facts, active);
    if (!logical || !operand.valid ||
        (operand.dependsOnTensorAxis && logical.getValue() != "negate"))
      return finish({});
    return finish(operand);
  }
  if (name != "intent.binary" || definition->getNumOperands() != 2)
    return finish({});
  StructuredTensorIndex lhs =
      classifyStructuredIndex(definition->getOperand(0), facts, active);
  StructuredTensorIndex rhs =
      classifyStructuredIndex(definition->getOperand(1), facts, active);
  auto logical = definition->getAttrOfType<StringAttr>("intent.operator");
  if (!lhs.valid || !rhs.valid || !logical)
    return finish({});
  bool tensorAxis = lhs.dependsOnTensorAxis || rhs.dependsOnTensorAxis;
  if (!tensorAxis)
    return finish({true, false, false});
  if (lhs.dependsOnTensorAxis && rhs.dependsOnTensorAxis)
    return finish({});
  std::optional<int64_t> divisor =
      integerSplatConstant(definition->getOperand(1));
  if (logical.getValue() == "floor_divide" && lhs.dependsOnTensorAxis &&
      !rhs.dependsOnTensorAxis && divisor && *divisor > 1 &&
      hasNonnegativeIntegerOperandsImpl(*definition, facts))
    return finish({true, true, true});
  if (logical.getValue() != "add" && logical.getValue() != "subtract")
    return finish({});
  return finish({true, true, lhs.compact || rhs.compact});
}

LogicalResult classifyTensorIndices(Operation &operation, KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  bool hasTensorIndex = false;
  bool dataDependent = false;
  bool compact = false;
  unsigned tensorIndexCount = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind != "value_index" || term.operands.size() != 1 ||
        !term.operands.front())
      continue;
    Value indexed = operation.getOperand(*term.operands.front());
    if (!isa<RankedTensorType>(indexed.getType()))
      continue;
    hasTensorIndex = true;
    ++tensorIndexCount;
    llvm::DenseSet<Value> active;
    StructuredTensorIndex structured =
        classifyStructuredIndex(indexed, facts, active);
    if (!structured.valid || !structured.dependsOnTensorAxis) {
      dataDependent = true;
      continue;
    }
    compact |= structured.compact;
    auto axes = facts.valueAxes.find(indexed);
    if (axes == facts.valueAxes.end() ||
        llvm::count_if(axes->second, [](const LogicalAxis &axis) {
          return axis.extent != "1" && axis.domain;
        }) != 1)
      dataDependent = true;
  }
  if (compact && tensorIndexCount != 1)
    dataDependent = true;
  facts.tensorIndexing[&operation] =
      dataDependent ? TensorIndexingKind::dataDependent
      : compact ? TensorIndexingKind::compact
      : hasTensorIndex ? TensorIndexingKind::structured
                       : TensorIndexingKind::none;
  return success();
}

LogicalResult propagatePointwiseAxes(Operation &operation, KernelFacts &facts) {
  if (operation.getNumResults() != 1)
    return operation.emitOpError("pointwise provenance requires one result");
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  if (!result)
    return success();
  FailureOr<SmallVector<StringRef>> labels = resultShapeLabels(operation, 0);
  if (failed(labels))
    return failure();
  SmallVector<LogicalAxis> axes(result.getRank(), LogicalAxis{nullptr, "1"});
  for (Value operand : operation.getOperands()) {
    auto source = facts.valueAxes.find(operand);
    if (source == facts.valueAxes.end())
      continue;
    if (source->second.size() > axes.size())
      return operation.emitOpError(
          "pointwise operand provenance rank exceeds the result rank");
    size_t offset = axes.size() - source->second.size();
    for (auto [position, axis] : llvm::enumerate(source->second)) {
      LogicalAxis &destination = axes[offset + position];
      if (destination.extent == "1")
        destination = axis;
      else if (axis.extent != "1" && destination != axis) {
        bool sameExtent = destination.extent == axis.extent;
        bool implicitAlias = sameExtent &&
                             (!destination.domain || !axis.domain);
        if (!implicitAlias)
          return operation.emitOpError(
              "pointwise operands have incompatible logical-axis provenance");
        destination = mergeLogicalAxis(destination, axis);
      }
    }
  }
  if (labels->size() != axes.size())
    return operation.emitOpError(
        "pointwise result shape does not match its tensor rank");
  for (auto [position, label] : llvm::enumerate(*labels))
    if (axes[position].extent == "1" && label != "1") {
      FailureOr<LogicalAxis> fallback = axisFromLabel(label, facts, operation);
      if (failed(fallback))
        return failure();
      axes[position] = *fallback;
    }
  return bindResultAxes(operation, 0, std::move(axes), facts);
}

bool isUnitAxis(const LogicalAxis &axis, const KernelFacts &facts) {
  if (axis.extent == "1")
    return true;
  if (!axis.domain)
    return false;
  return llvm::any_of(facts.stateStreams, [&](const auto &entry) {
    Operation *stream = entry.first;
    if (stream->getNumOperands() == 0 ||
        stream->getOperand(0).getDefiningOp() != axis.domain)
      return false;
    auto extentIndex =
        stream->getAttrOfType<IntegerAttr>("intent.extent_operand_index");
    if (!extentIndex || extentIndex.getInt() < 0 ||
        static_cast<unsigned>(extentIndex.getInt()) >= stream->getNumOperands())
      return false;
    Operation *extent =
        stream->getOperand(extentIndex.getInt()).getDefiningOp();
    auto value = extent
                     ? extent->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
    return extent && extent->getName().getStringRef() == "intent.constant" &&
           value && value.getInt() == 1;
  });
}

std::optional<uint64_t> staticAxisExtent(const LogicalAxis &axis) {
  uint64_t value = 0;
  if (StringRef(axis.extent).getAsInteger(10, value) || value == 0)
    return std::nullopt;
  return value;
}

bool multiplyExtent(uint64_t &product, uint64_t extent) {
  if (product > std::numeric_limits<uint64_t>::max() / extent)
    return false;
  product *= extent;
  return true;
}

LogicalResult propagateReshapeAxes(Operation &operation, KernelFacts &facts) {
  if (operation.getNumOperands() != 1 || operation.getNumResults() != 1)
    return operation.emitOpError("has no canonical reshape schema");
  auto source = facts.valueAxes.find(operation.getOperand(0));
  FailureOr<SmallVector<LogicalAxis>> result =
      axesFromResultShape(operation, 0, facts);
  if (source == facts.valueAxes.end() || failed(result))
    return operation.emitOpError("reshape has no logical-axis provenance");

  size_t sourceIndex = 0, resultIndex = 0;
  while (sourceIndex < source->second.size() && resultIndex < result->size()) {
    if (compatibleLogicalAxis(source->second[sourceIndex],
                              (*result)[resultIndex])) {
      (*result)[resultIndex] = mergeLogicalAxis(
          source->second[sourceIndex], (*result)[resultIndex]);
      ++sourceIndex;
      ++resultIndex;
      continue;
    }
    if (isUnitAxis(source->second[sourceIndex], facts)) {
      ++sourceIndex;
      continue;
    }
    if (isUnitAxis((*result)[resultIndex], facts)) {
      ++resultIndex;
      continue;
    }
    std::optional<uint64_t> sourceProduct =
        staticAxisExtent(source->second[sourceIndex++]);
    std::optional<uint64_t> resultProduct =
        staticAxisExtent((*result)[resultIndex++]);
    if (!sourceProduct || !resultProduct)
      return operation.emitOpError(
          "reshape changes a symbolic logical-axis group");
    while (*sourceProduct != *resultProduct) {
      if (*sourceProduct < *resultProduct) {
        if (sourceIndex == source->second.size())
          return operation.emitOpError(
              "reshape changes the logical element count");
        std::optional<uint64_t> extent =
            staticAxisExtent(source->second[sourceIndex++]);
        if (!extent || !multiplyExtent(*sourceProduct, *extent))
          return operation.emitOpError(
              "reshape has an unsupported static source extent group");
      } else {
        if (resultIndex == result->size())
          return operation.emitOpError(
              "reshape changes the logical element count");
        std::optional<uint64_t> extent =
            staticAxisExtent((*result)[resultIndex++]);
        if (!extent || !multiplyExtent(*resultProduct, *extent))
          return operation.emitOpError(
              "reshape has an unsupported static result extent group");
      }
    }
  }
  while (sourceIndex < source->second.size() &&
         isUnitAxis(source->second[sourceIndex], facts))
    ++sourceIndex;
  while (resultIndex < result->size() &&
         isUnitAxis((*result)[resultIndex], facts))
    ++resultIndex;
  if (sourceIndex != source->second.size() || resultIndex != result->size())
    return operation.emitOpError("reshape changes the logical element count");
  return bindResultAxes(operation, 0, std::move(*result), facts);
}

LogicalResult registerFactHandlers(OperationHandlerRegistry &registry,
                                   KernelFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.yield",
                         "intent.condition", "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.assume_in_bounds",
          [&](Operation &operation) -> LogicalResult {
            auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
            Type indexType = operation.getNumOperands() > 0
                                 ? operation.getOperand(0).getType()
                                 : Type();
            auto tensorIndex = dyn_cast_or_null<RankedTensorType>(indexType);
            bool integerIndex =
                isa<IntegerType, IndexType, intent::LogicalIndexType>(indexType) ||
                (tensorIndex &&
                 isa<IntegerType, IndexType>(tensorIndex.getElementType()));
            if (operation.getNumOperands() != 2 || operation.getNumResults() != 0 ||
                !axis || axis.getInt() < 0 || !integerIndex)
              return operation.emitOpError(
                  "has no canonical in-bounds precondition schema");
            Type targetType = operation.getOperand(1).getType();
            if (auto view = dyn_cast<intent::ViewType>(targetType)) {
              auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
              if (!tensor || axis.getInt() >= tensor.getRank())
                return operation.emitOpError(
                    "in-bounds precondition axis exceeds its view rank");
              return success();
            }
            Operation *buffer = operation.getOperand(1).getDefiningOp();
            FailureOr<LogicalBufferInfo> info = buffer
                                                    ? getLogicalBufferInfo(*buffer)
                                                    : FailureOr<LogicalBufferInfo>(
                                                          failure());
            if (!isa<intent::BufferType>(targetType) || failed(info) ||
                axis.getInt() >= static_cast<int64_t>(info->shape.size()))
              return operation.emitOpError(
                  "in-bounds precondition axis exceeds its logical-buffer rank");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.make_record",
          [&](Operation &operation) -> LogicalResult {
            auto fields = operation.getAttrOfType<ArrayAttr>("intent.fields");
            llvm::StringSet<> names;
            if (operation.getNumResults() != 1 ||
                !isa<intent::RecordType>(operation.getResult(0).getType()) ||
                !fields || fields.empty() ||
                fields.size() != operation.getNumOperands())
              return operation.emitOpError("has no canonical record schema");
            for (Attribute attribute : fields) {
              auto name = dyn_cast<StringAttr>(attribute);
              if (!name || name.getValue().empty() ||
                  !names.insert(name.getValue()).second)
                return operation.emitOpError(
                    "record fields must be unique non-empty names");
            }
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.extract",
          [&](Operation &operation) -> LogicalResult {
            FailureOr<Value> field = resolveRecordField(operation);
            if (failed(field))
              return failure();
            if (!isa<RankedTensorType>(field->getType()))
              return success();
            auto axes = facts.valueAxes.find(*field);
            if (axes == facts.valueAxes.end())
              return operation.emitOpError(
                  "tensor record field has no logical-axis provenance");
            return bindResultAxes(operation, 0, axes->second, facts);
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.domain", [&](Operation &operation) -> LogicalResult {
            if ((operation.getNumOperands() != 2 &&
                 operation.getNumOperands() != 3) ||
                operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical domain schema");
            auto bindDomainResultAxis = [&]() -> LogicalResult {
              FailureOr<int64_t> node = getValueID(
                  operation.getResult(0), facts.kernel, operation,
                  "domain result logical-axis binding");
              FailureOr<LogicalAxis> logical =
                  axisFromDomain(operation, facts, operation);
              if (failed(node) || failed(logical))
                return failure();
              facts.axisLabels["?region_" + std::to_string(*node) + "_0"] =
                  *logical;
              return success();
            };
            Operation *start = operation.getOperand(0).getDefiningOp();
            Operation *stop = operation.getOperand(1).getDefiningOp();
            auto startValue =
                start ? start->getAttrOfType<IntegerAttr>("intent.value")
                      : IntegerAttr();
            auto axis = stop ? stop->getAttrOfType<IntegerAttr>("intent.axis")
                             : IntegerAttr();
            Operation *step = operation.getNumOperands() == 3
                                  ? operation.getOperand(2).getDefiningOp()
                                  : nullptr;
            auto stepValue =
                step ? step->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
            if (step && (step->getName().getStringRef() != "intent.constant" ||
                         !stepValue || stepValue.getInt() != 1))
              return operation.emitOpError(
                  "runtime sequential domains currently require unit step");
            auto stopValue =
                stop ? stop->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
            bool staticStart =
                start && start->getName().getStringRef() == "intent.constant" &&
                startValue;
            if (staticStart && stop &&
                stop->getName().getStringRef() == "intent.constant" &&
                stopValue && stopValue.getInt() > startValue.getInt()) {
              int64_t extent;
              if (llvm::SubOverflow(stopValue.getInt(), startValue.getInt(),
                                    extent) ||
                  extent <= 0)
                return operation.emitOpError("has an overflowing static domain");
              facts.staticDomainExtents[&operation] = extent;
              facts.staticDomainBounds[&operation] =
                  {startValue.getInt(), stopValue.getInt()};
              return bindDomainResultAxis();
            }
            bool zeroBased = staticStart && startValue.getInt() == 0;
            if (zeroBased && stop &&
                stop->getName().getStringRef() == "intent.dim" && axis &&
                stop->getNumOperands() == 1) {
              facts.domainSources[&operation] = stop->getOperand(0);
              facts.domainSourceAxes[&operation] = axis.getInt();
              return bindDomainResultAxis();
            }
            auto validLoopBound = [](Type type) {
              return type.isIntOrIndex() || isa<intent::LogicalIndexType>(type);
            };
            if (!validLoopBound(operation.getOperand(0).getType()) ||
                !validLoopBound(operation.getOperand(1).getType()) ||
                !llvm::all_of(operation.getResult(0).getUsers(), [](Operation *user) {
                  return user->getName().getStringRef() == "intent.for";
                }))
              return operation.emitOpError(
                  "runtime-bounded domains require an ordinary scalar sequential loop");
            facts.runtimeSequentialDomains.insert(&operation);
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.domain_product",
          [&](Operation &operation) -> LogicalResult {
            auto type = operation.getNumResults() == 1
                            ? dyn_cast<intent::DomainType>(
                                  operation.getResult(0).getType())
                            : intent::DomainType();
            if (operation.getNumOperands() == 0 || !type)
              return operation.emitOpError(
                  "has no canonical domain-product schema");
            FailureOr<SmallVector<Operation *>> domains =
                expandDomainSource(operation.getResult(0), operation);
            if (failed(domains) || domains->size() != operation.getNumOperands() ||
                type.getSpec() !=
                    ("domain<product," + std::to_string(domains->size()) + ">"))
              return operation.emitOpError(
                  "domain product requires direct rank-one logical domains");
            for (Operation *domain : *domains)
              if (!facts.domainSourceAxes.count(domain) &&
                  !facts.staticDomainExtents.count(domain))
                return operation.emitOpError(
                    "domain product contains an unrealized logical domain");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.partition", [&](Operation &operation) -> LogicalResult {
            if ((operation.getNumOperands() != 1 &&
                 operation.getNumOperands() != 2) ||
                operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical partition schema");
            Operation *domain = operation.getOperand(0).getDefiningOp();
            if (!domain || (!facts.domainSourceAxes.count(domain) &&
                            !facts.staticDomainExtents.count(domain)))
              return operation.emitOpError(
                  "tiled partitions currently require a source domain");
            auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
            auto extent =
                operation.getAttrOfType<DictionaryAttr>("intent.extent");
            auto name = extent ? extent.getAs<StringAttr>("name") : StringAttr();
            bool namedAuto = name && !name.getValue().empty();
            bool fixedExtent = operation.getNumOperands() == 2 &&
                               integerConstant(operation.getOperand(1));
            if (!mode || mode.getValue() != "extent" ||
                (!namedAuto && !fixedExtent))
              return operation.emitOpError(
                  "tiled partitions require a named auto or fixed extent");
            facts.partitionDomains[&operation] = domain;
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.parallel", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                operation.getNumRegions() != 1 ||
                !llvm::hasSingleElement(operation.getRegion(0)) ||
                operation.getNumResults() != 0)
              return operation.emitOpError(
                  "parallel ownership requires one stateless region");
            FailureOr<SmallVector<Operation *>> domains =
                expandDomainSource(operation.getOperand(0), operation);
            if (failed(domains) || domains->empty() ||
                operation.getRegion(0).front().getNumArguments() !=
                    domains->size())
              return operation.emitOpError(
                  "parallel ownership must match its logical source axes");
            for (auto [index, domain] : llvm::enumerate(*domains)) {
              if (!facts.domainSourceAxes.count(domain) &&
                  !facts.staticDomainExtents.count(domain))
                return operation.emitOpError(
                    "parallel ownership requires canonical logical domains");
              if (failed(bindRegionArgumentAxis(operation, index, *domain, facts)))
                return failure();
            }
            facts.parallels.push_back(&operation);
            facts.parallelDomains[&operation] = std::move(*domains);
            return success();
          })))
    return failure();

  auto analyzeSequentialLoop = [&](Operation &operation) -> LogicalResult {
            bool ordered =
                operation.getName().getStringRef() == "intent.ordered";
            FailureOr<SmallVector<Operation *>> domains =
                operation.getNumOperands() > 0
                    ? expandDomainSource(operation.getOperand(0), operation)
                    : FailureOr<SmallVector<Operation *>>(failure());
            Operation *source = operation.getNumOperands() > 0
                                    ? operation.getOperand(0).getDefiningOp()
                                    : nullptr;
            if (failed(domains) || domains->empty() ||
                (!ordered &&
                 (domains->size() != 1 || !source ||
                  source->getName().getStringRef() != "intent.domain")) ||
                operation.getNumRegions() != 1 ||
                !llvm::hasSingleElement(operation.getRegion(0)) ||
                operation.getNumOperands() != operation.getNumResults() + 1 ||
                operation.getRegion(0).front().getNumArguments() !=
                    operation.getNumResults() + domains->size())
              return operation.emitOpError(
                  "has no canonical sequential-loop schema");
            for (auto [index, domain] : llvm::enumerate(*domains)) {
              bool runtime = facts.runtimeSequentialDomains.contains(domain);
              if ((!facts.domainSourceAxes.count(domain) &&
                   !facts.staticDomainExtents.count(domain) && !runtime) ||
                  !isa<intent::LogicalIndexType>(
                      operation.getRegion(0).front().getArgument(index).getType()) ||
                  (!runtime &&
                   failed(bindRegionArgumentAxis(operation, index, *domain, facts))))
                return operation.emitOpError(
                    "sequential loop has an unrealized logical axis");
            }
            Operation &terminator = operation.getRegion(0).front().back();
            if (terminator.getName().getStringRef() != "intent.yield" ||
                terminator.getNumOperands() != operation.getNumResults())
              return operation.emitOpError(
                  "sequential loop must yield every carried value");
            for (unsigned index = 0; index < operation.getNumResults(); ++index) {
              Value initialValue = operation.getOperand(index + 1);
              Type initial = initialValue.getType();
              Type argument = operation.getRegion(0).front()
                                  .getArgument(domains->size() + index)
                                  .getType();
              Type result = operation.getResult(index).getType();
              if ((!initial.isIntOrIndexOrFloat() &&
                   !isa<RankedTensorType>(initial)) ||
                  initial != argument ||
                  initial != result ||
                  terminator.getOperand(index).getType() != result)
                return operation.emitOpError(
                    "sequential loop requires scalar or tensor type-stable carried values");
              if (isa<RankedTensorType>(initial)) {
                auto axes = facts.valueAxes.find(initialValue);
                if (axes == facts.valueAxes.end())
                  return operation.emitOpError(
                      "tensor loop state has no logical-axis provenance");
                facts.valueAxes[operation.getRegion(0).front().getArgument(
                    domains->size() + index)] = axes->second;
                facts.valueAxes[operation.getResult(index)] = axes->second;
              }
            }
            if (ordered)
              for (Operation *domain : *domains) {
                facts.orderedDomains.insert(domain);
                facts.serialLoopDomains.insert(domain);
              }
            return success();
          };
  auto leaveSequentialLoop = [&](Operation &operation) -> LogicalResult {
    FailureOr<SmallVector<Operation *>> domains =
        expandDomainSource(operation.getOperand(0), operation);
    if (failed(domains))
      return failure();
    Operation &terminator = operation.getRegion(0).front().back();
    for (unsigned index = 0; index < operation.getNumResults(); ++index) {
      Value initial = operation.getOperand(index + 1);
      if (!isa<RankedTensorType>(initial.getType()))
        continue;
      Value yielded = terminator.getOperand(index);
      auto initialAxes = facts.valueAxes.find(initial);
      auto yieldedAxes = facts.valueAxes.find(yielded);
      SmallVector<LogicalAxis> merged;
      if (initialAxes == facts.valueAxes.end() ||
          yieldedAxes == facts.valueAxes.end() ||
          !mergeLogicalAxes(initialAxes->second, yieldedAxes->second, merged))
        return operation.emitOpError(
            "yielded tensor loop state changes its logical axis identity");
      facts.valueAxes[operation.getResult(index)] = std::move(merged);
    }
    return success();
  };
  OperationHandler sequentialLoopHandler{analyzeSequentialLoop,
                                         leaveSequentialLoop};
  if (failed(registry.add("intent.for", sequentialLoopHandler)) ||
      failed(registry.add("intent.ordered", sequentialLoopHandler)))
    return failure();

  if (failed(addHandler(
          registry, "intent.if", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                !operation.getOperand(0).getType().isInteger(1) ||
                operation.getNumRegions() != 2)
              return operation.emitOpError("has no canonical scalar-if schema");
            for (Region &branch : operation.getRegions()) {
              if (!llvm::hasSingleElement(branch) || branch.front().empty() ||
                  branch.front().getNumArguments() != 0)
                return operation.emitOpError(
                    "scalar if requires two single-block branches");
              Operation &terminator = branch.front().back();
              if (terminator.getName().getStringRef() != "intent.yield" ||
                  terminator.getNumOperands() != operation.getNumResults())
                return operation.emitOpError(
                    "scalar if branches must yield every result");
              for (auto [yielded, result] :
                   llvm::zip(terminator.getOperands(), operation.getResults()))
                if (!result.getType().isIntOrIndexOrFloat() ||
                    yielded.getType() != result.getType())
                  return operation.emitOpError(
                      "scalar if currently requires scalar type-stable results");
            }
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.while", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumRegions() != 2 ||
                !llvm::hasSingleElement(operation.getRegion(0)) ||
                !llvm::hasSingleElement(operation.getRegion(1)) ||
                operation.getNumOperands() != operation.getNumResults())
              return operation.emitOpError(
                  "has no canonical scalar while schema");
            Block &before = operation.getRegion(0).front();
            Block &after = operation.getRegion(1).front();
            if (before.getNumArguments() != operation.getNumResults() ||
                after.getNumArguments() != operation.getNumResults() ||
                before.empty() || after.empty())
              return operation.emitOpError(
                  "while regions do not match their carried state");
            for (unsigned index = 0; index < operation.getNumResults(); ++index) {
              Type type = operation.getOperand(index).getType();
              if (!type.isIntOrIndexOrFloat() ||
                  operation.getResult(index).getType() != type ||
                  before.getArgument(index).getType() != type ||
                  after.getArgument(index).getType() != type)
                return operation.emitOpError(
                    "while requires scalar type-stable carried values");
            }
            Operation &condition = before.back();
            if (condition.getName().getStringRef() != "intent.condition" ||
                condition.getNumOperands() != operation.getNumResults() + 1 ||
                !condition.getOperand(0).getType().isInteger(1))
              return operation.emitOpError(
                  "while before-region must end in one scalar condition");
            for (unsigned index = 0; index < operation.getNumResults(); ++index)
              if (condition.getOperand(index + 1) != before.getArgument(index))
                return operation.emitOpError(
                    "while condition must forward every carried value unchanged");
            Operation &yield = after.back();
            if (yield.getName().getStringRef() != "intent.yield" ||
                yield.getNumOperands() != operation.getNumResults())
              return operation.emitOpError(
                  "while body must yield every carried value");
            for (auto [yielded, result] :
                 llvm::zip(yield.getOperands(), operation.getResults()))
              if (yielded.getType() != result.getType())
                return operation.emitOpError(
                    "while body changes a carried value type");
            for (Operation &nested : before.without_terminator()) {
              StringRef name = nested.getName().getStringRef();
              if (!llvm::is_contained(
                      {StringRef("intent.constant"), StringRef("intent.dim"),
                       StringRef("intent.make_record"),
                       StringRef("intent.extract"), StringRef("intent.unary"),
                       StringRef("intent.binary"), StringRef("intent.compare"),
                       StringRef("intent.select"), StringRef("intent.cast")},
                      name))
                return nested.emitOpError(
                    "cannot be inlined into a scalar while condition");
              auto logical = nested.getAttrOfType<StringAttr>("intent.operator");
              if (name == "intent.binary" && logical &&
                  (logical.getValue() == "floor_divide" ||
                   logical.getValue() == "remainder") &&
                  !hasNonnegativeIntegerOperandsImpl(nested, facts))
                return nested.emitOpError(
                    "requires multi-statement lowering in a while condition");
              for (Value result : nested.getResults())
                if (!result.getType().isIntOrIndexOrFloat() &&
                    !isa<intent::RecordType>(result.getType()))
                  return nested.emitOpError(
                      "while condition values must remain scalar");
            }
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.buffer", [&](Operation &operation) -> LogicalResult {
            FailureOr<LogicalBufferInfo> info = getLogicalBufferInfo(operation);
            Operation *owner = nearestParallelOwner(operation);
            if (failed(info) || !owner || operation.getParentOp() != owner)
              return operation.emitOpError(
                  "private logical buffer must be a direct child of one parallel owner");
            facts.logicalBuffers[&operation] =
                LogicalBufferFact{owner, std::move(*info)};
            return success();
          })))
    return failure();

  auto validateBufferAccess = [&](Operation &operation) -> LogicalResult {
    Operation *buffer = operation.getNumOperands() > 0
                            ? operation.getOperand(0).getDefiningOp()
                            : nullptr;
    auto found = facts.logicalBuffers.find(buffer);
    FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
    if (found == facts.logicalBuffers.end() || failed(relation) ||
        relation->size() != found->second.info.shape.size() ||
        nearestParallelOwner(operation) != found->second.owner)
      return operation.emitOpError(
          "private logical buffer access rank does not match its owner buffer");
    for (auto [axis, term] : llvm::enumerate(*relation)) {
      int64_t extent = found->second.info.shape[axis];
      if (term.kind == "value_index") {
        if (term.operands.size() != 1 || !term.operands.front())
          return operation.emitOpError(
              "private logical buffer dynamic index is not canonical");
        found->second.hasDynamicAccess = true;
        unsigned operandIndex = *term.operands.front();
        if (operandIndex >= operation.getNumOperands())
          return operation.emitOpError(
              "private logical buffer index references a missing operand");
        Value indexed = operation.getOperand(operandIndex);
        auto iterator = dyn_cast<BlockArgument>(indexed);
        Operation *loop = iterator ? iterator.getOwner()->getParentOp() : nullptr;
        Operation *domain =
            loop && loop->getName().getStringRef() == "intent.for" &&
                    iterator.getArgNumber() == 0 && loop->getNumOperands() > 0
                ? loop->getOperand(0).getDefiningOp()
                : nullptr;
        std::optional<std::pair<int64_t, int64_t>> iteratorRange =
            sequentialDomainRange(domain, facts, operation);
        bool boundedIterator = iteratorRange && iteratorRange->first >= 0 &&
                               iteratorRange->second < extent;
        llvm::DenseSet<Value> active;
        std::optional<AffineIndexExpression> expression =
            affineIndexExpression(indexed, facts, operation, active);
        std::optional<std::pair<int64_t, int64_t>> range =
            expression ? staticAffineRange(*expression, facts) : std::nullopt;
        bool boundedExpression =
            range && range->first >= 0 && range->second < extent;
        if (!boundedIterator && !boundedExpression &&
            !hasInBoundsPrecondition(indexed, operation.getOperand(0), axis,
                                     operation))
          return operation.emitOpError()
                 << "private logical buffer dynamic index for axis " << axis
                 << " requires a bounded iterator or preceding in-bounds declaration";
        if (!boundedIterator && !boundedExpression)
          found->second.hasUnstructuredDynamicAccess = true;
        continue;
      }
      if (term.kind != "static_index" || term.staticValues.size() != 1 ||
          !term.staticValues.front() || *term.staticValues.front() < 0 ||
          *term.staticValues.front() >= extent)
        return operation.emitOpError()
               << "private logical buffer static index for axis " << axis
               << " is outside its extent";
    }
    bool load = operation.getName().getStringRef() == "intent.buffer_load";
    if (load &&
        (operation.getNumResults() != 1 ||
         operation.getResult(0).getType() != found->second.info.elementType))
      return operation.emitOpError(
          "private logical buffer load must produce one matching scalar");
    auto valueIndex =
        operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
    if (!load &&
        (operation.getNumResults() != 0 || !valueIndex ||
         valueIndex.getInt() != 1 || operation.getNumOperands() < 2 ||
         operation.getOperand(1).getType() != found->second.info.elementType))
      return operation.emitOpError(
          "private logical buffer store must consume one matching scalar value");
    return success();
  };
  if (failed(addHandler(registry, "intent.buffer_load", validateBufferAccess)) ||
      failed(addHandler(registry, "intent.buffer_store", validateBufferAccess)))
    return failure();

  if (failed(addHandler(
          registry, "intent.ragged", [&](Operation &operation) -> LogicalResult {
            if ((operation.getNumOperands() != 3 &&
                 operation.getNumOperands() != 4) ||
                operation.getNumResults() != 1 ||
                !isa<intent::RaggedType>(operation.getResult(0).getType()))
              return operation.emitOpError("has no canonical ragged schema");
            Operation *outer = operation.getOperand(0).getDefiningOp();
            Operation *members = operation.getOperand(1).getDefiningOp();
            auto offsets = dyn_cast<RankedTensorType>(
                operation.getOperand(2).getType());
            auto indices = operation.getNumOperands() == 4
                               ? dyn_cast<RankedTensorType>(
                                     operation.getOperand(3).getType())
                               : RankedTensorType();
            bool supportedOuter =
                outer && (facts.domainSourceAxes.count(outer) ||
                          facts.staticDomainExtents.count(outer));
            if (!supportedOuter || !members ||
                !facts.domainSourceAxes.count(members) || !offsets ||
                offsets.getRank() != 1 ||
                !isa<IntegerType, IndexType>(offsets.getElementType()) ||
                (operation.getNumOperands() == 4 &&
                 (!indices || indices.getRank() != 1 ||
                  !isa<IntegerType, IndexType>(indices.getElementType()))))
              return operation.emitOpError(
                  "ragged relation requires outer/member domains, rank-one integer offsets, and an optional rank-one integer index map");
            if (failed(backingView(operation.getOperand(2), facts, operation)) ||
                (operation.getNumOperands() == 4 &&
                 failed(backingView(operation.getOperand(3), facts,
                                    operation))))
              return failure();
            facts.raggedRelations[&operation] =
                RaggedRelationFact{
                    &operation, outer, members, nullptr,
                    operation.getOperand(2),
                    operation.getNumOperands() == 4 ? operation.getOperand(3)
                                                    : Value(),
                    {}};
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.ragged_outer",
          [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                operation.getNumResults() != 1)
              return operation.emitOpError(
                  "has no canonical ragged-outer schema");
            Operation *relation = operation.getOperand(0).getDefiningOp();
            auto found = facts.raggedRelations.find(relation);
            if (found == facts.raggedRelations.end() ||
                found->second.outerDomain)
              return operation.emitOpError(
                  "does not reference one unresolved ragged relation");
            Operation *source = found->second.outerSource;
            if (facts.staticDomainExtents.count(source)) {
              facts.staticDomainExtents[&operation] =
                  facts.staticDomainExtents.lookup(source);
              facts.staticDomainBounds[&operation] =
                  facts.staticDomainBounds.lookup(source);
            } else {
              facts.domainSources[&operation] = facts.domainSources.lookup(source);
              facts.domainSourceAxes[&operation] =
                  facts.domainSourceAxes.lookup(source);
            }
            found->second.outerDomain = &operation;
            facts.raggedOuterRelations[&operation] = relation;
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.ragged_member",
          [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 2 ||
                operation.getNumResults() != 1)
              return operation.emitOpError(
                  "has no canonical ragged-member schema");
            Operation *relation = operation.getOperand(0).getDefiningOp();
            auto found = facts.raggedRelations.find(relation);
            if (found == facts.raggedRelations.end())
              return operation.emitOpError(
                  "does not reference a canonical ragged relation");
            FailureOr<Operation *> selector = resolveDomain(
                operation.getOperand(1), facts, operation);
            if (succeeded(selector) && !found->second.outerDomain) {
              Operation *outerSource = found->second.outerSource;
              Value selectorSource = facts.domainSources.lookup(*selector);
              bool sameDynamicSource =
                  outerSource && selectorSource &&
                  facts.domainSources.lookup(outerSource) == selectorSource &&
                  facts.domainSourceAxes.lookup(outerSource) ==
                      facts.domainSourceAxes.lookup(*selector);
              bool sameStaticDomain =
                  outerSource && facts.staticDomainBounds.count(outerSource) &&
                  facts.staticDomainBounds.lookup(outerSource) ==
                      facts.staticDomainBounds.lookup(*selector);
              if (sameDynamicSource || sameStaticDomain) {
                found->second.outerDomain = *selector;
                facts.raggedOuterRelations[*selector] = relation;
              }
            }
            if (failed(selector) || *selector != found->second.outerDomain)
              return operation.emitOpError(
                  "member selector is not owned by the ragged outer domain");
            Operation *memberSource = found->second.memberSource;
            if (!memberSource || !facts.domainSources.count(memberSource) ||
                !facts.domainSourceAxes.count(memberSource))
              return operation.emitOpError(
                  "ragged relation has no canonical member source domain");
            facts.domainSources[&operation] =
                facts.domainSources.lookup(memberSource);
            facts.domainSourceAxes[&operation] =
                facts.domainSourceAxes.lookup(memberSource);
            facts.raggedMembers[&operation] =
                RaggedMemberFact{relation, &operation, operation.getOperand(1)};
            found->second.memberDomains.push_back(&operation);
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.members", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical members schema");
            FailureOr<Operation *> domain =
                resolveDomain(operation.getOperand(0), facts, operation);
            if (failed(domain) || !facts.raggedMembers.count(*domain))
              return operation.emitOpError(
                  "members requires a ragged-member region");
            auto result = dyn_cast<RankedTensorType>(
                operation.getResult(0).getType());
            if (!result || result.getRank() != 1 ||
                !isa<IntegerType, IndexType>(result.getElementType()))
              return operation.emitOpError(
                  "members must produce a rank-one index tensor");
            FailureOr<LogicalAxis> axis =
                axisFromDomain(**domain, facts, operation);
            if (failed(axis) ||
                failed(bindResultAxes(operation, 0, {*axis}, facts)))
              return failure();
            facts.memberValues[operation.getResult(0)] = *domain;
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.view_load", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() < 1 ||
                operation.getNumResults() != 1 ||
                !isa<intent::ViewType>(operation.getOperand(0).getType()))
              return operation.emitOpError(
                  "has no canonical external-view load schema");
            FailureOr<SmallVector<IndexTerm>> relation =
                parseIndexRelation(operation);
            if (failed(relation))
              return failure();
            bool wholeView = !relation->empty() && llvm::all_of(
                *relation, [](const IndexTerm &term) {
                  return term.kind == "full_slice";
                });
            if (wholeView) {
              facts.wholeViewLoads.insert(&operation);
              return recordLoadAxes(operation, facts);
            }
            bool feedsContract = llvm::any_of(
                operation.getResult(0).getUsers(), [](Operation *user) {
                  return user->getName().getStringRef() == "intent.contract";
                });
            FailureOr<bool> masked = requiresRuntimeBoundary(operation, facts);
            if (failed(masked))
              return failure();
            std::optional<std::string> fill =
                !*masked ? std::optional<std::string>("none")
                : feedsContract ? std::optional<std::string>("zero")
                                : inferMaskedLaneFill(operation.getResult(0));
            if (!fill) {
              InFlightDiagnostic diagnostic = operation.emitOpError(
                  "cannot prove a semantics-preserving masked-load fill");
              if (auto names = operation.getAttrOfType<ArrayAttr>(
                      "intent.result_names"))
                diagnostic << " for " << names;
              return failure();
            }
            if (failed(analyzeBoundary(operation, facts, *fill)) ||
                failed(recordLoadAxes(operation, facts)))
              return failure();
            if (failed(classifyTensorIndices(operation, facts)))
              return failure();
            return recordAccessRanges(operation, facts);
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.view_store", [&](Operation &operation) -> LogicalResult {
            auto valueIndex = operation.getAttrOfType<IntegerAttr>(
                "intent.value_operand_index");
            if (!valueIndex || valueIndex.getInt() <= 0 ||
                static_cast<unsigned>(valueIndex.getInt()) >=
                    operation.getNumOperands())
              return operation.emitOpError(
                  "has no canonical external-view store schema");
            if (failed(analyzeBoundary(operation, facts, "none")))
              return failure();
            if (failed(classifyTensorIndices(operation, facts)))
              return failure();
            return recordAccessRanges(operation, facts);
          })))
    return failure();

  auto bindReductionAxes = [&](Operation &operation) -> LogicalResult {
    auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
    auto components =
        operation.getAttrOfType<IntegerAttr>("intent.component_count");
    auto input = operation.getNumOperands() > 0
                     ? facts.valueAxes.find(operation.getOperand(0))
                     : facts.valueAxes.end();
    if (!axes || axes.empty() || !components || components.getInt() <= 0 ||
        operation.getNumResults() != static_cast<unsigned>(components.getInt()) ||
        input == facts.valueAxes.end())
      return operation.emitOpError(
          "reduction axes have no logical-axis provenance");
    for (unsigned component = 1;
         component < static_cast<unsigned>(components.getInt()); ++component) {
      auto componentAxes = facts.valueAxes.find(operation.getOperand(component));
      if (componentAxes == facts.valueAxes.end() ||
          componentAxes->second != input->second)
        return operation.emitOpError(
            "reduction components must share one logical-axis schema");
    }
    SmallVector<unsigned> reducedAxes;
    for (Attribute attribute : axes) {
      auto axis = dyn_cast<IntegerAttr>(attribute);
      if (!axis || axis.getInt() < 0 ||
          static_cast<size_t>(axis.getInt()) >= input->second.size())
        return operation.emitOpError(
            "reduction axis has no logical-axis provenance");
      reducedAxes.push_back(axis.getInt());
      if (Operation *domain = input->second[axis.getInt()].domain) {
        facts.vectorDomains.insert(domain);
        facts.reductionDomains.insert(domain);
      }
    }
    llvm::sort(reducedAxes, std::greater<unsigned>());
    SmallVector<LogicalAxis> resultAxes = input->second;
    for (unsigned axis : reducedAxes)
      resultAxes.erase(resultAxes.begin() + axis);
    for (unsigned result = 0; result < operation.getNumResults(); ++result)
      if (failed(bindResultAxes(operation, result, resultAxes, facts)))
        return failure();
    return success();
  };
  if (failed(addHandler(registry, "intent.reduce", bindReductionAxes)))
    return failure();

  if (failed(addHandler(
          registry, "intent.scan", [&](Operation &operation) -> LogicalResult {
            auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
            auto components =
                operation.getAttrOfType<IntegerAttr>("intent.component_count");
            auto input = operation.getNumOperands() > 0
                             ? facts.valueAxes.find(operation.getOperand(0))
                             : facts.valueAxes.end();
            if (!axis || axis.getInt() < 0 || !components ||
                components.getInt() <= 0 ||
                operation.getNumResults() !=
                    static_cast<unsigned>(components.getInt()) ||
                input == facts.valueAxes.end() ||
                static_cast<size_t>(axis.getInt()) >= input->second.size() ||
                operation.getNumResults() == 0)
              return operation.emitOpError(
                  "scan axis has no logical-axis provenance");
            for (unsigned component = 1;
                 component < static_cast<unsigned>(components.getInt());
                 ++component) {
              auto componentAxes =
                  facts.valueAxes.find(operation.getOperand(component));
              if (componentAxes == facts.valueAxes.end() ||
                  componentAxes->second != input->second)
                return operation.emitOpError(
                    "scan components must share one logical-axis schema");
            }
            if (Operation *domain = input->second[axis.getInt()].domain)
              facts.vectorDomains.insert(domain);
            for (unsigned result = 0; result < operation.getNumResults(); ++result)
              if (failed(bindResultAxes(operation, result, input->second, facts)))
                return failure();
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast", "intent.compare", "intent.select",
                         "intent.mask", "intent.random"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              return propagatePointwiseAxes(operation, facts);
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.reshape", [&](Operation &operation) -> LogicalResult {
            return propagateReshapeAxes(operation, facts);
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.transpose",
          [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                operation.getNumResults() != 1)
              return operation.emitOpError(
                  "has no canonical transpose provenance schema");
            auto source = facts.valueAxes.find(operation.getOperand(0));
            auto permutation =
                operation.getAttrOfType<ArrayAttr>("intent.permutation");
            if (source == facts.valueAxes.end() || !permutation ||
                permutation.size() != source->second.size())
              return operation.emitOpError(
                  "transpose has no logical-axis provenance");
            SmallVector<LogicalAxis> resultAxes;
            resultAxes.reserve(permutation.size());
            SmallVector<bool> covered(source->second.size(), false);
            for (Attribute attribute : permutation) {
              auto axis = dyn_cast<IntegerAttr>(attribute);
              if (!axis || axis.getInt() < 0 ||
                  static_cast<size_t>(axis.getInt()) >= source->second.size() ||
                  covered[axis.getInt()])
                return operation.emitOpError(
                    "transpose permutation has no logical-axis provenance");
              covered[axis.getInt()] = true;
              resultAxes.push_back(source->second[axis.getInt()]);
            }
            return bindResultAxes(operation, 0, std::move(resultAxes), facts);
          })))
    return failure();

  for (StringRef name : {"intent.full", "intent.zeros"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              FailureOr<SmallVector<LogicalAxis>> axes =
                  axesFromResultShape(operation, 0, facts);
              if (failed(axes))
                return failure();
              return bindResultAxes(operation, 0, std::move(*axes), facts);
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.indices", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 || operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical indices schema");
            auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
            if (mode && mode.getValue() == "tensor_axis") {
              auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
              auto source = facts.valueAxes.find(operation.getOperand(0));
              if (!axis || axis.getInt() < 0 || source == facts.valueAxes.end() ||
                  static_cast<size_t>(axis.getInt()) >= source->second.size())
                return operation.emitOpError(
                    "tensor-axis indices have no logical-axis provenance");
              if (Operation *domain = source->second[axis.getInt()].domain)
                facts.vectorDomains.insert(domain);
              return bindResultAxes(operation, 0, source->second, facts);
            }
            FailureOr<Operation *> domain =
                resolveDomain(operation.getOperand(0), facts, operation);
            if (failed(domain))
              return failure();
            FailureOr<LogicalAxis> axis =
                axisFromDomain(**domain, facts, operation);
            if (failed(axis))
              return failure();
            return bindResultAxes(operation, 0, {*axis}, facts);
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.region_end",
          [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                operation.getNumResults() != 1 ||
                !isa<IntegerType, IndexType>(operation.getResult(0).getType()))
              return operation.emitOpError(
                  "has no canonical logical range-end schema");
            FailureOr<Operation *> domain =
                resolveDomain(operation.getOperand(0), facts, operation);
            if (failed(domain))
              return failure();
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.gather", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() < 1 || operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical gather schema");
            FailureOr<SmallVector<LogicalAxis>> resultAxes = inferIndexedAxes(
                operation, operation.getOperand(0), facts);
            if (failed(resultAxes))
              return failure();
            return bindResultAxes(operation, 0, std::move(*resultAxes), facts);
          })))
    return failure();

  auto bindIndexedWrite = [&](Operation &operation,
                              bool recordScatter) -> LogicalResult {
    auto valueIndex =
        operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
    bool reduction =
        operation.getName().getStringRef() == "intent.scatter_reduce";
    auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
    if (!valueIndex || valueIndex.getInt() <= 0 ||
        static_cast<unsigned>(valueIndex.getInt()) >=
            operation.getNumOperands() ||
        (reduction && (!combine || combine.getValue() != "add")) ||
        (!reduction && combine) ||
        !isa<intent::ViewType>(operation.getOperand(0).getType()))
      return operation.emitOpError(
          "has no canonical indexed external-write schema");
    FailureOr<SmallVector<LogicalAxis>> indexedAxes =
        inferIndexedAxes(operation, operation.getOperand(0), facts);
    auto valueAxes =
        facts.valueAxes.find(operation.getOperand(valueIndex.getInt()));
    SmallVector<LogicalAxis> scalarAxes;
    ArrayRef<LogicalAxis> writtenAxes =
        valueAxes == facts.valueAxes.end()
            ? ArrayRef<LogicalAxis>(scalarAxes)
            : ArrayRef<LogicalAxis>(valueAxes->second);
    if (failed(indexedAxes) ||
        (valueAxes == facts.valueAxes.end() &&
         isa<RankedTensorType>(operation.getOperand(valueIndex.getInt()).getType())) ||
        ArrayRef<LogicalAxis>(*indexedAxes) != writtenAxes)
      return operation.emitOpError(
          "indexed write value does not match its destination");
    if (failed(analyzeBoundary(operation, facts, "none")) ||
        failed(classifyTensorIndices(operation, facts)))
      return failure();
    if (recordScatter)
      facts.scatterWrites.insert(&operation);
    return success();
  };
  auto bindScatterWrite = [bindIndexedWrite](Operation &operation) {
    return bindIndexedWrite(operation, true);
  };
  auto bindAtomicWrite = [bindIndexedWrite](Operation &operation) {
    return bindIndexedWrite(operation, false);
  };
  if (failed(addHandler(registry, "intent.scatter_unique", bindScatterWrite)) ||
      failed(addHandler(registry, "intent.scatter_reduce", bindScatterWrite)) ||
      failed(addHandler(registry, "intent.atomic_add", bindAtomicWrite)) ||
      failed(addHandler(registry, "intent.atomic_cas", bindAtomicWrite)))
    return failure();

  auto enterStateStream = [&](Operation &operation) -> LogicalResult {
    auto stateCount = operation.getAttrOfType<IntegerAttr>("intent.state_count");
    auto extentIndex =
        operation.getAttrOfType<IntegerAttr>("intent.extent_operand_index");
    auto stopIndex =
        operation.getAttrOfType<IntegerAttr>("intent.stop_operand_index");
    auto extent = operation.getAttrOfType<DictionaryAttr>("intent.extent");
    auto tile = extent ? extent.getAs<StringAttr>("name") : StringAttr();
    if (!stateCount || stateCount.getInt() <= 0 || operation.getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation.getRegion(0)) ||
        operation.getNumResults() !=
            static_cast<unsigned>(stateCount.getInt()))
      return operation.emitOpError("has no canonical state-stream schema");
    bool namedExtent = tile && !tile.getValue().empty();
    if (namedExtent == static_cast<bool>(extentIndex))
      return operation.emitOpError(
          "state stream requires exactly one named or runtime extent");
    int64_t trailingIndex = stateCount.getInt() + 1;
    if (extentIndex) {
      if (extentIndex.getInt() != trailingIndex ||
          extentIndex.getInt() < 0 ||
          static_cast<unsigned>(extentIndex.getInt()) >=
              operation.getNumOperands() ||
          !isa<IntegerType, IndexType>(
              operation.getOperand(extentIndex.getInt()).getType()))
        return operation.emitOpError(
            "state stream runtime extent is not the canonical trailing index");
      ++trailingIndex;
    }
    if (stopIndex) {
      if (stopIndex.getInt() != trailingIndex || stopIndex.getInt() < 0 ||
          static_cast<unsigned>(stopIndex.getInt()) >= operation.getNumOperands())
        return operation.emitOpError(
            "state stream stop is not the canonical trailing operand");
      ++trailingIndex;
    }
    if (operation.getNumOperands() != static_cast<unsigned>(trailingIndex))
      return operation.emitOpError("has no canonical state-stream operand layout");
    Operation *axisDomain = operation.getOperand(0).getDefiningOp();
    if (!axisDomain || !facts.domainSourceAxes.count(axisDomain))
      return operation.emitOpError(
          "state stream requires a canonical source domain");
    if (failed(bindRegionArgumentAxis(operation, 0, *axisDomain, facts)))
      return failure();
    if (extentIndex) {
      Operation *extentValue =
          operation.getOperand(extentIndex.getInt()).getDefiningOp();
      auto constant = extentValue
                          ? extentValue->getAttrOfType<IntegerAttr>("intent.value")
                          : IntegerAttr();
      if (!extentValue ||
          extentValue->getName().getStringRef() != "intent.constant" ||
          !constant || constant.getInt() <= 0)
        return operation.emitOpError(
            "state stream runtime extent must be a positive constant");
      auto previous = facts.orderedStreamFixedExtents.find(axisDomain);
      if (previous != facts.orderedStreamFixedExtents.end() &&
          previous->second != constant.getInt())
        return operation.emitOpError(
            "assigns incompatible fixed extents to one ordered axis");
      facts.orderedStreamFixedExtents[axisDomain] = constant.getInt();
    }
    Operation *stopBound = nullptr;
    if (stopIndex) {
      stopBound = operation.getOperand(stopIndex.getInt()).getDefiningOp();
      if (!stopBound ||
          stopBound->getName().getStringRef() != "intent.region_end")
        return operation.emitOpError(
            "state stream stop must come from I.end(domain_or_region)");
    }
    Block &body = operation.getRegion(0).front();
    if (body.getNumArguments() !=
        static_cast<unsigned>(stateCount.getInt() + 1))
      return operation.emitOpError(
          "state stream body does not match carried state");
    StateStreamFact fact{stateCount.getInt(), &body, {}};
    for (int64_t index = 0; index < stateCount.getInt(); ++index) {
      Value initial = operation.getOperand(index + 1);
      Value argument = body.getArgument(index + 1);
      fact.initialState.push_back(initial);
      auto axes = facts.valueAxes.find(initial);
      if (isa<RankedTensorType>(initial.getType()) &&
          axes == facts.valueAxes.end())
        return operation.emitOpError(
            "tensor state has no logical-axis provenance");
      if (axes != facts.valueAxes.end()) {
        facts.valueAxes[argument] = axes->second;
        facts.valueAxes[operation.getResult(index)] = axes->second;
      }
    }
    facts.orderedDomains.insert(axisDomain);
    facts.stateStreams[&operation] = std::move(fact);
    return success();
  };
  auto leaveStateStream = [&](Operation &operation) -> LogicalResult {
    StateStreamFact &fact = facts.stateStreams[&operation];
    Operation &terminator = fact.body->back();
    if (terminator.getName().getStringRef() != "intent.yield" ||
        terminator.getNumOperands() != static_cast<unsigned>(fact.stateCount))
      return operation.emitOpError(
          "state stream must yield every carried value");
    for (int64_t index = 0; index < fact.stateCount; ++index) {
      Value yielded = terminator.getOperand(index);
      auto initialAxes = facts.valueAxes.find(fact.initialState[index]);
      auto yieldedAxes = facts.valueAxes.find(yielded);
      if (initialAxes != facts.valueAxes.end()) {
        SmallVector<LogicalAxis> merged;
        if (yieldedAxes == facts.valueAxes.end() ||
            !mergeLogicalAxes(initialAxes->second, yieldedAxes->second, merged))
          return operation.emitOpError(
              "yielded state changes its logical axis identity");
        facts.valueAxes[operation.getResult(index)] = std::move(merged);
      }
    }
    return success();
  };
  if (failed(registry.add(
          "intent.state_stream",
          OperationHandler{enterStateStream, leaveStateStream})))
    return failure();

  auto analyzeContraction = [&](Operation &operation) -> LogicalResult {
            unsigned operandCount =
                operation.getName().getStringRef() == "intent.scaled_contract"
                    ? 4
                    : 2;
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto batch = operation.getAttrOfType<ArrayAttr>("intent.batch");
            auto lhs = operation.getNumOperands() == operandCount
                           ? facts.valueAxes.find(operation.getOperand(0))
                           : facts.valueAxes.end();
            auto rhs = operation.getNumOperands() == operandCount
                           ? facts.valueAxes.find(operation.getOperand(1))
                           : facts.valueAxes.end();
            if (!reduce || reduce.empty() || !batch || lhs == facts.valueAxes.end() ||
                rhs == facts.valueAxes.end() || operation.getNumResults() != 1)
              return operation.emitOpError(
                  "contraction has no logical-axis provenance");
            llvm::DenseSet<unsigned> lhsReduced;
            llvm::DenseSet<unsigned> rhsReduced;
            for (Attribute attribute : reduce) {
              auto pair = dyn_cast<ArrayAttr>(attribute);
              auto lhsAxis = pair && pair.size() == 2
                                 ? dyn_cast<IntegerAttr>(pair[0])
                                 : IntegerAttr();
              auto rhsAxis = pair && pair.size() == 2
                                 ? dyn_cast<IntegerAttr>(pair[1])
                                 : IntegerAttr();
              if (!lhsAxis || !rhsAxis || lhsAxis.getInt() < 0 ||
                  rhsAxis.getInt() < 0 ||
                  static_cast<size_t>(lhsAxis.getInt()) >= lhs->second.size() ||
                  static_cast<size_t>(rhsAxis.getInt()) >= rhs->second.size() ||
                  lhs->second[lhsAxis.getInt()] != rhs->second[rhsAxis.getInt()] ||
                  !lhsReduced.insert(lhsAxis.getInt()).second ||
                  !rhsReduced.insert(rhsAxis.getInt()).second)
                return operation.emitOpError(
                    "contraction pair has invalid logical-domain provenance");
              if (Operation *domain = lhs->second[lhsAxis.getInt()].domain)
                facts.contractionDomains.insert(domain);
            }
            llvm::DenseSet<unsigned> lhsBatched;
            llvm::DenseSet<unsigned> rhsBatched;
            for (Attribute attribute : batch) {
              auto pair = dyn_cast<ArrayAttr>(attribute);
              auto lhsAxis = pair && pair.size() == 2
                                 ? dyn_cast<IntegerAttr>(pair[0])
                                 : IntegerAttr();
              auto rhsAxis = pair && pair.size() == 2
                                 ? dyn_cast<IntegerAttr>(pair[1])
                                 : IntegerAttr();
              if (!lhsAxis || !rhsAxis || lhsAxis.getInt() < 0 ||
                  rhsAxis.getInt() < 0 ||
                  static_cast<size_t>(lhsAxis.getInt()) >= lhs->second.size() ||
                  static_cast<size_t>(rhsAxis.getInt()) >= rhs->second.size() ||
                  lhs->second[lhsAxis.getInt()] != rhs->second[rhsAxis.getInt()] ||
                  lhsReduced.contains(lhsAxis.getInt()) ||
                  rhsReduced.contains(rhsAxis.getInt()) ||
                  !lhsBatched.insert(lhsAxis.getInt()).second ||
                  !rhsBatched.insert(rhsAxis.getInt()).second)
                return operation.emitOpError(
                    "contraction batch pair has invalid logical-domain provenance");
            }
            SmallVector<LogicalAxis> resultAxes;
            for (auto [axis, logicalAxis] : llvm::enumerate(lhs->second))
              if (!lhsReduced.contains(axis))
                resultAxes.push_back(logicalAxis);
            for (auto [axis, logicalAxis] : llvm::enumerate(rhs->second))
              if (!rhsReduced.contains(axis) && !rhsBatched.contains(axis))
                resultAxes.push_back(logicalAxis);
            ContractionFact fact;
            fact.operation = &operation;
            fact.lhsAxes = lhs->second;
            fact.rhsAxes = rhs->second;
            fact.resultAxes = resultAxes;
            fact.lhsReductionAxes.assign(lhsReduced.begin(), lhsReduced.end());
            fact.rhsReductionAxes.assign(rhsReduced.begin(), rhsReduced.end());
            fact.lhsBatchAxes.assign(lhsBatched.begin(), lhsBatched.end());
            fact.rhsBatchAxes.assign(rhsBatched.begin(), rhsBatched.end());
            llvm::sort(fact.lhsReductionAxes);
            llvm::sort(fact.rhsReductionAxes);
            llvm::sort(fact.lhsBatchAxes);
            llvm::sort(fact.rhsBatchAxes);
            facts.contractions[&operation] = std::move(fact);
            if (operation.getName().getStringRef() == "intent.scaled_contract") {
              Operation *stream = operation.getParentOp();
              while (stream && stream->getName().getStringRef() !=
                                   "intent.state_stream")
                stream = stream->getParentOp();
              if (stream && stream->getNumOperands() > 0) {
                FailureOr<SmallVector<Operation *>> domains =
                    expandDomainSource(stream->getOperand(0), operation);
                if (failed(domains))
                  return failure();
                facts.scaledStreamDomains.insert(domains->begin(), domains->end());
              }
            }
            return bindResultAxes(operation, 0, std::move(resultAxes), facts);
          };
  if (failed(addHandler(registry, "intent.contract", analyzeContraction)) ||
      failed(addHandler(registry, "intent.scaled_contract", analyzeContraction)))
    return failure();
  if (failed(addHandler(
          registry, "intent.sparse_contract",
          [&](Operation &operation) -> LogicalResult {
            auto format = operation.getAttrOfType<StringAttr>("intent.format");
            auto compressedAxis =
                operation.getAttrOfType<IntegerAttr>("intent.compressed_axis");
            auto metadataAxis =
                operation.getAttrOfType<IntegerAttr>("intent.metadata_axis");
            auto rhsReductionAxis = operation.getAttrOfType<IntegerAttr>(
                "intent.rhs_reduction_axis");
            auto compressed = operation.getNumOperands() == 3
                                  ? facts.valueAxes.find(operation.getOperand(0))
                                  : facts.valueAxes.end();
            auto metadata = operation.getNumOperands() == 3
                                ? facts.valueAxes.find(operation.getOperand(1))
                                : facts.valueAxes.end();
            auto rhs = operation.getNumOperands() == 3
                           ? facts.valueAxes.find(operation.getOperand(2))
                           : facts.valueAxes.end();
            auto compressedType = operation.getNumOperands() == 3
                                      ? dyn_cast<RankedTensorType>(
                                            operation.getOperand(0).getType())
                                      : RankedTensorType();
            auto metadataType = operation.getNumOperands() == 3
                                    ? dyn_cast<RankedTensorType>(
                                          operation.getOperand(1).getType())
                                    : RankedTensorType();
            auto rhsType = operation.getNumOperands() == 3
                               ? dyn_cast<RankedTensorType>(
                                     operation.getOperand(2).getType())
                               : RankedTensorType();
            auto resultType = operation.getNumResults() == 1
                                  ? dyn_cast<RankedTensorType>(
                                        operation.getResult(0).getType())
                                  : RankedTensorType();
            if (!format || format.getValue() != "two_of_four" ||
                !compressedAxis || compressedAxis.getInt() != 1 ||
                !metadataAxis || metadataAxis.getInt() != 1 ||
                !rhsReductionAxis || rhsReductionAxis.getInt() != 0 ||
                !compressedType || !metadataType || !rhsType || !resultType ||
                compressedType.getRank() != 2 || metadataType.getRank() != 2 ||
                rhsType.getRank() != 2 || resultType.getRank() != 2 ||
                compressedType.getElementType() != rhsType.getElementType() ||
                !metadataType.getElementType().isInteger(16) ||
                compressed == facts.valueAxes.end() ||
                metadata == facts.valueAxes.end() || rhs == facts.valueAxes.end() ||
                compressed->second.size() != 2 || metadata->second.size() != 2 ||
                rhs->second.size() != 2 ||
                compressed->second[0] != metadata->second[0])
              return operation.emitOpError(
                  "2:4 sparse contraction does not match its canonical format, dtype, or axis schema");
            SmallVector<LogicalAxis> resultAxes{
                compressed->second[0], rhs->second[1]};
            Operation *row = compressed->second[0].domain;
            Operation *column = rhs->second[1].domain;
            Operation *reduction = rhs->second[0].domain;
            if (reduction)
              facts.contractionDomains.insert(reduction);
            facts.sparseContractions[&operation] =
                SparseContractionFact{&operation, row, column, reduction};
            return bindResultAxes(operation, 0, std::move(resultAxes), facts);
          })))
    return failure();
  return success();
}

} // namespace

FailureOr<Operation *> resolveDomain(Value indexedValue,
                                     const KernelFacts &facts,
                                     Operation &consumer) {
  if (Operation *definition = indexedValue.getDefiningOp())
    if (facts.domainSourceAxes.count(definition) ||
        facts.staticDomainExtents.count(definition))
      return definition;
  auto argument = dyn_cast<BlockArgument>(indexedValue);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (owner && owner->getName().getStringRef() == "intent.state_stream" &&
      argument.getArgNumber() == 0 && owner->getNumOperands() > 0) {
    Operation *domain = owner->getOperand(0).getDefiningOp();
    if (domain && facts.domainSourceAxes.count(domain))
      return domain;
    consumer.emitOpError("cannot resolve a state stream to its source domain");
    return failure();
  }
  if (owner && owner->getName().getStringRef() == "intent.for" &&
      argument.getArgNumber() == 0 && owner->getNumOperands() > 0) {
    Operation *domain = owner->getOperand(0).getDefiningOp();
    if (domain && (facts.domainSourceAxes.count(domain) ||
                   facts.staticDomainExtents.count(domain)))
      return domain;
    consumer.emitOpError("cannot resolve a sequential for to its source domain");
    return failure();
  }
  if (owner && owner->getName().getStringRef() == "intent.ordered" &&
      owner->getNumOperands() > 0) {
    FailureOr<SmallVector<Operation *>> domains =
        expandDomainSource(owner->getOperand(0), consumer);
    if (succeeded(domains) && argument.getArgNumber() < domains->size())
      return (*domains)[argument.getArgNumber()];
    consumer.emitOpError("cannot resolve an ordered loop to its source domain");
    return failure();
  }
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("indexes with a value not owned by a parallel region");
    return failure();
  }
  auto domains = facts.parallelDomains.find(owner);
  if (domains != facts.parallelDomains.end() &&
      argument.getArgNumber() < domains->second.size())
    return domains->second[argument.getArgNumber()];
  consumer.emitOpError("cannot resolve an indexed region to its source domain");
  return failure();
}

bool hasNonnegativeIntegerOperands(Operation &operation,
                                   const KernelFacts &facts) {
  return hasNonnegativeIntegerOperandsImpl(operation, facts);
}

TensorIndexingKind tensorIndexingKind(Operation &operation,
                                      const KernelFacts &facts) {
  auto found = facts.tensorIndexing.find(&operation);
  return found == facts.tensorIndexing.end() ? TensorIndexingKind::none
                                              : found->second;
}

LogicalResult analyzeKernelFacts(KernelFacts &facts) {
  for (const ABIArgument &argument : facts.kernel.abi.arguments) {
    auto shape = argument.metadata.getAs<ArrayAttr>("shape");
    if (!shape)
      continue;
    for (Attribute attribute : shape) {
      auto label = dyn_cast<StringAttr>(attribute);
      if (!label || label.getValue().empty())
        return facts.kernel.entry.emitOpError(
            "has non-canonical ABI axis provenance metadata");
      facts.axisLabels.try_emplace(
          label.getValue(), LogicalAxis{nullptr, label.getValue().str()});
    }
  }
  OperationHandlerRegistry registry;
  if (failed(registerFactHandlers(registry, facts)))
    return facts.kernel.entry.emitOpError(
        "failed to construct canonical realization handlers");
  if (failed(traverseKernel(facts.kernel.entry, registry,
                            "canonical realization analysis")))
    return failure();

  llvm::DenseSet<Operation *> programDomains;
  for (Operation *parallel : facts.parallels)
    for (Operation *domain : facts.parallelDomains.lookup(parallel))
      programDomains.insert(domain);
  for (const auto &entry : facts.valueAxes)
    for (const LogicalAxis &axis : entry.second)
      if (axis.domain && !programDomains.contains(axis.domain) &&
          !facts.orderedDomains.contains(axis.domain) &&
          !facts.contractionDomains.contains(axis.domain))
        facts.vectorDomains.insert(axis.domain);

  auto collectScanProducers = [&](Operation &scan) -> LogicalResult {
    auto axis = scan.getAttrOfType<IntegerAttr>("intent.axis");
    auto components =
        scan.getAttrOfType<IntegerAttr>("intent.component_count");
    auto axes = scan.getNumOperands() > 0
                    ? facts.valueAxes.find(scan.getOperand(0))
                    : facts.valueAxes.end();
    if (!axis || axis.getInt() < 0 || !components || components.getInt() <= 0 ||
        scan.getNumResults() != static_cast<unsigned>(components.getInt()) ||
        axes == facts.valueAxes.end() ||
        static_cast<size_t>(axis.getInt()) >= axes->second.size() ||
        !axes->second[axis.getInt()].domain)
      return scan.emitOpError("has no canonical scan producer axis");

    llvm::DenseSet<Value> visited;
    llvm::DenseSet<Operation *> slice;
    std::function<LogicalResult(Value)> collect = [&](Value value) -> LogicalResult {
      if (!visited.insert(value).second)
        return success();
      Operation *producer = value.getDefiningOp();
      if (!producer)
        return success();
      StringRef name = producer->getName().getStringRef();
      if (name == "intent.constant" || name == "intent.dim" ||
          name == "intent.domain" || name == "intent.domain_product" ||
          name == "intent.partition")
        return success();
      bool pure = name == "intent.view_load" || name == "intent.indices" ||
                  name == "intent.broadcast" || name == "intent.unary" ||
                  name == "intent.binary" || name == "intent.compare" ||
                  name == "intent.mask" || name == "intent.select" ||
                  name == "intent.cast" || name == "intent.full" ||
                  name == "intent.zeros" || name == "intent.reshape" ||
                  name == "intent.transpose" ||
                  name == "intent.make_record" || name == "intent.extract";
      if (!pure) {
        producer->emitOpError(
            "cannot be replayed inside a physical scan chunk");
        return failure();
      }
      slice.insert(producer);
      if (name == "intent.indices")
        return success();
      unsigned firstOperand = name == "intent.view_load" ? 1 : 0;
      for (Value operand : producer->getOperands().drop_front(firstOperand))
        if (failed(collect(operand)))
          return failure();
      return success();
    };
    for (unsigned component = 0;
         component < static_cast<unsigned>(components.getInt()); ++component)
      if (failed(collect(scan.getOperand(component))))
        return failure();
    ScanFact fact;
    fact.axis = axes->second[axis.getInt()].domain;
    fact.scalarConsumers = llvm::all_of(scan.getResults(), [&](Value result) {
      return !result.use_empty() && llvm::all_of(result.getUsers(), [&](Operation *user) {
        return user->getName().getStringRef() == "intent.gather" &&
               user->getNumOperands() > 0 && user->getOperand(0) == result &&
               user->getNumResults() == 1 &&
               !isa<RankedTensorType>(user->getResult(0).getType());
      });
    });
    for (Operation *producer : slice) {
      FailureOr<int64_t> producerNode =
          target::getNodeID(*producer, "scan producer slice");
      if (failed(producerNode))
        return failure();
      fact.producers.push_back(*producerNode);
      if (producer->getNumResults() != 1)
        continue;
      Value result = producer->getResult(0);
      bool escapes = llvm::any_of(result.getUsers(), [&](Operation *user) {
        return user != &scan && !slice.contains(user) &&
               user->getName().getStringRef() != "intent.assume_in_bounds";
      });
      if (escapes) {
        if (fact.scalarConsumers &&
            llvm::any_of(result.getUsers(), [&](Operation *user) {
              if (user == &scan || slice.contains(user) ||
                  user->getName().getStringRef() == "intent.assume_in_bounds")
                return false;
              return user->getName().getStringRef() != "intent.gather" ||
                     user->getNumOperands() == 0 ||
                     user->getOperand(0) != result ||
                     user->getNumResults() != 1 ||
                     isa<RankedTensorType>(user->getResult(0).getType());
            }))
          return producer->emitOpError(
              "escapes a chunked scan through a non-scalar-gather consumer");
        FailureOr<int64_t> valueID = target::getValueID(
            result, facts.kernel, scan, "scan producer materialization");
        if (failed(valueID))
          return failure();
        fact.materializedValues.push_back(*valueID);
      }
    }
    llvm::sort(fact.producers);
    llvm::sort(fact.materializedValues);
    facts.scans[&scan] = std::move(fact);
    return success();
  };
  WalkResult scanWalk = facts.kernel.entry.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() != "intent.scan")
      return WalkResult::advance();
    return failed(collectScanProducers(*operation)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
  });
  if (scanWalk.wasInterrupted())
    return failure();
  return success();
}

} // namespace intent::target

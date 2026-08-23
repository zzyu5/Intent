#include "Intent/Target/GPU/Transforms/Passes.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Pass/Pass.h"

#include <cmath>

using namespace mlir;

namespace intent::gpu {
namespace {

struct PaddingBinding {
  Value value;
  SmallVector<int64_t> axes;
  SmallVector<int64_t> nodes;
  std::string fill;
};

class BoundaryNeutralizationProof {
public:
  BoundaryNeutralizationProof(const target::KernelFacts &facts,
                              ArrayRef<PaddingBinding> paddings)
      : facts(facts), paddings(paddings) {}

  bool prove(Operation &load) const {
    auto domains = facts.boundaryDomains.find(&load);
    auto fill = facts.boundaryFills.find(&load);
    if (load.getNumResults() != 1 || domains == facts.boundaryDomains.end() ||
        domains->second.empty() || fill == facts.boundaryFills.end() ||
        fill->second == "none")
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

  bool isRedundantPadding(Value value, Operation *domain,
                          StringRef expected) const {
    std::optional<std::string> derived =
        paddingForDomain(value, domain, value);
    return derived && *derived == expected;
  }

private:
  const target::KernelFacts &facts;
  ArrayRef<PaddingBinding> paddings;

  static bool containsDomain(ArrayRef<Operation *> domains,
                             Operation *domain) {
    return llvm::is_contained(domains, domain);
  }

  std::optional<int64_t> nodeOf(Operation *operation) const {
    auto node = operation
                    ? operation->getAttrOfType<IntegerAttr>("intent.node")
                    : IntegerAttr();
    return node ? std::optional<int64_t>(node.getInt()) : std::nullopt;
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

  std::optional<std::string>
  materializedPadding(Value value, Operation *domain,
                      Value ignoredMaterialization = {}) const {
    Operation *definition = value.getDefiningOp();
    if (definition &&
        ::intent::target::semanticOperationName(*definition) == "intent.view_load")
      return std::nullopt;
    std::optional<int64_t> domainNode = nodeOf(domain);
    std::optional<unsigned> tensorAxis = uniqueAxis(value, domain);
    if (!domainNode || !tensorAxis)
      return std::nullopt;
    for (const PaddingBinding &padding : paddings) {
      if (padding.value != value ||
          (ignoredMaterialization && value == ignoredMaterialization))
        continue;
      for (auto [axis, node] : llvm::zip(padding.axes, padding.nodes))
        if (axis == *tensorAxis && node == *domainNode)
          return padding.fill;
    }
    return std::nullopt;
  }

  std::optional<std::string>
  paddingForDomain(Value value, Operation *domain,
                   Value ignoredMaterialization = {}) const {
    if (std::optional<std::string> materialized =
            materializedPadding(value, domain, ignoredMaterialization))
      return materialized;
    Operation *definition = value.getDefiningOp();
    if (!definition)
      return std::nullopt;
    StringRef name = ::intent::target::semanticOperationName(*definition);
    if (name == "intent.view_load") {
      auto bounded = facts.boundaryDomains.find(definition);
      auto fill = facts.boundaryFills.find(definition);
      if (bounded != facts.boundaryDomains.end() &&
          containsDomain(bounded->second, domain) &&
          fill != facts.boundaryFills.end() && fill->second != "none")
        return fill->second;
      return std::nullopt;
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
      return paddingForDomain(definition->getOperand(0), domain,
                              ignoredMaterialization);
    if ((name == "intent.cast" || name == "intent.broadcast" ||
         name == "intent.reshape" || name == "intent.transpose") &&
        definition->getNumOperands() >= 1)
      return paddingForDomain(definition->getOperand(0), domain,
                              ignoredMaterialization);
    if (name == "intent.gather" && definition->getNumOperands() >= 1) {
      FailureOr<SmallVector<target::IndexTerm>> relation =
          target::parseIndexRelation(*definition);
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
          llvm::all_of(*relation, [](const target::IndexTerm &term) {
            return term.kind == "full_slice" || term.kind == "new_axis";
          });
      if (pureExpansion && validLiteral &&
          !validLiteral.getValue().isZero())
        return paddingForDomain(definition->getOperand(0), domain,
                                ignoredMaterialization);
    }
    if (name == "intent.unary" &&
        definition->getNumOperands() == 1) {
      std::optional<std::string> operand =
          paddingForDomain(definition->getOperand(0), domain,
                           ignoredMaterialization);
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
          paddingForDomain(definition->getOperand(0), domain,
                           ignoredMaterialization);
      std::optional<std::string> rhs =
          paddingForDomain(definition->getOperand(1), domain,
                           ignoredMaterialization);
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
    return std::nullopt;
  }

  bool valueCarriesDomains(Value value,
                           ArrayRef<Operation *> domains) const {
    return llvm::all_of(domains, [&](Operation *domain) {
      return uniqueAxis(value, domain).has_value();
    });
  }

  bool proveContractUse(Operation &contract, Value value,
                        ArrayRef<Operation *> domains,
                        llvm::DenseSet<Value> &active) const {
    auto fact = facts.contractions.find(&contract);
    if (fact == facts.contractions.end() || contract.getNumOperands() != 2 ||
        contract.getNumResults() != 1)
      return false;
    int64_t operandNumber = -1;
    for (auto [index, operand] : llvm::enumerate(contract.getOperands())) {
      if (operand != value)
        continue;
      if (operandNumber >= 0)
        return false;
      operandNumber = index;
    }
    if (operandNumber < 0)
      return false;
    ArrayRef<target::LogicalAxis> axes =
        operandNumber == 0 ? ArrayRef(fact->second.lhsAxes)
                           : ArrayRef(fact->second.rhsAxes);
    auto reductionPairs =
        contract.getAttrOfType<ArrayAttr>("intent.reduce");
    if (!reductionPairs)
      return false;
    SmallVector<Operation *> surviving;
    for (Operation *domain : domains) {
      std::optional<unsigned> axisNumber = uniqueAxis(value, domain);
      if (!axisNumber || *axisNumber >= axes.size())
        return false;
      std::optional<unsigned> pairedAxis;
      for (Attribute attribute : reductionPairs) {
        auto pair = dyn_cast<ArrayAttr>(attribute);
        auto lhs = pair && pair.size() == 2 ? dyn_cast<IntegerAttr>(pair[0])
                                            : IntegerAttr();
        auto rhs = pair && pair.size() == 2 ? dyn_cast<IntegerAttr>(pair[1])
                                            : IntegerAttr();
        if (!lhs || !rhs)
          return false;
        int64_t current = operandNumber == 0 ? lhs.getInt() : rhs.getInt();
        if (current == static_cast<int64_t>(*axisNumber))
          pairedAxis = operandNumber == 0 ? rhs.getInt() : lhs.getInt();
      }
      if (!pairedAxis) {
        surviving.push_back(domain);
        continue;
      }
      Value other = contract.getOperand(operandNumber == 0 ? 1 : 0);
      ArrayRef<target::LogicalAxis> otherAxes =
          operandNumber == 0 ? ArrayRef(fact->second.rhsAxes)
                             : ArrayRef(fact->second.lhsAxes);
      if (*pairedAxis >= otherAxes.size() ||
          otherAxes[*pairedAxis].domain != domain)
        return false;
      std::optional<std::string> otherPadding =
          paddingForDomain(other, domain);
      if (!otherPadding || *otherPadding != "zero")
        return false;
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
        static_cast<unsigned>(components.getInt()) != reduction.getNumResults())
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
    StringRef ownerName = owner ? ::intent::target::semanticOperationName(*owner) : StringRef();
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
      if (!materializedPadding(value, domain))
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
          name == "intent.scatter_reduce" || name == "intent.atomic_add") {
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
      if (name == "intent.contract") {
        if (!proveContractUse(*user, value, remaining, active))
          return finish(false);
        continue;
      }
      if (name == "intent.reduce") {
        if (!proveReductionUse(*user, value, remaining, active))
          return finish(false);
        continue;
      }
      if (name == "intent.yield") {
        if (!proveYieldUse(*user, value, remaining, active))
          return finish(false);
        continue;
      }
      if (name != "intent.cast" && name != "intent.broadcast" &&
          name != "intent.reshape" && name != "intent.transpose" &&
          name != "intent.unary" &&
          name != "intent.binary" && name != "intent.compare" &&
          name != "intent.select" && name != "intent.mask")
        return finish(false);
      if (user->getNumResults() != 1 ||
          !valueCarriesDomains(user->getResult(0), remaining) ||
          !proveUses(user->getResult(0), remaining, active))
        return finish(false);
    }
    return finish(true);
  }
};

class RefineBoundaryNeutralizationPass final
    : public PassWrapper<RefineBoundaryNeutralizationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RefineBoundaryNeutralizationPass)

  StringRef getArgument() const final {
    return "intent-refine-gpu-boundary-neutralization";
  }
  StringRef getDescription() const final {
    return "Refine conservative GPU transfers using consumer neutralization proofs";
  }

  void runOnOperation() final {
    SmallVector<plan::ProgramOp> programs(
        getOperation().getOps<plan::ProgramOp>());
    if (programs.size() != 1) {
      getOperation().emitError(
          "boundary neutralization refinement requires one physical program");
      signalPassFailure();
      return;
    }
    plan::ProgramOp program = programs.front();
    FailureOr<std::unique_ptr<PhysicalProgramAnalysis>> analysis =
        PhysicalProgramAnalysis::compute(program);
    if (failed(analysis) || failed(refine(program, **analysis)))
      signalPassFailure();
  }

private:
  static LogicalResult refine(plan::ProgramOp program,
                              PhysicalProgramAnalysis &analysis) {
    SmallVector<PaddingBinding> paddings;
    target::KernelModel &kernel = analysis.getKernel();
    for (plan::PaddingOp padding :
         program.getBody().getOps<plan::PaddingOp>()) {
      Value value = kernel.values.lookup(padding.getValue());
      if (!value)
        return padding.emitOpError(
            "does not resolve a physical-program value for padding");
      paddings.push_back(PaddingBinding{
          value, SmallVector<int64_t>(padding.getTensorAxes()),
          SmallVector<int64_t>(padding.getDomainNodes()),
          padding.getFill().str()});
    }
    BoundaryNeutralizationProof proof(analysis.getFacts(), paddings);
    SmallVector<plan::PaddingOp> redundantPaddings;
    for (plan::PaddingOp padding :
         program.getBody().getOps<plan::PaddingOp>()) {
      Value value = kernel.values.lookup(padding.getValue());
      bool redundant = value && !padding.getDomainNodes().empty() &&
                       llvm::all_of(
                           padding.getDomainNodes(), [&](int64_t node) {
                             Operation *domain = kernel.nodes.lookup(node);
                             return domain && proof.isRedundantPadding(
                                                  value, domain,
                                                  padding.getFill());
                           });
      if (redundant)
        redundantPaddings.push_back(padding);
    }
    for (plan::PaddingOp padding : redundantPaddings)
      padding.erase();
    OpBuilder builder(program.getContext());
    for (plan::TransferOp transfer :
         program.getBody().getOps<plan::TransferOp>()) {
      Operation *operation = kernel.nodes.lookup(transfer.getNode());
      bool neutralized =
          operation &&
          ::intent::target::semanticOperationName(*operation) == "intent.view_load" &&
          proof.prove(*operation);
      transfer->setAttr("consumer_neutralized",
                        builder.getBoolAttr(neutralized));
    }
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createRefineBoundaryNeutralizationPass() {
  return std::make_unique<RefineBoundaryNeutralizationPass>();
}

} // namespace intent::gpu

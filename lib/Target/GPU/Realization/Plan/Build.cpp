#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Support/Decisions.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <limits>

using namespace mlir;

namespace intent::gpu::realization {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter) {
  return registry.add(name,
                      target::OperationHandler{std::move(enter), {}});
}

struct PrivateBufferCandidate {
  Operation *operation = nullptr;
  int64_t registerUnits = 0;
  bool scalarized = false;
};

FailureOr<int64_t> privateBufferRegisterUnits(
    Operation &operation, const target::LogicalBufferInfo &info,
    bool roundToVectorExtent) {
  int64_t elements = 1;
  for (int64_t extent : info.shape) {
    if (elements > std::numeric_limits<int64_t>::max() / extent) {
      operation.emitOpError(
          "private logical-buffer capacity overflows the machine plan");
      return failure();
    }
    elements *= extent;
  }
  if (roundToVectorExtent) {
    int64_t physicalExtent = 1;
    while (physicalExtent < elements) {
      if (physicalExtent > std::numeric_limits<int64_t>::max() / 2) {
        operation.emitOpError(
            "private logical-buffer vector extent overflows the machine plan");
        return failure();
      }
      physicalExtent *= 2;
    }
    elements = physicalExtent;
  }
  int64_t elementBits = info.elementType.isIndex()
                            ? 64
                            : info.elementType.getIntOrFloatBitWidth();
  int64_t unitsPerElement = std::max<int64_t>(1, (elementBits + 31) / 32);
  if (elements > std::numeric_limits<int64_t>::max() / unitsPerElement) {
    operation.emitOpError(
        "private logical-buffer capacity overflows the machine plan");
    return failure();
  }
  return elements * unitsPerElement;
}

FailureOr<llvm::DenseMap<Operation *, std::string>>
choosePrivateBufferResidencies(const target::KernelFacts &facts,
                               const DeviceCapabilities &device) {
  llvm::DenseMap<Operation *, std::string> spaces;
  llvm::DenseMap<Operation *, SmallVector<PrivateBufferCandidate>> byOwner;
  int64_t scalarizationBudget =
      std::max<int64_t>(1, device.registersPerUnit / 1024);
  int64_t localRegisterBudget =
      std::max<int64_t>(1, device.registersPerUnit / 128);

  for (const auto &entry : facts.logicalBuffers) {
    Operation *operation = entry.first;
    const target::LogicalBufferFact &buffer = entry.second;
    if (!buffer.owner) {
      operation->emitOpError("private logical buffer has no parallel owner");
      return failure();
    }
    FailureOr<int64_t> logicalUnits =
        privateBufferRegisterUnits(*operation, buffer.info, false);
    if (failed(logicalUnits))
      return failure();
    bool scalarized = buffer.info.shape.size() == 1 &&
                      *logicalUnits <= scalarizationBudget;
    if (buffer.hasUnstructuredDynamicAccess) {
      spaces[operation] = "private_workspace";
      continue;
    }
    FailureOr<int64_t> physicalUnits = scalarized
                                           ? logicalUnits
                                           : privateBufferRegisterUnits(
                                                 *operation, buffer.info, true);
    if (failed(physicalUnits))
      return failure();
    byOwner[buffer.owner].push_back(
        PrivateBufferCandidate{operation, *physicalUnits, scalarized});
  }

  for (auto &entry : byOwner) {
    llvm::sort(entry.second, [](const PrivateBufferCandidate &lhs,
                                const PrivateBufferCandidate &rhs) {
      if (lhs.registerUnits != rhs.registerUnits)
        return lhs.registerUnits < rhs.registerUnits;
      auto lhsNode = lhs.operation->getAttrOfType<IntegerAttr>("intent.node");
      auto rhsNode = rhs.operation->getAttrOfType<IntegerAttr>("intent.node");
      return lhsNode && rhsNode && lhsNode.getInt() < rhsNode.getInt();
    });
    int64_t used = 0;
    for (const PrivateBufferCandidate &candidate : entry.second) {
      bool fits = candidate.registerUnits <= localRegisterBudget - used;
      spaces[candidate.operation] =
          fits ? candidate.scalarized ? "private_scalar_array"
                                      : "private_vector"
               : "private_workspace";
      if (fits)
        used += candidate.registerUnits;
    }
  }
  return spaces;
}

int64_t reusableOperand(Operation &operation) {
  if (operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()) ||
      operation.getName().getStringRef() == "intent.broadcast" ||
      operation.getName().getStringRef() == "intent.transpose")
    return -1;
  Value result = operation.getResult(0);
  auto isStateCarrier = [](Value operand) {
    auto argument = dyn_cast<BlockArgument>(operand);
    Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
    return owner && owner->getName().getStringRef() == "intent.state_stream" &&
           argument.getArgNumber() > 0;
  };
  auto reusable = [&](Value operand) {
    if (operand.getType() != result.getType() ||
        operand.getParentBlock() != operation.getBlock() ||
        (!operand.getDefiningOp() && !isStateCarrier(operand)))
      return false;
    return llvm::all_of(operand.getUsers(), [&](Operation *user) {
      return user == &operation ||
             (user->getBlock() == operation.getBlock() &&
              user->isBeforeInBlock(&operation));
    });
  };
  for (bool requireCarrier : {true, false})
    for (auto [index, operand] : llvm::enumerate(operation.getOperands()))
      if (isStateCarrier(operand) == requireCarrier && reusable(operand))
        return index;
  return -1;
}

struct ValidityBinding {
  SmallVector<int64_t> axes;
  SmallVector<int64_t> nodes;
};

struct PaddingDecision {
  Value value;
  int64_t valueID;
  ValidityBinding validity;
  std::string fill;
};

FailureOr<ValidityBinding>
validityBinding(Value value, ArrayRef<unsigned> requiredAxes,
                const target::KernelFacts &facts, Operation &consumer) {
  ValidityBinding binding;
  auto axes = facts.valueAxes.find(value);
  if (axes == facts.valueAxes.end()) {
    if (isa<RankedTensorType>(value.getType())) {
      consumer.emitOpError("tensor operand has no logical validity provenance");
      return failure();
    }
    return binding;
  }
  for (unsigned position : requiredAxes) {
    if (position >= axes->second.size()) {
      consumer.emitOpError("logical validity axis is outside its tensor rank");
      return failure();
    }
    const target::LogicalAxis &axis = axes->second[position];
    if (!axis.domain)
      continue;
    FailureOr<int64_t> node =
        target::getNodeID(*axis.domain, "logical validity binding");
    if (failed(node))
      return failure();
    binding.axes.push_back(position);
    binding.nodes.push_back(*node);
  }
  return binding;
}

struct PaddingState {
  const target::KernelFacts &facts;
  SmallVectorImpl<PaddingDecision> &decisions;
  llvm::DenseMap<Value, std::string> assumed;

  PaddingState(const target::KernelFacts &facts,
               SmallVectorImpl<PaddingDecision> &decisions)
      : facts(facts), decisions(decisions) {}

  std::optional<std::string> paddingOf(Value value) const {
    return target::inferValuePadding(value, facts, assumed);
  }

  LogicalResult require(Value value, ArrayRef<unsigned> requiredAxes,
                        StringRef fill, Operation &consumer) {
    FailureOr<ValidityBinding> validity =
        validityBinding(value, requiredAxes, facts, consumer);
    if (failed(validity))
      return failure();
    if (validity->nodes.empty())
      return success();
    FailureOr<int64_t> valueID = target::getValueID(
        value, facts.kernel, consumer, "padding binding");
    if (failed(valueID))
      return failure();
    auto existing =
        llvm::find_if(decisions, [&](const PaddingDecision &decision) {
          return decision.value == value;
        });
    if (existing == decisions.end()) {
      decisions.push_back(
          PaddingDecision{value, *valueID, std::move(*validity), fill.str()});
    } else {
      if (existing->fill != fill)
        return consumer.emitOpError(
            "requires incompatible physical padding values");
      for (auto [axis, node] : llvm::zip(validity->axes, validity->nodes)) {
        auto found = llvm::find(existing->validity.axes, axis);
        if (found == existing->validity.axes.end()) {
          existing->validity.axes.push_back(axis);
          existing->validity.nodes.push_back(node);
          continue;
        }
        unsigned position =
            std::distance(existing->validity.axes.begin(), found);
        if (existing->validity.nodes[position] != node)
          return consumer.emitOpError(
              "binds one tensor axis to incompatible validity domains");
      }
    }
    assumed[value] = fill.str();
    return success();
  }
};

class BoundaryNeutralizationProof {
public:
  BoundaryNeutralizationProof(const target::KernelFacts &facts,
                              ArrayRef<PaddingDecision> paddings)
      : facts(facts), paddings(paddings) {}

  bool prove(Operation &load) const {
    auto domains = facts.boundaryDomains.find(&load);
    auto fill = facts.boundaryFills.find(&load);
    if (load.getNumResults() != 1 || domains == facts.boundaryDomains.end() ||
        domains->second.empty() || fill == facts.boundaryFills.end() ||
        fill->second == "none")
      return false;
    llvm::DenseSet<Value> active;
    return proveUses(load.getResult(0), domains->second, active);
  }

private:
  const target::KernelFacts &facts;
  ArrayRef<PaddingDecision> paddings;

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

  std::optional<std::string> materializedPadding(Value value,
                                                 Operation *domain) const {
    std::optional<int64_t> domainNode = nodeOf(domain);
    std::optional<unsigned> tensorAxis = uniqueAxis(value, domain);
    if (!domainNode || !tensorAxis)
      return std::nullopt;
    for (const PaddingDecision &padding : paddings) {
      if (padding.value != value)
        continue;
      for (auto [axis, node] :
           llvm::zip(padding.validity.axes, padding.validity.nodes))
        if (axis == *tensorAxis && node == *domainNode)
          return padding.fill;
    }
    return std::nullopt;
  }

  std::optional<std::string> paddingForDomain(Value value,
                                              Operation *domain) const {
    if (std::optional<std::string> materialized =
            materializedPadding(value, domain))
      return materialized;
    Operation *definition = value.getDefiningOp();
    if (!definition)
      return std::nullopt;
    StringRef name = definition->getName().getStringRef();
    if (name == "intent.view_load")
      return std::nullopt;
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
      return paddingForDomain(definition->getOperand(0), domain);
    if ((name == "intent.cast" || name == "intent.broadcast" ||
         name == "intent.reshape" || name == "intent.transpose") &&
        definition->getNumOperands() >= 1)
      return paddingForDomain(definition->getOperand(0), domain);
    if (name == "intent.unary" && definition->getNumOperands() == 1) {
      std::optional<std::string> operand =
          paddingForDomain(definition->getOperand(0), domain);
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
          paddingForDomain(definition->getOperand(0), domain);
      std::optional<std::string> rhs =
          paddingForDomain(definition->getOperand(1), domain);
      auto logical =
          definition->getAttrOfType<StringAttr>("intent.operator");
      if (!logical)
        return std::nullopt;
      if (logical.getValue() == "multiply" &&
          lhs && rhs && *lhs == "zero" && *rhs == "zero")
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
        auto lhs = pair && pair.size() == 2
                       ? dyn_cast<IntegerAttr>(pair[0])
                       : IntegerAttr();
        auto rhs = pair && pair.size() == 2
                       ? dyn_cast<IntegerAttr>(pair[1])
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
      StringRef name = user->getName().getStringRef();
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

LogicalResult registerPlanHandlers(target::OperationHandlerRegistry &registry,
                                   const KernelFacts &facts,
                                   const PhysicalDecisions &decisions,
                                   const llvm::DenseMap<Operation *, std::string>
                                       &bufferSpaces,
                                   OpBuilder &builder,
                                   PaddingState &paddingState) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.domain_product",
                         "intent.region_end", "intent.assume_in_bounds",
                         "intent.partition",
                         "intent.parallel", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member",
                         "intent.state_stream", "intent.scatter_reduce",
                         "intent.make_record", "intent.extract",
                         "intent.for", "intent.ordered", "intent.if", "intent.while",
                         "intent.condition", "intent.buffer_load",
                         "intent.buffer_store", "intent.yield", "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.buffer", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "logical-buffer binding");
            if (failed(node) || !facts.logicalBuffers.count(&operation))
              return operation.emitOpError(
                  "has no canonical private logical-buffer facts");
            const target::LogicalBufferFact &buffer =
                facts.logicalBuffers.lookup(&operation);
            auto placement = bufferSpaces.find(&operation);
            if (placement == bufferSpaces.end())
              return operation.emitOpError(
                  "has no private logical-buffer residency decision");
            StringRef space = placement->second;
            bool workspace = space == "private_workspace";
            SmallVector<int64_t> ownerNodes;
            if (workspace) {
              auto domains = facts.parallelDomains.find(buffer.owner);
              if (domains == facts.parallelDomains.end() || domains->second.empty())
                return operation.emitOpError(
                    "private workspace has no physical program owners");
              for (Operation *domain : domains->second) {
                FailureOr<int64_t> owner =
                    target::getNodeID(*domain, "private-workspace owner");
                if (failed(owner))
                  return failure();
                ownerNodes.push_back(*owner);
              }
            }
            builder.create<intent::plan::BufferOp>(
                operation.getLoc(), i64(builder, *node), string(builder, space),
                builder.getDenseI64ArrayAttr(ownerNodes));
            return success();
          })))
    return failure();

  auto isStagedContract = [&decisions](Operation &operation) {
    auto node = operation.getAttrOfType<IntegerAttr>("intent.node");
    return node && llvm::any_of(decisions.stages, [&](intent::plan::StageOp stage) {
             return static_cast<int64_t>(stage.getNode()) == node.getInt();
           });
  };
  auto isDirectViewLoad = [](Value value) {
    Operation *definition = value.getDefiningOp();
    return definition &&
           definition->getName().getStringRef() == "intent.view_load";
  };
  auto isContractionDerived = [](Value value) {
    llvm::DenseSet<Value> visited;
    std::function<bool(Value)> reachesContraction = [&](Value current) {
      if (!visited.insert(current).second)
        return false;
      Operation *definition = current.getDefiningOp();
      if (!definition)
        return false;
      if (definition->getName().getStringRef() == "intent.contract")
        return true;
      return llvm::any_of(definition->getOperands(), [&](Value operand) {
        return isa<RankedTensorType>(operand.getType()) &&
               reachesContraction(operand);
      });
    };
    return reachesContraction(value);
  };
  auto sharedContractOperand =
      [isStagedContract, isDirectViewLoad,
       isContractionDerived](Value operand, Operation &contract) {
    if (isStagedContract(contract))
      return true;
    if (!isDirectViewLoad(operand))
      return false;
    if (llvm::all_of(contract.getOperands(), isDirectViewLoad))
      return true;
    return llvm::any_of(contract.getOperands(), [&](Value other) {
      return other != operand && isContractionDerived(other);
    });
  };

  auto bindTransfer = [&, sharedContractOperand](Operation &operation) -> LogicalResult {
    FailureOr<int64_t> node = target::getNodeID(operation, "transfer binding");
    if (failed(node))
      return failure();
    bool load = operation.getName().getStringRef() == "intent.view_load";
    bool returnedAtomic =
        operation.getName().getStringRef() == "intent.atomic_cas" ||
        (operation.getName().getStringRef() == "intent.atomic_add" &&
         operation.getNumResults() == 1);
    bool contractOperand =
        load && operation.getNumResults() == 1 &&
        llvm::any_of(operation.getResult(0).getUsers(), [](Operation *user) {
          return user->getName().getStringRef() == "intent.contract";
        });
    Operation *soleUser = nullptr;
    if (contractOperand &&
        llvm::hasSingleElement(operation.getResult(0).getUsers()))
      soleUser = *operation.getResult(0).user_begin();
    bool sharedOperand =
        soleUser && sharedContractOperand(operation.getResult(0), *soleUser);
    SmallVector<int64_t> domains;
    for (Operation *domain : facts.boundaryDomains.lookup(&operation)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "transfer boundary binding");
      if (failed(domainNode))
        return failure();
      domains.push_back(*domainNode);
    }
    StringRef fill = "none";
    if (load) {
      auto found = facts.boundaryFills.find(&operation);
      if (found != facts.boundaryFills.end())
        fill = found->second;
      else if (contractOperand)
        fill = "zero";
      else if (domains.empty())
        fill = "none";
      else
        return operation.emitOpError("has no resolved load boundary fill");
    }
    StringRef resultSpace = "none";
    if (load || returnedAtomic) {
      resultSpace = sharedOperand
                        ? StringRef("shared")
                        : isa<RankedTensorType>(operation.getResult(0).getType())
                              ? StringRef("private_fragment")
                              : StringRef("private_scalar");
    }
    builder.create<intent::plan::TransferOp>(
        operation.getLoc(), i64(builder, *node),
        builder.getDenseI64ArrayAttr(domains), string(builder, fill),
        builder.getBoolAttr(false),
        string(builder, resultSpace));
    return success();
  };
  if (failed(addHandler(registry, "intent.view_load", bindTransfer)) ||
      failed(addHandler(registry, "intent.view_store", bindTransfer)) ||
      failed(addHandler(registry, "intent.scatter_unique", bindTransfer)) ||
      failed(addHandler(registry, "intent.atomic_add", bindTransfer)) ||
      failed(addHandler(registry, "intent.atomic_cas", bindTransfer)))
    return failure();

  auto bindReduction = [&](Operation &operation) -> LogicalResult {
    FailureOr<int64_t> node =
        target::getNodeID(operation, "reduction binding");
    auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
    auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                         : IntegerAttr();
    if (failed(node) || !axis)
      return failure();
    if (axis.getInt() < 0)
      return operation.emitOpError("has a negative reduction axis");
    std::optional<std::string> inputPadding =
        paddingState.paddingOf(operation.getOperand(0));
    std::optional<std::string> identityPadding =
        operation.getNumOperands() > 1
            ? paddingState.paddingOf(operation.getOperand(1))
            : std::nullopt;
    if (!inputPadding || !identityPadding ||
        *inputPadding != *identityPadding) {
      if (!identityPadding)
        return operation.emitOpError(
            "cannot realize reduction-lane padding without an identity");
      if (failed(paddingState.require(
              operation.getOperand(0),
              {static_cast<unsigned>(axis.getInt())}, *identityPadding,
              operation)))
        return failure();
    }
    builder.create<intent::plan::ReductionOp>(
        operation.getLoc(), i64(builder, *node),
        string(builder, "private_fragment"));
    return success();
  };
  if (failed(addHandler(registry, "intent.reduce", bindReduction)) ||
      failed(addHandler(registry, "intent.arg_reduce", bindReduction)))
    return failure();

  if (failed(addHandler(
          registry, "intent.scan", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "scan binding");
            auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
            if (failed(node) || !axis || axis.getInt() < 0)
              return operation.emitOpError("has no physical scan axis");
            auto inputAxes = facts.valueAxes.find(operation.getOperand(0));
            if (inputAxes == facts.valueAxes.end() ||
                static_cast<size_t>(axis.getInt()) >= inputAxes->second.size() ||
                !inputAxes->second[axis.getInt()].domain)
              return operation.emitOpError(
                  "has no logical domain for its physical scan axis");
            FailureOr<int64_t> axisNode = target::getNodeID(
                *inputAxes->second[axis.getInt()].domain, "scan axis binding");
            auto scanFact = facts.scans.find(&operation);
            Operation *owner = nullptr;
            for (Operation *parent = operation.getParentOp(); parent;
                 parent = parent->getParentOp())
              if (parent->getName().getStringRef() == "intent.parallel") {
                owner = parent;
                break;
              }
            auto ownerDomains = owner ? facts.parallelDomains.find(owner)
                                      : facts.parallelDomains.end();
            if (failed(axisNode) || scanFact == facts.scans.end() ||
                ownerDomains == facts.parallelDomains.end() ||
                ownerDomains->second.empty())
              return failure();
            SmallVector<int64_t> ownerNodes;
            for (Operation *domain : ownerDomains->second) {
              FailureOr<int64_t> ownerNode =
                  target::getNodeID(*domain, "scan result owner");
              if (failed(ownerNode))
                return failure();
              ownerNodes.push_back(*ownerNode);
            }
            std::optional<std::string> inputPadding =
                paddingState.paddingOf(operation.getOperand(0));
            std::optional<std::string> identityPadding =
                paddingState.paddingOf(operation.getOperand(1));
            if (!inputPadding || !identityPadding ||
                *inputPadding != *identityPadding) {
              if (!identityPadding)
                return operation.emitOpError(
                    "cannot realize scan-lane padding without an identity");
              if (failed(paddingState.require(
                      operation.getOperand(0),
                      {static_cast<unsigned>(axis.getInt())}, *identityPadding,
                      operation)))
                return failure();
            }
            bool scalarConsumers = scanFact->second.scalarConsumers;
            ArrayRef<int64_t> producers =
                scalarConsumers ? ArrayRef<int64_t>(scanFact->second.producers)
                                : ArrayRef<int64_t>();
            ArrayRef<int64_t> materializedValues =
                scalarConsumers
                    ? ArrayRef<int64_t>(scanFact->second.materializedValues)
                    : ArrayRef<int64_t>();
            builder.create<intent::plan::ScanOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, "scan_inclusive_add"), i64(builder, *axisNode),
                i64(builder, axis.getInt()),
                string(builder, scalarConsumers ? "private_workspace"
                                                : "private_fragment"),
                string(builder, "private_scalar"),
                string(builder, scalarConsumers ? "scalar_access"
                                                 : "fragment_access"),
                builder.getDenseI64ArrayAttr(ownerNodes),
                builder.getDenseI64ArrayAttr(producers),
                builder.getDenseI64ArrayAttr(materializedValues));
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.indices", "intent.broadcast", "intent.unary",
                         "intent.binary", "intent.compare", "intent.mask",
                         "intent.select", "intent.cast", "intent.full",
                         "intent.zeros", "intent.members", "intent.gather",
                         "intent.reshape", "intent.transpose", "intent.random"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              FailureOr<int64_t> node =
                  target::getNodeID(operation, "pointwise binding");
              if (failed(node) || operation.getNumResults() != 1)
                return failure();
              bool tensor =
                  isa<RankedTensorType>(operation.getResult(0).getType());
              builder.create<intent::plan::PointwiseOp>(
                  operation.getLoc(), i64(builder, *node),
                  string(builder,
                         tensor ? "private_fragment" : "private_scalar"),
                  i64(builder, reusableOperand(operation)),
                  builder.getBoolAttr(
                      target::hasNonnegativeIntegerOperands(operation, facts)));
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.contract",
          [&, sharedContractOperand,
           isStagedContract](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "contract binding");
            if (failed(node))
              return failure();
            bool staged = isStagedContract(operation);
            auto contraction = facts.contractions.find(&operation);
            if (contraction == facts.contractions.end())
              return operation.emitOpError("has no canonical contraction facts");
            auto zeroPadded = [&](Value value) {
              std::optional<std::string> padding = paddingState.paddingOf(value);
              return padding && *padding == "zero";
            };
            if (!staged && !zeroPadded(operation.getOperand(0)) &&
                failed(paddingState.require(
                    operation.getOperand(0),
                    contraction->second.lhsReductionAxes, "zero", operation)))
              return failure();
            if (!staged && !zeroPadded(operation.getOperand(1)) &&
                failed(paddingState.require(
                    operation.getOperand(1),
                    contraction->second.rhsReductionAxes, "zero", operation)))
              return failure();
            auto operandSpace = [&](Value operand) -> StringRef {
              return sharedContractOperand(operand, operation)
                         ? StringRef("shared")
                         : StringRef("private_fragment");
            };
            builder.create<intent::plan::ContractOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, operandSpace(operation.getOperand(0))),
                string(builder, operandSpace(operation.getOperand(1))),
                string(builder, "private_fragment"));
            return success();
          })))
    return failure();

  return success();
}

void emitPaddings(OpBuilder &builder, ArrayRef<PaddingDecision> paddings) {
  for (const PaddingDecision &padding : paddings) {
    builder.create<intent::plan::PaddingOp>(
        padding.value.getLoc(), i64(builder, padding.valueID),
        builder.getDenseI64ArrayAttr(padding.validity.axes),
        builder.getDenseI64ArrayAttr(padding.validity.nodes),
        string(builder, padding.fill));
  }
}

} // namespace

LogicalResult emitMachinePlan(ModuleOp module, const DeviceCapabilities &device,
                              const KernelFacts &facts) {
  func::FuncOp entry = facts.kernel.entry;
  if (!device.matrixUnits && !facts.contractions.empty())
    return entry.emitOpError(
        "requires matrix units unavailable on the selected GPU");
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto realization = builder.create<intent::plan::RealizationOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "gpu"));
  Block &body = realization.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<intent::plan::DeviceOp>(
      entry.getLoc(), i64(builder, device.device));
  FailureOr<PhysicalDecisions> decisions =
      emitPhysicalDecisions(builder, facts);
  if (failed(decisions))
    return failure();
  FailureOr<llvm::DenseMap<Operation *, std::string>> bufferSpaces =
      choosePrivateBufferResidencies(facts, device);
  if (failed(bufferSpaces))
    return failure();
  target::OperationHandlerRegistry registry;
  SmallVector<PaddingDecision> paddings;
  PaddingState paddingState(facts, paddings);
  if (failed(registerPlanHandlers(registry, facts, *decisions, *bufferSpaces,
                                  builder, paddingState)) ||
      failed(target::traverseKernel(entry, registry,
                                    "GPU machine-plan construction")))
    return failure();
  BoundaryNeutralizationProof neutralization(facts, paddings);
  for (intent::plan::TransferOp transfer :
       body.getOps<intent::plan::TransferOp>()) {
    Operation *operation = facts.kernel.nodes.lookup(transfer.getNode());
    if (operation &&
        operation->getName().getStringRef() == "intent.view_load" &&
        neutralization.prove(*operation))
      transfer->setAttr("consumer_neutralized", builder.getBoolAttr(true));
  }
  builder.setInsertionPointToEnd(&body);
  emitPaddings(builder, paddings);
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  if (failed(intent::plan::verifyGpuRealization(realization)))
    return failure();
  return emitSearchSpace(module, facts, *decisions);
}

} // namespace intent::gpu::realization

#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Support/Decisions.h"
#include "Support/Operations.h"
#include "llvm/ADT/STLExtras.h"

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

FailureOr<llvm::DenseMap<Operation *, std::string>>
baselinePrivateBufferResidencies(const target::KernelFacts &facts) {
  llvm::DenseMap<Operation *, std::string> spaces;
  for (const auto &entry : facts.logicalBuffers) {
    Operation *operation = entry.first;
    const target::LogicalBufferFact &buffer = entry.second;
    if (!buffer.owner)
      return operation->emitOpError(
          "private logical buffer has no parallel owner");
    spaces[operation] = buffer.hasUnstructuredDynamicAccess
                            ? "private_workspace"
                            : "private_vector";
  }
  return spaces;
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
    Value materialized = value;
    if (Operation *definition = value.getDefiningOp();
        definition &&
        ::intent::target::semanticOperationName(*definition) == "intent.extract") {
      FailureOr<Value> field = target::resolveRecordField(*definition);
      if (failed(field))
        return failure();
      materialized = *field;
    }
    FailureOr<ValidityBinding> validity =
        validityBinding(materialized, requiredAxes, facts, consumer);
    if (failed(validity))
      return failure();
    if (validity->nodes.empty())
      return success();
    FailureOr<int64_t> valueID = target::getValueID(
        materialized, facts.kernel, consumer, "padding binding");
    if (failed(valueID))
      return failure();
    auto existing =
        llvm::find_if(decisions, [&](const PaddingDecision &decision) {
          return decision.value == materialized;
        });
    if (existing == decisions.end()) {
      decisions.push_back(
          PaddingDecision{materialized, *valueID, std::move(*validity), fill.str()});
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
    assumed[materialized] = fill.str();
    assumed[value] = fill.str();
    return success();
  }
};

LogicalResult registerPlanHandlers(target::OperationHandlerRegistry &registry,
                                   const KernelFacts &facts,
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
                         "intent.for", "intent.if", "intent.while",
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

  auto bindTransfer = [&](Operation &operation) -> LogicalResult {
    FailureOr<int64_t> node = target::getNodeID(operation, "transfer binding");
    if (failed(node))
      return failure();
    bool load = ::intent::target::semanticOperationName(operation) == "intent.view_load";
    bool returnedAtomic =
        ::intent::target::semanticOperationName(operation) == "intent.atomic_cas" ||
        (::intent::target::semanticOperationName(operation) == "intent.atomic_add" &&
         operation.getNumResults() == 1);
    bool contractOperand =
        load && operation.getNumResults() == 1 &&
        llvm::any_of(operation.getResult(0).getUsers(), [](Operation *user) {
          StringRef name = ::intent::target::semanticOperationName(*user);
                 return name == "intent.contract" || name == "intent.scaled_contract" ||
                 name == "intent.sparse_contract";
        });
    target::TensorIndexingKind tensorIndexing =
        target::tensorIndexingKind(operation, facts);
    if (!load && tensorIndexing == target::TensorIndexingKind::compact)
      return operation.emitOpError(
          "compact tensor-indexed stores have no physical coverage realization");
    SmallVector<int64_t> domains;
    for (Operation *domain : facts.boundaryDomains.lookup(&operation)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "transfer boundary binding");
      if (failed(domainNode))
        return failure();
      domains.push_back(*domainNode);
    }
    Value transferred;
    if (load && operation.getNumResults() == 1) {
      transferred = operation.getResult(0);
    } else if (auto valueIndex = operation.getAttrOfType<IntegerAttr>(
                   "intent.value_operand_index");
               valueIndex && valueIndex.getInt() >= 0 &&
               static_cast<unsigned>(valueIndex.getInt()) <
                   operation.getNumOperands()) {
      transferred = operation.getOperand(valueIndex.getInt());
    }
    SmallVector<int64_t> validityTensorAxes;
    SmallVector<int64_t> validityDomainNodes;
    auto transferredAxes = transferred ? facts.valueAxes.find(transferred)
                                       : facts.valueAxes.end();
    if (transferredAxes != facts.valueAxes.end()) {
      for (Operation *domain : facts.boundaryDomains.lookup(&operation)) {
        std::optional<unsigned> tensorAxis;
        for (auto [axis, logical] : llvm::enumerate(transferredAxes->second)) {
          if (logical.domain != domain)
            continue;
          if (tensorAxis) {
            tensorAxis.reset();
            break;
          }
          tensorAxis = axis;
        }
        if (!tensorAxis)
          continue;
        FailureOr<int64_t> domainNode =
            target::getNodeID(*domain, "transfer validity binding");
        if (failed(domainNode))
          return failure();
        validityTensorAxes.push_back(*tensorAxis);
        validityDomainNodes.push_back(*domainNode);
      }
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
      resultSpace = isa<RankedTensorType>(operation.getResult(0).getType())
                        ? StringRef("private_fragment")
                        : StringRef("private_scalar");
    }
    StringRef materialization = "direct";
    StringRef tensorIndexingName =
        tensorIndexing == target::TensorIndexingKind::dataDependent
            ? "data_dependent"
        : tensorIndexing == target::TensorIndexingKind::compact
            ? "compact"
        : tensorIndexing == target::TensorIndexingKind::structured
            ? "structured"
            : "none";
    StringRef coverageSpace =
        tensorIndexing == target::TensorIndexingKind::compact
            ? resultSpace
            : StringRef("none");
    builder.create<intent::plan::TransferOp>(
        operation.getLoc(), i64(builder, *node),
        builder.getDenseI64ArrayAttr(domains),
        builder.getDenseI64ArrayAttr(validityTensorAxes),
        builder.getDenseI64ArrayAttr(validityDomainNodes),
        string(builder, fill),
        builder.getBoolAttr(false),
        string(builder, materialization),
        string(builder, tensorIndexingName),
        string(builder, coverageSpace),
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
    auto components =
        operation.getAttrOfType<IntegerAttr>("intent.component_count");
    auto captures =
        operation.getAttrOfType<IntegerAttr>("intent.capture_count");
    if (!components || components.getInt() <= 0 || !captures ||
        captures.getInt() < 0 ||
        operation.getNumOperands() !=
            2 * static_cast<unsigned>(components.getInt()) +
                static_cast<unsigned>(captures.getInt()))
      return operation.emitOpError("has no physical reduction component schema");
    for (unsigned component = 0;
         component < static_cast<unsigned>(components.getInt()); ++component) {
      std::optional<std::string> inputPadding =
          paddingState.paddingOf(operation.getOperand(component));
      std::optional<std::string> identityPadding = paddingState.paddingOf(
          operation.getOperand(components.getInt() + component));
      if (!inputPadding || !identityPadding ||
          *inputPadding != *identityPadding) {
        if (!identityPadding)
          return operation.emitOpError(
              "cannot realize reduction-lane padding without a literal identity");
        if (failed(paddingState.require(
                operation.getOperand(component),
                {static_cast<unsigned>(axis.getInt())}, *identityPadding,
                operation)))
          return failure();
      }
    }
    auto combineAttr = operation.getAttrOfType<StringAttr>("intent.combine");
    StringRef combine = combineAttr ? combineAttr.getValue() : StringRef();
    if (combine == "maximum" &&
        failed(paddingState.require(
            operation.getOperand(0),
            {static_cast<unsigned>(axis.getInt())}, "negative_infinity",
            operation)))
      return failure();
    builder.create<intent::plan::ReductionOp>(
        operation.getLoc(), i64(builder, *node),
        string(builder, "private_fragment"));
    return success();
  };
  if (failed(addHandler(registry, "intent.reduce", bindReduction)))
    return failure();

  if (failed(addHandler(
          registry, "intent.scan", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "scan binding");
            auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
            auto inclusive =
                operation.getAttrOfType<BoolAttr>("intent.inclusive");
            if (failed(node) || !axis || axis.getInt() < 0 || !inclusive)
              return operation.emitOpError("has no physical scan axis");
            if (!inclusive.getValue())
              return operation.emitOpError(
                  "GPU targets expose inclusive scan only; express an exclusive "
                  "scan explicitly from the inclusive result and its identity");
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
              if (::intent::target::semanticOperationName(*parent) == "intent.parallel") {
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
            auto components = operation.getAttrOfType<IntegerAttr>(
                "intent.component_count");
            auto captures = operation.getAttrOfType<IntegerAttr>(
                "intent.capture_count");
            if (!components || components.getInt() <= 0 || !captures ||
                captures.getInt() < 0 ||
                operation.getNumOperands() !=
                    2 * static_cast<unsigned>(components.getInt()) +
                        static_cast<unsigned>(captures.getInt()))
              return operation.emitOpError(
                  "has no physical scan component schema");
            for (unsigned component = 0;
                 component < static_cast<unsigned>(components.getInt());
                 ++component) {
              inputPadding =
                  paddingState.paddingOf(operation.getOperand(component));
              std::optional<std::string> identityPadding = paddingState.paddingOf(
                  operation.getOperand(components.getInt() + component));
              if (!inputPadding || !identityPadding ||
                  *inputPadding != *identityPadding) {
                if (!identityPadding)
                  return operation.emitOpError(
                      "cannot realize scan-lane padding without a literal identity");
                if (failed(paddingState.require(
                        operation.getOperand(component),
                        {static_cast<unsigned>(axis.getInt())}, *identityPadding,
                        operation)))
                  return failure();
              }
            }
            builder.create<intent::plan::ScanOp>(
                operation.getLoc(), i64(builder, *node),
                i64(builder, *axisNode), i64(builder, axis.getInt()),
                string(builder, "private_fragment"),
                string(builder, "private_scalar"),
                string(builder, "fragment_access"),
                builder.getDenseI64ArrayAttr(ownerNodes),
                builder.getDenseI64ArrayAttr({}),
                builder.getDenseI64ArrayAttr({}));
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
              SmallVector<int64_t> axisNodes;
              auto resultAxes = facts.valueAxes.find(operation.getResult(0));
              if (tensor && resultAxes == facts.valueAxes.end())
                return operation.emitOpError(
                    "has no logical-axis provenance for its physical result");
              if (resultAxes != facts.valueAxes.end()) {
                axisNodes.reserve(resultAxes->second.size());
                for (const target::LogicalAxis &axis : resultAxes->second) {
                  if (!axis.domain) {
                    axisNodes.push_back(-1);
                    continue;
                  }
                  FailureOr<int64_t> axisNode = target::getNodeID(
                      *axis.domain, "pointwise result-axis binding");
                  if (failed(axisNode))
                    return failure();
                  axisNodes.push_back(*axisNode);
                }
              }
              builder.create<intent::plan::PointwiseOp>(
                  operation.getLoc(), i64(builder, *node),
                  string(builder,
                         tensor ? "private_fragment" : "private_scalar"),
                  builder.getBoolAttr(
                      target::hasNonnegativeIntegerOperands(operation, facts)),
                  builder.getDenseI64ArrayAttr(axisNodes));
              return success();
            })))
      return failure();

  auto bindContraction = [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "contract binding");
            if (failed(node))
              return failure();
            auto contraction = facts.contractions.find(&operation);
            if (contraction == facts.contractions.end())
              return operation.emitOpError("has no canonical contraction facts");
            auto zeroPadded = [&](Value value) {
              std::optional<std::string> padding = paddingState.paddingOf(value);
              return padding && *padding == "zero";
            };
            if (!zeroPadded(operation.getOperand(0)) &&
                failed(paddingState.require(
                    operation.getOperand(0),
                    contraction->second.lhsReductionAxes, "zero", operation)))
              return failure();
            if (!zeroPadded(operation.getOperand(1)) &&
                failed(paddingState.require(
                    operation.getOperand(1),
                    contraction->second.rhsReductionAxes, "zero", operation)))
              return failure();
            builder.create<intent::plan::ContractOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder,
                       ::intent::target::semanticOperationName(operation) ==
                               "intent.scaled_contract"
                           ? "scaled_direct"
                           : "direct"),
                string(builder, "none"), string(builder, "none"),
                IntegerAttr(), IntegerAttr(), IntegerAttr(),
                string(builder, "private_fragment"),
                string(builder, "private_fragment"),
                string(builder, "private_fragment"),
                string(builder, "none"), IntegerAttr(), IntegerAttr(),
                IntegerAttr(), IntegerAttr(), IntegerAttr(),
                builder.getBoolAttr(false));
            return success();
          };
  if (failed(addHandler(registry, "intent.contract", bindContraction)) ||
      failed(addHandler(registry, "intent.scaled_contract", bindContraction)))
    return failure();

  if (failed(addHandler(
          registry, "intent.sparse_contract",
          [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "sparse-contract binding");
            auto format = operation.getAttrOfType<StringAttr>("intent.format");
            if (failed(node) || !format || format.getValue() != "two_of_four" ||
                facts.sparseContractions.find(&operation) ==
                    facts.sparseContractions.end())
              return operation.emitOpError(
                  "has no canonical 2:4 sparse contraction facts");
            const target::SparseContractionFact &fact =
                facts.sparseContractions.find(&operation)->second;
            FailureOr<int64_t> row = target::getNodeID(
                *fact.rowDomain, "sparse-contract row-axis binding");
            FailureOr<int64_t> column = target::getNodeID(
                *fact.columnDomain, "sparse-contract column-axis binding");
            FailureOr<int64_t> reduction = target::getNodeID(
                *fact.reductionDomain, "sparse-contract reduction-axis binding");
            if (failed(row) || failed(column) || failed(reduction))
              return failure();
            builder.create<intent::plan::SparseContractOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, format.getValue()), i64(builder, *row),
                i64(builder, *column), i64(builder, *reduction),
                string(builder, "shared"),
                string(builder, "shared"), string(builder, "shared"),
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

LogicalResult buildPhysicalProgram(ModuleOp module,
                                   const DeviceCapabilities &device,
                                   const KernelFacts &facts) {
  func::FuncOp entry = facts.kernel.entry;
  if (!device.matrixUnits && !facts.contractions.empty())
    return entry.emitOpError(
        "requires matrix units unavailable on the selected GPU");
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto program = builder.create<intent::plan::ProgramOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "gpu"));
  Block &body = program.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<intent::plan::DeviceOp>(
      entry.getLoc(), i64(builder, device.device));
  FailureOr<PhysicalDecisions> decisions =
      emitPhysicalDecisions(builder, facts, true);
  if (failed(decisions))
    return failure();
  FailureOr<llvm::DenseMap<Operation *, std::string>> bufferSpaces =
      baselinePrivateBufferResidencies(facts);
  if (failed(bufferSpaces))
    return failure();
  target::OperationHandlerRegistry registry;
  SmallVector<PaddingDecision> paddings;
  PaddingState paddingState(facts, paddings);
  if (failed(registerPlanHandlers(registry, facts, *bufferSpaces,
                                  builder, paddingState)) ||
      failed(target::traverseKernel(entry, registry,
                                    "GPU machine-plan construction")))
    return failure();
  builder.setInsertionPointToEnd(&body);
  emitPaddings(builder, paddings);
  func::FuncOp physicalEntry = cast<func::FuncOp>(entry->clone());
  physicalEntry->setAttr("intent.kind", string(builder, "physical"));
  if (failed(materializePhysicalOperations(physicalEntry))) {
    physicalEntry->destroy();
    return failure();
  }
  builder.insert(physicalEntry.getOperation());
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  return success();
}

} // namespace intent::gpu::realization

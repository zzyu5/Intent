#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Support/Decisions.h"
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

int64_t reusableOperand(Operation &operation) {
  if (operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()) ||
      operation.getName().getStringRef() == "intent.broadcast")
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
    std::optional<std::string> safeFill = target::inferMaskedLaneFill(value);
    if (!safeFill || *safeFill != fill)
      return consumer.emitOpError()
             << "cannot choose one semantics-preserving padding for value; "
                "required "
             << fill;
    Operation *definition = value.getDefiningOp();
    if (!definition ||
        (definition->getName().getStringRef() != "intent.binary" &&
         definition->getName().getStringRef() != "intent.mask"))
      return consumer.emitOpError(
          "producer-fused padding requires a pointwise binary or mask value");
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

LogicalResult registerPlanHandlers(target::OperationHandlerRegistry &registry,
                                   const KernelFacts &facts,
                                   const PhysicalDecisions &decisions,
                                   OpBuilder &builder,
                                   PaddingState &paddingState) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.region_end", "intent.partition",
                         "intent.parallel", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member",
                         "intent.state_stream", "intent.scatter_reduce",
                         "intent.yield", "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  auto bindTransfer = [&](Operation &operation) -> LogicalResult {
    FailureOr<int64_t> node = target::getNodeID(operation, "transfer binding");
    if (failed(node))
      return failure();
    bool load = operation.getName().getStringRef() == "intent.view_load";
    bool contractOperand =
        load && operation.getNumResults() == 1 &&
        llvm::any_of(operation.getResult(0).getUsers(), [](Operation *user) {
          return user->getName().getStringRef() == "intent.contract";
        });
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
    if (load) {
      resultSpace = contractOperand
                        ? StringRef("shared")
                        : isa<RankedTensorType>(operation.getResult(0).getType())
                              ? StringRef("private_fragment")
                              : StringRef("private_scalar");
    }
    builder.create<intent::plan::TransferOp>(
        operation.getLoc(), i64(builder, *node),
        builder.getDenseI64ArrayAttr(domains), string(builder, fill),
        string(builder, resultSpace));
    return success();
  };
  if (failed(addHandler(registry, "intent.view_load", bindTransfer)) ||
      failed(addHandler(registry, "intent.view_store", bindTransfer)) ||
      failed(addHandler(registry, "intent.scatter_unique", bindTransfer)))
    return failure();

  if (failed(addHandler(
          registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "reduction binding");
            auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
            auto axis = axes && axes.size() == 1
                            ? dyn_cast<IntegerAttr>(axes[0])
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
          })))
    return failure();

  for (StringRef name : {"intent.indices", "intent.broadcast", "intent.unary",
                         "intent.binary", "intent.compare", "intent.mask",
                         "intent.cast", "intent.full", "intent.zeros",
                         "intent.members", "intent.gather"})
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
                  i64(builder, reusableOperand(operation)));
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "contract binding");
            if (failed(node))
              return failure();
            bool staged = succeeded(node) && llvm::any_of(
                decisions.stages, [&](intent::plan::StageOp stage) {
                  return static_cast<int64_t>(stage.getNode()) == *node;
                });
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
              if (staged)
                return "shared";
              Operation *definition = operand.getDefiningOp();
              return definition &&
                             definition->getName().getStringRef() ==
                                 "intent.view_load"
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
    Operation *definition = padding.value.getDefiningOp();
    builder.create<intent::plan::PaddingOp>(
        definition->getLoc(), i64(builder, padding.valueID),
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
  target::OperationHandlerRegistry registry;
  SmallVector<PaddingDecision> paddings;
  PaddingState paddingState(facts, paddings);
  if (failed(registerPlanHandlers(registry, facts, *decisions, builder,
                                  paddingState)) ||
      failed(target::traverseKernel(entry, registry,
                                    "GPU machine-plan construction")))
    return failure();
  builder.setInsertionPointToEnd(&body);
  emitPaddings(builder, paddings);
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  if (failed(intent::plan::verifyGpuRealization(realization)))
    return failure();
  return emitSearchSpace(module, facts, *decisions);
}

} // namespace intent::gpu::realization

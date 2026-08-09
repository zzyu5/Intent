#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
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

ArrayAttr strings(OpBuilder &builder, ArrayRef<std::string> values) {
  SmallVector<Attribute> attributes;
  for (const std::string &value : values)
    attributes.push_back(string(builder, value));
  return builder.getArrayAttr(attributes);
}

bool hasTraversal(const ScheduleDecision &schedule, StringRef traversal) {
  return llvm::is_contained(schedule.traversals, traversal);
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

LogicalResult emitStorage(OpBuilder &builder, const target::KernelModel &kernel,
                          const ScheduleDecision &schedule) {
  func::FuncOp entry = kernel.entry;
  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view)
      continue;
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return entry.emitOpError("GPU realization requires ranked views");
    builder.create<intent::plan::StorageOp>(
        entry.getLoc(), i64(builder, argument.valueID),
        string(builder, "external"));
  }

  llvm::DenseSet<int64_t> materialized;
  for (const target::ContractionStage &stage : schedule.stages)
    for (Value value : stage.outputs) {
      FailureOr<int64_t> valueID = target::getValueID(
          value, kernel, *stage.contraction, "staged workspace binding");
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      if (failed(valueID) || !tensor || !materialized.insert(*valueID).second)
        return stage.contraction->emitOpError(
            "has an invalid staged workspace value");
      builder.create<intent::plan::StorageOp>(
          stage.contraction->getLoc(), i64(builder, *valueID),
          string(builder, "workspace"));
    }
  return success();
}

LogicalResult registerPlanHandlers(target::OperationHandlerRegistry &registry,
                                   const OperationFacts &facts,
                                   const ScheduleDecision &schedule,
                                   OpBuilder &builder) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.partition",
                         "intent.yield", "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  auto bindAxis = [&](Operation &operation) -> LogicalResult {
    auto choice = llvm::find_if(schedule.axes, [&](const target::AxisDecision &axis) {
      return axis.domain == &operation;
    });
    if (choice == schedule.axes.end()) {
      bool raggedSource = llvm::any_of(
          facts.semantics.raggedRelations, [&](const auto &entry) {
            return entry.second.outerSource == &operation ||
                   entry.second.memberSource == &operation;
          });
      if (raggedSource)
        return success();
      return operation.emitOpError("is not assigned a GPU physical role");
    }
    FailureOr<int64_t> node = target::getNodeID(operation, "plan binding");
    if (failed(node))
      return failure();
    builder.create<intent::plan::AxisOp>(
        operation.getLoc(), i64(builder, *node),
        i64(builder, choice->sourceAxis), string(builder, choice->role),
        string(builder, choice->tile));
    return success();
  };
  for (StringRef name :
       {"intent.domain", "intent.ragged_outer", "intent.ragged_member"})
    if (failed(addHandler(registry, name, bindAxis)))
      return failure();

  if (failed(addHandler(
          registry, "intent.parallel", [&](Operation &operation) -> LogicalResult {
            if (&operation != schedule.programRoot)
              return success();
            FailureOr<int64_t> node =
                target::getNodeID(operation, "program binding");
            if (failed(node))
              return failure();
            builder.create<intent::plan::ProgramOp>(
                operation.getLoc(), i64(builder, *node),
                builder.getDenseI64ArrayAttr(schedule.workerAxes),
                string(builder, schedule.ownership),
                strings(builder, schedule.traversals));
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.ragged", [&](Operation &operation) -> LogicalResult {
            auto found = facts.semantics.raggedRelations.find(&operation);
            if (found == facts.semantics.raggedRelations.end() ||
                found->second.memberDomains.empty())
              return operation.emitOpError("has no resolved GPU ragged ownership");
            FailureOr<int64_t> node =
                target::getNodeID(operation, "ragged binding");
            FailureOr<int64_t> outer = target::getNodeID(
                *found->second.outerDomain, "ragged outer binding");
            SmallVector<int64_t> members;
            for (Operation *memberDomain : found->second.memberDomains) {
              FailureOr<int64_t> member = target::getNodeID(
                  *memberDomain, "ragged member binding");
              if (failed(member))
                return failure();
              members.push_back(*member);
            }
            if (failed(node) || failed(outer))
              return failure();
            builder.create<intent::plan::RaggedOp>(
                operation.getLoc(), i64(builder, *node), i64(builder, *outer),
                builder.getDenseI64ArrayAttr(members),
                string(builder, found->second.indices
                                    ? "expert_offset_ranges"
                                    : "compact_offset_tiles"));
            return success();
          })))
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
    bool defer = contractOperand &&
                 (hasTraversal(schedule, "grouped") ||
                  hasTraversal(schedule, "staged"));
    SmallVector<int64_t> domains;
    for (Operation *domain : facts.semantics.boundaryDomains.lookup(&operation)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "transfer boundary binding");
      if (failed(domainNode))
        return failure();
      domains.push_back(*domainNode);
    }
    StringRef fill = "none";
    if (load) {
      auto found = facts.semantics.boundaryFills.find(&operation);
      if (found != facts.semantics.boundaryFills.end())
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
        builder.getDenseI64ArrayAttr(domains),
        string(builder, load ? "load" : "store"), string(builder, fill),
        string(builder, contractOperand ? "contract_operand" : "direct"),
        string(builder, resultSpace), builder.getBoolAttr(defer));
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
            auto role = facts.primitiveRoles.find(&operation);
            if (role == facts.primitiveRoles.end())
              return operation.emitOpError("has no resolved GPU reduction role");
            builder.create<intent::plan::ReductionOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, role->second),
                i64(builder, axis.getInt()), builder.getBoolAttr(true),
                string(builder, "private_fragment"),
                string(builder, "private_fragment"));
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast", "intent.full", "intent.zeros",
                         "intent.members", "intent.gather"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              FailureOr<int64_t> node =
                  target::getNodeID(operation, "pointwise binding");
              auto role = facts.primitiveRoles.find(&operation);
              if (failed(node) || role == facts.primitiveRoles.end() ||
                  operation.getNumResults() != 1)
                return failure();
              bool tensor =
                  isa<RankedTensorType>(operation.getResult(0).getType());
              bool feedsContract = tensor && llvm::any_of(
                  operation.getResult(0).getUsers(), [](Operation *user) {
                    return user->getName().getStringRef() == "intent.contract";
                  });
              builder.create<intent::plan::PointwiseOp>(
                  operation.getLoc(), i64(builder, *node),
                  string(builder, role->second),
                  string(builder,
                         tensor ? "private_fragment" : "private_scalar"),
                  i64(builder, reusableOperand(operation)),
                  string(builder,
                         feedsContract ? "contract_operand" : "elementwise"),
                  builder.getBoolAttr(feedsContract &&
                                      hasTraversal(schedule, "staged")));
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "contract binding");
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto pair = reduce && reduce.size() == 1
                            ? dyn_cast<ArrayAttr>(reduce[0])
                            : ArrayAttr();
            auto rhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[1])
                               : IntegerAttr();
            if (failed(node) || !rhsAxis)
              return operation.emitOpError("has no canonical contraction pair");
            bool staged = llvm::any_of(schedule.stages, [&](const auto &stage) {
              return stage.contraction == &operation;
            });
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
                string(builder, "matrix_multiply"),
                string(builder, hasTraversal(schedule, "ordered_stream")
                                    ? "full_row"
                                    : "square"),
                string(builder, "f32"), builder.getBoolAttr(false),
                builder.getBoolAttr(rhsAxis.getInt() == 1),
                string(builder, operandSpace(operation.getOperand(0))),
                string(builder, operandSpace(operation.getOperand(1))),
                string(builder, "private_fragment"));
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.state_stream",
          [&](Operation &operation) -> LogicalResult {
            if (!llvm::is_contained(schedule.stateStreams, &operation))
              return operation.emitOpError("is not the resolved ordered stream");
            FailureOr<int64_t> node =
                target::getNodeID(operation, "stream binding");
            Operation *domain = operation.getOperand(0).getDefiningOp();
            FailureOr<int64_t> axisNode =
                domain ? target::getNodeID(*domain, "stream axis binding")
                       : FailureOr<int64_t>(failure());
            if (failed(node) || failed(axisNode))
              return failure();
            builder.create<intent::plan::StreamOp>(
                operation.getLoc(), i64(builder, *node), i64(builder, *axisNode),
                string(builder, "stream"), string(builder, "forward"),
                string(builder, "private_fragment"), builder.getBoolAttr(true));
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.scatter_reduce",
          [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "atomic binding");
            if (failed(node))
              return failure();
            builder.create<intent::plan::AtomicOp>(
                operation.getLoc(), i64(builder, *node), string(builder, "add"),
                string(builder, "relaxed"), string(builder, "device"));
            return success();
          })))
    return failure();
  return success();
}

LogicalResult emitStages(OpBuilder &builder, const target::KernelModel &kernel,
                         const ScheduleDecision &schedule) {
  for (auto [ordinal, stage] : llvm::enumerate(schedule.stages)) {
    FailureOr<int64_t> node =
        target::getNodeID(*stage.contraction, "stage binding");
    if (failed(node))
      return failure();
    SmallVector<int64_t> inputs;
    SmallVector<int64_t> outputs;
    for (Value value : stage.inputs) {
      FailureOr<int64_t> valueID = target::getValueID(
          value, kernel, *stage.contraction, "stage input binding");
      if (failed(valueID))
        return failure();
      inputs.push_back(*valueID);
    }
    for (Value value : stage.outputs) {
      FailureOr<int64_t> valueID = target::getValueID(
          value, kernel, *stage.contraction, "stage output binding");
      if (failed(valueID))
        return failure();
      outputs.push_back(*valueID);
    }
    builder.create<intent::plan::StageOp>(
        stage.contraction->getLoc(), i64(builder, ordinal), i64(builder, *node),
        builder.getDenseI64ArrayAttr(inputs),
        builder.getDenseI64ArrayAttr(outputs));
  }
  return success();
}

} // namespace

LogicalResult emitMachinePlan(ModuleOp module, const DeviceCapabilities &device,
                              const OperationFacts &facts,
                              const ScheduleDecision &schedule) {
  func::FuncOp entry = facts.semantics.kernel.entry;
  if (!device.matrixUnits && !facts.semantics.contractions.empty())
    return entry.emitOpError(
        "requires matrix units unavailable on the selected GPU");
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  const target::KernelModel &kernel = facts.semantics.kernel;
  auto realization = builder.create<intent::plan::RealizationOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "gpu"));
  Block &body = realization.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<intent::plan::DeviceOp>(
      entry.getLoc(), i64(builder, device.device),
      i64(builder, device.computeUnits),
      i64(builder, device.sharedMemoryPerUnit),
      i64(builder, device.registersPerUnit),
      builder.getBoolAttr(device.matrixUnits),
      builder.getBoolAttr(device.dynamicVectorWidth));
  if (failed(emitStorage(builder, kernel, schedule)))
    return failure();
  target::OperationHandlerRegistry registry;
  if (failed(registerPlanHandlers(registry, facts, schedule, builder)) ||
      failed(target::traverseKernel(entry, registry,
                                    "GPU machine-plan construction")))
    return failure();
  builder.setInsertionPointToEnd(&body);
  if (failed(emitStages(builder, kernel, schedule)))
    return failure();
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  if (failed(intent::plan::verifyGpuRealization(realization)))
    return failure();

  if (!schedule.usesAutotuner)
    return success();
  builder.setInsertionPointToEnd(module.getBody());
  auto search = builder.create<intent::plan::SearchSpaceOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "gpu"));
  Block &searchBody = search.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&searchBody);
  SmallVector<Attribute> keys;
  for (const std::string &key : schedule.autotuneKeys)
    keys.push_back(string(builder, key));
  SmallVector<Attribute> parameters;
  for (const std::string &parameter : schedule.autotuneParameters)
    parameters.push_back(string(builder, parameter));
  builder.create<intent::plan::AutotuneOp>(
      entry.getLoc(), builder.getArrayAttr(keys),
      builder.getArrayAttr(parameters));
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  return intent::plan::verifyGpuSearchSpace(search);
}

} // namespace intent::gpu::realization

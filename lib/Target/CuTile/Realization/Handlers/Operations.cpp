#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::cutile::realization {
namespace {

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter) {
  return registry.add(name,
                      target::OperationHandler{std::move(enter), {}});
}

StringRef reductionLowering(Operation &operation) {
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
  if (!combine || !axes || axes.size() != 1)
    return {};
  if (combine.getValue() == "maximum")
    return "ct.max";
  if (combine.getValue() == "add")
    return "ct.sum";
  return {};
}

StringRef pointwiseLowering(Operation &operation) {
  StringRef name = operation.getName().getStringRef();
  if (name == "intent.broadcast")
    return "alias";
  if (name == "intent.cast")
    return "ct.astype";
  auto logical = operation.getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return {};
  if (name == "intent.unary") {
    if (logical.getValue() == "exp")
      return "ct.exp";
    if (logical.getValue() == "exp2")
      return "ct.exp2";
    if (logical.getValue() == "log")
      return "ct.log";
    if (logical.getValue() == "rsqrt")
      return "ct.rsqrt";
    if (logical.getValue() == "negate")
      return "python_negate";
    return {};
  }
  if (name == "intent.binary") {
    if (logical.getValue() == "add")
      return "python_add";
    if (logical.getValue() == "subtract")
      return "python_subtract";
    if (logical.getValue() == "multiply")
      return "python_multiply";
    if (logical.getValue() == "true_divide")
      return "python_true_divide";
    if (logical.getValue() == "maximum")
      return "ct.maximum";
  }
  return {};
}

LogicalResult registerAnalysisHandlers(target::OperationHandlerRegistry &registry,
                                       OperationFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.partition", "intent.parallel",
                         "intent.view_load", "intent.view_store", "intent.yield",
                         "intent.return", "intent.state_stream", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  for (auto [name, lowering] :
       {std::pair<StringRef, StringRef>{"intent.full", "ct.full"},
        {"intent.zeros", "ct.zeros"}, {"intent.members", "ct.members"},
        {"intent.gather", "expand_dims"}})
    if (failed(addHandler(
            registry, name,
            [&, name, lowering](Operation &operation) -> LogicalResult {
              if (name == "intent.gather") {
                FailureOr<SmallVector<target::IndexTerm>> relation =
                    target::parseIndexRelation(operation);
                if (failed(relation))
                  return failure();
                bool expand = relation->size() == 2 &&
                              (*relation)[0].kind == "full_slice" &&
                              (*relation)[1].kind == "new_axis";
                bool indirect = llvm::any_of(*relation, [](const target::IndexTerm &term) {
                  return term.kind == "value_index";
                });
                if (!expand && !indirect)
                  return operation.emitOpError(
                      "has no mechanical cuTile gather lowering");
                facts.primitiveLowerings[&operation] =
                    expand ? "expand_dims" : "ct.indirect_gather";
                return success();
              }
              facts.primitiveLowerings[&operation] = lowering.str();
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.scatter_reduce",
          [&](Operation &operation) -> LogicalResult {
            facts.primitiveLowerings[&operation] = "ct.atomic_add";
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
            StringRef lowering = reductionLowering(operation);
            if (lowering.empty())
              return operation.emitOpError("has no cuTile reduction primitive");
            facts.primitiveLowerings[&operation] = lowering.str();
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              StringRef lowering = pointwiseLowering(operation);
              if (lowering.empty())
                return operation.emitOpError("has no cuTile pointwise primitive");
              facts.primitiveLowerings[&operation] = lowering.str();
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto accType =
                operation.getAttrOfType<StringAttr>("intent.acc_dtype");
            auto multiply =
                operation.getAttrOfType<StringAttr>("intent.multiply");
            auto combine =
                operation.getAttrOfType<StringAttr>("intent.combine");
            auto pair = reduce && reduce.size() == 1
                            ? dyn_cast<ArrayAttr>(reduce[0])
                            : ArrayAttr();
            auto lhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[0])
                               : IntegerAttr();
            auto rhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[1])
                               : IntegerAttr();
            if (operation.getNumOperands() != 2 ||
                operation.getNumResults() != 1 || !accType ||
                accType.getValue() != "f32" || !multiply ||
                multiply.getValue() != "multiply" || !combine ||
                combine.getValue() != "add" || !lhsAxis || !rhsAxis ||
                lhsAxis.getInt() != 1 ||
                (rhsAxis.getInt() != 0 && rhsAxis.getInt() != 1))
              return operation.emitOpError(
                  "has no semantics-preserving cuTile ct.mma binding");
            facts.primitiveLowerings[&operation] = "ct.mma";
            return success();
          })))
    return failure();
  return success();
}

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

LogicalResult registerBindingHandlers(target::OperationHandlerRegistry &registry,
                                      const OperationFacts &facts,
                                      const PolicyDecision &policy,
                                      OpBuilder &builder) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.partition",
                         "intent.yield", "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  auto bindAxis = [&](Operation &operation) -> LogicalResult {
            auto choice = llvm::find_if(
                policy.axes, [&](const AxisDecision &axis) {
                  return axis.domain == &operation;
                });
            if (choice == policy.axes.end()) {
              bool raggedSource = llvm::any_of(
                  facts.semantics.raggedRelations, [&](const auto &entry) {
                    return entry.second.outerSource == &operation;
                  });
              if (raggedSource)
                return success();
              return operation.emitOpError(
                  "is not assigned a cuTile physical role");
            }
            FailureOr<int64_t> node =
                target::getNodeID(operation, "plan binding");
            if (failed(node))
              return failure();
            builder.create<plan::AxisOp>(
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
          registry, "intent.ragged", [&](Operation &operation) -> LogicalResult {
            auto found = facts.semantics.raggedRelations.find(&operation);
            if (found == facts.semantics.raggedRelations.end() ||
                found->second.memberDomains.size() != 1)
              return operation.emitOpError(
                  "has no resolved cuTile ragged ownership");
            FailureOr<int64_t> node =
                target::getNodeID(operation, "ragged binding");
            FailureOr<int64_t> outer = target::getNodeID(
                *found->second.outerDomain, "ragged outer binding");
            FailureOr<int64_t> member = target::getNodeID(
                *found->second.memberDomains.front(), "ragged member binding");
            if (failed(node) || failed(outer) || failed(member))
              return failure();
            builder.create<plan::RaggedOp>(
                operation.getLoc(), i64(builder, *node), i64(builder, *outer),
                i64(builder, *member),
                string(builder, "expert_offset_ranges"));
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.parallel", [&](Operation &operation) -> LogicalResult {
            if (&operation != policy.programRoot)
              return success();
            FailureOr<int64_t> node =
                target::getNodeID(operation, "program binding");
            if (failed(node))
              return failure();
            builder.create<plan::ProgramOp>(
                operation.getLoc(), i64(builder, *node),
                builder.getDenseI64ArrayAttr(policy.workerAxes),
                string(builder, policy.traversal), string(builder, policy.mapping),
                i64(builder, policy.groupSize));
            return success();
          })))
    return failure();

  auto bindBoundary = [&](Operation &operation) -> LogicalResult {
    if (facts.semantics.wholeViewLoads.contains(&operation))
      return success();
    FailureOr<int64_t> node =
        target::getNodeID(operation, "boundary binding");
    if (failed(node))
      return failure();
    SmallVector<int64_t> domains;
    for (Operation *domain :
         facts.semantics.boundaryDomains.lookup(&operation)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "boundary binding");
      if (failed(domainNode))
        return failure();
      domains.push_back(*domainNode);
    }
    bool load = operation.getName().getStringRef() == "intent.view_load";
    bool persistent = policy.mapping == "persistent_rows";
    StringRef access = persistent ? (load ? "gather" : "scatter")
                                  : (load ? "load" : "store");
    builder.create<plan::BoundaryOp>(
        operation.getLoc(), i64(builder, *node),
        builder.getDenseI64ArrayAttr(domains), string(builder, access),
        string(builder, facts.semantics.boundaryFills.lookup(&operation)),
        builder.getBoolAttr(persistent));
    return success();
  };
  if (failed(addHandler(registry, "intent.view_load", bindBoundary)) ||
      failed(addHandler(registry, "intent.view_store", bindBoundary)))
    return failure();

  if (failed(addHandler(
          registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "primitive binding");
            auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
            auto axis = axes && axes.size() == 1
                            ? dyn_cast<IntegerAttr>(axes[0])
                            : IntegerAttr();
            if (failed(node) || !axis)
              return failure();
            builder.create<plan::ReductionOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, facts.primitiveLowerings.lookup(&operation)),
                i64(builder, axis.getInt()), builder.getBoolAttr(true));
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast", "intent.full", "intent.zeros",
                         "intent.members"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              FailureOr<int64_t> node =
                  target::getNodeID(operation, "primitive binding");
              if (failed(node))
                return failure();
              builder.create<plan::PointwiseOp>(
                  operation.getLoc(), i64(builder, *node),
                  string(builder, facts.primitiveLowerings.lookup(&operation)));
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.gather", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "primitive binding");
            if (failed(node))
              return failure();
            StringRef lowering = policy.mapping == "multi_axis_stream"
                                     ? "alias_column"
                                     : facts.primitiveLowerings.lookup(&operation);
            builder.create<plan::PointwiseOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, lowering));
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
            builder.create<plan::AtomicOp>(
                operation.getLoc(), i64(builder, *node),
                string(builder, "ct.atomic_add"), string(builder, "relaxed"),
                string(builder, "device"));
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            FailureOr<int64_t> node =
                target::getNodeID(operation, "primitive binding");
            if (failed(node))
              return failure();
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto pair = reduce && reduce.size() == 1
                            ? dyn_cast<ArrayAttr>(reduce[0])
                            : ArrayAttr();
            auto lhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[0])
                               : IntegerAttr();
            auto rhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[1])
                               : IntegerAttr();
            if (!lhsAxis || !rhsAxis)
              return operation.emitOpError("has no canonical contraction pair");
            builder.create<plan::ContractOp>(
                operation.getLoc(), i64(builder, *node), string(builder, "ct.mma"),
                string(builder, "f32"), builder.getBoolAttr(false),
                builder.getBoolAttr(rhsAxis.getInt() == 1));
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.state_stream",
          [&](Operation &operation) -> LogicalResult {
            if (&operation != policy.stateStream)
              return operation.emitOpError("is not the resolved ordered stream");
            FailureOr<int64_t> node =
                target::getNodeID(operation, "stream binding");
            Operation *domain = operation.getOperand(0).getDefiningOp();
            FailureOr<int64_t> axisNode = domain
                                              ? target::getNodeID(
                                                    *domain, "stream axis binding")
                                              : FailureOr<int64_t>(failure());
            if (failed(node) || failed(axisNode))
              return failure();
            builder.create<plan::StreamOp>(
                operation.getLoc(), i64(builder, *node), i64(builder, *axisNode),
                string(builder, "TILE_SIZE_N"), string(builder, "forward"),
                string(builder, "register"));
            return success();
          })))
    return failure();
  return success();
}

} // namespace

LogicalResult analyzeOperations(OperationFacts &facts) {
  if (failed(target::analyzeKernelFacts(facts.semantics)))
    return failure();
  target::OperationHandlerRegistry registry;
  if (failed(registerAnalysisHandlers(registry, facts)))
    return facts.semantics.kernel.entry.emitOpError(
        "failed to construct cuTile handlers");
  return target::traverseKernel(facts.semantics.kernel.entry, registry,
                                "cuTile realization analysis");
}

LogicalResult emitPlan(ModuleOp module, const TargetOptions &targetOptions,
                       const OperationFacts &facts,
                       const PolicyDecision &policy) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  const target::KernelModel &kernel = facts.semantics.kernel;
  func::FuncOp entry = kernel.entry;
  auto realization = builder.create<intent::plan::RealizationOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "cutile"));
  Block &body = realization.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<plan::TargetOp>(
      entry.getLoc(), string(builder, targetOptions.architecture),
      i64(builder, targetOptions.device));

  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view)
      continue;
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return entry.emitOpError("cuTile requires ranked external views");
    SmallVector<int64_t> order;
    for (int64_t dimension = 0; dimension < tensor.getRank(); ++dimension)
      order.push_back(dimension);
    builder.create<plan::StorageOp>(entry.getLoc(), i64(builder, argument.valueID),
                                    string(builder, "global"));
    builder.create<plan::LayoutOp>(entry.getLoc(), i64(builder, argument.valueID),
                                   builder.getDenseI64ArrayAttr(order));
  }

  llvm::DenseSet<int64_t> materialized;
  for (const target::ContractionStage &stage : policy.stages)
    for (Value value : stage.outputs) {
      FailureOr<int64_t> valueID =
          target::getValueID(value, kernel, *stage.contraction,
                             "staged workspace binding");
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      if (failed(valueID) || !tensor || !materialized.insert(*valueID).second)
        return stage.contraction->emitOpError(
            "has an invalid staged workspace value");
      SmallVector<int64_t> order;
      for (int64_t dimension = 0; dimension < tensor.getRank(); ++dimension)
        order.push_back(dimension);
      builder.create<plan::StorageOp>(
          stage.contraction->getLoc(), i64(builder, *valueID),
          string(builder, "global"));
      builder.create<plan::LayoutOp>(
          stage.contraction->getLoc(), i64(builder, *valueID),
          builder.getDenseI64ArrayAttr(order));
    }

  target::OperationHandlerRegistry registry;
  if (failed(registerBindingHandlers(registry, facts, policy, builder)) ||
      failed(target::traverseKernel(entry, registry,
                                    "cuTile plan construction")))
    return failure();

  builder.setInsertionPointToEnd(&body);
  for (auto [ordinal, stage] : llvm::enumerate(policy.stages)) {
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
    builder.create<plan::StageOp>(
        stage.contraction->getLoc(), i64(builder, ordinal), i64(builder, *node),
        builder.getDenseI64ArrayAttr(inputs),
        builder.getDenseI64ArrayAttr(outputs));
  }
  if (!policy.usesAutotuner) {
    FailureOr<int64_t> loopNode =
        target::getNodeID(*policy.programRoot, "fixed launch binding");
    if (failed(loopNode))
      return failure();
    builder.create<plan::LaunchOp>(
        policy.programRoot->getLoc(), i64(builder, *loopNode),
        string(builder, "static_persistent"),
        i64(builder, policy.fixedOccupancy));
  }
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  if (failed(plan::verifyCuTileRealization(realization)))
    return failure();
  if (policy.usesAutotuner) {
    if (failed(emitAutotuneSpace(module, entry, targetOptions, policy, builder)))
      return failure();
    auto search = *module.getOps<intent::plan::SearchSpaceOp>().begin();
    if (failed(plan::verifyCuTileSearchSpace(search)))
      return failure();
  }
  return success();
}

} // namespace intent::cutile::realization

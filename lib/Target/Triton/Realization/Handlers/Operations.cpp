#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::triton::realization {
namespace {

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter) {
  if (failed(registry.add(name, target::OperationHandler{std::move(enter), {}})))
    return failure();
  return success();
}

StringRef reductionLowering(Operation &operation) {
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
  auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                      : IntegerAttr();
  if (!combine || !axis)
    return {};
  if (combine.getValue() == "maximum")
    return "tl.max";
  if (combine.getValue() == "add")
    return "tl.sum";
  return {};
}

StringRef pointwiseLowering(Operation &operation) {
  StringRef name = operation.getName().getStringRef();
  if (name == "intent.broadcast")
    return "alias";
  if (name == "intent.cast")
    return "tl.cast";
  auto logical = operation.getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return {};
  if (name == "intent.unary") {
    if (logical.getValue() == "exp")
      return "tl.exp";
    if (logical.getValue() == "exp2")
      return "tl.exp2";
    if (logical.getValue() == "log")
      return "tl.log";
    if (logical.getValue() == "rsqrt")
      return "tl.rsqrt";
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
      return "tl.maximum";
  }
  return {};
}

LogicalResult analyzeBoundary(Operation &operation, OperationFacts &facts,
                              StringRef fill) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Operation *> domains;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind == "slice")
      continue;
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "Triton boundary analysis requires one value per index term");
    FailureOr<Operation *> domain =
        resolveDomain(operation.getOperand(*term.operands.front()), facts, operation);
    if (failed(domain))
      return failure();
    domains.push_back(*domain);
  }
  if (domains.empty())
    return operation.emitOpError("has no domain-bound index for Triton");
  facts.boundaryDomains[&operation] = std::move(domains);
  facts.boundaryFills[&operation] = fill.str();
  return success();
}

LogicalResult recordLoadDomains(Operation &operation, OperationFacts &facts) {
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Operation *> domains;
  for (const target::IndexTerm &term : *relation) {
    if (term.kind != "region_index")
      continue;
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "region_index provenance requires one dynamic operand");
    FailureOr<Operation *> domain = resolveDomain(
        operation.getOperand(*term.operands.front()), facts, operation);
    if (failed(domain))
      return failure();
    domains.push_back(*domain);
  }
  auto tensor = operation.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                    : RankedTensorType();
  if (!tensor || static_cast<size_t>(tensor.getRank()) != domains.size())
    return operation.emitOpError(
        "indexed region provenance does not match the loaded tensor rank");
  facts.valueDomains[operation.getResult(0)] = std::move(domains);
  return success();
}

LogicalResult propagatePointwiseDomains(Operation &operation,
                                        OperationFacts &facts) {
  if (operation.getNumResults() != 1)
    return operation.emitOpError("pointwise provenance requires one result");
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  if (!result)
    return success();
  SmallVector<Operation *> selected;
  for (Value operand : operation.getOperands()) {
    auto found = facts.valueDomains.find(operand);
    if (found == facts.valueDomains.end())
      continue;
    if (selected.empty() || found->second.size() > selected.size())
      selected = found->second;
    else if (found->second.size() == selected.size() && found->second != selected)
      return operation.emitOpError(
          "pointwise operands carry incompatible logical domains");
  }
  if (!selected.empty() &&
      selected.size() != static_cast<size_t>(result.getRank()))
    return operation.emitOpError(
        "pointwise logical domains do not match the result rank");
  facts.valueDomains[operation.getResult(0)] = std::move(selected);
  return success();
}

LogicalResult registerAnalysisHandlers(target::OperationHandlerRegistry &registry,
                                       OperationFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.yield",
                         "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(registry, "intent.domain", [&](Operation &operation) -> LogicalResult {
        if (operation.getNumOperands() < 2 || operation.getNumResults() != 1)
          return operation.emitOpError("has no canonical domain schema");
        Operation *start = operation.getOperand(0).getDefiningOp();
        Operation *stop = operation.getOperand(1).getDefiningOp();
        auto startValue =
            start ? start->getAttrOfType<IntegerAttr>("intent.value")
                  : IntegerAttr();
        auto axis = stop ? stop->getAttrOfType<IntegerAttr>("intent.axis")
                         : IntegerAttr();
        if (!start || start->getName().getStringRef() != "intent.constant" ||
            !startValue || startValue.getInt() != 0 || !stop ||
            stop->getName().getStringRef() != "intent.dim" || !axis ||
            stop->getNumOperands() != 1)
          return operation.emitOpError(
              "Triton currently requires zero-based ABI dimension domains");
        facts.domainSources[&operation] = stop->getOperand(0);
        facts.domainSourceAxes[&operation] = axis.getInt();
        return success();
      })))
    return failure();

  if (failed(addHandler(registry, "intent.partition", [&](Operation &operation) -> LogicalResult {
        if (operation.getNumOperands() != 1 || operation.getNumResults() != 1)
          return operation.emitOpError("has no canonical partition schema");
        Operation *domain = operation.getOperand(0).getDefiningOp();
        if (!domain || !facts.domainSourceAxes.count(domain))
          return operation.emitOpError(
              "Triton tiled partitions currently require a source domain");
        auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
        auto extent = operation.getAttrOfType<DictionaryAttr>("intent.extent");
        auto name = extent ? extent.getAs<StringAttr>("name") : StringAttr();
        if (!mode || mode.getValue() != "extent" || !name || name.getValue().empty())
          return operation.emitOpError(
              "Triton tiled partitions require a named auto extent");
        facts.partitionDomains[&operation] = domain;
        return success();
      })))
    return failure();

  if (failed(addHandler(registry, "intent.parallel", [&](Operation &operation) -> LogicalResult {
        if (operation.getNumOperands() != 1 || operation.getNumRegions() != 1 ||
            !llvm::hasSingleElement(operation.getRegion(0)) ||
            operation.getRegion(0).front().getNumArguments() != 1 ||
            operation.getNumResults() != 0)
          return operation.emitOpError(
              "Triton program ownership requires one stateless region");
        Operation *source = operation.getOperand(0).getDefiningOp();
        if (!source || (!facts.domainSourceAxes.count(source) &&
                        !facts.partitionDomains.count(source)))
          return operation.emitOpError(
              "Triton parallel ownership requires a domain or partition");
        facts.parallels.push_back(&operation);
        return success();
      })))
    return failure();

  if (failed(addHandler(registry, "intent.view_load", [&](Operation &operation) -> LogicalResult {
        if (operation.getNumOperands() < 2 || operation.getNumResults() != 1 ||
            !isa<intent::ViewType>(operation.getOperand(0).getType()))
          return operation.emitOpError("has no canonical external-view load schema");
        bool feedsContract = llvm::any_of(operation.getResult(0).getUsers(),
                                         [](Operation *user) {
                                           return user->getName().getStringRef() ==
                                                  "intent.contract";
                                         });
        if (!feedsContract && !proveMaskedLaneNeutrality(operation.getResult(0)))
          return operation.emitOpError(
              "cannot prove a semantics-preserving masked-load fill");
        if (failed(analyzeBoundary(
                operation, facts,
                feedsContract ? "zero" : "negative_infinity")))
          return failure();
        return recordLoadDomains(operation, facts);
      })))
    return failure();

  if (failed(addHandler(registry, "intent.view_store", [&](Operation &operation) -> LogicalResult {
        auto valueIndex =
            operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
        if (!valueIndex || valueIndex.getInt() <= 0 ||
            static_cast<unsigned>(valueIndex.getInt()) >= operation.getNumOperands())
          return operation.emitOpError("has no canonical external-view store schema");
        return analyzeBoundary(operation, facts, "none");
      })))
    return failure();

  if (failed(addHandler(registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
        StringRef lowering = reductionLowering(operation);
        if (lowering.empty())
          return operation.emitOpError("has no Triton reduction primitive");
        auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
        auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                            : IntegerAttr();
        auto input = operation.getNumOperands() > 0
                         ? facts.valueDomains.find(operation.getOperand(0))
                         : facts.valueDomains.end();
        if (!axis || axis.getInt() < 0 || input == facts.valueDomains.end() ||
            static_cast<size_t>(axis.getInt()) >= input->second.size())
          return operation.emitOpError(
              "reduction axis has no logical-domain provenance");
        facts.vectorDomains.insert(input->second[axis.getInt()]);
        SmallVector<Operation *> resultDomains = input->second;
        resultDomains.erase(resultDomains.begin() + axis.getInt());
        if (operation.getNumResults() == 1)
          facts.valueDomains[operation.getResult(0)] = std::move(resultDomains);
        facts.primitiveLowerings[&operation] = lowering.str();
        return success();
      })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast"})
    if (failed(addHandler(registry, name, [&](Operation &operation) -> LogicalResult {
          StringRef lowering = pointwiseLowering(operation);
          if (lowering.empty())
            return operation.emitOpError("has no Triton pointwise primitive");
          if (failed(propagatePointwiseDomains(operation, facts)))
            return failure();
          facts.primitiveLowerings[&operation] = lowering.str();
          return success();
        })))
      return failure();

  if (failed(addHandler(registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
        auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
        auto accType = operation.getAttrOfType<StringAttr>("intent.acc_dtype");
        auto multiply = operation.getAttrOfType<StringAttr>("intent.multiply");
        auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
        if (operation.getNumOperands() != 2 || operation.getNumResults() != 1 ||
            !reduce || reduce.size() != 1 || !accType ||
            accType.getValue() != "f32" || !multiply ||
            multiply.getValue() != "multiply" || !combine ||
            combine.getValue() != "add")
          return operation.emitOpError(
              "has no semantics-preserving Triton tl.dot binding");
        auto pair = dyn_cast<ArrayAttr>(reduce[0]);
        auto lhsAxis = pair && pair.size() == 2
                           ? dyn_cast<IntegerAttr>(pair[0])
                           : IntegerAttr();
        auto rhsAxis = pair && pair.size() == 2
                           ? dyn_cast<IntegerAttr>(pair[1])
                           : IntegerAttr();
        if (!lhsAxis || !rhsAxis || lhsAxis.getInt() != 1 ||
            rhsAxis.getInt() != 0)
          return operation.emitOpError(
              "Triton tl.dot currently binds contraction pair (1, 0)");
        auto lhs = facts.valueDomains.find(operation.getOperand(0));
        auto rhs = facts.valueDomains.find(operation.getOperand(1));
        if (lhs == facts.valueDomains.end() || rhs == facts.valueDomains.end() ||
            static_cast<size_t>(lhsAxis.getInt()) >= lhs->second.size() ||
            static_cast<size_t>(rhsAxis.getInt()) >= rhs->second.size() ||
            lhs->second[lhsAxis.getInt()] != rhs->second[rhsAxis.getInt()])
          return operation.emitOpError(
              "contraction pair has no shared logical-domain provenance");
        Operation *reductionDomain = lhs->second[lhsAxis.getInt()];
        facts.streamedReductionDomains.insert(reductionDomain);
        SmallVector<Operation *> resultDomains;
        for (auto [axis, domain] : llvm::enumerate(lhs->second))
          if (axis != static_cast<size_t>(lhsAxis.getInt()))
            resultDomains.push_back(domain);
        for (auto [axis, domain] : llvm::enumerate(rhs->second))
          if (axis != static_cast<size_t>(rhsAxis.getInt()))
            resultDomains.push_back(domain);
        facts.valueDomains[operation.getResult(0)] = std::move(resultDomains);
        facts.primitiveLowerings[&operation] = "tl.dot";
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

  if (failed(addHandler(registry, "intent.domain", [&](Operation &operation) -> LogicalResult {
        auto choice = llvm::find_if(policy.axes, [&](const AxisDecision &axis) {
          return axis.domain == &operation;
        });
        if (choice == policy.axes.end())
          return operation.emitOpError("is not assigned a Triton physical role");
        FailureOr<int64_t> node = target::getNodeID(operation, "plan binding");
        if (failed(node))
          return failure();
        builder.create<plan::AxisOp>(
            operation.getLoc(), i64(builder, *node),
            i64(builder, choice->sourceAxis), string(builder, choice->role),
            string(builder, choice->tile));
        return success();
      })))
    return failure();

  if (failed(addHandler(registry, "intent.parallel", [&](Operation &operation) -> LogicalResult {
        if (&operation != policy.programRoot)
          return success();
        FailureOr<int64_t> node = target::getNodeID(operation, "program binding");
        if (failed(node))
          return failure();
        builder.create<plan::ProgramOp>(
            operation.getLoc(), i64(builder, *node),
            builder.getDenseI64ArrayAttr(policy.workerAxes),
            string(builder, policy.traversal), string(builder, policy.mapping));
        return success();
      })))
    return failure();

  auto bindBoundary = [&](Operation &operation) -> LogicalResult {
    FailureOr<int64_t> node = target::getNodeID(operation, "boundary binding");
    if (failed(node))
      return failure();
    SmallVector<int64_t> domains;
    for (Operation *domain : facts.boundaryDomains.lookup(&operation)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "boundary binding");
      if (failed(domainNode))
        return failure();
      domains.push_back(*domainNode);
    }
    builder.create<plan::BoundaryOp>(
        operation.getLoc(), i64(builder, *node),
        builder.getDenseI64ArrayAttr(domains), string(builder, "index_lt_extent"),
        string(builder, facts.boundaryFills.lookup(&operation)),
        string(builder, "predicate"));
    return success();
  };
  if (failed(addHandler(registry, "intent.view_load", bindBoundary)) ||
      failed(addHandler(registry, "intent.view_store", bindBoundary)))
    return failure();

  if (failed(addHandler(registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
        FailureOr<int64_t> node = target::getNodeID(operation, "primitive binding");
        auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
        auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                            : IntegerAttr();
        if (failed(node) || !axis)
          return failure();
        builder.create<plan::ReductionOp>(
            operation.getLoc(), i64(builder, *node),
            string(builder, facts.primitiveLowerings.lookup(&operation)),
            i64(builder, axis.getInt()));
        return success();
      })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast"})
    if (failed(addHandler(registry, name, [&](Operation &operation) -> LogicalResult {
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

  if (failed(addHandler(registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
        FailureOr<int64_t> node = target::getNodeID(operation, "primitive binding");
        if (failed(node))
          return failure();
        builder.create<plan::ContractOp>(
            operation.getLoc(), i64(builder, *node), string(builder, "tl.dot"),
            string(builder, "f32"));
        return success();
      })))
    return failure();
  return success();
}

} // namespace

FailureOr<Operation *> resolveDomain(Value indexedValue,
                                     const OperationFacts &facts,
                                     Operation &consumer) {
  if (Operation *definition = indexedValue.getDefiningOp()) {
    if (facts.domainSourceAxes.count(definition))
      return definition;
  }
  auto argument = dyn_cast<BlockArgument>(indexedValue);
  Operation *owner = argument ? argument.getOwner()->getParentOp() : nullptr;
  if (!owner || owner->getName().getStringRef() != "intent.parallel" ||
      owner->getNumOperands() != 1) {
    consumer.emitOpError("indexes with a value not owned by a parallel region");
    return failure();
  }
  Operation *source = owner->getOperand(0).getDefiningOp();
  if (facts.domainSourceAxes.count(source))
    return source;
  auto partition = facts.partitionDomains.find(source);
  if (partition != facts.partitionDomains.end())
    return partition->second;
  consumer.emitOpError("cannot resolve an indexed region to its source domain");
  return failure();
}

LogicalResult analyzeOperations(OperationFacts &facts) {
  target::OperationHandlerRegistry registry;
  if (failed(registerAnalysisHandlers(registry, facts)))
    return facts.kernel.entry.emitOpError("failed to construct Triton handlers");
  return target::traverseKernel(facts.kernel.entry, registry,
                                "Triton realization analysis");
}

LogicalResult emitPlan(ModuleOp module, const TargetOptions &targetOptions,
                       const OperationFacts &facts,
                       const PolicyDecision &policy) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto realization = builder.create<intent::plan::RealizationOp>(
      facts.kernel.entry.getLoc(),
      FlatSymbolRefAttr::get(module.getContext(), facts.kernel.entry.getName()),
      string(builder, "triton"));
  Block &body = realization.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<plan::TargetOp>(
      facts.kernel.entry.getLoc(), string(builder, targetOptions.architecture),
      i64(builder, targetOptions.device), i64(builder, targetOptions.warpSize));

  for (const target::ABIArgument &argument : facts.kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view)
      continue;
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return facts.kernel.entry.emitOpError(
          "Triton requires ranked external views");
    SmallVector<int64_t> order;
    for (int64_t dimension = 0; dimension < tensor.getRank(); ++dimension)
      order.push_back(dimension);
    builder.create<plan::StorageOp>(
        facts.kernel.entry.getLoc(), i64(builder, argument.valueID),
        string(builder, "global"));
    builder.create<plan::LayoutOp>(
        facts.kernel.entry.getLoc(), i64(builder, argument.valueID),
        string(builder, "row_major"), builder.getDenseI64ArrayAttr(order));
  }

  target::OperationHandlerRegistry registry;
  if (failed(registerBindingHandlers(registry, facts, policy, builder)) ||
      failed(target::traverseKernel(facts.kernel.entry, registry,
                                    "Triton plan construction")))
    return failure();

  builder.setInsertionPointToEnd(&body);
  if (!policy.usesAutotuner) {
    FailureOr<int64_t> loopNode =
        target::getNodeID(*policy.programRoot, "fixed pipeline binding");
    if (failed(loopNode))
      return failure();
    builder.create<plan::PipelineOp>(
        policy.programRoot->getLoc(), i64(builder, *loopNode), i64(builder, 2),
        i64(builder, 4), i64(builder, 200000), builder.getBoolAttr(false),
        builder.getBoolAttr(false));
    builder.create<plan::LaunchOp>(
        policy.programRoot->getLoc(), i64(builder, *loopNode),
        string(builder, "persistent_occupancy"), i64(builder, 8));
  }
  builder.create<intent::plan::YieldOp>(facts.kernel.entry.getLoc());
  if (failed(plan::verifyTritonRealization(realization)))
    return failure();
  if (policy.usesAutotuner) {
    emitAutotuneSpace(module, facts.kernel.entry, policy, builder);
    auto search = *module.getOps<intent::plan::SearchSpaceOp>().begin();
    if (failed(plan::verifyTritonSearchSpace(search)))
      return failure();
  }
  return success();
}

} // namespace intent::triton::realization

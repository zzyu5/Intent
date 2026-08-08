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

LogicalResult registerAnalysisHandlers(target::OperationHandlerRegistry &registry,
                                       OperationFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.partition", "intent.parallel",
                         "intent.view_load", "intent.view_store", "intent.yield",
                         "intent.return", "intent.state_stream"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
        StringRef lowering = reductionLowering(operation);
        if (lowering.empty())
          return operation.emitOpError("has no Triton reduction primitive");
        auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
        auto axis = axes && axes.size() == 1 ? dyn_cast<IntegerAttr>(axes[0])
                                            : IntegerAttr();
        if (!axis)
          return operation.emitOpError("requires one Triton reduction axis");
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
          facts.primitiveLowerings[&operation] = lowering.str();
          return success();
        })))
      return failure();

  for (auto [name, lowering] :
       {std::pair<StringRef, StringRef>{"intent.full", "tl.full"},
        {"intent.zeros", "tl.zeros"}, {"intent.gather", "expand_dims"}})
    if (failed(addHandler(
            registry, name,
            [&, name, lowering](Operation &operation) -> LogicalResult {
              if (name == "intent.gather") {
                FailureOr<SmallVector<target::IndexTerm>> relation =
                    target::parseIndexRelation(operation);
                if (failed(relation) || relation->size() != 2 ||
                    (*relation)[0].kind != "full_slice" ||
                    (*relation)[1].kind != "new_axis")
                  return operation.emitOpError(
                      "has no mechanical Triton gather lowering");
              }
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
            (rhsAxis.getInt() != 0 && rhsAxis.getInt() != 1))
          return operation.emitOpError(
              "Triton tl.dot requires lhs axis 1 and rhs axis 0 or 1");
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
    for (Operation *domain :
         facts.semantics.boundaryDomains.lookup(&operation)) {
      FailureOr<int64_t> domainNode =
          target::getNodeID(*domain, "boundary binding");
      if (failed(domainNode))
        return failure();
      domains.push_back(*domainNode);
    }
    builder.create<plan::BoundaryOp>(
        operation.getLoc(), i64(builder, *node),
        builder.getDenseI64ArrayAttr(domains), string(builder, "index_lt_extent"),
        string(builder, facts.semantics.boundaryFills.lookup(&operation)),
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
                         "intent.cast", "intent.full", "intent.zeros",
                         "intent.gather"})
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
            operation.getLoc(), i64(builder, *node), string(builder, "tl.dot"),
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
                string(builder, "BLOCK_SIZE_K"), string(builder, "forward"),
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
        "failed to construct Triton handlers");
  return target::traverseKernel(facts.semantics.kernel.entry, registry,
                                "Triton realization analysis");
}

LogicalResult emitPlan(ModuleOp module, const TargetOptions &targetOptions,
                       const OperationFacts &facts,
                       const PolicyDecision &policy) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  const target::KernelModel &kernel = facts.semantics.kernel;
  func::FuncOp entry = kernel.entry;
  auto realization = builder.create<intent::plan::RealizationOp>(
      entry.getLoc(),
      FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "triton"));
  Block &body = realization.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<plan::TargetOp>(
      entry.getLoc(), string(builder, targetOptions.architecture),
      i64(builder, targetOptions.device), i64(builder, targetOptions.warpSize));

  for (const target::ABIArgument &argument : kernel.abi.arguments) {
    auto view = dyn_cast<intent::ViewType>(argument.type);
    if (!view)
      continue;
    auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
    if (!tensor)
      return entry.emitOpError(
          "Triton requires ranked external views");
    SmallVector<int64_t> order;
    for (int64_t dimension = 0; dimension < tensor.getRank(); ++dimension)
      order.push_back(dimension);
    builder.create<plan::StorageOp>(
        entry.getLoc(), i64(builder, argument.valueID),
        string(builder, "global"));
    builder.create<plan::LayoutOp>(
        entry.getLoc(), i64(builder, argument.valueID),
        string(builder, "row_major"), builder.getDenseI64ArrayAttr(order));
  }

  target::OperationHandlerRegistry registry;
  if (failed(registerBindingHandlers(registry, facts, policy, builder)) ||
      failed(target::traverseKernel(entry, registry,
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
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  if (failed(plan::verifyTritonRealization(realization)))
    return failure();
  if (policy.usesAutotuner) {
    emitAutotuneSpace(module, entry, policy, builder);
    auto search = *module.getOps<intent::plan::SearchSpaceOp>().begin();
    if (failed(plan::verifyTritonSearchSpace(search)))
      return failure();
  }
  return success();
}

} // namespace intent::triton::realization

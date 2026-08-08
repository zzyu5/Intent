#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

using namespace mlir;

namespace intent::target {
namespace {

LogicalResult addHandler(OperationHandlerRegistry &registry, StringRef name,
                         OperationCallback enter) {
  return registry.add(name, OperationHandler{std::move(enter), {}});
}

LogicalResult analyzeBoundary(Operation &operation, KernelFacts &facts,
                              StringRef fill) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Operation *> domains;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "full_slice" || term.kind == "new_axis" ||
        term.kind == "static_index" || term.kind == "slice")
      continue;
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "boundary analysis requires one value per dynamic index term");
    FailureOr<Operation *> domain = resolveDomain(
        operation.getOperand(*term.operands.front()), facts, operation);
    if (failed(domain))
      return failure();
    domains.push_back(*domain);
  }
  if (domains.empty())
    return operation.emitOpError("has no domain-bound index for realization");
  facts.boundaryDomains[&operation] = std::move(domains);
  facts.boundaryFills[&operation] = fill.str();
  return success();
}

LogicalResult recordLoadDomains(Operation &operation, KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Operation *> domains;
  for (const IndexTerm &term : *relation) {
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
                                        KernelFacts &facts) {
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

LogicalResult registerFactHandlers(OperationHandlerRegistry &registry,
                                   KernelFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.yield",
                         "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.domain", [&](Operation &operation) -> LogicalResult {
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
                  "realization currently requires zero-based ABI dimensions");
            facts.domainSources[&operation] = stop->getOperand(0);
            facts.domainSourceAxes[&operation] = axis.getInt();
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.partition", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 || operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical partition schema");
            Operation *domain = operation.getOperand(0).getDefiningOp();
            if (!domain || !facts.domainSourceAxes.count(domain))
              return operation.emitOpError(
                  "tiled partitions currently require a source domain");
            auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
            auto extent =
                operation.getAttrOfType<DictionaryAttr>("intent.extent");
            auto name = extent ? extent.getAs<StringAttr>("name") : StringAttr();
            if (!mode || mode.getValue() != "extent" || !name ||
                name.getValue().empty())
              return operation.emitOpError(
                  "tiled partitions require a named auto extent");
            facts.partitionDomains[&operation] = domain;
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.parallel", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                operation.getNumRegions() != 1 ||
                !llvm::hasSingleElement(operation.getRegion(0)) ||
                operation.getRegion(0).front().getNumArguments() != 1 ||
                operation.getNumResults() != 0)
              return operation.emitOpError(
                  "parallel ownership requires one stateless region");
            Operation *source = operation.getOperand(0).getDefiningOp();
            if (!source || (!facts.domainSourceAxes.count(source) &&
                            !facts.partitionDomains.count(source)))
              return operation.emitOpError(
                  "parallel ownership requires a domain or partition");
            facts.parallels.push_back(&operation);
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.view_load", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() < 2 ||
                operation.getNumResults() != 1 ||
                !isa<intent::ViewType>(operation.getOperand(0).getType()))
              return operation.emitOpError(
                  "has no canonical external-view load schema");
            bool feedsContract = llvm::any_of(
                operation.getResult(0).getUsers(), [](Operation *user) {
                  return user->getName().getStringRef() == "intent.contract";
                });
            if (!feedsContract &&
                !proveMaskedLaneNeutrality(operation.getResult(0)))
              return operation.emitOpError(
                  "cannot prove a semantics-preserving masked-load fill");
            if (failed(analyzeBoundary(
                    operation, facts,
                    feedsContract ? "zero" : "negative_infinity")))
              return failure();
            return recordLoadDomains(operation, facts);
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
            return analyzeBoundary(operation, facts, "none");
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
            auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
            auto input = operation.getNumOperands() > 0
                             ? facts.valueDomains.find(operation.getOperand(0))
                             : facts.valueDomains.end();
            if (!axes || axes.empty() || input == facts.valueDomains.end())
              return operation.emitOpError(
                  "reduction axes have no logical-domain provenance");
            SmallVector<unsigned> reducedAxes;
            for (Attribute attribute : axes) {
              auto axis = dyn_cast<IntegerAttr>(attribute);
              if (!axis || axis.getInt() < 0 ||
                  static_cast<size_t>(axis.getInt()) >= input->second.size())
                return operation.emitOpError(
                    "reduction axis has no logical-domain provenance");
              reducedAxes.push_back(axis.getInt());
              facts.vectorDomains.insert(input->second[axis.getInt()]);
            }
            llvm::sort(reducedAxes, std::greater<unsigned>());
            SmallVector<Operation *> resultDomains = input->second;
            for (unsigned axis : reducedAxes)
              resultDomains.erase(resultDomains.begin() + axis);
            if (operation.getNumResults() == 1)
              facts.valueDomains[operation.getResult(0)] =
                  std::move(resultDomains);
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              return propagatePointwiseDomains(operation, facts);
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto lhs = operation.getNumOperands() == 2
                           ? facts.valueDomains.find(operation.getOperand(0))
                           : facts.valueDomains.end();
            auto rhs = operation.getNumOperands() == 2
                           ? facts.valueDomains.find(operation.getOperand(1))
                           : facts.valueDomains.end();
            if (!reduce || reduce.empty() || lhs == facts.valueDomains.end() ||
                rhs == facts.valueDomains.end() || operation.getNumResults() != 1)
              return operation.emitOpError(
                  "contraction has no logical-domain provenance");
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
              facts.streamedReductionDomains.insert(
                  lhs->second[lhsAxis.getInt()]);
            }
            SmallVector<Operation *> resultDomains;
            for (auto [axis, domain] : llvm::enumerate(lhs->second))
              if (!lhsReduced.contains(axis))
                resultDomains.push_back(domain);
            for (auto [axis, domain] : llvm::enumerate(rhs->second))
              if (!rhsReduced.contains(axis))
                resultDomains.push_back(domain);
            facts.valueDomains[operation.getResult(0)] =
                std::move(resultDomains);
            return success();
          })))
    return failure();
  return success();
}

} // namespace

FailureOr<Operation *> resolveDomain(Value indexedValue,
                                     const KernelFacts &facts,
                                     Operation &consumer) {
  if (Operation *definition = indexedValue.getDefiningOp())
    if (facts.domainSourceAxes.count(definition))
      return definition;
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

LogicalResult analyzeKernelFacts(KernelFacts &facts) {
  OperationHandlerRegistry registry;
  if (failed(registerFactHandlers(registry, facts)))
    return facts.kernel.entry.emitOpError(
        "failed to construct canonical realization handlers");
  return traverseKernel(facts.kernel.entry, registry,
                        "canonical realization analysis");
}

} // namespace intent::target

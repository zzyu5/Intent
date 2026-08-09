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
    Value indexed = operation.getOperand(*term.operands.front());
    auto indexedAxes = facts.valueAxes.find(indexed);
    if (term.kind == "value_index" && indexedAxes != facts.valueAxes.end()) {
      for (const LogicalAxis &axis : indexedAxes->second)
        if (axis.domain && !llvm::is_contained(domains, axis.domain))
          domains.push_back(axis.domain);
      continue;
    }
    FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
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
  auto known = facts.axisLabels.find(label);
  if (known != facts.axisLabels.end())
    return known->second;
  for (const auto &binding : facts.domainSourceAxes) {
    FailureOr<LogicalAxis> axis = axisFromDomain(*binding.first, facts, consumer);
    if (failed(axis))
      return failure();
    if (axis->extent == label) {
      facts.axisLabels[label] = *axis;
      return axis;
    }
  }
  LogicalAxis implicit{nullptr, label.str()};
  facts.axisLabels[label] = implicit;
  return implicit;
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
      FailureOr<LogicalAxis> axis = axisFromDomain(**domain, facts, operation);
      if (failed(axis))
        return failure();
      axes.push_back(*axis);
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
      FailureOr<LogicalAxis> axis =
          axisFromDomain(**domain, facts, operation);
      if (failed(axis))
        return failure();
      resultAxes.push_back(*axis);
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

LogicalResult propagatePointwiseAxes(Operation &operation, KernelFacts &facts) {
  if (operation.getNumResults() != 1)
    return operation.emitOpError("pointwise provenance requires one result");
  auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  if (!result)
    return success();
  FailureOr<SmallVector<LogicalAxis>> axes =
      axesFromResultShape(operation, 0, facts);
  if (failed(axes))
    return failure();
  return bindResultAxes(operation, 0, std::move(*axes), facts);
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
            if (!outer || !facts.domainSourceAxes.count(outer) || !members ||
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
            facts.domainSources[&operation] = facts.domainSources.lookup(source);
            facts.domainSourceAxes[&operation] =
                facts.domainSourceAxes.lookup(source);
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
            if (found == facts.raggedRelations.end() ||
                !found->second.outerDomain)
              return operation.emitOpError(
                  "does not reference a resolved ragged relation");
            FailureOr<Operation *> selector = resolveDomain(
                operation.getOperand(1), facts, operation);
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
            std::optional<std::string> fill =
                feedsContract ? std::optional<std::string>("zero")
                              : inferMaskedLaneFill(operation.getResult(0));
            if (!fill)
              return operation.emitOpError(
                  "cannot prove a semantics-preserving masked-load fill");
            if (failed(analyzeBoundary(
                    operation, facts, *fill)))
              return failure();
            return recordLoadAxes(operation, facts);
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
                             ? facts.valueAxes.find(operation.getOperand(0))
                             : facts.valueAxes.end();
            if (!axes || axes.empty() || input == facts.valueAxes.end())
              return operation.emitOpError(
                  "reduction axes have no logical-axis provenance");
            SmallVector<unsigned> reducedAxes;
            for (Attribute attribute : axes) {
              auto axis = dyn_cast<IntegerAttr>(attribute);
              if (!axis || axis.getInt() < 0 ||
                  static_cast<size_t>(axis.getInt()) >= input->second.size())
                return operation.emitOpError(
                    "reduction axis has no logical-axis provenance");
              reducedAxes.push_back(axis.getInt());
              if (Operation *domain = input->second[axis.getInt()].domain)
                facts.vectorDomains.insert(domain);
            }
            llvm::sort(reducedAxes, std::greater<unsigned>());
            SmallVector<LogicalAxis> resultAxes = input->second;
            for (unsigned axis : reducedAxes)
              resultAxes.erase(resultAxes.begin() + axis);
            if (operation.getNumResults() == 1)
              return bindResultAxes(operation, 0, std::move(resultAxes), facts);
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast", "intent.compare", "intent.select",
                         "intent.mask"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              return propagatePointwiseAxes(operation, facts);
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

  auto bindScatterWrite = [&](Operation &operation) -> LogicalResult {
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
          "has no canonical unique or additive scatter schema");
    FailureOr<SmallVector<LogicalAxis>> indexedAxes =
        inferIndexedAxes(operation, operation.getOperand(0), facts);
    auto valueAxes =
        facts.valueAxes.find(operation.getOperand(valueIndex.getInt()));
    if (failed(indexedAxes) || valueAxes == facts.valueAxes.end() ||
        *indexedAxes != valueAxes->second)
      return operation.emitOpError(
          "scatter value does not match its indexed destination");
    if (failed(analyzeBoundary(operation, facts, "none")))
      return failure();
    facts.scatterWrites.insert(&operation);
    return success();
  };
  if (failed(addHandler(registry, "intent.scatter_unique", bindScatterWrite)) ||
      failed(addHandler(registry, "intent.scatter_reduce", bindScatterWrite)))
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
    facts.orderedStreamDomains.insert(axisDomain);
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
      if (initialAxes != facts.valueAxes.end() &&
          (yieldedAxes == facts.valueAxes.end() ||
           yieldedAxes->second != initialAxes->second))
        return operation.emitOpError(
            "yielded state changes its logical axis identity");
    }
    return success();
  };
  if (failed(registry.add(
          "intent.state_stream",
          OperationHandler{enterStateStream, leaveStateStream})))
    return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto lhs = operation.getNumOperands() == 2
                           ? facts.valueAxes.find(operation.getOperand(0))
                           : facts.valueAxes.end();
            auto rhs = operation.getNumOperands() == 2
                           ? facts.valueAxes.find(operation.getOperand(1))
                           : facts.valueAxes.end();
            if (!reduce || reduce.empty() || lhs == facts.valueAxes.end() ||
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
            SmallVector<LogicalAxis> resultAxes;
            for (auto [axis, logicalAxis] : llvm::enumerate(lhs->second))
              if (!lhsReduced.contains(axis))
                resultAxes.push_back(logicalAxis);
            for (auto [axis, logicalAxis] : llvm::enumerate(rhs->second))
              if (!rhsReduced.contains(axis))
                resultAxes.push_back(logicalAxis);
            ContractionFact fact;
            fact.operation = &operation;
            fact.lhsAxes = lhs->second;
            fact.rhsAxes = rhs->second;
            fact.resultAxes = resultAxes;
            fact.lhsReductionAxes.assign(lhsReduced.begin(), lhsReduced.end());
            fact.rhsReductionAxes.assign(rhsReduced.begin(), rhsReduced.end());
            llvm::sort(fact.lhsReductionAxes);
            llvm::sort(fact.rhsReductionAxes);
            facts.contractions[&operation] = std::move(fact);
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
    if (facts.domainSourceAxes.count(definition))
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
  if (failed(traverseKernel(facts.kernel.entry, registry,
                            "canonical realization analysis")))
    return failure();

  llvm::DenseSet<Operation *> programDomains;
  for (Operation *parallel : facts.parallels) {
    Operation *source = parallel->getOperand(0).getDefiningOp();
    auto partition = facts.partitionDomains.find(source);
    programDomains.insert(partition == facts.partitionDomains.end()
                              ? source
                              : partition->second);
  }
  for (const auto &entry : facts.valueAxes)
    for (const LogicalAxis &axis : entry.second)
      if (axis.domain && !programDomains.contains(axis.domain) &&
          !facts.orderedStreamDomains.contains(axis.domain) &&
          !facts.contractionDomains.contains(axis.domain))
        facts.vectorDomains.insert(axis.domain);
  return success();
}

} // namespace intent::target

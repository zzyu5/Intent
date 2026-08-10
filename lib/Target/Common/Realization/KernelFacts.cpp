#include "Intent/Target/Common/Realization/KernelFacts.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>

using namespace mlir;

namespace intent::target {
namespace {

LogicalResult addHandler(OperationHandlerRegistry &registry, StringRef name,
                         OperationCallback enter) {
  return registry.add(name, OperationHandler{std::move(enter), {}});
}

Operation *nearestParallelOwner(Operation &operation) {
  for (Operation *parent = operation.getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "intent.parallel")
      return parent;
  return nullptr;
}

LogicalResult analyzeBoundary(Operation &operation, KernelFacts &facts,
                              StringRef fill) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  SmallVector<Operation *> domains;
  bool hasOpaqueScalarIndex = false;
  bool hasInBoundsIndex = false;
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice" || term.kind == "static_index" ||
        term.kind == "slice") {
      ++sourceAxis;
      continue;
    }
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "boundary analysis requires one value per dynamic index term");
    Value indexed = operation.getOperand(*term.operands.front());
    auto indexedAxes = facts.valueAxes.find(indexed);
    if (term.kind == "value_index") {
      if (hasInBoundsPrecondition(indexed, operation.getOperand(0), sourceAxis,
                                  operation)) {
        hasInBoundsIndex = true;
        ++sourceAxis;
        continue;
      }
      if (indexedAxes != facts.valueAxes.end()) {
        for (const LogicalAxis &axis : indexedAxes->second)
          if (axis.domain && !llvm::is_contained(domains, axis.domain))
            domains.push_back(axis.domain);
        ++sourceAxis;
        continue;
      }
      FailureOr<ScalarIndexSource> source =
          traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      if (source->domain) {
        if (!llvm::is_contained(domains, source->domain))
          domains.push_back(source->domain);
      } else if (source->opaque) {
        hasOpaqueScalarIndex = true;
      }
      ++sourceAxis;
      continue;
    }
    FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
    if (failed(domain))
      return failure();
    domains.push_back(*domain);
    ++sourceAxis;
  }
  if (domains.empty() && !hasOpaqueScalarIndex && !hasInBoundsIndex)
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
  auto staticExtent = facts.staticDomainExtents.find(&domain);
  if (staticExtent != facts.staticDomainExtents.end())
    return LogicalAxis{&domain, std::to_string(staticExtent->second)};
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

FailureOr<bool> requiresRuntimeBoundary(Operation &operation,
                                        KernelFacts &facts) {
  FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
  if (failed(relation))
    return failure();
  unsigned sourceAxis = 0;
  for (const IndexTerm &term : *relation) {
    if (term.kind == "new_axis")
      continue;
    if (term.kind == "full_slice" || term.kind == "static_index") {
      ++sourceAxis;
      continue;
    }
    if (term.kind == "slice")
      return true;
    if (term.operands.size() != 1 || !term.operands.front())
      return operation.emitOpError(
          "boundary classification requires one dynamic index operand");
    Value indexed = operation.getOperand(*term.operands.front());
    if (term.kind == "value_index") {
      if (hasInBoundsPrecondition(indexed, operation.getOperand(0), sourceAxis,
                                  operation)) {
        ++sourceAxis;
        continue;
      }
      if (isa<RankedTensorType>(indexed.getType()))
        return true;
      FailureOr<ScalarIndexSource> source =
          traceScalarIndexSource(indexed, operation);
      if (failed(source))
        return failure();
      if (source->opaque && !source->domain) {
        return operation.emitOpError(
            "opaque scalar index requires a preceding I.assume_in_bounds declaration");
      }
      if (source->transformed)
        return true;
      if (!source->domain) {
        ++sourceAxis;
        continue;
      }
      FailureOr<LogicalAxis> logical =
          axisFromDomain(*source->domain, facts, operation);
      FailureOr<LogicalAxis> destination =
          axisFromView(operation.getOperand(0), sourceAxis++, facts, operation);
      if (failed(logical) || failed(destination))
        return failure();
      if (logical->extent != destination->extent)
        return true;
      continue;
    }
    if (term.kind != "region_index")
      return operation.emitOpError("has an unsupported boundary index relation");
    if (!isa<intent::LogicalIndexType, IntegerType, IndexType>(
            indexed.getType()))
      return true;
    FailureOr<Operation *> domain = resolveDomain(indexed, facts, operation);
    FailureOr<LogicalAxis> logical =
        succeeded(domain) ? axisFromDomain(**domain, facts, operation)
                          : FailureOr<LogicalAxis>(failure());
    FailureOr<LogicalAxis> destination =
        axisFromView(operation.getOperand(0), sourceAxis++, facts, operation);
    if (failed(domain) || failed(logical) || failed(destination))
      return failure();
    if (logical->extent != destination->extent)
      return true;
  }
  return false;
}

LogicalResult bindRegionArgumentAxis(Operation &owner, unsigned argumentIndex,
                                     Operation &domain, KernelFacts &facts) {
  auto regions = owner.getAttrOfType<ArrayAttr>("intent.region_argument_nodes");
  auto blocks = regions && !regions.empty() ? dyn_cast<ArrayAttr>(regions[0])
                                            : ArrayAttr();
  auto arguments = blocks && !blocks.empty() ? dyn_cast<ArrayAttr>(blocks[0])
                                             : ArrayAttr();
  auto node = arguments && argumentIndex < arguments.size()
                  ? dyn_cast<IntegerAttr>(arguments[argumentIndex])
                  : IntegerAttr();
  FailureOr<LogicalAxis> axis = axisFromDomain(domain, facts, owner);
  if (!node || failed(axis))
    return owner.emitOpError("has no canonical region-axis identity");
  facts.axisLabels["?region_" + std::to_string(node.getInt()) + "_0"] = *axis;
  return success();
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
      if (!isa<intent::LogicalIndexType, IntegerType, IndexType>(
              indexed.getType())) {
        FailureOr<LogicalAxis> axis = axisFromDomain(**domain, facts, operation);
        if (failed(axis))
          return failure();
        axes.push_back(*axis);
      }
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
      if (!isa<intent::LogicalIndexType, IntegerType, IndexType>(
              indexed.getType())) {
        FailureOr<LogicalAxis> axis =
            axisFromDomain(**domain, facts, operation);
        if (failed(axis))
          return failure();
        resultAxes.push_back(*axis);
      }
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
  if (!result) {
    FailureOr<ScalarIndexSource> source =
        traceScalarIndexSource(operation.getResult(0), operation);
    if (failed(source))
      return failure();
    return success();
  }
  FailureOr<SmallVector<LogicalAxis>> axes =
      axesFromResultShape(operation, 0, facts);
  if (failed(axes))
    return failure();
  return bindResultAxes(operation, 0, std::move(*axes), facts);
}

bool isUnitAxis(const LogicalAxis &axis, const KernelFacts &facts) {
  if (axis.extent == "1")
    return true;
  if (!axis.domain)
    return false;
  return llvm::any_of(facts.stateStreams, [&](const auto &entry) {
    Operation *stream = entry.first;
    if (stream->getNumOperands() == 0 ||
        stream->getOperand(0).getDefiningOp() != axis.domain)
      return false;
    auto extentIndex =
        stream->getAttrOfType<IntegerAttr>("intent.extent_operand_index");
    if (!extentIndex || extentIndex.getInt() < 0 ||
        static_cast<unsigned>(extentIndex.getInt()) >= stream->getNumOperands())
      return false;
    Operation *extent =
        stream->getOperand(extentIndex.getInt()).getDefiningOp();
    auto value = extent
                     ? extent->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
    return extent && extent->getName().getStringRef() == "intent.constant" &&
           value && value.getInt() == 1;
  });
}

std::optional<uint64_t> staticAxisExtent(const LogicalAxis &axis) {
  uint64_t value = 0;
  if (StringRef(axis.extent).getAsInteger(10, value) || value == 0)
    return std::nullopt;
  return value;
}

bool multiplyExtent(uint64_t &product, uint64_t extent) {
  if (product > std::numeric_limits<uint64_t>::max() / extent)
    return false;
  product *= extent;
  return true;
}

LogicalResult propagateReshapeAxes(Operation &operation, KernelFacts &facts) {
  if (operation.getNumOperands() != 1 || operation.getNumResults() != 1)
    return operation.emitOpError("has no canonical reshape schema");
  auto source = facts.valueAxes.find(operation.getOperand(0));
  FailureOr<SmallVector<LogicalAxis>> result =
      axesFromResultShape(operation, 0, facts);
  if (source == facts.valueAxes.end() || failed(result))
    return operation.emitOpError("reshape has no logical-axis provenance");

  size_t sourceIndex = 0, resultIndex = 0;
  while (sourceIndex < source->second.size() && resultIndex < result->size()) {
    if (source->second[sourceIndex] == (*result)[resultIndex]) {
      ++sourceIndex;
      ++resultIndex;
      continue;
    }
    if (isUnitAxis(source->second[sourceIndex], facts)) {
      ++sourceIndex;
      continue;
    }
    if (isUnitAxis((*result)[resultIndex], facts)) {
      ++resultIndex;
      continue;
    }
    std::optional<uint64_t> sourceProduct =
        staticAxisExtent(source->second[sourceIndex++]);
    std::optional<uint64_t> resultProduct =
        staticAxisExtent((*result)[resultIndex++]);
    if (!sourceProduct || !resultProduct)
      return operation.emitOpError(
          "reshape changes a symbolic logical-axis group");
    while (*sourceProduct != *resultProduct) {
      if (*sourceProduct < *resultProduct) {
        if (sourceIndex == source->second.size())
          return operation.emitOpError(
              "reshape changes the logical element count");
        std::optional<uint64_t> extent =
            staticAxisExtent(source->second[sourceIndex++]);
        if (!extent || !multiplyExtent(*sourceProduct, *extent))
          return operation.emitOpError(
              "reshape has an unsupported static source extent group");
      } else {
        if (resultIndex == result->size())
          return operation.emitOpError(
              "reshape changes the logical element count");
        std::optional<uint64_t> extent =
            staticAxisExtent((*result)[resultIndex++]);
        if (!extent || !multiplyExtent(*resultProduct, *extent))
          return operation.emitOpError(
              "reshape has an unsupported static result extent group");
      }
    }
  }
  while (sourceIndex < source->second.size() &&
         isUnitAxis(source->second[sourceIndex], facts))
    ++sourceIndex;
  while (resultIndex < result->size() &&
         isUnitAxis((*result)[resultIndex], facts))
    ++resultIndex;
  if (sourceIndex != source->second.size() || resultIndex != result->size())
    return operation.emitOpError("reshape changes the logical element count");
  return bindResultAxes(operation, 0, std::move(*result), facts);
}

LogicalResult registerFactHandlers(OperationHandlerRegistry &registry,
                                   KernelFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.yield",
                         "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.assume_in_bounds",
          [&](Operation &operation) -> LogicalResult {
            auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
            Type indexType = operation.getNumOperands() > 0
                                 ? operation.getOperand(0).getType()
                                 : Type();
            auto tensorIndex = dyn_cast_or_null<RankedTensorType>(indexType);
            bool integerIndex =
                isa<IntegerType, IndexType, intent::LogicalIndexType>(indexType) ||
                (tensorIndex &&
                 isa<IntegerType, IndexType>(tensorIndex.getElementType()));
            if (operation.getNumOperands() != 2 || operation.getNumResults() != 0 ||
                !axis || axis.getInt() < 0 || !integerIndex ||
                !isa<intent::ViewType>(operation.getOperand(1).getType()))
              return operation.emitOpError(
                  "has no canonical in-bounds precondition schema");
            auto view = cast<intent::ViewType>(operation.getOperand(1).getType());
            auto tensor = dyn_cast<RankedTensorType>(view.getTensor());
            if (!tensor || axis.getInt() >= tensor.getRank())
              return operation.emitOpError(
                  "in-bounds precondition axis exceeds its view rank");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.domain", [&](Operation &operation) -> LogicalResult {
            if ((operation.getNumOperands() != 2 &&
                 operation.getNumOperands() != 3) ||
                operation.getNumResults() != 1)
              return operation.emitOpError("has no canonical domain schema");
            Operation *start = operation.getOperand(0).getDefiningOp();
            Operation *stop = operation.getOperand(1).getDefiningOp();
            auto startValue =
                start ? start->getAttrOfType<IntegerAttr>("intent.value")
                      : IntegerAttr();
            auto axis = stop ? stop->getAttrOfType<IntegerAttr>("intent.axis")
                             : IntegerAttr();
            Operation *step = operation.getNumOperands() == 3
                                  ? operation.getOperand(2).getDefiningOp()
                                  : nullptr;
            auto stepValue =
                step ? step->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
            if (!start || start->getName().getStringRef() != "intent.constant" ||
                !startValue || startValue.getInt() != 0 ||
                (step && (step->getName().getStringRef() != "intent.constant" ||
                          !stepValue || stepValue.getInt() != 1)))
              return operation.emitOpError(
                  "realization requires a zero-based unit-step domain");
            auto stopValue =
                stop ? stop->getAttrOfType<IntegerAttr>("intent.value")
                     : IntegerAttr();
            if (stop && stop->getName().getStringRef() == "intent.constant" &&
                stopValue && stopValue.getInt() > 0) {
              facts.staticDomainExtents[&operation] = stopValue.getInt();
              return success();
            }
            if (!stop || stop->getName().getStringRef() != "intent.dim" || !axis ||
                stop->getNumOperands() != 1)
              return operation.emitOpError(
                  "domain stop must be a positive constant or ABI dimension");
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
            Operation *domain = facts.domainSourceAxes.count(source)
                                    ? source
                                    : facts.partitionDomains.lookup(source);
            if (failed(bindRegionArgumentAxis(operation, 0, *domain, facts)))
              return failure();
            facts.parallels.push_back(&operation);
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.for", [&](Operation &operation) -> LogicalResult {
            Operation *domain = operation.getNumOperands() > 0
                                    ? operation.getOperand(0).getDefiningOp()
                                    : nullptr;
            if (!domain ||
                (!facts.domainSourceAxes.count(domain) &&
                 !facts.staticDomainExtents.count(domain)) ||
                operation.getNumRegions() != 1 ||
                !llvm::hasSingleElement(operation.getRegion(0)) ||
                operation.getNumOperands() != operation.getNumResults() + 1 ||
                operation.getRegion(0).front().getNumArguments() !=
                    operation.getNumResults() + 1 ||
                !isa<intent::LogicalIndexType>(
                    operation.getRegion(0).front().getArgument(0).getType()))
              return operation.emitOpError(
                  "has no canonical sequential-for schema");
            Operation &terminator = operation.getRegion(0).front().back();
            if (terminator.getName().getStringRef() != "intent.yield" ||
                terminator.getNumOperands() != operation.getNumResults())
              return operation.emitOpError(
                  "sequential for must yield every carried value");
            for (unsigned index = 0; index < operation.getNumResults(); ++index) {
              Type initial = operation.getOperand(index + 1).getType();
              Type argument = operation.getRegion(0).front()
                                  .getArgument(index + 1)
                                  .getType();
              Type result = operation.getResult(index).getType();
              if (!initial.isIntOrIndexOrFloat() || initial != argument ||
                  initial != result ||
                  terminator.getOperand(index).getType() != result)
                return operation.emitOpError(
                    "sequential for currently requires scalar type-stable carried values");
            }
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.if", [&](Operation &operation) -> LogicalResult {
            if (operation.getNumOperands() != 1 ||
                !operation.getOperand(0).getType().isInteger(1) ||
                operation.getNumRegions() != 2)
              return operation.emitOpError("has no canonical scalar-if schema");
            for (Region &branch : operation.getRegions()) {
              if (!llvm::hasSingleElement(branch) || branch.front().empty() ||
                  branch.front().getNumArguments() != 0)
                return operation.emitOpError(
                    "scalar if requires two single-block branches");
              Operation &terminator = branch.front().back();
              if (terminator.getName().getStringRef() != "intent.yield" ||
                  terminator.getNumOperands() != operation.getNumResults())
                return operation.emitOpError(
                    "scalar if branches must yield every result");
              for (auto [yielded, result] :
                   llvm::zip(terminator.getOperands(), operation.getResults()))
                if (!result.getType().isIntOrIndexOrFloat() ||
                    yielded.getType() != result.getType())
                  return operation.emitOpError(
                      "scalar if currently requires scalar type-stable results");
            }
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.buffer", [&](Operation &operation) -> LogicalResult {
            FailureOr<LogicalBufferInfo> info = getLogicalBufferInfo(operation);
            Operation *owner = nearestParallelOwner(operation);
            if (failed(info) || !owner || operation.getParentOp() != owner ||
                info->shape.size() != 1)
              return operation.emitOpError(
                  "private logical buffer must be a rank-one direct child of one parallel owner");
            facts.logicalBuffers[&operation] =
                LogicalBufferFact{owner, std::move(*info)};
            return success();
          })))
    return failure();

  auto validateBufferAccess = [&](Operation &operation) -> LogicalResult {
    Operation *buffer = operation.getNumOperands() > 0
                            ? operation.getOperand(0).getDefiningOp()
                            : nullptr;
    auto found = facts.logicalBuffers.find(buffer);
    FailureOr<SmallVector<IndexTerm>> relation = parseIndexRelation(operation);
    if (found == facts.logicalBuffers.end() || failed(relation) ||
        relation->size() != 1 ||
        ((*relation)[0].kind != "value_index" &&
         (*relation)[0].kind != "static_index") ||
        nearestParallelOwner(operation) != found->second.owner)
      return operation.emitOpError(
          "private logical buffer access must use one scalar index under its owner");
    if ((*relation)[0].kind == "value_index" &&
        ((*relation)[0].operands.size() != 1 ||
         !(*relation)[0].operands.front()))
      return operation.emitOpError(
          "private logical buffer dynamic index is not canonical");
    if ((*relation)[0].kind == "value_index") {
      unsigned operandIndex = *(*relation)[0].operands.front();
      auto iterator = operandIndex < operation.getNumOperands()
                          ? dyn_cast<BlockArgument>(
                                operation.getOperand(operandIndex))
                          : BlockArgument();
      Operation *loop = iterator ? iterator.getOwner()->getParentOp() : nullptr;
      Operation *domain = loop && loop->getName().getStringRef() == "intent.for" &&
                                  iterator.getArgNumber() == 0 &&
                                  loop->getNumOperands() > 0
                              ? loop->getOperand(0).getDefiningOp()
                              : nullptr;
      auto extent = facts.staticDomainExtents.find(domain);
      if (!domain || extent == facts.staticDomainExtents.end() ||
          extent->second > found->second.info.shape.front())
        return operation.emitOpError(
            "private logical buffer dynamic index must be a bounded static sequential iterator");
    } else {
      std::optional<int64_t> index = (*relation)[0].staticValues.size() == 1
                                         ? (*relation)[0].staticValues.front()
                                         : std::nullopt;
      if (!index || *index < 0 || *index >= found->second.info.shape.front())
        return operation.emitOpError(
            "private logical buffer static index is outside its extent");
    }
    bool load = operation.getName().getStringRef() == "intent.buffer_load";
    if (load &&
        (operation.getNumResults() != 1 ||
         operation.getResult(0).getType() != found->second.info.elementType))
      return operation.emitOpError(
          "private logical buffer load must produce one matching scalar");
    auto valueIndex =
        operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
    if (!load &&
        (operation.getNumResults() != 0 || !valueIndex ||
         valueIndex.getInt() != 1 || operation.getNumOperands() < 2 ||
         operation.getOperand(1).getType() != found->second.info.elementType))
      return operation.emitOpError(
          "private logical buffer store must consume one matching scalar value");
    return success();
  };
  if (failed(addHandler(registry, "intent.buffer_load", validateBufferAccess)) ||
      failed(addHandler(registry, "intent.buffer_store", validateBufferAccess)))
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
            FailureOr<bool> masked = requiresRuntimeBoundary(operation, facts);
            if (failed(masked))
              return failure();
            std::optional<std::string> fill =
                !*masked ? std::optional<std::string>("none")
                : feedsContract ? std::optional<std::string>("zero")
                                : inferMaskedLaneFill(operation.getResult(0));
            if (!fill) {
              InFlightDiagnostic diagnostic = operation.emitOpError(
                  "cannot prove a semantics-preserving masked-load fill");
              if (auto names = operation.getAttrOfType<ArrayAttr>(
                      "intent.result_names"))
                diagnostic << " for " << names;
              return failure();
            }
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

  auto bindReductionAxes = [&](Operation &operation) -> LogicalResult {
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
    for (unsigned result = 0; result < operation.getNumResults(); ++result)
      if (failed(bindResultAxes(operation, result, resultAxes, facts)))
        return failure();
    return success();
  };
  if (failed(addHandler(registry, "intent.reduce", bindReductionAxes)) ||
      failed(addHandler(registry, "intent.arg_reduce", bindReductionAxes)))
    return failure();

  if (failed(addHandler(
          registry, "intent.scan", [&](Operation &operation) -> LogicalResult {
            auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
            auto input = operation.getNumOperands() > 0
                             ? facts.valueAxes.find(operation.getOperand(0))
                             : facts.valueAxes.end();
            if (!axis || axis.getInt() < 0 || input == facts.valueAxes.end() ||
                static_cast<size_t>(axis.getInt()) >= input->second.size() ||
                operation.getNumResults() != 1)
              return operation.emitOpError(
                  "scan axis has no logical-axis provenance");
            if (Operation *domain = input->second[axis.getInt()].domain)
              facts.vectorDomains.insert(domain);
            return bindResultAxes(operation, 0, input->second, facts);
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast", "intent.compare", "intent.select",
                         "intent.mask", "intent.random"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              return propagatePointwiseAxes(operation, facts);
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.reshape", [&](Operation &operation) -> LogicalResult {
            return propagateReshapeAxes(operation, facts);
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

  auto bindIndexedWrite = [&](Operation &operation,
                              bool recordScatter) -> LogicalResult {
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
          "has no canonical indexed external-write schema");
    FailureOr<SmallVector<LogicalAxis>> indexedAxes =
        inferIndexedAxes(operation, operation.getOperand(0), facts);
    auto valueAxes =
        facts.valueAxes.find(operation.getOperand(valueIndex.getInt()));
    if (failed(indexedAxes) || valueAxes == facts.valueAxes.end() ||
        *indexedAxes != valueAxes->second)
      return operation.emitOpError(
          "indexed write value does not match its destination");
    if (failed(analyzeBoundary(operation, facts, "none")))
      return failure();
    if (recordScatter)
      facts.scatterWrites.insert(&operation);
    return success();
  };
  auto bindScatterWrite = [bindIndexedWrite](Operation &operation) {
    return bindIndexedWrite(operation, true);
  };
  auto bindAtomicWrite = [bindIndexedWrite](Operation &operation) {
    return bindIndexedWrite(operation, false);
  };
  if (failed(addHandler(registry, "intent.scatter_unique", bindScatterWrite)) ||
      failed(addHandler(registry, "intent.scatter_reduce", bindScatterWrite)) ||
      failed(addHandler(registry, "intent.atomic_add", bindAtomicWrite)))
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
    if (failed(bindRegionArgumentAxis(operation, 0, *axisDomain, facts)))
      return failure();
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
  if (owner && owner->getName().getStringRef() == "intent.for" &&
      argument.getArgNumber() == 0 && owner->getNumOperands() > 0) {
    Operation *domain = owner->getOperand(0).getDefiningOp();
    if (domain && (facts.domainSourceAxes.count(domain) ||
                   facts.staticDomainExtents.count(domain)))
      return domain;
    consumer.emitOpError("cannot resolve a sequential for to its source domain");
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

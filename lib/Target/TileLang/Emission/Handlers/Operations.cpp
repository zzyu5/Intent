#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::tilelang::emission {
namespace {

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter,
                         target::OperationCallback leave = {}) {
  return registry.add(name,
                      target::OperationHandler{std::move(enter), std::move(leave)});
}

} // namespace

LogicalResult registerEmissionHandlers(target::OperationHandlerRegistry &registry,
                                       SourceEmitter &emitter) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.domain", "intent.region_end",
                         "intent.partition", "intent.return", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();
  if (failed(addHandler(registry, "intent.dim", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitDimension(op);
      })) ||
      failed(addHandler(registry, "intent.constant", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitConstant(op);
      })) ||
      failed(addHandler(
          registry, "intent.parallel",
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.enterParallel(op)
                                               : success();
          },
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.leaveParallel(op)
                                               : success();
          })) ||
      failed(addHandler(
          registry, "intent.for",
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.enterFor(op) : success();
          },
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.leaveFor(op) : success();
          })) ||
      failed(addHandler(
          registry, "intent.if",
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.enterIf(op) : success();
          },
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.leaveIf(op) : success();
          })) ||
      failed(addHandler(registry, "intent.yield", [&](Operation &op) {
        return emitter.selectOperation(op) ? emitter.emitYield(op) : success();
      })) ||
      failed(addHandler(registry, "intent.buffer", [&](Operation &op) {
        return emitter.selectOperation(op) ? emitter.emitBuffer(op) : success();
      })) ||
      failed(addHandler(registry, "intent.buffer_load", [&](Operation &op) {
        return emitter.selectOperation(op) ? emitter.emitBufferLoad(op)
                                           : success();
      })) ||
      failed(addHandler(registry, "intent.buffer_store", [&](Operation &op) {
        return emitter.selectOperation(op) ? emitter.emitBufferStore(op)
                                           : success();
      })) ||
      failed(addHandler(registry, "intent.assume_in_bounds", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitAssumeInBounds(op);
      })) ||
      failed(addHandler(registry, "intent.view_load", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitLoad(op);
      })) ||
      failed(addHandler(registry, "intent.indices", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitIndices(op);
      })) ||
      failed(addHandler(registry, "intent.random", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitRandom(op);
      })) ||
      failed(addHandler(registry, "intent.reduce", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitReduction(op);
      })) ||
      failed(addHandler(registry, "intent.arg_reduce", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitReduction(op);
      })) ||
      failed(addHandler(registry, "intent.scan", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitScan(op);
      })) ||
      failed(addHandler(registry, "intent.broadcast", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitBroadcast(op);
      })) ||
      failed(addHandler(registry, "intent.unary", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitUnary(op);
      })) ||
      failed(addHandler(registry, "intent.binary", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitBinary(op);
      })) ||
      failed(addHandler(registry, "intent.compare", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitBinary(op);
      })) ||
      failed(addHandler(registry, "intent.mask", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitMask(op);
      })) ||
      failed(addHandler(registry, "intent.select", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitSelect(op);
      })) ||
      failed(addHandler(registry, "intent.cast", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitCast(op);
      })) ||
      failed(addHandler(registry, "intent.reshape", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitReshape(op);
      })) ||
      failed(addHandler(registry, "intent.transpose", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitTranspose(op);
      })) ||
      failed(addHandler(registry, "intent.full", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitFull(op);
      })) ||
      failed(addHandler(registry, "intent.zeros", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitZeros(op);
      })) ||
      failed(addHandler(registry, "intent.members", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitMembers(op);
      })) ||
      failed(addHandler(registry, "intent.gather", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitGather(op);
      })) ||
      failed(addHandler(
          registry, "intent.state_stream",
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.enterStateStream(op)
                                               : success();
          },
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.leaveStateStream(op)
                                               : success();
          })) ||
      failed(addHandler(registry, "intent.contract", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitContract(op);
      })) ||
      failed(addHandler(registry, "intent.view_store", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitStore(op);
      })) ||
      failed(addHandler(registry, "intent.scatter_unique", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitUniqueStore(op);
      })) ||
      failed(addHandler(registry, "intent.scatter_reduce", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitAtomic(op);
      })) ||
      failed(addHandler(registry, "intent.atomic_add", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitAtomic(op);
      })))
    return failure();
  return success();
}

LogicalResult SourceEmitter::emitConstant(Operation &operation) {
  if (operation.getBlock() == &kernel.entry.getBody().front())
    return success();
  if (operation.getNumResults() != 1)
    return operation.emitOpError("constant emission requires one result");
  std::string result = makeResultName(operation, 0);
  Attribute value = operation.getAttr("intent.value");
  std::string expression;
  if (auto floating = dyn_cast<FloatAttr>(value)) {
    double number = floating.getValueAsDouble();
    if (std::isinf(number))
      expression = number < 0 ? "-T.infinity(T.float32)"
                              : "T.infinity(T.float32)";
    else if (std::isnan(number))
      return operation.emitOpError("TileLang constants cannot materialize NaN");
    else
      expression = std::to_string(number);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported TileLang constant value");
  }
  line(result + " = " + expression);
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::emitDimension(Operation &operation) {
  auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  FailureOr<ABIView *> view =
      operation.getNumOperands() == 1
          ? lookupView(operation.getOperand(0), operation)
          : FailureOr<ABIView *>(failure());
  if (operation.getNumResults() != 1 || !axis || axis.getInt() < 0 ||
      failed(view) ||
      static_cast<size_t>(axis.getInt()) >= (*view)->shape.size())
    return operation.emitOpError(
        "lacks a mechanical TileLang dimension binding");
  bindResult(operation, 0, (*view)->shape[axis.getInt()]);
  return success();
}

LogicalResult SourceEmitter::emitAssumeInBounds(Operation &operation) {
  auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  FailureOr<StringRef> index = lookupValue(operation, 0);
  FailureOr<ABIView *> view =
      operation.getNumOperands() == 2
          ? lookupView(operation.getOperand(1), operation)
          : FailureOr<ABIView *>(failure());
  if (operation.getNumOperands() != 2 || operation.getNumResults() != 0 ||
      !axis || axis.getInt() < 0 || failed(index) || failed(view) ||
      axis.getInt() >= (*view)->tensor.getRank())
    return operation.emitOpError(
        "lacks a mechanical TileLang in-bounds assumption binding");

  Type indexType = operation.getOperand(0).getType();
  std::string expression = index->str();
  if (auto tensor = dyn_cast<RankedTensorType>(indexType)) {
    auto result = dyn_cast<OpResult>(operation.getOperand(0));
    FailureOr<SmallVector<std::string>> extents =
        result ? tensorExtents(*result.getOwner(), result.getResultNumber())
               : FailureOr<SmallVector<std::string>>(failure());
    if (tensor.getRank() != 1 || failed(extents) || extents->size() != 1 ||
        extents->front() != "1")
      return operation.emitOpError(
          "TileLang can project only singleton tensor index assumptions");
    auto assumed = assumedIndexNames.find(operation.getOperand(0));
    if (assumed == assumedIndexNames.end()) {
      auto node = operation.getAttrOfType<IntegerAttr>("intent.node");
      if (!node)
        return operation.emitOpError(
            "TileLang in-bounds assumption has no canonical node identity");
      expression = uniqueName("assumed_index", node.getInt());
      line(expression + " = " + index->str() + "[0]");
      assumedIndexNames[operation.getOperand(0)] = expression;
    } else {
      expression = assumed->second;
    }
  } else if (!isa<IntegerType, IndexType, intent::LogicalIndexType>(indexType)) {
    return operation.emitOpError(
        "TileLang in-bounds assumptions require an integer index");
  }

  line("T.assume(0 <= " + expression + ")");
  line("T.assume(" + expression + " < " +
       (*view)->shape[axis.getInt()] + ")");
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() != 1)
    return operation.emitOpError(
        "TileLang parallel ownership requires one region argument");
  BlockArgument argument = operation.getRegion(0).front().getArgument(0);
  FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
  if (failed(axis))
    return failure();

  if (!planIndex.stages.empty()) {
    bool outer = false;
    bool member = false;
    for (const auto &entry : stageRaggedRuntime) {
      const RaggedRuntime &runtime = raggedRuntimes[entry.second];
      outer |= runtime.binding.getOuterNode() == axis->getNode();
      member |= llvm::any_of(runtime.ownedMembers, [&](Operation *candidate) {
        auto node = candidate->getAttrOfType<IntegerAttr>("intent.node");
        return node && node.getInt() == axis->getNode();
      });
    }
    if (outer == member)
      return operation.emitOpError("has no staged program-axis binding");
    valueNames[argument] = outer ? "expert" : "member_start";
    return success();
  }

  if (&operation == programRoot &&
      !planIndex.components.orderedRaggedProgramAxes.empty()) {
    for (int64_t memberNode :
         planIndex.components.orderedRaggedProgramAxes) {
      FailureOr<plan::RaggedOp> relation =
          target::emission::uniqueRaggedRelation(planIndex, memberNode, operation);
      std::string outer = succeeded(relation)
                              ? axisIndices.lookup(relation->getOuterNode())
                              : std::string();
      std::string block = programBlocks.lookup(memberNode);
      auto runtime = succeeded(relation)
                         ? raggedRuntimeByRelation.find(relation->getNode())
                         : raggedRuntimeByRelation.end();
      if (failed(relation) || outer.empty() || block.empty() ||
          runtime == raggedRuntimeByRelation.end())
        return operation.emitOpError(
            "ordered ragged axis has no physical metadata binding");
      RaggedRuntime &ragged = raggedRuntimes[runtime->second];
      auto ordered =
          planIndex.components.orderedAxesByRelation.find(relation->getNode());
      if (ordered == planIndex.components.orderedAxesByRelation.end() ||
          ordered->second.empty())
        return operation.emitOpError(
            "ordered ragged relation has no stream axis");
      for (int64_t orderedAxis : ordered->second) {
        std::string suffix = std::to_string(orderedAxis);
        line("sequence_begin_" + suffix + " = " +
             ragged.offsets->argument->name + "[" + addressIndex(outer) + "]");
        line("sequence_end_" + suffix + " = " +
             ragged.offsets->argument->name + "[" +
             addressIndex(outer + " + 1") + "]");
        line("sequence_length_" + suffix + " = sequence_end_" + suffix +
             " - sequence_begin_" + suffix);
      }
      std::string suffix = std::to_string(ordered->second.front());
      std::string query = std::to_string(memberNode);
      line("query_start_" + query + " = sequence_begin_" + suffix + " + " +
           addressIndex(block) + " * " +
           planIndex.axes.lookup(memberNode).getTile().str());
      axisIndices[memberNode] = "query_start_" + query;
    }
  }
  std::string value =
      planIndex.components.orderedRaggedProgramAxes.contains(axis->getNode())
          ? axisIndices.lookup(axis->getNode())
          : programBlocks.lookup(axis->getNode());
  if (value.empty())
    return axis->emitOpError("has no emitted per-axis program index");
  valueNames[argument] = value;
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &operation) {
  if (&operation == programRoot && planIndex.program.getPersistent())
    indentation -= 2;
  return success();
}

LogicalResult SourceEmitter::enterFor(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "for emission");
  Operation *domain = operation.getNumOperands() > 0
                          ? operation.getOperand(0).getDefiningOp()
                          : nullptr;
  if (failed(node) || !domain ||
      domain->getName().getStringRef() != "intent.domain" ||
      domain->getNumOperands() < 2 || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical TileLang sequential loop");
  Block &body = operation.getRegion(0).front();
  std::string stop;
  Operation *stopDefinition = domain->getOperand(1).getDefiningOp();
  auto stopLiteral = stopDefinition
                         ? stopDefinition->getAttrOfType<IntegerAttr>("intent.value")
                         : IntegerAttr();
  if (stopDefinition &&
      stopDefinition->getName().getStringRef() == "intent.constant" &&
      stopLiteral)
    stop = std::to_string(stopLiteral.getInt());
  else {
    auto found = valueNames.find(domain->getOperand(1));
    if (found == valueNames.end())
      return operation.emitOpError("has no emitted TileLang loop bound");
    stop = found->second;
  }
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    std::string dtype = dtypeName(operation.getResult(index).getType(), operation);
    if (failed(initial) || dtype.empty())
      return failure();
    std::string carrier =
        uniqueName("loop_state_" + std::to_string(index), *node);
    line(carrier + " = T.alloc_local((1,), " + dtype + ")");
    line(carrier + "[0] = " + initial->str());
    carriers.push_back(carrier);
    valueNames[body.getArgument(index + 1)] = carrier + "[0]";
  }
  loopCarriers[&operation] = carriers;
  std::string iterator = makeRegionArgumentName(operation, 0);
  valueNames[body.getArgument(0)] = iterator;
  line("for " + iterator + " in T.serial(" + stop + "):");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active TileLang sequential loop");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index] + "[0]");
  return success();
}

LogicalResult SourceEmitter::enterIf(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "if emission");
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (failed(node) || failed(condition) || operation.getNumRegions() != 2)
    return operation.emitOpError("lacks a mechanical TileLang scalar branch");
  SmallVector<std::string> results;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    std::string dtype = dtypeName(operation.getResult(index).getType(), operation);
    if (dtype.empty())
      return failure();
    std::string result = makeResultName(operation, index) + "_branch";
    line(result + " = T.alloc_local((1,), " + dtype + ")");
    results.push_back(std::move(result));
  }
  ifResults[&operation] = std::move(results);
  line("if " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveIf(Operation &operation) {
  auto results = ifResults.find(&operation);
  if (results == ifResults.end())
    return operation.emitOpError("has no active TileLang scalar branch");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, results->second[index] + "[0]");
  return success();
}

LogicalResult SourceEmitter::emitYield(Operation &operation) {
  Operation *owner = operation.getParentOp();
  if (!owner)
    return operation.emitOpError("has no structured-control owner");
  StringRef name = owner->getName().getStringRef();
  if (name == "intent.for") {
    auto carriers = loopCarriers.find(owner);
    if (carriers == loopCarriers.end() ||
        carriers->second.size() != operation.getNumOperands())
      return operation.emitOpError("does not match its TileLang loop state");
    for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
      FailureOr<StringRef> yielded = lookupValue(operation, index);
      if (failed(yielded))
        return failure();
      line(carriers->second[index] + "[0] = " + yielded->str());
    }
    return success();
  }
  if (name != "intent.if")
    return success();
  auto results = ifResults.find(owner);
  if (results == ifResults.end() ||
      results->second.size() != operation.getNumOperands())
    return operation.emitOpError("does not match its TileLang branch results");
  if (operation.getNumOperands() == 0)
    line("pass");
  for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(operation, index);
    if (failed(yielded))
      return failure();
    line(results->second[index] + "[0] = " + yielded->str());
  }
  if (operation.getParentRegion() == &owner->getRegion(0)) {
    --indentation;
    line("else:");
    ++indentation;
  }
  return success();
}

LogicalResult SourceEmitter::emitBuffer(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "buffer emission");
  plan::BufferOp binding =
      succeeded(node) ? planIndex.buffers.lookup(*node) : plan::BufferOp();
  FailureOr<target::LogicalBufferInfo> info =
      target::getLogicalBufferInfo(operation);
  FailureOr<StringRef> initializer = lookupValue(operation, 0);
  if (failed(node) || !binding ||
      binding.getSpace() != "private_scalar_array" || failed(info) ||
      info->shape.size() != 1 || failed(initializer))
    return operation.emitOpError(
        "lacks a private scalar-array TileLang buffer binding");
  std::string dtype = dtypeName(info->elementType, operation);
  if (dtype.empty())
    return failure();
  std::string base = makeResultName(operation, 0);
  SmallVector<std::string> elements;
  for (int64_t index = 0; index < info->shape.front(); ++index) {
    std::string storage = base + "_" + std::to_string(index);
    line(storage + " = T.alloc_local((1,), " + dtype + ")");
    line(storage + "[0] = " + initializer->str());
    elements.push_back(storage + "[0]");
  }
  scalarBuffers[operation.getResult(0)] = std::move(elements);
  return success();
}

LogicalResult SourceEmitter::emitBufferLoad(Operation &operation) {
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
  FailureOr<target::LogicalBufferIndex> index =
      target::getLogicalBufferIndex(operation);
  if (buffer == scalarBuffers.end() || failed(index) ||
      buffer->second.empty() || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a scalarized TileLang buffer load");
  std::string expression;
  if (index->constant) {
    if (*index->constant < 0 ||
        static_cast<size_t>(*index->constant) >= buffer->second.size())
      return operation.emitOpError("indexes outside its private buffer");
    expression = buffer->second[*index->constant];
  } else {
    FailureOr<StringRef> dynamic = lookupValue(operation, *index->operand);
    if (failed(dynamic))
      return failure();
    expression = buffer->second.back();
    for (int64_t position = static_cast<int64_t>(buffer->second.size()) - 2;
         position >= 0; --position)
      expression = "T.if_then_else(" + dynamic->str() + " == " +
                   std::to_string(position) + ", " + buffer->second[position] +
                   ", " + expression + ")";
  }
  std::string result = makeResultName(operation, 0);
  line(result + " = " + expression);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitBufferStore(Operation &operation) {
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
  FailureOr<target::LogicalBufferIndex> index =
      target::getLogicalBufferIndex(operation);
  FailureOr<StringRef> stored = lookupValue(operation, 1);
  if (buffer == scalarBuffers.end() || failed(index) || failed(stored) ||
      buffer->second.empty())
    return operation.emitOpError("lacks a scalarized TileLang buffer store");
  if (index->constant) {
    if (*index->constant < 0 ||
        static_cast<size_t>(*index->constant) >= buffer->second.size())
      return operation.emitOpError("indexes outside its private buffer");
    line(buffer->second[*index->constant] + " = " + stored->str());
    return success();
  }
  FailureOr<StringRef> dynamic = lookupValue(operation, *index->operand);
  if (failed(dynamic))
    return failure();
  for (auto [position, element] : llvm::enumerate(buffer->second))
    line(element + " = T.if_then_else(" + dynamic->str() + " == " +
         std::to_string(position) + ", " + stored->str() + ", " + element +
         ")");
  return success();
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a TileLang load binding");
  FailureOr<bool> wholeView = target::isWholeViewAccess(operation);
  if (failed(wholeView))
    return failure();
  if (*wholeView && boundary.getDomainNodes().empty() &&
      boundary.getPadding() == "none") {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view))
      return failure();
    bindResult(operation, 0, (*view)->argument->name);
    return success();
  }
  if (boundary.getDefer()) {
    deferredLoads[operation.getResult(0)] = &operation;
    return success();
  }
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<std::string> physicalFill = transferPhysicalExtentFill(operation);
  bool scalarResult = operation.getNumResults() == 1 &&
                      !isa<RankedTensorType>(operation.getResult(0).getType());
  Type resultElementType = scalarResult
                               ? operation.getResult(0).getType()
                               : cast<RankedTensorType>(
                                     operation.getResult(0).getType())
                                     .getElementType();
  StringRef zeroFill = isa<IntegerType, IndexType>(resultElementType) ? "0"
                                                                      : "0.0";
  if (failed(physicalFill))
    return failure();
  if (!scalarResult) {
    auto resultType =
        dyn_cast<RankedTensorType>(operation.getResult(0).getType());
    FailureOr<SmallVector<target::IndexTerm>> relation =
        target::parseIndexRelation(operation);
    if (!resultType || failed(relation))
      return failure();
    unsigned tensorIndices = llvm::count_if(
        *relation, [&](const target::IndexTerm &term) {
          return term.kind == "value_index" && term.operands.size() == 1 &&
                 term.operands.front() &&
                 isa<RankedTensorType>(
                     operation.getOperand(*term.operands.front()).getType());
        });
    if (resultType.getRank() > 2 && tensorIndices > 1)
      return operation.emitOpError(
          "requires a multi-axis broadcasted indirect read footprint that "
          "TileLang cannot project as one parallel fragment");
  }
  bool expanded = !physicalFill->empty();
  if (scalarResult) {
    FailureOr<std::string> indices = accessIndices(operation);
    if (failed(view) || failed(indices))
      return operation.emitOpError(
          "lacks a mechanical TileLang scalar load binding");
    std::string expression =
        (*view)->argument->name + "[" + *indices + "]";
    if (boundary.getCheckBounds() && boundary.getPadding() != "none") {
      FailureOr<std::string> predicate =
          elementBoundsPredicate(operation, {});
      if (failed(predicate))
        return failure();
      StringRef fill = boundary.getPadding() == "negative_infinity"
                           ? "-T.infinity(T.float32)"
                           : zeroFill;
      expression = "T.if_then_else(" + *predicate + ", " + expression +
                   ", " + fill.str() + ")";
    }
    std::string result = makeResultName(operation, 0);
    line(result + " = " + expression);
    bindResult(operation, 0, result);
    return success();
  }
  FailureOr<std::string> result =
      allocateResult(operation, 0, boundary.getResultSpace());
  if (failed(view) || failed(result))
    return operation.emitOpError("lacks a mechanical TileLang load binding");
  if (boundary.getTransfer() == "parallel_elements" || expanded) {
    StringRef padding = boundary.getPadding();
    if (padding == "none" && expanded)
      padding = *physicalFill;
    FailureOr<bool> tensorIndirect = target::hasTensorIndirectIndex(operation);
    bool stagePhysicalPadding =
        expanded && succeeded(tensorIndirect) && *tensorIndirect;
    FailureOr<SmallVector<std::string>> extents =
        tensorExtents(operation, 0, !stagePhysicalPadding);
    if (failed(extents) || extents->empty() ||
        failed(tensorIndirect) ||
        (padding != "zero" && padding != "negative_infinity"))
      return operation.emitOpError(
          "has an invalid parallel TileLang load transfer");
    std::string elementResult = *result;
    if (stagePhysicalPadding) {
      auto resultType = cast<RankedTensorType>(operation.getResult(0).getType());
      std::string dtype = dtypeName(resultType.getElementType(), operation);
      StringRef allocator = boundary.getResultSpace() == "shared"
                                ? "T.alloc_shared"
                            : boundary.getResultSpace() == "fragment"
                                ? "T.alloc_fragment"
                                : StringRef();
      if (dtype.empty() || allocator.empty())
        return operation.emitOpError(
            "cannot stage indirect TileLang physical padding");
      std::string shape = "(";
      for (auto [axis, extent] : llvm::enumerate(*extents)) {
        if (axis)
          shape += ", ";
        shape += extent;
      }
      if (extents->size() == 1)
        shape += ",";
      shape += ")";
      elementResult += "_logical";
      line(elementResult + " = " + allocator.str() + "(" + shape + ", " +
           dtype + ")");
    }
    SmallVector<std::string> tileIndices;
    std::string loop = "for ";
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis)
        loop += ", ";
      tileIndices.push_back("transfer_i" + std::to_string(axis));
      loop += tileIndices.back();
    }
    loop += " in T.Parallel(";
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis)
        loop += ", ";
      loop += (*extents)[axis];
    }
    FailureOr<std::string> indices =
        elementAccessIndices(operation, tileIndices);
    if (failed(indices))
      return failure();
    FailureOr<std::string> logicalPredicate =
        boundary.getCheckBounds()
            ? elementBoundsPredicate(operation, tileIndices, false, false)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> physicalPredicate =
        expanded && !stagePhysicalPadding
            ? elementBoundsPredicate(operation, tileIndices, true)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> wholeTile =
        boundary.getCheckBounds() && succeeded(tensorIndirect) &&
                !*tensorIndirect
            ? wholeTileBoundsPredicate(operation, *extents)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> bulkIndices =
        boundary.getCheckBounds() && succeeded(tensorIndirect) &&
                !*tensorIndirect
            ? accessIndices(operation)
            : FailureOr<std::string>(std::string());
    if (failed(logicalPredicate) || failed(physicalPredicate) ||
        failed(tensorIndirect) || failed(wholeTile) || failed(bulkIndices))
      return failure();
    bool hasBulkFastPath = !expanded && !wholeTile->empty();
    if (hasBulkFastPath) {
      line("if " + *wholeTile + ":");
      ++indentation;
      line("T.copy(" + (*view)->argument->name + "[" + *bulkIndices + "], " +
           *result + ")");
      --indentation;
      line("else:");
      ++indentation;
    }
    std::string target = elementResult + "[";
    for (auto [axis, index] : llvm::enumerate(tileIndices)) {
      if (axis)
        target += ", ";
      target += index;
    }
    StringRef fill = padding == "negative_infinity"
                         ? "-T.infinity(T.float32)"
                         : zeroFill;
    auto emitElementwise = [&](bool includePhysicalBounds) {
      line(loop + "):");
      ++indentation;
      if (boundary.getCheckBounds()) {
        line("if " + *logicalPredicate + ":");
        ++indentation;
      }
      if (includePhysicalBounds) {
        line("if " + *physicalPredicate + ":");
        ++indentation;
      }
      line(target + "] = " + (*view)->argument->name + "[" + *indices + "]");
      if (includePhysicalBounds) {
        --indentation;
        line("else:");
        ++indentation;
        line(target + "] = " + fill.str());
        --indentation;
      }
      if (boundary.getCheckBounds()) {
        --indentation;
        line("else:");
        ++indentation;
        line(target + "] = " + fill.str());
        --indentation;
      }
      --indentation;
      if (boundary.getResultSpace() == "shared")
        line("T.sync_threads()");
    };
    emitElementwise(expanded && !stagePhysicalPadding);
    if (stagePhysicalPadding) {
      line("T.clear(" + *result + ")");
      if (boundary.getResultSpace() == "shared")
        line("T.sync_threads()");
      std::string copyLoop = "for ";
      std::string resultIndices;
      for (unsigned axis = 0; axis < extents->size(); ++axis) {
        if (axis) {
          copyLoop += ", ";
          resultIndices += ", ";
        }
        std::string index = "physical_copy_i" + std::to_string(axis);
        copyLoop += index;
        resultIndices += index;
      }
      copyLoop += " in T.Parallel(";
      for (auto [axis, extent] : llvm::enumerate(*extents)) {
        if (axis)
          copyLoop += ", ";
        copyLoop += extent;
      }
      line(copyLoop + "):");
      ++indentation;
      line(*result + "[" + resultIndices + "] = " + elementResult + "[" +
           resultIndices + "]");
      --indentation;
      if (boundary.getResultSpace() == "shared")
        line("T.sync_threads()");
    }
    if (hasBulkFastPath)
      --indentation;
    bindResult(operation, 0, *result);
    return success();
  }
  if (boundary.getTransfer() != "bulk_copy")
    return operation.emitOpError("has no TileLang load transfer emitter");
  FailureOr<std::string> indices = accessIndices(operation);
  if (failed(indices))
    return failure();
  line("T.copy(" + (*view)->argument->name + "[" + *indices + "], " +
       *result + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitIndices(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "indices emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<plan::AxisOp> axis =
      operation.getNumOperands() == 1
          ? resolveAxis(operation.getOperand(0), operation)
          : FailureOr<plan::AxisOp>(failure());
  auto resultType = operation.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                        : RankedTensorType();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "logical_indices" ||
      failed(axis) || !resultType || resultType.getRank() != 1 ||
      !isa<IntegerType, IndexType>(resultType.getElementType()) ||
      failed(result) || failed(extents) || extents->size() != 1)
    return operation.emitOpError("lacks a mechanical TileLang indices binding");
  std::string base = axisIndices.lookup(axis->getNode());
  if (base.empty() && axis->hasRole("lane")) {
    base = "0";
    axisIndices[axis->getNode()] = base;
  }
  if (base.empty())
    return operation.emitOpError("has no TileLang vector index realization");
  line("for indices_i in T.Parallel(" + extents->front() + "):");
  ++indentation;
  line(*result + "[indices_i] = " + base + " + indices_i");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitRandom(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "random emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding ||
      binding.getLowering() != "counter_xorshift32" ||
      binding.getSpace() != (tensorResult ? "fragment" : "local") ||
      operation.getNumOperands() != 2 || operation.getNumResults() != 1)
    return operation.emitOpError(
        "lacks a mechanical TileLang counter RNG binding");

  std::string result = makeResultName(operation, 0);
  SmallVector<std::string> indices;
  if (tensorResult) {
    FailureOr<std::string> storage = allocateResult(operation, 0, "fragment");
    FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
    if (failed(storage) || failed(extents))
      return failure();
    result = *storage;
    std::string loop = "for ";
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis)
        loop += ", ";
      indices.push_back("random_i" + std::to_string(axis));
      loop += indices.back();
    }
    loop += " in T.Parallel(";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        loop += ", ";
      loop += extent;
    }
    line(loop + "):");
    ++indentation;
  }

  FailureOr<std::string> seed =
      tensorElement(operation.getOperand(0), indices, operation);
  FailureOr<std::string> counter =
      tensorElement(operation.getOperand(1), indices, operation);
  if (failed(seed) || failed(counter))
    return failure();
  std::string bits0 = makeResultName(operation, 0) + "_bits0";
  std::string bits1 = makeResultName(operation, 0) + "_bits1";
  std::string bits2 = makeResultName(operation, 0) + "_bits2";
  std::string bits3 = makeResultName(operation, 0) + "_bits3";
  line(bits0 + " = T.bitwise_xor(T.bitwise_xor(T.cast(" + *counter +
       ", T.uint32), T.cast(" + *seed +
       ", T.uint32)), T.cast(1831565813, T.uint32))");
  line(bits1 + " = T.bitwise_xor(" + bits0 + ", T.shift_left(" + bits0 +
       ", 13))");
  line(bits2 + " = T.bitwise_xor(" + bits1 + ", T.shift_right(" + bits1 +
       ", 17))");
  line(bits3 + " = T.bitwise_xor(" + bits2 + ", T.shift_left(" + bits2 +
       ", 5))");
  std::string uniform = "T.cast(T.shift_right(" + bits3 +
                        ", 8), T.float32) * 5.960464477539063e-08";
  if (tensorResult) {
    std::string target = result + "[";
    for (auto [axis, index] : llvm::enumerate(indices)) {
      if (axis)
        target += ", ";
      target += index;
    }
    line(target + "] = " + uniform);
    --indentation;
    bindResult(operation, 0, result);
  } else {
    line(result + " = " + uniform);
    valueNames[operation.getResult(0)] = result;
  }
  return success();
}

LogicalResult SourceEmitter::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  bool argReduction =
      binding && binding.getLowering() == "T.reduce_max_with_index";
  unsigned expectedResults = argReduction ? 2 : 1;
  if (failed(node) || !binding || failed(operand) ||
      operation.getNumResults() != expectedResults)
    return operation.emitOpError("lacks a TileLang reduction binding");
  bool logicalReduction = binding.getLowering() == "T.any_of" ||
                          binding.getLowering() == "T.all_of";
  if (logicalReduction) {
    auto input = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
    if (!input || input.getRank() != 1 || binding.getAxis() != 0 ||
        isa<RankedTensorType>(operation.getResult(0).getType()))
      return operation.emitOpError(
          "TileLang any/all projection requires a one-dimensional input "
          "reduced to a scalar");
    std::string result = makeResultName(operation, 0);
    line(result + " = " + binding.getLowering().str() + "(" + operand->str() +
         ")");
    valueNames[operation.getResult(0)] = result;
    return success();
  }
  if (argReduction) {
    auto input = dyn_cast<OpResult>(operation.getOperand(0));
    FailureOr<SmallVector<std::string>> inputExtents =
        input ? tensorExtents(*input.getOwner(), input.getResultNumber())
              : FailureOr<SmallVector<std::string>>(failure());
    int64_t axis = binding.getAxis();
    bool tensorResults = isa<RankedTensorType>(operation.getResult(0).getType());
    if (failed(inputExtents) || axis < 0 ||
        static_cast<size_t>(axis) >= inputExtents->size() ||
        tensorResults !=
            isa<RankedTensorType>(operation.getResult(1).getType()))
      return operation.emitOpError(
          "has no mechanical TileLang arg-reduction shape");

    std::string valueStorage;
    std::string indexStorage;
    if (tensorResults) {
      FailureOr<std::string> value = allocateResult(operation, 0, "fragment");
      FailureOr<std::string> index = allocateResult(operation, 1, "fragment");
      if (failed(value) || failed(index))
        return failure();
      valueStorage = *value;
      indexStorage = *index;
    } else {
      std::string valueDtype =
          dtypeName(operation.getResult(0).getType(), operation);
      std::string indexDtype =
          dtypeName(operation.getResult(1).getType(), operation);
      if (valueDtype.empty() || indexDtype.empty())
        return failure();
      valueStorage = makeResultName(operation, 0) + "_fragment";
      indexStorage = makeResultName(operation, 1) + "_fragment";
      line(valueStorage + " = T.alloc_fragment((1,), " + valueDtype + ")");
      line(indexStorage + " = T.alloc_fragment((1,), " + indexDtype + ")");
    }
    line("T.reduce_max(" + operand->str() + ", " + valueStorage + ", dim=" +
         std::to_string(axis) + ", clear=True)");

    std::string candidateShape = "(";
    for (auto [position, extent] : llvm::enumerate(*inputExtents)) {
      if (position)
        candidateShape += ", ";
      candidateShape += extent;
    }
    if (inputExtents->size() == 1)
      candidateShape += ",";
    candidateShape += ")";
    std::string candidates = makeResultName(operation, 1) + "_candidates";
    line(candidates + " = T.alloc_fragment(" + candidateShape + ", T.int32)");

    SmallVector<std::string> inputIndices;
    std::string loop = "for ";
    for (unsigned dimension = 0; dimension < inputExtents->size(); ++dimension) {
      if (dimension)
        loop += ", ";
      inputIndices.push_back("arg_reduce_i" + std::to_string(dimension));
      loop += inputIndices.back();
    }
    loop += " in T.Parallel(";
    for (auto [position, extent] : llvm::enumerate(*inputExtents)) {
      if (position)
        loop += ", ";
      loop += extent;
    }
    line(loop + "):");
    ++indentation;
    auto access = [](StringRef value, ArrayRef<std::string> indices) {
      std::string expression = value.str() + "[";
      for (auto [position, index] : llvm::enumerate(indices)) {
        if (position)
          expression += ", ";
        expression += index;
      }
      return expression + "]";
    };
    SmallVector<std::string> resultIndices;
    for (auto [dimension, index] : llvm::enumerate(inputIndices))
      if (static_cast<int64_t>(dimension) != axis)
        resultIndices.push_back(index);
    std::string maximum = tensorResults
                              ? access(valueStorage, resultIndices)
                              : valueStorage + "[0]";
    line(access(candidates, inputIndices) + " = T.if_then_else(" +
         access(*operand, inputIndices) + " == " + maximum + ", " +
         inputIndices[axis] + ", " + (*inputExtents)[axis] + ")");
    --indentation;
    line("T.reduce_min(" + candidates + ", " + indexStorage + ", dim=" +
         std::to_string(axis) + ", clear=True)");
    if (tensorResults) {
      bindResult(operation, 0, valueStorage);
      bindResult(operation, 1, indexStorage);
    } else {
      valueNames[operation.getResult(0)] = valueStorage + "[0]";
      valueNames[operation.getResult(1)] = indexStorage + "[0]";
    }
    return success();
  }
  if (isa<RankedTensorType>(operation.getResult(0).getType())) {
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
    if (failed(result))
      return failure();
    line(binding.getLowering().str() + "(" + operand->str() + ", " + *result +
         ", dim=" + std::to_string(binding.getAxis()) + ", clear=True)");
    bindResult(operation, 0, *result);
    return success();
  }
  std::string dtype = dtypeName(operation.getResult(0).getType(), operation);
  if (dtype.empty())
    return failure();
  std::string result = makeResultName(operation, 0) + "_fragment";
  line(result + " = T.alloc_fragment((1,), " + dtype + ")");
  line(binding.getLowering().str() + "(" + operand->str() + ", " + result +
       ", dim=" + std::to_string(binding.getAxis()) + ", clear=True)");
  valueNames[operation.getResult(0)] = result + "[0]";
  return success();
}

LogicalResult SourceEmitter::emitScan(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "scan emission");
  plan::ScanOp binding =
      succeeded(node) ? planIndex.scans.lookup(*node) : plan::ScanOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> result =
      operation.getNumResults() == 1
          ? allocateResult(operation, 0, binding.getResultSpace())
          : FailureOr<std::string>(failure());
  if (failed(node) || !binding || binding.getLowering() != "T.cumsum" ||
      failed(operand) || failed(result))
    return operation.emitOpError("lacks a mechanical TileLang scan binding");
  line("T.copy(" + operand->str() + ", " + *result + ")");
  line("T.cumsum(" + *result + ", dim=" +
       std::to_string(binding.getAxis()) + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitBroadcast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "broadcast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "alias" ||
      binding.getReuseOperandAttr().getInt() != -1 ||
      failed(result) || failed(extents))
    return operation.emitOpError("lacks a TileLang broadcast binding");
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back("broadcast_i" + std::to_string(axis));
    loop += indices.back();
  }
  loop += " in T.Parallel(";
  for (auto [axis, extent] : llvm::enumerate(*extents)) {
    if (axis)
      loop += ", ";
    loop += extent;
  }
  loop += "):";
  line(loop);
  ++indentation;
  FailureOr<std::string> operand =
      tensorElement(operation.getOperand(0), indices, operation);
  if (failed(operand))
    return failure();
  std::string target = *result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + *operand);
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitUnary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "unary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding || operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local") ||
      (!tensorResult && binding.getReuseOperandAttr().getInt() != -1) ||
      (tensorResult && binding.getReuseOperandAttr().getInt() != -1 &&
       binding.getReuseOperandAttr().getInt() != 0))
    return operation.emitOpError("lacks a TileLang unary binding");
  if (!tensorResult) {
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    if (failed(operand))
      return failure();
    std::string expression =
        binding.getLowering() == "python_negate"
            ? "-(" + operand->str() + ")"
            : binding.getLowering().str() + "(" + operand->str() + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = " + expression);
    valueNames[operation.getResult(0)] = result;
    return success();
  }
  std::string result;
  if (binding.getReuseOperandAttr().getInt() == 0) {
    FailureOr<StringRef> reused = lookupValue(operation, 0);
    if (failed(reused))
      return failure();
    result = reused->str();
  } else {
    FailureOr<std::string> allocated =
        allocateResult(operation, 0, "fragment");
    if (failed(allocated))
      return failure();
    result = *allocated;
  }
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(extents))
    return failure();
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back("unary_i" + std::to_string(axis));
    loop += indices.back();
  }
  loop += " in T.Parallel(";
  for (auto [axis, extent] : llvm::enumerate(*extents)) {
    if (axis)
      loop += ", ";
    loop += extent;
  }
  line(loop + "):");
  ++indentation;
  FailureOr<std::string> operand =
      tensorElement(operation.getOperand(0), indices, operation);
  if (failed(operand))
    return failure();
  std::string expression = binding.getLowering() == "python_negate"
                               ? "-(" + *operand + ")"
                               : binding.getLowering().str() + "(" + *operand + ")";
  std::string target = result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + expression);
  --indentation;
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitBinary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "binary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding || operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local") ||
      (!tensorResult && binding.getReuseOperandAttr().getInt() != -1) ||
      (tensorResult &&
       (binding.getReuseOperandAttr().getInt() < -1 || binding.getReuseOperandAttr().getInt() > 1)))
    return operation.emitOpError("lacks a TileLang binary binding");
  std::string resultName = makeResultName(operation, 0);
  auto makeExpression = [&](StringRef lhs,
                            StringRef rhs) -> FailureOr<std::string> {
    if (binding.getLowering() == "T.max" ||
        binding.getLowering() == "T.min")
      return binding.getLowering().str() + "(" + lhs.str() + ", " +
             rhs.str() + ")";
    if (binding.getLowering().starts_with("T.bitwise_") ||
        binding.getLowering().starts_with("T.shift_"))
      return binding.getLowering().str() + "(" + lhs.str() + ", " +
             rhs.str() + ")";
    if (binding.getLowering() == "python_floor_divide" ||
        binding.getLowering() == "python_remainder") {
      std::string quotient = resultName + "_quotient";
      std::string remainder = resultName + "_remainder";
      std::string adjust = resultName + "_adjust";
      line(quotient + " = (" + lhs.str() + ") // (" + rhs.str() + ")");
      line(remainder + " = (" + lhs.str() + ") - " + quotient + " * (" +
           rhs.str() + ")");
      line(adjust + " = (" + remainder + " != 0) & ((" + remainder +
           " < 0) != (" + rhs.str() + " < 0))");
      return binding.getLowering() == "python_floor_divide"
                 ? quotient + " - T.if_then_else(" + adjust + ", 1, 0)"
                 : remainder + " + T.if_then_else(" + adjust + ", " +
                       rhs.str() + ", 0)";
    }
    StringRef symbol;
    if (binding.getLowering() == "python_add")
      symbol = "+";
    else if (binding.getLowering() == "python_subtract")
      symbol = "-";
    else if (binding.getLowering() == "python_multiply")
      symbol = "*";
    else if (binding.getLowering() == "python_true_divide")
      symbol = "/";
    else if (binding.getLowering() == "python_equal")
      symbol = "==";
    else if (binding.getLowering() == "python_not_equal")
      symbol = "!=";
    else if (binding.getLowering() == "python_less")
      symbol = "<";
    else if (binding.getLowering() == "python_less_equal")
      symbol = "<=";
    else if (binding.getLowering() == "python_greater")
      symbol = ">";
    else if (binding.getLowering() == "python_greater_equal")
      symbol = ">=";
    else {
      operation.emitOpError("uses an unsupported TileLang binary lowering");
      return failure();
    }
    return "(" + lhs.str() + ") " + symbol.str() + " (" + rhs.str() + ")";
  };
  if (!tensorResult) {
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    if (failed(lhs) || failed(rhs))
      return failure();
    FailureOr<std::string> expression = makeExpression(*lhs, *rhs);
    if (failed(expression))
      return failure();
    line(resultName + " = " + *expression);
    valueNames[operation.getResult(0)] = resultName;
    return success();
  }
  std::string result;
  if (binding.getReuseOperandAttr().getInt() >= 0) {
    FailureOr<StringRef> reused =
        lookupValue(operation, binding.getReuseOperandAttr().getInt());
    if (failed(reused))
      return failure();
    result = reused->str();
  } else {
    FailureOr<std::string> allocated =
        allocateResult(operation, 0, "fragment");
    if (failed(allocated))
      return failure();
    result = *allocated;
  }
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(extents))
    return failure();
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back("binary_i" + std::to_string(axis));
    loop += indices.back();
  }
  loop += " in T.Parallel(";
  for (auto [axis, extent] : llvm::enumerate(*extents)) {
    if (axis)
      loop += ", ";
    loop += extent;
  }
  line(loop + "):");
  ++indentation;
  FailureOr<std::string> lhs =
      tensorElement(operation.getOperand(0), indices, operation);
  FailureOr<std::string> rhs =
      tensorElement(operation.getOperand(1), indices, operation);
  if (failed(lhs) || failed(rhs))
    return failure();
  FailureOr<std::string> expression = makeExpression(*lhs, *rhs);
  if (failed(expression))
    return failure();
  FailureOr<std::string> padded = padElementExpression(
      operation.getResult(0), *expression, indices, operation);
  if (failed(padded))
    return failure();
  std::string target = result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + *padded);
  --indentation;
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitConditional(Operation &operation, bool mask) {
  FailureOr<int64_t> node = target::getNodeID(
      operation, mask ? "mask emission" : "select emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  int64_t reuse = binding ? binding.getReuseOperandAttr().getInt() : -2;
  unsigned conditionOperand = mask ? 1 : 0;
  unsigned trueOperand = mask ? 0 : 1;
  if (failed(node) || !binding ||
      binding.getLowering() != "T.if_then_else" ||
      operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local") ||
      (mask && !tensorResult) || (!tensorResult && reuse != -1) ||
      (tensorResult && reuse != -1 &&
       reuse != static_cast<int64_t>(trueOperand) && reuse != 2))
    return operation.emitOpError(
        "lacks a mechanical TileLang conditional binding");
  if (!tensorResult) {
    FailureOr<StringRef> condition = lookupValue(operation, conditionOperand);
    FailureOr<StringRef> trueValue = lookupValue(operation, trueOperand);
    FailureOr<StringRef> falseValue = lookupValue(operation, 2);
    if (failed(condition) || failed(trueValue) || failed(falseValue))
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = T.if_then_else(" + condition->str() + ", " +
         trueValue->str() + ", " + falseValue->str() + ")");
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  std::string result;
  if (reuse >= 0) {
    FailureOr<StringRef> reused = lookupValue(operation, reuse);
    if (failed(reused))
      return failure();
    result = reused->str();
  } else {
    FailureOr<std::string> allocated = allocateResult(operation, 0, "fragment");
    if (failed(allocated))
      return failure();
    result = *allocated;
  }
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(extents) || extents->empty())
    return failure();
  SmallVector<std::string> indices;
  std::string loop = "for ";
  StringRef prefix = mask ? "mask_i" : "select_i";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back(prefix.str() + std::to_string(axis));
    loop += indices.back();
  }
  loop += " in T.Parallel(";
  for (auto [axis, extent] : llvm::enumerate(*extents)) {
    if (axis)
      loop += ", ";
    loop += extent;
  }
  line(loop + "):");
  ++indentation;
  FailureOr<std::string> condition =
      tensorElement(operation.getOperand(conditionOperand), indices, operation);
  FailureOr<std::string> trueValue =
      tensorElement(operation.getOperand(trueOperand), indices, operation);
  FailureOr<std::string> falseValue =
      tensorElement(operation.getOperand(2), indices, operation);
  if (failed(condition) || failed(trueValue) || failed(falseValue))
    return failure();
  std::string expression = "T.if_then_else(" + *condition + ", " +
                           *trueValue + ", " + *falseValue + ")";
  FailureOr<std::string> padded = padElementExpression(
      operation.getResult(0), expression, indices, operation);
  if (failed(padded))
    return failure();
  std::string target = result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + *padded);
  --indentation;
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitMask(Operation &operation) {
  return emitConditional(operation, true);
}

LogicalResult SourceEmitter::emitSelect(Operation &operation) {
  return emitConditional(operation, false);
}

LogicalResult SourceEmitter::emitCast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "cast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding ||
      (binding.getLowering() != "T.cast" &&
       binding.getLowering() != "T.copy_cast") ||
      operation.getNumResults() != 1 ||
      binding.getReuseOperandAttr().getInt() != -1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local"))
    return operation.emitOpError("lacks a TileLang cast binding");
  if (!tensorResult) {
    if (binding.getLowering() != "T.cast")
      return operation.emitOpError("cannot copy-cast a scalar TileLang value");
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    std::string dtype = dtypeName(operation.getResult(0).getType(), operation);
    if (failed(operand) || dtype.empty())
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = T.cast(" + operand->str() + ", " + dtype + ")");
    valueNames[operation.getResult(0)] = result;
    return success();
  }
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  auto tensor = operation.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                    : RankedTensorType();
  if (failed(result) || failed(extents) || !tensor)
    return operation.emitOpError("lacks a TileLang cast binding");
  std::string dtype = dtypeName(tensor.getElementType(), operation);
  if (dtype.empty())
    return failure();
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back("cast_i" + std::to_string(axis));
    loop += indices.back();
  }
  loop += " in T.Parallel(";
  for (auto [axis, extent] : llvm::enumerate(*extents)) {
    if (axis)
      loop += ", ";
    loop += extent;
  }
  line(loop + "):");
  ++indentation;
  FailureOr<std::string> operand =
      tensorElement(operation.getOperand(0), indices, operation);
  if (failed(operand))
    return failure();
  std::string target = *result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = T.cast(" + *operand + ", " + dtype + ")");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitReshape(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reshape emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> shape = tensorShape(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "T.reshape" ||
      binding.getReuseOperandAttr().getInt() != -1 ||
      binding.getSpace() != "fragment" || failed(operand) || failed(shape) ||
      operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a TileLang reshape binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = T.reshape(" + operand->str() + ", " + *shape + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitTranspose(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "transpose emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<SmallVector<int64_t>> permutation =
      target::emission::transposePermutation(operation);
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(node) || !binding ||
      binding.getLowering() != "fragment_permute" ||
      binding.getReuseOperandAttr().getInt() != -1 ||
      binding.getSpace() != "fragment" || failed(permutation) ||
      failed(extents) || failed(result))
    return operation.emitOpError(
        "lacks a mechanical TileLang transpose binding");

  SmallVector<std::string> resultIndices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    resultIndices.push_back("transpose_i" + std::to_string(axis));
    loop += resultIndices.back();
  }
  loop += " in T.Parallel(";
  for (auto [axis, extent] : llvm::enumerate(*extents)) {
    if (axis)
      loop += ", ";
    loop += extent;
  }
  line(loop + "):");
  ++indentation;

  SmallVector<std::string> sourceIndices(resultIndices.size());
  for (auto [resultAxis, sourceAxis] : llvm::enumerate(*permutation))
    sourceIndices[sourceAxis] = resultIndices[resultAxis];
  FailureOr<std::string> operand =
      tensorElement(operation.getOperand(0), sourceIndices, operation);
  if (failed(operand))
    return failure();
  std::string target = *result + "[";
  for (auto [axis, index] : llvm::enumerate(resultIndices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + *operand);
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitFull(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "full emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> fill = lookupValue(operation, 0);
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(node) || !binding || binding.getLowering() != "T.fill" ||
      binding.getReuseOperandAttr().getInt() != -1 || failed(fill) || failed(result))
    return operation.emitOpError("lacks a TileLang full binding");
  line("T.fill(" + *result + ", " + fill->str() + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitZeros(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "zeros emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(node) || !binding || binding.getLowering() != "T.clear" ||
      binding.getReuseOperandAttr().getInt() != -1 || failed(result))
    return operation.emitOpError("lacks a TileLang zeros binding");
  line("T.clear(" + *result + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitGather(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "gather emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  auto validIndex =
      operation.getAttrOfType<IntegerAttr>("intent.valid_operand_index");
  auto fillIndex =
      operation.getAttrOfType<IntegerAttr>("intent.fill_operand_index");
  int64_t reuseOperand =
      binding ? binding.getReuseOperandAttr().getInt() : int64_t{-2};
  if (failed(node) || !binding || reuseOperand < -1 ||
      reuseOperand >= static_cast<int64_t>(operation.getNumOperands()) ||
      failed(relation) || !validIndex || !fillIndex)
    return operation.emitOpError("lacks a TileLang gather binding");
  auto resultStorage = [&]() -> FailureOr<std::string> {
    if (reuseOperand < 0)
      return allocateResult(operation, 0, "fragment");
    FailureOr<StringRef> reused = lookupValue(operation, reuseOperand);
    if (failed(reused))
      return failure();
    return reused->str();
  };
  if (!planIndex.stages.empty() && binding.getLowering() == "T.indirect_gather") {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view))
      return failure();
    if (binding.getDefer() && (*view)->tensor.getRank() == 2) {
      deferredLoads[operation.getResult(0)] = &operation;
      return success();
    }
    if ((*view)->tensor.getRank() != 1 || relation->size() != 1 ||
        (*relation)[0].kind != "value_index" ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front())
      return operation.emitOpError(
          "staged indirect gather requires one indexed vector source");
    FailureOr<StringRef> indices =
        lookupValue(operation, *(*relation)[0].operands.front());
    auto element = [&](IntegerAttr operandIndex) -> FailureOr<std::string> {
      FailureOr<StringRef> value = lookupValue(operation, operandIndex.getInt());
      if (failed(value))
        return failure();
      auto tensor = dyn_cast<RankedTensorType>(
          operation.getOperand(operandIndex.getInt()).getType());
      if (!tensor)
        return value->str();
      if (tensor.getRank() != 1) {
        operation.emitOpError(
            "requires scalar or rank-one ragged gather predicates and fills");
        return failure();
      }
      return value->str() + "[gather_i]";
    };
    FailureOr<std::string> valid = element(validIndex);
    FailureOr<std::string> fill = element(fillIndex);
    FailureOr<std::string> allocated = resultStorage();
    if (failed(indices) || failed(valid) || failed(fill) || failed(allocated))
      return failure();
    std::string result = *allocated;
    line("for gather_i in T.Parallel(TILE_SIZE_M):");
    ++indentation;
    line(result + "[gather_i] = T.if_then_else(member_start + gather_i < "
         "route_end and " + *valid + ", " +
         (*view)->argument->name + "[" +
         addressIndex(indices->str() + "[gather_i]") + "], " +
         *fill + ")");
    --indentation;
    bindResult(operation, 0, result);
    return success();
  }
  bool appendAxis = relation->size() == 2 &&
                    (*relation)[0].kind == "full_slice" &&
                    (*relation)[1].kind == "new_axis";
  bool prependAxis = relation->size() == 2 &&
                     (*relation)[0].kind == "new_axis" &&
                     (*relation)[1].kind == "full_slice";
  if (binding.getLowering() != "expand_dims" ||
      (!appendAxis && !prependAxis))
    return operation.emitOpError("has no mechanical TileLang gather relation");
  FailureOr<StringRef> source = lookupValue(operation, 0);
  FailureOr<StringRef> valid = lookupValue(operation, validIndex.getInt());
  FailureOr<StringRef> fill = lookupValue(operation, fillIndex.getInt());
  FailureOr<std::string> result = resultStorage();
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(source) || failed(valid) || failed(fill) || failed(result) ||
      failed(extents) || extents->size() != 2)
    return failure();
  auto sourceTensor = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  line("for gather_i, gather_j in T.Parallel(" + (*extents)[0] + ", " +
       (*extents)[1] + "):");
  ++indentation;
  std::string sourceElement = sourceTensor && sourceTensor.getRank() == 1
                                  ? source->str() +
                                        (appendAxis ? "[gather_i]"
                                                    : "[gather_j]")
                                  : sourceTensor
                                        ? source->str() + "[gather_i, gather_j]"
                                        : source->str();
  line(*result + "[gather_i, gather_j] = T.if_then_else(" + valid->str() +
       ", " + sourceElement + ", " + fill->str() + ")");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitMembers(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "members emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  if (failed(node) || !binding || binding.getLowering() != "T.members" ||
      binding.getReuseOperandAttr().getInt() != -1 ||
      operation.getNumOperands() != 1 || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a staged TileLang members binding");
  FailureOr<plan::AxisOp> memberAxis =
      resolveAxis(operation.getOperand(0), operation);
  FailureOr<plan::RaggedOp> relation =
      succeeded(memberAxis)
          ? target::emission::uniqueRaggedRelation(
                planIndex, memberAxis->getNode(), operation)
          : FailureOr<plan::RaggedOp>(failure());
  auto runtime = succeeded(relation)
                     ? raggedRuntimeByRelation.find(relation->getNode())
                     : raggedRuntimeByRelation.end();
  if (failed(memberAxis) || failed(relation) ||
      runtime == raggedRuntimeByRelation.end())
    return operation.emitOpError("has no ragged runtime for its member axis");
  RaggedRuntime &ragged = raggedRuntimes[runtime->second];
  if (planIndex.components.orderedRaggedAxes.contains(memberAxis->getNode())) {
    if (!activeStages.empty())
      return operation.emitOpError(
          "cannot mix ordered and staged ragged member projection");
    std::string position = axisIndices.lookup(memberAxis->getNode());
    if (position.empty())
      return operation.emitOpError("has no ordered ragged runtime position");
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
    if (failed(result))
      return failure();
    std::string memberIndex = "ordered_member_i_" + std::to_string(*node);
    std::string absolute = position + " + " + memberIndex;
    std::string suffix = std::to_string(memberAxis->getNode());
    line("for " + memberIndex + " in T.Parallel(" +
         memberAxis->getTile().str() + "):");
    ++indentation;
    std::string member = ragged.indices
                             ? ragged.indices->argument->name + "[" +
                                   addressIndex(absolute) + "]"
                             : absolute;
    line(*result + "[" + memberIndex + "] = T.if_then_else(" + absolute +
         " < sequence_end_" + suffix + ", " + member + ", 0)");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }
  if (activeStages.empty())
    return operation.emitOpError(
        "has neither ordered traversal nor staged ragged ownership");
  for (unsigned stage : activeStages)
    if (!stageRaggedRuntime.count(stage) ||
        stageRaggedRuntime.lookup(stage) != runtime->second)
      return operation.emitOpError(
          "is shared by stages with inconsistent ragged ownership");
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(result))
    return failure();
  line("for member_i in T.Parallel(TILE_SIZE_M):");
  ++indentation;
  std::string member = ragged.indices
                           ? ragged.indices->argument->name +
                                 "[" + addressIndex("member_start + member_i") +
                                 "]"
                           : "member_start + member_i";
  line(*result + "[member_i] = T.if_then_else(member_start + member_i < "
       "route_end, " + member + ", 0)");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical TileLang stream binding");
  Block &body = operation.getRegion(0).front();
  bool hasStop = static_cast<bool>(binding.getStopNodeAttr());
  bool hasRuntimeExtent = static_cast<bool>(
      operation.getAttrOfType<IntegerAttr>("intent.extent_operand_index"));
  if (operation.getNumOperands() !=
          operation.getNumResults() + 1 + (hasRuntimeExtent ? 1 : 0) +
              (hasStop ? 1 : 0) ||
      body.getNumArguments() != operation.getNumResults() + 1)
    return operation.emitOpError("has inconsistent TileLang stream state");
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    if (failed(initial))
      return failure();
    if (operation.getOperand(index + 1).getType() !=
        operation.getResult(index).getType())
      return operation.emitOpError(
          "has a TileLang stream carrier type mismatch");
    Type carrierType = operation.getResult(index).getType();
    if (carrierType.isIntOrIndexOrFloat()) {
      std::string dtype = dtypeName(carrierType, operation);
      if (dtype.empty())
        return failure();
      std::string carrier = uniqueName("stream_state_" + std::to_string(index),
                                       *node);
      line(carrier + " = T.alloc_local((1,), " + dtype + ")");
      line(carrier + "[0] = " + initial->str());
      carriers.push_back(carrier);
      valueNames[body.getArgument(index + 1)] = carrier + "[0]";
    } else {
      carriers.push_back(initial->str());
      valueNames[body.getArgument(index + 1)] = initial->str();
    }
  }
  streamCarriers[&operation] = carriers;
  bool raggedStream =
      planIndex.components.orderedRaggedAxes.contains(binding.getAxisNode());
  Operation *streamDomain = kernel.nodes.lookup(binding.getAxisNode());
  std::string raggedSuffix = std::to_string(binding.getAxisNode());
  if (raggedStream) {
    FailureOr<plan::RaggedOp> relation =
        target::emission::uniqueRaggedRelation(
            planIndex, binding.getAxisNode(), operation);
    auto runtime = succeeded(relation)
                       ? raggedRuntimeByRelation.find(relation->getNode())
                       : raggedRuntimeByRelation.end();
    std::string outer = succeeded(relation)
                            ? axisIndices.lookup(relation->getOuterNode())
                            : std::string();
    if (failed(relation) || runtime == raggedRuntimeByRelation.end() ||
        outer.empty())
      return operation.emitOpError(
          "ordered ragged stream has no outer-axis metadata binding");
    RaggedRuntime &ragged = raggedRuntimes[runtime->second];
    line("sequence_begin_" + raggedSuffix + " = " +
         ragged.offsets->argument->name + "[" + addressIndex(outer) + "]");
    line("sequence_end_" + raggedSuffix + " = " +
         ragged.offsets->argument->name + "[" +
         addressIndex(outer + " + 1") + "]");
    line("sequence_length_" + raggedSuffix + " = sequence_end_" +
         raggedSuffix + " - sequence_begin_" + raggedSuffix);
  }
  FailureOr<std::string> logicalExtent =
      streamDomain ? dimensionName(*streamDomain)
                   : FailureOr<std::string>(failure());
  if (failed(logicalExtent))
    return binding.emitOpError("has no logical ordered-axis extent");
  std::string streamExtent =
      raggedStream ? "sequence_length_" + raggedSuffix : *logicalExtent;
  if (hasStop) {
    auto stopIndex =
        operation.getAttrOfType<IntegerAttr>("intent.stop_operand_index");
    int64_t expectedStop = operation.getNumResults() + 1 +
                           (hasRuntimeExtent ? 1 : 0);
    auto stop = kernel.nodes.find(binding.getStopNodeAttr().getInt());
    Operation *stopOperation =
        stop == kernel.nodes.end() ? nullptr : stop->second;
    if (!stopIndex || stopIndex.getInt() != expectedStop ||
        !stopOperation ||
        stopOperation->getName().getStringRef() != "intent.region_end" ||
        stopOperation->getNumOperands() != 1 ||
        operation.getOperand(stopIndex.getInt()).getDefiningOp() != stopOperation)
      return operation.emitOpError(
          "does not match its planned logical stream stop");
    FailureOr<plan::AxisOp> stopAxis =
        resolveAxis(stopOperation->getOperand(0), operation);
    if (failed(stopAxis))
      return failure();
    if (stopAxis->hasRole("parallel") && !stopAxis->isScalar()) {
      std::string block = programBlocks.lookup(stopAxis->getNode());
      if (block.empty())
        return stopAxis->emitOpError("has no physical program block index");
      streamExtent = "T.min((" + block + " + 1) * " +
                     stopAxis->getTile().str() + ", " + streamExtent + ")";
    } else if (stopAxis->getNode() != binding.getAxisNode()) {
      return operation.emitOpError(
          "has no TileLang spelling for its planned logical stream stop");
    }
  }
  std::string streamTile = "stream_tile_" + std::to_string(*node);
  line("for " + streamTile + " in T.Pipelined(T.ceildiv(" + streamExtent +
       ", " + binding.getTile().str() +
       "), num_stages=num_stages):");
  ++indentation;
  std::string streamStart = "axis_index_" + std::to_string(binding.getAxisNode());
  line(streamStart + " = " +
       std::string(raggedStream ? "sequence_begin_" + raggedSuffix + " + " : "") +
       addressIndex(streamTile) + " * " + binding.getTile().str());
  axisIndices[binding.getAxisNode()] = streamStart;
  valueNames[body.getArgument(0)] = streamStart;
  return success();
}

LogicalResult SourceEmitter::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  if (carriers == streamCarriers.end())
    return operation.emitOpError("has no active TileLang stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (terminator.getName().getStringRef() != "intent.yield" ||
      terminator.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("does not yield every TileLang stream state");
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(terminator, index);
    if (failed(yielded))
      return failure();
    if (operation.getResult(index).getType().isIntOrIndexOrFloat()) {
      std::string destination = carriers->second[index] + "[0]";
      if (*yielded != destination)
        line(destination + " = " + yielded->str());
    } else if (*yielded != carriers->second[index]) {
      line("T.copy(" + yielded->str() + ", " + carriers->second[index] + ")");
    }
  }
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    bool scalar = operation.getResult(index).getType().isIntOrIndexOrFloat();
    valueNames[operation.getResult(index)] =
        carriers->second[index] + (scalar ? "[0]" : "");
  }
  return success();
}

LogicalResult SourceEmitter::emitContract(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "T.gemm" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a TileLang contraction binding");
  FailureOr<target::emission::ContractionOrientation> orientation =
      target::emission::contractionOrientation(operation);
  if (failed(orientation))
    return failure();
  if (!planIndex.stages.empty()) {
    if (binding.getLhsSpace() != "shared" ||
        binding.getRhsSpace() != "shared" ||
        binding.getAccumulatorSpace() != "fragment")
      return operation.emitOpError(
          "staged TileLang contraction has inconsistent operand spaces");
    if (activeStages.size() != 1)
      return operation.emitOpError(
          "must belong to exactly one resolved physical stage");
    unsigned stage = activeStages.front();
    Operation *lhsAccess = deferredLoads.lookup(operation.getOperand(0));
    Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
    FailureOr<ABIView *> rhsView =
        rhsLoad && rhsLoad->getNumOperands() > 0
            ? lookupView(rhsLoad->getOperand(0), *rhsLoad)
            : FailureOr<ABIView *>(failure());
    if (!rhsLoad || failed(rhsView) || (*rhsView)->tensor.getRank() != 3 ||
        orientation->lhsTranspose || orientation->rhsTranspose)
      return operation.emitOpError(
          "staged contraction requires one expert-selected rank-three weight");
    auto operandElementType = [](Value value) -> Type {
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      return tensor ? tensor.getElementType() : Type();
    };
    Type lhsElement = operandElementType(operation.getOperand(0));
    Type rhsElement = operandElementType(operation.getOperand(1));
    bool promoteToF32 = lhsElement.isF32() || rhsElement.isF32();
    if (!lhsElement || !rhsElement ||
        (!promoteToF32 && lhsElement != rhsElement) ||
        (!lhsElement.isF16() && !lhsElement.isBF16() && !lhsElement.isF32()) ||
        (!rhsElement.isF16() && !rhsElement.isBF16() && !rhsElement.isF32()))
      return operation.emitOpError(
          "has unsupported staged matrix operand types");
    std::string reduction = stageReductionDimensions.lookup(stage);
    std::string matrixDtype =
        promoteToF32 ? "T.float32" : dtypeName(lhsElement, operation);
    if (matrixDtype.empty())
      return failure();
    std::string lhs = makeResultName(operation, 0) + "_lhs";
    std::string rhs = makeResultName(*rhsLoad, 0) + "_shared";
    std::string result = makeResultName(operation, 0);
    line(lhs + " = T.alloc_shared((TILE_SIZE_M, TILE_SIZE_K), " +
         matrixDtype + ")");
    line(rhs + " = T.alloc_shared((TILE_SIZE_K, TILE_SIZE_N), " +
         matrixDtype + ")");
    line(result +
         " = T.alloc_fragment((TILE_SIZE_M, TILE_SIZE_N), T.float32)");
    line("T.clear(" + result + ")");
    line("for k_tile in T.Pipelined(T.ceildiv(" + reduction +
         ", TILE_SIZE_K), num_stages=num_stages):");
    ++indentation;
    if (lhsAccess) {
      if (lhsAccess->getName().getStringRef() != "intent.gather")
        return lhsAccess->emitOpError(
            "is not a staged indirect contraction input");
      FailureOr<SmallVector<target::IndexTerm>> relation =
          target::parseIndexRelation(*lhsAccess);
      FailureOr<ABIView *> lhsView =
          lookupView(lhsAccess->getOperand(0), *lhsAccess);
      auto runtime = stageRaggedRuntime.find(stage);
      bool contiguous =
          runtime != stageRaggedRuntime.end() &&
          runtime->second < raggedRuntimes.size() &&
          !raggedRuntimes[runtime->second].indices;
      if (failed(relation) || failed(lhsView) || relation->size() != 2 ||
          (*relation)[0].kind != "value_index" ||
          (*relation)[0].operands.size() != 1 ||
          !(*relation)[0].operands.front() ||
          (*relation)[1].kind != "full_slice")
        return lhsAccess->emitOpError(
            "has no staged row-gather contraction relation");
      FailureOr<StringRef> rows =
          lookupValue(*lhsAccess, *(*relation)[0].operands.front());
      if (failed(rows))
        return failure();
      if (contiguous) {
        line("T.copy(" + (*lhsView)->argument->name +
             "[" + addressIndex("member_start") + " : " +
             addressIndex("member_start") + " + TILE_SIZE_M, " +
             addressIndex("k_tile") + " * TILE_SIZE_K : " +
             addressIndex("k_tile + 1") + " * TILE_SIZE_K], " +
             lhs + ")");
      } else {
        line("for load_i, load_k in T.Parallel(TILE_SIZE_M, TILE_SIZE_K):");
        ++indentation;
        line(lhs + "[load_i, load_k] = T.if_then_else(member_start + load_i < "
             "route_end and k_tile * TILE_SIZE_K + load_k < " + reduction +
             ", " + (*lhsView)->argument->name + "[" +
             addressIndex(rows->str() + "[load_i]") + ", " +
             addressIndex("k_tile * TILE_SIZE_K + load_k") + "], 0.0)");
        --indentation;
      }
    } else {
      auto workspace = workspaceNames.find(operation.getOperand(0));
      if (workspace == workspaceNames.end())
        return operation.emitOpError(
            "staged contraction input has no materialized workspace");
      line("for load_i, load_k in T.Parallel(TILE_SIZE_M, TILE_SIZE_K):");
      ++indentation;
      line(lhs + "[load_i, load_k] = T.if_then_else(member_start + load_i < "
           "route_end and k_tile * TILE_SIZE_K + load_k < " + reduction +
           ", " + workspace->second + "[" +
           addressIndex("member_start + load_i") + ", " +
           addressIndex("k_tile * TILE_SIZE_K + load_k") + "], 0.0)");
      --indentation;
    }
    line("T.copy(" + (*rhsView)->argument->name +
         "[" + addressIndex("expert") + ", " + addressIndex("k_tile") +
         " * TILE_SIZE_K : " + addressIndex("k_tile + 1") +
         " * TILE_SIZE_K, " + addressIndex("bid_feature") +
         " * TILE_SIZE_N : " + addressIndex("bid_feature + 1") +
         " * TILE_SIZE_N], " +
         rhs + ")");
    line("T.gemm(" + lhs + ", " + rhs + ", " + result +
         ", policy=T.GemmWarpPolicy.FullRow)");
    --indentation;
    bindResult(operation, 0, result);
    return success();
  }

  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (lhsLoad || rhsLoad) {
    if (!lhsLoad || !rhsLoad)
      return operation.emitOpError(
          "deferred TileLang contraction has inconsistent operand residency");
    if (binding.getLhsSpace() != "shared" ||
        binding.getRhsSpace() != "shared" ||
        binding.getAccumulatorSpace() != "fragment")
      return operation.emitOpError(
          "deferred TileLang contraction has inconsistent plan spaces");
    FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
    FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
    FailureOr<plan::AxisOp> reductionAxis =
        target::emission::contractionReductionAxis(planIndex, *lhsLoad,
                                                   *rhsLoad, operation);
    if (failed(reductionAxis))
      return failure();
    axisIndices[reductionAxis->getNode()] =
        addressIndex("k_tile") + " * " + reductionAxis->getTile().str();
    FailureOr<std::string> lhsIndices = accessIndices(*lhsLoad);
    FailureOr<std::string> rhsIndices = accessIndices(*rhsLoad);
    FailureOr<std::string> lhsShape = tensorShape(*lhsLoad, 0);
    FailureOr<std::string> rhsShape = tensorShape(*rhsLoad, 0);
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
    if (failed(lhsView) || failed(rhsView) || failed(lhsIndices) ||
        failed(rhsIndices) || failed(lhsShape) || failed(rhsShape) ||
        failed(result))
      return failure();
    std::string lhs = makeResultName(*lhsLoad, 0) + "_shared";
    std::string rhs = makeResultName(*rhsLoad, 0) + "_shared";
    line(lhs + " = T.alloc_shared(" + *lhsShape + ", " +
         dtypeName((*lhsView)->tensor.getElementType(), *lhsLoad) + ")");
    line(rhs + " = T.alloc_shared(" + *rhsShape + ", " +
         dtypeName((*rhsView)->tensor.getElementType(), *rhsLoad) + ")");
    line("T.clear(" + *result + ")");
    line("for k_tile in T.Pipelined(T.ceildiv(" +
         roleDimensions.lookup(reductionAxis->getRole()) + ", " +
         reductionAxis->getTile().str() + "), num_stages=num_stages):");
    ++indentation;
    line("T.copy(" + (*lhsView)->argument->name + "[" + *lhsIndices + "], " +
         lhs + ")");
    line("T.copy(" + (*rhsView)->argument->name + "[" + *rhsIndices + "], " +
         rhs + ")");
    std::string call = "T.gemm(" + lhs + ", " + rhs + ", " + *result;
    if (orientation->lhsTranspose)
      call += ", transpose_A=True";
    if (orientation->rhsTranspose)
      call += ", transpose_B=True";
    line(call + ", policy=T.GemmWarpPolicy.FullRow)");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }

  FailureOr<StringRef> lhs = lookupValue(operation, 0);
  FailureOr<StringRef> rhs = lookupValue(operation, 1);
  if ((binding.getLhsSpace() != "shared" &&
       binding.getLhsSpace() != "fragment") ||
      (binding.getRhsSpace() != "shared" &&
       binding.getRhsSpace() != "fragment") ||
      binding.getAccumulatorSpace() != "fragment")
    return operation.emitOpError(
        "direct TileLang contraction has invalid projected spaces");
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  line("T.clear(" + *result + ")");
  auto valueExtents = [&](Value value)
      -> FailureOr<SmallVector<std::string>> {
    auto opResult = dyn_cast<OpResult>(value);
    if (!opResult)
      return failure();
    return tensorExtents(*opResult.getOwner(), opResult.getResultNumber());
  };
  FailureOr<SmallVector<std::string>> lhsExtents =
      valueExtents(operation.getOperand(0));
  FailureOr<SmallVector<std::string>> rhsExtents =
      valueExtents(operation.getOperand(1));
  FailureOr<SmallVector<std::string>> resultExtents =
      tensorExtents(operation, 0);
  auto reduction = operation.getAttrOfType<ArrayAttr>("intent.reduce");
  auto pair = reduction && reduction.size() == 1
                  ? dyn_cast<ArrayAttr>(reduction[0])
                  : ArrayAttr();
  auto lhsReduction = pair && pair.size() == 2
                          ? dyn_cast<IntegerAttr>(pair[0])
                          : IntegerAttr();
  auto rhsReduction = pair && pair.size() == 2
                          ? dyn_cast<IntegerAttr>(pair[1])
                          : IntegerAttr();
  bool unitRow = succeeded(lhsExtents) && succeeded(rhsExtents) &&
                 succeeded(resultExtents) && lhsExtents->size() == 2 &&
                 rhsExtents->size() == 2 && resultExtents->size() == 2 &&
                 (*resultExtents)[0] == "1" && lhsReduction && rhsReduction &&
                 lhsReduction.getInt() == 1 &&
                 (rhsReduction.getInt() == 0 || rhsReduction.getInt() == 1);
  if (unitRow) {
    auto resultTensor =
        dyn_cast<RankedTensorType>(operation.getResult(0).getType());
    std::string accumulatorDtype =
        resultTensor ? dtypeName(resultTensor.getElementType(), operation) : "";
    if (accumulatorDtype.empty())
      return failure();
    std::string reductionExtent = (*lhsExtents)[1];
    std::string outputExtent =
        (*rhsExtents)[rhsReduction.getInt() == 0 ? 1 : 0];
    std::string products = makeResultName(operation, 0) + "_products";
    line(products + " = T.alloc_fragment((1, " + outputExtent + ", " +
         reductionExtent + "), " + accumulatorDtype + ")");
    line("for contract_j, contract_k in T.Parallel(" + outputExtent + ", " +
         reductionExtent + "):");
    ++indentation;
    std::string rhsElement = rhsReduction.getInt() == 0
                                 ? rhs->str() + "[contract_k, contract_j]"
                                 : rhs->str() + "[contract_j, contract_k]";
    line(products + "[0, contract_j, contract_k] = T.cast(" + lhs->str() +
         "[0, contract_k], " + accumulatorDtype + ") * T.cast(" + rhsElement +
         ", " + accumulatorDtype + ")");
    --indentation;
    line("T.reduce_sum(" + products + ", " + *result +
         ", dim=2, clear=True)");
    bindResult(operation, 0, *result);
    return success();
  }
  std::string call = "T.gemm(" + lhs->str() + ", " + rhs->str() + ", " +
                     *result;
  if (orientation->lhsTranspose)
    call += ", transpose_A=True";
  if (orientation->rhsTranspose)
    call += ", transpose_B=True";
  line(call + ", policy=T.GemmWarpPolicy.FullRow)");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitStore(Operation &operation) {
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<int64_t> node = target::getNodeID(operation, "store emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  FailureOr<StringRef> stored =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (!valueIndex || failed(node) || !boundary || failed(stored) ||
      failed(view))
    return operation.emitOpError("lacks a TileLang store binding");
  FailureOr<std::string> physicalFill = transferPhysicalExtentFill(operation);
  if (failed(physicalFill))
    return failure();
  bool expanded = !physicalFill->empty();
  if (boundary.getTransfer() == "parallel_elements" || expanded) {
    Value storedValue = operation.getOperand(valueIndex.getInt());
    auto result = dyn_cast<OpResult>(storedValue);
    Operation *definition = result ? result.getOwner() : nullptr;
    FailureOr<SmallVector<std::string>> extents =
        definition ? tensorExtents(*definition, result.getResultNumber())
                   : FailureOr<SmallVector<std::string>>(failure());
    if (failed(extents) || extents->empty() ||
        boundary.getPadding() != "none")
      return operation.emitOpError(
          "has an invalid parallel TileLang store transfer");
    SmallVector<std::string> tileIndices;
    std::string loop = "for ";
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis)
        loop += ", ";
      tileIndices.push_back("transfer_i" + std::to_string(axis));
      loop += tileIndices.back();
    }
    loop += " in T.Parallel(";
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis)
        loop += ", ";
      loop += (*extents)[axis];
    }
    FailureOr<std::string> indices =
        elementAccessIndices(operation, tileIndices);
    if (failed(indices))
      return failure();
    FailureOr<std::string> logicalPredicate =
        boundary.getCheckBounds()
            ? elementBoundsPredicate(operation, tileIndices, false, false)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> physicalPredicate =
        expanded ? elementBoundsPredicate(operation, tileIndices, true)
                 : FailureOr<std::string>(std::string());
    if (failed(logicalPredicate) || failed(physicalPredicate))
      return failure();
    line(loop + "):");
    ++indentation;
    if (boundary.getCheckBounds()) {
      line("if " + *logicalPredicate + ":");
      ++indentation;
    }
    if (expanded) {
      line("if " + *physicalPredicate + ":");
      ++indentation;
    }
    std::string source = stored->str() + "[";
    for (auto [axis, index] : llvm::enumerate(tileIndices)) {
      if (axis)
        source += ", ";
      source += index;
    }
    line((*view)->argument->name + "[" + *indices + "] = " + source + "]");
    if (expanded)
      --indentation;
    if (boundary.getCheckBounds())
      --indentation;
    --indentation;
    return success();
  }
  if (boundary.getTransfer() != "bulk_copy")
    return operation.emitOpError("has no TileLang store transfer emitter");
  FailureOr<std::string> indices = accessIndices(operation);
  if (failed(indices))
    return failure();
  if (!isa<RankedTensorType>(operation.getOperand(valueIndex.getInt()).getType())) {
    line((*view)->argument->name + "[" + *indices + "] = " + stored->str());
    return success();
  }
  line("T.copy(" + stored->str() + ", " + (*view)->argument->name + "[" +
       *indices + "])");
  return success();
}

LogicalResult SourceEmitter::emitUniqueStore(Operation &operation) {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "unique-store emission");
  plan::BoundaryOp binding =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<StringRef> stored =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  if (failed(node) || !binding || binding.getAccess() != "store" ||
      !binding.getCheckBounds() || binding.getDefer() || !valueIndex ||
      failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "value_index" ||
      (*relation)[0].operands.size() != 1 ||
      !(*relation)[0].operands.front() ||
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical TileLang unique store");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  line("for store_i, store_j in T.Parallel(TILE_SIZE_M, TILE_SIZE_N):");
  ++indentation;
  line("if member_start + store_i < route_end and bid_feature * "
       "TILE_SIZE_N + store_j < " +
       stageFeatureDimensions.lookup(activeStages.front()) + ":");
  ++indentation;
  line((*view)->argument->name + "[" + rows->str() +
       "[store_i], " + addressIndex("bid_feature * TILE_SIZE_N + store_j") +
       "] = " + stored->str() +
       "[store_i, store_j]");
  --indentation;
  --indentation;
  return success();
}

LogicalResult SourceEmitter::emitAtomic(Operation &operation) {
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<StringRef> stored =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  if (!valueIndex || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical TileLang atomic merge");
  if (planIndex.stages.empty()) {
    Value storedValue = operation.getOperand(valueIndex.getInt());
    if (!isa<RankedTensorType>(storedValue.getType())) {
      FailureOr<int64_t> node =
          target::getNodeID(operation, "atomic emission");
      plan::BoundaryOp boundary =
          succeeded(node) ? planIndex.boundaries.lookup(*node)
                          : plan::BoundaryOp();
      FailureOr<std::string> indices = elementAccessIndices(operation, {});
      if (failed(node) || !boundary || failed(indices))
        return operation.emitOpError(
            "lacks a mechanical scalar TileLang atomic merge");
      if (boundary.getCheckBounds()) {
        FailureOr<std::string> predicate =
            elementBoundsPredicate(operation, {});
        if (failed(predicate))
          return failure();
        line("if " + *predicate + ":");
        ++indentation;
      }
      line("T.atomic_add(" + (*view)->argument->name + "[" + *indices +
           "], " + stored->str() + ", memory_order=\"relaxed\")");
      if (boundary.getCheckBounds())
        --indentation;
      return success();
    }
    auto result = dyn_cast<OpResult>(storedValue);
    FailureOr<SmallVector<std::string>> extents =
        result ? tensorExtents(*result.getOwner(), result.getResultNumber())
               : FailureOr<SmallVector<std::string>>(failure());
    if (failed(extents) || extents->empty())
      return operation.emitOpError(
          "regular TileLang atomic merge requires a ranked value");
    SmallVector<std::string> tileIndices;
    std::string loop = "for ";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        loop += ", ";
      tileIndices.push_back("atomic_i" + std::to_string(axis));
      loop += tileIndices.back();
    }
    loop += " in T.Parallel(";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        loop += ", ";
      loop += extent;
    }
    loop += "):";
    FailureOr<std::string> indices =
        elementAccessIndices(operation, tileIndices);
    FailureOr<std::string> predicate =
        elementBoundsPredicate(operation, tileIndices);
    FailureOr<std::string> value =
        tensorElement(storedValue, tileIndices, operation);
    if (failed(indices) || failed(predicate) || failed(value))
      return failure();
    line(loop);
    ++indentation;
    line("if " + *predicate + ":");
    ++indentation;
    line("T.atomic_add(" + (*view)->argument->name + "[" + *indices + "], " +
         *value + ", memory_order=\"relaxed\")");
    --indentation;
    --indentation;
    return success();
  }
  if (!valueIndex || failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "value_index" ||
      (*relation)[0].operands.size() != 1 ||
      !(*relation)[0].operands.front() ||
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical TileLang atomic merge");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  line("for atomic_i, atomic_j in T.Parallel(TILE_SIZE_M, TILE_SIZE_N):");
  ++indentation;
  line("if member_start + atomic_i < route_end and bid_feature * "
       "TILE_SIZE_N + atomic_j < " + stageFeatureDimensions.lookup(activeStages.front()) +
       ":");
  ++indentation;
  line("T.atomic_add(" + (*view)->argument->name + "[" + rows->str() +
       "[atomic_i], " + addressIndex("bid_feature * TILE_SIZE_N + atomic_j") +
       "], " + stored->str() +
       "[atomic_i, atomic_j], memory_order=\"relaxed\")");
  --indentation;
  --indentation;
  return success();
}

} // namespace intent::tilelang::emission

#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Analysis/StructuredControl.h"
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
  for (StringRef name : {"intent.domain", "intent.domain_product",
                         "intent.make_record",
                         "intent.region_end",
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
      failed(addHandler(registry, "intent.extract", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitExtract(op);
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
          registry, "intent.ordered",
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
      failed(addHandler(
          registry, "intent.while",
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.enterWhile(op)
                                               : success();
          },
          [&](Operation &op) {
            return emitter.selectOperation(op) ? emitter.leaveWhile(op)
                                               : success();
          })) ||
      failed(addHandler(registry, "intent.condition", [&](Operation &op) {
        return emitter.selectOperation(op) ? emitter.emitCondition(op) : success();
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
      })) ||
      failed(addHandler(registry, "intent.atomic_cas",
                        [&](Operation &op) -> LogicalResult {
        if (!emitter.selectOperation(op))
          return success();
        return op.emitOpError(
            "cannot lower compare-and-swap because the TileLang surface has no CAS primitive");
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
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::emitExtract(Operation &operation) {
  FailureOr<Value> field = target::resolveRecordField(operation);
  if (failed(field))
    return failure();
  auto found = valueNames.find(*field);
  if (found == valueNames.end())
    return operation.emitOpError("record field has no emitted TileLang SSA value");
  bindResult(operation, 0, found->second);
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
  auto producer = operation.getNumOperands() > 0
                      ? operation.getOperand(0).getDefiningOp()
                      : nullptr;
  if (producer && scanProducerOwners.count(producer))
    return success();
  FailureOr<StringRef> index = lookupValue(operation, 0);
  if (operation.getNumOperands() != 2 || operation.getNumResults() != 0 ||
      !axis || axis.getInt() < 0 || failed(index))
    return operation.emitOpError(
        "lacks a mechanical TileLang in-bounds assumption binding");
  std::string bound;
  if (isa<intent::ViewType>(operation.getOperand(1).getType())) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(1), operation);
    if (failed(view) || axis.getInt() >= (*view)->tensor.getRank())
      return operation.emitOpError(
          "lacks a mechanical TileLang view-bound assumption");
    bound = (*view)->shape[axis.getInt()];
  } else {
    Operation *buffer = operation.getOperand(1).getDefiningOp();
    FailureOr<target::LogicalBufferInfo> info =
        buffer ? target::getLogicalBufferInfo(*buffer)
               : FailureOr<target::LogicalBufferInfo>(failure());
    if (failed(info) ||
        axis.getInt() >= static_cast<int64_t>(info->shape.size()))
      return operation.emitOpError(
          "lacks a mechanical TileLang logical-buffer assumption");
    bound = std::to_string(info->shape[axis.getInt()]);
  }

  Type indexType = operation.getOperand(0).getType();
  std::string expression = index->str();
  if (auto tensor = dyn_cast<RankedTensorType>(indexType)) {
    auto result = dyn_cast<OpResult>(operation.getOperand(0));
    FailureOr<SmallVector<std::string>> extents =
        result ? tensorExtents(*result.getOwner(), result.getResultNumber())
               : FailureOr<SmallVector<std::string>>(failure());
    if (tensor.getRank() != 1 || failed(extents) || extents->size() != 1)
      return operation.emitOpError(
          "TileLang in-bounds assumption has no ranked index schema");
    if (extents->front() != "1")
      return success();
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
  line("T.assume(" + expression + " < " + bound + ")");
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() == 0)
    return operation.emitOpError(
        "TileLang parallel ownership requires region arguments");
  Block &body = operation.getRegion(0).front();
  SmallVector<plan::AxisOp> axes;
  for (BlockArgument argument : body.getArguments()) {
    FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
    if (failed(axis))
      return failure();
    axes.push_back(*axis);
  }

  if (!planIndex.stages.empty()) {
    if (axes.size() != 1)
      return operation.emitOpError(
          "staged TileLang ownership requires one logical axis");
    BlockArgument argument = body.getArgument(0);
    plan::AxisOp axis = axes.front();
    bool outer = llvm::any_of(planIndex.stages, [&](plan::StageOp stage) {
      unsigned position = stage.getOrdinal();
      auto runtime = stageRaggedRuntime.find(position);
      return runtime != stageRaggedRuntime.end() &&
             raggedRuntimes[runtime->second].binding.getOuterNode() ==
                 axis.getNode();
    });
    bool member = llvm::any_of(planIndex.stages, [&](plan::StageOp stage) {
      auto stageAxes = planIndex.stageAxes.find(stage.getNode());
      plan::StageAxisOp binding =
          stageAxes == planIndex.stageAxes.end()
              ? plan::StageAxisOp()
              : stageAxes->second.lookup("member");
      return binding && binding.getAxisNodeAttr() &&
             binding.getAxisNodeAttr().getInt() == axis.getNode();
    });
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
      plan::AxisOp memberAxis = planIndex.axes.lookup(memberNode);
      const target::emission::RangeBinding *ownership =
          memberAxis ? memberAxis.getRange("ownership", 0) : nullptr;
      if (!ownership)
        return operation.emitOpError(
            "ordered ragged axis has no ownership range");
      line("query_start_" + query + " = sequence_begin_" + suffix + " + " +
           addressIndex(block) + " * " +
           ownership->getTile().str());
      axisIndices[memberNode] = "query_start_" + query;
    }
  }
  for (const auto &entry : planIndex.axesByRole) {
    plan::AxisOp lane = entry.getValue();
    if (entry.getKey().starts_with("lane_") &&
        axisIndices.lookup(lane.getNode()).empty())
      axisIndices[lane.getNode()] = "0";
  }
  for (auto [argument, axis] : llvm::zip(body.getArguments(), axes)) {
    std::string value =
        planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode())
            ? axisIndices.lookup(axis.getNode())
            : programBlocks.lookup(axis.getNode());
    if (value.empty())
      return axis.emitOpError("has no emitted per-axis program index");
    if (target::emission::isPackedScalarAxis(axis)) {
      const target::emission::RangeBinding *ownership =
          axis.getRange("ownership");
      if (!ownership)
        return axis.emitOpError("has no packed-lane ownership range");
      std::string lane =
          "packed_lane_" + std::to_string(axis.getNode());
      line("for " + lane + " in T.Parallel(" + ownership->getTile().str() +
           "):");
      ++indentation;
      value = value + " * " + ownership->getTile().str() + " + " + lane;
      axisIndices[axis.getNode()] = value;
    }
    valueNames[argument] = value;
  }
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &operation) {
  if (operation.getNumRegions() == 1 &&
      llvm::hasSingleElement(operation.getRegion(0))) {
    for (BlockArgument argument : operation.getRegion(0).front().getArguments()) {
      FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
      if (succeeded(axis) && target::emission::isPackedScalarAxis(*axis))
        --indentation;
    }
  }
  if (&operation == programRoot && planIndex.program.getPersistent())
    indentation -= 2;
  return success();
}

LogicalResult SourceEmitter::enterFor(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "for emission");
  FailureOr<SmallVector<Operation *>> domains =
      operation.getNumOperands() > 0
          ? target::expandDomainSource(operation.getOperand(0), operation)
          : FailureOr<SmallVector<Operation *>>(failure());
  bool ordered = operation.getName().getStringRef() == "intent.ordered";
  if (failed(node) || failed(domains) || domains->empty() ||
      (!ordered && domains->size() != 1) || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical TileLang sequential loop");
  Block &body = operation.getRegion(0).front();
  if (body.getNumArguments() != domains->size() + operation.getNumResults())
    return operation.emitOpError("does not match its TileLang sequential axes");
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
    valueNames[body.getArgument(domains->size() + index)] = carrier + "[0]";
  }
  loopCarriers[&operation] = carriers;
  for (auto [index, domain] : llvm::enumerate(*domains)) {
    if (!domain)
      return operation.emitOpError("has a non-canonical TileLang loop axis");
    FailureOr<int64_t> domainNode =
        target::getNodeID(*domain, "ordered traversal range");
    bool raggedAxis =
        domain->getName().getStringRef() == "intent.ragged_member";
    auto rangeValue = [&](unsigned operand) -> FailureOr<std::string> {
      Operation *definition = domain->getOperand(operand).getDefiningOp();
      auto literal = definition
                         ? definition->getAttrOfType<IntegerAttr>("intent.value")
                         : IntegerAttr();
      if (definition &&
          definition->getName().getStringRef() == "intent.constant" && literal)
        return std::to_string(literal.getInt());
      auto found = valueNames.find(domain->getOperand(operand));
      if (found == valueNames.end()) {
        operation.emitOpError("has no emitted TileLang loop bound");
        return failure();
      }
      return found->second;
    };
    FailureOr<std::string> start = failure();
    FailureOr<std::string> stop = failure();
    FailureOr<std::string> step = std::string("1");
    if (domain->getName().getStringRef() == "intent.domain" &&
        domain->getNumOperands() >= 2) {
      start = rangeValue(0);
      stop = rangeValue(1);
      if (domain->getNumOperands() == 3)
        step = rangeValue(2);
    } else if (ordered && raggedAxis && succeeded(domainNode)) {
      FailureOr<plan::RaggedOp> relation =
          target::emission::uniqueRaggedRelation(planIndex, *domainNode,
                                                  operation);
      std::string outer = succeeded(relation)
                              ? axisIndices.lookup(relation->getOuterNode())
                              : std::string();
      auto runtime = succeeded(relation)
                         ? raggedRuntimeByRelation.find(relation->getNode())
                         : raggedRuntimeByRelation.end();
      if (failed(relation) || outer.empty() ||
          runtime == raggedRuntimeByRelation.end())
        return operation.emitOpError(
            "ordered ragged traversal has no outer-axis or offsets binding");
      RaggedRuntime &ragged = raggedRuntimes[runtime->second];
      std::string suffix = std::to_string(*domainNode);
      std::string begin = "sequence_begin_" + suffix;
      std::string end = "sequence_end_" + suffix;
      line(begin + " = " + ragged.offsets->argument->name + "[" +
           addressIndex(outer) + "]");
      line(end + " = " + ragged.offsets->argument->name + "[" +
           addressIndex(outer + " + 1") + "]");
      start = begin;
      stop = end;
    } else {
      return operation.emitOpError("has a non-canonical TileLang loop axis");
    }
    if (failed(start) || failed(stop) || failed(step))
      return failure();
    if (*step != "1")
      return operation.emitOpError(
          "TileLang serial traversal requires a unit-step domain");
    if (ordered && !raggedAxis && *start != "0")
      return operation.emitOpError(
          "TileLang ordered traversal requires a zero-based domain");
    std::string iterator = makeRegionArgumentName(operation, index);
    valueNames[body.getArgument(index)] = iterator;
    if (raggedAxis && succeeded(domainNode))
      axisIndices[*domainNode] = iterator;
    plan::AxisOp axis = succeeded(domainNode)
                            ? planIndex.axes.lookup(*domainNode)
                            : plan::AxisOp();
    const target::emission::RangeBinding *outer =
        ordered && axis ? axis.getRange("traversal", 0) : nullptr;
    const target::emission::RangeBinding *inner =
        ordered && axis ? axis.getRange("traversal", 1) : nullptr;
    if (inner) {
      if (!outer || inner->getTileRole() != "one")
        return operation.emitOpError(
            "has an invalid two-level TileLang ordered traversal");
      std::string chunk = iterator + "_chunk";
      line("for " + chunk + " in T.serial(T.ceildiv(" + *stop + " - " +
           *start + ", " +
           outer->getTile().str() + ")):");
      ++indentation;
      line("for " + iterator + " in T.serial(" + *start + " + " + chunk +
           " * " + outer->getTile().str() + ", T.min(" + *start + " + (" +
           chunk + " + 1) * " + outer->getTile().str() + ", " + *stop +
           ")):");
      ++indentation;
    } else {
      line("for " + iterator + " in T.serial(" + *start + ", " + *stop +
           "):");
      ++indentation;
    }
  }
  return success();
}

LogicalResult SourceEmitter::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active TileLang sequential loop");
  FailureOr<SmallVector<Operation *>> domains =
      target::expandDomainSource(operation.getOperand(0), operation);
  if (failed(domains) || domains->empty())
    return failure();
  unsigned depth = 0;
  bool ordered = operation.getName().getStringRef() == "intent.ordered";
  for (Operation *domain : *domains) {
    FailureOr<int64_t> domainNode =
        target::getNodeID(*domain, "ordered traversal range");
    plan::AxisOp axis = succeeded(domainNode)
                            ? planIndex.axes.lookup(*domainNode)
                            : plan::AxisOp();
    depth += ordered && axis && axis.getRange("traversal", 1) ? 2 : 1;
  }
  indentation -= depth;
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

LogicalResult SourceEmitter::enterWhile(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "while emission");
  if (failed(node) || operation.getNumRegions() != 2 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      !llvm::hasSingleElement(operation.getRegion(1)) ||
      operation.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("lacks a mechanical TileLang scalar while");
  Block &before = operation.getRegion(0).front();
  Block &after = operation.getRegion(1).front();
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index);
    std::string dtype = dtypeName(operation.getResult(index).getType(), operation);
    if (failed(initial) || dtype.empty())
      return failure();
    std::string carrier =
        uniqueName("while_state_" + std::to_string(index), *node);
    line(carrier + " = T.alloc_local((1,), " + dtype + ")");
    line(carrier + "[0] = " + initial->str());
    carriers.push_back(carrier);
    valueNames[before.getArgument(index)] = carrier + "[0]";
    valueNames[after.getArgument(index)] = carrier + "[0]";
  }
  whileCarriers[&operation] = std::move(carriers);
  return success();
}

LogicalResult SourceEmitter::emitCondition(Operation &operation) {
  Operation *owner = target::whileConditionOwner(operation);
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (!owner || failed(condition) || operation.getNumOperands() !=
                                          owner->getNumResults() + 1)
    return operation.emitOpError("does not match a TileLang scalar while");
  line("while " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveWhile(Operation &operation) {
  auto carriers = whileCarriers.find(&operation);
  if (carriers == whileCarriers.end())
    return operation.emitOpError("has no active TileLang scalar while");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index] + "[0]");
  return success();
}

LogicalResult SourceEmitter::emitYield(Operation &operation) {
  Operation *owner = operation.getParentOp();
  if (!owner)
    return operation.emitOpError("has no structured-control owner");
  StringRef name = owner->getName().getStringRef();
  if (name == "intent.for" || name == "intent.ordered") {
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
  if (name == "intent.while") {
    auto carriers = whileCarriers.find(owner);
    if (carriers == whileCarriers.end() ||
        carriers->second.size() != operation.getNumOperands())
      return operation.emitOpError("does not match its TileLang while state");
    for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
      FailureOr<StringRef> yielded = lookupValue(operation, index);
      if (failed(yielded))
        return failure();
      std::string destination = carriers->second[index] + "[0]";
      if (*yielded != destination)
        line(destination + " = " + yielded->str());
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
  if (binding && binding.getSpace() == "private_workspace") {
    if (failed(info) || info->shape.empty() ||
        !workspaceNames.count(operation.getResult(0)))
      return operation.emitOpError(
          "lacks a planned TileLang private workspace parameter");
    return success();
  }
  if (failed(node) || !binding || failed(info) || failed(initializer))
    return operation.emitOpError(
        "lacks a TileLang logical-buffer binding");
  std::string dtype = dtypeName(info->elementType, operation);
  if (dtype.empty())
    return failure();
  std::string base = makeResultName(operation, 0);
  if (binding.getSpace() == "private_vector") {
    FailureOr<int64_t> elements =
        target::logicalBufferElementCount(*info, operation);
    if (failed(elements))
      return failure();
    line(base + " = T.alloc_local((" + std::to_string(*elements) +
         ",), " + dtype + ")");
    line("T.fill(" + base + ", " + initializer->str() + ")");
    localBuffers[operation.getResult(0)] = base;
    return success();
  }
  if (binding.getSpace() != "private_scalar_array")
    return operation.emitOpError("has an unsupported TileLang buffer residency");
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
  auto workspace = operation.getNumOperands() > 0
                       ? workspaceNames.find(operation.getOperand(0))
                       : workspaceNames.end();
  if (workspace != workspaceNames.end()) {
    FailureOr<std::string> index = privateWorkspaceIndex(operation);
    if (failed(index) || operation.getNumResults() != 1)
      return operation.emitOpError(
          "lacks a mechanical TileLang private-workspace load");
    std::string result = makeResultName(operation, 0);
    line(result + " = " + workspace->second + "[" + *index + "]");
    bindResult(operation, 0, result);
    return success();
  }
  auto local = operation.getNumOperands() > 0
                   ? localBuffers.find(operation.getOperand(0))
                   : localBuffers.end();
  Operation *localOwner = operation.getNumOperands() > 0
                              ? operation.getOperand(0).getDefiningOp()
                              : nullptr;
  FailureOr<target::LogicalBufferInfo> localInfo =
      localOwner ? target::getLogicalBufferInfo(*localOwner)
                 : FailureOr<target::LogicalBufferInfo>(failure());
  FailureOr<SmallVector<target::LogicalBufferIndex>> localIndices =
      succeeded(localInfo)
          ? target::getLogicalBufferIndices(operation, localInfo->shape.size())
          : FailureOr<SmallVector<target::LogicalBufferIndex>>(failure());
  FailureOr<target::LogicalBufferIndex> index =
      target::getLogicalBufferIndex(operation);
  if (local != localBuffers.end()) {
    if (failed(localInfo) || failed(localIndices) || operation.getNumResults() != 1)
      return operation.emitOpError("lacks an addressable TileLang buffer load");
    std::string subscript;
    for (auto [axis, logical] : llvm::enumerate(*localIndices)) {
      std::string component;
      if (logical.constant)
        component = std::to_string(*logical.constant);
      else {
        FailureOr<StringRef> dynamic = lookupValue(operation, *logical.operand);
        if (failed(dynamic))
          return failure();
        component = dynamic->str();
      }
      subscript = subscript.empty()
                      ? component
                      : "(" + subscript + ") * " +
                            std::to_string(localInfo->shape[axis]) + " + (" +
                            component + ")";
    }
    std::string result = makeResultName(operation, 0);
    line(result + " = " + local->second + "[" + subscript + "]");
    bindResult(operation, 0, result);
    return success();
  }
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
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
  auto workspace = operation.getNumOperands() > 0
                       ? workspaceNames.find(operation.getOperand(0))
                       : workspaceNames.end();
  if (workspace != workspaceNames.end()) {
    FailureOr<std::string> index = privateWorkspaceIndex(operation);
    FailureOr<StringRef> stored = lookupValue(operation, 1);
    if (failed(index) || failed(stored))
      return operation.emitOpError(
          "lacks a mechanical TileLang private-workspace store");
    line(workspace->second + "[" + *index + "] = " + stored->str());
    return success();
  }
  auto local = operation.getNumOperands() > 0
                   ? localBuffers.find(operation.getOperand(0))
                   : localBuffers.end();
  Operation *localOwner = operation.getNumOperands() > 0
                              ? operation.getOperand(0).getDefiningOp()
                              : nullptr;
  FailureOr<target::LogicalBufferInfo> localInfo =
      localOwner ? target::getLogicalBufferInfo(*localOwner)
                 : FailureOr<target::LogicalBufferInfo>(failure());
  FailureOr<SmallVector<target::LogicalBufferIndex>> localIndices =
      succeeded(localInfo)
          ? target::getLogicalBufferIndices(operation, localInfo->shape.size())
          : FailureOr<SmallVector<target::LogicalBufferIndex>>(failure());
  FailureOr<target::LogicalBufferIndex> index =
      target::getLogicalBufferIndex(operation);
  FailureOr<StringRef> stored = lookupValue(operation, 1);
  if (local != localBuffers.end()) {
    if (failed(localInfo) || failed(localIndices) || failed(stored))
      return operation.emitOpError("lacks an addressable TileLang buffer store");
    std::string subscript;
    for (auto [axis, logical] : llvm::enumerate(*localIndices)) {
      std::string component;
      if (logical.constant)
        component = std::to_string(*logical.constant);
      else {
        FailureOr<StringRef> dynamic = lookupValue(operation, *logical.operand);
        if (failed(dynamic))
          return failure();
        component = dynamic->str();
      }
      subscript = subscript.empty()
                      ? component
                      : "(" + subscript + ") * " +
                            std::to_string(localInfo->shape[axis]) + " + (" +
                            component + ")";
    }
    line(local->second + "[" + subscript + "] = " + stored->str());
    return success();
  }
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
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

FailureOr<std::string>
SourceEmitter::privateWorkspaceIndex(Operation &operation) {
  Operation *buffer = operation.getNumOperands() > 0
                          ? operation.getOperand(0).getDefiningOp()
                          : nullptr;
  FailureOr<int64_t> node =
      buffer ? target::getNodeID(*buffer, "private-workspace access")
             : FailureOr<int64_t>(failure());
  plan::BufferOp binding =
      succeeded(node) ? planIndex.buffers.lookup(*node) : plan::BufferOp();
  FailureOr<target::LogicalBufferInfo> info =
      buffer ? target::getLogicalBufferInfo(*buffer)
             : FailureOr<target::LogicalBufferInfo>(failure());
  FailureOr<SmallVector<target::LogicalBufferIndex>> indices =
      succeeded(info)
          ? target::getLogicalBufferIndices(operation, info->shape.size())
          : FailureOr<SmallVector<target::LogicalBufferIndex>>(failure());
  if (failed(node) || !binding || binding.getSpace() != "private_workspace" ||
      failed(info) || failed(indices))
    return operation.emitOpError(
        "does not resolve a planned TileLang private workspace");

  auto spellIndex = [&](const target::LogicalBufferIndex &index)
      -> FailureOr<std::string> {
    if (index.constant)
      return std::to_string(*index.constant);
    FailureOr<StringRef> dynamic = lookupValue(operation, *index.operand);
    if (failed(dynamic))
      return failure();
    return dynamic->str();
  };
  FailureOr<std::string> offset =
      target::emission::projectPrivateWorkspaceOffset(
          binding, *info, *indices, planIndex, axisIndices, axisDimensions,
          spellIndex, operation);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a TileLang load binding");
  FailureOr<std::string> physicalFill =
      transferPhysicalExtentFill(operation);
  FailureOr<bool> wholeView = target::isWholeViewAccess(operation);
  if (failed(physicalFill) || failed(wholeView))
    return failure();
  if (*wholeView && boundary.getDomainNodes().empty() &&
      boundary.getPadding() == "none" && physicalFill->empty()) {
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
  bool scalarResult = operation.getNumResults() == 1 &&
                      !isa<RankedTensorType>(operation.getResult(0).getType());
  Type resultElementType = scalarResult
                               ? operation.getResult(0).getType()
                               : cast<RankedTensorType>(
                                     operation.getResult(0).getType())
                                     .getElementType();
  FailureOr<bool> derivedScalar = target::hasDerivedScalarIndex(operation);
  if (failed(derivedScalar))
    return failure();
  StringRef zeroFill = isa<IntegerType, IndexType>(resultElementType) ? "0"
                                                                      : "0.0";
  if (failed(physicalFill))
    return failure();
  auto accessRanges =
      target::emission::accessRangesForTransfer(planIndex, *node);
  if (accessRanges.size() > 1)
    return operation.emitOpError(
        "TileLang cannot project a multi-axis checked access footprint as one "
        "cooperative transfer");
  bool expanded = !physicalFill->empty();
  bool tensorIndirect = boundary.hasDataDependentTensorIndex();
  bool guardedF16Bulk = !scalarResult && resultElementType.isF16() &&
                        *derivedScalar && !tensorIndirect &&
                        boundary.getCheckBounds() &&
                        boundary.getPadding() != "none";
  FailureOr<SmallVector<target::IndexTerm>> relation = failure();
  if (guardedF16Bulk) {
    relation = target::parseIndexRelation(operation);
    if (failed(relation))
      return failure();
    for (const target::IndexTerm &term : *relation) {
      if (term.kind != "region_index")
        continue;
      if (term.operands.size() != 1 || !term.operands.front())
        return operation.emitOpError(
            "has no mechanical guarded float16 region relation");
      Value indexed = operation.getOperand(*term.operands.front());
      FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
      if (failed(axis))
        return failure();
      if (!axis->hasRole("lane") || axis->hasRole("parallel") ||
          axis->hasRole("ordered") || axis->hasRole("reduction") ||
          axis->hasRole("ragged_member")) {
        guardedF16Bulk = false;
        break;
      }
    }
  }
  if (guardedF16Bulk) {
    if (expanded) {
      if (failed(view) || failed(relation) ||
          relation->size() != (*view)->shape.size())
        return operation.emitOpError(
            "cannot resolve guarded float16 bulk extents");
      for (auto [axis, term] : llvm::enumerate(*relation)) {
        auto extent = planIndex.blockExtents.find((*view)->shape[axis]);
        bool vectorAccess = term.kind == "full_slice" ||
                            term.kind == "region_index";
        if (!vectorAccess || extent == planIndex.blockExtents.end())
          continue;
        if (extent->second.getRounding() != "power_of_two")
          return extent->second.emitOpError(
              "has no exact-extent TileLang bulk capability check");
        exactBulkExtents.insert((*view)->shape[axis]);
      }
    }
    FailureOr<std::string> result =
        allocateResult(operation, 0, boundary.getResultSpace());
    FailureOr<SmallVector<std::string>> extents =
        tensorExtents(operation, 0);
    FailureOr<std::string> indices = accessIndices(operation);
    FailureOr<std::string> predicate =
        succeeded(extents)
            ? wholeTileBoundsPredicate(operation, *extents)
            : FailureOr<std::string>(failure());
    if (failed(view) || failed(result) || failed(extents) ||
        failed(indices) || failed(predicate) || predicate->empty())
      return operation.emitOpError(
          "has no guarded TileLang float16 bulk-transfer projection");
    line("if " + *predicate + ":");
    ++indentation;
    line("T.copy(" + (*view)->argument->name + "[" + *indices + "], " +
         *result + ")");
    --indentation;
    line("else:");
    ++indentation;
    if (boundary.getPadding() == "negative_infinity")
      line("T.fill(" + *result + ", -T.infinity(T.float32))");
    else
      line("T.clear(" + *result + ")");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }
  if (scalarResult) {
    FailureOr<std::string> indices = accessIndices(operation);
    if (failed(view) || failed(indices))
      return operation.emitOpError(
          "lacks a mechanical TileLang scalar load binding");
    std::string expression =
        (*view)->argument->name + "[" + *indices + "]";
    bool packedScalar =
        target::emission::hasPackedScalarDomain(planIndex, boundary);
    if (boundary.getCheckBounds() &&
        (boundary.getPadding() != "none" || packedScalar)) {
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
    bool materializeLogicalBounds =
        boundary.getCheckBounds() && padding != "none";
    bool stagePhysicalPadding =
        expanded && tensorIndirect;
    FailureOr<SmallVector<std::string>> extents =
        tensorExtents(operation, 0, !stagePhysicalPadding);
    if (failed(extents) || extents->empty() ||
        (padding != "none" && padding != "zero" &&
         padding != "negative_infinity"))
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
        materializeLogicalBounds
            ? elementBoundsPredicate(operation, tileIndices, false, false)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> physicalPredicate =
        expanded && !stagePhysicalPadding
            ? elementBoundsPredicate(operation, tileIndices, true)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> wholeTile =
        materializeLogicalBounds && !tensorIndirect
            ? wholeTileBoundsPredicate(operation, *extents)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> bulkIndices =
        materializeLogicalBounds && !tensorIndirect
            ? accessIndices(operation)
            : FailureOr<std::string>(std::string());
    if (failed(logicalPredicate) || failed(physicalPredicate) ||
        failed(wholeTile) || failed(bulkIndices))
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
      unsigned loopDepth = 1;
      if (materializeLogicalBounds) {
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
      if (materializeLogicalBounds) {
        --indentation;
        line("else:");
        ++indentation;
        line(target + "] = " + fill.str());
        --indentation;
      }
      indentation -= loopDepth;
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
  bool logicalReduction = binding &&
                          (binding.getLowering() == "T.reduce_any_i32" ||
                           binding.getLowering() == "T.reduce_all_i32");
  unsigned expectedResults = argReduction ? 2 : 1;
  if (failed(node) || !binding || failed(operand) ||
      operation.getNumResults() != expectedResults)
    return operation.emitOpError("lacks a TileLang reduction binding");
  if (logicalReduction) {
    auto inputType = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
    if (!inputType || inputType.getRank() != 1 || binding.getAxis() != 0 ||
        isa<RankedTensorType>(operation.getResult(0).getType()))
      return operation.emitOpError(
          "TileLang logical reduction requires a one-dimensional input "
          "reduced to a scalar");
    auto inputResult = dyn_cast<OpResult>(operation.getOperand(0));
    FailureOr<SmallVector<std::string>> extents =
        inputResult
            ? tensorExtents(*inputResult.getOwner(), inputResult.getResultNumber())
            : FailureOr<SmallVector<std::string>>(failure());
    if (failed(extents) || extents->size() != 1)
      return operation.emitOpError("has no logical reduction fragment extent");
    std::string inputBuffer = makeResultName(operation, 0) + "_logical_input";
    std::string reduced = makeResultName(operation, 0) + "_logical_result";
    line(inputBuffer + " = T.alloc_fragment((" + extents->front() +
         ",), T.int32)");
    line("for logical_i in T.Parallel(" + extents->front() + "):");
    ++indentation;
    line(inputBuffer + "[logical_i] = T.cast(" + operand->str() +
         "[logical_i], T.int32)");
    --indentation;
    line(reduced + " = T.alloc_fragment((1,), T.int32)");
    StringRef primitive = binding.getLowering() == "T.reduce_any_i32"
                              ? StringRef("T.reduce_max")
                              : StringRef("T.reduce_min");
    line(primitive.str() + "(" + inputBuffer + ", " + reduced +
         ", dim=0, clear=True)");
    valueNames[operation.getResult(0)] = reduced + "[0] != 0";
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
  if (binding && binding.getResultSpace() == "global") {
    plan::AxisOp axis = planIndex.axes.lookup(binding.getAxisNode());
    const target::emission::RangeBinding *range =
        axis ? axis.getRange("traversal", 0) : nullptr;
    std::string extent = scanExtents.lookup(binding.getNode());
    Operation *scan = kernel.nodes.lookup(binding.getNode());
    if (failed(node) || binding.getLowering() != "T.cumsum" ||
        !range || extent.empty() || !scan ||
        scan->getNumResults() != 1)
      return operation.emitOpError(
          "lacks a workspace-backed TileLang scan binding");
    std::string result = makeResultName(operation, 0);
    std::string carry = result + "_carry";
    std::string block = result + "_block";
    std::string index = result + "_index";
    auto resultType = dyn_cast<RankedTensorType>(scan->getResult(0).getType());
    std::string carryDtype =
        resultType ? dtypeName(resultType.getElementType(), operation)
                   : std::string();
    if (!resultType || carryDtype.empty())
      return operation.emitOpError("has no TileLang scan carry dtype");
    line(carry + " = T.alloc_local((1,), " + carryDtype + ")");
    line(carry + "[0] = 0");
    line("for " + block + " in T.serial(T.ceildiv(" + extent + ", " +
         range->getTile().str() + ")):");
    ++indentation;
    line("for " + index + " in T.serial(" + range->getTile().str() + "):");
    ++indentation;
    std::string logical = block + " * " + range->getTile().str() + " + " + index;
    FailureOr<std::string> workspace =
        scanWorkspaceIndex(binding, logical, operation);
    if (failed(workspace))
      return failure();
    if (failed(replayScanProducers(binding, logical)))
      return failure();
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    if (failed(operand))
      return failure();
    for (int64_t valueID : binding.getMaterializedValues()) {
      Value value = kernel.values.lookup(valueID);
      auto name = valueNames.find(value);
      FailureOr<std::string> materialized =
          scanMaterializedIndex(value, logical, operation);
      if (name == valueNames.end() || failed(materialized))
        return binding.emitOpError(
            "has no emitted TileLang value for scan materialization");
      line("if " + logical + " < " + extent + ":");
      ++indentation;
      line(workspaceNames.lookup(value) + "[" + *materialized + "] = " +
           name->second + "[0]");
      --indentation;
    }
    line(carry + "[0] = " + carry + "[0] + T.if_then_else(" + logical +
         " < " + extent + ", " + operand->str() + "[0], 0)");
    line("if " + logical + " < " + extent + ":");
    ++indentation;
    line(workspaceNames.lookup(scan->getResult(0)) + "[" + *workspace + "] = " +
         carry + "[0]");
    --indentation;
    indentation -= 2;
    valueNames.erase(operation.getResult(0));
    return success();
  }
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

LogicalResult SourceEmitter::replayScanProducers(const plan::ScanOp &binding,
                                                 StringRef logicalIndex) {
  if (!operationRegistry())
    return binding.emitOpError(
        "has no TileLang operation registry for scan replay");
  auto oldAxis = axisIndices.find(binding.getAxisNode());
  std::optional<std::string> savedAxis =
      oldAxis == axisIndices.end() ? std::nullopt
                                   : std::optional<std::string>(oldAxis->second);
  axisIndices[binding.getAxisNode()] = logicalIndex.str();
  llvm::StringMap<std::optional<std::string>> savedTiles;
  for (int64_t node : binding.getProducers()) {
    Operation *producer = kernel.nodes.lookup(node);
    if (!producer)
      return binding.emitOpError(
          "references an unknown TileLang scan producer");
    auto shapes = producer->getAttrOfType<ArrayAttr>("intent.result_shapes");
    if (!shapes)
      continue;
    for (Attribute shapeAttr : shapes) {
      auto shape = dyn_cast<ArrayAttr>(shapeAttr);
      if (!shape)
        continue;
      for (Attribute labelAttr : shape) {
        auto label = dyn_cast<StringAttr>(labelAttr);
        if (!label || !label.getValue().starts_with("?region_") ||
            savedTiles.count(label.getValue()))
          continue;
        auto found = regionTiles.find(label.getValue());
        savedTiles[label.getValue()] =
            found == regionTiles.end()
                ? std::nullopt
                : std::optional<std::string>(found->getValue());
        regionTiles[label.getValue()] = "1";
      }
    }
  }
  auto restore = [&]() {
    activeScanReplay = -1;
    if (savedAxis)
      axisIndices[binding.getAxisNode()] = *savedAxis;
    else
      axisIndices.erase(binding.getAxisNode());
    for (const auto &entry : savedTiles) {
      if (entry.getValue())
        regionTiles[entry.getKey()] = *entry.getValue();
      else
        regionTiles.erase(entry.getKey());
    }
  };
  activeScanReplay = binding.getNode();
  for (int64_t node : binding.getProducers()) {
    Operation *producer = kernel.nodes.lookup(node);
    if (!producer || failed(operationRegistry()->dispatch(*producer, stage()))) {
      restore();
      return failure();
    }
  }
  restore();
  return success();
}

FailureOr<std::string>
SourceEmitter::scanWorkspaceIndex(const plan::ScanOp &binding,
                                  StringRef logicalIndex,
                                  Operation &consumer) {
  FailureOr<std::string> offset = target::emission::projectScanWorkspaceOffset(
      binding, scanExtents.lookup(binding.getNode()), logicalIndex, planIndex,
      axisIndices, axisDimensions, consumer);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

FailureOr<std::string>
SourceEmitter::scanMaterializedIndex(Value value, StringRef logicalIndex,
                                     Operation &consumer) {
  auto binding = scanMaterializedValues.find(value);
  if (binding == scanMaterializedValues.end())
    return consumer.emitOpError(
        "has no TileLang scan materialization binding");
  return scanWorkspaceIndex(binding->second, logicalIndex, consumer);
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
        : binding.getLowering() == "python_not"
            ? "(" + operand->str() + ") == False"
            : binding.getLowering().str() + "(" + operand->str() + ")";
    if (target::whileConditionOwner(operation)) {
      bindResult(operation, 0, expression);
      return success();
    }
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
                           : binding.getLowering() == "python_not"
                               ? "(" + *operand + ") == False"
                               : binding.getLowering().str() + "(" + *operand + ")";
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
      if (binding.getNonnegativeOperands()) {
        StringRef symbol =
            binding.getLowering() == "python_floor_divide" ? "//" : "%";
        return "(" + lhs.str() + ") " + symbol.str() + " (" + rhs.str() + ")";
      }
      Type elementType = operation.getResult(0).getType();
      if (auto tensor = dyn_cast<RankedTensorType>(elementType))
        elementType = tensor.getElementType();
      std::string resultDtype = dtypeName(elementType, operation);
      if (resultDtype.empty())
        return failure();
      std::string wideLhs = resultName + "_wide_lhs";
      std::string wideRhs = resultName + "_wide_rhs";
      std::string lhsMagnitude = resultName + "_lhs_magnitude";
      std::string rhsMagnitude = resultName + "_rhs_magnitude";
      std::string quotientMagnitude = resultName + "_quotient_magnitude";
      std::string quotient = resultName + "_truncating_quotient";
      std::string remainder = resultName + "_remainder";
      std::string adjust = resultName + "_adjust";
      line(wideLhs + " = T.cast(" + lhs.str() + ", T.int64)");
      line(wideRhs + " = T.cast(" + rhs.str() + ", T.int64)");
      line(lhsMagnitude + " = T.if_then_else(" + wideLhs + " < 0, -" +
           wideLhs + ", " + wideLhs + ")");
      line(rhsMagnitude + " = T.if_then_else(" + wideRhs + " < 0, -" +
           wideRhs + ", " + wideRhs + ")");
      line(quotientMagnitude + " = " + lhsMagnitude + " // " + rhsMagnitude);
      line(quotient + " = T.if_then_else((" + wideLhs + " < 0) != (" +
           wideRhs + " < 0), -" + quotientMagnitude + ", " +
           quotientMagnitude + ")");
      line(remainder + " = " + wideLhs + " - " + quotient + " * " + wideRhs);
      line(adjust + " = (" + remainder + " != 0) & ((" + remainder +
           " < 0) != (" + wideRhs + " < 0))");
      return binding.getLowering() == "python_floor_divide"
                 ? "T.cast(" + quotient + " - T.if_then_else(" + adjust +
                       ", 1, 0), " + resultDtype + ")"
                 : "T.cast(" + remainder + " + T.if_then_else(" + adjust +
                       ", " + wideRhs + ", 0), " + resultDtype + ")";
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
    else if (binding.getLowering() == "python_logical_and")
      symbol = "&";
    else if (binding.getLowering() == "python_logical_or")
      symbol = "|";
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
    if (target::whileConditionOwner(operation)) {
      bindResult(operation, 0, *expression);
      return success();
    }
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
    std::string expression = "T.if_then_else(" + condition->str() + ", " +
                             trueValue->str() + ", " + falseValue->str() + ")";
    if (target::whileConditionOwner(operation)) {
      bindResult(operation, 0, expression);
      return success();
    }
    line(result + " = " + expression);
    valueNames[operation.getResult(0)] = result;
    return success();
  }

  FailureOr<std::string> allocated = allocateResult(operation, 0, "fragment");
  if (failed(allocated))
    return failure();
  std::string result = *allocated;
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
    std::string expression =
        "T.cast(" + operand->str() + ", " + dtype + ")";
    if (target::whileConditionOwner(operation)) {
      bindResult(operation, 0, expression);
      return success();
    }
    line(result + " = " + expression);
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
  auto resultTensor = operation.getNumResults() == 1
                          ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                          : RankedTensorType();
  SmallVector<std::string> extents;
  auto sourceTensor = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  bool introducesUnitAxis = sourceTensor && resultTensor &&
                            sourceTensor.getRank() + 1 == resultTensor.getRank() &&
                            llvm::count(resultTensor.getShape(), 1) == 1;
  if (introducesUnitAxis && binding &&
      binding.getAxisNodes().size() == static_cast<size_t>(resultTensor.getRank())) {
    for (int64_t axisNode : binding.getAxisNodes()) {
      if (axisNode < 0) {
        extents.push_back("1");
        continue;
      }
      plan::AxisOp axis = planIndex.axes.lookup(axisNode);
      const target::emission::RangeBinding *range =
          axis ? axis.getRange("reduction", 0) : nullptr;
      if (!range) {
        extents.clear();
        break;
      }
      extents.push_back(range->getTile().str());
    }
  }
  FailureOr<std::string> shape = failure();
  if (!extents.empty()) {
    std::string spelling = "(";
    for (auto [index, extent] : llvm::enumerate(extents)) {
      if (index)
        spelling += ", ";
      spelling += extent;
    }
    if (extents.size() == 1)
      spelling += ",";
    shape = spelling + ")";
  } else {
    shape = tensorShape(operation, 0);
  }
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
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> result = allocateResult(operation, 0, "shared");
  if (failed(node) || !binding ||
      binding.getLowering() != "T.transpose" ||
      binding.getReuseOperandAttr().getInt() != -1 ||
      binding.getSpace() != "fragment" || failed(permutation) ||
      failed(operand) || failed(result))
    return operation.emitOpError(
        "lacks a mechanical TileLang transpose binding");
  if (permutation->size() != 2 || (*permutation)[0] != 1 ||
      (*permutation)[1] != 0)
    return operation.emitOpError(
        "TileLang transpose only supports swapping the final two axes");
  line("T.transpose(" + operand->str() + ", " + *result + ")");
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
  bool scalarFragmentGather =
      binding.getLowering() == "T.indirect_gather" &&
      isa<RankedTensorType>(operation.getOperand(0).getType()) &&
      !isa<RankedTensorType>(operation.getResult(0).getType()) &&
      relation->size() == 1;
  if (scalarFragmentGather) {
    auto scanSource = scanResults.find(operation.getOperand(0));
    auto materializedSource =
        scanMaterializedValues.find(operation.getOperand(0));
    FailureOr<StringRef> source =
        scanSource == scanResults.end() &&
                materializedSource == scanMaterializedValues.end()
            ? lookupValue(operation, 0)
            : FailureOr<StringRef>(StringRef());
    FailureOr<StringRef> valid = lookupValue(operation, validIndex.getInt());
    FailureOr<StringRef> fill = lookupValue(operation, fillIndex.getInt());
    const target::IndexTerm &term = relation->front();
    std::string index;
    if (term.kind == "static_index" && term.staticValues.size() == 1 &&
        term.staticValues.front()) {
      index = std::to_string(*term.staticValues.front());
    } else if ((term.kind == "value_index" || term.kind == "region_index") &&
               term.operands.size() == 1 && term.operands.front()) {
      FailureOr<StringRef> dynamic =
          lookupValue(operation, *term.operands.front());
      if (failed(dynamic))
        return failure();
      index = dynamic->str();
    } else {
      return operation.emitOpError(
          "scalar TileLang gather has no mechanical index relation");
    }
    if ((scanSource == scanResults.end() &&
         materializedSource == scanMaterializedValues.end() && failed(source)) ||
        failed(valid) || failed(fill))
      return failure();
    std::string result = makeResultName(operation, 0);
    if (scanSource != scanResults.end()) {
      FailureOr<std::string> workspace =
          scanWorkspaceIndex(scanSource->second, index, operation);
      Operation *scan = kernel.nodes.lookup(scanSource->second.getNode());
      if (failed(workspace) || !scan)
        return failure();
      line(result + " = T.if_then_else(" + valid->str() + ", " +
           workspaceNames.lookup(scan->getResult(0)) + "[" + *workspace +
           "], " + fill->str() + ")");
    } else if (materializedSource != scanMaterializedValues.end()) {
      FailureOr<std::string> workspace = scanMaterializedIndex(
          operation.getOperand(0), index, operation);
      if (failed(workspace))
        return failure();
      line(result + " = T.if_then_else(" + valid->str() + ", " +
           workspaceNames.lookup(operation.getOperand(0)) + "[" + *workspace +
           "], " + fill->str() + ")");
    } else {
      line(result + " = T.if_then_else(" + valid->str() + ", " + source->str() +
           "[" + addressIndex(index) + "], " + fill->str() + ")");
    }
    bindResult(operation, 0, result);
    return success();
  }
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
  if (binding.getLowering() == "T.indirect_gather" &&
      isa<intent::ViewType>(operation.getOperand(0).getType())) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
    FailureOr<std::string> result = resultStorage();
    if (failed(view) || failed(extents) || extents->empty() || failed(result))
      return failure();
    SmallVector<std::string> indices;
    std::string loop = "for ";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        loop += ", ";
      indices.push_back("gather_i" + std::to_string(axis));
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
    FailureOr<std::string> sourceIndices =
        elementAccessIndices(operation, indices);
    FailureOr<std::string> bounds =
        elementBoundsPredicate(operation, indices, false, false);
    FailureOr<std::string> valid =
        tensorElement(operation.getOperand(validIndex.getInt()), indices, operation);
    FailureOr<std::string> fill =
        tensorElement(operation.getOperand(fillIndex.getInt()), indices, operation);
    if (failed(sourceIndices) || failed(bounds) || failed(valid) || failed(fill))
      return failure();
    std::string target = *result + "[";
    for (auto [axis, index] : llvm::enumerate(indices)) {
      if (axis)
        target += ", ";
      target += index;
    }
    target += "]";
    std::string predicate = *valid;
    if (*bounds != "True")
      predicate = "(" + *bounds + ") and (" + predicate + ")";
    line(target + " = T.if_then_else(" + predicate + ", " +
         (*view)->argument->name + "[" + *sourceIndices + "], " + *fill +
         ")");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }
  bool expand = binding.getLowering() == "expand_dims" &&
                llvm::any_of(*relation, [](const target::IndexTerm &term) {
                  return term.kind == "new_axis";
                }) &&
                llvm::all_of(*relation, [](const target::IndexTerm &term) {
                  return term.kind == "full_slice" || term.kind == "new_axis";
                });
  if (!expand)
    return operation.emitOpError("has no mechanical TileLang gather relation");
  FailureOr<StringRef> source = lookupValue(operation, 0);
  FailureOr<StringRef> valid = lookupValue(operation, validIndex.getInt());
  FailureOr<StringRef> fill = lookupValue(operation, fillIndex.getInt());
  FailureOr<std::string> result = resultStorage();
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(source) || failed(valid) || failed(fill) || failed(result) ||
      failed(extents) || extents->size() != relation->size())
    return failure();
  auto sourceTensor = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  Operation *validDefinition =
      operation.getOperand(validIndex.getInt()).getDefiningOp();
  Attribute validLiteral =
      validDefinition ? validDefinition->getAttr("intent.value") : Attribute();
  bool alwaysValid = false;
  if (auto boolean = dyn_cast_if_present<BoolAttr>(validLiteral))
    alwaysValid = boolean.getValue();
  else if (auto integer = dyn_cast_if_present<IntegerAttr>(validLiteral))
    alwaysValid = !integer.getValue().isZero();
  if (sourceTensor && alwaysValid) {
    std::string shape = "(";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        shape += ", ";
      shape += extent;
    }
    if (extents->size() == 1)
      shape += ",";
    shape += ")";
    line(*result + " = T.reshape(" + source->str() + ", " + shape + ")");
    bindResult(operation, 0, *result);
    return success();
  }
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back("gather_i" + std::to_string(axis));
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
  std::string target = *result + "[";
  std::string sourceElement = source->str() + "[";
  unsigned sourceAxis = 0;
  for (auto [axis, term] : llvm::enumerate(*relation)) {
    if (axis)
      target += ", ";
    target += indices[axis];
    if (term.kind == "new_axis")
      continue;
    if (sourceAxis)
      sourceElement += ", ";
    sourceElement += indices[axis];
    ++sourceAxis;
  }
  target += "]";
  sourceElement += "]";
  line(target + " = T.if_then_else(" + valid->str() + ", " + sourceElement +
       ", " + fill->str() + ")");
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
  streamOuterAxisIndices[&operation] =
      axisIndices.lookup(binding.getAxisNode());
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
  std::string streamStart = "stream_axis_index_" + std::to_string(*node);
  line(streamStart + " = " +
       std::string(raggedStream ? "sequence_begin_" + raggedSuffix + " + " : "") +
       addressIndex(streamTile) + " * " + binding.getTile().str());
  axisIndices[binding.getAxisNode()] = streamStart;
  valueNames[body.getArgument(0)] = streamStart;
  for (int64_t axisNode : binding.getInnerReductionAxes()) {
    if (axisNode == binding.getAxisNode())
      continue;
    plan::AxisOp axis = planIndex.axes.lookup(axisNode);
    const target::emission::RangeBinding *range =
        axis ? axis.getRange("reduction", 0) : nullptr;
    if (!range)
      return binding.emitOpError("has no inner reduction range");
    axisIndices[axisNode] = "0";
  }
  return success();
}

LogicalResult SourceEmitter::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  auto outerIndex = streamOuterAxisIndices.find(&operation);
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (carriers == streamCarriers.end() ||
      outerIndex == streamOuterAxisIndices.end() || failed(node) || !binding)
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
  if (outerIndex->second.empty())
    axisIndices.erase(binding.getAxisNode());
  else
    axisIndices[binding.getAxisNode()] = outerIndex->second;
  for (int64_t axisNode : binding.getInnerReductionAxes())
    if (axisNode != binding.getAxisNode())
      axisIndices.erase(axisNode);
  streamOuterAxisIndices.erase(outerIndex);
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
  if (orientation->batched)
    return operation.emitOpError(
        "TileLang 0.1.13 has no mechanical batched GEMM projection");
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
    FailureOr<int64_t> lhsNode =
        target::getNodeID(*lhsLoad, "TileLang contraction lhs transfer");
    FailureOr<int64_t> rhsNode =
        target::getNodeID(*rhsLoad, "TileLang contraction rhs transfer");
    plan::BoundaryOp lhsBoundary =
        succeeded(lhsNode) ? planIndex.boundaries.lookup(*lhsNode)
                           : plan::BoundaryOp();
    plan::BoundaryOp rhsBoundary =
        succeeded(rhsNode) ? planIndex.boundaries.lookup(*rhsNode)
                           : plan::BoundaryOp();
    if (failed(lhsNode) || failed(rhsNode) || !lhsBoundary || !rhsBoundary)
      return failure();
    if (lhsBoundary.hasDataDependentTensorIndex() ||
        rhsBoundary.hasDataDependentTensorIndex())
      return operation.emitOpError(
          "TileLang cannot bulk-copy a contraction operand with a noncontiguous index tile");
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
    return operation.emitOpError(
        "TileLang 0.1.13 has no native single-row contraction projection; "
        "the scalar product-and-reduce fallback is intentionally unsupported");
  }
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  line("T.clear(" + *result + ")");
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
  Value storedValue = operation.getOperand(valueIndex.getInt());
  if (!isa<RankedTensorType>(storedValue.getType())) {
    if (expanded || (boundary.getTransfer() != "bulk_copy" &&
                     boundary.getTransfer() != "parallel_elements"))
      return operation.emitOpError("has no scalar TileLang store transfer");
    FailureOr<std::string> indices = accessIndices(operation);
    FailureOr<std::string> predicate =
        scalarTransferPredicate(operation, boundary);
    if (failed(indices) || failed(predicate))
      return failure();
    bool singleLane =
        !target::emission::hasPackedScalarDomain(planIndex, boundary);
    bool guard = boundary.getCheckBounds() && !predicate->empty();
    if (singleLane) {
      line("if T.get_thread_binding() == 0:");
      ++indentation;
    }
    if (guard) {
      line("if " + *predicate + ":");
      ++indentation;
    }
    line((*view)->argument->name + "[" + *indices + "] = " + stored->str());
    if (guard)
      --indentation;
    if (singleLane)
      --indentation;
    return success();
  }
  if (boundary.getTransfer() == "parallel_elements" || expanded) {
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
  Value storedValue = valueIndex ? operation.getOperand(valueIndex.getInt()) : Value();
  if (failed(node) || !binding ||
      (binding.getAccess() != "store" && binding.getAccess() != "scatter") ||
      binding.getDefer() || !valueIndex ||
      failed(relation) || failed(view) || failed(stored) ||
      !isa<RankedTensorType>(storedValue.getType()))
    return operation.emitOpError("lacks a mechanical TileLang unique store");
  if (binding.getTensorIndexing() == "none") {
    FailureOr<std::string> indices = accessIndices(operation);
    if (failed(indices))
      return failure();
    line("T.copy(" + stored->str() + ", " + (*view)->argument->name + "[" +
         *indices + "])");
    return success();
  }
  if (binding.getTensorIndexing() == "structured" && planIndex.stages.empty()) {
    FailureOr<std::string> indices = accessIndices(operation);
    if (failed(indices))
      return failure();
    line("T.copy(" + stored->str() + ", " + (*view)->argument->name + "[" +
         *indices + "])");
    return success();
  }
  auto result = dyn_cast<OpResult>(storedValue);
  Operation *definition = result ? result.getOwner() : nullptr;
  FailureOr<SmallVector<std::string>> extents =
      definition ? tensorExtents(*definition, result.getResultNumber())
                 : FailureOr<SmallVector<std::string>>(failure());
  if (failed(extents) || extents->empty())
    return operation.emitOpError("has no TileLang unique-store value shape");
  SmallVector<std::string> tileIndices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    tileIndices.push_back("unique_i" + std::to_string(axis));
    loop += tileIndices.back();
  }
  loop += " in T.Parallel(";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    loop += (*extents)[axis];
  }
  loop += "):";
  FailureOr<std::string> indices =
      elementAccessIndices(operation, tileIndices);
  FailureOr<std::string> predicate =
      elementBoundsPredicate(operation, tileIndices, false, false);
  if (failed(indices) || failed(predicate))
    return failure();
  if (!planIndex.stages.empty()) {
    auto storedTensor = dyn_cast<RankedTensorType>(storedValue.getType());
    if (activeStages.size() != 1 || relation->size() != 2 ||
        (*relation)[0].kind != "value_index" ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front() ||
        !isa<RankedTensorType>(
            operation.getOperand(*(*relation)[0].operands.front()).getType()) ||
        (*relation)[1].kind != "full_slice" || !storedTensor ||
        storedTensor.getRank() != 2 || tileIndices.size() != 2)
      return operation.emitOpError(
          "staged TileLang unique store requires one member-indexed matrix");
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
         "] = " + stored->str() + "[store_i, store_j]");
    --indentation;
    --indentation;
    return success();
  }
  line(loop);
  ++indentation;
  line("if " + *predicate + ":");
  ++indentation;
  std::string source = stored->str() + "[";
  for (auto [axis, index] : llvm::enumerate(tileIndices)) {
    if (axis)
      source += ", ";
    source += index;
  }
  source += "]";
  line((*view)->argument->name + "[" + *indices + "] = " + source);
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
  Operation *deferredValue =
      planIndex.stages.empty() && valueIndex
          ? deferredLoads.lookup(operation.getOperand(valueIndex.getInt()))
          : nullptr;
  FailureOr<StringRef> stored =
      deferredValue ? FailureOr<StringRef>(StringRef())
      : valueIndex  ? lookupValue(operation, valueIndex.getInt())
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
      bool singleLane =
          !target::emission::hasPackedScalarDomain(planIndex, boundary);
      bool returnsPrevious =
          operation.getNumResults() == 1 && !operation.getResult(0).use_empty();
      std::string result;
      if (returnsPrevious) {
        std::string dtype = dtypeName(operation.getResult(0).getType(), operation);
        if (dtype.empty())
          return failure();
        result = makeResultName(operation, 0);
        line(result + " = T.alloc_var(dtype=" + dtype + ")");
      }
      if (singleLane) {
        line("if T.get_thread_binding() == 0:");
        ++indentation;
      }
      FailureOr<std::string> predicate =
          scalarTransferPredicate(operation, boundary);
      if (failed(predicate))
        return failure();
      bool guard = boundary.getCheckBounds() && !predicate->empty();
      if (guard) {
        line("if " + *predicate + ":");
        ++indentation;
      }
      std::string call =
          "T.atomic_add(" + (*view)->argument->name + "[" + *indices + "], " +
          stored->str() + ", memory_order=\"relaxed\"";
      if (returnsPrevious) {
        line(result + " = " + call + ", return_prev=True)");
      } else {
        line(call + ")");
      }
      if (guard)
        --indentation;
      if (singleLane)
        --indentation;
      if (returnsPrevious)
        bindResult(operation, 0, result);
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
    auto atomicValue = [&]() -> FailureOr<std::string> {
      Operation *sourceLoad = deferredValue;
      if (!sourceLoad)
        return tensorElement(storedValue, tileIndices, operation);
      FailureOr<ABIView *> sourceView =
          lookupView(sourceLoad->getOperand(0), operation);
      FailureOr<std::string> sourceIndices =
          elementAccessIndices(*sourceLoad, tileIndices);
      if (failed(sourceView) || failed(sourceIndices))
        return failure();
      return (*sourceView)->argument->name + "[" + *sourceIndices + "]";
    };
    FailureOr<std::string> value = atomicValue();
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

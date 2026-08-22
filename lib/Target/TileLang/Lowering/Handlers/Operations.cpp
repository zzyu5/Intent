#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/TileLang/Lowering/Passes.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Analysis/StructuredControl.h"
#include "Intent/Target/Common/Lowering/Literal.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::tilelang::lowering {
namespace {

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter,
                         target::OperationCallback leave = {}) {
  return registry.add(name,
                      target::OperationHandler{std::move(enter), std::move(leave)});
}

} // namespace

LogicalResult registerEmissionHandlers(target::OperationHandlerRegistry &registry,
                                       ProgramMaterializer &emitter) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.domain", "intent.domain_product",
                         "intent.make_record",
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
      failed(addHandler(registry, "intent.region_end", [&](Operation &op) {
        return emitter.selectOperation(op) ? emitter.emitRegionEnd(op) : success();
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
      failed(addHandler(registry, "intent.bitcast", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitBitcast(op);
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
      failed(addHandler(registry, "intent.scaled_contract", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitScaledContract(op);
      })) ||
      failed(addHandler(registry, "intent.sparse_contract", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitSparseContract(op);
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

LogicalResult ProgramMaterializer::emitConstant(Operation &operation) {
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
      expression = target::lowering::spellFiniteFloatLiteral(floating);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported TileLang constant value");
  }
  if (operation.getBlock() == &kernel.entry.getBody().front()) {
    bindResult(operation, 0, expression);
    return success();
  }
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult ProgramMaterializer::emitExtract(Operation &operation) {
  FailureOr<Value> field = target::resolveRecordField(operation);
  if (failed(field))
    return failure();
  auto found = valueNames.find(*field);
  if (found == valueNames.end())
    return operation.emitOpError("record field has no emitted TileLang SSA value");
  bindResult(operation, 0, found->second);
  return success();
}

LogicalResult ProgramMaterializer::emitRegionEnd(Operation &operation) {
  if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
      !operation.getResult(0).getType().isIntOrIndex())
    return operation.emitOpError(
        "lacks a mechanical TileLang region-end binding");
  FailureOr<plan::AxisOp> axis = resolveAxis(operation.getOperand(0), operation);
  if (failed(axis))
    return failure();
  std::string extent = axisDimensions.lookup(axis->getNode());
  if (extent.empty())
    return operation.emitOpError("has no planned logical extent for region end");
  std::string expression = extent;
  if (isa<intent::RegionType>(operation.getOperand(0).getType())) {
    if (axis->hasRole("parallel") && !axis->isScalar()) {
      std::string block = programBlocks.lookup(axis->getNode());
      if (block.empty())
        return operation.emitOpError(
            "has no planned program block for parallel region end");
      expression = "T.min((" + block + " + 1) * " + axis->getTile().str() +
                   ", " + extent + ")";
    } else if (axis->hasRole("ordered")) {
      const target::lowering::RangeBinding *range =
          axis->getRange("traversal", 0);
      std::string start = axisIndices.lookup(axis->getNode());
      if (!range || start.empty())
        return operation.emitOpError(
            "has no planned ordered range for region end");
      expression = "T.min((" + start + ") + " + range->getTile().str() +
                   ", " + extent + ")";
    }
  }
  bindResult(operation, 0, expression);
  return success();
}

LogicalResult ProgramMaterializer::emitDimension(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitAssumeInBounds(Operation &operation) {
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
    if (failed(extents) || extents->size() != static_cast<size_t>(tensor.getRank()))
      return operation.emitOpError(
          "TileLang in-bounds assumption has no ranked index schema");
    if (tensor.getRank() != 1 || extents->front() != "1")
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

LogicalResult ProgramMaterializer::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() == 0)
    return operation.emitOpError(
        "TileLang parallel ownership requires region arguments");
  Block &body = operation.getRegion(0).front();
  plan::PartitionBindingOp partition =
      target::lowering::countPartitionForIteration(planIndex, kernel, operation);
  if (partition) {
    if (!planIndex.components.reusedAxes.empty())
      return partition.emitOpError(
          "count partition cannot use worker-reused TileLang ownership");
    plan::AxisOp axis = planIndex.axes.lookup(partition.getAxisNode());
    std::string part = programBlocks.lookup(partition.getAxisNode());
    std::string region = axisIndices.lookup(partition.getAxisNode());
    if (!axis || part.empty() || region.empty())
      return partition.emitOpError(
          "has no emitted TileLang part and region projection");
    valueNames[body.getArgument(0)] = part;
    valueNames[body.getArgument(1)] = region;
    regionIndices[body.getArgument(1)] = region;
    return success();
  }
  SmallVector<plan::AxisOp> axes;
  for (BlockArgument argument : body.getArguments()) {
    FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
    if (failed(axis))
      return failure();
    axes.push_back(*axis);
  }

  if (&operation == programRoot &&
      !planIndex.components.orderedRaggedProgramAxes.empty()) {
    for (int64_t memberNode :
         planIndex.components.orderedRaggedProgramAxes) {
      FailureOr<plan::RaggedOp> relation =
          target::lowering::uniqueRaggedRelation(planIndex, memberNode, operation);
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
      const target::lowering::RangeBinding *ownership =
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
    if (target::lowering::isPackedScalarAxis(axis)) {
      const target::lowering::RangeBinding *ownership =
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
    std::string regionIndex = axisIndices.lookup(axis.getNode());
    if (regionIndex.empty())
      return axis.emitOpError("has no emitted ownership-range index");
    regionIndices[argument] = std::move(regionIndex);
  }
  return success();
}

LogicalResult ProgramMaterializer::leaveParallel(Operation &operation) {
  if (target::lowering::countPartitionForIteration(planIndex, kernel, operation)) {
    if (&operation == programRoot && planIndex.program.getPersistent())
      indentation -= 2;
    return success();
  }
  if (operation.getNumRegions() == 1 &&
      llvm::hasSingleElement(operation.getRegion(0))) {
    for (BlockArgument argument : operation.getRegion(0).front().getArguments()) {
      FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
      if (succeeded(axis) && target::lowering::isPackedScalarAxis(*axis))
        --indentation;
    }
  }
  if (&operation == programRoot && planIndex.program.getPersistent())
    indentation -= 2;
  return success();
}

LogicalResult ProgramMaterializer::enterFor(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "for emission");
  FailureOr<SmallVector<Operation *>> domains =
      operation.getNumOperands() > 0
          ? target::expandDomainSource(operation.getOperand(0), operation)
          : FailureOr<SmallVector<Operation *>>(failure());
  if (failed(node) || failed(domains) || domains->empty() ||
      operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical TileLang sequential loop");
  Block &body = operation.getRegion(0).front();
  if (body.getNumArguments() != domains->size() + operation.getNumResults())
    return operation.emitOpError("does not match its TileLang sequential axes");
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    if (failed(initial))
      return failure();
    Type carrierType = operation.getResult(index).getType();
    if (carrierType.isIntOrIndexOrFloat()) {
      std::string dtype = dtypeName(carrierType, operation);
      if (dtype.empty())
        return failure();
      std::string carrier =
          uniqueName("loop_state_" + std::to_string(index), *node);
      line(carrier + " = T.alloc_local((1,), " + dtype + ")");
      line(carrier + "[0] = " + initial->str());
      carriers.push_back(carrier);
      valueNames[body.getArgument(domains->size() + index)] = carrier + "[0]";
    } else if (isa<RankedTensorType>(carrierType)) {
      carriers.push_back(initial->str());
      valueNames[body.getArgument(domains->size() + index)] = initial->str();
    } else {
      return operation.emitOpError(
          "has an unsupported TileLang sequential-loop carrier type");
    }
  }
  loopCarriers[&operation] = carriers;
  for (auto [index, domain] : llvm::enumerate(*domains)) {
    if (!domain)
      return operation.emitOpError("has a non-canonical TileLang loop axis");
    FailureOr<int64_t> domainNode =
        target::getNodeID(*domain, "ordered traversal range");
    bool raggedAxis =
        ::intent::target::semanticOperationName(*domain) == "intent.ragged_member";
    auto rangeValue = [&](unsigned operand) -> FailureOr<std::string> {
      Operation *definition = domain->getOperand(operand).getDefiningOp();
      auto literal = definition
                         ? definition->getAttrOfType<IntegerAttr>("intent.value")
                         : IntegerAttr();
      if (definition &&
          ::intent::target::semanticOperationName(*definition) == "intent.constant" && literal)
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
    ABIView *raggedIndices = nullptr;
    if (::intent::target::semanticOperationName(*domain) == "intent.domain" &&
        domain->getNumOperands() >= 2) {
      start = rangeValue(0);
      stop = rangeValue(1);
      if (domain->getNumOperands() == 3)
        step = rangeValue(2);
    } else if (raggedAxis && succeeded(domainNode)) {
      FailureOr<plan::RaggedOp> relation =
          target::lowering::uniqueRaggedRelation(planIndex, *domainNode,
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
      raggedIndices = ragged.indices;
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
    std::string iterator = makeRegionArgumentName(operation, index);
    plan::AxisOp axis = succeeded(domainNode)
                            ? planIndex.axes.lookup(*domainNode)
                            : plan::AxisOp();
    const target::lowering::RangeBinding *outer =
        axis ? axis.getRange("traversal", 0) : nullptr;
    const target::lowering::RangeBinding *inner =
        axis ? axis.getRange("traversal", 1) : nullptr;
    if (inner) {
      if (!outer || inner->getTileRole() != "one" ||
          (!raggedAxis && *start != "0"))
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
    std::string logicalIterator = iterator;
    if (raggedIndices) {
      logicalIterator = iterator + "_member";
      line(logicalIterator + " = " + raggedIndices->argument->name + "[" +
           addressIndex(iterator) + "]");
    }
    valueNames[body.getArgument(index)] = logicalIterator;
    regionIndices[body.getArgument(index)] = logicalIterator;
    if (raggedAxis && succeeded(domainNode))
      axisIndices[*domainNode] = logicalIterator;
  }
  return success();
}

LogicalResult ProgramMaterializer::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active TileLang sequential loop");
  FailureOr<SmallVector<Operation *>> domains =
      target::expandDomainSource(operation.getOperand(0), operation);
  if (failed(domains) || domains->empty())
    return failure();
  unsigned depth = 0;
  for (Operation *domain : *domains) {
    FailureOr<int64_t> domainNode =
        target::getNodeID(*domain, "ordered traversal range");
    plan::AxisOp axis = succeeded(domainNode)
                            ? planIndex.axes.lookup(*domainNode)
                            : plan::AxisOp();
    depth += axis && axis.getRange("traversal", 1) ? 2 : 1;
  }
  indentation -= depth;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    bool scalar = operation.getResult(index).getType().isIntOrIndexOrFloat();
    bindResult(operation, index,
               carriers->second[index] + (scalar ? "[0]" : ""));
  }
  return success();
}

LogicalResult ProgramMaterializer::enterIf(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "if emission");
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (failed(node) || failed(condition) || operation.getNumRegions() != 2)
    return operation.emitOpError("lacks a mechanical TileLang scalar branch");
  SmallVector<std::string> results;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    Type type = operation.getResult(index).getType();
    if (isa<RankedTensorType>(type)) {
      std::optional<std::string> carried;
      bool ambiguous = false;
      for (const auto &entry : planIndex.contracts) {
        plan::ContractOp contract = entry.second;
        std::optional<int64_t> conditional =
            contract.getAccumulatorConditionalNode();
        std::optional<int64_t> conditionalResult =
            contract.getAccumulatorConditionalResult();
        std::optional<int64_t> previousValue =
            contract.getAccumulatorValue();
        if (contract.getAccumulatorFlow() != "loop_carried" || !conditional ||
            !conditionalResult || !previousValue || *conditional != *node ||
            *conditionalResult != static_cast<int64_t>(index))
          continue;
        Value previous = kernel.values.lookup(*previousValue);
        auto previousName = valueNames.find(previous);
        if (!previous || previousName == valueNames.end())
          return operation.emitOpError(
              "cannot resolve its planned TileLang loop-carried accumulator");
        if (carried) {
          ambiguous = true;
          continue;
        }
        carried = previousName->second;
      }
      if (ambiguous)
        return operation.emitOpError(
            "has multiple TileLang in-place accumulator flows for one result");
      if (carried) {
        results.push_back(std::move(*carried));
      } else {
        FailureOr<std::string> result =
            allocateResult(operation, index, "fragment");
        if (failed(result))
          return failure();
        results.push_back(std::move(*result));
      }
    } else {
      std::string dtype = dtypeName(type, operation);
      if (dtype.empty())
        return failure();
      std::string result = makeResultName(operation, index) + "_branch";
      line(result + " = T.alloc_local((1,), " + dtype + ")");
      results.push_back(std::move(result));
    }
  }
  ifResults[&operation] = std::move(results);
  line("if " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult ProgramMaterializer::leaveIf(Operation &operation) {
  auto results = ifResults.find(&operation);
  if (results == ifResults.end())
    return operation.emitOpError("has no active TileLang scalar branch");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    bool tensor = isa<RankedTensorType>(operation.getResult(index).getType());
    bindResult(operation, index,
               results->second[index] + (tensor ? "" : "[0]"));
  }
  ifResults.erase(results);
  return success();
}

LogicalResult ProgramMaterializer::enterWhile(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitCondition(Operation &operation) {
  Operation *owner = target::whileConditionOwner(operation);
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (!owner || failed(condition) || operation.getNumOperands() !=
                                          owner->getNumResults() + 1)
    return operation.emitOpError("does not match a TileLang scalar while");
  line("while " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult ProgramMaterializer::leaveWhile(Operation &operation) {
  auto carriers = whileCarriers.find(&operation);
  if (carriers == whileCarriers.end())
    return operation.emitOpError("has no active TileLang scalar while");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index] + "[0]");
  return success();
}

LogicalResult ProgramMaterializer::emitYield(Operation &operation) {
  Operation *owner = operation.getParentOp();
  if (!owner)
    return operation.emitOpError("has no structured-control owner");
  StringRef name = ::intent::target::semanticOperationName(*owner);
  if (name == "intent.for") {
    auto carriers = loopCarriers.find(owner);
    if (carriers == loopCarriers.end() ||
        carriers->second.size() != operation.getNumOperands())
      return operation.emitOpError("does not match its TileLang loop state");
    for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
      FailureOr<StringRef> yielded = lookupValue(operation, index);
      if (failed(yielded))
        return failure();
      bool scalar = owner->getResult(index).getType().isIntOrIndexOrFloat();
      std::string destination =
          carriers->second[index] + (scalar ? "[0]" : "");
      if (*yielded == destination)
        continue;
      if (scalar)
        line(destination + " = " + yielded->str());
      else
        line("T.copy(" + yielded->str() + ", " + destination + ")");
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
  bool emitted = false;
  for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(operation, index);
    if (failed(yielded))
      return failure();
    bool tensor = isa<RankedTensorType>(owner->getResult(index).getType());
    if (tensor && *yielded != results->second[index]) {
      line("T.copy(" + yielded->str() + ", " + results->second[index] + ")");
      emitted = true;
    } else if (!tensor) {
      line(results->second[index] + "[0] = " + yielded->str());
      emitted = true;
    }
  }
  if (operation.getNumOperands() != 0 && !emitted)
    line("pass");
  if (operation.getParentRegion() == &owner->getRegion(0)) {
    --indentation;
    line("else:");
    ++indentation;
  }
  return success();
}

LogicalResult ProgramMaterializer::emitBuffer(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitBufferLoad(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitBufferStore(Operation &operation) {
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
ProgramMaterializer::privateWorkspaceIndex(Operation &operation) {
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
      target::lowering::projectPrivateWorkspaceOffset(
          binding, *info, *indices, planIndex, axisIndices, axisDimensions,
          spellIndex, operation);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

LogicalResult ProgramMaterializer::emitLoad(Operation &operation) {
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
  if (boundary.getDefer() && !activeDeferredContract) {
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
  StringRef zeroFill = isa<IntegerType, IndexType>(resultElementType) ? "0"
                                                                      : "0.0";
  auto accessRanges =
      target::lowering::accessRangesForTransfer(planIndex, *node);
  if (accessRanges.size() > 1)
    return operation.emitOpError(
        "TileLang cannot project a multi-axis checked access footprint as one "
        "cooperative transfer");
  if (boundary.getTransfer() == "compact") {
    if (scalarResult || failed(view) || accessRanges.size() != 1 ||
        !accessRanges.front().isCompact())
      return operation.emitOpError(
          "has no proven compact TileLang transfer coverage");
    const target::lowering::RangeBinding &compact = accessRanges.front();
    int64_t divisor = compact.getDivisor();
    int64_t offset = compact.getOffset();
    auto compactAxis = planIndex.axes.find(compact.getAxisNode());
    auto active = activeTraversalIndices.find(compact.getAxisNode());
    std::string logicalBase =
        active != activeTraversalIndices.end() && !active->second.empty()
            ? active->second.back()
        : compactAxis == planIndex.axes.end()
            ? std::string()
            : axisIndices.lookup(compactAxis->second.getNode());
    std::string logicalTile = compact.getTile().str();
    if (compactAxis == planIndex.axes.end() || logicalBase.empty() ||
        logicalTile.empty() || divisor <= 1 || offset < 0)
      return compact.emitOpError(
          "has no mechanical compact TileLang source-axis projection");

    FailureOr<SmallVector<target::IndexTerm>> relation =
        target::parseIndexRelation(operation);
    FailureOr<SmallVector<std::string>> resultExtents =
        tensorExtents(operation, 0);
    auto resultType = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
    if (failed(relation) || failed(resultExtents) || !resultType ||
        static_cast<size_t>(resultType.getRank()) != resultExtents->size())
      return operation.emitOpError(
          "has no ranked compact TileLang transfer result");

    std::string physicalBase = "((" + logicalBase + ") + " +
                               std::to_string(offset) + ") // " +
                               std::to_string(divisor);
    std::string physicalExtent = "T.ceildiv(" + logicalTile + ", " +
                                 std::to_string(divisor) + ")";
    SmallVector<std::string> sourceBases;
    SmallVector<std::string> sourceExtents;
    SmallVector<int64_t> sourceBufferAxes;
    SmallVector<std::string> compactShape;
    unsigned sourceAxis = 0;
    int64_t bufferAxis = 0;
    int64_t compactBufferAxis = -1;
    for (const target::IndexTerm &term : *relation) {
      if (term.kind == "new_axis")
        return operation.emitOpError(
            "compact TileLang transfer does not support inserted axes");
      if (sourceAxis >= (*view)->shape.size())
        return operation.emitOpError(
            "compact TileLang transfer exceeds its source rank");
      if (term.kind == "static_index") {
        if (term.staticValues.size() != 1 || !term.staticValues.front())
          return operation.emitOpError(
              "compact TileLang transfer has an invalid static index");
        sourceBases.push_back(std::to_string(*term.staticValues.front()));
        sourceExtents.emplace_back();
        sourceBufferAxes.push_back(-1);
        ++sourceAxis;
        continue;
      }
      if (term.kind == "full_slice") {
        sourceBases.push_back("0");
        sourceExtents.push_back((*view)->shape[sourceAxis]);
        sourceBufferAxes.push_back(bufferAxis++);
        compactShape.push_back((*view)->shape[sourceAxis]);
        ++sourceAxis;
        continue;
      }
      if ((term.kind != "region_index" && term.kind != "value_index") ||
          term.operands.size() != 1 || !term.operands.front())
        return operation.emitOpError(
            "compact TileLang transfer has no mechanical index relation");
      Value indexed = operation.getOperand(*term.operands.front());
      if (sourceAxis == static_cast<unsigned>(compact.getSourceAxis())) {
        auto tensor = dyn_cast<RankedTensorType>(indexed.getType());
        if (term.kind != "value_index" || !tensor || tensor.getRank() != 1)
          return operation.emitOpError(
              "compact TileLang source axis requires one rank-one quotient index");
        sourceBases.push_back(physicalBase);
        sourceExtents.push_back(physicalExtent);
        sourceBufferAxes.push_back(bufferAxis);
        compactBufferAxis = bufferAxis++;
        compactShape.push_back(physicalExtent);
        ++sourceAxis;
        continue;
      }
      if (term.kind == "region_index") {
        FailureOr<plan::AxisOp> axis = resolveAxis(indexed, operation);
        std::string base = succeeded(axis)
                               ? axisIndices.lookup(axis->getNode())
                               : std::string();
        std::string extent = succeeded(axis) ? axis->getTile().str()
                                             : std::string();
        if (failed(axis) || base.empty() || extent.empty() || axis->isScalar())
          return operation.emitOpError(
              "compact TileLang transfer has no tiled region projection");
        sourceBases.push_back(base);
        sourceExtents.push_back(extent);
        sourceBufferAxes.push_back(bufferAxis++);
        compactShape.push_back(extent);
      } else {
        if (isa<RankedTensorType>(indexed.getType()))
          return operation.emitOpError(
              "compact TileLang transfer supports one tensor-valued source index");
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(exact))
          return failure();
        sourceBases.push_back(exact->str());
        sourceExtents.emplace_back();
        sourceBufferAxes.push_back(-1);
      }
      ++sourceAxis;
    }
    if (sourceAxis != (*view)->shape.size() || compactBufferAxis < 0 ||
        compactShape.size() != resultExtents->size())
      return operation.emitOpError(
          "compact TileLang transfer rank does not match its logical result");

    FailureOr<std::string> result =
        allocateResult(operation, 0, boundary.getResultSpace());
    std::string dtype = dtypeName(resultType.getElementType(), operation);
    if (failed(result) || dtype.empty())
      return operation.emitOpError(
          "cannot allocate a compact TileLang transfer result");
    std::string compactResult = *result + "_compact";
    std::string compactShapeText = "(";
    for (auto [axis, extent] : llvm::enumerate(compactShape)) {
      if (axis)
        compactShapeText += ", ";
      compactShapeText += extent;
    }
    if (compactShape.size() == 1)
      compactShapeText += ",";
    compactShapeText += ")";
    StringRef compactAllocator = boundary.getCoverageSpace() == "shared"
                                     ? "T.alloc_shared"
                                     : "T.alloc_fragment";
    line(compactResult + " = " + compactAllocator.str() + "(" +
         compactShapeText + ", " + dtype + ")");

    auto join = [](ArrayRef<std::string> values) {
      std::string result;
      for (auto [index, value] : llvm::enumerate(values)) {
        if (index)
          result += ", ";
        result += value;
      }
      return result;
    };
    auto conjunction = [](ArrayRef<std::string> values) {
      std::string result;
      for (auto [index, value] : llvm::enumerate(values)) {
        if (index)
          result += " and ";
        result += value;
      }
      return result;
    };
    SmallVector<std::string> sourceSlices;
    SmallVector<std::string> wholePredicates;
    for (unsigned axis = 0; axis < sourceBases.size(); ++axis) {
      if (sourceExtents[axis].empty()) {
        sourceSlices.push_back(sourceBases[axis]);
        wholePredicates.push_back("0 <= " + sourceBases[axis] + " and " +
                                  sourceBases[axis] + " < " +
                                  (*view)->shape[axis]);
        continue;
      }
      sourceSlices.push_back(sourceBases[axis] + " : " + sourceBases[axis] +
                             " + " + sourceExtents[axis]);
      wholePredicates.push_back("0 <= " + sourceBases[axis] + " and " +
                                sourceBases[axis] + " + " +
                                sourceExtents[axis] + " <= " +
                                (*view)->shape[axis]);
    }
    line("if " + conjunction(wholePredicates) + ":");
    ++indentation;
    line("T.copy(" + (*view)->argument->name + "[" + join(sourceSlices) +
         "], " + compactResult + ")");
    --indentation;
    line("else:");
    ++indentation;
    SmallVector<std::string> compactIndices;
    std::string compactLoop = "for ";
    for (unsigned axis = 0; axis < compactShape.size(); ++axis) {
      if (axis)
        compactLoop += ", ";
      compactIndices.push_back("compact_load_i" + std::to_string(axis));
      compactLoop += compactIndices.back();
    }
    compactLoop += " in T.Parallel(" + join(compactShape) + "):";
    line(compactLoop);
    ++indentation;
    SmallVector<std::string> sourceElements;
    SmallVector<std::string> elementPredicates;
    for (unsigned axis = 0; axis < sourceBases.size(); ++axis) {
      std::string element = sourceBases[axis];
      if (sourceBufferAxes[axis] >= 0)
        element += " + " + compactIndices[sourceBufferAxes[axis]];
      sourceElements.push_back(element);
      elementPredicates.push_back("0 <= " + element + " and " + element +
                                  " < " + (*view)->shape[axis]);
    }
    line("if " + conjunction(elementPredicates) + ":");
    ++indentation;
    line(compactResult + "[" + join(compactIndices) + "] = " +
         (*view)->argument->name + "[" + join(sourceElements) + "]");
    --indentation;
    line("else:");
    ++indentation;
    line(compactResult + "[" + join(compactIndices) + "] = 0");
    --indentation;
    --indentation;
    --indentation;
    if (boundary.getCoverageSpace() == "shared")
      line("T.sync_threads()");

    SmallVector<std::string> logicalIndices;
    std::string expandLoop = "for ";
    for (unsigned axis = 0; axis < resultExtents->size(); ++axis) {
      if (axis)
        expandLoop += ", ";
      logicalIndices.push_back("compact_expand_i" + std::to_string(axis));
      expandLoop += logicalIndices.back();
    }
    expandLoop += " in T.Parallel(" + join(*resultExtents) + "):";
    line(expandLoop);
    ++indentation;
    SmallVector<std::string> physicalIndices = logicalIndices;
    physicalIndices[compactBufferAxis] =
        "((" + logicalBase + ") + " +
        logicalIndices[compactBufferAxis] + " + " + std::to_string(offset) +
        ") // " + std::to_string(divisor) + " - (" + physicalBase + ")";
    line(*result + "[" + join(logicalIndices) + "] = " + compactResult + "[" +
         join(physicalIndices) + "]");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }
  bool expanded = !physicalFill->empty();
  bool tensorIndirect = boundary.hasDataDependentTensorIndex() ||
                        boundary.hasCompactTensorIndex();
  bool plannedValidity = !boundary.getConsumerNeutralized() &&
                         boundary.getPadding() != "none" &&
                         !boundary.getValidityDomainNodes().empty();
  bool guardedF16Bulk = boundary.getTransfer() == "guarded_f16_bulk";
  FailureOr<SmallVector<target::IndexTerm>> relation = failure();
  if (guardedF16Bulk) {
    relation = target::parseIndexRelation(operation);
    if (failed(relation))
      return failure();
  }
  if (guardedF16Bulk) {
    if (expanded) {
      if (failed(view) || failed(relation) ||
          relation->size() != (*view)->shape.size())
        return operation.emitOpError(
            "cannot resolve guarded float16 bulk extents");
      for (auto [axis, term] : llvm::enumerate(*relation)) {
        auto extent = planIndex.blockExtents.find((*view)->shape[axis]);
        if (term.kind != "full_slice" ||
            extent == planIndex.blockExtents.end())
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
    if (boundary.getResultSpace() == "shared")
      line("T.sync_threads()");
    --indentation;
    line("else:");
    ++indentation;
    SmallVector<std::string> tileIndices;
    std::string loop = "for ";
    std::string targetIndices;
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis) {
        loop += ", ";
        targetIndices += ", ";
      }
      std::string index = "guarded_transfer_i" + std::to_string(axis);
      tileIndices.push_back(index);
      loop += index;
      targetIndices += index;
    }
    loop += " in T.Parallel(";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        loop += ", ";
      loop += extent;
    }
    FailureOr<std::string> elementIndices =
        elementAccessIndices(operation, tileIndices);
    FailureOr<std::string> elementPredicate =
        elementBoundsPredicate(operation, tileIndices, false, false);
    if (failed(elementIndices) || failed(elementPredicate))
      return failure();
    line(loop + "):");
    ++indentation;
    line("if " + *elementPredicate + ":");
    ++indentation;
    line(*result + "[" + targetIndices + "] = " +
         (*view)->argument->name + "[" + *elementIndices + "]");
    --indentation;
    line("else:");
    ++indentation;
    StringRef fill = boundary.getPadding() == "negative_infinity"
                         ? "-T.infinity(T.float32)"
                         : zeroFill;
    line(*result + "[" + targetIndices + "] = " + fill.str());
    --indentation;
    --indentation;
    if (boundary.getResultSpace() == "shared")
      line("T.sync_threads()");
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
        target::lowering::hasPackedScalarDomain(planIndex, boundary);
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
  if (boundary.getTransfer() == "parallel_elements" || expanded ||
      boundary.getCheckBounds() || tensorIndirect) {
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
    FailureOr<std::string> validityPredicate =
        plannedValidity
            ? elementValidityPredicate(boundary.getValidityTensorAxes(),
                                       boundary.getValidityDomainNodes(),
                                       tileIndices, operation)
            : FailureOr<std::string>(std::string("True"));
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
    if (failed(logicalPredicate) || failed(validityPredicate) ||
        failed(physicalPredicate) ||
        failed(wholeTile) || failed(bulkIndices))
      return failure();
    if (*validityPredicate != "True") {
      if (logicalPredicate->empty())
        *logicalPredicate = *validityPredicate;
      else
        *logicalPredicate = "(" + *logicalPredicate + ") and (" +
                            *validityPredicate + ")";
    }
    bool hasBulkFastPath =
        !expanded && !wholeTile->empty() && !plannedValidity;
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
    std::string physicalFallback =
        padding == "negative_infinity" ? "-T.infinity(T.float32)"
                                        : zeroFill.str();
    FailureOr<std::string> semanticFallback =
        paddingFillExpression(operation.getResult(0), operation);
    if (failed(semanticFallback))
      return failure();
    if (!semanticFallback->empty())
      physicalFallback = *semanticFallback;
    std::string logicalFallback = semanticFallback->empty()
                                      ? physicalFallback
                                      : *semanticFallback;
    auto emitElementwise = [&](bool includePhysicalBounds) -> LogicalResult {
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
      std::string source = (*view)->argument->name + "[" + *indices + "]";
      FailureOr<std::string> padded = padElementExpression(
          operation.getResult(0), source, tileIndices, operation);
      if (failed(padded))
        return failure();
      line(target + "] = " + *padded);
      if (includePhysicalBounds) {
        --indentation;
        line("else:");
        ++indentation;
        line(target + "] = " + physicalFallback);
        --indentation;
      }
      if (materializeLogicalBounds) {
        --indentation;
        line("else:");
        ++indentation;
        line(target + "] = " + logicalFallback);
        --indentation;
      }
      indentation -= loopDepth;
      if (boundary.getResultSpace() == "shared")
        line("T.sync_threads()");
      return success();
    };
    if (failed(emitElementwise(expanded && !stagePhysicalPadding)))
      return failure();
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

LogicalResult ProgramMaterializer::emitIndices(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "indices emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  auto resultType = operation.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                        : RankedTensorType();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "logical_indices" ||
      !resultType || resultType.getRank() <= 0 ||
      !isa<IntegerType, IndexType>(resultType.getElementType()) ||
      failed(result) || failed(extents) ||
      extents->size() != static_cast<size_t>(resultType.getRank()))
    return operation.emitOpError("lacks a mechanical TileLang indices binding");
  auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
  auto tensorAxis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  FailureOr<plan::AxisOp> axis = failure();
  unsigned emittedAxis = 0;
  if (mode && mode.getValue() == "tensor_axis") {
    if (!tensorAxis || tensorAxis.getInt() < 0 ||
        tensorAxis.getInt() >= resultType.getRank() ||
        static_cast<size_t>(tensorAxis.getInt()) >= binding.getAxisNodes().size())
      return operation.emitOpError("has no tensor-axis indices binding");
    axis = planIndex.axes.lookup(binding.getAxisNodes()[tensorAxis.getInt()]);
    emittedAxis = tensorAxis.getInt();
  } else {
    if (resultType.getRank() != 1)
      return operation.emitOpError("domain indices require one result axis");
    axis = resolveAxis(operation.getOperand(0), operation);
  }
  if (failed(axis))
    return failure();
  std::string base;
  if (auto argument = dyn_cast<BlockArgument>(operation.getOperand(0))) {
    FailureOr<target::lowering::RegionRangeBinding> region =
        target::lowering::selectedRegionArgumentRange(planIndex, kernel, argument,
                                                      operation);
    if (failed(region))
      return failure();
    base = regionIndices.lookup(argument);
  } else {
    base = axisIndices.lookup(axis->getNode());
  }
  if (base.empty() && axis->hasRole("lane")) {
    base = "0";
    axisIndices[axis->getNode()] = base;
  }
  if (base.empty())
    return operation.emitOpError("has no TileLang vector-index projection");
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned index = 0; index < extents->size(); ++index) {
    if (index)
      loop += ", ";
    std::string name = "indices_i" + std::to_string(index);
    loop += name;
    indices.push_back(std::move(name));
  }
  loop += " in T.Parallel(" + llvm::join(*extents, ", ") + ")";
  line(loop + ":");
  ++indentation;
  std::string expression = base + " + " + indices[emittedAxis];
  FailureOr<std::string> padded = padElementExpression(
      operation.getResult(0), expression, indices, operation);
  if (failed(padded))
    return failure();
  line(*result + "[" + llvm::join(indices, ", ") + "] = " + *padded);
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitRandom(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  bool argReduction =
      binding && binding.getLowering() == "T.reduce_max_with_index";
  bool logicalReduction = binding &&
                          (binding.getLowering() == "T.reduce_any_i32" ||
                           binding.getLowering() == "T.reduce_all_i32");
  unsigned expectedResults =
      argReduction ? 2 : components ? components.getInt() : 0;
  if (failed(node) || !binding || !components || components.getInt() <= 0 ||
      operation.getNumResults() != expectedResults)
    return operation.emitOpError("lacks a TileLang reduction binding");
  SmallVector<std::string> operands;
  for (unsigned component = 0; component < expectedResults; ++component) {
    FailureOr<StringRef> operand = lookupValue(operation, component);
    if (failed(operand))
      return failure();
    operands.push_back(operand->str());
  }
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
    line(inputBuffer + "[logical_i] = T.cast(" + operands.front() +
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
    FailureOr<StringRef> indexIdentity = lookupValue(operation, 3);
    auto input = dyn_cast<OpResult>(operation.getOperand(0));
    FailureOr<SmallVector<std::string>> inputExtents =
        input ? tensorExtents(*input.getOwner(), input.getResultNumber())
              : FailureOr<SmallVector<std::string>>(failure());
    int64_t axis = binding.getAxis();
    bool tensorResults = isa<RankedTensorType>(operation.getResult(0).getType());
    if (failed(indexIdentity) || operands.size() != 2 || failed(inputExtents) || axis < 0 ||
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
    line("T.reduce_max(" + operands.front() + ", " + valueStorage + ", dim=" +
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
         access(operands.front(), inputIndices) + " == " + maximum + ", " +
         access(operands[1], inputIndices) + ", " + indexIdentity->str() + ")");
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
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    if (isa<RankedTensorType>(operation.getResult(component).getType())) {
      FailureOr<std::string> result =
          allocateResult(operation, component, "fragment");
      if (failed(result))
        return failure();
      line(binding.getLowering().str() + "(" + operands[component] + ", " +
           *result + ", dim=" + std::to_string(binding.getAxis()) +
           ", clear=True)");
      bindResult(operation, component, *result);
      continue;
    }
    std::string dtype =
        dtypeName(operation.getResult(component).getType(), operation);
    if (dtype.empty())
      return failure();
    std::string result =
        makeResultName(operation, component) + "_fragment";
    line(result + " = T.alloc_fragment((1,), " + dtype + ")");
    line(binding.getLowering().str() + "(" + operands[component] + ", " +
         result + ", dim=" + std::to_string(binding.getAxis()) +
         ", clear=True)");
    valueNames[operation.getResult(component)] = result + "[0]";
  }
  return success();
}

LogicalResult ProgramMaterializer::emitScan(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "scan emission");
  plan::ScanOp binding =
      succeeded(node) ? planIndex.scans.lookup(*node) : plan::ScanOp();
  if (binding && binding.getResultSpace() == "global") {
    plan::AxisOp axis = planIndex.axes.lookup(binding.getAxisNode());
    const target::lowering::RangeBinding *range =
        axis ? axis.getRange("traversal", 0) : nullptr;
    std::string extent = scanExtents.lookup(binding.getNode());
    Operation *scan = kernel.nodes.lookup(binding.getNode());
    if (failed(node) || binding.getLowering() != "T.cumsum" || !range ||
        extent.empty() || !scan || scan->getNumResults() == 0)
      return operation.emitOpError(
          "lacks a workspace-backed TileLang scan binding");
    SmallVector<std::string> carries;
    for (unsigned component = 0; component < operation.getNumResults(); ++component) {
      auto resultType =
          dyn_cast<RankedTensorType>(scan->getResult(component).getType());
      std::string carryDtype =
          resultType ? dtypeName(resultType.getElementType(), operation)
                     : std::string();
      FailureOr<StringRef> identity =
          lookupValue(operation, operation.getNumResults() + component);
      if (!resultType || carryDtype.empty() || failed(identity))
        return operation.emitOpError("has no TileLang scan carry schema");
      carries.push_back(makeResultName(operation, component) + "_carry");
      line(carries.back() + " = T.alloc_local((1,), " + carryDtype + ")");
      line(carries.back() + "[0] = " + identity->str());
    }
    std::string stem = makeResultName(operation, 0);
    std::string block = stem + "_block";
    std::string index = stem + "_index";
    line("for " + block + " in T.serial(T.ceildiv(" + extent + ", " +
         range->getTile().str() + ")):");
    ++indentation;
    line("for " + index + " in T.serial(" + range->getTile().str() + "):");
    ++indentation;
    std::string logical = block + " * " + range->getTile().str() + " + " + index;
    if (failed(replayScanProducers(binding, logical)))
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
    FailureOr<std::string> workspace =
        scanWorkspaceIndex(binding, logical, operation);
    if (failed(workspace))
      return failure();
    for (unsigned component = 0; component < operation.getNumResults(); ++component) {
      FailureOr<StringRef> operand = lookupValue(operation, component);
      FailureOr<StringRef> identity =
          lookupValue(operation, operation.getNumResults() + component);
      if (failed(operand) || failed(identity))
        return failure();
      line(carries[component] + "[0] = " + carries[component] +
           "[0] + T.if_then_else(" + logical + " < " + extent + ", " +
           operand->str() + "[0], " + identity->str() + ")");
    }
    line("if " + logical + " < " + extent + ":");
    ++indentation;
    for (unsigned component = 0; component < operation.getNumResults(); ++component)
      line(workspaceNames.lookup(scan->getResult(component)) + "[" + *workspace +
           "] = " + carries[component] + "[0]");
    --indentation;
    indentation -= 2;
    for (Value result : operation.getResults())
      valueNames.erase(result);
    return success();
  }
  if (failed(node) || !binding || binding.getLowering() != "T.cumsum" ||
      operation.getNumResults() == 0)
    return operation.emitOpError("lacks a mechanical TileLang scan binding");
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    FailureOr<StringRef> operand = lookupValue(operation, component);
    FailureOr<std::string> result =
        allocateResult(operation, component, binding.getResultSpace());
    if (failed(operand) || failed(result))
      return failure();
    line("T.copy(" + operand->str() + ", " + *result + ")");
    line("T.cumsum(" + *result + ", dim=" +
         std::to_string(binding.getAxis()) + ")");
    bindResult(operation, component, *result);
  }
  return success();
}

LogicalResult ProgramMaterializer::replayScanProducers(const plan::ScanOp &binding,
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

LogicalResult ProgramMaterializer::replayContractProducers(
    ArrayRef<Operation *> producers) {
  if (!activeDeferredContract || !operationRegistry())
    return failure();
  for (Operation *producer : producers)
    if (!producer || failed(operationRegistry()->dispatch(
                         *producer,
                         "TileLang deferred contraction producer replay")))
      return failure();
  return success();
}

FailureOr<std::string>
ProgramMaterializer::scanWorkspaceIndex(const plan::ScanOp &binding,
                                  StringRef logicalIndex,
                                  Operation &consumer) {
  FailureOr<std::string> offset = target::lowering::projectScanWorkspaceOffset(
      binding, scanExtents.lookup(binding.getNode()), logicalIndex, planIndex,
      axisIndices, axisDimensions, consumer);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

FailureOr<std::string>
ProgramMaterializer::scanMaterializedIndex(Value value, StringRef logicalIndex,
                                     Operation &consumer) {
  auto binding = scanMaterializedValues.find(value);
  if (binding == scanMaterializedValues.end())
    return consumer.emitOpError(
        "has no TileLang scan materialization binding");
  return scanWorkspaceIndex(binding->second, logicalIndex, consumer);
}

LogicalResult ProgramMaterializer::emitBroadcast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "broadcast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "alias" ||
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

LogicalResult ProgramMaterializer::emitUnary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "unary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding || operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local"))
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
  FailureOr<std::string> allocated = allocateResult(operation, 0, "fragment");
  if (failed(allocated))
    return failure();
  std::string result = *allocated;
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

LogicalResult ProgramMaterializer::emitBinary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "binary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding || operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local"))
    return operation.emitOpError("lacks a TileLang binary binding");
  auto inPlace = inPlacePointwiseResults.find(&operation);
  if (inPlace != inPlacePointwiseResults.end()) {
    std::string result = inPlace->second;
    inPlacePointwiseResults.erase(inPlace);
    bindResult(operation, 0, result);
    return success();
  }
  std::string resultName = makeResultName(operation, 0);
  auto makeExpression = [&](StringRef lhs,
                            StringRef rhs) -> FailureOr<std::string> {
    if (binding.getLowering() == "T.max" ||
        binding.getLowering() == "T.min")
      return binding.getLowering().str() + "(" + lhs.str() + ", " +
             rhs.str() + ")";
    if (binding.getLowering() == "T.pow")
      return "T.pow(" + lhs.str() + ", " + rhs.str() + ")";
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
  FailureOr<std::string> allocated = allocateResult(operation, 0, "fragment");
  if (failed(allocated))
    return failure();
  std::string result = *allocated;
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

LogicalResult ProgramMaterializer::emitConditional(Operation &operation, bool mask) {
  FailureOr<int64_t> node = target::getNodeID(
      operation, mask ? "mask emission" : "select emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  unsigned conditionOperand = mask ? 1 : 0;
  unsigned trueOperand = mask ? 0 : 1;
  if (failed(node) || !binding ||
      binding.getLowering() != "T.if_then_else" ||
      operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local"))
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

LogicalResult ProgramMaterializer::emitMask(Operation &operation) {
  return emitConditional(operation, true);
}

LogicalResult ProgramMaterializer::emitSelect(Operation &operation) {
  return emitConditional(operation, false);
}

LogicalResult ProgramMaterializer::emitCast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "cast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding ||
      (binding.getLowering() != "T.cast" &&
       binding.getLowering() != "T.copy_cast") ||
      operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local"))
    return operation.emitOpError("lacks a TileLang cast binding");
  Type sourceElementType = operation.getOperand(0).getType();
  if (auto tensor = dyn_cast<RankedTensorType>(sourceElementType))
    sourceElementType = tensor.getElementType();
  Type resultElementType = operation.getResult(0).getType();
  if (auto tensor = dyn_cast<RankedTensorType>(resultElementType))
    resultElementType = tensor.getElementType();
  bool decodeE8M0 = isa<Float8E8M0FNUType>(sourceElementType) &&
                    !isa<Float8E8M0FNUType>(resultElementType);
  auto castExpression = [&](StringRef value, StringRef dtype) {
    return syntax::cast(value, dtype, decodeE8M0, resultElementType.isF32());
  };
  if (!tensorResult) {
    if (binding.getLowering() != "T.cast")
      return operation.emitOpError("cannot copy-cast a scalar TileLang value");
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    std::string dtype = dtypeName(operation.getResult(0).getType(), operation);
    if (failed(operand) || dtype.empty())
      return failure();
    std::string result = makeResultName(operation, 0);
    std::string expression = castExpression(*operand, dtype);
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
  line(target + "] = " + castExpression(*operand, dtype));
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitBitcast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "bitcast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  bool tensorResult = operation.getNumResults() == 1 &&
                      isa<RankedTensorType>(operation.getResult(0).getType());
  if (failed(node) || !binding || binding.getLowering() != "T.reinterpret" ||
      operation.getNumResults() != 1 ||
      binding.getSpace() != (tensorResult ? "fragment" : "local"))
    return operation.emitOpError("lacks a TileLang bitcast binding");
  if (!tensorResult) {
    FailureOr<StringRef> operand = lookupValue(operation, 0);
    std::string dtype = dtypeName(operation.getResult(0).getType(), operation);
    if (failed(operand) || dtype.empty())
      return failure();
    std::string result = makeResultName(operation, 0);
    std::string expression = syntax::bitcast(*operand, dtype);
    if (target::whileConditionOwner(operation)) {
      bindResult(operation, 0, expression);
      return success();
    }
    line(result + " = " + expression);
    bindResult(operation, 0, result);
    return success();
  }
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  auto tensor = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  if (failed(result) || failed(extents) || !tensor)
    return operation.emitOpError("lacks a TileLang bitcast binding");
  std::string dtype = dtypeName(tensor.getElementType(), operation);
  if (dtype.empty())
    return failure();
  SmallVector<std::string> indices;
  std::string loop = "for ";
  for (unsigned axis = 0; axis < extents->size(); ++axis) {
    if (axis)
      loop += ", ";
    indices.push_back("bitcast_i" + std::to_string(axis));
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
  line(target + "] = " + syntax::bitcast(*operand, dtype));
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitReshape(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reshape emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  auto sourceType = operation.getNumOperands() == 1
                        ? dyn_cast<RankedTensorType>(
                              operation.getOperand(0).getType())
                        : RankedTensorType();
  auto resultType = operation.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                        : RankedTensorType();
  if (sourceType && resultType && sourceType.getRank() == 1 &&
      resultType.getRank() == 0) {
    bool singleton = sourceType.getDimSize(0) == 1;
    if (!singleton) {
      auto source = dyn_cast<OpResult>(operation.getOperand(0));
      FailureOr<SmallVector<std::string>> extents =
          source ? tensorExtents(*source.getOwner(), source.getResultNumber())
                 : FailureOr<SmallVector<std::string>>(failure());
      singleton = succeeded(extents) && extents->size() == 1 &&
                  extents->front() == "1";
    }
    if (!singleton)
      return operation.emitOpError(
          "TileLang scalar reshape requires a singleton physical source");
    if (failed(operand))
      return failure();
    bindResult(operation, 0, operand->str() + "[0]");
    return success();
  }
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
      const target::lowering::RangeBinding *range =
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
      binding.getSpace() != "fragment" || failed(operand) || failed(shape) ||
      operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a TileLang reshape binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = T.reshape(" + operand->str() + ", " + *shape + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitTranspose(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "transpose emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<SmallVector<int64_t>> permutation =
      target::lowering::transposePermutation(operation);
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> result = allocateResult(operation, 0, "shared");
  if (failed(node) || !binding ||
      binding.getLowering() != "T.transpose" ||
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

LogicalResult ProgramMaterializer::emitFull(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "full emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> fill = lookupValue(operation, 0);
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(node) || !binding || binding.getLowering() != "T.fill" ||
      failed(fill) || failed(result))
    return operation.emitOpError("lacks a TileLang full binding");
  FailureOr<int64_t> valueID = target::getValueID(
      operation.getResult(0), kernel, operation, "TileLang full padding lookup");
  if (failed(valueID))
    return failure();
  if (!planIndex.paddings.contains(*valueID)) {
    line("T.fill(" + *result + ", " + fill->str() + ")");
  } else {
    FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
    if (failed(extents))
      return failure();
    SmallVector<std::string> indices;
    std::string loop = "for ";
    for (unsigned axis = 0; axis < extents->size(); ++axis) {
      if (axis)
        loop += ", ";
      indices.push_back("full_i" + std::to_string(axis));
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
    FailureOr<std::string> padded = padElementExpression(
        operation.getResult(0), *fill, indices, operation);
    if (failed(padded))
      return failure();
    std::string target = *result + "[";
    for (auto [axis, index] : llvm::enumerate(indices)) {
      if (axis)
        target += ", ";
      target += index;
    }
    line(target + "] = " + *padded);
    --indentation;
  }
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitZeros(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "zeros emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(node) || !binding || binding.getLowering() != "T.clear" ||
      failed(result))
    return operation.emitOpError("lacks a TileLang zeros binding");
  line("T.clear(" + *result + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitGather(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "gather emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  auto formAttr = binding
                      ? binding.operation->getAttrOfType<StringAttr>(gatherFormAttr)
                      : StringAttr();
  StringRef form = formAttr ? formAttr.getValue() : StringRef();
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  auto validIndex =
      operation.getAttrOfType<IntegerAttr>("intent.valid_operand_index");
  auto fillIndex =
      operation.getAttrOfType<IntegerAttr>("intent.fill_operand_index");
  if (failed(node) || !binding || failed(relation) || !validIndex || !fillIndex)
    return operation.emitOpError("lacks a TileLang gather binding");
  auto resultStorage = [&]() -> FailureOr<std::string> {
    return allocateResult(operation, 0, "fragment");
  };
  bool scalarFragmentGather =
      form == "scalar_fragment" &&
      isa<RankedTensorType>(operation.getOperand(0).getType()) &&
      !isa<RankedTensorType>(operation.getResult(0).getType()) &&
      relation->size() == 1;
  if (form == "extract_first_scalar" &&
      relation->size() == 1 && relation->front().kind == "static_index") {
    FailureOr<StringRef> source = lookupValue(operation, 0);
    if (failed(source))
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = " + source->str() + "[0]");
    bindResult(operation, 0, result);
    return success();
  }
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
      if (failed(workspace))
        return failure();
      line(result + " = T.if_then_else(" + valid->str() + ", " +
           workspaceNames.lookup(operation.getOperand(0)) + "[" + *workspace +
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
  if (form == "view_indirect" &&
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
  bool expand = form == "expand_dims" &&
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
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(source) || failed(valid) || failed(fill) || failed(extents) ||
      extents->size() != relation->size())
    return failure();
  Operation *validDefinition =
      operation.getOperand(validIndex.getInt()).getDefiningOp();
  auto validConstant =
      validDefinition &&
              ::intent::target::semanticOperationName(*validDefinition) == "intent.constant"
          ? validDefinition->getAttrOfType<BoolAttr>("intent.value")
          : BoolAttr();
  if (validConstant && validConstant.getValue()) {
    std::string shape = "(";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        shape += ", ";
      shape += extent;
    }
    if (extents->size() == 1)
      shape += ",";
    shape += ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = T.reshape(" + source->str() + ", " + shape + ")");
    bindResult(operation, 0, result);
    return success();
  }
  FailureOr<std::string> result = resultStorage();
  if (failed(result))
    return failure();
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

LogicalResult ProgramMaterializer::emitMembers(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "members emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  if (failed(node) || !binding || binding.getLowering() != "T.members" ||
      operation.getNumOperands() != 1 || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a TileLang members binding");
  FailureOr<plan::AxisOp> memberAxis =
      resolveAxis(operation.getOperand(0), operation);
  FailureOr<plan::RaggedOp> relation =
      succeeded(memberAxis)
          ? target::lowering::uniqueRaggedRelation(
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
    std::string position = axisIndices.lookup(memberAxis->getNode());
    if (auto argument = dyn_cast<BlockArgument>(operation.getOperand(0))) {
      FailureOr<target::lowering::RegionRangeBinding> selected =
          target::lowering::selectedRegionArgumentRange(planIndex, kernel,
                                                        argument, operation);
      if (failed(selected) ||
          selected->axis.getNode() != memberAxis->getNode())
        return operation.emitOpError(
            "has no exact TileLang region projection for ragged members");
      position = regionIndices.lookup(argument);
    }
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
  return operation.emitOpError(
      "has no single-launch ordered ragged traversal realization");
}

LogicalResult ProgramMaterializer::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical TileLang stream binding");
  Block &body = operation.getRegion(0).front();
  bool hasStop = static_cast<bool>(binding.getStopValueAttr());
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
  bool partitionedStream = binding.hasPartition();
  if (raggedStream && partitionedStream)
    return binding.emitOpError(
        "cannot mechanically combine ragged and count-partition stream bounds");
  std::string raggedSuffix = std::to_string(binding.getAxisNode());
  if (raggedStream) {
    FailureOr<plan::RaggedOp> relation =
        target::lowering::uniqueRaggedRelation(
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
  std::string partitionBegin;
  std::string partitionEnd;
  std::string activeEnd;
  std::string streamExtent =
      raggedStream ? "sequence_length_" + raggedSuffix
                   : logicalExtent(binding.getExtent());
  if (partitionedStream) {
    plan::PartitionBindingOp partition = binding.getPartition();
    plan::AxisOp axis = planIndex.axes.lookup(partition.getAxisNode());
    const target::lowering::RangeBinding *ownership =
        axis ? axis.getRange("ownership", 0) : nullptr;
    std::string part = programBlocks.lookup(partition.getAxisNode());
    std::string logical = axisDimensions.lookup(partition.getAxisNode());
    if (!axis || !ownership || part.empty() || logical.empty())
      return binding.emitOpError(
          "has no exact TileLang count-partition stream interval");
    std::string suffix = std::to_string(*node);
    partitionBegin = "stream_segment_begin_" + suffix;
    partitionEnd = "stream_segment_end_" + suffix;
    line(partitionBegin + " = T.min(" + addressIndex(part) + " * " +
         ownership->getTile().str() + ", " + logical + ")");
    line(partitionEnd + " = T.min(" + partitionBegin + " + " +
         ownership->getTile().str() + ", " + logical + ")");
    streamExtent =
        "T.max(0, " + partitionEnd + " - " + partitionBegin + ")";
    activeEnd = partitionEnd;
  }
  if (hasStop) {
    auto stopIndex =
        operation.getAttrOfType<IntegerAttr>("intent.stop_operand_index");
    int64_t expectedStop = operation.getNumResults() + 1 +
                           (hasRuntimeExtent ? 1 : 0);
    if (!stopIndex || stopIndex.getInt() != expectedStop ||
        !binding.getStopValueAttr())
      return operation.emitOpError(
          "does not match its planned logical stream stop");
    Value stopValue = operation.getOperand(stopIndex.getInt());
    auto valueID = kernel.valueIDs.find(stopValue);
    FailureOr<StringRef> stop = lookupValue(operation, stopIndex.getInt());
    if (valueID == kernel.valueIDs.end() ||
        valueID->second != binding.getStopValueAttr().getInt())
      return operation.emitOpError(
          "does not consume the planned logical stream stop value");
    if (failed(stop))
      return failure();
    if (partitionedStream) {
      activeEnd = "stream_effective_end_" + std::to_string(*node);
      line(activeEnd + " = T.min(" + stop->str() + ", " + partitionEnd + ")");
      streamExtent =
          "T.max(0, " + activeEnd + " - " + partitionBegin + ")";
    } else {
      streamExtent =
          "T.max(0, T.min(" + stop->str() + ", " + streamExtent + "))";
    }
  }
  if (partitionedStream) {
    auto previous = activeTraversalEnds.find(binding.getAxisNode());
    std::optional<std::string> restore;
    if (previous != activeTraversalEnds.end())
      restore = previous->second;
    streamEndRestores[&operation].emplace_back(binding.getAxisNode(),
                                                std::move(restore));
    activeTraversalEnds[binding.getAxisNode()] = activeEnd;
  }
  std::string streamTile = "stream_tile_" + std::to_string(*node);
  line("for " + streamTile + " in T.Pipelined(T.ceildiv(" + streamExtent +
       ", " + binding.getTile().str() +
       "), num_stages=num_stages):");
  ++indentation;
  std::string streamStart = "stream_axis_index_" + std::to_string(*node);
  line(streamStart + " = " +
       std::string(partitionedStream
                       ? partitionBegin + " + "
                       : raggedStream ? "sequence_begin_" + raggedSuffix + " + "
                                      : "") +
       addressIndex(streamTile) + " * " + binding.getTile().str());
  valueNames[body.getArgument(0)] = streamStart;
  regionIndices[body.getArgument(0)] = streamStart;
  activeTraversalIndices[binding.getAxisNode()].push_back(streamStart);
  auto bindScopedAxis = [&](int64_t axisNode, std::string value) {
    auto previous = axisIndices.find(axisNode);
    std::optional<std::string> restore;
    if (previous != axisIndices.end())
      restore = previous->second;
    streamAxisRestores[&operation].emplace_back(axisNode, std::move(restore));
    axisIndices[axisNode] = std::move(value);
  };
  bindScopedAxis(binding.getAxisNode(), streamStart);
  for (int64_t axisNode : binding.getInnerReductionAxes()) {
    if (axisNode == binding.getAxisNode())
      continue;
    plan::AxisOp axis = planIndex.axes.lookup(axisNode);
    const target::lowering::RangeBinding *range =
        axis ? axis.getRange("reduction", 0) : nullptr;
    if (!range)
      return binding.emitOpError("has no inner reduction range");
    bindScopedAxis(axisNode, "0");
  }
  return success();
}

LogicalResult ProgramMaterializer::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (carriers == streamCarriers.end() || failed(node) || !binding)
    return operation.emitOpError("has no active TileLang stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (::intent::target::semanticOperationName(terminator) != "intent.yield" ||
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
  auto active = activeTraversalIndices.find(binding.getAxisNode());
  if (active == activeTraversalIndices.end() || active->second.empty())
    return operation.emitOpError("has no active TileLang traversal projection");
  active->second.pop_back();
  if (active->second.empty())
    activeTraversalIndices.erase(active);
  if (auto restores = streamAxisRestores.find(&operation);
      restores != streamAxisRestores.end()) {
    for (auto value = restores->second.rbegin();
         value != restores->second.rend(); ++value) {
      if (value->second)
        axisIndices[value->first] = *value->second;
      else
        axisIndices.erase(value->first);
    }
    streamAxisRestores.erase(restores);
  }
  if (auto restores = streamEndRestores.find(&operation);
      restores != streamEndRestores.end()) {
    for (auto value = restores->second.rbegin();
         value != restores->second.rend(); ++value) {
      if (value->second)
        activeTraversalEnds[value->first] = *value->second;
      else
        activeTraversalEnds.erase(value->first);
    }
    streamEndRestores.erase(restores);
  }
  return success();
}

LogicalResult ProgramMaterializer::emitSparseContract(Operation &operation) {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "sparse-contract emission");
  intent::plan::SparseContractOp binding =
      succeeded(node) ? planIndex.sparseContracts.lookup(*node)
                      : intent::plan::SparseContractOp();
  if (failed(node) || !binding || binding.getFormat() != "two_of_four" ||
      operation.getNumOperands() != 3 || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a TileLang 2:4 sparse contraction binding");
  Operation *compressedLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *metadataLoad = deferredLoads.lookup(operation.getOperand(1));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(2));
  if (!compressedLoad || !metadataLoad || !rhsLoad)
    return operation.emitOpError(
        "requires three planned deferred sparse-matrix transfers");
  FailureOr<ABIView *> compressedView =
      lookupView(compressedLoad->getOperand(0), *compressedLoad);
  FailureOr<ABIView *> metadataView =
      lookupView(metadataLoad->getOperand(0), *metadataLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  plan::AxisOp rowAxis = planIndex.axes.lookup(binding.getRowAxisNode());
  plan::AxisOp columnAxis = planIndex.axes.lookup(binding.getColumnAxisNode());
  plan::AxisOp reductionAxis =
      planIndex.axes.lookup(binding.getReductionAxisNode());
  std::string rowStart = axisIndices.lookup(binding.getRowAxisNode());
  std::string columnStart = axisIndices.lookup(binding.getColumnAxisNode());
  std::string reductionExtent =
      axisDimensions.lookup(binding.getReductionAxisNode());
  if (failed(compressedView) || failed(metadataView) || failed(rhsView) ||
      !rowAxis || !columnAxis || !reductionAxis || rowStart.empty() ||
      columnStart.empty() || reductionExtent.empty() || rowAxis.getTile().empty() ||
      columnAxis.getTile().empty() || reductionAxis.getTile().empty())
    return operation.emitOpError(
        "has an incomplete planned 2:4 sparse-matrix projection");
  std::string compressed = makeResultName(*compressedLoad, 0) + "_shared";
  std::string metadata = makeResultName(*metadataLoad, 0) + "_shared";
  std::string rhs = makeResultName(*rhsLoad, 0) + "_shared";
  line(compressed + " = T.alloc_shared((" + rowAxis.getTile().str() + ", " +
       reductionAxis.getTile().str() + " // 2), " +
       dtypeName((*compressedView)->tensor.getElementType(), *compressedLoad) +
       ")");
  line(metadata + " = T.alloc_shared((" + rowAxis.getTile().str() + ", " +
       reductionAxis.getTile().str() + " // 16), " +
       dtypeName((*metadataView)->tensor.getElementType(), *metadataLoad) + ")");
  line(rhs + " = T.alloc_shared((" + reductionAxis.getTile().str() + ", " +
       columnAxis.getTile().str() + "), " +
       dtypeName((*rhsView)->tensor.getElementType(), *rhsLoad) + ")");
  FailureOr<std::string> accumulator =
      allocateResult(operation, 0, "fragment");
  if (failed(accumulator))
    return failure();
  line("T.clear(" + *accumulator + ")");
  line("for k_tile in T.Pipelined(T.ceildiv(" + reductionExtent + ", " +
       reductionAxis.getTile().str() + "), num_stages=num_stages):");
  ++indentation;
  line("T.copy(" + (*compressedView)->argument->name + "[" +
       addressIndex(rowStart) + ", " + addressIndex("k_tile") + " * " +
       reductionAxis.getTile().str() + " // 2], " + compressed + ")");
  line("T.copy(" + (*metadataView)->argument->name + "[" +
       addressIndex(rowStart) + ", " + addressIndex("k_tile") + " * " +
       reductionAxis.getTile().str() + " // 16], " + metadata + ")");
  line("T.copy(" + (*rhsView)->argument->name + "[" +
       addressIndex("k_tile") + " * " + reductionAxis.getTile().str() +
       ", " + addressIndex(columnStart) + "], " + rhs + ")");
  line("T.gemm_sp(" + compressed + ", " + metadata + ", " + rhs + ", " +
       *accumulator + ", policy=GEMM_WARP_POLICY)");
  --indentation;
  bindResult(operation, 0, *accumulator);
  return success();
}

LogicalResult ProgramMaterializer::emitContract(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "T.gemm" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a TileLang contraction binding");
  StringRef form = binding.getForm();
  FailureOr<target::lowering::ContractionOrientation> orientation =
      target::lowering::selectedContractionOrientation(binding);
  if (failed(orientation))
    return failure();
  bool carriedFlow = binding.getAccumulatorFlow() == "loop_carried";
  std::optional<int64_t> accumulatorOwner =
      binding.getAccumulatorOwnerNode();
  std::optional<int64_t> accumulatorUpdate =
      binding.getAccumulatorUpdateNode();
  std::optional<int64_t> accumulatorValue = binding.getAccumulatorValue();
  std::optional<int64_t> accumulatorConditional =
      binding.getAccumulatorConditionalNode();
  bool producerReplay = form == "replay";
  bool oneSidedDeferredOperand = form == "deferred_one";
  bool reuseCarriedAccumulator =
      carriedFlow &&
      (accumulatorConditional || producerReplay ||
       (accumulatorOwner && planIndex.streams.count(*accumulatorOwner) &&
        oneSidedDeferredOperand));
  std::optional<std::string> carriedAccumulator;
  if (reuseCarriedAccumulator) {
    if (!accumulatorUpdate || !accumulatorValue)
      return operation.emitOpError(
          "has an incomplete planned loop-carried accumulator binding");
    Value previous = kernel.values.lookup(*accumulatorValue);
    Operation *update = kernel.nodes.lookup(*accumulatorUpdate);
    auto previousName = valueNames.find(previous);
    if (!previous || !update || previousName == valueNames.end())
      return operation.emitOpError(
          "has no emitted TileLang loop-carried contraction accumulator");
    carriedAccumulator = previousName->second;
    inPlacePointwiseResults[update] = *carriedAccumulator;
  }
  auto replay = deferredContractReplays.find(&operation);
  if (form == "replay") {
    if (replay == deferredContractReplays.end())
      return operation.emitOpError(
          "TileLang replay form has no physical producer slice");
    if ((binding.getLhsSpace() != "shared" &&
         binding.getLhsSpace() != "fragment") ||
        (binding.getRhsSpace() != "shared" &&
         binding.getRhsSpace() != "fragment") ||
        binding.getAccumulatorSpace() != "fragment")
      return operation.emitOpError(
          "producer replay has inconsistent TileLang plan spaces");
    std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
    plan::AxisOp reductionAxis =
        reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
    FailureOr<std::string> result =
        carriedAccumulator
            ? FailureOr<std::string>(*carriedAccumulator)
            : allocateResult(operation, 0, "fragment");
    if (!reductionAxis || failed(result))
      return operation.emitOpError(
          "replay form has no selected physical reduction axis");
    bool streamReduction = target::lowering::isEnclosingStreamReductionAxis(
        planIndex, operation, reductionAxis.getNode());
    if (!carriedAccumulator)
      line("T.clear(" + *result + ")");
    if (!streamReduction) {
      line("for k_tile in T.Pipelined(T.ceildiv(" +
           roleDimensions.lookup(reductionAxis.getRole()) + ", " +
           reductionAxis.getTile().str() +
           "), num_stages=num_stages):");
      ++indentation;
      axisIndices[reductionAxis.getNode()] =
          addressIndex("k_tile") + " * " + reductionAxis.getTile().str();
    } else if (axisIndices.lookup(reductionAxis.getNode()).empty()) {
      return operation.emitOpError(
          "has no active TileLang stream-bound reduction range");
    }
    llvm::DenseMap<Value, std::optional<std::string>> savedValues;
    auto saveValues = [&](ArrayRef<Operation *> producers) {
      for (Operation *producer : producers)
        for (Value value : producer->getResults())
          if (!savedValues.count(value)) {
            auto found = valueNames.find(value);
            savedValues[value] =
                found == valueNames.end()
                    ? std::nullopt
                    : std::optional<std::string>(found->second);
          }
    };
    saveValues(replay->second.lhsReplayProducers);
    saveValues(replay->second.rhsReplayProducers);
    auto restoreValues = [&]() {
      for (const auto &entry : savedValues)
        if (entry.second)
          valueNames[entry.first] = *entry.second;
        else
          valueNames.erase(entry.first);
    };
    activeDeferredContract = &operation;
    LogicalResult lhsReplay =
        replayContractProducers(replay->second.lhsReplayProducers);
    LogicalResult rhsReplay = succeeded(lhsReplay)
                                  ? replayContractProducers(
                                        replay->second.rhsReplayProducers)
                                  : failure();
    activeDeferredContract = nullptr;
    if (failed(lhsReplay) || failed(rhsReplay)) {
      restoreValues();
      return failure();
    }
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    if (failed(lhs) || failed(rhs)) {
      restoreValues();
      return failure();
    }
    std::string call =
        "T.gemm(" + lhs->str() + ", " + rhs->str() + ", " + *result;
    if (orientation->lhsTranspose)
      call += ", transpose_A=True";
    if (orientation->rhsTranspose)
      call += ", transpose_B=True";
    line(call + ", policy=GEMM_WARP_POLICY)");
    restoreValues();
    if (!streamReduction)
      --indentation;
    bindResult(operation, 0, *result);
    return success();
  }

  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (form == "deferred_one" || form == "deferred_two") {
    if (form == "deferred_one") {
      if (static_cast<bool>(lhsLoad) == static_cast<bool>(rhsLoad))
        return operation.emitOpError(
            "TileLang one-sided deferred form does not own exactly one deferred operand");
      Operation *load = lhsLoad ? lhsLoad : rhsLoad;
      unsigned loadOperand = lhsLoad ? 0 : 1;
      unsigned directOperand = lhsLoad ? 1 : 0;
      StringRef loadSpace = lhsLoad ? binding.getLhsSpace()
                                    : binding.getRhsSpace();
      StringRef directSpace = lhsLoad ? binding.getRhsSpace()
                                      : binding.getLhsSpace();
      if (loadSpace != "shared" || directSpace != "fragment" ||
          binding.getAccumulatorSpace() != "fragment")
        return operation.emitOpError(
            "one-sided deferred TileLang contraction has inconsistent plan "
            "spaces");
      std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
      plan::AxisOp reductionAxis =
          reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
      if (!reductionAxis)
        return operation.emitOpError(
            "one-sided deferred form has no selected physical reduction axis");
      if (!target::lowering::isEnclosingStreamReductionAxis(
              planIndex, operation, reductionAxis.getNode()))
        return operation.emitOpError(
            "one-sided deferred TileLang contraction requires an active "
            "stream-bound reduction range");
      if (axisIndices.lookup(reductionAxis.getNode()).empty())
        return operation.emitOpError(
            "has no active TileLang stream-bound reduction range");
      FailureOr<int64_t> loadNode =
          target::getNodeID(*load, "TileLang contraction transfer");
      plan::BoundaryOp boundary =
          succeeded(loadNode) ? planIndex.boundaries.lookup(*loadNode)
                              : plan::BoundaryOp();
      FailureOr<ABIView *> view = lookupView(load->getOperand(0), *load);
      FailureOr<std::string> indices = accessIndices(*load);
      FailureOr<std::string> shape = tensorShape(*load, 0);
      FailureOr<StringRef> direct = lookupValue(operation, directOperand);
      FailureOr<std::string> result =
          carriedAccumulator
              ? FailureOr<std::string>(*carriedAccumulator)
              : allocateResult(operation, 0, "fragment");
      if (failed(loadNode) || !boundary || failed(view) || failed(indices) ||
          failed(shape) || failed(direct) || failed(result))
        return failure();
      if (boundary.hasDataDependentTensorIndex())
        return operation.emitOpError(
            "TileLang cannot bulk-copy a stream contraction operand with a "
            "noncontiguous index tile");
      std::string shared = makeResultName(*load, 0) + "_shared";
      line(shared + " = T.alloc_shared(" + *shape + ", " +
           dtypeName((*view)->tensor.getElementType(), *load) + ")");
      line("T.copy(" + (*view)->argument->name + "[" + *indices + "], " +
           shared + ")");
      if (!carriedAccumulator)
        line("T.clear(" + *result + ")");
      std::string lhs = loadOperand == 0 ? shared : direct->str();
      std::string rhs = loadOperand == 1 ? shared : direct->str();
      std::string call = "T.gemm(" + lhs + ", " + rhs + ", " + *result;
      if (orientation->lhsTranspose)
        call += ", transpose_A=True";
      if (orientation->rhsTranspose)
        call += ", transpose_B=True";
      line(call + ", policy=GEMM_WARP_POLICY)");
      bindResult(operation, 0, *result);
      return success();
    }
    if (!lhsLoad || !rhsLoad)
      return operation.emitOpError(
          "TileLang two-sided deferred form does not own both deferred operands");
    if (binding.getLhsSpace() != "shared" ||
        binding.getRhsSpace() != "shared" ||
        binding.getAccumulatorSpace() != "fragment")
      return operation.emitOpError(
          "deferred TileLang contraction has inconsistent plan spaces");
    FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
    FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
    std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
    std::optional<int64_t> lhsResultNode = binding.getLhsResultAxisNode();
    std::optional<int64_t> rhsResultNode = binding.getRhsResultAxisNode();
    plan::AxisOp reductionAxis =
        reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
    plan::AxisOp lhsResult =
        lhsResultNode ? planIndex.axes.lookup(*lhsResultNode) : plan::AxisOp();
    plan::AxisOp rhsResult =
        rhsResultNode ? planIndex.axes.lookup(*rhsResultNode) : plan::AxisOp();
    auto operandElementType = [](Value value) -> Type {
      auto tensor = dyn_cast<RankedTensorType>(value.getType());
      return tensor ? tensor.getElementType() : Type();
    };
    Type lhsElement = operandElementType(operation.getOperand(0));
    Type rhsElement = operandElementType(operation.getOperand(1));
    bool fp8Operands =
        lhsElement && rhsElement &&
        isa<Float8E4M3FNType, Float8E5M2Type>(lhsElement) &&
        isa<Float8E4M3FNType, Float8E5M2Type>(rhsElement);
    if (!reductionAxis || !lhsResult || !rhsResult)
      return operation.emitOpError(
          "two-sided deferred form has incomplete selected physical axes");
    if (raggedRuntimesByAxis.count(reductionAxis.getNode()))
      return operation.emitOpError(
          "TileLang has no mechanical masked bulk-copy projection for a "
          "ragged contraction reduction axis");
    bool streamReduction = target::lowering::isEnclosingStreamReductionAxis(
        planIndex, operation, reductionAxis.getNode());
    if (fp8Operands && lhsResult.hasRole("lane"))
      return operation.emitOpError(
          "TileLang 0.1.13 cannot project an FP8 contraction whose matrix-M "
          "axis is a runtime lane extent to a supported MMA primitive");
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
    if (!streamReduction)
      axisIndices[reductionAxis.getNode()] =
          addressIndex("k_tile") + " * " + reductionAxis.getTile().str();
    else if (axisIndices.lookup(reductionAxis.getNode()).empty())
      return operation.emitOpError(
          "has no active TileLang stream-bound reduction range");
    FailureOr<std::string> lhsIndices = accessIndices(*lhsLoad);
    FailureOr<std::string> rhsIndices = accessIndices(*rhsLoad);
    FailureOr<std::string> lhsShape = tensorShape(*lhsLoad, 0);
    FailureOr<std::string> rhsShape = tensorShape(*rhsLoad, 0);
    FailureOr<std::string> result =
        carriedAccumulator
            ? FailureOr<std::string>(*carriedAccumulator)
            : allocateResult(operation, 0, "fragment");
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
    if (!carriedAccumulator)
      line("T.clear(" + *result + ")");
    if (!streamReduction) {
      line("for k_tile in T.Pipelined(T.ceildiv(" +
           roleDimensions.lookup(reductionAxis.getRole()) + ", " +
           reductionAxis.getTile().str() + "), num_stages=num_stages):");
      ++indentation;
    }
    line("T.copy(" + (*lhsView)->argument->name + "[" + *lhsIndices + "], " +
         lhs + ")");
    line("T.copy(" + (*rhsView)->argument->name + "[" + *rhsIndices + "], " +
         rhs + ")");
    std::string call = "T.gemm(" + lhs + ", " + rhs + ", " + *result;
    if (orientation->lhsTranspose)
      call += ", transpose_A=True";
    if (orientation->rhsTranspose)
      call += ", transpose_B=True";
    line(call + ", policy=GEMM_WARP_POLICY)");
    if (!streamReduction)
      --indentation;
    bindResult(operation, 0, *result);
    return success();
  }

  if (form != "direct" || lhsLoad || rhsLoad)
    return operation.emitOpError("has an inconsistent TileLang contraction form");

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
  auto isolateRepeatedContractionOperand =
      [&](Value operand, StringRef emitted,
          unsigned operandNumber) -> FailureOr<std::string> {
    auto isolate = binding.operation->getAttrOfType<BoolAttr>(
        operandNumber == 0 ? isolateLhsAttr : isolateRhsAttr);
    if (!isolate)
      return binding.emitOpError(
          "has no realized TileLang contraction-operand form");
    if (!isolate.getValue())
      return emitted.str();
    auto tensor = dyn_cast<RankedTensorType>(operand.getType());
    FailureOr<SmallVector<std::string>> extents = valueExtents(operand);
    if (!tensor || failed(extents) || extents->empty())
      return operation.emitOpError(
          "cannot isolate a repeated TileLang contraction operand");
    std::string dtype = dtypeName(tensor.getElementType(), operation);
    if (dtype.empty())
      return failure();
    std::string isolated = makeResultName(operation, 0) + "_operand_" +
                           std::to_string(operandNumber);
    std::string shape = "(";
    for (auto [axis, extent] : llvm::enumerate(*extents)) {
      if (axis)
        shape += ", ";
      shape += extent;
    }
    if (extents->size() == 1)
      shape += ",";
    shape += ")";
    line(isolated + " = T.alloc_shared(" + shape + ", " + dtype + ")");
    line("T.copy(" + emitted.str() + ", " + isolated + ")");
    line("T.sync_threads()");
    return isolated;
  };
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
  FailureOr<std::string> result =
      carriedAccumulator ? FailureOr<std::string>(*carriedAccumulator)
                         : allocateResult(operation, 0, "fragment");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  FailureOr<std::string> isolatedLhs =
      isolateRepeatedContractionOperand(operation.getOperand(0), *lhs, 0);
  FailureOr<std::string> isolatedRhs =
      isolateRepeatedContractionOperand(operation.getOperand(1), *rhs, 1);
  if (failed(isolatedLhs) || failed(isolatedRhs))
    return failure();
  if (!carriedAccumulator)
    line("T.clear(" + *result + ")");
  std::string call =
      "T.gemm(" + *isolatedLhs + ", " + *isolatedRhs + ", " + *result;
  if (orientation->lhsTranspose)
    call += ", transpose_A=True";
  if (orientation->rhsTranspose)
    call += ", transpose_B=True";
  line(call + ", policy=GEMM_WARP_POLICY)");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitScaledContract(Operation &operation) {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "scaled-contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding ||
      binding.getLowering() != "T.scaled_gemm_fallback" ||
      operation.getNumOperands() != 4 || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a TileLang scaled-contraction binding");
  if (binding.getForm() != "scaled_direct")
    return operation.emitOpError(
        "has no realized direct TileLang scaled-contraction form");
  for (Value operand : operation.getOperands())
    if (deferredLoads.count(operand))
      return operation.emitOpError(
          "TileLang scaled contraction requires materialized operand tiles");
  auto valueExtents = [&](Value value)
      -> FailureOr<SmallVector<std::string>> {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return failure();
    return tensorExtents(*result.getOwner(), result.getResultNumber());
  };
  FailureOr<SmallVector<std::string>> lhsExtents =
      valueExtents(operation.getOperand(0));
  FailureOr<SmallVector<std::string>> rhsExtents =
      valueExtents(operation.getOperand(1));
  auto lhsGroup =
      operation.getAttrOfType<IntegerAttr>("intent.lhs_group_size");
  auto rhsGroup =
      operation.getAttrOfType<IntegerAttr>("intent.rhs_group_size");
  StringRef layout = binding.getScaledLayout();
  if (layout == "flattened_rank_three" && succeeded(lhsExtents) &&
      succeeded(rhsExtents) &&
      lhsExtents->size() == 3 && rhsExtents->size() == 3 && lhsGroup &&
      rhsGroup && lhsGroup.getInt() > 0 && rhsGroup.getInt() > 0) {
    std::string resultName = makeResultName(operation, 0);
    std::string scaledLhs = resultName + "_scaled_lhs";
    std::string scaledRhs = resultName + "_scaled_rhs";
    std::string lhsReduction = (*lhsExtents)[1] + " * " + (*lhsExtents)[2];
    std::string rhsReduction = (*rhsExtents)[0] + " * " + (*rhsExtents)[1];
    line(scaledLhs + " = T.alloc_fragment((" + (*lhsExtents)[0] + ", " +
         lhsReduction + "), T.float32)");
    line("for scaled_i, scaled_g, scaled_k in T.Parallel(" +
         (*lhsExtents)[0] + ", " + (*lhsExtents)[1] + ", " +
         (*lhsExtents)[2] + "):");
    ++indentation;
    FailureOr<std::string> lhs = tensorElement(
        operation.getOperand(0), {"scaled_i", "scaled_g", "scaled_k"},
        operation);
    FailureOr<std::string> lhsScale = tensorElement(
        operation.getOperand(2), {"scaled_i", "scaled_g"}, operation);
    if (failed(lhs) || failed(lhsScale))
      return failure();
    line(scaledLhs + "[scaled_i, scaled_g * " + (*lhsExtents)[2] +
         " + scaled_k] = " +
         syntax::cast(*lhs, "T.float32", false, true) + " * " +
         syntax::cast(*lhsScale, "T.float32", true, true));
    --indentation;
    line(scaledRhs + " = T.alloc_fragment((" + rhsReduction + ", " +
         (*rhsExtents)[2] + "), T.float32)");
    line("for scaled_g, scaled_k, scaled_j in T.Parallel(" +
         (*rhsExtents)[0] + ", " + (*rhsExtents)[1] + ", " +
         (*rhsExtents)[2] + "):");
    ++indentation;
    FailureOr<std::string> rhs = tensorElement(
        operation.getOperand(1), {"scaled_g", "scaled_k", "scaled_j"},
        operation);
    FailureOr<std::string> rhsScale = tensorElement(
        operation.getOperand(3), {"scaled_g", "scaled_j"}, operation);
    if (failed(rhs) || failed(rhsScale))
      return failure();
    line(scaledRhs + "[scaled_g * " + (*rhsExtents)[1] +
         " + scaled_k, scaled_j] = " +
         syntax::cast(*rhs, "T.float32", false, true) + " * " +
         syntax::cast(*rhsScale, "T.float32", true, true));
    --indentation;
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
    if (failed(result))
      return failure();
    line("T.clear(" + *result + ")");
    line("T.gemm(" + scaledLhs + ", " + scaledRhs + ", " + *result +
         ", policy=GEMM_WARP_POLICY)");
    bindResult(operation, 0, *result);
    return success();
  }
  if (layout != "grouped_rank_two" || failed(lhsExtents) ||
      failed(rhsExtents) || lhsExtents->size() != 2 ||
      rhsExtents->size() != 2 || !lhsGroup || !rhsGroup ||
      lhsGroup.getInt() <= 0 || rhsGroup.getInt() <= 0)
    return operation.emitOpError(
        "has no rank-two TileLang scaled-contraction projection");
  std::string resultName = makeResultName(operation, 0);
  std::string scaledLhs = resultName + "_scaled_lhs";
  std::string scaledRhs = resultName + "_scaled_rhs";
  line(scaledLhs + " = T.alloc_fragment((" + (*lhsExtents)[0] + ", " +
       (*lhsExtents)[1] + "), T.float32)");
  line("for scaled_i, scaled_k in T.Parallel(" + (*lhsExtents)[0] + ", " +
       (*lhsExtents)[1] + "):");
  ++indentation;
  FailureOr<std::string> lhs = tensorElement(
      operation.getOperand(0), {"scaled_i", "scaled_k"}, operation);
  FailureOr<std::string> lhsScale = tensorElement(
      operation.getOperand(2),
      {"scaled_i", "scaled_k // " + std::to_string(lhsGroup.getInt())},
      operation);
  if (failed(lhs) || failed(lhsScale))
    return failure();
  line(scaledLhs + "[scaled_i, scaled_k] = " +
       syntax::cast(*lhs, "T.float32", false, true) + " * " +
       syntax::cast(*lhsScale, "T.float32", true, true));
  --indentation;
  line(scaledRhs + " = T.alloc_fragment((" + (*rhsExtents)[0] + ", " +
       (*rhsExtents)[1] + "), T.float32)");
  line("for scaled_k, scaled_j in T.Parallel(" + (*rhsExtents)[0] + ", " +
       (*rhsExtents)[1] + "):");
  ++indentation;
  FailureOr<std::string> rhs = tensorElement(
      operation.getOperand(1), {"scaled_k", "scaled_j"}, operation);
  FailureOr<std::string> rhsScale = tensorElement(
      operation.getOperand(3),
      {"scaled_k // " + std::to_string(rhsGroup.getInt()), "scaled_j"},
      operation);
  if (failed(rhs) || failed(rhsScale))
    return failure();
  line(scaledRhs + "[scaled_k, scaled_j] = " +
       syntax::cast(*rhs, "T.float32", false, true) + " * " +
       syntax::cast(*rhsScale, "T.float32", true, true));
  --indentation;
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(result))
    return failure();
  line("T.clear(" + *result + ")");
  line("T.gemm(" + scaledLhs + ", " + scaledRhs + ", " + *result +
       ", policy=GEMM_WARP_POLICY)");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult ProgramMaterializer::emitStore(Operation &operation) {
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
        !target::lowering::hasPackedScalarDomain(planIndex, boundary);
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
    FailureOr<SmallVector<target::IndexTerm>> relation =
        boundary.getCheckBounds()
            ? target::parseIndexRelation(operation)
            : FailureOr<SmallVector<target::IndexTerm>>(
                  SmallVector<target::IndexTerm>());
    bool tensorIndexed =
        succeeded(relation) && llvm::any_of(*relation, [&](const auto &term) {
          return term.kind == "value_index" && term.operands.size() == 1 &&
                 term.operands.front() &&
                 isa<RankedTensorType>(
                     operation.getOperand(*term.operands.front()).getType());
        });
    bool mayUseBulkFastPath =
        boundary.getCheckBounds() && !tensorIndexed &&
        boundary.getTensorIndexing() == "none";
    FailureOr<std::string> tileFits =
        mayUseBulkFastPath
            ? tileFitsViewPredicate(operation, *extents)
            : FailureOr<std::string>(std::string());
    FailureOr<std::string> bulkIndices =
        mayUseBulkFastPath
            ? accessIndices(operation)
            : FailureOr<std::string>(std::string());
    if (failed(logicalPredicate) || failed(physicalPredicate) ||
        failed(relation) || failed(tileFits) || failed(bulkIndices))
      return failure();
    bool hasBulkFastPath =
        !expanded && !tileFits->empty() && !bulkIndices->empty();
    if (hasBulkFastPath) {
      line("if " + *tileFits + ":");
      ++indentation;
      line("T.copy(" + stored->str() + ", " + (*view)->argument->name + "[" +
           *bulkIndices + "])");
      --indentation;
      line("else:");
      ++indentation;
    }
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
    if (hasBulkFastPath)
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

LogicalResult ProgramMaterializer::emitUniqueStore(Operation &operation) {
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
  if (binding.getTensorIndexing() == "structured") {
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

LogicalResult ProgramMaterializer::emitAtomic(Operation &operation) {
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  Operation *deferredValue =
      valueIndex ? deferredLoads.lookup(operation.getOperand(valueIndex.getInt()))
                 : nullptr;
  FailureOr<StringRef> stored =
      deferredValue ? FailureOr<StringRef>(StringRef())
      : valueIndex  ? lookupValue(operation, valueIndex.getInt())
                    : FailureOr<StringRef>(failure());
  if (!valueIndex || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical TileLang atomic merge");
  Value storedValue = operation.getOperand(valueIndex.getInt());
  {
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
          !target::lowering::hasPackedScalarDomain(planIndex, boundary);
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
}

} // namespace intent::tilelang::lowering

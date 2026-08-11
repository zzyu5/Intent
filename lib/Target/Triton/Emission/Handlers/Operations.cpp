#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::triton::emission {
namespace {

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter,
                         target::OperationCallback leave = {}) {
  return registry.add(name,
                      target::OperationHandler{std::move(enter), std::move(leave)});
}

StringRef tritonDtype(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (type.isF16())
    return "tl.float16";
  if (type.isF32())
    return "tl.float32";
  if (type.isBF16())
    return "tl.bfloat16";
  if (type.isInteger(8))
    return "tl.int8";
  if (type.isInteger(32))
    return "tl.int32";
  if (isa<IndexType>(type))
    return "tl.int64";
  return {};
}

} // namespace

LogicalResult registerEmissionHandlers(target::OperationHandlerRegistry &registry,
                                       SourceEmitter &emitter) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.domain", "intent.domain_product",
                         "intent.region_end",
                         "intent.assume_in_bounds", "intent.partition", "intent.return", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();
  if (failed(addHandler(registry, "intent.dim", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitDimension(op);
      })) ||
      failed(addHandler(registry, "intent.constant",
                        [&](Operation &op) {
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
      failed(addHandler(registry, "intent.view_load",
                        [&](Operation &op) {
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
      failed(addHandler(registry, "intent.reduce",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitReduction(op);
                        })) ||
      failed(addHandler(registry, "intent.arg_reduce",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitReduction(op);
                        })) ||
      failed(addHandler(registry, "intent.scan", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitScan(op);
      })) ||
      failed(addHandler(registry, "intent.broadcast",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitBroadcast(op);
                        })) ||
      failed(addHandler(registry, "intent.unary",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitUnary(op);
                        })) ||
      failed(addHandler(registry, "intent.binary",
                        [&](Operation &op) {
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
      failed(addHandler(registry, "intent.cast",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitCast(op);
                        })) ||
      failed(addHandler(registry, "intent.reshape",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitReshape(op);
                        })) ||
      failed(addHandler(registry, "intent.transpose",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitTranspose(op);
                        })) ||
      failed(addHandler(registry, "intent.full",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitFull(op);
                        })) ||
      failed(addHandler(registry, "intent.zeros",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitZeros(op);
                        })) ||
      failed(addHandler(registry, "intent.members", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitMembers(op);
      })) ||
      failed(addHandler(registry, "intent.gather",
                        [&](Operation &op) {
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
      failed(addHandler(registry, "intent.contract",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitContract(op);
                        })) ||
      failed(addHandler(registry, "intent.view_store",
                        [&](Operation &op) {
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
      failed(addHandler(registry, "intent.atomic_cas", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitAtomicCas(op);
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
      expression = number < 0 ? "-float('inf')" : "float('inf')";
    else if (std::isnan(number))
      expression = "float('nan')";
    else
      expression = std::to_string(number);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported Triton constant value");
  }
  line(result + " = " + expression);
  bindResult(operation, 0, result);
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
    return operation.emitOpError("lacks a mechanical Triton dimension binding");
  std::string dimension = (*view)->shape[axis.getInt()];
  if (!planIndex.components.reusedAxes.empty()) {
    if (dimension == roleDimensions.lookup("program_0"))
      dimension = "n_rows";
    else if (dimension == roleDimensions.lookup("lane_0"))
      dimension = "n_cols";
  }
  bindResult(operation, 0, dimension);
  return success();
}

LogicalResult SourceEmitter::emitProgramBindings() {
  if (programBindingsEmitted)
    return success();
  programBindingsEmitted = true;

  auto axisExtent = [&](plan::AxisOp axis) {
    std::string role =
        "program_" + std::to_string(axis.getProgramOrder());
    std::string extent = roleDimensions.lookup(role);
    return axis.isScalar()
               ? extent
               : "tl.cdiv(" + extent + ", " + axis.getTile().str() + ")";
  };
  bool persistent = planIndex.program.getPersistent();
  std::string linear = "persistent_program";
  if (persistent) {
    line("total_program_tiles = " +
         target::emission::projectProgramVolume(planIndex, axisExtent));
    line("program_start = " + addressIndex("tl.program_id(0)"));
    line("program_step = " + addressIndex("tl.num_programs(0)"));
    line("for " + linear +
         " in tl.range(program_start, total_program_tiles, program_step):");
    ++indentation;
  }

  for (const auto &entry : planIndex.components.groups) {
    SmallVector<plan::AxisOp> axes(entry.getValue().begin(), entry.getValue().end());
    llvm::sort(axes, [](plan::AxisOp lhs, plan::AxisOp rhs) {
      return lhs.getProgramOrder() < rhs.getProgramOrder();
    });
    if (axes.size() != 2 ||
        axes.front().getWorkerAxis() != axes.back().getWorkerAxis())
      return axes.front().emitOpError(
          "Triton supports two-axis program grouping on one worker axis");
    StringRef group = axes.front().getGroupSpelling();
    if (group.empty())
      return axes.front().emitOpError("has no Triton group spelling");
    plan::AxisOp lhs = axes[0];
    plan::AxisOp rhs = axes[1];
    std::string lhsRole =
        "program_" + std::to_string(lhs.getProgramOrder());
    std::string rhsRole =
        "program_" + std::to_string(rhs.getProgramOrder());
    std::string pid = "group_pid_" + std::to_string(lhs.getNode());
    std::string lhsCount = "group_count_" + std::to_string(lhs.getNode());
    std::string rhsCount = "group_count_" + std::to_string(rhs.getNode());
    std::string groupSpan = "group_span_" + std::to_string(lhs.getNode());
    std::string groupID = "group_id_" + std::to_string(lhs.getNode());
    std::string first = "group_first_" + std::to_string(lhs.getNode());
    std::string size = "group_size_" + std::to_string(lhs.getNode());
    std::string lhsBlock = "block_axis_" + std::to_string(lhs.getNode());
    std::string rhsBlock = "block_axis_" + std::to_string(rhs.getNode());
    if (persistent) {
      FailureOr<std::string> groupIndex =
          target::emission::projectLinearGroupIndex(
              planIndex, axes, axisExtent, linear,
              *lhs.operation.getOperation());
      if (failed(groupIndex))
        return failure();
      line(pid + " = " + *groupIndex);
    } else {
      line(pid + " = " + addressIndex("tl.program_id(axis=" +
                                        std::to_string(lhs.getWorkerAxis()) +
                                        ")"));
    }
    line(lhsCount + " = " +
         addressIndex("tl.cdiv(" + roleDimensions.lookup(lhsRole) + ", " +
                      lhs.getTile().str() + ")"));
    line(rhsCount + " = " +
         addressIndex("tl.cdiv(" + roleDimensions.lookup(rhsRole) + ", " +
                      rhs.getTile().str() + ")"));
    line(groupSpan + " = " + group.str() + " * " + rhsCount);
    line(groupID + " = " + pid + " // " + groupSpan);
    line(first + " = " + groupID + " * " + group.str());
    line(size + " = min(" + lhsCount + " - " + first + ", " + group.str() +
         ")");
    line(lhsBlock + " = " + first + " + ((" + pid + " % " + groupSpan +
         ") % " + size + ")");
    line(rhsBlock + " = (" + pid + " % " + groupSpan + ") // " + size);
    programBlocks[lhs.getNode()] = lhsBlock;
    programBlocks[rhs.getNode()] = rhsBlock;
    axisIndices[lhs.getNode()] = lhsBlock + " * " + lhs.getTile().str() +
                                 " + tl.arange(0, " + lhs.getTile().str() + ")";
    axisIndices[rhs.getNode()] = rhsBlock + " * " + rhs.getTile().str() +
                                 " + tl.arange(0, " + rhs.getTile().str() + ")";
    line("axis_index_" + std::to_string(lhs.getNode()) + " = " +
         axisIndices.lookup(lhs.getNode()));
    line("axis_index_" + std::to_string(rhs.getNode()) + " = " +
         axisIndices.lookup(rhs.getNode()));
    axisIndices[lhs.getNode()] = "axis_index_" + std::to_string(lhs.getNode());
    axisIndices[rhs.getNode()] = "axis_index_" + std::to_string(rhs.getNode());
  }

  SmallVector<target::emission::ProgramIndexProjection> projections =
      persistent
          ? target::emission::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::emission::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("tl.program_id(axis=" +
                                      std::to_string(worker) + ")");
                });
  for (const target::emission::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    line(block + " = " + projection.expression);
    programBlocks[axis.getNode()] = block;
    if (axis.isScalar()) {
      axisIndices[axis.getNode()] = block;
      continue;
    }
    std::string value = "axis_index_" + std::to_string(axis.getNode());
    if (planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode())) {
      FailureOr<plan::RaggedOp> relation =
          target::emission::uniqueRaggedRelation(planIndex, axis.getNode(),
                                                  *axis.operation.getOperation());
      std::string outer = succeeded(relation)
                              ? axisIndices.lookup(relation->getOuterNode())
                              : std::string();
      auto runtime = succeeded(relation)
                         ? raggedRuntimeByRelation.find(relation->getNode())
                         : raggedRuntimeByRelation.end();
      if (failed(relation) || outer.empty() ||
          runtime == raggedRuntimeByRelation.end())
        return axis.emitOpError(
            "ordered ragged axis has no outer-axis or offsets binding");
      ABIView *offsets = raggedRuntimes[runtime->second].offsets;
      auto ordered =
          planIndex.components.orderedAxesByRelation.find(relation->getNode());
      if (ordered == planIndex.components.orderedAxesByRelation.end() ||
          ordered->second.empty())
        return axis.emitOpError("has no ordered axis in its ragged relation");
      for (int64_t orderedAxis : ordered->second) {
        std::string suffix = std::to_string(orderedAxis);
        line("sequence_begin_" + suffix + " = tl.load(" + offsets->pointer +
             " + " + addressIndex(outer) + " * " +
             addressIndex(offsets->strides[0]) + ")");
        line("sequence_end_" + suffix + " = tl.load(" + offsets->pointer +
             " + " + addressIndex(outer + " + 1") + " * " +
             addressIndex(offsets->strides[0]) + ")");
        line("sequence_length_" + suffix + " = sequence_end_" + suffix +
             " - sequence_begin_" + suffix);
      }
      std::string suffix = std::to_string(ordered->second.front());
      line(value + " = sequence_begin_" + suffix + " + " + block + " * " +
           axis.getTile().str() + " + tl.arange(0, " + axis.getTile().str() +
           ")");
    } else {
      line(value + " = " + block + " * " + axis.getTile().str() +
           " + tl.arange(0, " + axis.getTile().str() + ")");
    }
    axisIndices[axis.getNode()] = value;
  }
  for (const auto &entry : planIndex.axesByRole) {
    plan::AxisOp axis = entry.getValue();
    if (!entry.getKey().starts_with("lane_") ||
        !axisIndices.lookup(axis.getNode()).empty())
      continue;
    std::string extent = roleDimensions.lookup(entry.getKey());
    if (extent.empty())
      return axis.emitOpError("has no Triton lane extent");
    std::string value = "axis_index_" + std::to_string(axis.getNode());
    line(value + " = tl.arange(0, " + physicalExtent(extent) + ")");
    axisIndices[axis.getNode()] = value;
  }
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() == 0)
    return operation.emitOpError("parallel ownership requires region arguments");
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
          "staged Triton ownership requires one logical axis");
    BlockArgument argument = body.getArgument(0);
    plan::AxisOp axis = axes.front();
    bool outer = false;
    bool member = false;
    for (const auto &entry : stageRaggedRuntime) {
      const RaggedRuntime &runtime = raggedRuntimes[entry.second];
      outer |= runtime.binding.getOuterNode() == axis.getNode();
      member |= llvm::any_of(runtime.ownedMembers, [&](Operation *candidate) {
        auto node = candidate->getAttrOfType<IntegerAttr>("intent.node");
        return node && node.getInt() == axis.getNode();
      });
    }
    if (outer == member)
      return operation.emitOpError("has no staged program-axis binding");
    valueNames[argument] = outer ? "expert" : "member_offsets";
    return success();
  }

  if (!planIndex.components.reusedAxes.empty()) {
    if (axes.size() != 1)
      return operation.emitOpError(
          "worker-reused Triton ownership requires one logical axis");
    BlockArgument argument = body.getArgument(0);
    plan::AxisOp axis = axes.front();
    if (&operation != programRoot ||
        axis.getNode() != planIndex.components.reusedAxes.front().getNode())
      return operation.emitOpError("is not the worker-reused program axis");
    int64_t workerAxis = axis.getWorkerAxis();
    line("program_start = " + addressIndex("tl.program_id(" +
                                            std::to_string(workerAxis) + ")"));
    line("program_step = " + addressIndex("tl.num_programs(" +
                                           std::to_string(workerAxis) + ")"));
    line("for " + programIndex +
         " in tl.range(program_start, n_rows, program_step, "
         "num_stages=num_stages):");
    ++indentation;
    line(vectorIndex + " = tl.arange(0, BLOCK_SIZE)");
    axisIndices[axis.getNode()] = programIndex;
    valueNames[argument] = programIndex;
    return success();
  }

  if (&operation == programRoot && failed(emitProgramBindings()))
    return failure();
  for (auto [argument, axis] : llvm::zip(body.getArguments(), axes)) {
    std::string value = axisIndices.lookup(axis.getNode());
    if (value.empty())
      return axis.emitOpError(planIndex.program.getPersistent()
                                  ? "has no persistent Triton program index"
                                  : "has no emitted per-axis program index");
    valueNames[argument] = value;
  }
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &operation) {
  if (&operation == programRoot &&
      (!planIndex.components.reusedAxes.empty() ||
       planIndex.program.getPersistent()))
    --indentation;
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
    return operation.emitOpError("lacks a mechanical Triton sequential loop");
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
      return operation.emitOpError("has no emitted Triton loop bound");
    stop = found->second;
  }
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    if (failed(initial))
      return failure();
    std::string carrier =
        uniqueName("loop_state_" + std::to_string(index), *node);
    line(carrier + " = " + initial->str());
    carriers.push_back(carrier);
    valueNames[body.getArgument(index + 1)] = carrier;
  }
  loopCarriers[&operation] = carriers;
  std::string iterator = makeRegionArgumentName(operation, 0);
  valueNames[body.getArgument(0)] = iterator;
  line("for " + iterator + " in tl.range(0, " + stop + "):");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active Triton sequential loop");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index]);
  return success();
}

LogicalResult SourceEmitter::enterIf(Operation &operation) {
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (failed(condition) || operation.getNumRegions() != 2)
    return operation.emitOpError("lacks a mechanical Triton scalar branch");
  SmallVector<std::string> results;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    results.push_back(makeResultName(operation, index));
  ifResults[&operation] = std::move(results);
  line("if " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveIf(Operation &operation) {
  auto results = ifResults.find(&operation);
  if (results == ifResults.end())
    return operation.emitOpError("has no active Triton scalar branch");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, results->second[index]);
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
      return operation.emitOpError("does not match its Triton loop state");
    for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
      FailureOr<StringRef> yielded = lookupValue(operation, index);
      if (failed(yielded))
        return failure();
      line(carriers->second[index] + " = " + yielded->str());
    }
    return success();
  }
  if (name != "intent.if")
    return success();
  auto results = ifResults.find(owner);
  if (results == ifResults.end() ||
      results->second.size() != operation.getNumOperands())
    return operation.emitOpError("does not match its Triton branch results");
  if (operation.getNumOperands() == 0)
    line("pass");
  for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(operation, index);
    if (failed(yielded))
      return failure();
    line(results->second[index] + " = " + yielded->str());
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
        "lacks a private scalar-array Triton buffer binding");
  std::string base = makeResultName(operation, 0);
  SmallVector<std::string> elements;
  for (int64_t index = 0; index < info->shape.front(); ++index) {
    std::string element = base + "_" + std::to_string(index);
    line(element + " = " + initializer->str());
    elements.push_back(std::move(element));
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
    return operation.emitOpError("lacks a scalarized Triton buffer load");
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
      expression = "tl.where(" + dynamic->str() + " == " +
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
    return operation.emitOpError("lacks a scalarized Triton buffer store");
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
    line(element + " = tl.where(" + dynamic->str() + " == " +
         std::to_string(position) + ", " + stored->str() + ", " + element +
         ")");
  return success();
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a resolved Triton load binding");
  FailureOr<bool> wholeView = target::isWholeViewAccess(operation);
  if (failed(wholeView))
    return failure();
  if (*wholeView && boundary.getDomainNodes().empty() &&
      boundary.getLoadFill() == "none") {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view))
      return failure();
    bindResult(operation, 0, (*view)->pointer);
    return success();
  }
  if (boundary.getDefer()) {
    deferredLoads[operation.getResult(0)] = &operation;
    return success();
  }
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view))
    return failure();
  FailureOr<std::string> pointers =
      emitPointerExpression(operation, **view, false);
  FailureOr<std::string> mask = emitMaskExpression(operation, false);
  FailureOr<std::string> physicalFill =
      transferPhysicalExtentFill(operation);
  if (failed(pointers) || failed(mask) || failed(physicalFill))
    return failure();
  std::string result = makeResultName(operation, 0);
  StringRef loadFill = boundary.getLoadFill();
  if (loadFill == "none" && !physicalFill->empty())
    loadFill = *physicalFill;
  if (loadFill == "none") {
    line(result + " = tl.load(" + *pointers + ")");
  } else {
    StringRef fill = loadFill == "negative_infinity"
                         ? "-float('inf')"
                         : "0.0";
    line(result + " = tl.load(" + *pointers + ", mask=" + *mask +
         ", other=" + fill.str() + ")");
  }
  bindResult(operation, 0, result);
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
  auto result = operation.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                    : RankedTensorType();
  if (failed(node) || !binding || binding.getLowering() != "logical_indices" ||
      failed(axis) || !result || result.getRank() != 1 ||
      !isa<IntegerType, IndexType>(result.getElementType()))
    return operation.emitOpError("lacks a mechanical Triton indices binding");
  if (axisIndices.lookup(axis->getNode()).empty() && axis->hasRole("lane"))
    axisIndices[axis->getNode()] =
        "tl.arange(0, " + axis->getTile().str() + ")";
  FailureOr<std::string> expression =
      indexExpression(*axis, false, 0, 1, operation);
  if (failed(expression))
    return failure();
  bindResult(operation, 0, *expression);
  return success();
}

LogicalResult SourceEmitter::emitRandom(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "random emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> seed = lookupValue(operation, 0);
  FailureOr<StringRef> counter = lookupValue(operation, 1);
  if (failed(node) || !binding ||
      binding.getLowering() != "counter_xorshift32" || failed(seed) ||
      failed(counter) || operation.getNumResults() != 1)
    return operation.emitOpError(
        "lacks a mechanical Triton counter RNG binding");
  std::string result = makeResultName(operation, 0);
  std::string bits = result + "_bits";
  line(bits + " = tl.cast(" + counter->str() +
       ", tl.uint32) ^ tl.cast(" + seed->str() +
       ", tl.uint32) ^ tl.cast(1831565813, tl.uint32)");
  line(bits + " = " + bits + " ^ (" + bits + " << 13)");
  line(bits + " = " + bits + " ^ (" + bits + " >> 17)");
  line(bits + " = " + bits + " ^ (" + bits + " << 5)");
  line(result + " = tl.cast(" + bits +
       " >> 8, tl.float32) * 5.960464477539063e-08");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  if (failed(node) || !binding)
    return operation.emitOpError("lacks a Triton reduction binding");
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(operand))
    return failure();
  if (binding.getLowering() == "tl.max_with_index") {
    if (operation.getNumResults() != 2)
      return operation.emitOpError(
          "Triton arg-reduction requires value and index results");
    std::string value = makeResultName(operation, 0);
    std::string index = makeResultName(operation, 1);
    line(value + ", " + index + " = tl.max(" + operand->str() + ", axis=" +
         std::to_string(binding.getAxis()) +
         ", return_indices=True, return_indices_tie_break_left=True)");
    bindResult(operation, 0, value);
    bindResult(operation, 1, index);
    return success();
  }
  if (operation.getNumResults() != 1)
    return operation.emitOpError("Triton reduction requires one result");
  std::string result = makeResultName(operation, 0);
  if (binding.getLowering() == "tl.reduce_all")
    line(result + " = ~tl.reduce_or(~(" + operand->str() + "), axis=" +
         std::to_string(binding.getAxis()) + ")");
  else
    line(result + " = " + binding.getLowering().str() + "(" + operand->str() +
         ", axis=" + std::to_string(binding.getAxis()) + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitScan(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "scan emission");
  plan::ScanOp binding =
      succeeded(node) ? planIndex.scans.lookup(*node) : plan::ScanOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "tl.cumsum" ||
      failed(operand) || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a mechanical Triton scan binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.cumsum(" + operand->str() + ", axis=" +
       std::to_string(binding.getAxis()) + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitBroadcast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "broadcast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  if (failed(node) || !binding || binding.getLowering() != "alias")
    return operation.emitOpError("lacks a Triton broadcast binding");
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(operand))
    return failure();
  bindResult(operation, 0, operand->str());
  return success();
}

LogicalResult SourceEmitter::emitUnary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "unary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(node) || !binding || failed(operand))
    return failure();
  std::string result = makeResultName(operation, 0);
  std::string expression = binding.getLowering() == "python_negate"
                               ? "-(" + operand->str() + ")"
                               : binding.getLowering().str() + "(" +
                                     operand->str() + ")";
  line(result + " = " + expression);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitBinary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "binary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> lhs = lookupValue(operation, 0);
  FailureOr<StringRef> rhs = lookupValue(operation, 1);
  if (failed(node) || !binding || failed(lhs) || failed(rhs))
    return failure();
  std::string result = makeResultName(operation, 0);
  std::string expression;
  if (binding.getLowering() == "tl.maximum" ||
      binding.getLowering() == "tl.minimum")
    expression = binding.getLowering().str() + "(" + lhs->str() + ", " +
                 rhs->str() + ")";
  else if (binding.getLowering() == "python_floor_divide" ||
           binding.getLowering() == "python_remainder") {
    std::string quotient = result + "_quotient";
    std::string remainder = result + "_remainder";
    std::string adjust = result + "_adjust";
    line(quotient + " = (" + lhs->str() + ") // (" + rhs->str() + ")");
    line(remainder + " = (" + lhs->str() + ") - " + quotient + " * (" +
         rhs->str() + ")");
    line(adjust + " = (" + remainder + " != 0) & ((" + remainder +
         " < 0) != (" + rhs->str() + " < 0))");
    expression = binding.getLowering() == "python_floor_divide"
                     ? quotient + " - tl.where(" + adjust + ", 1, 0)"
                     : remainder + " + tl.where(" + adjust + ", " +
                           rhs->str() + ", 0)";
  } else {
    StringRef symbol;
    if (binding.getLowering() == "python_add")
      symbol = "+";
    else if (binding.getLowering() == "python_subtract")
      symbol = "-";
    else if (binding.getLowering() == "python_multiply")
      symbol = "*";
    else if (binding.getLowering() == "python_true_divide")
      symbol = "/";
    else if (binding.getLowering() == "python_bitwise_and")
      symbol = "&";
    else if (binding.getLowering() == "python_bitwise_or")
      symbol = "|";
    else if (binding.getLowering() == "python_bitwise_xor")
      symbol = "^";
    else if (binding.getLowering() == "python_left_shift")
      symbol = "<<";
    else if (binding.getLowering() == "python_right_shift")
      symbol = ">>";
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
    else
      return operation.emitOpError("uses an unsupported binary lowering");
    expression = "(" + lhs->str() + ") " + symbol.str() + " (" +
                 rhs->str() + ")";
  }
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), expression, operation);
  if (failed(padded))
    return failure();
  line(result + " = " + *padded);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitConditional(Operation &operation, bool mask) {
  FailureOr<int64_t> node = target::getNodeID(
      operation, mask ? "mask emission" : "select emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  unsigned conditionOperand = mask ? 1 : 0;
  unsigned trueOperand = mask ? 0 : 1;
  FailureOr<StringRef> condition = lookupValue(operation, conditionOperand);
  FailureOr<StringRef> trueValue = lookupValue(operation, trueOperand);
  FailureOr<StringRef> falseValue = lookupValue(operation, 2);
  if (failed(node) || !binding || binding.getLowering() != "tl.where" ||
      operation.getNumResults() != 1 || failed(condition) || failed(trueValue) ||
      failed(falseValue))
    return operation.emitOpError("lacks a mechanical Triton conditional binding");
  std::string result = makeResultName(operation, 0);
  std::string expression = "tl.where(" + condition->str() + ", " +
                           trueValue->str() + ", " + falseValue->str() + ")";
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), expression, operation);
  if (failed(padded))
    return failure();
  line(result + " = " + *padded);
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
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  Type resultType = operation.getNumResults() == 1
                        ? operation.getResult(0).getType()
                        : Type();
  if (failed(node) || !binding || binding.getLowering() != "tl.cast" ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical Triton cast binding");
  StringRef targetType = tritonDtype(resultType);
  if (targetType.empty())
    return operation.emitOpError("casts to an unsupported Triton type");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.cast(" + operand->str() + ", " + targetType.str() + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitReshape(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reshape emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> shape = emitTensorShape(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "tl.reshape" ||
      binding.getReuseOperandAttr().getInt() != -1 || failed(operand) ||
      failed(shape) || operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a mechanical Triton reshape binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.reshape(" + operand->str() + ", " + *shape + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitTranspose(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "transpose emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<SmallVector<int64_t>> permutation =
      target::emission::transposePermutation(operation);
  if (failed(node) || !binding || binding.getLowering() != "tl.permute" ||
      binding.getReuseOperandAttr().getInt() != -1 || failed(operand) ||
      failed(permutation))
    return operation.emitOpError("lacks a mechanical Triton transpose binding");
  std::string result = makeResultName(operation, 0);
  std::string expression = "tl.permute(" + operand->str();
  for (int64_t axis : *permutation)
    expression += ", " + std::to_string(axis);
  line(result + " = " + expression + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitFull(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "full emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> fill = lookupValue(operation, 0);
  auto resultType = operation.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                        : RankedTensorType();
  FailureOr<std::string> shape = emitTensorShape(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "tl.full" ||
      failed(fill) || !resultType || failed(shape))
    return operation.emitOpError("lacks a mechanical Triton full binding");
  StringRef dtype = tritonDtype(resultType);
  if (dtype.empty())
    return operation.emitOpError("uses an unsupported Triton full dtype");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.full(" + *shape + ", " + fill->str() +
       ", dtype=" + dtype.str() + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitZeros(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "zeros emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  auto resultType = operation.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                        : RankedTensorType();
  FailureOr<std::string> shape = emitTensorShape(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "tl.zeros" ||
      !resultType || failed(shape))
    return operation.emitOpError("lacks a mechanical Triton zeros binding");
  StringRef dtype = tritonDtype(resultType);
  if (dtype.empty())
    return operation.emitOpError("uses an unsupported Triton zeros dtype");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.zeros(" + *shape + ", dtype=" + dtype.str() + ")");
  bindResult(operation, 0, result);
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
  if (!planIndex.stages.empty() && binding &&
      binding.getLowering() == "tl.indirect_gather") {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view) || failed(relation))
      return failure();
    if (binding.getDefer() && (*view)->tensor.getRank() == 2) {
      deferredLoads[operation.getResult(0)] = &operation;
      return success();
    }
    FailureOr<StringRef> valid =
        validIndex ? lookupValue(operation, validIndex.getInt())
                   : FailureOr<StringRef>(failure());
    FailureOr<StringRef> fill =
        fillIndex ? lookupValue(operation, fillIndex.getInt())
                  : FailureOr<StringRef>(failure());
    if (failed(valid) || failed(fill))
      return failure();
    if ((*view)->tensor.getRank() != 1 || relation->size() != 1 ||
        (*relation)[0].kind != "value_index" ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front())
      return operation.emitOpError(
          "staged indirect gather requires one indexed vector source");
    FailureOr<StringRef> index =
        lookupValue(operation, *(*relation)[0].operands.front());
    if (failed(index))
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = tl.load(" + (*view)->pointer + " + " +
         addressIndex(*index) + " * " + addressIndex((*view)->strides[0]) +
         ", mask=member_mask & " +
         valid->str() + ", other=" + fill->str() + ")");
    bindResult(operation, 0, result);
    return success();
  }
  FailureOr<StringRef> source = lookupValue(operation, 0);
  FailureOr<StringRef> valid =
      validIndex ? lookupValue(operation, validIndex.getInt())
                 : FailureOr<StringRef>(failure());
  FailureOr<StringRef> fill =
      fillIndex ? lookupValue(operation, fillIndex.getInt())
                : FailureOr<StringRef>(failure());
  bool appendAxis = succeeded(relation) && relation->size() == 2 &&
                    (*relation)[0].kind == "full_slice" &&
                    (*relation)[1].kind == "new_axis";
  bool prependAxis = succeeded(relation) && relation->size() == 2 &&
                     (*relation)[0].kind == "new_axis" &&
                     (*relation)[1].kind == "full_slice";
  if (failed(node) || !binding || binding.getLowering() != "expand_dims" ||
      failed(relation) || (!appendAxis && !prependAxis) || failed(source) ||
      failed(valid) || failed(fill))
    return operation.emitOpError("lacks a mechanical Triton gather binding");
  std::string result = makeResultName(operation, 0);
  std::string expanded = source->str() + (appendAxis ? "[:, None]" : "[None, :]");
  line(result + " = tl.where(" + valid->str() + ", " + expanded + ", " +
       fill->str() + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitMembers(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "members emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  if (failed(node) || !binding || binding.getLowering() != "tl.members" ||
      operation.getNumResults() != 1)
    return operation.emitOpError("lacks a Triton members binding");
  FailureOr<plan::AxisOp> memberAxis =
      operation.getNumOperands() == 1
          ? resolveAxis(operation.getOperand(0), operation)
          : FailureOr<plan::AxisOp>(failure());
  if (failed(memberAxis))
    return operation.emitOpError("has no physical ragged-member axis");
  if (!planIndex.components.orderedRaggedAxes.contains(memberAxis->getNode())) {
    if (activeStages.empty())
      return operation.emitOpError(
          "has neither ordered traversal nor staged ragged ownership");
    bindResult(operation, 0, "routes");
    return success();
  }
  FailureOr<plan::RaggedOp> relation =
      target::emission::uniqueRaggedRelation(planIndex, memberAxis->getNode(),
                                              operation);
  auto runtime = succeeded(relation)
                     ? raggedRuntimeByRelation.find(relation->getNode())
                     : raggedRuntimeByRelation.end();
  std::string position = axisIndices.lookup(memberAxis->getNode());
  if (failed(relation) || runtime == raggedRuntimeByRelation.end() ||
      position.empty())
    return operation.emitOpError("has no ordered ragged runtime position");
  RaggedRuntime &ragged = raggedRuntimes[runtime->second];
  std::string suffix = std::to_string(memberAxis->getNode());
  std::string valid = position + " < sequence_end_" + suffix;
  std::string result = makeResultName(operation, 0);
  if (ragged.indices)
    line(result + " = tl.load(" + ragged.indices->pointer + " + " +
         addressIndex(position) + " * " +
         addressIndex(ragged.indices->strides[0]) + ", mask=" + valid +
         ", other=0)");
  else
    line(result + " = tl.where(" + valid + ", " + position + ", 0)");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical Triton stream binding");
  Block &body = operation.getRegion(0).front();
  bool hasStop = static_cast<bool>(binding.getStopNodeAttr());
  bool hasRuntimeExtent = static_cast<bool>(
      operation.getAttrOfType<IntegerAttr>("intent.extent_operand_index"));
  if (operation.getNumOperands() !=
          operation.getNumResults() + 1 + (hasRuntimeExtent ? 1 : 0) +
              (hasStop ? 1 : 0) ||
      body.getNumArguments() != operation.getNumResults() + 1)
    return operation.emitOpError("has inconsistent stream carried state");

  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    if (failed(initial))
      return failure();
    std::string carrier =
        uniqueName("stream_state_" + std::to_string(index), *node);
    line(carrier + " = " + initial->str());
    carriers.push_back(carrier);
    valueNames[body.getArgument(index + 1)] = carrier;
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
    line("sequence_begin_" + raggedSuffix + " = tl.load(" +
         ragged.offsets->pointer + " + " + addressIndex(outer) + " * " +
         addressIndex(ragged.offsets->strides[0]) + ")");
    line("sequence_end_" + raggedSuffix + " = tl.load(" +
         ragged.offsets->pointer + " + " + addressIndex(outer + " + 1") +
         " * " + addressIndex(ragged.offsets->strides[0]) + ")");
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
      streamExtent = "tl.minimum((" + block + " + 1) * " +
                     stopAxis->getTile().str() + ", " + streamExtent + ")";
    } else if (stopAxis->getNode() != binding.getAxisNode()) {
      return operation.emitOpError(
          "has no Triton spelling for its planned logical stream stop");
    }
  }
  std::string block = "stream_block_" + std::to_string(*node);
  std::string offsets = "axis_index_" + std::to_string(binding.getAxisNode());
  line("for " + block + " in range(0, tl.cdiv(" + streamExtent + ", " +
       binding.getTile().str() + ")):");
  ++indentation;
  line(offsets + " = " +
       std::string(raggedStream ? "sequence_begin_" + raggedSuffix + " + " : "") +
       addressIndex(block) + " * " + binding.getTile().str() + " + " +
       addressIndex("tl.arange(0, " + binding.getTile().str() + ")"));
  axisIndices[binding.getAxisNode()] = offsets;
  valueNames[body.getArgument(0)] = offsets;
  return success();
}

LogicalResult SourceEmitter::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  if (carriers == streamCarriers.end())
    return operation.emitOpError("has no active Triton stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (terminator.getName().getStringRef() != "intent.yield" ||
      terminator.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("does not yield every Triton stream state");
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(terminator, index);
    if (failed(yielded))
      return failure();
    line(carriers->second[index] + " = " + yielded->str());
  }
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    valueNames[operation.getResult(index)] = carriers->second[index];
  return success();
}

LogicalResult SourceEmitter::emitContract(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "tl.dot")
    return operation.emitOpError("lacks a Triton contraction binding");
  if (operation.getNumOperands() != 2)
    return operation.emitOpError("Triton contraction requires two operands");
  FailureOr<target::emission::ContractionOrientation> orientation =
      target::emission::contractionOrientation(operation);
  if (failed(orientation))
    return failure();
  if (!planIndex.stages.empty()) {
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
    std::string feature = stageFeatureDimensions.lookup(stage);
    std::string reduction = stageReductionDimensions.lookup(stage);
    std::string result = makeResultName(operation, 0);
    line(result +
         " = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)");
    line("for reduction_block in range(0, tl.cdiv(" + reduction +
         ", BLOCK_SIZE_K)):");
    ++indentation;
    line("offs_reduction = " + addressIndex("reduction_block") +
         " * BLOCK_SIZE_K + " +
         addressIndex("tl.arange(0, BLOCK_SIZE_K)"));

    std::string lhs;
    if (lhsAccess) {
      if (lhsAccess->getName().getStringRef() != "intent.gather")
        return lhsAccess->emitOpError(
            "is not a staged indirect contraction input");
      FailureOr<SmallVector<target::IndexTerm>> relation =
          target::parseIndexRelation(*lhsAccess);
      FailureOr<ABIView *> lhsView =
          lookupView(lhsAccess->getOperand(0), *lhsAccess);
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
      lhs = makeResultName(*lhsAccess, 0);
      line(lhs + " = tl.load(" + (*lhsView)->pointer + " + " +
           addressIndex(rows->str() + "[:, None]") + " * " +
           addressIndex((*lhsView)->strides[0]) + " + " +
           addressIndex("offs_reduction[None, :]") + " * " +
           addressIndex((*lhsView)->strides[1]) +
           ", mask=member_mask[:, None] & (offs_reduction[None, :] < " +
           reduction + "), other=0.0)");
    } else {
      auto workspace = workspaceNames.find(operation.getOperand(0));
      if (workspace == workspaceNames.end())
        return operation.emitOpError(
            "staged contraction input has no materialized workspace");
      lhs = "stage_input";
      line(lhs + " = tl.load(" + workspace->second +
           " + " + addressIndex("member_offsets[:, None]") + " * " +
           addressIndex(reduction) + " + " +
           addressIndex("offs_reduction[None, :]") +
           ", mask=member_mask[:, None] & "
           "(offs_reduction[None, :] < " + reduction + "), other=0.0)");
    }
    std::string rhs = makeResultName(*rhsLoad, 0);
    line(rhs + " = tl.load(" + (*rhsView)->pointer + " + " +
         addressIndex("expert") + " * " + addressIndex((*rhsView)->strides[0]) +
         " + " + addressIndex("offs_reduction[:, None]") + " * " +
         addressIndex((*rhsView)->strides[1]) + " + " +
         addressIndex("offs_feature[None, :]") + " * " +
         addressIndex((*rhsView)->strides[2]) +
         ", mask=(offs_reduction[:, None] < " + reduction +
         ") & feature_mask[None, :], other=0.0)");
    if (promoteToF32 && !lhsElement.isF32())
      line(lhs + " = " + lhs + ".to(tl.float32)");
    if (promoteToF32 && !rhsElement.isF32())
      line(rhs + " = " + rhs + ".to(tl.float32)");
    line(result + " = tl.dot(" + lhs + ", " + rhs + ", " + result + ")");
    --indentation;
    bindResult(operation, 0, result);
    return success();
  }
  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (!lhsLoad && !rhsLoad) {
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    if (failed(lhs) || failed(rhs))
      return failure();
    std::string lhsExpression = lhs->str();
    std::string rhsExpression = rhs->str();
    if (orientation->lhsTranspose)
      lhsExpression = "tl.trans(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = "tl.trans(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression +
         ", out_dtype=tl.float32)");
    bindResult(operation, 0, result);
    return success();
  }
  if (!lhsLoad || !rhsLoad)
    return operation.emitOpError(
        "deferred Triton contraction has inconsistent operand residency");
  FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  FailureOr<plan::AxisOp> reductionAxis =
      target::emission::contractionReductionAxis(planIndex, *lhsLoad, *rhsLoad,
                                                 operation);
  if (failed(lhsView) || failed(rhsView) || failed(reductionAxis))
    return failure();

  std::string reductionRole = reductionAxis->getRole().str();
  std::string reductionTile = reductionAxis->getTile().str();
  std::string reductionOffset = "offs_" + reductionRole;
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)");
  line("for reduction_block in range(0, tl.cdiv(" +
       roleDimensions.lookup(reductionRole) + ", " + reductionTile + ")):");
  ++indentation;
  line(reductionOffset + " = " + addressIndex("reduction_block") + " * " +
       reductionTile + " + " +
       addressIndex("tl.arange(0, " + reductionTile + ")"));
  axisIndices[reductionAxis->getNode()] = reductionOffset;
  FailureOr<std::string> lhsPointers =
      emitPointerExpression(*lhsLoad, **lhsView, false);
  FailureOr<std::string> rhsPointers =
      emitPointerExpression(*rhsLoad, **rhsView, false);
  FailureOr<std::string> lhsMask = emitMaskExpression(*lhsLoad, false);
  FailureOr<std::string> rhsMask = emitMaskExpression(*rhsLoad, false);
  if (failed(lhsPointers) || failed(rhsPointers) || failed(lhsMask) ||
      failed(rhsMask))
    return failure();
  std::string lhs = makeResultName(*lhsLoad, 0);
  std::string rhs = makeResultName(*rhsLoad, 0);
  line(lhs + " = tl.load(" + *lhsPointers + ", mask=" + *lhsMask +
       ", other=0.0)");
  line(rhs + " = tl.load(" + *rhsPointers + ", mask=" + *rhsMask +
       ", other=0.0)");
  std::string lhsExpression = lhs;
  std::string rhsExpression = rhs;
  if (orientation->lhsTranspose)
    lhsExpression = "tl.trans(" + lhsExpression + ")";
  if (orientation->rhsTranspose)
    rhsExpression = "tl.trans(" + rhsExpression + ")";
  line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression + ", " +
       result + ")");
  --indentation;
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitStore(Operation &operation) {
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<int64_t> node = target::getNodeID(operation, "store emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (!valueIndex || failed(node) || !boundary)
    return operation.emitOpError("lacks a Triton store boundary binding");
  FailureOr<StringRef> stored = lookupValue(operation, valueIndex.getInt());
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(stored) || failed(view))
    return failure();
  FailureOr<std::string> pointers =
      emitPointerExpression(operation, **view, true);
  FailureOr<std::string> mask = emitMaskExpression(operation, true);
  if (failed(pointers) || failed(mask))
    return failure();
  line("tl.store(" + *pointers + ", " + stored->str() + ", mask=" + *mask +
       ")");
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
  if (failed(node) || !binding || binding.getDefer() ||
      binding.getStoreMask() != "predicate" || !valueIndex ||
      failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "value_index" ||
      (*relation)[0].operands.size() != 1 ||
      !(*relation)[0].operands.front() ||
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored) ||
      (*view)->tensor.getRank() != 2)
    return operation.emitOpError("lacks a mechanical Triton unique store");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  std::string pointer = (*view)->pointer + " + " +
                        addressIndex(rows->str() + "[:, None]") + " * " +
                        addressIndex((*view)->strides[0]) + " + " +
                        addressIndex("offs_feature[None, :]") + " * " +
                        addressIndex((*view)->strides[1]);
  line("tl.store(" + pointer + ", " + stored->str() +
       ", mask=member_mask[:, None] & feature_mask[None, :])");
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
    return operation.emitOpError("lacks a mechanical Triton atomic merge");
  if (planIndex.stages.empty()) {
    FailureOr<std::string> pointer =
        emitPointerExpression(operation, **view, true);
    FailureOr<std::string> mask = emitMaskExpression(operation, true);
    if (failed(pointer) || failed(mask))
      return failure();
    line("tl.atomic_add(" + *pointer + ", " + stored->str() + ", mask=" +
         *mask + ", sem='relaxed', scope='gpu')");
    return success();
  }
  if (!valueIndex || failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "value_index" ||
      (*relation)[0].operands.size() != 1 ||
      !(*relation)[0].operands.front() ||
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored) ||
      (*view)->tensor.getRank() != 2)
    return operation.emitOpError("lacks a mechanical Triton atomic merge");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  std::string pointer = (*view)->pointer + " + " +
                        addressIndex(rows->str() + "[:, None]") + " * " +
                        addressIndex((*view)->strides[0]) + " + " +
                        addressIndex("offs_feature[None, :]") + " * " +
                        addressIndex((*view)->strides[1]);
  line("tl.atomic_add(" + pointer + ", " + stored->str() +
       ", mask=member_mask[:, None] & feature_mask[None, :], "
       "sem='relaxed', scope='gpu')");
  return success();
}

LogicalResult SourceEmitter::emitAtomicCas(Operation &operation) {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "compare-and-swap emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  auto compareIndex = operation.getAttrOfType<IntegerAttr>(
      "intent.compare_operand_index");
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  auto ordering = operation.getAttrOfType<StringAttr>("intent.ordering");
  auto scope = operation.getAttrOfType<StringAttr>("intent.scope");
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<StringRef> expected =
      compareIndex ? lookupValue(operation, compareIndex.getInt())
                   : FailureOr<StringRef>(failure());
  FailureOr<StringRef> desired =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  if (failed(node) || !boundary || boundary.getAccess() != "store" ||
      boundary.getCheckBounds() || !planIndex.stages.empty() || !compareIndex ||
      !valueIndex || !ordering || !scope || failed(view) || failed(expected) ||
      failed(desired) || operation.getNumResults() != 1 ||
      !operation.getResult(0).getType().isInteger(32) ||
      boundary.getResultSpace() != "private_scalar")
    return operation.emitOpError(
        "lacks an in-bounds scalar Triton compare-and-swap binding");

  StringRef semantic = ordering.getValue();
  std::string targetScope;
  if (scope.getValue() == "workgroup")
    targetScope = "cta";
  else if (scope.getValue() == "device")
    targetScope = "gpu";
  else if (scope.getValue() == "system")
    targetScope = "sys";
  else
    return operation.emitOpError("has no Triton atomic scope spelling");
  FailureOr<std::string> pointer =
      emitPointerExpression(operation, **view, true);
  if (failed(pointer))
    return failure();
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.atomic_cas(" + *pointer + ", " + expected->str() +
       ", " + desired->str() + ", sem='" + semantic.str() +
       "', scope='" + targetScope + "')");
  bindResult(operation, 0, result);
  return success();
}

} // namespace intent::triton::emission

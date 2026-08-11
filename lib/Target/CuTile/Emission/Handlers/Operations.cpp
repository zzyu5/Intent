#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <optional>

using namespace mlir;

namespace intent::cutile::emission {
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
      expression = number < 0 ? "-math.inf" : "math.inf";
    else if (std::isnan(number))
      expression = "math.nan";
    else
      expression = std::to_string(number);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported cuTile constant value");
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
    return operation.emitOpError("lacks a mechanical cuTile dimension binding");
  std::string dimension = (*view)->shape[axis.getInt()];
  if (!planIndex.components.reusedAxes.empty()) {
    if (dimension == roleDimensions.lookup("program_0"))
      dimension = "N_ROWS";
    else if (dimension == roleDimensions.lookup("lane_0"))
      dimension = "DIM_COLS";
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
    std::string extent =
        dimensionOwners.lookup(roleDimensions.lookup(role));
    return axis.isScalar()
               ? extent
               : "ct.cdiv(" + extent + ", " + axis.getTile().str() + ")";
  };
  bool persistent = planIndex.program.getPersistent();
  std::string linear = "persistent_program";
  if (persistent) {
    line("total_program_tiles = " +
         target::emission::projectProgramVolume(planIndex, axisExtent));
    line("program_start = " + addressIndex("ct.bid(0)"));
    line("program_step = " + addressIndex("ct.num_blocks(0)"));
    line("for " + linear +
         " in range(program_start, total_program_tiles, program_step):");
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
          "cuTile supports two-axis program grouping on one worker axis");
    StringRef group = axes.front().getGroupSpelling();
    if (group.empty())
      return axes.front().emitOpError("has no cuTile group spelling");
    plan::AxisOp lhs = axes[0];
    plan::AxisOp rhs = axes[1];
    std::string lhsRole =
        "program_" + std::to_string(lhs.getProgramOrder());
    std::string rhsRole =
        "program_" + std::to_string(rhs.getProgramOrder());
    std::string pid = "group_bid_" + std::to_string(lhs.getNode());
    std::string lhsCount = "group_count_" + std::to_string(lhs.getNode());
    std::string rhsCount = "group_count_" + std::to_string(rhs.getNode());
    std::string span = "group_span_" + std::to_string(lhs.getNode());
    std::string id = "group_id_" + std::to_string(lhs.getNode());
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
      line(pid + " = " + addressIndex("ct.bid(" +
                                       std::to_string(lhs.getWorkerAxis()) +
                                       ")"));
    }
    line(lhsCount + " = " +
         addressIndex("ct.cdiv(" +
                      dimensionOwners.lookup(roleDimensions.lookup(lhsRole)) +
                      ", " + lhs.getTile().str() + ")"));
    line(rhsCount + " = " +
         addressIndex("ct.cdiv(" +
                      dimensionOwners.lookup(roleDimensions.lookup(rhsRole)) +
                      ", " + rhs.getTile().str() + ")"));
    line(span + " = " + group.str() + " * " + rhsCount);
    line(id + " = " + pid + " // " + span);
    line(first + " = " + id + " * " + group.str());
    line(size + " = min(" + lhsCount + " - " + first + ", " + group.str() +
         ")");
    line(lhsBlock + " = " + first + " + (" + pid + " % " + size + ")");
    line(rhsBlock + " = (" + pid + " % " + span + ") // " + size);
    programBlocks[lhs.getNode()] = lhsBlock;
    programBlocks[rhs.getNode()] = rhsBlock;
    axisIndices[lhs.getNode()] = lhsBlock;
    axisIndices[rhs.getNode()] = rhsBlock;
  }

  SmallVector<target::emission::ProgramIndexProjection> projections =
      persistent
          ? target::emission::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::emission::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("ct.bid(" + std::to_string(worker) + ")");
                });
  for (const target::emission::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    line(block + " = " + projection.expression);
    programBlocks[axis.getNode()] = block;
    axisIndices[axis.getNode()] = block;
    if (!planIndex.components.orderedRaggedProgramAxes.contains(axis.getNode()))
      continue;
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
          "ordered ragged axis has no outer-axis or metadata binding");
    RaggedRuntime &ragged = raggedRuntimes[runtime->second];
    auto ordered =
        planIndex.components.orderedAxesByRelation.find(relation->getNode());
    if (ordered == planIndex.components.orderedAxesByRelation.end() ||
        ordered->second.empty())
      return axis.emitOpError("has no ordered axis in its ragged relation");
    for (int64_t orderedAxis : ordered->second) {
      std::string suffix = std::to_string(orderedAxis);
      line("sequence_begin_" + suffix + " = ct.gather(" +
           ragged.offsets->argument->name + ", " + addressIndex(outer) +
           ", padding_value=0)");
      line("sequence_end_" + suffix + " = ct.gather(" +
           ragged.offsets->argument->name + ", " + addressIndex(outer + " + 1") +
           ", padding_value=0)");
      line("sequence_length_" + suffix + " = sequence_end_" + suffix +
           " - sequence_begin_" + suffix);
    }
    std::string suffix = std::to_string(ordered->second.front());
    std::string values = "axis_index_" + std::to_string(axis.getNode());
    line(values + " = sequence_begin_" + suffix + " + " + block + " * " +
         axis.getTile().str() + " + " +
         addressIndex("ct.arange(" + axis.getTile().str() +
                      ", dtype=ct.int32)"));
    line(values + " = ct.where(" + values + " < sequence_end_" + suffix +
         ", " + values + ", " + ragged.membersView->argument->name +
         ".shape[0])");
    axisIndices[axis.getNode()] = values;
  }
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() != 1)
    return operation.emitOpError(
        "parallel ownership requires one region argument");
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
    valueNames[argument] = outer ? "expert" : "member_offsets";
    return success();
  }

  if (!planIndex.components.reusedAxes.empty()) {
    if (&operation != programRoot ||
        axis->getNode() != planIndex.components.reusedAxes.front().getNode())
      return operation.emitOpError("is not the worker-reused program axis");
    int64_t workerAxis = axis->getWorkerAxis();
    line("program_start = " +
         addressIndex("ct.bid(" + std::to_string(workerAxis) + ")"));
    line("program_step = " +
         addressIndex("ct.num_blocks(" + std::to_string(workerAxis) + ")"));
    line(vectorIndex + " = " +
         addressIndex("ct.arange(TILE_SIZE, dtype=ct.int32)"));
    line("for " + programIndex +
         " in range(program_start, N_ROWS, program_step):");
    ++indentation;
    axisIndices[axis->getNode()] = programIndex;
    valueNames[argument] = programIndex;
    return success();
  }

  if (planIndex.program.getPersistent()) {
    if (&operation == programRoot && failed(emitProgramBindings()))
      return failure();
    std::string value = axisIndices.lookup(axis->getNode());
    if (value.empty())
      return axis->emitOpError("has no persistent cuTile program index");
    valueNames[argument] = value;
    return success();
  }

  if (&operation == programRoot && failed(emitProgramBindings()))
    return failure();
  std::string value = axisIndices.lookup(axis->getNode());
  if (value.empty())
    return axis->emitOpError("has no emitted per-axis program index");
  valueNames[argument] = value;
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
    return operation.emitOpError("lacks a mechanical cuTile sequential loop");
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
      return operation.emitOpError("has no emitted cuTile loop bound");
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
  line("for " + iterator + " in range(" + stop + "):");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active cuTile sequential loop");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index]);
  return success();
}

LogicalResult SourceEmitter::enterIf(Operation &operation) {
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (failed(condition) || operation.getNumRegions() != 2)
    return operation.emitOpError("lacks a mechanical cuTile scalar branch");
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
    return operation.emitOpError("has no active cuTile scalar branch");
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
      return operation.emitOpError("does not match its cuTile loop state");
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
    return operation.emitOpError("does not match its cuTile branch results");
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
        "lacks a private scalar-array cuTile buffer binding");
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
    return operation.emitOpError("lacks a scalarized cuTile buffer load");
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
      expression = "ct.where(" + dynamic->str() + " == " +
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
    return operation.emitOpError("lacks a scalarized cuTile buffer store");
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
    line(element + " = ct.where(" + dynamic->str() + " == " +
         std::to_string(position) + ", " + stored->str() + ", " + element +
         ")");
  return success();
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a cuTile load boundary");
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
  FailureOr<std::string> indices =
      indexTuple(operation, boundary.getAccess() == "gather");
  FailureOr<std::string> physicalFill =
      transferPhysicalExtentFill(operation);
  if (failed(view) || failed(indices) || failed(physicalFill))
    return failure();
  StringRef loadFill = boundary.getPadding();
  if (loadFill == "none" && !physicalFill->empty())
    loadFill = *physicalFill;
  Type elementType = (*view)->tensor.getElementType();
  StringRef padding = loadFill == "negative_infinity"
                          ? "-math.inf"
                      : isa<IntegerType, IndexType>(elementType) ? "0"
                                                                : "0.0";
  std::string result = makeResultName(operation, 0);
  if (boundary.getAccess() == "gather") {
    line(result + " = ct.gather(" + (*view)->argument->name + ", " +
         *indices + ", check_bounds=True, padding_value=" + padding.str() +
         ")");
  } else if (boundary.getAccess() == "load") {
    FailureOr<std::string> shape = tileShape(operation);
    bool scalarResult = operation.getNumResults() == 1 &&
                        !isa<RankedTensorType>(operation.getResult(0).getType());
    FailureOr<std::string> resultShape =
        scalarResult ? FailureOr<std::string>(std::string())
                     : emitTensorShape(operation, 0);
    if (failed(shape) || failed(resultShape))
      return failure();
    StringRef paddingMode = loadFill == "negative_infinity"
                                ? "ct.PaddingMode.NEG_INF"
                                : "ct.PaddingMode.ZERO";
    std::string expression =
        "ct.load(" + (*view)->argument->name + ", index=" + *indices +
        ", shape=" + *shape;
    if (loadFill != "none")
      expression += ", padding_mode=" + paddingMode.str();
    expression += ")";
    if (scalarResult)
      expression += ".item()";
    else
      expression += ".reshape(" + *resultShape + ")";
    line(result + " = " + expression);
  } else {
    return boundary.emitOpError("is not a load-like cuTile access");
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
    return operation.emitOpError("lacks a mechanical cuTile indices binding");
  std::string base = axisIndices.lookup(axis->getNode());
  if (base.empty() && axis->hasRole("lane")) {
    base = "0";
    axisIndices[axis->getNode()] = base;
  }
  if (base.empty())
    return operation.emitOpError("has no cuTile vector index realization");
  bool directVector =
      target::emission::isRaggedBoundAxis(planIndex.components,
                                          axis->getNode()) ||
      (axis->hasRole("lane") &&
       kernel.nodes.lookup(axis->getNode()) == vectorDomain);
  std::string expression =
      directVector
          ? addressIndex(base)
          : addressIndex(base) + " * " + axis->getTile().str() + " + " +
                addressIndex("ct.arange(" + axis->getTile().str() +
                             ", dtype=ct.int32)");
  bindResult(operation, 0, expression);
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
        "lacks a mechanical cuTile counter RNG binding");
  std::string result = makeResultName(operation, 0);
  std::string bits = result + "_bits";
  line(bits + " = ct.bitwise_xor(ct.bitwise_xor(ct.astype(" +
       counter->str() + ", ct.uint32), ct.astype(" + seed->str() +
       ", ct.uint32)), ct.astype(1831565813, ct.uint32))");
  line(bits + " = ct.bitwise_xor(" + bits + ", ct.bitwise_lshift(" + bits +
       ", 13))");
  line(bits + " = ct.bitwise_xor(" + bits + ", ct.bitwise_rshift(" + bits +
       ", 17))");
  line(bits + " = ct.bitwise_xor(" + bits + ", ct.bitwise_lshift(" + bits +
       ", 5))");
  line(result + " = ct.astype(ct.bitwise_rshift(" + bits +
       ", 8), ct.float32) * 5.960464477539063e-08");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(node) || !binding || failed(operand))
    return failure();
  if (binding.getLowering() == "ct.max_with_index") {
    if (operation.getNumResults() != 2)
      return operation.emitOpError(
          "cuTile arg-reduction requires value and index results");
    std::string value = makeResultName(operation, 0);
    std::string index = makeResultName(operation, 1);
    std::string axis = std::to_string(binding.getAxis());
    std::string keepDims = binding.getKeepDims() ? "True" : "False";
    line(value + " = ct.max(" + operand->str() + ", " + axis +
         ", keepdims=" + keepDims + ")");
    line(index + " = ct.argmax(" + operand->str() + ", " + axis +
         ", keepdims=" + keepDims + ")");
    bindResult(operation, 0, value);
    bindResult(operation, 1, index);
    return success();
  }
  if (operation.getNumResults() != 1)
    return operation.emitOpError("cuTile reduction requires one result");
  std::string result = makeResultName(operation, 0);
  line(result + " = " + binding.getLowering().str() + "(" + operand->str() +
       ", " + std::to_string(binding.getAxis()) + ", keepdims=" +
       (binding.getKeepDims() ? "True" : "False") + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitScan(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "scan emission");
  plan::ScanOp binding =
      succeeded(node) ? planIndex.scans.lookup(*node) : plan::ScanOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "ct.cumsum" ||
      failed(operand) || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a mechanical cuTile scan binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.cumsum(" + operand->str() + ", axis=" +
       std::to_string(binding.getAxis()) + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitBroadcast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "broadcast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "alias" ||
      failed(operand))
    return operation.emitOpError("lacks a cuTile broadcast binding");
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
  std::string expression;
  if (binding.getLowering() == "python_negate")
    expression = "-(" + operand->str() + ")";
  else if (binding.getLowering() == "python_sigmoid")
    expression = "1.0 / (1.0 + ct.exp(-(" + operand->str() + ")))";
  else
    expression = binding.getLowering().str() + "(" + operand->str() + ")";
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
  if (binding.getLowering() == "ct.maximum" ||
      binding.getLowering() == "ct.minimum")
    expression = binding.getLowering().str() + "(" + lhs->str() + ", " +
                 rhs->str() + ")";
  else if (binding.getLowering().starts_with("ct.bitwise_"))
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
                     ? quotient + " - " + adjust
                     : remainder + " + " + adjust + " * " + rhs->str();
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
      return operation.emitOpError("uses an unsupported cuTile binary lowering");
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

LogicalResult SourceEmitter::emitMask(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "mask emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> value = lookupValue(operation, 0);
  FailureOr<StringRef> predicate = lookupValue(operation, 1);
  FailureOr<StringRef> fill = lookupValue(operation, 2);
  if (failed(node) || !binding || binding.getLowering() != "ct.where" ||
      operation.getNumResults() != 1 || failed(value) || failed(predicate) ||
      failed(fill))
    return operation.emitOpError("lacks a mechanical cuTile mask binding");
  std::string result = makeResultName(operation, 0);
  std::string expression = "ct.where(" + predicate->str() + ", " +
                           value->str() + ", " + fill->str() + ")";
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), expression, operation);
  if (failed(padded))
    return failure();
  line(result + " = " + *padded);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitCast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "cast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  Type resultType = operation.getNumResults() == 1
                        ? operation.getResult(0).getType()
                        : Type();
  if (failed(node) || !binding ||
      (binding.getLowering() != "ct.astype" &&
       binding.getLowering() != "ct.full_cast") ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical cuTile cast binding");
  Type elementType = resultType;
  if (auto tensor = dyn_cast<RankedTensorType>(resultType))
    elementType = tensor.getElementType();
  std::string targetType = dtypeName(elementType, operation);
  if (targetType.empty())
    return failure();
  std::string result = makeResultName(operation, 0);
  if (binding.getLowering() == "ct.full_cast")
    line(result + " = ct.full((), " + operand->str() + ", dtype=" +
         targetType + ")");
  else
    line(result + " = ct.astype(" + operand->str() + ", " + targetType + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitReshape(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reshape emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> shape = emitTensorShape(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "ct.reshape" ||
      binding.getReuseOperandAttr().getInt() != -1 || failed(operand) ||
      failed(shape) || operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a mechanical cuTile reshape binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.reshape(" + operand->str() + ", " + *shape + ")");
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
  if (failed(node) || !binding || binding.getLowering() != "ct.permute" ||
      binding.getReuseOperandAttr().getInt() != -1 || failed(operand) ||
      failed(permutation))
    return operation.emitOpError("lacks a mechanical cuTile transpose binding");
  std::string result = makeResultName(operation, 0);
  std::string axes = "(";
  for (auto [position, axis] : llvm::enumerate(*permutation)) {
    if (position)
      axes += ", ";
    axes += std::to_string(axis);
  }
  if (permutation->size() == 1)
    axes += ",";
  axes += ")";
  line(result + " = ct.permute(" + operand->str() + ", " + axes + ")");
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
  if (failed(node) || !binding || binding.getLowering() != "ct.full" ||
      failed(fill) || !resultType || failed(shape))
    return operation.emitOpError("lacks a mechanical cuTile full binding");
  std::string dtype = dtypeName(resultType.getElementType(), operation);
  if (dtype.empty())
    return failure();
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.full(" + *shape + ", " + fill->str() +
       ", dtype=" + dtype + ")");
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
  if (failed(node) || !binding || binding.getLowering() != "ct.zeros" ||
      !resultType || failed(shape))
    return operation.emitOpError("lacks a mechanical cuTile zeros binding");
  std::string dtype = dtypeName(resultType.getElementType(), operation);
  if (dtype.empty())
    return failure();
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.zeros(" + *shape + ", dtype=" + dtype + ")");
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
      binding.getLowering() == "ct.indirect_gather") {
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
    line(result + " = ct.gather(" + (*view)->argument->name + ", " +
         addressIndex(*index) + ", check_bounds=True, padding_value=" +
         fill->str() + ")");
    line(result + " = ct.where(member_mask & " + valid->str() + ", " +
         result + ", " + fill->str() + ")");
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
    return operation.emitOpError("lacks a mechanical cuTile gather binding");
  std::string result = makeResultName(operation, 0);
  std::string expanded =
      source->str() + (appendAxis ? "[:, None]" : "[None, :]");
  line(result + " = ct.where(" + valid->str() + ", " + expanded + ", " +
       fill->str() + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitMembers(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "members emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  if (failed(node) || !binding || binding.getLowering() != "ct.members" ||
      operation.getNumResults() != 1)
    return operation.emitOpError("lacks a cuTile members binding");
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
  if (ragged.indices) {
    line(result + " = ct.gather(" + ragged.indices->argument->name + ", " +
         addressIndex(position) + ", check_bounds=True, padding_value=0)");
    line(result + " = ct.where(" + valid + ", " + result + ", 0)");
  } else {
    line(result + " = ct.where(" + valid + ", " + position + ", 0)");
  }
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical cuTile stream binding");
  Block &body = operation.getRegion(0).front();
  bool hasStop = static_cast<bool>(binding.getStopNodeAttr());
  bool hasRuntimeExtent = static_cast<bool>(
      operation.getAttrOfType<IntegerAttr>("intent.extent_operand_index"));
  if (operation.getNumOperands() !=
          operation.getNumResults() + 1 + (hasRuntimeExtent ? 1 : 0) +
              (hasStop ? 1 : 0) ||
      body.getNumArguments() != operation.getNumResults() + 1)
    return operation.emitOpError("has inconsistent cuTile stream state");
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
    line("sequence_begin_" + raggedSuffix + " = ct.gather(" +
         ragged.offsets->argument->name + ", " + addressIndex(outer) +
         ", padding_value=0)");
    line("sequence_end_" + raggedSuffix + " = ct.gather(" +
         ragged.offsets->argument->name + ", " + addressIndex(outer + " + 1") +
         ", padding_value=0)");
    line("sequence_length_" + raggedSuffix + " = sequence_end_" +
         raggedSuffix + " - sequence_begin_" + raggedSuffix);
  }
  FailureOr<std::string> logicalExtent =
      streamDomain ? dimensionName(*streamDomain)
                   : FailureOr<std::string>(failure());
  if (failed(logicalExtent))
    return binding.emitOpError("has no logical ordered-axis extent");
  std::string streamExtent = raggedStream
                                 ? "sequence_length_" + raggedSuffix
                                 : dimensionOwners.lookup(*logicalExtent);
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
      streamExtent = "min((" + block + " + 1) * " +
                     stopAxis->getTile().str() + ", " + streamExtent + ")";
    } else if (stopAxis->getNode() != binding.getAxisNode()) {
      return operation.emitOpError(
          "has no cuTile spelling for its planned logical stream stop");
    }
  }
  std::string streamTile = "stream_tile_" + std::to_string(*node);
  line("for " + streamTile + " in range(" + addressIndex("0") + ", " +
       addressIndex("ct.cdiv(" + streamExtent + ", " +
                    binding.getTile().str() + ")") +
       ", " + addressIndex("1") + "):");
  ++indentation;
  if (raggedStream) {
    auto runtimes = raggedRuntimesByAxis.find(binding.getAxisNode());
    if (runtimes == raggedRuntimesByAxis.end() ||
        runtimes->second.size() != 1)
      return binding.emitOpError("has no unique ragged stream runtime");
    RaggedRuntime &ragged = raggedRuntimes[runtimes->second.front()];
    std::string offsets = "axis_index_" + std::to_string(binding.getAxisNode());
    line(offsets + " = sequence_begin_" + raggedSuffix + " + " + streamTile +
         " * " + binding.getTile().str() + " + " +
         addressIndex("ct.arange(" + binding.getTile().str() +
                      ", dtype=ct.int32)"));
    line(offsets + " = ct.where(" + offsets + " < sequence_end_" +
         raggedSuffix + ", " + offsets + ", " +
         ragged.membersView->argument->name +
         ".shape[0])");
    axisIndices[binding.getAxisNode()] = offsets;
  } else {
    axisIndices[binding.getAxisNode()] = streamTile;
  }
  valueNames[body.getArgument(0)] = streamTile;
  return success();
}

LogicalResult SourceEmitter::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  if (carriers == streamCarriers.end())
    return operation.emitOpError("has no active cuTile stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (terminator.getName().getStringRef() != "intent.yield" ||
      terminator.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("does not yield every cuTile stream state");
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
  if (failed(node) || !binding || binding.getLowering() != "ct.mma" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a cuTile contraction binding");
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
    std::string reduction = stageReductionDimensions.lookup(stage);
    std::string result = makeResultName(operation, 0);
    line(result +
         " = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)");
    line("for k_tile in range(ct.cdiv(" + reduction + ", TILE_SIZE_K)):");
    ++indentation;
    line("offs_reduction = " + addressIndex("k_tile") +
         " * TILE_SIZE_K + " +
         addressIndex("ct.arange(TILE_SIZE_K, dtype=ct.int32)"));

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
      line(lhs + " = ct.gather(" + (*lhsView)->argument->name + ", (" +
           addressIndex(rows->str() + "[:, None]") + ", " +
           addressIndex("offs_reduction[None, :]") + "), "
           "check_bounds=True, padding_value=0.0)");
      line(lhs + " = ct.where(member_mask[:, None], " + lhs + ", 0.0)");
    } else {
      auto workspace = workspaceNames.find(operation.getOperand(0));
      if (workspace == workspaceNames.end())
        return operation.emitOpError(
            "staged contraction input has no materialized workspace");
      lhs = "stage_input";
      line(lhs + " = ct.gather(" + workspace->second +
           ", (" + addressIndex("safe_member_offsets[:, None]") + ", " +
           addressIndex("offs_reduction[None, :]") + "), "
           "check_bounds=True, padding_value=0.0)");
    }
    std::string rhs = makeResultName(*rhsLoad, 0);
    line(rhs + " = ct.load(" + (*rhsView)->argument->name +
         ", index=(" + addressIndex("expert") + ", " +
         addressIndex("k_tile") + ", " + addressIndex("bid_feature") +
         "), shape=(1, TILE_SIZE_K, "
         "TILE_SIZE_N), padding_mode=ct.PaddingMode.ZERO).reshape((TILE_SIZE_K, "
         "TILE_SIZE_N))");
    if (promoteToF32 && !lhsElement.isF32())
      line(lhs + " = " + lhs + ".astype(ct.tfloat32)");
    if (promoteToF32 && !rhsElement.isF32())
      line(rhs + " = " + rhs + ".astype(ct.tfloat32)");
    line(result + " = ct.mma(" + lhs + ", " + rhs + ", " + result + ")");
    --indentation;
    bindResult(operation, 0, result);
    return success();
  }
  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (!lhsLoad && !rhsLoad) {
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    FailureOr<std::string> shape = emitTensorShape(operation, 0);
    if (failed(lhs) || failed(rhs) || failed(shape))
      return failure();
    std::string lhsExpression = lhs->str();
    std::string rhsExpression = rhs->str();
    if (orientation->lhsTranspose)
      lhsExpression = "ct.transpose(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = "ct.transpose(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.full(" + *shape +
         ", 0.0, dtype=ct.float32)");
    line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression +
         ", " + result + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (!lhsLoad || !rhsLoad)
    return operation.emitOpError(
        "deferred cuTile contraction has inconsistent operand residency");
  FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  FailureOr<plan::AxisOp> reductionAxis =
      target::emission::contractionReductionAxis(planIndex, *lhsLoad, *rhsLoad,
                                                 operation);
  if (failed(reductionAxis))
    return failure();
  auto physicalReductionAxis = [&](Operation &load) -> FailureOr<int64_t> {
    FailureOr<SmallVector<target::IndexTerm>> relation =
        target::parseIndexRelation(load);
    if (failed(relation))
      return failure();
    std::optional<int64_t> result;
    for (auto [axisNumber, term] : llvm::enumerate(*relation)) {
      if ((term.kind != "region_index" && term.kind != "value_index") ||
          term.operands.size() != 1 || !term.operands.front())
        continue;
      FailureOr<plan::AxisOp> axis =
          resolveAxis(load.getOperand(*term.operands.front()), load);
      if (failed(axis))
        return failure();
      if (axis->getNode() != reductionAxis->getNode())
        continue;
      if (result)
        return load.emitOpError(
            "maps its contraction reduction axis more than once");
      result = axisNumber;
    }
    if (!result)
      return load.emitOpError(
          "does not map its contraction reduction axis to a source dimension");
    return *result;
  };
  FailureOr<int64_t> lhsReductionDimension = physicalReductionAxis(*lhsLoad);
  axisIndices[reductionAxis->getNode()] = "k_tile";
  FailureOr<std::string> lhsIndex = indexTuple(*lhsLoad, false);
  FailureOr<std::string> rhsIndex = indexTuple(*rhsLoad, false);
  FailureOr<std::string> lhsShape = tileShape(*lhsLoad);
  FailureOr<std::string> rhsShape = tileShape(*rhsLoad);
  bool permuteLhs =
      orientation->lhsTranspose && (*lhsView)->tensor.getRank() > 2;
  bool permuteRhs =
      orientation->rhsTranspose && (*rhsView)->tensor.getRank() > 2;
  FailureOr<std::string> lhsResultShape =
      emitTensorShape(*lhsLoad, 0, permuteLhs);
  FailureOr<std::string> rhsResultShape =
      emitTensorShape(*rhsLoad, 0, permuteRhs);
  if (failed(lhsView) || failed(rhsView) || failed(lhsIndex) ||
      failed(rhsIndex) || failed(lhsShape) || failed(rhsShape) ||
      failed(lhsResultShape) || failed(rhsResultShape) ||
      failed(lhsReductionDimension))
    return failure();

  std::string result = makeResultName(operation, 0);
  line("num_tiles_k = ct.num_tiles(" + (*lhsView)->argument->name +
       ", axis=" + std::to_string(*lhsReductionDimension) +
       ", shape=" + *lhsShape + ")");
  line(result +
       " = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)");
  std::string operandDtype =
      dtypeName((*lhsView)->tensor.getElementType(), operation);
  if (operandDtype.empty())
    return failure();
  line("for k_tile in range(num_tiles_k):");
  ++indentation;
  std::string lhs = makeResultName(*lhsLoad, 0);
  std::string rhs = makeResultName(*rhsLoad, 0);
  auto emitOperand = [&](StringRef name, ABIView &view, StringRef index,
                         StringRef shape, StringRef resultShape,
                         bool transpose) {
    std::string physical = name.str() + "_physical";
    line(physical + " = ct.load(" + view.argument->name + ", index=" +
         index.str() + ", shape=" + shape.str() +
         ", padding_mode=ct.PaddingMode.ZERO)");
    if (transpose && view.tensor.getRank() > 2) {
      std::string permutation = "(";
      for (int64_t axis = 0; axis < view.tensor.getRank(); ++axis) {
        if (axis)
          permutation += ", ";
        int64_t projected = axis;
        if (axis == view.tensor.getRank() - 2)
          projected = axis + 1;
        else if (axis == view.tensor.getRank() - 1)
          projected = axis - 1;
        permutation += std::to_string(projected);
      }
      permutation += ")";
      line(physical + " = ct.permute(" + physical + ", " + permutation + ")");
    }
    line(name.str() + " = " + physical + ".reshape(" + resultShape.str() +
         ").astype(" + operandDtype + ")");
  };
  emitOperand(lhs, **lhsView, *lhsIndex, *lhsShape, *lhsResultShape, permuteLhs);
  emitOperand(rhs, **rhsView, *rhsIndex, *rhsShape, *rhsResultShape, permuteRhs);
  std::string lhsExpression = lhs;
  std::string rhsExpression = rhs;
  if (orientation->lhsTranspose && (*lhsView)->tensor.getRank() == 2)
    lhsExpression = "ct.transpose(" + lhsExpression + ")";
  if (orientation->rhsTranspose && (*rhsView)->tensor.getRank() == 2)
    rhsExpression = "ct.transpose(" + rhsExpression + ")";
  line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression + ", " +
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
  FailureOr<StringRef> stored =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<std::string> physicalFill =
      transferPhysicalExtentFill(operation);
  bool expanded = succeeded(physicalFill) && !physicalFill->empty();
  bool scatter = boundary.getAccess() == "scatter" ||
                 expanded;
  FailureOr<std::string> indices = indexTuple(operation, scatter);
  if (!valueIndex || failed(node) || !boundary || failed(stored) ||
      failed(view) || failed(physicalFill) || failed(indices))
    return operation.emitOpError("lacks a cuTile store binding");
  if (scatter)
    line("ct.scatter(" + (*view)->argument->name + ", " + *indices + ", " +
         stored->str() + ", check_bounds=True)");
  else if (boundary.getAccess() == "store") {
    FailureOr<unsigned> storedRank = emittedTensorRank(operation, true);
    if (failed(storedRank))
      return failure();
    std::string tile = stored->str();
    if (*storedRank != static_cast<unsigned>((*view)->tensor.getRank())) {
      FailureOr<std::string> physicalShape = tileShape(operation);
      if (failed(physicalShape))
        return failure();
      tile += ".reshape(" + *physicalShape + ")";
    }
    SmallVector<std::string> scalarBounds;
    if (boundary.getCheckBounds()) {
      FailureOr<SmallVector<target::IndexTerm>> relation =
          target::parseIndexRelation(operation);
      if (failed(relation) || relation->size() != (*view)->shape.size())
        return failure();
      for (auto [axisNumber, term] : llvm::enumerate(*relation)) {
        if (term.kind != "value_index" || term.operands.size() != 1 ||
            !term.operands.front())
          continue;
        Value indexed = operation.getOperand(*term.operands.front());
        if (isa<RankedTensorType>(indexed.getType()))
          continue;
        FailureOr<target::ScalarIndexSource> source =
            target::traceScalarIndexSource(indexed, operation);
        FailureOr<StringRef> exact =
            lookupValue(operation, *term.operands.front());
        if (failed(source) || failed(exact))
          return failure();
        if (source->domain && source->transformed) {
          std::string extent = (*view)->shape[axisNumber];
          if (!planIndex.components.reusedAxes.empty()) {
            if (roleDimensions.lookup("program_0") == extent)
              extent = "N_ROWS";
            else if (roleDimensions.lookup("lane_0") == extent)
              extent = "DIM_COLS";
          }
          scalarBounds.push_back("(0 <= (" + exact->str() + ") < " +
                                 extent + ")");
        }
      }
    }
    if (!scalarBounds.empty()) {
      std::string predicate = scalarBounds.front();
      for (StringRef next : llvm::drop_begin(scalarBounds))
        predicate += " and " + next.str();
      line("if " + predicate + ":");
      ++indentation;
    }
    line("ct.store(" + (*view)->argument->name + ", index=" + *indices +
         ", tile=" + tile + ")");
    if (!scalarBounds.empty())
      --indentation;
  } else
    return boundary.emitOpError("is not a store-like cuTile access");
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
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored) ||
      (*view)->tensor.getRank() != 2)
    return operation.emitOpError("lacks a mechanical cuTile unique store");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  line("unique_rows = ct.where(member_mask, " + rows->str() + ", " +
       (*view)->argument->name + ".shape[0])");
  line("ct.scatter(" + (*view)->argument->name +
       ", (" + addressIndex("unique_rows[:, None]") + ", " +
       addressIndex("offs_feature[None, :]") + "), " + stored->str() +
       ", check_bounds=True)");
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
    return operation.emitOpError("lacks a mechanical cuTile atomic merge");
  if (planIndex.stages.empty()) {
    FailureOr<std::string> indices = indexTuple(operation, true);
    if (failed(indices))
      return failure();
    line("ct.atomic_add(" + (*view)->argument->name + ", " + *indices + ", " +
         stored->str() +
         ", check_bounds=True, memory_order=ct.MemoryOrder.RELAXED, "
         "memory_scope=ct.MemoryScope.DEVICE)");
    return success();
  }
  if (!valueIndex || failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "value_index" ||
      (*relation)[0].operands.size() != 1 ||
      !(*relation)[0].operands.front() ||
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored) ||
      (*view)->tensor.getRank() != 2)
    return operation.emitOpError("lacks a mechanical cuTile atomic merge");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  line("atomic_rows = ct.where(member_mask, " + rows->str() + ", " +
       (*view)->argument->name + ".shape[0])");
  line("ct.atomic_add(" + (*view)->argument->name +
       ", (" + addressIndex("atomic_rows[:, None]") + ", " +
       addressIndex("offs_feature[None, :]") + "), " + stored->str() +
       ", check_bounds=True, memory_order=ct.MemoryOrder.RELAXED, "
       "memory_scope=ct.MemoryScope.DEVICE)");
  return success();
}

} // namespace intent::cutile::emission

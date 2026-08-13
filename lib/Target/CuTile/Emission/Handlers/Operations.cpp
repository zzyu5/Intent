#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Analysis/StructuredControl.h"
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
  for (StringRef name : {"intent.domain", "intent.domain_product",
                         "intent.make_record",
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
      expression = number < 0 ? "-math.inf" : "math.inf";
    else if (std::isnan(number))
      expression = "math.nan";
    else
      expression = std::to_string(number);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else if (operation.getResult(0).getType().isInteger(64))
      expression = "ct.full((), " + std::to_string(integer.getInt()) +
                   ", dtype=ct.int64).item()";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported cuTile constant value");
  }
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitExtract(Operation &operation) {
  FailureOr<Value> field = target::resolveRecordField(operation);
  if (failed(field))
    return failure();
  auto found = valueNames.find(*field);
  if (found == valueNames.end())
    return operation.emitOpError("record field has no emitted cuTile SSA value");
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
  for (const auto &entry : planIndex.axesByRole) {
    plan::AxisOp axis = entry.getValue();
    if (!entry.getKey().starts_with("lane_") ||
        !axisIndices.lookup(axis.getNode()).empty())
      continue;
    std::string extent = roleDimensions.lookup(entry.getKey());
    if (extent.empty())
      return axis.emitOpError("has no cuTile lane extent");
    axisIndices[axis.getNode()] = "0";
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
          "staged cuTile ownership requires one logical axis");
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
    valueNames[argument] = outer ? "expert" : "member_offsets";
    return success();
  }

  if (!planIndex.components.reusedAxes.empty()) {
    if (axes.size() != 1)
      return operation.emitOpError(
          "worker-reused cuTile ownership requires one logical axis");
    BlockArgument argument = body.getArgument(0);
    plan::AxisOp axis = axes.front();
    if (&operation != programRoot ||
        axis.getNode() != planIndex.components.reusedAxes.front().getNode())
      return operation.emitOpError("is not the worker-reused program axis");
    int64_t workerAxis = axis.getWorkerAxis();
    line("program_start = " +
         addressIndex("ct.bid(" + std::to_string(workerAxis) + ")"));
    line("program_step = " +
         addressIndex("ct.num_blocks(" + std::to_string(workerAxis) + ")"));
    line(vectorIndex + " = " +
         addressIndex("ct.arange(TILE_SIZE, dtype=ct.int32)"));
    line("for " + programIndex +
         " in range(program_start, N_ROWS, program_step):");
    ++indentation;
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
                                  ? "has no persistent cuTile program index"
                                  : "has no emitted per-axis program index");
    if (target::emission::isPackedScalarAxis(axis)) {
      FailureOr<std::string> tile = physicalAxisTile(axis);
      if (failed(tile))
        return failure();
      value = addressIndex(value) + " * " + *tile + " + " +
              addressIndex("ct.arange(" + *tile + ", dtype=ct.int32)");
      axisIndices[axis.getNode()] = value;
    }
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
  FailureOr<SmallVector<Operation *>> domains =
      operation.getNumOperands() > 0
          ? target::expandDomainSource(operation.getOperand(0), operation)
          : FailureOr<SmallVector<Operation *>>(failure());
  bool ordered = operation.getName().getStringRef() == "intent.ordered";
  if (failed(node) || failed(domains) || domains->empty() ||
      (!ordered && domains->size() != 1) || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical cuTile sequential loop");
  Block &body = operation.getRegion(0).front();
  if (body.getNumArguments() != domains->size() + operation.getNumResults())
    return operation.emitOpError("does not match its cuTile sequential axes");
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    if (failed(initial))
      return failure();
    std::string carrier =
        uniqueName("loop_state_" + std::to_string(index), *node);
    line(carrier + " = " + initial->str());
    carriers.push_back(carrier);
    valueNames[body.getArgument(domains->size() + index)] = carrier;
  }
  loopCarriers[&operation] = carriers;
  for (auto [index, domain] : llvm::enumerate(*domains)) {
    if (!domain)
      return operation.emitOpError("has a non-canonical cuTile loop axis");
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
        operation.emitOpError("has no emitted cuTile loop bound");
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
      line(begin + " = ct.gather(" + ragged.offsets->argument->name + ", " +
           addressIndex(outer) + ", padding_value=0)");
      line(end + " = ct.gather(" + ragged.offsets->argument->name + ", " +
           addressIndex(outer + " + 1") + ", padding_value=0)");
      start = begin;
      stop = end;
    } else {
      return operation.emitOpError("has a non-canonical cuTile loop axis");
    }
    if (failed(start) || failed(stop) || failed(step))
      return failure();
    if (ordered && ((!raggedAxis && *start != "0") || *step != "1"))
      return operation.emitOpError(
          "cuTile ordered traversal requires a zero-based unit-step domain");
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
            "has an invalid two-level cuTile ordered traversal");
      std::string chunk = iterator + "_chunk";
      line("for " + chunk + " in range(ct.cdiv(" + *stop + " - " + *start +
           ", " +
           outer->getTile().str() + ")):");
      ++indentation;
      line("for " + iterator + " in range(" + *start + " + " + chunk +
           " * " + outer->getTile().str() + ", min(" + *start + " + (" +
           chunk + " + 1) * " + outer->getTile().str() + ", " + *stop +
           ")):");
      ++indentation;
    } else {
      line("for " + iterator + " in range(" + *start + ", " + *stop +
           ", " + *step + "):");
      ++indentation;
    }
  }
  return success();
}

LogicalResult SourceEmitter::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active cuTile sequential loop");
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

LogicalResult SourceEmitter::enterWhile(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "while emission");
  if (failed(node) || operation.getNumRegions() != 2 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      !llvm::hasSingleElement(operation.getRegion(1)) ||
      operation.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("lacks a mechanical cuTile scalar while");
  Block &before = operation.getRegion(0).front();
  Block &after = operation.getRegion(1).front();
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index);
    if (failed(initial))
      return failure();
    std::string carrier =
        uniqueName("while_state_" + std::to_string(index), *node);
    line(carrier + " = " + initial->str());
    carriers.push_back(carrier);
    valueNames[before.getArgument(index)] = carrier;
    valueNames[after.getArgument(index)] = carrier;
  }
  whileCarriers[&operation] = std::move(carriers);
  return success();
}

LogicalResult SourceEmitter::emitCondition(Operation &operation) {
  Operation *owner = target::whileConditionOwner(operation);
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (!owner || failed(condition) || operation.getNumOperands() !=
                                          owner->getNumResults() + 1)
    return operation.emitOpError("does not match a cuTile scalar while");
  line("while " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult SourceEmitter::leaveWhile(Operation &operation) {
  auto carriers = whileCarriers.find(&operation);
  if (carriers == whileCarriers.end())
    return operation.emitOpError("has no active cuTile scalar while");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index]);
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
      return operation.emitOpError("does not match its cuTile loop state");
    for (unsigned index = 0; index < operation.getNumOperands(); ++index) {
      FailureOr<StringRef> yielded = lookupValue(operation, index);
      if (failed(yielded))
        return failure();
      line(carriers->second[index] + " = " + yielded->str());
    }
    return success();
  }
  if (name == "intent.while") {
    auto carriers = whileCarriers.find(owner);
    if (carriers == whileCarriers.end() ||
        carriers->second.size() != operation.getNumOperands())
      return operation.emitOpError("does not match its cuTile while state");
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
  if (binding && binding.getSpace() == "private_workspace") {
    if (failed(info) || info->shape.empty() ||
        !workspaceNames.count(operation.getResult(0)))
      return operation.emitOpError(
          "lacks a planned cuTile private workspace parameter");
    return success();
  }
  if (binding && binding.getSpace() == "private_vector") {
    std::string dtype = succeeded(info) ? dtypeName(info->elementType, operation)
                                        : std::string();
    if (failed(info) || failed(initializer) ||
        dtype.empty())
      return operation.emitOpError(
          "private cuTile vectors require a supported static shape");
    std::string base = makeResultName(operation, 0);
    FailureOr<int64_t> elements =
        target::logicalBufferElementCount(*info, operation);
    if (failed(elements))
      return failure();
    int64_t physicalExtent = 1;
    while (physicalExtent < *elements)
      physicalExtent *= 2;
    line(base + "_lanes = ct.arange(" +
         std::to_string(physicalExtent) + ", dtype=ct.int32)");
    line(base + " = ct.full((" + std::to_string(physicalExtent) +
         ",), " + initializer->str() + ", dtype=" + dtype + ")");
    vectorBuffers[operation.getResult(0)] = base;
    return success();
  }
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
  auto workspace = operation.getNumOperands() > 0
                       ? workspaceNames.find(operation.getOperand(0))
                       : workspaceNames.end();
  if (workspace != workspaceNames.end()) {
    FailureOr<std::string> index = privateWorkspaceIndex(operation);
    if (failed(index) || operation.getNumResults() != 1)
      return operation.emitOpError(
          "lacks a mechanical cuTile private-workspace load");
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.load(" + workspace->second + ", index=" + *index +
         ", shape=()).item()");
    bindResult(operation, 0, result);
    return success();
  }
  auto vector = operation.getNumOperands() > 0
                    ? vectorBuffers.find(operation.getOperand(0))
                    : vectorBuffers.end();
  Operation *vectorOwner = operation.getNumOperands() > 0
                               ? operation.getOperand(0).getDefiningOp()
                               : nullptr;
  FailureOr<target::LogicalBufferInfo> vectorInfo =
      vectorOwner ? target::getLogicalBufferInfo(*vectorOwner)
                  : FailureOr<target::LogicalBufferInfo>(failure());
  FailureOr<SmallVector<target::LogicalBufferIndex>> vectorIndices =
      succeeded(vectorInfo)
          ? target::getLogicalBufferIndices(operation, vectorInfo->shape.size())
          : FailureOr<SmallVector<target::LogicalBufferIndex>>(failure());
  FailureOr<target::LogicalBufferIndex> index =
      target::getLogicalBufferIndex(operation);
  if (vector != vectorBuffers.end()) {
    if (failed(vectorInfo) || failed(vectorIndices) ||
        operation.getNumResults() != 1)
      return operation.emitOpError("lacks a private cuTile vector load");
    std::string selected;
    for (auto [axis, logical] : llvm::enumerate(*vectorIndices)) {
      std::string component;
      if (logical.constant)
        component = std::to_string(*logical.constant);
      else {
        FailureOr<StringRef> dynamic = lookupValue(operation, *logical.operand);
        if (failed(dynamic))
          return failure();
        component = dynamic->str();
      }
      selected = selected.empty()
                     ? component
                     : "(" + selected + ") * " +
                           std::to_string(vectorInfo->shape[axis]) + " + (" +
                           component + ")";
    }
    std::string result = makeResultName(operation, 0);
    if (operation.getResult(0).getType().isInteger(1)) {
      line(result + " = ct.sum(ct.where(" + vector->second + "_lanes == " +
           selected + ", ct.astype(" + vector->second +
           ", ct.int32), 0), axis=0) != 0");
    } else {
      line(result + "_tile = ct.extract(" + vector->second + ", (" +
           addressIndex(selected) + ",), shape=(1,))");
      line(result + " = " + result + "_tile.item()");
    }
    bindResult(operation, 0, result);
    return success();
  }
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
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
  auto workspace = operation.getNumOperands() > 0
                       ? workspaceNames.find(operation.getOperand(0))
                       : workspaceNames.end();
  if (workspace != workspaceNames.end()) {
    Operation *buffer = operation.getOperand(0).getDefiningOp();
    FailureOr<target::LogicalBufferInfo> info =
        buffer ? target::getLogicalBufferInfo(*buffer)
               : FailureOr<target::LogicalBufferInfo>(failure());
    FailureOr<std::string> index = privateWorkspaceIndex(operation);
    FailureOr<StringRef> stored = lookupValue(operation, 1);
    std::string dtype = succeeded(info) ? dtypeName(info->elementType, operation)
                                        : std::string();
    if (failed(index) || failed(stored) || dtype.empty())
      return operation.emitOpError(
          "lacks a mechanical cuTile private-workspace store");
    line("ct.store(" + workspace->second + ", index=" + *index +
         ", tile=ct.full((), " + stored->str() + ", dtype=" + dtype + "))");
    return success();
  }
  auto vector = operation.getNumOperands() > 0
                    ? vectorBuffers.find(operation.getOperand(0))
                    : vectorBuffers.end();
  Operation *vectorOwner = operation.getNumOperands() > 0
                               ? operation.getOperand(0).getDefiningOp()
                               : nullptr;
  FailureOr<target::LogicalBufferInfo> vectorInfo =
      vectorOwner ? target::getLogicalBufferInfo(*vectorOwner)
                  : FailureOr<target::LogicalBufferInfo>(failure());
  FailureOr<SmallVector<target::LogicalBufferIndex>> vectorIndices =
      succeeded(vectorInfo)
          ? target::getLogicalBufferIndices(operation, vectorInfo->shape.size())
          : FailureOr<SmallVector<target::LogicalBufferIndex>>(failure());
  FailureOr<target::LogicalBufferIndex> index =
      target::getLogicalBufferIndex(operation);
  FailureOr<StringRef> stored = lookupValue(operation, 1);
  if (vector != vectorBuffers.end()) {
    if (failed(vectorInfo) || failed(vectorIndices) || failed(stored))
      return operation.emitOpError("lacks a private cuTile vector store");
    std::string selected;
    for (auto [axis, logical] : llvm::enumerate(*vectorIndices)) {
      std::string component;
      if (logical.constant)
        component = std::to_string(*logical.constant);
      else {
        FailureOr<StringRef> dynamic = lookupValue(operation, *logical.operand);
        if (failed(dynamic))
          return failure();
        component = dynamic->str();
      }
      selected = selected.empty()
                     ? component
                     : "(" + selected + ") * " +
                           std::to_string(vectorInfo->shape[axis]) + " + (" +
                           component + ")";
    }
    line(vector->second + " = ct.where(" + vector->second +
         "_lanes == " + selected + ", " + stored->str() + ", " +
         vector->second + ")");
    return success();
  }
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
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
        "does not resolve a planned cuTile private workspace");

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

FailureOr<std::string>
SourceEmitter::scanWorkspaceIndex(const plan::ScanOp &binding,
                                  StringRef logicalIndex,
                                  Operation &consumer) {
  std::string extent = scanExtents.lookup(binding.getNode());
  if (extent.empty())
    return binding.emitOpError("has no cuTile scan workspace binding");
  FailureOr<std::string> offset = target::emission::projectScanWorkspaceOffset(
      binding, extent, logicalIndex, planIndex, axisIndices, axisDimensions,
      consumer);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

FailureOr<std::string>
SourceEmitter::scanMaterializedIndex(Value value, StringRef logicalIndex,
                                     Operation &consumer) {
  auto materialized = scanMaterializedValues.find(value);
  if (materialized == scanMaterializedValues.end())
    return consumer.emitOpError("has no cuTile materialized scan value");
  FailureOr<std::string> offset = target::emission::projectScanWorkspaceOffset(
      materialized->second, scanExtents.lookup(materialized->second.getNode()),
      logicalIndex, planIndex, axisIndices, axisDimensions, consumer);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a cuTile load boundary");
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
  bool packedScalar =
      target::emission::hasPackedScalarDomain(planIndex, boundary);
  bool scanReplay = activeScanReplay >= 0;
  FailureOr<std::string> indices = indexTuple(
      operation, boundary.getAccess() == "gather" || packedScalar || scanReplay);
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
  if (boundary.getAccess() == "gather" || packedScalar || scanReplay) {
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
  if (activeScanReplay >= 0) {
    std::string replay = axisIndices.lookup(axis->getNode());
    if (replay.empty())
      return operation.emitOpError("has no active cuTile scan replay index");
    bindResult(operation, 0, replay);
    return success();
  }
  if (axis->hasRole("lane") && !axis->getReuseWorker() &&
      axis->getTileRole().starts_with("row_vector")) {
    FailureOr<std::string> tile = physicalAxisTile(*axis);
    if (failed(tile))
      return failure();
    bindResult(operation, 0,
               addressIndex("ct.arange(" + *tile +
                            ", dtype=ct.int32)"));
    return success();
  }
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
  FailureOr<std::string> tile = physicalAxisTile(*axis);
  if (failed(tile))
    return failure();
  std::string expression =
      directVector
          ? addressIndex(base)
          : addressIndex(base) + " * " + *tile + " + " +
                addressIndex("ct.arange(" + *tile +
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
  if (binding.getLowering() == "ct.any_via_max" ||
      binding.getLowering() == "ct.all_via_min") {
    StringRef reduction = binding.getLowering() == "ct.any_via_max" ? "ct.max"
                                                                      : "ct.min";
    line(result + " = ct.astype(" + reduction.str() + "(" + operand->str() +
         ", " + std::to_string(binding.getAxis()) + ", keepdims=" +
         (binding.getKeepDims() ? "True" : "False") + "), ct.bool_)");
  } else {
    line(result + " = " + binding.getLowering().str() + "(" + operand->str() +
         ", " + std::to_string(binding.getAxis()) + ", keepdims=" +
         (binding.getKeepDims() ? "True" : "False") + ")");
  }
  bindResult(operation, 0, result);
  return success();
}

LogicalResult SourceEmitter::emitScan(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "scan emission");
  plan::ScanOp binding =
      succeeded(node) ? planIndex.scans.lookup(*node) : plan::ScanOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (binding && binding.getResultSpace() == "private_fragment") {
    if (failed(node) || binding.getLowering() != "ct.cumsum" ||
        failed(operand) || operation.getNumResults() != 1)
      return operation.emitOpError("lacks a fragment cuTile scan binding");
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.cumsum(" + operand->str() + ", axis=" +
         std::to_string(binding.getAxis()) + ")");
    bindResult(operation, 0, result);
    return success();
  }
  plan::AxisOp axis = binding ? planIndex.axes.lookup(binding.getAxisNode())
                              : plan::AxisOp();
  const target::emission::RangeBinding *range =
      axis ? axis.getRange("traversal", 0) : nullptr;
  std::string extent = binding ? scanExtents.lookup(binding.getNode())
                               : std::string();
  if (failed(node) || !binding || binding.getLowering() != "ct.cumsum" ||
      !axis || !range || extent.empty() ||
      operation.getNumResults() != 1)
    return operation.emitOpError("lacks a mechanical cuTile scan binding");
  std::string result = makeResultName(operation, 0);
  std::string carry = result + "_carry";
  std::string block = result + "_block";
  std::string offsets = result + "_offsets";
  std::string values = result + "_values";
  std::string scanned = result + "_scanned";
  line(carry + " = 0");
  line("for " + block + " in range(ct.cdiv(" + extent + ", " +
       range->getTile().str() + ")):");
  ++indentation;
  line(offsets + " = " + addressIndex(block) + " * " +
       range->getTile().str() + " + " +
       addressIndex("ct.arange(" + range->getTile().str() +
                    ", dtype=ct.int32)"));
  if (failed(replayScanProducers(binding, offsets)))
    return failure();
  operand = lookupValue(operation, 0);
  if (failed(operand))
    return failure();
  line(values + " = " + operand->str());
  for (int64_t valueID : binding.getMaterializedValues()) {
    Value value = kernel.values.lookup(valueID);
    auto name = valueNames.find(value);
    FailureOr<std::string> workspaceIndex =
        scanMaterializedIndex(value, offsets, operation);
    if (name == valueNames.end() || failed(workspaceIndex))
      return binding.emitOpError(
          "has no emitted cuTile value for scan materialization");
    line("ct.scatter(" + workspaceNames.lookup(value) + ", " +
         *workspaceIndex + ", " + name->second + ", check_bounds=True)");
  }
  line(scanned + " = ct.cumsum(ct.where(" + offsets + " < " + extent +
       ", " + values + ", 0), axis=" +
       std::to_string(binding.getAxis()) + ") + " + carry);
  FailureOr<std::string> workspaceIndex =
      scanWorkspaceIndex(binding, offsets, operation);
  Operation *scan = kernel.nodes.lookup(binding.getNode());
  if (failed(workspaceIndex) || !scan || scan->getNumResults() != 1)
    return failure();
  line("ct.scatter(" + workspaceNames.lookup(scan->getResult(0)) + ", " +
       *workspaceIndex + ", " + scanned + ", check_bounds=True)");
  std::string lastLane = "ct.minimum(" + range->getTile().str() + " - 1, " +
                         extent + " - " + block + " * " +
                         range->getTile().str() + " - 1)";
  line(carry + " = ct.extract(" + scanned + ", (" +
       lastLane + ",), shape=(1,)).item()");
  --indentation;
  valueNames.erase(operation.getResult(0));
  return success();
}

LogicalResult SourceEmitter::replayScanProducers(const plan::ScanOp &binding,
                                                 StringRef offsets) {
  if (!operationRegistry())
    return binding.emitOpError("has no cuTile operation registry for scan replay");
  auto oldAxis = axisIndices.find(binding.getAxisNode());
  std::optional<std::string> savedAxis =
      oldAxis == axisIndices.end() ? std::nullopt
                                   : std::optional<std::string>(oldAxis->second);
  axisIndices[binding.getAxisNode()] = offsets.str();
  const auto *range = planIndex.axes.lookup(binding.getAxisNode())
                          .getRange("traversal", 0);
  if (!range)
    return binding.emitOpError("has no cuTile scan traversal tile");
  scanAxisTiles[binding.getAxisNode()] = range->getTile().str();
  llvm::StringMap<std::optional<std::string>> savedTiles;
  for (int64_t node : binding.getProducers()) {
    Operation *producer = kernel.nodes.lookup(node);
    if (!producer)
      return binding.emitOpError("references an unknown cuTile scan producer");
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
        regionTiles[label.getValue()] = range->getTile().str();
      }
    }
  }
  activeScanReplay = binding.getNode();
  auto restore = [&]() {
    activeScanReplay = -1;
    scanAxisTiles.erase(binding.getAxisNode());
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
  else if (binding.getLowering() == "python_not")
    expression = "(" + operand->str() + ") == False";
  else if (binding.getLowering() == "python_sigmoid")
    expression = "1.0 / (1.0 + ct.exp(-(" + operand->str() + ")))";
  else
    expression = binding.getLowering().str() + "(" + operand->str() + ")";
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
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
  else if ((binding.getLowering() == "python_floor_divide" ||
            binding.getLowering() == "python_remainder") &&
           binding.getNonnegativeOperands()) {
    StringRef symbol = binding.getLowering() == "python_floor_divide" ? "//" : "%";
    expression = "(" + lhs->str() + ") " + symbol.str() + " (" + rhs->str() + ")";
  } else if (binding.getLowering() == "python_floor_divide" ||
           binding.getLowering() == "python_remainder") {
    Type elementType = operation.getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(elementType))
      elementType = tensor.getElementType();
    std::string resultDtype = dtypeName(elementType, operation);
    if (resultDtype.empty())
      return failure();
    std::string wideLhs = result + "_wide_lhs";
    std::string wideRhs = result + "_wide_rhs";
    std::string lhsMagnitude = result + "_lhs_magnitude";
    std::string rhsMagnitude = result + "_rhs_magnitude";
    std::string quotientMagnitude = result + "_quotient_magnitude";
    std::string quotient = result + "_truncating_quotient";
    std::string remainder = result + "_remainder";
    std::string adjust = result + "_adjust";
    line(wideLhs + " = ct.astype(" + lhs->str() + ", ct.int64)");
    line(wideRhs + " = ct.astype(" + rhs->str() + ", ct.int64)");
    line(lhsMagnitude + " = ct.where(" + wideLhs + " < 0, -" + wideLhs +
         ", " + wideLhs + ")");
    line(rhsMagnitude + " = ct.where(" + wideRhs + " < 0, -" + wideRhs +
         ", " + wideRhs + ")");
    line(quotientMagnitude + " = " + lhsMagnitude + " // " + rhsMagnitude);
    line(quotient + " = ct.where((" + wideLhs + " < 0) != (" + wideRhs +
         " < 0), -" + quotientMagnitude + ", " + quotientMagnitude + ")");
    line(remainder + " = " + wideLhs + " - " + quotient + " * " + wideRhs);
    line(adjust + " = (" + remainder + " != 0) & ((" + remainder +
         " < 0) != (" + wideRhs + " < 0))");
    expression = binding.getLowering() == "python_floor_divide"
                     ? "ct.astype(" + quotient + " - ct.where(" + adjust +
                           ", 1, 0), " + resultDtype + ")"
                     : "ct.astype(" + remainder + " + ct.where(" + adjust +
                           ", " + wideRhs + ", 0), " + resultDtype + ")";
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
    else
      return operation.emitOpError("uses an unsupported cuTile binary lowering");
    expression = "(" + lhs->str() + ") " + symbol.str() + " (" +
                 rhs->str() + ")";
  }
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
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
  if (failed(node) || !binding || binding.getLowering() != "ct.where" ||
      operation.getNumResults() != 1 || failed(condition) || failed(trueValue) ||
      failed(falseValue))
    return operation.emitOpError("lacks a mechanical cuTile conditional binding");
  std::string result = makeResultName(operation, 0);
  std::string expression = "ct.where(" + condition->str() + ", " +
                           trueValue->str() + ", " + falseValue->str() + ")";
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
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
  std::string expression =
      binding.getLowering() == "ct.full_cast"
          ? "ct.full((), " + operand->str() + ", dtype=" + targetType + ")"
          : "ct.astype(" + operand->str() + ", " + targetType + ")";
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
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
  bool scalarFragmentGather =
      binding && binding.getLowering() == "ct.indirect_gather" &&
      isa<RankedTensorType>(operation.getOperand(0).getType()) &&
      !isa<RankedTensorType>(operation.getResult(0).getType()) &&
      succeeded(relation) && relation->size() == 1 &&
      (*relation)[0].kind == "value_index" &&
      (*relation)[0].operands.size() == 1 &&
      (*relation)[0].operands.front();
  if (scalarFragmentGather) {
    auto scanSource = scanResults.find(operation.getOperand(0));
    auto materializedSource =
        scanMaterializedValues.find(operation.getOperand(0));
    FailureOr<StringRef> source =
        scanSource == scanResults.end() &&
                materializedSource == scanMaterializedValues.end()
            ? lookupValue(operation, 0)
            : FailureOr<StringRef>(StringRef());
    FailureOr<StringRef> index =
        lookupValue(operation, *(*relation)[0].operands.front());
    FailureOr<StringRef> valid =
        validIndex ? lookupValue(operation, validIndex.getInt())
                   : FailureOr<StringRef>(failure());
    FailureOr<StringRef> fill =
        fillIndex ? lookupValue(operation, fillIndex.getInt())
                  : FailureOr<StringRef>(failure());
    if ((scanSource == scanResults.end() &&
         materializedSource == scanMaterializedValues.end() && failed(source)) ||
        failed(index) ||
        failed(valid) || failed(fill))
      return failure();
    std::string result = makeResultName(operation, 0);
    if (scanSource != scanResults.end()) {
      FailureOr<std::string> workspaceIndex = scanWorkspaceIndex(
          scanSource->second, index->str(), operation);
      Operation *scan = kernel.nodes.lookup(scanSource->second.getNode());
      if (failed(workspaceIndex) || !scan || scan->getNumResults() != 1)
        return failure();
      line(result + " = ct.where(" + valid->str() + ", ct.load(" +
           workspaceNames.lookup(scan->getResult(0)) + ", index=(" +
           *workspaceIndex + ",), shape=()).item(), " + fill->str() + ")");
    } else if (materializedSource != scanMaterializedValues.end()) {
      FailureOr<std::string> workspaceIndex = scanMaterializedIndex(
          operation.getOperand(0), index->str(), operation);
      if (failed(workspaceIndex))
        return failure();
      line(result + " = ct.where(" + valid->str() + ", ct.load(" +
           workspaceNames.lookup(operation.getOperand(0)) + ", index=(" +
           *workspaceIndex + ",), shape=()).item(), " + fill->str() + ")");
    } else {
      line(result + " = ct.where(" + valid->str() + ", ct.extract(" +
           source->str() + ", (" + addressIndex(index->str()) +
           ",), shape=(1,)).item(), " + fill->str() + ")");
    }
    bindResult(operation, 0, result);
    return success();
  }
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
  FailureOr<StringRef> valid =
      validIndex ? lookupValue(operation, validIndex.getInt())
                 : FailureOr<StringRef>(failure());
  FailureOr<StringRef> fill =
      fillIndex ? lookupValue(operation, fillIndex.getInt())
                : FailureOr<StringRef>(failure());
  if (failed(node) || !binding || failed(relation) || failed(valid) ||
      failed(fill))
    return operation.emitOpError("lacks a mechanical cuTile gather binding");
  if (binding.getLowering() == "ct.indirect_gather" &&
      isa<intent::ViewType>(operation.getOperand(0).getType())) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    FailureOr<std::string> indices = indexTuple(operation, true);
    if (failed(view) || failed(indices))
      return failure();
    StringRef padding = isa<IntegerType, IndexType>(
                            (*view)->tensor.getElementType())
                            ? "0"
                            : "0.0";
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.gather(" + (*view)->argument->name + ", " + *indices +
         ", check_bounds=True, padding_value=" + padding.str() + ")");
    line(result + " = ct.where(" + valid->str() + ", " + result + ", " +
         fill->str() + ")");
    bindResult(operation, 0, result);
    return success();
  }
  FailureOr<StringRef> source = lookupValue(operation, 0);
  bool expand = binding.getLowering() == "expand_dims" &&
                llvm::any_of(*relation, [](const target::IndexTerm &term) {
                  return term.kind == "new_axis";
                }) &&
                llvm::all_of(*relation, [](const target::IndexTerm &term) {
                  return term.kind == "full_slice" || term.kind == "new_axis";
                });
  if (!expand || failed(source))
    return operation.emitOpError("lacks a mechanical cuTile gather binding");
  std::string result = makeResultName(operation, 0);
  std::string expanded = source->str() + "[";
  for (auto [index, term] : llvm::enumerate(*relation)) {
    if (index)
      expanded += ", ";
    expanded += term.kind == "new_axis" ? "None" : ":";
  }
  expanded += "]";
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
    std::string offsets = "stream_axis_index_" + std::to_string(*node);
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
  if (failed(node) || !binding || binding.getLowering() != "ct.mma" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a cuTile contraction binding");
  FailureOr<target::emission::ContractionOrientation> orientation =
      target::emission::contractionOrientation(operation);
  if (failed(orientation))
    return failure();
  auto resultTensor = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  std::string accumulatorDtype =
      resultTensor ? dtypeName(resultTensor.getElementType(), operation) : "";
  if (accumulatorDtype.empty())
    return operation.emitOpError("has no supported cuTile accumulator dtype");
  if (!planIndex.stages.empty()) {
    if (orientation->batched)
      return operation.emitOpError(
          "cuTile staged batch contraction is not supported");
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
    line(result + " = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=" +
         accumulatorDtype + ")");
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
      lhsExpression = orientation->batched
                          ? "ct.transpose(" + lhsExpression + ", 1, 2)"
                          : "ct.transpose(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = orientation->batched
                          ? "ct.transpose(" + rhsExpression + ", 1, 2)"
                          : "ct.transpose(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.full(" + *shape + ", 0, dtype=" +
         accumulatorDtype + ")");
    line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression +
         ", " + result + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (!lhsLoad || !rhsLoad)
    return operation.emitOpError(
        "deferred cuTile contraction has inconsistent operand residency");
  if (orientation->batched)
    return operation.emitOpError(
        "deferred cuTile batch contraction is not supported");
  FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  auto transferBoundary = [&](Operation &load) -> FailureOr<plan::BoundaryOp> {
    FailureOr<int64_t> node =
        target::getNodeID(load, "deferred contraction transfer");
    plan::BoundaryOp boundary =
        succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
    if (failed(node) || !boundary ||
        (boundary.getAccess() != "load" && boundary.getAccess() != "gather"))
      return load.emitOpError(
          "has no load-or-gather physical transfer binding");
    return boundary;
  };
  FailureOr<plan::BoundaryOp> lhsBoundary = transferBoundary(*lhsLoad);
  FailureOr<plan::BoundaryOp> rhsBoundary = transferBoundary(*rhsLoad);
  FailureOr<target::emission::ContractionAxes> axes =
      target::emission::contractionAxes(planIndex, *lhsLoad, *rhsLoad,
                                        operation);
  if (failed(lhsView) || failed(rhsView) || failed(lhsBoundary) ||
      failed(rhsBoundary) || failed(axes))
    return failure();
  plan::AxisOp reductionAxis = axes->reduction;
  axisIndices[reductionAxis.getNode()] = "k_tile";
  bool gatherLhs = lhsBoundary->getAccess() == "gather";
  bool gatherRhs = rhsBoundary->getAccess() == "gather";
  FailureOr<std::string> lhsIndex = indexTuple(*lhsLoad, gatherLhs);
  FailureOr<std::string> rhsIndex = indexTuple(*rhsLoad, gatherRhs);
  FailureOr<std::string> lhsShape =
      gatherLhs ? FailureOr<std::string>(std::string()) : tileShape(*lhsLoad);
  FailureOr<std::string> rhsShape =
      gatherRhs ? FailureOr<std::string>(std::string()) : tileShape(*rhsLoad);
  bool permuteLhs =
      orientation->lhsTranspose && (*lhsView)->tensor.getRank() > 2;
  bool permuteRhs =
      orientation->rhsTranspose && (*rhsView)->tensor.getRank() > 2;
  FailureOr<std::string> lhsResultShape =
      emitTensorShape(*lhsLoad, 0, permuteLhs);
  FailureOr<std::string> rhsResultShape =
      emitTensorShape(*rhsLoad, 0, permuteRhs);
  FailureOr<std::string> lhsTile = physicalAxisTile(axes->lhsResult);
  FailureOr<std::string> rhsTile = physicalAxisTile(axes->rhsResult);
  if (failed(lhsIndex) || failed(rhsIndex) || failed(lhsShape) || failed(rhsShape) ||
      failed(lhsResultShape) || failed(rhsResultShape) || failed(lhsTile) ||
      failed(rhsTile))
    return failure();

  std::string result = makeResultName(operation, 0);
  std::string reductionExtent = roleDimensions.lookup(reductionAxis.getRole());
  if (reductionExtent.empty())
    return reductionAxis.emitOpError(
        "has no cuTile physical reduction extent binding");
  line("num_tiles_k = ct.cdiv(" + reductionExtent + ", " +
       reductionAxis.getTile().str() + ")");
  line(result + " = ct.full((" + *lhsTile + ", " + *rhsTile +
       "), 0, dtype=" + accumulatorDtype + ")");
  std::string operandDtype =
      dtypeName((*lhsView)->tensor.getElementType(), operation);
  if (operandDtype.empty())
    return failure();
  line("for k_tile in range(num_tiles_k):");
  ++indentation;
  std::string lhs = makeResultName(*lhsLoad, 0);
  std::string rhs = makeResultName(*rhsLoad, 0);
  auto emitOperand = [&](StringRef name, ABIView &view, StringRef index,
                         StringRef shape, StringRef resultShape, bool gather,
                         bool transpose) {
    std::string physical = name.str() + "_physical";
    if (gather) {
      StringRef padding = isa<IntegerType, IndexType>(view.tensor.getElementType())
                              ? "0"
                              : "0.0";
      line(physical + " = ct.gather(" + view.argument->name + ", " +
           index.str() + ", check_bounds=True, padding_value=" +
           padding.str() + ")");
    } else {
      line(physical + " = ct.load(" + view.argument->name + ", index=" +
           index.str() + ", shape=" + shape.str() +
           ", padding_mode=ct.PaddingMode.ZERO)");
    }
    line(name.str() + " = " + physical + ".reshape(" + resultShape.str() +
         ").astype(" + operandDtype + ")");
    if (transpose)
      line(name.str() + " = ct.transpose(" + name.str() + ")");
  };
  emitOperand(lhs, **lhsView, *lhsIndex, *lhsShape, *lhsResultShape, gatherLhs,
              permuteLhs);
  emitOperand(rhs, **rhsView, *rhsIndex, *rhsShape, *rhsResultShape, gatherRhs,
              permuteRhs);
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
                 expanded ||
                 target::emission::hasPackedScalarDomain(planIndex, boundary);
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
        if (source->hasDomain() && source->transformed) {
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
  if (failed(node) || !binding ||
      (binding.getAccess() != "store" && binding.getAccess() != "scatter") ||
      binding.getDefer() || !valueIndex ||
      failed(relation) || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical cuTile unique store");
  FailureOr<std::string> indices = failure();
  std::string mask;
  if (!planIndex.stages.empty()) {
    auto storedTensor = dyn_cast<RankedTensorType>(
        operation.getOperand(valueIndex.getInt()).getType());
    if (activeStages.size() != 1 || relation->size() != 2 ||
        (*relation)[0].kind != "value_index" ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front() ||
        !isa<RankedTensorType>(
            operation.getOperand(*(*relation)[0].operands.front()).getType()) ||
        (*relation)[1].kind != "full_slice" || !storedTensor ||
        storedTensor.getRank() != 2)
      return operation.emitOpError(
          "staged cuTile unique store requires one member-indexed matrix");
    FailureOr<StringRef> rows =
        lookupValue(operation, *(*relation)[0].operands.front());
    if (failed(rows))
      return failure();
    indices = "(" + addressIndex(rows->str() + "[:, None]") + ", " +
              addressIndex("offs_feature[None, :]") +
              ")";
    mask = "member_mask[:, None] & feature_mask[None, :]";
  } else
    indices = indexTuple(operation, true);
  if (failed(indices))
    return failure();
  line("ct.scatter(" + (*view)->argument->name + ", " + *indices + ", " +
       stored->str() + ", check_bounds=True" +
       (mask.empty() ? std::string() : ", mask=" + mask) + ")");
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
    std::string call =
        "ct.atomic_add(" + (*view)->argument->name + ", " + *indices + ", " +
        stored->str() +
        ", check_bounds=True, memory_order=ct.MemoryOrder.RELAXED, "
        "memory_scope=ct.MemoryScope.DEVICE)";
    if (operation.getNumResults() == 1 && !operation.getResult(0).use_empty()) {
      std::string result = makeResultName(operation, 0);
      line(result + " = " + call);
      bindResult(operation, 0, result);
    } else {
      line(call);
    }
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
      !planIndex.stages.empty() || !compareIndex || !valueIndex || !ordering ||
      !scope || failed(view) || failed(expected) || failed(desired) ||
      operation.getNumResults() != 1 ||
      !operation.getResult(0).getType().isInteger(32) ||
      boundary.getResultSpace() != "private_scalar")
    return operation.emitOpError(
        "lacks a scalar cuTile compare-and-swap binding");

  std::string targetOrdering;
  if (ordering.getValue() == "relaxed")
    targetOrdering = "RELAXED";
  else if (ordering.getValue() == "acquire")
    targetOrdering = "ACQUIRE";
  else if (ordering.getValue() == "release")
    targetOrdering = "RELEASE";
  else if (ordering.getValue() == "acq_rel")
    targetOrdering = "ACQ_REL";
  else
    return operation.emitOpError("has no cuTile atomic ordering spelling");
  std::string targetScope;
  if (scope.getValue() == "workgroup")
    targetScope = "BLOCK";
  else if (scope.getValue() == "device")
    targetScope = "DEVICE";
  else if (scope.getValue() == "system")
    targetScope = "SYS";
  else
    return operation.emitOpError("has no cuTile atomic scope spelling");
  FailureOr<std::string> indices = indexTuple(operation, true);
  if (failed(indices))
    return failure();
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.atomic_cas(" + (*view)->argument->name + ", " +
       *indices + ", " + expected->str() + ", " + desired->str() +
       ", check_bounds=True, memory_order=ct.MemoryOrder." + targetOrdering +
       ", memory_scope=ct.MemoryScope." + targetScope + ")");
  bindResult(operation, 0, result);
  return success();
}

} // namespace intent::cutile::emission

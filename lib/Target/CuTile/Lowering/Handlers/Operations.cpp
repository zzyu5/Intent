#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/CuTile/Lowering/Passes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Analysis/StructuredControl.h"
#include "Intent/Target/Common/Lowering/Literal.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include <cmath>
#include <numeric>
#include <optional>

using namespace mlir;

namespace intent::cutile::lowering {
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
      failed(addHandler(registry, "intent.bitcast",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitBitcast(op);
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
      failed(addHandler(registry, "intent.scaled_contract",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitScaledContract(op);
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

LogicalResult ProgramMaterializer::emitConstant(Operation &operation) {
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
      expression = target::lowering::spellFiniteFloatLiteral(floating);
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
  if (operation.getBlock() == &kernel.entry.getBody().front() ||
      target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitExtract(Operation &operation) {
  FailureOr<Value> field = target::resolveRecordField(operation);
  if (failed(field))
    return failure();
  auto found = valueNames.find(*field);
  if (found == valueNames.end())
    return operation.emitOpError("record field has no emitted cuTile SSA value");
  bindResult(operation, 0, found->second);
  return success();
}

LogicalResult ProgramMaterializer::emitRegionEnd(Operation &operation) {
  if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
      !operation.getResult(0).getType().isIntOrIndex())
    return operation.emitOpError("lacks a mechanical cuTile region-end binding");
  FailureOr<plan::AxisOp> axis = resolveAxis(operation.getOperand(0), operation);
  if (failed(axis))
    return failure();
  std::string extent = axisDimensions.lookup(axis->getNode());
  auto owner = dimensionOwners.find(extent);
  if (owner != dimensionOwners.end())
    extent = owner->second;
  if (extent.empty())
    return operation.emitOpError("has no planned logical extent for region end");
  std::string expression = extent;
  if (target::lowering::isRaggedBoundAxis(planIndex.components,
                                           axis->getNode())) {
    expression = "sequence_end_" + std::to_string(axis->getNode());
  } else if (isa<intent::RegionType>(operation.getOperand(0).getType())) {
    if (axis->hasRole("parallel") && !axis->isScalar()) {
      std::string block = programBlocks.lookup(axis->getNode());
      if (block.empty())
        return operation.emitOpError(
            "has no planned program block for parallel region end");
      expression = "min((" + block + " + 1) * " + axis->getTile().str() +
                   ", " + extent + ")";
    } else if (axis->hasRole("ordered")) {
      const target::lowering::RangeBinding *range =
          axis->getRange("traversal", 0);
      std::string start = axisIndices.lookup(axis->getNode());
      if (!range || start.empty())
        return operation.emitOpError(
            "has no planned ordered range for region end");
      expression = "min((" + start + ") + " + range->getTile().str() + ", " +
                   extent + ")";
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

LogicalResult ProgramMaterializer::emitProgramBindings() {
  if (programBindingsEmitted)
    return success();
  programBindingsEmitted = true;

  auto roleExtent = [&](StringRef role) {
    std::string dimension = roleDimensions.lookup(role);
    std::string owner = dimensionOwners.lookup(dimension);
    return owner.empty() ? dimension : owner;
  };
  auto axisExtent = [&](plan::AxisOp axis) {
    std::string role =
        "program_" + std::to_string(axis.getProgramOrder());
    std::string extent = roleExtent(role);
    return axis.isScalar()
               ? extent
               : "ct.cdiv(" + extent + ", " + axis.getTile().str() + ")";
  };
  bool persistent = planIndex.program.getPersistent();
  std::string linear = "persistent_program";
  if (persistent) {
    line("total_program_tiles = " +
         target::lowering::projectProgramVolume(planIndex, axisExtent));
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
          target::lowering::projectLinearGroupIndex(
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
                      roleExtent(lhsRole) + ", " + lhs.getTile().str() + ")"));
    line(rhsCount + " = " +
         addressIndex("ct.cdiv(" +
                      roleExtent(rhsRole) + ", " + rhs.getTile().str() + ")"));
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

  SmallVector<target::lowering::ProgramIndexProjection> projections =
      persistent
          ? target::lowering::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::lowering::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("ct.bid(" + std::to_string(worker) + ")");
                });
  for (const target::lowering::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    line(block + " = " + projection.expression);
    programBlocks[axis.getNode()] = block;
    axisIndices[axis.getNode()] = block;
  }
  for (int64_t memberNode : planIndex.components.raggedProgramAxes) {
    plan::AxisOp axis = planIndex.axes.lookup(memberNode);
    std::string block = programBlocks.lookup(memberNode);
    FailureOr<plan::RaggedOp> relation =
        axis ? target::lowering::uniqueRaggedRelation(
                   planIndex, memberNode, *axis.operation.getOperation())
             : FailureOr<plan::RaggedOp>(failure());
    std::string outer = succeeded(relation)
                            ? axisIndices.lookup(relation->getOuterNode())
                            : std::string();
    auto runtime = succeeded(relation)
                       ? raggedRuntimeByRelation.find(relation->getNode())
                       : raggedRuntimeByRelation.end();
    if (!axis || axis.isScalar() || block.empty() || failed(relation) || outer.empty() ||
        runtime == raggedRuntimeByRelation.end())
      return programRoot->emitOpError(
          "ragged program axis has no tiled outer-axis or metadata binding");
    RaggedRuntime &ragged = raggedRuntimes[runtime->second];
    SmallVector<int64_t> boundAxes{memberNode};
    auto ordered =
        planIndex.components.orderedAxesByRelation.find(relation->getNode());
    if (ordered != planIndex.components.orderedAxesByRelation.end())
      for (int64_t orderedAxis : ordered->second)
        if (!llvm::is_contained(boundAxes, orderedAxis))
          boundAxes.push_back(orderedAxis);
    for (int64_t boundAxis : boundAxes) {
      std::string suffix = std::to_string(boundAxis);
      line("sequence_begin_" + suffix + " = ct.gather(" +
           ragged.offsets->argument->name + ", " + addressIndex(outer) +
           ", padding_value=0)");
      line("sequence_end_" + suffix + " = ct.gather(" +
           ragged.offsets->argument->name + ", " + addressIndex(outer + " + 1") +
           ", padding_value=0)");
      line("sequence_length_" + suffix + " = sequence_end_" + suffix +
           " - sequence_begin_" + suffix);
    }
    std::string suffix = std::to_string(memberNode);
    std::string values = "axis_index_" + suffix;
    line(values + " = sequence_begin_" + suffix + " + " + block + " * " +
         axis.getTile().str() + " + " +
         addressIndex("ct.arange(" + axis.getTile().str() +
                      ", dtype=ct.int32)"));
    line(values + " = ct.where(" + values + " < sequence_end_" + suffix +
         ", " + values + ", " + ragged.membersView->argument->name +
         ".shape[0])");
    axisIndices[memberNode] = values;
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
  for (const auto &entry : planIndex.axes) {
    plan::AxisOp axis = entry.second;
    if (!axisIndices.lookup(axis.getNode()).empty() ||
        !axis.getRange("reduction", 0))
      continue;
    axisIndices[axis.getNode()] = "0";
  }
  for (const auto &entry : planIndex.regionBindings) {
    Value value = kernel.values.lookup(entry.first);
    plan::RegionBindingOp binding = entry.second;
    if (!value || isa<BlockArgument>(value) || binding.getPurpose() != "ownership")
      continue;
    std::string projected = axisIndices.lookup(binding.getAxisNode());
    if (projected.empty())
      return binding.emitOpError(
          "has no active cuTile projection for its selected region value");
    selectedRegionIndices[value] = std::move(projected);
  }
  return success();
}

LogicalResult ProgramMaterializer::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() == 0)
    return operation.emitOpError("parallel ownership requires region arguments");
  Block &body = operation.getRegion(0).front();
  plan::PartitionBindingOp partition =
      target::lowering::countPartitionForIteration(planIndex, kernel, operation);
  if (partition) {
    if (!planIndex.components.reusedAxes.empty())
      return partition.emitOpError(
          "count partition cannot use worker-reused cuTile ownership");
    if (&operation == programRoot && failed(emitProgramBindings()))
      return failure();
    plan::AxisOp axis = planIndex.axes.lookup(partition.getAxisNode());
    std::string part = programBlocks.lookup(partition.getAxisNode());
    std::string region = axisIndices.lookup(partition.getAxisNode());
    if (!axis || part.empty() || region.empty())
      return partition.emitOpError(
          "has no emitted cuTile part and region projection");
    valueNames[body.getArgument(0)] = part;
    valueNames[body.getArgument(1)] = region;
    return success();
  }
  SmallVector<plan::AxisOp> axes;
  for (BlockArgument argument : body.getArguments()) {
    FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
    if (failed(axis))
      return failure();
    axes.push_back(*axis);
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
    if (target::lowering::isPackedScalarAxis(axis)) {
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

LogicalResult ProgramMaterializer::leaveParallel(Operation &operation) {
  if (&operation == programRoot &&
      (!planIndex.components.reusedAxes.empty() ||
       planIndex.program.getPersistent()))
    --indentation;
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
        operation.emitOpError("has no emitted cuTile loop bound");
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
          (!raggedAxis && *start != "0") || *step != "1")
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
    std::string logicalIterator = iterator;
    if (raggedIndices) {
      logicalIterator = iterator + "_member";
      line(logicalIterator + " = ct.gather(" + raggedIndices->argument->name +
           ", " + addressIndex(iterator) + ", padding_value=0)");
    }
    valueNames[body.getArgument(index)] = logicalIterator;
    if (raggedAxis && succeeded(domainNode))
      axisIndices[*domainNode] = logicalIterator;
  }
  return success();
}

LogicalResult ProgramMaterializer::leaveFor(Operation &operation) {
  auto carriers = loopCarriers.find(&operation);
  if (carriers == loopCarriers.end())
    return operation.emitOpError("has no active cuTile sequential loop");
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
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index]);
  return success();
}

LogicalResult ProgramMaterializer::enterIf(Operation &operation) {
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

LogicalResult ProgramMaterializer::leaveIf(Operation &operation) {
  auto results = ifResults.find(&operation);
  if (results == ifResults.end())
    return operation.emitOpError("has no active cuTile scalar branch");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, results->second[index]);
  return success();
}

LogicalResult ProgramMaterializer::enterWhile(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitCondition(Operation &operation) {
  Operation *owner = target::whileConditionOwner(operation);
  FailureOr<StringRef> condition = lookupValue(operation, 0);
  if (!owner || failed(condition) || operation.getNumOperands() !=
                                          owner->getNumResults() + 1)
    return operation.emitOpError("does not match a cuTile scalar while");
  line("while " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult ProgramMaterializer::leaveWhile(Operation &operation) {
  auto carriers = whileCarriers.find(&operation);
  if (carriers == whileCarriers.end())
    return operation.emitOpError("has no active cuTile scalar while");
  --indentation;
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    bindResult(operation, index, carriers->second[index]);
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

LogicalResult ProgramMaterializer::emitBufferLoad(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitBufferStore(Operation &operation) {
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
      target::lowering::projectPrivateWorkspaceOffset(
          binding, *info, *indices, planIndex, axisIndices, axisDimensions,
          spellIndex, operation);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

FailureOr<std::string>
ProgramMaterializer::scanWorkspaceIndex(const plan::ScanOp &binding,
                                  StringRef logicalIndex,
                                  Operation &consumer) {
  std::string extent = scanExtents.lookup(binding.getNode());
  if (extent.empty())
    return binding.emitOpError("has no cuTile scan workspace binding");
  FailureOr<std::string> offset = target::lowering::projectScanWorkspaceOffset(
      binding, extent, logicalIndex, planIndex, axisIndices, axisDimensions,
      consumer);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

FailureOr<std::string>
ProgramMaterializer::scanMaterializedIndex(Value value, StringRef logicalIndex,
                                     Operation &consumer) {
  auto materialized = scanMaterializedValues.find(value);
  if (materialized == scanMaterializedValues.end())
    return consumer.emitOpError("has no cuTile materialized scan value");
  FailureOr<std::string> offset = target::lowering::projectScanWorkspaceOffset(
      materialized->second, scanExtents.lookup(materialized->second.getNode()),
      logicalIndex, planIndex, axisIndices, axisDimensions, consumer);
  if (failed(offset))
    return failure();
  return addressIndex(*offset);
}

LogicalResult ProgramMaterializer::emitLoad(Operation &operation) {
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
  if (boundary.getDefer() && !activeDeferredContract) {
    deferredLoads[operation.getResult(0)] = &operation;
    return success();
  }
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  bool packedScalar =
      target::lowering::hasPackedScalarDomain(planIndex, boundary);
  bool scanReplay = activeScanReplay >= 0;
  bool partitionStreamTile =
      boundary.getTransfer() == "partition_stream_tile";
  auto loadShapeNode =
      boundary.operation->getAttrOfType<IntegerAttr>(loadShapeNodeAttr);
  Operation *loadShape =
      loadShapeNode ? kernel.nodes.lookup(loadShapeNode.getInt()) : nullptr;
  auto loadShapeType =
      loadShape && loadShape->getNumResults() == 1
          ? dyn_cast<RankedTensorType>(loadShape->getResult(0).getType())
          : RankedTensorType();
  bool scalarLoadShape = loadShapeType && loadShapeType.getRank() == 0;
  FailureOr<std::string> indices = indexTuple(
      operation,
      (boundary.getAccess() == "gather" && !scalarLoadShape) || packedScalar ||
          scanReplay,
      partitionStreamTile);
  if (failed(view) || failed(indices) || failed(physicalFill))
    return failure();
  StringRef loadFill = boundary.getPadding();
  if (loadFill == "none" && !physicalFill->empty())
    loadFill = *physicalFill;
  bool materializeValidity = !boundary.getConsumerNeutralized() &&
                             boundary.getAccess() == "gather" &&
                             (boundary.getPadding() != "none" ||
                              boundary.hasDataDependentTensorIndex()) &&
                             !boundary.getValidityDomainNodes().empty();
  FailureOr<std::string> validity =
      materializeValidity
          ? emitValidityExpression(boundary.getValidityTensorAxes(),
                                   boundary.getValidityDomainNodes(),
                                   operation.getResult(0), operation)
          : FailureOr<std::string>(std::string("True"));
  if (failed(validity))
    return failure();
  Type elementType = (*view)->tensor.getElementType();
  StringRef padding = loadFill == "negative_infinity"
                          ? "-math.inf"
                      : isa<IntegerType, IndexType>(elementType) ? "0"
                                                                : "0.0";
  std::string result = makeResultName(operation, 0);
  if (boundary.getAccess() == "gather" || packedScalar || scanReplay) {
    std::string expression =
        result + " = ct.gather(" + (*view)->argument->name + ", " + *indices +
        ", check_bounds=" +
        (boundary.getCheckBounds() ? std::string("True")
                                   : std::string("False")) +
        ", padding_value=" + padding.str() + ")";
    line(expression);
  } else if (boundary.getAccess() == "load") {
    FailureOr<std::string> shape = tileShape(operation);
    bool scalarResult = operation.getNumResults() == 1 &&
                        !isa<RankedTensorType>(operation.getResult(0).getType());
    FailureOr<std::string> resultShape =
        scalarResult ? FailureOr<std::string>(std::string())
                     : emitTensorShape(operation, 0);
    if (loadShape)
      resultShape = emitTensorShape(*loadShape, 0);
    if (failed(shape) || failed(resultShape))
      return failure();
    StringRef paddingMode = loadFill == "negative_infinity"
                                ? "ct.PaddingMode.NEG_INF"
                                : "ct.PaddingMode.ZERO";
    std::string expression =
        "ct.load(" + (*view)->argument->name + ", index=" + *indices +
        ", shape=" + *shape;
    if (partitionStreamTile) {
      std::string order = "(";
      for (int64_t axis = 0; axis < (*view)->tensor.getRank(); ++axis) {
        if (axis)
          order += ", ";
        order += std::to_string(axis);
      }
      if ((*view)->tensor.getRank() == 1)
        order += ",";
      expression += ", order=" + order + "), allow_tma=True";
    }
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
  FailureOr<int64_t> resultID = target::getValueID(
      operation.getResult(0), kernel, operation, "cuTile load padding fusion");
  plan::PaddingOp plannedPadding =
      succeeded(resultID) ? planIndex.paddings.lookup(*resultID)
                          : plan::PaddingOp();
  bool paddingCoveredByValidity =
      *validity != "True" && plannedPadding &&
      plannedPadding.getFill() == loadFill &&
      llvm::all_of(
          llvm::zip(plannedPadding.getTensorAxes(),
                    plannedPadding.getDomainNodes()),
          [&](auto entry) {
            auto [paddingAxis, paddingNode] = entry;
            return llvm::any_of(
                llvm::zip(boundary.getValidityTensorAxes(),
                          boundary.getValidityDomainNodes()),
                [&](auto boundaryEntry) {
                  auto [boundaryAxis, boundaryNode] = boundaryEntry;
                  return paddingAxis == boundaryAxis &&
                         paddingNode == boundaryNode;
                });
          });
  if (*validity != "True")
    line(result + " = ct.where(" + *validity + ", " + result + ", " +
         padding.str() + ")");
  FailureOr<std::string> padded =
      paddingCoveredByValidity
          ? FailureOr<std::string>(result)
          : padExpression(operation.getResult(0), result, operation);
  if (failed(padded))
    return failure();
  if (*padded != result)
    line(result + " = " + *padded);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitIndices(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "indices emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  auto result = operation.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                    : RankedTensorType();
  if (failed(node) || !binding || binding.getLowering() != "logical_indices" ||
      !result || result.getRank() <= 0 ||
      !isa<IntegerType, IndexType>(result.getElementType()))
    return operation.emitOpError("lacks a mechanical cuTile indices binding");
  auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
  auto tensorAxis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  FailureOr<plan::AxisOp> axis = failure();
  std::optional<target::lowering::RegionRangeBinding> region;
  unsigned emittedAxis = 0;
  if (mode && mode.getValue() == "tensor_axis") {
    if (!tensorAxis || tensorAxis.getInt() < 0 ||
        tensorAxis.getInt() >= result.getRank() ||
        static_cast<size_t>(tensorAxis.getInt()) >= binding.getAxisNodes().size())
      return operation.emitOpError("has no tensor-axis indices binding");
    axis = planIndex.axes.lookup(binding.getAxisNodes()[tensorAxis.getInt()]);
    emittedAxis = tensorAxis.getInt();
  } else {
    if (result.getRank() != 1)
      return operation.emitOpError("domain indices require one result axis");
    Value indexed = operation.getOperand(0);
    FailureOr<std::optional<target::lowering::RegionRangeBinding>> selected =
        target::lowering::selectedRegionValueRange(planIndex, kernel, indexed,
                                                   operation);
    if (failed(selected))
      return failure();
    if (*selected) {
      region = **selected;
      axis = region->axis;
    } else {
      axis = resolveAxis(indexed, operation);
    }
  }
  if (failed(axis))
    return failure();
  if (activeScanReplay >= 0) {
    std::string replay = axisIndices.lookup(axis->getNode());
    if (replay.empty())
      return operation.emitOpError("has no active cuTile scan replay index");
    bindResult(operation, 0, replay);
    return success();
  }
  std::string expression;
  std::string selectedTile;
  if (region) {
    selectedTile = region->range.getTile().str();
    if (!region->axis.getReuseWorker() &&
        region->range.getTileRole().starts_with("row_vector")) {
      if (!planIndex.blockExtents.count(region->range.getExtent()))
        return operation.emitOpError(
            "row-vector region indices lack their selected physical extent");
      selectedTile = physicalExtent(region->range.getExtent());
    }
  }
  if (!region && axis->hasRole("lane") && !axis->getReuseWorker() &&
      axis->getTileRole().starts_with("row_vector")) {
    FailureOr<std::string> tile = physicalAxisTile(*axis);
    if (failed(tile))
      return failure();
    expression = addressIndex("ct.arange(" + *tile + ", dtype=ct.int32)");
  } else {
    Value indexed = operation.getOperand(0);
    FailureOr<StringRef> argumentProjection =
        region && isa<BlockArgument>(indexed)
            ? lookupValue(operation, 0)
            : FailureOr<StringRef>(failure());
    std::string base =
        region ? isa<BlockArgument>(indexed)
                     ? succeeded(argumentProjection) ? argumentProjection->str()
                                                     : std::string()
                     : selectedRegionIndices.lookup(indexed)
               : axisIndices.lookup(axis->getNode());
    if (base.empty() && region && isa<intent::DomainType>(indexed.getType()))
      base = axisIndices.lookup(axis->getNode());
    if (region && isa<BlockArgument>(indexed) && failed(argumentProjection))
      return failure();
    if (base.empty() && axis->hasRole("lane")) {
      base = "0";
      axisIndices[axis->getNode()] = base;
    }
    if (base.empty())
      return operation.emitOpError("has no cuTile vector-index projection");
    bool directVector =
        target::lowering::isRaggedBoundAxis(planIndex.components,
                                            axis->getNode()) ||
        (region && absoluteRegionArguments.contains(operation.getOperand(0))) ||
        (axis->hasRole("lane") &&
         kernel.nodes.lookup(axis->getNode()) == vectorDomain);
    FailureOr<std::string> axisTile =
        region ? FailureOr<std::string>(selectedTile) : physicalAxisTile(*axis);
    if (failed(axisTile))
      return failure();
    expression = directVector
                     ? addressIndex(base)
                     : addressIndex(base) + " * " + *axisTile + " + " +
                           addressIndex("ct.arange(" + *axisTile +
                                        ", dtype=ct.int32)");
  }
  if (result.getRank() > 1) {
    FailureOr<std::string> tile =
        region ? FailureOr<std::string>(selectedTile) : physicalAxisTile(*axis);
    if (failed(tile))
      return failure();
    SmallVector<std::string> shape(result.getRank(), "1");
    shape[emittedAxis] = *tile;
    expression = "ct.reshape(" + expression + ", (" +
                 llvm::join(shape, ", ") + "))";
  }
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), expression, operation);
  if (failed(padded))
    return failure();
  std::string emitted = makeResultName(operation, 0);
  line(emitted + " = " + *padded);
  bindResult(operation, 0, emitted);
  return success();
}

LogicalResult ProgramMaterializer::emitRandom(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  if (failed(node) || !binding || !components || components.getInt() <= 0 ||
      operation.getNumResults() != static_cast<unsigned>(components.getInt()))
    return failure();
  SmallVector<std::string> operands;
  SmallVector<std::string> identities;
  for (unsigned component = 0;
       component < static_cast<unsigned>(components.getInt()); ++component) {
    FailureOr<StringRef> operand = lookupValue(operation, component);
    FailureOr<StringRef> identity =
        lookupValue(operation, components.getInt() + component);
    if (failed(operand) || failed(identity))
      return failure();
    operands.push_back(operand->str());
    identities.push_back(identity->str());
  }
  if (binding.getLowering() == "ct.max_with_index") {
    if (operation.getNumResults() != 2 || operands.size() != 2 ||
        identities.size() != 2)
      return operation.emitOpError(
          "cuTile arg-reduction requires value and index results");
    std::string value = makeResultName(operation, 0);
    std::string index = makeResultName(operation, 1);
    std::string valueKeepDims = value + "_keep_dims";
    std::string candidates = index + "_candidates";
    std::string axis = std::to_string(binding.getAxis());
    std::string keepDims = binding.getKeepDims() ? "True" : "False";
    line(value + " = ct.max(" + operands.front() + ", " + axis +
         ", keepdims=" + keepDims + ")");
    line(valueKeepDims + " = ct.max(" + operands.front() + ", " + axis +
         ", keepdims=True)");
    line(candidates + " = ct.where(" + operands.front() + " == " +
         valueKeepDims + ", " + operands[1] + ", " + identities[1] + ")");
    line(index + " = ct.min(" + candidates + ", " + axis +
         ", keepdims=" + keepDims + ")");
    bindResult(operation, 0, value);
    bindResult(operation, 1, index);
    return success();
  }
  if (binding.getLowering() == "ct.reduce") {
    FailureOr<target::lowering::CombinerUse> combiner =
        target::lowering::resolveCombiner(operation);
    if (failed(combiner))
      return failure();
    SmallVector<std::string> results;
    for (unsigned component = 0; component < operation.getNumResults(); ++component)
      results.push_back(makeResultName(operation, component));
    SmallVector<std::string> projectedInputs = operands;
    SmallVector<std::string> projectedIdentities = identities;
    SmallVector<std::string> projectedResults = results;
    std::string function = combiner->function.getName().str();
    if (combiner->captureCount != 0) {
      FailureOr<std::string> projection =
          target::lowering::combinerProjectionName(operation);
      if (failed(projection))
        return failure();
      function = *projection;
      for (unsigned capture = 0; capture < combiner->captureCount; ++capture) {
        unsigned operandIndex = 2 * combiner->componentCount + capture;
        FailureOr<StringRef> value = lookupValue(operation, operandIndex);
        std::string dtype = dtypeName(operation.getOperand(operandIndex).getType(),
                                      operation);
        if (failed(value) || dtype.empty())
          return operation.emitOpError(
              "has no cuTile scalar capture projection");
        projectedInputs.push_back("ct.full(" + operands.front() + ".shape, " +
                                  value->str() + ", dtype=" + dtype + ")");
        projectedIdentities.push_back(
            operation.getOperand(operandIndex).getType().isInteger(1) ? "False"
                                                                      : "0");
        projectedResults.push_back(makeResultName(operation, 0) + "_capture_" +
                                   std::to_string(capture));
      }
      projectedInputs.push_back("ct.full(" + operands.front() +
                                ".shape, True, dtype=ct.bool_)");
      projectedIdentities.push_back("False");
      projectedResults.push_back(makeResultName(operation, 0) + "_capture_valid");
    }
    std::string input = projectedInputs.size() == 1
                            ? projectedInputs.front()
                            : "(" + llvm::join(projectedInputs, ", ") + ")";
    std::string identity = projectedIdentities.size() == 1
                               ? projectedIdentities.front()
                               : "(" + llvm::join(projectedIdentities, ", ") + ")";
    line(llvm::join(projectedResults, ", ") + " = ct.reduce(" + input + ", axis=" +
         std::to_string(binding.getAxis()) + ", func=" +
         function + ", identity=" + identity +
         ", keepdims=" + (binding.getKeepDims() ? "True" : "False") + ")");
    for (auto [index, result] : llvm::enumerate(results))
      bindResult(operation, index, result);
    return success();
  }
  if (operation.getNumResults() != operands.size())
    return operation.emitOpError(
        "built-in cuTile reduction components do not match results");
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    std::string result = makeResultName(operation, component);
  if (binding.getLowering() == "ct.any_via_max" ||
      binding.getLowering() == "ct.all_via_min") {
    StringRef reduction = binding.getLowering() == "ct.any_via_max" ? "ct.max"
                                                                      : "ct.min";
      line(result + " = ct.astype(" + reduction.str() + "(" + operands[component] +
         ", " + std::to_string(binding.getAxis()) + ", keepdims=" +
         (binding.getKeepDims() ? "True" : "False") + "), ct.bool_)");
  } else {
      line(result + " = " + binding.getLowering().str() + "(" + operands[component] +
         ", " + std::to_string(binding.getAxis()) + ", keepdims=" +
         (binding.getKeepDims() ? "True" : "False") + ")");
  }
    bindResult(operation, component, result);
  }
  return success();
}

LogicalResult ProgramMaterializer::emitScan(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "scan emission");
  plan::ScanOp binding =
      succeeded(node) ? planIndex.scans.lookup(*node) : plan::ScanOp();
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  if (!components || components.getInt() <= 0 ||
      operation.getNumResults() != static_cast<unsigned>(components.getInt()))
    return operation.emitOpError("has no canonical scan component schema");
  if (binding && binding.getResultSpace() == "private_fragment") {
    if (failed(node) ||
        (binding.getLowering() != "ct.cumsum" &&
         binding.getLowering() != "ct.scan"))
      return operation.emitOpError("lacks a fragment cuTile scan binding");
    SmallVector<std::string> operands;
    SmallVector<std::string> identities;
    for (unsigned component = 0;
         component < static_cast<unsigned>(components.getInt()); ++component) {
      FailureOr<StringRef> operand = lookupValue(operation, component);
      FailureOr<StringRef> identity =
          lookupValue(operation, components.getInt() + component);
      if (failed(operand) || failed(identity))
        return failure();
      operands.push_back(operand->str());
      identities.push_back(identity->str());
    }
    if (binding.getLowering() == "ct.scan") {
      FailureOr<target::lowering::CombinerUse> combiner =
          target::lowering::resolveCombiner(operation);
      if (failed(combiner))
        return failure();
      SmallVector<std::string> results;
      for (unsigned component = 0; component < operation.getNumResults(); ++component)
        results.push_back(makeResultName(operation, component));
      SmallVector<std::string> projectedInputs = operands;
      SmallVector<std::string> projectedIdentities = identities;
      SmallVector<std::string> projectedResults = results;
      std::string function = combiner->function.getName().str();
      if (combiner->captureCount != 0) {
        FailureOr<std::string> projection =
            target::lowering::combinerProjectionName(operation);
        if (failed(projection))
          return failure();
        function = *projection;
        for (unsigned capture = 0; capture < combiner->captureCount; ++capture) {
          unsigned operandIndex = 2 * combiner->componentCount + capture;
          FailureOr<StringRef> value = lookupValue(operation, operandIndex);
          std::string dtype = dtypeName(operation.getOperand(operandIndex).getType(),
                                        operation);
          if (failed(value) || dtype.empty())
            return operation.emitOpError(
                "has no cuTile scalar capture projection");
          projectedInputs.push_back("ct.full(" + operands.front() + ".shape, " +
                                    value->str() + ", dtype=" + dtype + ")");
          projectedIdentities.push_back(
              operation.getOperand(operandIndex).getType().isInteger(1) ? "False"
                                                                        : "0");
          projectedResults.push_back(makeResultName(operation, 0) + "_capture_" +
                                     std::to_string(capture));
        }
        projectedInputs.push_back("ct.full(" + operands.front() +
                                  ".shape, True, dtype=ct.bool_)");
        projectedIdentities.push_back("False");
        projectedResults.push_back(makeResultName(operation, 0) + "_capture_valid");
      }
      std::string input = projectedInputs.size() == 1
                              ? projectedInputs.front()
                              : "(" + llvm::join(projectedInputs, ", ") + ")";
      std::string identity = projectedIdentities.size() == 1
                                 ? projectedIdentities.front()
                                 : "(" + llvm::join(projectedIdentities, ", ") + ")";
      line(llvm::join(projectedResults, ", ") + " = ct.scan(" + input + ", axis=" +
           std::to_string(binding.getAxis()) + ", func=" +
           function + ", identity=" + identity +
           ")");
      for (auto [index, result] : llvm::enumerate(results))
        bindResult(operation, index, result);
      return success();
    }
    if (operation.getNumResults() != operands.size())
      return operation.emitOpError(
          "built-in cuTile scan components do not match results");
    for (unsigned component = 0; component < operation.getNumResults(); ++component) {
      std::string result = makeResultName(operation, component);
      line(result + " = ct.cumsum(" + operands[component] + ", axis=" +
           std::to_string(binding.getAxis()) + ")");
      bindResult(operation, component, result);
    }
    return success();
  }
  plan::AxisOp axis = binding ? planIndex.axes.lookup(binding.getAxisNode())
                              : plan::AxisOp();
  const target::lowering::RangeBinding *range =
      axis ? axis.getRange("traversal", 0) : nullptr;
  std::string extent = binding ? scanExtents.lookup(binding.getNode())
                               : std::string();
  bool generic = binding && binding.getLowering() == "ct.scan";
  if (failed(node) || !binding ||
      (binding.getLowering() != "ct.cumsum" && !generic) || !axis || !range ||
      extent.empty() || operation.getNumResults() == 0)
    return operation.emitOpError("lacks a mechanical cuTile scan binding");
  FailureOr<target::lowering::CombinerUse> combiner = failure();
  std::string function;
  if (generic) {
    combiner = target::lowering::resolveCombiner(operation);
    if (failed(combiner))
      return failure();
    function = combiner->function.getName().str();
    if (combiner->captureCount != 0) {
      FailureOr<std::string> projection =
          target::lowering::combinerProjectionName(operation);
      if (failed(projection))
        return failure();
      function = *projection;
    }
  }
  SmallVector<std::string> identities;
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    FailureOr<StringRef> identity =
        lookupValue(operation, components.getInt() + component);
    if (failed(identity))
      return failure();
    identities.push_back(identity->str());
  }
  SmallVector<std::string> carries;
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    carries.push_back(makeResultName(operation, component) + "_carry");
    line(carries.back() + " = " + identities[component]);
  }
  if (generic && combiner->captureCount != 0) {
    for (unsigned capture = 0; capture < combiner->captureCount; ++capture) {
      carries.push_back(makeResultName(operation, 0) + "_capture_" +
                        std::to_string(capture) + "_carry");
      line(carries.back() + " = 0");
    }
    carries.push_back(makeResultName(operation, 0) + "_capture_valid_carry");
    line(carries.back() + " = False");
  }
  std::string stem = makeResultName(operation, 0);
  std::string block = stem + "_block";
  std::string offsets = stem + "_offsets";
  line("for " + block + " in range(ct.cdiv(" + extent + ", " +
       range->getTile().str() + ")):");
  ++indentation;
  line(offsets + " = " + addressIndex(block) + " * " +
       range->getTile().str() + " + " +
       addressIndex("ct.arange(" + range->getTile().str() +
                    ", dtype=ct.int32)"));
  if (failed(replayScanProducers(binding, offsets)))
    return failure();
  SmallVector<std::string> blockInputs;
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    FailureOr<StringRef> operand = lookupValue(operation, component);
    if (failed(operand))
      return failure();
    blockInputs.push_back("ct.where(" + offsets + " < " + extent + ", " +
                          operand->str() + ", " + identities[component] + ")");
  }
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
  SmallVector<std::string> localInputs = blockInputs;
  SmallVector<std::string> localIdentities = identities;
  SmallVector<std::string> localResults;
  SmallVector<std::string> globalResults;
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    localResults.push_back(makeResultName(operation, component) + "_local_scan");
    globalResults.push_back(makeResultName(operation, component) + "_global_scan");
  }
  if (generic && combiner->captureCount != 0) {
    for (unsigned capture = 0; capture < combiner->captureCount; ++capture) {
      unsigned operandIndex = 2 * combiner->componentCount + capture;
      FailureOr<StringRef> captureValue = lookupValue(operation, operandIndex);
      std::string dtype =
          dtypeName(operation.getOperand(operandIndex).getType(), operation);
      if (failed(captureValue) || dtype.empty())
        return operation.emitOpError(
            "has no cuTile workspace-scan capture projection");
      localInputs.push_back("ct.full(" + blockInputs.front() + ".shape, " +
                            captureValue->str() + ", dtype=" + dtype + ")");
      localIdentities.push_back(
          operation.getOperand(operandIndex).getType().isInteger(1) ? "False"
                                                                    : "0");
      localResults.push_back(stem + "_capture_" + std::to_string(capture) +
                             "_local_scan");
      globalResults.push_back(stem + "_capture_" + std::to_string(capture) +
                              "_global_scan");
    }
    localInputs.push_back("ct.full(" + blockInputs.front() +
                          ".shape, True, dtype=ct.bool_)");
    localIdentities.push_back("False");
    localResults.push_back(stem + "_capture_valid_local_scan");
    globalResults.push_back(stem + "_capture_valid_global_scan");
  }
  if (generic) {
    std::string input = localInputs.size() == 1
                            ? localInputs.front()
                            : "(" + llvm::join(localInputs, ", ") + ")";
    std::string identity = localIdentities.size() == 1
                               ? localIdentities.front()
                               : "(" + llvm::join(localIdentities, ", ") + ")";
    line(llvm::join(localResults, ", ") + " = ct.scan(" + input + ", axis=" +
         std::to_string(binding.getAxis()) + ", func=" + function +
         ", identity=" + identity + ")");
    SmallVector<std::string> arguments = carries;
    arguments.append(localResults);
    line(llvm::join(globalResults, ", ") + " = " + function + "(" +
         llvm::join(arguments, ", ") + ")");
  } else {
    for (unsigned component = 0; component < operation.getNumResults(); ++component) {
      line(localResults[component] + " = ct.cumsum(" + blockInputs[component] +
           ", axis=" + std::to_string(binding.getAxis()) + ")");
      line(globalResults[component] + " = " + localResults[component] + " + " +
           carries[component]);
    }
  }
  FailureOr<std::string> workspaceIndex =
      scanWorkspaceIndex(binding, offsets, operation);
  if (failed(workspaceIndex))
    return failure();
  for (unsigned component = 0; component < operation.getNumResults(); ++component)
    line("ct.scatter(" + workspaceNames.lookup(operation.getResult(component)) +
         ", " + *workspaceIndex + ", " + globalResults[component] +
         ", check_bounds=True)");
  std::string lastLane = "ct.minimum(" + range->getTile().str() + " - 1, " +
                         extent + " - " + block + " * " +
                         range->getTile().str() + " - 1)";
  for (unsigned component = 0; component < carries.size(); ++component)
    line(carries[component] + " = ct.extract(" + globalResults[component] +
         ", (" + lastLane + ",), shape=(1,)).item()");
  --indentation;
  for (Value result : operation.getResults())
    valueNames.erase(result);
  return success();
}

LogicalResult ProgramMaterializer::replayScanProducers(const plan::ScanOp &binding,
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

LogicalResult ProgramMaterializer::replayContractProducers(
    ArrayRef<Operation *> producers) {
  if (!activeDeferredContract || !operationRegistry())
    return failure();
  for (Operation *producer : producers)
    if (!producer || failed(operationRegistry()->dispatch(
                         *producer,
                         "cuTile deferred contraction producer replay")))
      return failure();
  return success();
}

LogicalResult ProgramMaterializer::replayBlock(Block &block) {
  if (!operationRegistry())
    return block.getParentOp()->emitOpError(
        "has no cuTile operation registry for provider-form replay");
  for (Operation &operation : block) {
    const target::OperationHandler *handler = operationRegistry()->lookup(
        target::semanticOperationName(operation));
    if (!handler)
      return operation.emitOpError(
          "has no cuTile handler during provider-form replay");
    if (handler->enter && failed(handler->enter(operation)))
      return failure();
    for (Region &region : operation.getRegions())
      for (Block &nested : region)
        if (failed(replayBlock(nested)))
          return failure();
    if (handler->leave && failed(handler->leave(operation)))
      return failure();
  }
  return success();
}

LogicalResult ProgramMaterializer::emitBroadcast(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitUnary(Operation &operation) {
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

LogicalResult ProgramMaterializer::emitBinary(Operation &operation) {
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
  else if (binding.getLowering() == "ct.pow") {
    auto elementType = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        return tensor.getElementType();
      return type;
    };
    Type lhsType = elementType(operation.getOperand(0).getType());
    Type rhsType = elementType(operation.getOperand(1).getType());
    Type resultType = elementType(operation.getResult(0).getType());
    bool needsWidening =
        isa<Float8E4M3FNType, Float8E5M2Type>(lhsType) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(rhsType) ||
        isa<Float8E4M3FNType, Float8E5M2Type>(resultType);
    if (!needsWidening) {
      expression = "ct.pow(" + lhs->str() + ", " + rhs->str() + ")";
    } else {
      std::string resultDtype = dtypeName(resultType, operation);
      if (resultDtype.empty())
        return failure();
      std::string widened = "ct.pow(ct.astype(" + lhs->str() +
                            ", ct.float32), ct.astype(" + rhs->str() +
                            ", ct.float32))";
      expression = resultType.isF32()
                       ? std::move(widened)
                       : "ct.astype(" + widened + ", " + resultDtype + ")";
    }
  } else if (binding.getLowering().starts_with("ct.bitwise_"))
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

LogicalResult ProgramMaterializer::emitConditional(Operation &operation, bool mask) {
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

LogicalResult ProgramMaterializer::emitMask(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "mask emission");
  if (succeeded(node) && activeNeutralMasks.contains(*node)) {
    FailureOr<StringRef> input = lookupValue(operation, 0);
    if (failed(input))
      return failure();
    bindResult(operation, 0, *input);
    return success();
  }
  return emitConditional(operation, true);
}

LogicalResult ProgramMaterializer::emitSelect(Operation &operation) {
  return emitConditional(operation, false);
}

LogicalResult ProgramMaterializer::emitCast(Operation &operation) {
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
      syntax::cast(binding.getLowering(), *operand, targetType);
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitBitcast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "bitcast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  Type resultType = operation.getNumResults() == 1
                        ? operation.getResult(0).getType()
                        : Type();
  if (failed(node) || !binding || binding.getLowering() != "ct.bitcast" ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical cuTile bitcast binding");
  Type resultElement = resultType;
  if (auto tensor = dyn_cast<RankedTensorType>(resultElement))
    resultElement = tensor.getElementType();
  std::string targetType = dtypeName(resultElement, operation);
  if (targetType.empty())
    return failure();
  std::string value = operand->str();
  if (binding.getResultSpace() == "private_scalar") {
    Type sourceElement = operation.getOperand(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(sourceElement))
      sourceElement = tensor.getElementType();
    std::string sourceType = dtypeName(sourceElement, operation);
    if (sourceType.empty())
      return failure();
    value = "ct.full((), " + value + ", dtype=" + sourceType + ")";
  }
  std::string result = makeResultName(operation, 0);
  std::string expression = syntax::bitcast(value, targetType);
  if (target::whileConditionOwner(operation)) {
    bindResult(operation, 0, expression);
    return success();
  }
  line(result + " = " + expression);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitReshape(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reshape emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<std::string> shape = emitTensorShape(operation, 0);
  if (failed(node) || !binding || binding.getLowering() != "ct.reshape" ||
      failed(operand) || failed(shape) || operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a mechanical cuTile reshape binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.reshape(" + operand->str() + ", " + *shape + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitTranspose(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "transpose emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  FailureOr<SmallVector<int64_t>> permutation =
      target::lowering::transposePermutation(operation);
  if (failed(node) || !binding || binding.getLowering() != "ct.permute" ||
      failed(operand) || failed(permutation))
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

LogicalResult ProgramMaterializer::emitFull(Operation &operation) {
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
  std::string expression = "ct.full(" + *shape + ", " + fill->str() +
                           ", dtype=" + dtype + ")";
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), expression, operation);
  if (failed(padded))
    return failure();
  line(result + " = " + *padded);
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitZeros(Operation &operation) {
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

void ProgramMaterializer::emitGuardedGather(StringRef result, StringRef array,
                                      StringRef indices, StringRef padding,
                                      StringRef valid) {
  auto gather = [&](StringRef mask) {
    return result.str() + " = " +
           syntax::gather(array, indices, padding, mask);
  };
  if (tuneGatherSpelling) {
    line("if GATHER_SPELLING:");
    ++indentation;
    line(gather(valid));
    --indentation;
    line("else:");
    ++indentation;
    line(gather(""));
    line(result.str() + " = ct.where(" + valid.str() + ", " + result.str() +
         ", " + padding.str() + ")");
    --indentation;
    return;
  }
  line(gather(""));
  line(result.str() + " = ct.where(" + valid.str() + ", " + result.str() +
       ", " + padding.str() + ")");
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
  bool alwaysValid = false;
  if (validIndex && validIndex.getInt() >= 0 &&
      static_cast<unsigned>(validIndex.getInt()) < operation.getNumOperands()) {
    Operation *definition =
        operation.getOperand(validIndex.getInt()).getDefiningOp();
    auto literal =
        definition && target::semanticOperationName(*definition) ==
                          "intent.constant"
            ? definition->getAttrOfType<IntegerAttr>("intent.value")
            : IntegerAttr();
    alwaysValid = literal && !literal.getValue().isZero();
  }
  bool scalarFragmentGather =
      form == "scalar_fragment" &&
      isa<RankedTensorType>(operation.getOperand(0).getType()) &&
      !isa<RankedTensorType>(operation.getResult(0).getType()) &&
      succeeded(relation) && relation->size() == 1 &&
      (*relation)[0].kind == "value_index" &&
      (*relation)[0].operands.size() == 1 &&
      (*relation)[0].operands.front();
  if (form == "ragged_start_scalar") {
    Operation *indices = operation.getOperand(0).getDefiningOp();
    FailureOr<plan::AxisOp> axis =
        indices && indices->getNumOperands() == 1
            ? resolveAxis(indices->getOperand(0), operation)
            : FailureOr<plan::AxisOp>(failure());
    if (failed(axis) ||
        !target::lowering::isRaggedBoundAxis(planIndex.components,
                                             axis->getNode()))
      return operation.emitOpError(
          "ragged-start cuTile gather has no planned ragged axis");
    bindResult(operation, 0,
               "sequence_begin_" + std::to_string(axis->getNode()));
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
      if (failed(workspaceIndex))
        return failure();
      line(result + " = ct.where(" + valid->str() + ", ct.load(" +
           workspaceNames.lookup(operation.getOperand(0)) + ", index=(" +
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
  if (form == "extract_first_scalar" &&
      succeeded(relation) && relation->size() == 1 &&
      (*relation)[0].kind == "static_index") {
    FailureOr<StringRef> source = lookupValue(operation, 0);
    if (failed(source))
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.extract(" + source->str() +
         ", (0,), shape=(1,)).item()");
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
  if (form == "view_indirect" &&
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
    if (!alwaysValid)
      line(result + " = ct.where(" + valid->str() + ", " + result + ", " +
           fill->str() + ")");
    bindResult(operation, 0, result);
    return success();
  }
  FailureOr<StringRef> source = lookupValue(operation, 0);
  bool expand = form == "expand_dims" &&
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
  if (alwaysValid || *valid == "True")
    line(result + " = " + expanded);
  else
    line(result + " = ct.where(" + valid->str() + ", " + expanded + ", " +
         fill->str() + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitMembers(Operation &operation) {
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
  if (!target::lowering::isRaggedBoundAxis(planIndex.components,
                                           memberAxis->getNode()))
    return operation.emitOpError(
        "has no single-launch ragged-member realization");
  FailureOr<plan::RaggedOp> relation =
      target::lowering::uniqueRaggedRelation(planIndex, memberAxis->getNode(),
                                              operation);
  auto runtime = succeeded(relation)
                     ? raggedRuntimeByRelation.find(relation->getNode())
                     : raggedRuntimeByRelation.end();
  std::string position = axisIndices.lookup(memberAxis->getNode());
  if (auto argument = dyn_cast<BlockArgument>(operation.getOperand(0))) {
    FailureOr<target::lowering::RegionRangeBinding> selected =
        target::lowering::selectedRegionArgumentRange(planIndex, kernel, argument,
                                                      operation);
    FailureOr<StringRef> projected = lookupValue(operation, 0);
    if (failed(selected) || failed(projected) ||
        selected->axis.getNode() != memberAxis->getNode())
      return operation.emitOpError(
          "has no exact cuTile region projection for ragged members");
    position = projected->str();
  }
  if (failed(relation) || runtime == raggedRuntimeByRelation.end() ||
      position.empty())
    return operation.emitOpError("has no ragged runtime position");
  RaggedRuntime &ragged = raggedRuntimes[runtime->second];
  std::string suffix = std::to_string(memberAxis->getNode());
  std::string valid = position + " < sequence_end_" + suffix;
  std::string result = makeResultName(operation, 0);
  if (ragged.indices) {
    emitGuardedGather(result, ragged.indices->argument->name,
                      addressIndex(position), "0", valid);
  } else {
    line(result + " = ct.where(" + valid + ", " + position + ", 0)");
  }
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical cuTile stream binding");
  Block &body = operation.getRegion(0).front();
  bool hasStop = static_cast<bool>(binding.getStopValueAttr());
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
  bool partitionedStream = binding.hasPartition();
  if (raggedStream && partitionedStream)
    return binding.emitOpError(
        "cannot mechanically combine ragged and count-partition stream bounds");
  std::string raggedSuffix = std::to_string(binding.getAxisNode());
  if (raggedStream && !planIndex.components.raggedProgramAxes.contains(
                          binding.getAxisNode())) {
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
    line("sequence_begin_" + raggedSuffix + " = ct.gather(" +
         ragged.offsets->argument->name + ", " + addressIndex(outer) +
         ", padding_value=0)");
    line("sequence_end_" + raggedSuffix + " = ct.gather(" +
         ragged.offsets->argument->name + ", " + addressIndex(outer + " + 1") +
         ", padding_value=0)");
    line("sequence_length_" + raggedSuffix + " = sequence_end_" +
         raggedSuffix + " - sequence_begin_" + raggedSuffix);
  }
  std::string partitionBegin;
  std::string partitionEnd;
  std::string partitionExtent;
  std::string activeEnd;
  std::string streamExtent = raggedStream
                                 ? "sequence_length_" + raggedSuffix
                                 : binding.getExtent().str();
  if (!raggedStream && !partitionedStream) {
    auto owner = dimensionOwners.find(streamExtent);
    if (owner != dimensionOwners.end())
      streamExtent = owner->second;
  }
  if (partitionedStream) {
    plan::PartitionBindingOp partition = binding.getPartition();
    plan::AxisOp axis = planIndex.axes.lookup(partition.getAxisNode());
    const target::lowering::RangeBinding *ownership =
        axis ? axis.getRange("ownership", 0) : nullptr;
    std::string part = programBlocks.lookup(partition.getAxisNode());
    std::string logical = axisDimensions.lookup(partition.getAxisNode());
    if (!axis || !ownership || part.empty() || logical.empty())
      return binding.emitOpError(
          "has no exact cuTile count-partition stream interval");
    auto logicalOwner = dimensionOwners.find(logical);
    if (logicalOwner != dimensionOwners.end())
      logical = logicalOwner->second;
    std::string suffix = std::to_string(*node);
    partitionBegin = "stream_segment_begin_" + suffix;
    partitionEnd = "stream_segment_end_" + suffix;
    partitionExtent = ownership->getTile().str();
    line(partitionBegin + " = min(" + addressIndex(part) + " * " +
         ownership->getTile().str() + ", " + logical + ")");
    line(partitionEnd + " = min(" + partitionBegin + " + " +
         ownership->getTile().str() + ", " + logical + ")");
    streamExtent = "max(0, " + partitionEnd + " - " + partitionBegin + ")";
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
      line(activeEnd + " = min(" + stop->str() + ", " + partitionEnd + ")");
      streamExtent =
          "max(0, " + activeEnd + " - " + partitionBegin + ")";
    } else {
      streamExtent = "max(0, min(" + stop->str() + ", " + streamExtent + "))";
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
    bool tiledTransfer = false;
    operation.walk([&](Operation *nested) {
      auto nestedNode = nested->getAttrOfType<IntegerAttr>("intent.node");
      plan::BoundaryOp boundary =
          nestedNode ? planIndex.boundaries.lookup(nestedNode.getInt())
                     : plan::BoundaryOp();
      tiledTransfer |=
          boundary && boundary.getTransfer() == "partition_stream_tile";
    });
    if (tiledTransfer)
      line("ct.static_assert((" + partitionExtent + ") % (" +
           binding.getTile().str() +
           ") == 0, 'partition stream tile must align to the logical part')");
  }
  std::string streamTile = "stream_tile_" + std::to_string(*node);
  std::string projectedRegion = streamTile;
  auto streamForm = binding.physical
                        ? binding.physical->getAttrOfType<StringAttr>(streamFormAttr)
                        : StringAttr();
  bool prefixBoundary = streamForm && streamForm.getValue() == "prefix_boundary";
  if (prefixBoundary) {
    auto boundaryAxis =
        binding.physical->getAttrOfType<IntegerAttr>(streamBoundaryAxisAttr);
    auto neutralMasks =
        binding.physical->getAttrOfType<DenseI64ArrayAttr>(streamNeutralMasksAttr);
    plan::AxisOp boundary =
        boundaryAxis ? planIndex.axes.lookup(boundaryAxis.getInt()) : plan::AxisOp();
    const target::lowering::RangeBinding *ownership =
        boundary ? boundary.getRange("ownership", 0) : nullptr;
    std::string block =
        boundaryAxis ? programBlocks.lookup(boundaryAxis.getInt()) : std::string();
    if (!boundaryAxis || !neutralMasks || neutralMasks.empty() || !boundary ||
        !ownership || block.empty() || raggedStream || partitionedStream)
      return binding.emitOpError(
          "has no complete cuTile prefix-boundary stream form");
    std::string prefixBlocks =
        "stream_prefix_blocks_" + std::to_string(*node);
    std::string boundaryStart =
        addressIndex(block) + " * " + ownership->getTile().str();
    line(prefixBlocks + " = min(ct.cdiv(" + streamExtent + ", " +
         binding.getTile().str() + "), max(0, (" + boundaryStart +
         ") // " + binding.getTile().str() + "))");
    line("for " + streamTile + " in range(" + addressIndex("0") + ", " +
         addressIndex(prefixBlocks) + ", " + addressIndex("1") + "):");
    PrefixBoundaryStream replay{streamTile, streamExtent, prefixBlocks, {}};
    for (int64_t mask : neutralMasks.asArrayRef()) {
      replay.neutralMasks.push_back(mask);
      activeNeutralMasks.insert(mask);
    }
    prefixBoundaryStreams[&operation] = std::move(replay);
  } else {
    line("for " + streamTile + " in range(" + addressIndex("0") + ", " +
         addressIndex("ct.cdiv(" + streamExtent + ", " +
                      binding.getTile().str() + ")") +
         ", " + addressIndex("1") + "):");
  }
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
    projectedRegion = offsets;
  } else if (partitionedStream) {
    std::string offsets = "stream_axis_index_" + std::to_string(*node);
    line(offsets + " = " + partitionBegin + " + " + streamTile + " * " +
         binding.getTile().str() + " + " +
         addressIndex("ct.arange(" + binding.getTile().str() +
                      ", dtype=ct.int32)"));
    projectedRegion = offsets;
    partitionStreamBaseIndices[body.getArgument(0)] =
        "(" + partitionBegin + ") // (" + binding.getTile().str() + ") + " +
        streamTile;
    absoluteRegionArguments.insert(body.getArgument(0));
  }
  valueNames[body.getArgument(0)] = projectedRegion;
  auto bindScopedAxis = [&](int64_t axisNode, std::string value) {
    auto previous = axisIndices.find(axisNode);
    std::optional<std::string> restore;
    if (previous != axisIndices.end())
      restore = previous->second;
    streamAxisRestores[&operation].emplace_back(axisNode, std::move(restore));
    axisIndices[axisNode] = std::move(value);
  };
  bindScopedAxis(binding.getAxisNode(), projectedRegion);
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
    return operation.emitOpError("has no active cuTile stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (::intent::target::semanticOperationName(terminator) != "intent.yield" ||
      terminator.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("does not yield every cuTile stream state");
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(terminator, index);
    if (failed(yielded))
      return failure();
    line(carriers->second[index] + " = " + yielded->str());
  }
  --indentation;
  auto prefix = prefixBoundaryStreams.find(&operation);
  if (prefix != prefixBoundaryStreams.end()) {
    for (int64_t mask : prefix->second.neutralMasks)
      activeNeutralMasks.erase(mask);
    line("for " + prefix->second.block + " in range(" +
         addressIndex(prefix->second.prefixBlocks) + ", " +
         addressIndex("ct.cdiv(" + prefix->second.extent + ", " +
                      binding.getTile().str() + ")") +
         ", " + addressIndex("1") + "):");
    ++indentation;
    Block &body = operation.getRegion(0).front();
    valueNames[body.getArgument(0)] = prefix->second.block;
    axisIndices[binding.getAxisNode()] = prefix->second.block;
    if (failed(replayBlock(body)))
      return failure();
    for (unsigned index = 0; index < operation.getNumResults(); ++index) {
      FailureOr<StringRef> replayed = lookupValue(terminator, index);
      if (failed(replayed))
        return failure();
      line(carriers->second[index] + " = " + replayed->str());
    }
    --indentation;
    prefixBoundaryStreams.erase(prefix);
  }
  for (unsigned index = 0; index < operation.getNumResults(); ++index)
    valueNames[operation.getResult(index)] = carriers->second[index];
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
  absoluteRegionArguments.erase(operation.getRegion(0).front().getArgument(0));
  partitionStreamBaseIndices.erase(
      operation.getRegion(0).front().getArgument(0));
  return success();
}

LogicalResult ProgramMaterializer::emitContract(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "ct.mma" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a cuTile contraction binding");
  StringRef form = binding.getForm();
  FailureOr<target::lowering::ContractionOrientation> orientation =
      target::lowering::selectedContractionOrientation(binding);
  if (failed(orientation))
    return failure();
  auto resultTensor = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  std::string accumulatorDtype =
      resultTensor ? dtypeName(resultTensor.getElementType(), operation) : "";
  if (accumulatorDtype.empty())
    return operation.emitOpError("has no supported cuTile accumulator dtype");
  auto replay = deferredContractReplays.find(&operation);
  if (form == "replay") {
    if (replay == deferredContractReplays.end())
      return operation.emitOpError(
          "cuTile replay form has no physical producer slice");
    std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
    plan::AxisOp reductionAxis =
        reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
    FailureOr<std::string> resultShape = emitTensorShape(operation, 0);
    if (!reductionAxis || failed(resultShape))
      return operation.emitOpError(
          "replay form has no selected cuTile reduction axis");
    bool streamReduction = target::lowering::isEnclosingStreamReductionAxis(
        planIndex, operation, reductionAxis.getNode());
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.full(" + *resultShape + ", 0, dtype=" +
         accumulatorDtype + ")");
    if (!streamReduction) {
      std::string reductionExtent =
          roleDimensions.lookup(reductionAxis.getRole());
      if (reductionExtent.empty())
        return reductionAxis.emitOpError(
            "has no cuTile physical reduction extent binding");
      line("num_tiles_k = ct.cdiv(" + reductionExtent + ", " +
           reductionAxis.getTile().str() + ")");
      line("for k_tile in range(num_tiles_k):");
      ++indentation;
      axisIndices[reductionAxis.getNode()] = "k_tile";
    } else if (axisIndices.lookup(reductionAxis.getNode()).empty()) {
      return operation.emitOpError(
          "has no active cuTile stream-bound reduction range");
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
    std::string lhsExpression = lhs->str();
    std::string rhsExpression = rhs->str();
    if (orientation->lhsTranspose)
      lhsExpression = "ct.transpose(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = "ct.transpose(" + rhsExpression + ")";
    line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression +
         ", " + result + ")");
    restoreValues();
    if (!streamReduction)
      --indentation;
    bindResult(operation, 0, result);
    return success();
  }
  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (form == "direct") {
    if (lhsLoad || rhsLoad)
      return operation.emitOpError(
          "cuTile direct contraction unexpectedly owns deferred operands");
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    FailureOr<std::string> shape = emitTensorShape(operation, 0);
    auto mmaForm =
        binding.operation->getAttrOfType<StringAttr>(contractMmaFormAttr);
    bool transposeResult =
        mmaForm && mmaForm.getValue() == "transposed_result";
    FailureOr<std::string> transposedShape =
        transposeResult ? emitTensorShape(operation, 0, true)
                        : FailureOr<std::string>(std::string());
    if (failed(lhs) || failed(rhs) || failed(shape) || !mmaForm ||
        (mmaForm.getValue() != "native" && !transposeResult) ||
        failed(transposedShape))
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
    if (transposeResult) {
      std::string transposed = result + "_transposed";
      line(transposed + " = ct.full(" + *transposedShape +
           ", 0, dtype=" + accumulatorDtype + ")");
      line(transposed + " = ct.mma(ct.transpose(" + rhsExpression +
           "), ct.transpose(" + lhsExpression + "), " + transposed + ")");
      line(result + " = ct.transpose(" + transposed + ")");
    } else {
      line(result + " = ct.full(" + *shape + ", 0, dtype=" +
           accumulatorDtype + ")");
      line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression +
           ", " + result + ")");
    }
    bindResult(operation, 0, result);
    return success();
  }
  if (form == "deferred_one") {
    if (static_cast<bool>(lhsLoad) == static_cast<bool>(rhsLoad))
      return operation.emitOpError(
          "cuTile one-sided deferred form does not own exactly one deferred operand");
    Operation *load = lhsLoad ? lhsLoad : rhsLoad;
    unsigned loadOperand = lhsLoad ? 0 : 1;
    unsigned directOperand = lhsLoad ? 1 : 0;
    StringRef loadSpace =
        lhsLoad ? binding.getLhsSpace() : binding.getRhsSpace();
    StringRef directSpace =
        lhsLoad ? binding.getRhsSpace() : binding.getLhsSpace();
    if (loadSpace != "shared" || directSpace != "private_fragment" ||
        binding.getAccumulatorSpace() != "private_fragment")
      return operation.emitOpError(
          "one-sided deferred cuTile contraction has inconsistent plan spaces");
    std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
    plan::AxisOp reductionAxis =
        reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
    if (!reductionAxis)
      return operation.emitOpError(
          "one-sided deferred cuTile form has no selected reduction axis");
    if (!target::lowering::isEnclosingStreamReductionAxis(
            planIndex, operation, reductionAxis.getNode()) ||
        axisIndices.lookup(reductionAxis.getNode()).empty())
      return operation.emitOpError(
          "one-sided deferred cuTile contraction requires an active "
          "stream-bound reduction range");
    FailureOr<int64_t> loadNode =
        target::getNodeID(*load, "cuTile contraction transfer");
    plan::BoundaryOp boundary =
        succeeded(loadNode) ? planIndex.boundaries.lookup(*loadNode)
                            : plan::BoundaryOp();
    FailureOr<ABIView *> view = lookupView(load->getOperand(0), *load);
    bool gather = boundary && boundary.getAccess() == "gather";
    FailureOr<std::string> index =
        boundary ? indexTuple(*load, gather)
                 : FailureOr<std::string>(failure());
    FailureOr<std::string> shape =
        gather ? FailureOr<std::string>(std::string()) : tileShape(*load);
    bool transpose = loadOperand == 0 ? orientation->lhsTranspose
                                      : orientation->rhsTranspose;
    bool permutePhysical = transpose && succeeded(view) &&
                           (*view)->tensor.getRank() > 2 && !gather;
    FailureOr<std::string> resultShape =
        emitTensorShape(*load, 0, permutePhysical);
    FailureOr<std::string> contractShape = emitTensorShape(operation, 0);
    FailureOr<StringRef> direct = lookupValue(operation, directOperand);
    if (failed(loadNode) || !boundary ||
        (boundary.getAccess() != "load" && !gather) || failed(view) ||
        failed(index) || failed(shape) || failed(resultShape) ||
        failed(contractShape) || failed(direct))
      return operation.emitOpError(
          "has no one-sided deferred cuTile contraction transfer binding");
    std::string dtype =
        dtypeName((*view)->tensor.getElementType(), operation);
    if (dtype.empty())
      return failure();
    std::string loaded = makeResultName(*load, 0);
    std::string physical = loaded + "_physical";
    if (gather) {
      StringRef padding =
          isa<IntegerType, IndexType>((*view)->tensor.getElementType()) ? "0"
                                                                       : "0.0";
      line(physical + " = ct.gather(" + (*view)->argument->name + ", " +
           *index + ", check_bounds=True, padding_value=" + padding.str() +
           ")");
    } else {
      line(physical + " = ct.load(" + (*view)->argument->name +
           ", index=" + *index + ", shape=" + *shape +
           ", padding_mode=ct.PaddingMode.ZERO)");
    }
    if (permutePhysical) {
      SmallVector<unsigned> permutation((*view)->tensor.getRank());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::swap(permutation[permutation.size() - 2], permutation.back());
      std::string axes = "(";
      for (auto [axis, value] : llvm::enumerate(permutation)) {
        if (axis)
          axes += ", ";
        axes += std::to_string(value);
      }
      axes += ")";
      line(physical + " = ct.permute(" + physical + ", " + axes + ")");
    }
    line(loaded + " = " + physical + ".reshape(" + *resultShape +
         ").astype(" + dtype + ")");
    if (transpose && !permutePhysical)
      line(loaded + " = ct.transpose(" + loaded + ")");
    std::string lhsExpression = loadOperand == 0 ? loaded : direct->str();
    std::string rhsExpression = loadOperand == 1 ? loaded : direct->str();
    if (directOperand == 0 && orientation->lhsTranspose)
      lhsExpression = "ct.transpose(" + lhsExpression + ")";
    if (directOperand == 1 && orientation->rhsTranspose)
      rhsExpression = "ct.transpose(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.full(" + *contractShape + ", 0, dtype=" +
         accumulatorDtype + ")");
    line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression +
         ", " + result + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (form != "deferred_two" || !lhsLoad || !rhsLoad)
    return operation.emitOpError("has an inconsistent cuTile contraction form");
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
  std::optional<int64_t> lhsAxisNode = binding.getLhsResultAxisNode();
  std::optional<int64_t> rhsAxisNode = binding.getRhsResultAxisNode();
  std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
  plan::AxisOp lhsResult =
      lhsAxisNode ? planIndex.axes.lookup(*lhsAxisNode) : plan::AxisOp();
  plan::AxisOp rhsResult =
      rhsAxisNode ? planIndex.axes.lookup(*rhsAxisNode) : plan::AxisOp();
  plan::AxisOp reductionAxis =
      reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
  if (failed(lhsView) || failed(rhsView) || failed(lhsBoundary) ||
      failed(rhsBoundary) || !lhsResult || !rhsResult || !reductionAxis)
    return failure();
  bool streamReduction = target::lowering::isEnclosingStreamReductionAxis(
      planIndex, operation, reductionAxis.getNode());
  if (!streamReduction)
    axisIndices[reductionAxis.getNode()] = "k_tile";
  else if (axisIndices.lookup(reductionAxis.getNode()).empty())
    return operation.emitOpError(
        "has no active cuTile stream-bound reduction range");
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
  FailureOr<std::string> lhsTile = physicalAxisTile(lhsResult);
  FailureOr<std::string> rhsTile = physicalAxisTile(rhsResult);
  if (failed(lhsIndex) || failed(rhsIndex) || failed(lhsShape) || failed(rhsShape) ||
      failed(lhsResultShape) || failed(rhsResultShape) || failed(lhsTile) ||
      failed(rhsTile))
    return failure();

  std::string result = makeResultName(operation, 0);
  if (!streamReduction) {
    std::string reductionExtent = roleDimensions.lookup(reductionAxis.getRole());
    if (reductionExtent.empty())
      return reductionAxis.emitOpError(
          "has no cuTile physical reduction extent binding");
    line("num_tiles_k = ct.cdiv(" + reductionExtent + ", " +
         reductionAxis.getTile().str() + ")");
  }
  line(result + " = ct.full((" + *lhsTile + ", " + *rhsTile +
       "), 0, dtype=" + accumulatorDtype + ")");
  std::string operandDtype =
      dtypeName((*lhsView)->tensor.getElementType(), operation);
  if (operandDtype.empty())
    return failure();
  if (!streamReduction) {
    line("for k_tile in range(num_tiles_k):");
    ++indentation;
  }
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
    bool permutePhysical = transpose && !gather;
    if (permutePhysical) {
      SmallVector<unsigned> permutation(view.tensor.getRank());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::swap(permutation[permutation.size() - 2], permutation.back());
      std::string axes = "(";
      for (auto [axis, value] : llvm::enumerate(permutation)) {
        if (axis)
          axes += ", ";
        axes += std::to_string(value);
      }
      axes += ")";
      line(physical + " = ct.permute(" + physical + ", " + axes + ")");
    }
    line(name.str() + " = " + physical + ".reshape(" + resultShape.str() +
         ").astype(" + operandDtype + ")");
    if (transpose && !permutePhysical)
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
  if (!streamReduction)
    --indentation;
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitScaledContract(Operation &operation) {
  FailureOr<int64_t> node =
      target::getNodeID(operation, "scaled-contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "ct.mma_scaled" ||
      operation.getNumOperands() != 4 || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a cuTile scaled-contraction binding");
  if (binding.getForm() != "scaled_direct")
    return operation.emitOpError(
        "has no realized direct cuTile scaled-contraction form");
  for (Value operand : operation.getOperands())
    if (deferredLoads.count(operand))
      return operation.emitOpError(
          "cuTile scaled contraction requires materialized operand tiles");
  auto lhsGroup =
      operation.getAttrOfType<IntegerAttr>("intent.lhs_group_size");
  auto rhsGroup =
      operation.getAttrOfType<IntegerAttr>("intent.rhs_group_size");
  FailureOr<StringRef> lhs = lookupValue(operation, 0);
  FailureOr<StringRef> rhs = lookupValue(operation, 1);
  FailureOr<StringRef> lhsScale = lookupValue(operation, 2);
  FailureOr<StringRef> rhsScale = lookupValue(operation, 3);
  FailureOr<std::string> shape = emitTensorShape(operation, 0);
  if (!lhsGroup || !rhsGroup || lhsGroup.getInt() != 32 ||
      rhsGroup.getInt() != 32 || failed(lhs) || failed(rhs) ||
      failed(lhsScale) || failed(rhsScale) || failed(shape))
    return operation.emitOpError(
        "cuTile scaled contraction requires matching K-group size 32");
  StringRef layout = binding.getScaledLayout();
  if (layout != "grouped_rank_two" && layout != "flattened_rank_three")
    return operation.emitOpError(
        "has no selected cuTile scaled-contraction layout");
  std::string lhsExpression = lhs->str();
  std::string rhsExpression = rhs->str();
  if (layout == "flattened_rank_three") {
    lhsExpression += ".reshape((" + lhs->str() + ".shape[0], " + lhs->str() +
                     ".shape[1] * " + lhs->str() + ".shape[2]))";
    rhsExpression += ".reshape((" + rhs->str() + ".shape[0] * " + rhs->str() +
                     ".shape[1], " + rhs->str() + ".shape[2]))";
  }
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.full(" + *shape + ", 0, dtype=ct.float32)");
  line(result + " = ct.mma_scaled(" + lhsExpression + ", " + lhsScale->str() +
       ", " + rhsExpression + ", " + rhsScale->str() + ", " + result + ")");
  bindResult(operation, 0, result);
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
  bool scatter = boundary.getAccess() == "scatter" ||
                 target::lowering::hasPackedScalarDomain(planIndex, boundary);
  FailureOr<std::string> indices = indexTuple(operation, scatter);
  if (!valueIndex || failed(node) || !boundary || failed(stored) ||
      failed(view) || failed(indices))
    return operation.emitOpError("lacks a cuTile store binding");
  std::string scatterMask;
  if (scatter && boundary.getTensorIndexing() != "none" &&
      !boundary.getValidityTensorAxes().empty()) {
    FailureOr<std::string> validity = emitValidityExpression(
        boundary.getValidityTensorAxes(), boundary.getValidityDomainNodes(),
        operation.getOperand(valueIndex.getInt()), operation);
    if (failed(validity))
      return failure();
    if (*validity != "True")
      scatterMask = *validity;
  }
  if (scatter)
    line("ct.scatter(" + (*view)->argument->name + ", " + *indices + ", " +
         stored->str() + ", check_bounds=True" +
         (scatterMask.empty() ? std::string()
                              : ", mask=" + scatterMask) +
         ")");
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
  if (failed(node) || !binding ||
      (binding.getAccess() != "store" && binding.getAccess() != "scatter") ||
      binding.getDefer() || !valueIndex ||
      failed(relation) || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical cuTile unique store");
  if (binding.getTransfer() == "advanced_index_store") {
    FailureOr<std::string> indices = advancedIndexTuple(operation);
    FailureOr<std::string> shape = tileShape(operation);
    FailureOr<unsigned> storedRank = emittedTensorRank(operation, true);
    if (binding.getAccess() != "store" || failed(indices) || failed(shape) ||
        failed(storedRank))
      return operation.emitOpError(
          "lacks an advanced-index cuTile unique-store binding");
    std::string tile = stored->str();
    if (*storedRank != static_cast<unsigned>((*view)->tensor.getRank()))
      tile += ".reshape(" + *shape + ")";
    line("ct.store_advanced_indexing(" + (*view)->argument->name + ", " +
         *indices + ", " + tile + ")");
    return success();
  }
  if (binding.getAccess() == "store")
    return emitStore(operation);
  FailureOr<std::string> indices = indexTuple(operation, true);
  std::string mask;
  if (failed(indices))
    return failure();
  if (!binding.getValidityTensorAxes().empty()) {
    FailureOr<std::string> validity = emitValidityExpression(
        binding.getValidityTensorAxes(), binding.getValidityDomainNodes(),
        operation.getOperand(valueIndex.getInt()), operation);
    if (failed(validity))
      return failure();
    if (*validity != "True")
      mask = mask.empty() ? *validity
                          : "(" + mask + ") & (" + *validity + ")";
  }
  line("ct.scatter(" + (*view)->argument->name + ", " + *indices + ", " +
       stored->str() + ", check_bounds=True" +
       (mask.empty() ? std::string() : ", mask=" + mask) + ")");
  return success();
}

LogicalResult ProgramMaterializer::emitAtomic(Operation &operation) {
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<StringRef> stored =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  if (!valueIndex || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical cuTile atomic merge");
  if (::intent::target::semanticOperationName(operation) == "intent.scatter_reduce" &&
      isa<BFloat16Type>((*view)->tensor.getElementType()))
    return operation.emitOpError(
        "cannot project a high-throughput bfloat16 many-to-one scatter "
        "reduction with the configured cuTile native atomic surface; the "
        "available tile atomic projection is intentionally unsupported");
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

LogicalResult ProgramMaterializer::emitAtomicCas(Operation &operation) {
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
      !compareIndex || !valueIndex || !ordering ||
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

} // namespace intent::cutile::lowering

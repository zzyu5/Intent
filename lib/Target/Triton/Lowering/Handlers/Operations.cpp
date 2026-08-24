#include "Support/Model.h"
#include "Syntax/Spelling.h"

#include "Intent/Target/Triton/Lowering/Passes.h"
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

namespace intent::triton::lowering {
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
  if (type.isInteger(1))
    return "tl.int1";
  if (type.isF16())
    return "tl.float16";
  if (type.isF32())
    return "tl.float32";
  if (type.isBF16())
    return "tl.bfloat16";
  if (isa<Float8E4M3FNType>(type))
    return "tl.float8e4nv";
  if (isa<Float8E5M2Type>(type))
    return "tl.float8e5";
  if (isa<Float8E8M0FNUType>(type))
    return "tl.uint8";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 8)
    return integer.isUnsigned() ? "tl.uint8" : "tl.int8";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 16)
    return integer.isUnsigned() ? "tl.uint16" : "tl.int16";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 32)
    return integer.isUnsigned() ? "tl.uint32" : "tl.int32";
  if (auto integer = dyn_cast<IntegerType>(type);
      integer && integer.getWidth() == 64)
    return integer.isUnsigned() ? "tl.uint64" : "tl.int64";
  if (isa<IndexType>(type))
    return "tl.int64";
  return {};
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
      failed(addHandler(registry, "intent.join", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitJoin(op);
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
      expression = number < 0 ? "-float('inf')" : "float('inf')";
    else if (std::isnan(number))
      expression = "float('nan')";
    else
      expression = target::lowering::spellFiniteFloatLiteral(floating);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported Triton constant value");
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
    return operation.emitOpError("record field has no emitted Triton SSA value");
  bindResult(operation, 0, found->second);
  return success();
}

LogicalResult ProgramMaterializer::emitRegionEnd(Operation &operation) {
  if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
      !operation.getResult(0).getType().isIntOrIndex())
    return operation.emitOpError("lacks a mechanical Triton region-end binding");
  FailureOr<plan::AxisOp> axis = resolveAxis(operation.getOperand(0), operation);
  if (failed(axis))
    return failure();
  std::string extent = axisDimensions.lookup(axis->getNode());
  if (extent.empty())
    return operation.emitOpError("has no planned logical extent for region end");
  FailureOr<std::optional<target::lowering::RegionRangeBinding>> selected =
      target::lowering::selectedRegionValueRange(
          planIndex, kernel, operation.getOperand(0), operation);
  if (failed(selected))
    return failure();
  std::string logicalEnd = extent;
  if (target::lowering::isRaggedBoundAxis(planIndex.components,
                                           axis->getNode()))
    logicalEnd = "sequence_end_" + std::to_string(axis->getNode());
  std::string expression = logicalEnd;
  if (*selected) {
    if ((*selected)->axis.getNode() != axis->getNode())
      return operation.emitOpError(
          "selected region-end range does not bind its resolved axis");
    StringRef purpose = (*selected)->range.getPurpose();
    if ((purpose == "ownership" || purpose == "traversal") &&
        !axis->isScalar()) {
      auto start = selectedRegionStarts.find(operation.getOperand(0));
      if (start == selectedRegionStarts.end())
        return operation.emitOpError(
            "has no exact selected-region start for region end");
      expression = "tl.minimum((" + start->second + ") + " +
                   (*selected)->range.getTile().str() + ", " + logicalEnd + ")";
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

LogicalResult ProgramMaterializer::emitProgramBindings() {
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
         target::lowering::projectProgramVolume(planIndex, axisExtent));
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
          target::lowering::projectLinearGroupIndex(
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
    axisStarts[lhs.getNode()] = lhsBlock + " * " + lhs.getTile().str();
    axisStarts[rhs.getNode()] = rhsBlock + " * " + rhs.getTile().str();
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

  SmallVector<target::lowering::ProgramIndexProjection> projections =
      persistent
          ? target::lowering::projectLinearProgramIndices(
                planIndex, axisExtent, linear)
          : target::lowering::projectProgramIndices(
                planIndex, axisExtent, [&](unsigned worker) {
                  return addressIndex("tl.program_id(axis=" +
                                      std::to_string(worker) + ")");
                });
  for (const target::lowering::ProgramIndexProjection &projection : projections) {
    plan::AxisOp axis = projection.axis;
    std::string block = "block_axis_" + std::to_string(axis.getNode());
    line(block + " = " + projection.expression);
    programBlocks[axis.getNode()] = block;
    if (axis.isScalar()) {
      axisIndices[axis.getNode()] = block;
      axisStarts[axis.getNode()] = block;
      continue;
    }
    std::string value = "axis_index_" + std::to_string(axis.getNode());
    if (planIndex.components.raggedProgramAxes.contains(axis.getNode())) {
      axisIndices[axis.getNode()] = block;
      continue;
    }
    line(value + " = " + block + " * " + axis.getTile().str() +
         " + tl.arange(0, " + axis.getTile().str() + ")");
    axisIndices[axis.getNode()] = value;
    axisStarts[axis.getNode()] = block + " * " + axis.getTile().str();
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
    if (!axis || axis.isScalar() || block.empty() || failed(relation) ||
        outer.empty() || runtime == raggedRuntimeByRelation.end())
      return programRoot->emitOpError(
          "ragged program axis has no tiled outer-axis or offsets binding");
    ABIView *offsets = raggedRuntimes[runtime->second].offsets;
    SmallVector<int64_t> boundAxes{memberNode};
    auto ordered =
        planIndex.components.orderedAxesByRelation.find(relation->getNode());
    if (ordered != planIndex.components.orderedAxesByRelation.end())
      for (int64_t orderedAxis : ordered->second)
        if (!llvm::is_contained(boundAxes, orderedAxis))
          boundAxes.push_back(orderedAxis);
    for (int64_t boundAxis : boundAxes) {
      std::string suffix = std::to_string(boundAxis);
      line("sequence_begin_" + suffix + " = tl.load(" + offsets->pointer +
           " + " + addressIndex(outer) + " * " +
           addressIndex(offsets->strides[0]) + ")");
      line("sequence_end_" + suffix + " = tl.load(" + offsets->pointer +
           " + " + addressIndex(outer + " + 1") + " * " +
           addressIndex(offsets->strides[0]) + ")");
      line("sequence_length_" + suffix + " = sequence_end_" + suffix +
           " - sequence_begin_" + suffix);
    }
    std::string suffix = std::to_string(memberNode);
    std::string value = "axis_index_" + suffix;
    line(value + " = sequence_begin_" + suffix + " + " + block + " * " +
         axis.getTile().str() + " + tl.arange(0, " + axis.getTile().str() +
         ")");
    axisIndices[memberNode] = value;
    axisStarts[memberNode] = "sequence_begin_" + suffix + " + " + block +
                             " * " + axis.getTile().str();
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
    std::string physical = axis.getTileRole().starts_with("fixed_")
                               ? axis.getTile().str()
                               : physicalExtent(extent);
    line(value + " = tl.arange(0, " + physical + ")");
    axisIndices[axis.getNode()] = value;
    axisStarts[axis.getNode()] = "0";
  }
  for (const auto &entry : planIndex.axes) {
    plan::AxisOp axis = entry.second;
    if (!axisIndices.lookup(axis.getNode()).empty())
      continue;
    const target::lowering::RangeBinding *range =
        axis.getRange("reduction", 0);
    if (!range)
      continue;
    std::string value = "axis_index_" + std::to_string(axis.getNode());
    line(value + " = tl.arange(0, " + range->getTile().str() + ")");
    axisIndices[axis.getNode()] = value;
    axisStarts[axis.getNode()] = "0";
  }
  for (const auto &entry : planIndex.regionBindings) {
    Value value = kernel.values.lookup(entry.first);
    plan::RegionBindingOp binding = entry.second;
    if (!value || isa<BlockArgument>(value) || binding.getPurpose() != "ownership")
      continue;
    std::string projected = axisIndices.lookup(binding.getAxisNode());
    if (projected.empty())
      return binding.emitOpError(
          "has no active Triton projection for its selected region value");
    selectedRegionIndices[value] = std::move(projected);
    std::string start = axisStarts.lookup(binding.getAxisNode());
    if (start.empty())
      return binding.emitOpError(
          "has no active Triton start for its selected region value");
    selectedRegionStarts[value] = std::move(start);
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
          "count partition cannot use worker-reused Triton ownership");
    if (&operation == programRoot && failed(emitProgramBindings()))
      return failure();
    plan::AxisOp axis = planIndex.axes.lookup(partition.getAxisNode());
    std::string part = programBlocks.lookup(partition.getAxisNode());
    std::string region = axisIndices.lookup(partition.getAxisNode());
    if (!axis || part.empty() || region.empty())
      return partition.emitOpError(
          "has no emitted Triton part and region projection");
    valueNames[body.getArgument(0)] = part;
    valueNames[body.getArgument(1)] = region;
    selectedRegionStarts[body.getArgument(0)] = part;
    selectedRegionStarts[body.getArgument(1)] =
        axisStarts.lookup(partition.getAxisNode());
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
         "num_stages=PIPELINE_STAGES):");
    ++indentation;
    line(vectorIndex + " = tl.arange(0, BLOCK_SIZE)");
    axisIndices[axis.getNode()] = programIndex;
    valueNames[argument] = programIndex;
    selectedRegionStarts[argument] = programIndex;
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
    selectedRegionStarts[argument] = axisStarts.lookup(axis.getNode());
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
    return operation.emitOpError("lacks a mechanical Triton sequential loop");
  Block &body = operation.getRegion(0).front();
  if (body.getNumArguments() != domains->size() + operation.getNumResults())
    return operation.emitOpError("does not match its Triton sequential axes");
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
      return operation.emitOpError("has a non-canonical Triton loop axis");
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
        operation.emitOpError("has no emitted Triton loop bound");
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
      ABIView *offsets = raggedRuntimes[runtime->second].offsets;
      raggedIndices = raggedRuntimes[runtime->second].indices;
      std::string suffix = std::to_string(*domainNode);
      std::string begin = "sequence_begin_" + suffix;
      std::string end = "sequence_end_" + suffix;
      line(begin + " = tl.load(" + offsets->pointer + " + " +
           addressIndex(outer) + " * " + addressIndex(offsets->strides[0]) +
           ")");
      line(end + " = tl.load(" + offsets->pointer + " + " +
           addressIndex(outer + " + 1") + " * " +
           addressIndex(offsets->strides[0]) + ")");
      start = begin;
      stop = end;
    } else {
      return operation.emitOpError("has a non-canonical Triton loop axis");
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
            "has an invalid two-level Triton ordered traversal");
      std::string chunk = iterator + "_chunk";
      line("for " + chunk + " in tl.range(0, tl.cdiv(" + *stop + " - " +
           *start + ", " +
           outer->getTile().str() + "), flatten=True):");
      ++indentation;
      line("for " + iterator + " in tl.range(" + *start + " + " + chunk +
           " * " + outer->getTile().str() + ", tl.minimum(" + *start +
           " + (" + chunk + " + 1) * " + outer->getTile().str() + ", " +
           *stop + ")):");
      ++indentation;
    } else {
      line("for " + iterator + " in tl.range(" + *start + ", " + *stop +
           ", " + *step + "):");
      ++indentation;
    }
    std::string logicalIterator = iterator;
    if (raggedIndices) {
      logicalIterator = iterator + "_member";
      line(logicalIterator + " = tl.load(" + raggedIndices->pointer + " + " +
           addressIndex(iterator) + " * " +
           addressIndex(raggedIndices->strides[0]) + ")");
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
    return operation.emitOpError("has no active Triton sequential loop");
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
    return operation.emitOpError("lacks a mechanical Triton scalar branch");
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
    return operation.emitOpError("has no active Triton scalar branch");
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
    return operation.emitOpError("lacks a mechanical Triton scalar while");
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
    return operation.emitOpError("does not match a Triton scalar while");
  line("while " + condition->str() + ":");
  ++indentation;
  return success();
}

LogicalResult ProgramMaterializer::leaveWhile(Operation &operation) {
  auto carriers = whileCarriers.find(&operation);
  if (carriers == whileCarriers.end())
    return operation.emitOpError("has no active Triton scalar while");
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
      return operation.emitOpError("does not match its Triton loop state");
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
      return operation.emitOpError("does not match its Triton while state");
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
          "lacks a planned Triton private workspace parameter");
    return success();
  }
  if (binding && binding.getSpace() == "private_vector") {
    if (failed(info) || failed(initializer) ||
        tritonDtype(info->elementType).empty())
      return operation.emitOpError(
          "private Triton vectors require a supported static shape");
    std::string base = makeResultName(operation, 0);
    FailureOr<int64_t> elements =
        target::logicalBufferElementCount(*info, operation);
    if (failed(elements))
      return failure();
    int64_t physicalExtent = 1;
    while (physicalExtent < *elements)
      physicalExtent *= 2;
    line(base + "_lanes = tl.arange(0, " +
         std::to_string(physicalExtent) + ")");
    line(base + " = tl.full((" + std::to_string(physicalExtent) +
         ",), " + initializer->str() + ", " +
         tritonDtype(info->elementType).str() + ")");
    vectorBuffers[operation.getResult(0)] = base;
    return success();
  }
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

LogicalResult ProgramMaterializer::emitBufferLoad(Operation &operation) {
  auto workspace = operation.getNumOperands() > 0
                       ? workspaceNames.find(operation.getOperand(0))
                       : workspaceNames.end();
  if (workspace != workspaceNames.end()) {
    FailureOr<std::string> pointer = privateWorkspacePointer(operation);
    if (failed(pointer) || operation.getNumResults() != 1)
      return operation.emitOpError(
          "lacks a mechanical Triton private-workspace load");
    std::string result = makeResultName(operation, 0);
    line(result + " = tl.load(" + *pointer + ")");
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
      return operation.emitOpError("lacks a private Triton vector load");
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
      line(result + " = tl.sum(tl.where(" + vector->second + "_lanes == " +
           selected + ", tl.cast(" + vector->second +
           ", tl.int32), 0), axis=0) != 0");
    } else {
      line(result + "_index = tl.full((1,), " + selected + ", tl.int32)");
      line(result + "_tile = tl.gather(" + vector->second + ", " + result +
           "_index, axis=0)");
      line(result + " = tl.sum(" + result + "_tile, axis=0)");
    }
    bindResult(operation, 0, result);
    return success();
  }
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
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

LogicalResult ProgramMaterializer::emitBufferStore(Operation &operation) {
  auto workspace = operation.getNumOperands() > 0
                       ? workspaceNames.find(operation.getOperand(0))
                       : workspaceNames.end();
  if (workspace != workspaceNames.end()) {
    FailureOr<std::string> pointer = privateWorkspacePointer(operation);
    FailureOr<StringRef> stored = lookupValue(operation, 1);
    if (failed(pointer) || failed(stored))
      return operation.emitOpError(
          "lacks a mechanical Triton private-workspace store");
    line("tl.store(" + *pointer + ", " + stored->str() + ")");
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
      return operation.emitOpError("lacks a private Triton vector store");
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
    line(vector->second + " = tl.where(" + vector->second +
         "_lanes == " + selected + ", " + stored->str() + ", " +
         vector->second + ")");
    return success();
  }
  auto buffer = operation.getNumOperands() > 0
                    ? scalarBuffers.find(operation.getOperand(0))
                    : scalarBuffers.end();
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

FailureOr<std::string>
ProgramMaterializer::privateWorkspacePointer(Operation &operation) {
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
  auto workspace = buffer && buffer->getNumResults() == 1
                       ? workspaceNames.find(buffer->getResult(0))
                       : workspaceNames.end();
  if (failed(node) || !binding || binding.getSpace() != "private_workspace" ||
      failed(info) || failed(indices) || workspace == workspaceNames.end())
    return operation.emitOpError(
        "does not resolve a planned Triton private workspace");

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
  return workspace->second + " + " + addressIndex(*offset);
}

FailureOr<std::string>
ProgramMaterializer::scanWorkspacePointer(Value result, const plan::ScanOp &binding,
                                    StringRef logicalIndex,
                                    Operation &consumer) {
  Operation *scan = kernel.nodes.lookup(binding.getNode());
  auto workspace = workspaceNames.find(result);
  std::string extent = scanExtents.lookup(binding.getNode());
  if (!scan || result.getDefiningOp() != scan ||
      workspace == workspaceNames.end() || extent.empty())
    return binding.emitOpError("has no Triton scan workspace binding");
  FailureOr<std::string> offset = target::lowering::projectScanWorkspaceOffset(
      binding, extent, logicalIndex, planIndex, axisIndices, axisDimensions,
      consumer);
  if (failed(offset))
    return failure();
  return workspace->second + " + " + addressIndex(*offset);
}

FailureOr<std::string>
ProgramMaterializer::scanMaterializedPointer(Value value, StringRef logicalIndex,
                                       Operation &consumer) {
  auto materialized = scanMaterializedValues.find(value);
  auto workspace = workspaceNames.find(value);
  if (materialized == scanMaterializedValues.end() ||
      workspace == workspaceNames.end())
    return consumer.emitOpError("has no Triton materialized scan value");
  FailureOr<std::string> offset = target::lowering::projectScanWorkspaceOffset(
      materialized->second, scanExtents.lookup(materialized->second.getNode()),
      logicalIndex, planIndex, axisIndices, axisDimensions, consumer);
  if (failed(offset))
    return failure();
  return workspace->second + " + " + addressIndex(*offset);
}

LogicalResult ProgramMaterializer::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a resolved Triton load binding");
  FailureOr<std::string> physicalFill =
      transferPhysicalExtentFill(operation);
  FailureOr<bool> wholeView = target::isWholeViewAccess(operation);
  if (failed(physicalFill) || failed(wholeView))
    return failure();
  if (*wholeView && boundary.getDomainNodes().empty() &&
      boundary.getLoadFill() == "none" && physicalFill->empty()) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view))
      return failure();
    bindResult(operation, 0, (*view)->pointer);
    return success();
  }
  if (boundary.getDefer() && !activeDeferredContract) {
    deferredLoads[operation.getResult(0)] = &operation;
    return success();
  }
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view))
    return failure();
  bool descriptorCandidate =
      planIndex.transferForms.lookup(*node) == "pointer_or_descriptor";
  bool linearDescriptor =
      descriptorCandidate && planIndex.descriptorLayouts.lookup(*node) == "linear";
  FailureOr<std::string> descriptorOffset =
      descriptorCandidate ? descriptorOffsets(operation, **view)
                          : FailureOr<std::string>(std::string());
  FailureOr<std::string> descriptorShape =
      descriptorCandidate && !linearDescriptor
          ? emitTensorShape(operation, 0)
          : FailureOr<std::string>(std::string());
  FailureOr<std::string> pointers =
      emitPointerExpression(operation, **view, false);
  FailureOr<std::string> mask = emitMaskExpression(operation, false);
  StringRef loadFill = boundary.getLoadFill();
  if (loadFill == "none" && !physicalFill->empty())
    loadFill = *physicalFill;
  if (loadFill == "none" &&
      target::lowering::hasPackedScalarDomain(planIndex, boundary))
    loadFill = "zero";
  bool materializeValidity =
      (!boundary.getConsumerNeutralized() ||
       (descriptorCandidate && !linearDescriptor)) &&
      loadFill != "none" && !boundary.getValidityDomainNodes().empty();
  FailureOr<std::string> validity =
      materializeValidity
          ? emitValidityExpression(boundary.getValidityTensorAxes(),
                                   boundary.getValidityDomainNodes(),
                                   operation.getResult(0), operation)
          : FailureOr<std::string>(std::string("True"));
  if (failed(descriptorOffset) || failed(descriptorShape) || failed(pointers) ||
      failed(mask) || failed(validity) || failed(physicalFill))
    return failure();
  std::string result = makeResultName(operation, 0);
  StringRef loadFillExpression = loadFill == "negative_infinity"
                                     ? "-float('inf')"
                                     : loadFill == "positive_infinity"
                                           ? "float('inf')"
                                           : "0.0";
  if (*validity != "True" && *validity != *mask)
    *mask = *mask == "True" ? *validity
                            : "(" + *mask + ") & (" + *validity + ")";
  auto emitPointerLoad = [&]() {
    if (loadFill == "none") {
      line(result + " = tl.load(" + *pointers + ")");
    } else {
      line(result + " = tl.load(" + *pointers + ", mask=" + *mask +
           ", other=" + loadFillExpression.str() + ")");
    }
  };
  if (descriptorCandidate) {
    line("if USE_TMA:");
    ++indentation;
    if (linearDescriptor)
      line(result + " = " + descriptorName(operation) + ".load([" +
           *descriptorOffset + "])");
    else
      line(result + " = tl.reshape(" + descriptorName(operation) + ".load([" +
           *descriptorOffset + "]), " + *descriptorShape + ")");
    if (!linearDescriptor && *validity != "True")
      line(result + " = tl.where(" + *validity + ", " + result + ", " +
           loadFillExpression.str() + ")");
    --indentation;
    line("else:");
    ++indentation;
    emitPointerLoad();
    --indentation;
  } else {
    emitPointerLoad();
  }
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), result, operation);
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
      !result || !isa<IntegerType, IndexType>(result.getElementType()))
    return operation.emitOpError("lacks a mechanical Triton indices binding");
  auto mode = operation.getAttrOfType<StringAttr>("intent.mode");
  auto tensorAxis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  FailureOr<plan::AxisOp> axis = failure();
  unsigned emittedAxis = 0;
  if (mode && mode.getValue() == "tensor_axis") {
    if (!tensorAxis || tensorAxis.getInt() < 0 || !result ||
        tensorAxis.getInt() >= result.getRank() ||
        static_cast<size_t>(tensorAxis.getInt()) >= binding.getAxisNodes().size())
      return operation.emitOpError("has no tensor-axis indices binding");
    FailureOr<std::optional<target::lowering::ResultAxisRegionRangeBinding>>
        selected = target::lowering::selectedResultAxisRegionRange(
            planIndex, kernel, operation.getOperand(0), tensorAxis.getInt(),
            operation);
    if (failed(selected))
      return failure();
    if (*selected) {
      auto projected = valueNames.find((*selected)->argument);
      if (projected == valueNames.end())
        return operation.emitOpError(
            "has no active region projection for tensor-axis indices");
      std::string expression =
          (*selected)->selected.axis.isScalar()
              ? projected->second
              : broadcastIndex(projected->second, tensorAxis.getInt(),
                               result.getRank());
      FailureOr<std::string> padded =
          padExpression(operation.getResult(0), expression, operation);
      if (failed(padded))
        return failure();
      std::string emitted = makeResultName(operation, 0);
      line(emitted + " = " + *padded);
      bindResult(operation, 0, emitted);
      return success();
    }
    axis = planIndex.axes.lookup(binding.getAxisNodes()[tensorAxis.getInt()]);
    emittedAxis = tensorAxis.getInt();
  } else if (operation.getNumOperands() == 1) {
    Value indexed = operation.getOperand(0);
    FailureOr<std::optional<target::lowering::RegionRangeBinding>> selected =
        target::lowering::selectedRegionValueRange(planIndex, kernel, indexed,
                                                   operation);
    if (failed(selected))
      return failure();
    if (*selected) {
      FailureOr<StringRef> argumentProjection =
          isa<BlockArgument>(indexed) ? lookupValue(operation, 0)
                                      : FailureOr<StringRef>(failure());
      std::string projected = isa<BlockArgument>(indexed)
                                  ? succeeded(argumentProjection)
                                        ? argumentProjection->str()
                                        : std::string()
                                  : selectedRegionIndices.lookup(indexed);
      if (projected.empty() && isa<intent::DomainType>(indexed.getType()))
        projected = axisIndices.lookup((*selected)->axis.getNode());
      if (projected.empty())
        return operation.emitOpError(
                   "has no active projection for its selected region value on axis ")
               << (*selected)->axis.getNode() << " with roles "
               << (*selected)->axis.getRoles();
      if (isa<BlockArgument>(indexed) && failed(argumentProjection))
        return failure();
      FailureOr<std::string> padded =
          padExpression(operation.getResult(0), projected, operation);
      if (failed(padded))
        return failure();
      std::string emitted = makeResultName(operation, 0);
      line(emitted + " = " + *padded);
      bindResult(operation, 0, emitted);
      return success();
    }
    axis = resolveAxis(indexed, operation);
  }
  if (failed(axis) ||
      ((!mode || mode.getValue() != "tensor_axis") && result.getRank() != 1) ||
      result.getRank() <= 0)
    return operation.emitOpError("lacks a mechanical Triton indices binding");
  if (axisIndices.lookup(axis->getNode()).empty() && axis->hasRole("lane"))
    axisIndices[axis->getNode()] =
        "tl.arange(0, " + axis->getTile().str() + ")";
  FailureOr<std::string> expression =
      indexExpression(*axis, false, emittedAxis, result.getRank(), operation);
  if (failed(expression))
    return failure();
  FailureOr<std::string> padded =
      padExpression(operation.getResult(0), *expression, operation);
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

LogicalResult ProgramMaterializer::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  if (failed(node) || !binding)
    return operation.emitOpError("lacks a Triton reduction binding");
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  if (!components || components.getInt() <= 0 ||
      operation.getNumResults() != static_cast<unsigned>(components.getInt()))
    return operation.emitOpError("has no canonical reduction component schema");
  SmallVector<std::string> operands;
  for (unsigned component = 0;
       component < static_cast<unsigned>(components.getInt()); ++component) {
    FailureOr<StringRef> operand = lookupValue(operation, component);
    if (failed(operand))
      return failure();
    operands.push_back(operand->str());
  }
  std::string axis = binding.getAllAxes()
                         ? std::string("None")
                         : std::to_string(binding.getAxis());
  if (binding.getLowering() == "tl.max_with_index") {
    FailureOr<StringRef> indexIdentity = lookupValue(operation, 3);
    if (operation.getNumResults() != 2 || operands.size() != 2 ||
        failed(indexIdentity))
      return operation.emitOpError(
          "Triton arg-reduction requires value and index results");
    std::string value = makeResultName(operation, 0);
    std::string index = makeResultName(operation, 1);
    std::string valueKeepDims = value + "_keep_dims";
    std::string candidates = index + "_candidates";
    StringRef indexDtype = tritonDtype(operation.getResult(1).getType());
    if (indexDtype.empty())
      return operation.emitOpError("has an unsupported arg-reduction index type");
    line(value + " = tl.max(" + operands.front() + ", axis=" + axis + ")");
    line(valueKeepDims + " = tl.max(" + operands.front() + ", axis=" + axis +
         ", keep_dims=True)");
    line(candidates + " = tl.where(" + operands.front() + " == " + valueKeepDims +
         ", " + operands[1] + ", " + indexIdentity->str() + ")");
    line(index + " = tl.cast(tl.min(" + candidates + ", axis=" + axis + "), " +
         indexDtype.str() + ")");
    bindResult(operation, 0, value);
    bindResult(operation, 1, index);
    return success();
  }
  if (binding.getLowering() == "tl.reduce") {
    FailureOr<target::lowering::CombinerUse> combiner =
        target::lowering::resolveCombiner(operation);
    if (failed(combiner))
      return failure();
    SmallVector<std::string> results;
    for (unsigned component = 0; component < operation.getNumResults(); ++component)
      results.push_back(makeResultName(operation, component));
    SmallVector<std::string> projectedInputs = operands;
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
        StringRef dtype = tritonDtype(operation.getOperand(operandIndex).getType());
        if (failed(value) || dtype.empty())
          return operation.emitOpError(
              "has no Triton scalar capture projection");
        projectedInputs.push_back("tl.full(" + operands.front() + ".shape, " +
                                  value->str() + ", " + dtype.str() + ")");
        projectedResults.push_back(makeResultName(operation, 0) + "_capture_" +
                                   std::to_string(capture));
      }
      projectedInputs.push_back("tl.full(" + operands.front() +
                                ".shape, True, tl.int1)");
      projectedResults.push_back(makeResultName(operation, 0) + "_capture_valid");
    }
    std::string input = projectedInputs.size() == 1
                            ? projectedInputs.front()
                            : "(" + llvm::join(projectedInputs, ", ") + ")";
    line(llvm::join(projectedResults, ", ") + " = tl.reduce(" + input +
         ", axis=" + axis + ", combine_fn=" + function + ")");
    for (auto [index, result] : llvm::enumerate(results))
      bindResult(operation, index, result);
    return success();
  }
  if (operation.getNumResults() != operands.size())
    return operation.emitOpError(
        "built-in Triton reduction components do not match results");
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    std::string result = makeResultName(operation, component);
    if (binding.getLowering() == "tl.reduce_all")
      line(result + " = ~tl.reduce_or(~(" + operands[component] + "), axis=" +
           axis + ")");
    else
      line(result + " = " + binding.getLowering().str() + "(" + operands[component] +
           ", axis=" + axis + ")");
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
        (binding.getLowering() != "tl.cumsum" &&
         binding.getLowering() != "tl.associative_scan"))
      return operation.emitOpError("lacks a fragment Triton scan binding");
    SmallVector<std::string> operands;
    for (unsigned component = 0;
         component < static_cast<unsigned>(components.getInt()); ++component) {
      FailureOr<StringRef> operand = lookupValue(operation, component);
      if (failed(operand))
        return failure();
      operands.push_back(operand->str());
    }
    if (binding.getLowering() == "tl.associative_scan") {
      FailureOr<target::lowering::CombinerUse> combiner =
          target::lowering::resolveCombiner(operation);
      if (failed(combiner))
        return failure();
      SmallVector<std::string> results;
      for (unsigned component = 0; component < operation.getNumResults(); ++component)
        results.push_back(makeResultName(operation, component));
      SmallVector<std::string> projectedInputs = operands;
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
          StringRef dtype = tritonDtype(operation.getOperand(operandIndex).getType());
          if (failed(value) || dtype.empty())
            return operation.emitOpError(
                "has no Triton scalar capture projection");
          projectedInputs.push_back("tl.full(" + operands.front() + ".shape, " +
                                    value->str() + ", " + dtype.str() + ")");
          projectedResults.push_back(makeResultName(operation, 0) + "_capture_" +
                                     std::to_string(capture));
        }
        projectedInputs.push_back("tl.full(" + operands.front() +
                                  ".shape, True, tl.int1)");
        projectedResults.push_back(makeResultName(operation, 0) + "_capture_valid");
      }
      std::string input = projectedInputs.size() == 1
                              ? projectedInputs.front()
                              : "(" + llvm::join(projectedInputs, ", ") + ")";
      line(llvm::join(projectedResults, ", ") + " = tl.associative_scan(" + input +
           ", axis=" + std::to_string(binding.getAxis()) + ", combine_fn=" +
           function + ")");
      for (auto [index, result] : llvm::enumerate(results))
        bindResult(operation, index, result);
      return success();
    }
    if (operation.getNumResults() != operands.size())
      return operation.emitOpError(
          "built-in Triton scan components do not match results");
    for (unsigned component = 0; component < operation.getNumResults(); ++component) {
      std::string result = makeResultName(operation, component);
      line(result + " = tl.cumsum(" + operands[component] + ", axis=" +
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
  bool generic = binding && binding.getLowering() == "tl.associative_scan";
  if (failed(node) || !binding ||
      (binding.getLowering() != "tl.cumsum" && !generic) || !axis || !range ||
      extent.empty() || operation.getNumResults() == 0)
    return operation.emitOpError("lacks a mechanical Triton scan binding");
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
  SmallVector<std::string> carries;
  SmallVector<Type> carryTypes;
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    FailureOr<StringRef> identity =
        lookupValue(operation, operation.getNumResults() + component);
    if (failed(identity))
      return failure();
    identities.push_back(identity->str());
    carries.push_back(makeResultName(operation, component) + "_carry");
    carryTypes.push_back(operation.getResult(component).getType());
    line(carries.back() + " = " + identities.back());
  }
  if (generic && combiner->captureCount != 0) {
    for (unsigned capture = 0; capture < combiner->captureCount; ++capture) {
      carries.push_back(makeResultName(operation, 0) + "_capture_" +
                        std::to_string(capture) + "_carry");
      carryTypes.push_back(
          operation.getOperand(2 * combiner->componentCount + capture).getType());
      line(carries.back() + " = 0");
    }
    carries.push_back(makeResultName(operation, 0) + "_capture_valid_carry");
    carryTypes.push_back(IntegerType::get(operation.getContext(), 1));
    line(carries.back() + " = False");
  }
  std::string stem = makeResultName(operation, 0);
  std::string block = stem + "_block";
  std::string offsets = stem + "_offsets";
  line("for " + block + " in tl.range(0, tl.cdiv(" + extent + ", " +
       range->getTile().str() + "), flatten=True):");
  ++indentation;
  line(offsets + " = " + block + " * " + range->getTile().str() +
       " + tl.arange(0, " + range->getTile().str() + ")");
  if (failed(replayScanProducers(binding, offsets)))
    return failure();
  SmallVector<std::string> blockInputs;
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    FailureOr<StringRef> operand = lookupValue(operation, component);
    if (failed(operand))
      return failure();
    blockInputs.push_back("tl.where(" + offsets + " < " + extent + ", " +
                          operand->str() + ", " + identities[component] + ")");
  }
  for (int64_t valueID : binding.getMaterializedValues()) {
    Value value = kernel.values.lookup(valueID);
    auto name = valueNames.find(value);
    FailureOr<std::string> pointer =
        scanMaterializedPointer(value, offsets, operation);
    if (name == valueNames.end() || failed(pointer))
      return binding.emitOpError(
          "has no emitted Triton value for scan materialization");
    line("tl.store(" + *pointer + ", " + name->second + ", mask=" + offsets +
         " < " + extent + ")");
  }
  SmallVector<std::string> localInputs = blockInputs;
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
      StringRef dtype = tritonDtype(operation.getOperand(operandIndex).getType());
      if (failed(captureValue) || dtype.empty())
        return operation.emitOpError(
            "has no Triton workspace-scan capture projection");
      localInputs.push_back("tl.full(" + blockInputs.front() + ".shape, " +
                            captureValue->str() + ", " + dtype.str() + ")");
      localResults.push_back(stem + "_capture_" + std::to_string(capture) +
                             "_local_scan");
      globalResults.push_back(stem + "_capture_" + std::to_string(capture) +
                              "_global_scan");
    }
    localInputs.push_back("tl.full(" + blockInputs.front() +
                          ".shape, True, tl.int1)");
    localResults.push_back(stem + "_capture_valid_local_scan");
    globalResults.push_back(stem + "_capture_valid_global_scan");
  }
  if (generic) {
    std::string input = localInputs.size() == 1
                            ? localInputs.front()
                            : "(" + llvm::join(localInputs, ", ") + ")";
    line(llvm::join(localResults, ", ") + " = tl.associative_scan(" + input +
         ", axis=" + std::to_string(binding.getAxis()) + ", combine_fn=" +
         function + ")");
    SmallVector<std::string> arguments = carries;
    arguments.append(localResults);
    line(llvm::join(globalResults, ", ") + " = " + function + "(" +
         llvm::join(arguments, ", ") + ")");
  } else {
    for (unsigned component = 0; component < operation.getNumResults(); ++component) {
      line(localResults[component] + " = tl.cumsum(" + blockInputs[component] +
           ", axis=" + std::to_string(binding.getAxis()) + ")");
      line(globalResults[component] + " = " + localResults[component] + " + " +
           carries[component]);
    }
  }
  for (unsigned component = 0; component < operation.getNumResults(); ++component) {
    FailureOr<std::string> pointer = scanWorkspacePointer(
        operation.getResult(component), binding, offsets, operation);
    if (failed(pointer))
      return failure();
    line("tl.store(" + *pointer + ", " + globalResults[component] + ", mask=" +
         offsets + " < " + extent + ")");
  }
  std::string lastLane = "tl.minimum(" + range->getTile().str() +
                         " - 1, " + extent + " - " + block + " * " +
                         range->getTile().str() + " - 1)";
  for (unsigned component = 0; component < carries.size(); ++component) {
    StringRef dtype = tritonDtype(carryTypes[component]);
    if (dtype.empty())
      return operation.emitOpError("has no Triton scan carry dtype");
    line(carries[component] + " = tl.cast(tl.sum(tl.where(tl.arange(0, " +
         range->getTile().str() + ") == " + lastLane + ", " +
         globalResults[component] + ", 0), axis=0), " + dtype.str() + ")");
  }
  --indentation;
  for (Value result : operation.getResults())
    valueNames.erase(result);
  return success();
}

LogicalResult ProgramMaterializer::replayScanProducers(const plan::ScanOp &binding,
                                                 StringRef offsets) {
  if (!operationRegistry())
    return binding.emitOpError("has no Triton operation registry for scan replay");
  auto oldAxis = axisIndices.find(binding.getAxisNode());
  std::optional<std::string> savedAxis =
      oldAxis == axisIndices.end() ? std::nullopt
                                   : std::optional<std::string>(oldAxis->second);
  axisIndices[binding.getAxisNode()] = offsets.str();
  llvm::StringMap<std::optional<std::string>> savedTiles;
  for (int64_t node : binding.getProducers()) {
    Operation *producer = kernel.nodes.lookup(node);
    if (!producer)
      return binding.emitOpError("references an unknown Triton scan producer");
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
        regionTiles[label.getValue()] =
            planIndex.axes.lookup(binding.getAxisNode())
                .getRange("traversal", 0)
                ->getTile()
                .str();
      }
    }
  }
  activeScanReplay = binding.getNode();
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
    if (!producer ||
        failed(operationRegistry()->dispatch(
            *producer, "Triton deferred contraction producer replay")))
      return failure();
  return success();
}

LogicalResult ProgramMaterializer::replayBlock(Block &block) {
  if (!operationRegistry())
    return block.getParentOp()->emitOpError(
        "has no Triton operation registry for provider-form replay");
  for (Operation &operation : block) {
    const target::OperationHandler *handler = operationRegistry()->lookup(
        target::semanticOperationName(operation));
    if (!handler)
      return operation.emitOpError(
          "has no Triton handler during provider-form replay");
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
  if (failed(node) || !binding || binding.getLowering() != "alias")
    return operation.emitOpError("lacks a Triton broadcast binding");
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(operand))
    return failure();
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
  std::string expression = binding.getLowering() == "python_negate"
                               ? "-(" + operand->str() + ")"
                           : binding.getLowering() == "python_not"
                               ? "(" + operand->str() + ") == False"
                               : binding.getLowering().str() + "(" +
                                     operand->str() + ")";
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
  if (binding.getLowering() == "contract_accumulator_alias") {
    bool lhsContract = operation.getOperand(0).getDefiningOp() &&
                       target::semanticOperationName(
                           *operation.getOperand(0).getDefiningOp()) ==
                           "intent.contract";
    bool rhsContract = operation.getOperand(1).getDefiningOp() &&
                       target::semanticOperationName(
                           *operation.getOperand(1).getDefiningOp()) ==
                           "intent.contract";
    if (lhsContract == rhsContract)
      return operation.emitOpError(
          "does not have one selected Triton fused contraction result");
    bindResult(operation, 0, lhsContract ? lhs->str() : rhs->str());
    return success();
  }
  std::string result = makeResultName(operation, 0);
  std::string expression;
  if (binding.getLowering() == "tl.maximum" ||
      binding.getLowering() == "tl.minimum")
    expression = binding.getLowering().str() + "(" + lhs->str() + ", " +
                 rhs->str() + ")";
  else if (binding.getLowering() == "libdevice.pow") {
    auto elementType = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        return tensor.getElementType();
      return type;
    };
    Type lhsType = elementType(operation.getOperand(0).getType());
    Type rhsType = elementType(operation.getOperand(1).getType());
    Type resultType = operation.getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(resultType))
      resultType = tensor.getElementType();
    StringRef resultDtype = tritonDtype(resultType);
    if (resultDtype.empty())
      return operation.emitOpError("uses an unsupported Triton power type");
    if (lhsType.isF32() && rhsType.isF32() && resultType.isF32()) {
      expression =
          "libdevice.pow(" + lhs->str() + ", " + rhs->str() + ")";
    } else {
      std::string widened = "libdevice.pow(tl.cast(" + lhs->str() +
                            ", tl.float32), tl.cast(" + rhs->str() +
                            ", tl.float32))";
      expression = resultType.isF32()
                       ? std::move(widened)
                       : "tl.cast(" + widened + ", " + resultDtype.str() + ")";
    }
  } else if ((binding.getLowering() == "python_floor_divide" ||
            binding.getLowering() == "python_remainder") &&
           binding.getNonnegativeOperands()) {
    StringRef symbol = binding.getLowering() == "python_floor_divide" ? "//" : "%";
    expression = "(" + lhs->str() + ") " + symbol.str() + " (" + rhs->str() + ")";
  } else if (binding.getLowering() == "python_floor_divide" ||
           binding.getLowering() == "python_remainder") {
    Type elementType = operation.getResult(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(elementType))
      elementType = tensor.getElementType();
    StringRef resultDtype = tritonDtype(elementType);
    if (resultDtype.empty())
      return operation.emitOpError(
          "requires an integer result for Python division semantics");
    std::string wideLhs = result + "_wide_lhs";
    std::string wideRhs = result + "_wide_rhs";
    std::string lhsMagnitude = result + "_lhs_magnitude";
    std::string rhsMagnitude = result + "_rhs_magnitude";
    std::string quotientMagnitude = result + "_quotient_magnitude";
    std::string quotient = result + "_truncating_quotient";
    std::string remainder = result + "_remainder";
    std::string adjust = result + "_adjust";
    line(wideLhs + " = tl.cast(" + lhs->str() + ", tl.int64)");
    line(wideRhs + " = tl.cast(" + rhs->str() + ", tl.int64)");
    line(lhsMagnitude + " = tl.where(" + wideLhs + " < 0, -" + wideLhs +
         ", " + wideLhs + ")");
    line(rhsMagnitude + " = tl.where(" + wideRhs + " < 0, -" + wideRhs +
         ", " + wideRhs + ")");
    line(quotientMagnitude + " = " + lhsMagnitude + " // " + rhsMagnitude);
    line(quotient + " = tl.where((" + wideLhs + " < 0) != (" + wideRhs +
         " < 0), -" + quotientMagnitude + ", " + quotientMagnitude + ")");
    line(remainder + " = " + wideLhs + " - " + quotient + " * " + wideRhs);
    line(adjust + " = (" + remainder + " != 0) & ((" + remainder +
         " < 0) != (" + wideRhs + " < 0))");
    expression = binding.getLowering() == "python_floor_divide"
                     ? "tl.cast(" + quotient + " - tl.where(" + adjust +
                           ", 1, 0), " + resultDtype.str() + ")"
                     : "tl.cast(" + remainder + " + tl.where(" + adjust +
                           ", " + wideRhs + ", 0), " + resultDtype.str() + ")";
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
    else if (binding.getLowering() == "python_logical_and")
      symbol = "&";
    else if (binding.getLowering() == "python_logical_or")
      symbol = "|";
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
  if (failed(node) || !binding || binding.getLowering() != "tl.where" ||
      operation.getNumResults() != 1 || failed(condition) || failed(trueValue) ||
      failed(falseValue))
    return operation.emitOpError("lacks a mechanical Triton conditional binding");
  std::string result = makeResultName(operation, 0);
  std::string expression = "tl.where(" + condition->str() + ", " +
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
  if (failed(node) || !binding || binding.getLowering() != "tl.cast" ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical Triton cast binding");
  StringRef targetType = tritonDtype(resultType);
  if (targetType.empty())
    return operation.emitOpError("casts to an unsupported Triton type");
  Type operandType = operation.getOperand(0).getType();
  if (auto tensor = dyn_cast<RankedTensorType>(operandType))
    operandType = tensor.getElementType();
  Type resultElementType = resultType;
  if (auto tensor = dyn_cast<RankedTensorType>(resultElementType))
    resultElementType = tensor.getElementType();
  bool decodeE8M0 = isa<Float8E8M0FNUType>(operandType) &&
                    !isa<Float8E8M0FNUType>(resultElementType);
  std::string result = makeResultName(operation, 0);
  std::string expression = syntax::cast(*operand, targetType, decodeE8M0,
                                        resultElementType.isF32());
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
  if (failed(node) || !binding || binding.getLowering() != "tl.bitcast" ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical Triton bitcast binding");
  StringRef targetType = tritonDtype(resultType);
  if (targetType.empty())
    return operation.emitOpError("bitcasts to an unsupported Triton type");
  std::string result = makeResultName(operation, 0);
  std::string expression = syntax::bitcast(*operand, targetType);
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
  if (failed(node) || !binding || binding.getLowering() != "tl.reshape" ||
      failed(operand) || failed(shape) || operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a mechanical Triton reshape binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.reshape(" + operand->str() + ", " + *shape + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitJoin(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "join emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> lhs = lookupValue(operation, 0);
  FailureOr<StringRef> rhs = lookupValue(operation, 1);
  if (failed(node) || !binding || binding.getLowering() != "tl.join" ||
      failed(lhs) || failed(rhs) || operation.getNumResults() != 1 ||
      !isa<RankedTensorType>(operation.getResult(0).getType()))
    return operation.emitOpError("lacks a mechanical Triton join binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.join(" + lhs->str() + ", " + rhs->str() + ")");
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
  if (failed(node) || !binding || binding.getLowering() != "tl.permute" ||
      failed(operand) || failed(permutation))
    return operation.emitOpError("lacks a mechanical Triton transpose binding");
  std::string result = makeResultName(operation, 0);
  std::string expression = "tl.permute(" + operand->str();
  for (int64_t axis : *permutation)
    expression += ", " + std::to_string(axis);
  line(result + " = " + expression + ")");
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
  if (failed(node) || !binding || binding.getLowering() != "tl.full" ||
      failed(fill) || !resultType || failed(shape))
    return operation.emitOpError("lacks a mechanical Triton full binding");
  StringRef dtype = tritonDtype(resultType);
  if (dtype.empty())
    return operation.emitOpError("uses an unsupported Triton full dtype");
  std::string result = makeResultName(operation, 0);
  std::string expression = "tl.full(" + *shape + ", " + fill->str() +
                           ", dtype=" + dtype.str() + ")";
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
          "ragged-start Triton gather has no planned ragged axis");
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
      FailureOr<std::string> pointer = scanWorkspacePointer(
          operation.getOperand(0), scanSource->second, index->str(), operation);
      if (failed(pointer))
        return failure();
      line(result + " = tl.where(" + valid->str() + ", tl.load(" + *pointer +
           "), " + fill->str() + ")");
    } else if (materializedSource != scanMaterializedValues.end()) {
      FailureOr<std::string> pointer = scanMaterializedPointer(
          operation.getOperand(0), index->str(), operation);
      if (failed(pointer))
        return failure();
      line(result + " = tl.where(" + valid->str() + ", tl.load(" + *pointer +
           "), " + fill->str() + ")");
    } else {
      std::string gathered =
          "tl.gather(" + source->str() + ", tl.full(" + source->str() +
          ".shape, " + index->str() + ", tl.int64), axis=0)";
      line(result + " = tl.where(" + valid->str() + ", tl.max(" + gathered +
           ", axis=0), " + fill->str() + ")");
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
    line(result + " = tl.sum(tl.gather(" + source->str() +
         ", tl.full((1,), 0, tl.int32), axis=0), axis=0)");
    bindResult(operation, 0, result);
    return success();
  }
  if (form == "static_projection" && succeeded(relation)) {
    FailureOr<StringRef> source = lookupValue(operation, 0);
    auto sourceType = operation.getNumOperands() > 0
                          ? dyn_cast<RankedTensorType>(
                                operation.getOperand(0).getType())
                          : RankedTensorType();
    bool supported = succeeded(source) &&
                     target::isStaticFragmentProjection(operation, *relation) &&
                     sourceType && sourceType.getRank() > 0 &&
                     !sourceType.isDynamicDim(sourceType.getRank() - 1) &&
                     sourceType.getShape().back() == 2;
    std::optional<int64_t> component;
    for (auto [axis, term] : llvm::enumerate(*relation)) {
      if (term.kind == "full_slice")
        continue;
      if (term.kind != "static_index" || term.staticValues.size() != 1 ||
          !term.staticValues.front() ||
          !sourceType || axis + 1 != static_cast<unsigned>(sourceType.getRank()) ||
          (*term.staticValues.front() != 0 && *term.staticValues.front() != 1)) {
        supported = false;
        continue;
      }
      component = *term.staticValues.front();
    }
    if (!supported || !component)
      return operation.emitOpError(
          "has no mechanical Triton static fragment projection");
    std::string result = makeResultName(operation, 0);
    line(result + " = tl.split(" + source->str() + ")[" +
         std::to_string(*component) + "]");
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
    return operation.emitOpError("lacks a mechanical Triton gather binding");
  if (form == "view_indirect" &&
      isa<intent::ViewType>(operation.getOperand(0).getType())) {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    FailureOr<std::string> pointer =
        succeeded(view) ? emitPointerExpression(operation, **view, false)
                        : FailureOr<std::string>(failure());
    FailureOr<std::string> mask = emitMaskExpression(operation, false);
    if (failed(view) || failed(pointer) || failed(mask))
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = tl.load(" + *pointer + ", mask=(" + *mask + ") & (" +
         valid->str() + "), other=" + fill->str() + ")");
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
    return operation.emitOpError("lacks a mechanical Triton gather binding");
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
    line(result + " = tl.where(" + valid->str() + ", " + expanded + ", " +
         fill->str() + ")");
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitMembers(Operation &operation) {
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
          "has no exact Triton region projection for ragged members");
    position = projected->str();
  }
  if (failed(relation) || runtime == raggedRuntimeByRelation.end() ||
      position.empty())
    return operation.emitOpError("has no ragged runtime position");
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

LogicalResult ProgramMaterializer::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical Triton stream binding");
  Block &body = operation.getRegion(0).front();
  bool hasStop = static_cast<bool>(binding.getStopValueAttr());
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
    line("sequence_begin_" + raggedSuffix + " = tl.load(" +
         ragged.offsets->pointer + " + " + addressIndex(outer) + " * " +
         addressIndex(ragged.offsets->strides[0]) + ")");
    line("sequence_end_" + raggedSuffix + " = tl.load(" +
         ragged.offsets->pointer + " + " + addressIndex(outer + " + 1") +
         " * " + addressIndex(ragged.offsets->strides[0]) + ")");
    line("sequence_length_" + raggedSuffix + " = sequence_end_" +
         raggedSuffix + " - sequence_begin_" + raggedSuffix);
  }
  std::string partitionBegin;
  std::string partitionEnd;
  std::string activeEnd;
  std::string streamExtent =
      raggedStream ? "sequence_length_" + raggedSuffix
                   : binding.getExtent().str();
  if (partitionedStream) {
    plan::PartitionBindingOp partition = binding.getPartition();
    plan::AxisOp axis = planIndex.axes.lookup(partition.getAxisNode());
    const target::lowering::RangeBinding *ownership =
        axis ? axis.getRange("ownership", 0) : nullptr;
    std::string part = programBlocks.lookup(partition.getAxisNode());
    std::string logical = axisDimensions.lookup(partition.getAxisNode());
    if (!axis || !ownership || part.empty() || logical.empty())
      return binding.emitOpError(
          "has no exact Triton count-partition stream interval");
    std::string suffix = std::to_string(*node);
    partitionBegin = "stream_segment_begin_" + suffix;
    partitionEnd = "stream_segment_end_" + suffix;
    line(partitionBegin + " = tl.minimum(" + addressIndex(part) + " * " +
         ownership->getTile().str() + ", " + logical + ")");
    line(partitionEnd + " = tl.minimum(" + partitionBegin + " + " +
         ownership->getTile().str() + ", " + logical + ")");
    streamExtent = "tl.maximum(0, " + partitionEnd + " - " +
                   partitionBegin + ")";
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
      line(activeEnd + " = tl.minimum(" + stop->str() + ", " +
           partitionEnd + ")");
      streamExtent = "tl.maximum(0, " + activeEnd + " - " +
                     partitionBegin + ")";
    } else {
      streamExtent = "tl.maximum(0, tl.minimum(" + stop->str() + ", " +
                     streamExtent + "))";
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
  std::string block = "stream_block_" + std::to_string(*node);
  std::string offsets = "stream_axis_index_" + std::to_string(*node);
  auto streamForm = binding.physical
                        ? binding.physical->getAttrOfType<StringAttr>(streamFormAttr)
                        : StringAttr();
  bool prefixBoundary = streamForm && streamForm.getValue() == "prefix_boundary";
  std::string streamBase = partitionedStream
                               ? partitionBegin
                           : raggedStream
                               ? "sequence_begin_" + raggedSuffix
                               : "0";
  if (prefixBoundary) {
    auto boundaryAxis =
        binding.physical->getAttrOfType<IntegerAttr>(streamBoundaryAxisAttr);
    auto neutralMasks =
        binding.physical->getAttrOfType<DenseI64ArrayAttr>(streamNeutralMasksAttr);
    std::string boundaryStart =
        boundaryAxis ? axisStarts.lookup(boundaryAxis.getInt()) : std::string();
    if (!boundaryAxis || !neutralMasks || neutralMasks.empty() ||
        boundaryStart.empty())
      return binding.emitOpError(
          "has no complete Triton prefix-boundary stream form");
    std::string prefixBlocks =
        "stream_prefix_blocks_" + std::to_string(*node);
    line(prefixBlocks + " = tl.minimum(tl.cdiv(" + streamExtent + ", " +
         binding.getTile().str() + "), tl.maximum(0, (" + boundaryStart +
         " - " + streamBase + ") // " + binding.getTile().str() + "))");
    line("for " + block + " in range(0, " + prefixBlocks + "):");
    PrefixBoundaryStream replay{block, offsets, streamExtent, prefixBlocks,
                                streamBase, {}};
    for (int64_t mask : neutralMasks.asArrayRef()) {
      replay.neutralMasks.push_back(mask);
      activeNeutralMasks.insert(mask);
    }
    prefixBoundaryStreams[&operation] = std::move(replay);
  } else {
    line("for " + block + " in range(0, tl.cdiv(" + streamExtent + ", " +
         binding.getTile().str() + ")):");
  }
  ++indentation;
  line(offsets + " = " +
       std::string(partitionedStream
                       ? partitionBegin + " + "
                       : raggedStream ? "sequence_begin_" + raggedSuffix + " + "
                                      : "") +
       addressIndex(block) + " * " + binding.getTile().str() + " + " +
       addressIndex("tl.arange(0, " + binding.getTile().str() + ")"));
  valueNames[body.getArgument(0)] = offsets;
  std::string streamStart =
      std::string(partitionedStream
                      ? partitionBegin + " + "
                      : raggedStream ? "sequence_begin_" + raggedSuffix + " + "
                                     : "") +
      addressIndex(block) + " * " + binding.getTile().str();
  selectedRegionStarts[body.getArgument(0)] = streamStart;
  auto bindScopedAxis = [&](int64_t axisNode, std::string value) {
    auto previous = axisIndices.find(axisNode);
    std::optional<std::string> restore;
    if (previous != axisIndices.end())
      restore = previous->second;
    streamAxisRestores[&operation].emplace_back(axisNode, std::move(restore));
    axisIndices[axisNode] = std::move(value);
  };
  auto bindScopedStart = [&](int64_t axisNode, std::string value) {
    auto previous = axisStarts.find(axisNode);
    std::optional<std::string> restore;
    if (previous != axisStarts.end())
      restore = previous->second;
    streamStartRestores[&operation].emplace_back(axisNode, std::move(restore));
    axisStarts[axisNode] = std::move(value);
  };
  if (axisIndices.lookup(binding.getAxisNode()).empty())
    bindScopedAxis(binding.getAxisNode(), offsets);
  bindScopedStart(binding.getAxisNode(), streamStart);
  for (int64_t axisNode : binding.getInnerReductionAxes()) {
    if (axisNode == binding.getAxisNode())
      continue;
    plan::AxisOp axis = planIndex.axes.lookup(axisNode);
    const target::lowering::RangeBinding *range =
        axis ? axis.getRange("reduction", 0) : nullptr;
    if (!range)
      return binding.emitOpError("has no inner reduction range");
    bindScopedAxis(
        axisNode,
        addressIndex("tl.arange(0, " + range->getTile().str() + ")"));
    bindScopedStart(axisNode, "0");
  }
  return success();
}

LogicalResult ProgramMaterializer::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (carriers == streamCarriers.end() || failed(node) || !binding)
    return operation.emitOpError("has no active Triton stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (::intent::target::semanticOperationName(terminator) != "intent.yield" ||
      terminator.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("does not yield every Triton stream state");
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
         prefix->second.prefixBlocks + ", tl.cdiv(" + prefix->second.extent +
         ", " + binding.getTile().str() + ")):");
    ++indentation;
    line(prefix->second.offsets + " = " + prefix->second.base + " + " +
         addressIndex(prefix->second.block) + " * " + binding.getTile().str() +
         " + " +
         addressIndex("tl.arange(0, " + binding.getTile().str() + ")"));
    Block &body = operation.getRegion(0).front();
    valueNames[body.getArgument(0)] = prefix->second.offsets;
    selectedRegionStarts[body.getArgument(0)] =
        prefix->second.base + " + " + addressIndex(prefix->second.block) +
        " * " + binding.getTile().str();
    axisIndices[binding.getAxisNode()] = prefix->second.offsets;
    axisStarts[binding.getAxisNode()] = selectedRegionStarts[body.getArgument(0)];
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
  if (auto restores = streamStartRestores.find(&operation);
      restores != streamStartRestores.end()) {
    for (auto value = restores->second.rbegin();
         value != restores->second.rend(); ++value) {
      if (value->second)
        axisStarts[value->first] = *value->second;
      else
        axisStarts.erase(value->first);
    }
    streamStartRestores.erase(restores);
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

LogicalResult ProgramMaterializer::emitContract(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "tl.dot")
    return operation.emitOpError("lacks a Triton contraction binding");
  StringRef form = binding.getForm();
  if (operation.getNumOperands() != 2)
    return operation.emitOpError("Triton contraction requires two operands");
  FailureOr<target::lowering::ContractionOrientation> orientation =
      target::lowering::selectedContractionOrientation(binding);
  if (failed(orientation))
    return failure();
  auto resultTensor = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
  std::string accumulatorDtype =
      resultTensor ? tritonDtype(resultTensor.getElementType()).str() : "";
  if (accumulatorDtype.empty())
    return operation.emitOpError("has no supported Triton accumulator dtype");
  auto accumulatorForm =
      binding.operation->getAttrOfType<StringAttr>(contractAccumulatorAttr);
  std::optional<std::string> accumulatorInput;
  if (accumulatorForm && accumulatorForm.getValue() == "input") {
    std::optional<int64_t> inputID = binding.getAccumulatorInputValue();
    Value input = inputID ? kernel.values.lookup(*inputID) : Value();
    auto emitted = input ? valueNames.find(input) : valueNames.end();
    if (!inputID || !input || emitted == valueNames.end())
      return operation.emitOpError(
          "cannot resolve the selected Triton fused accumulator input");
    accumulatorInput = emitted->second;
  }
  auto replay = deferredContractReplays.find(&operation);
  if (form == "replay") {
    if (replay == deferredContractReplays.end())
      return operation.emitOpError(
          "Triton replay form has no physical producer slice");
    std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
    plan::AxisOp reductionAxis =
        reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
    FailureOr<std::string> resultShape = emitTensorShape(operation, 0);
    if (!reductionAxis || failed(resultShape))
      return operation.emitOpError(
          "replay form has no selected Triton reduction axis");
    std::string reductionRole = reductionAxis.getRole().str();
    std::string reductionTile = reductionAxis.getTile().str();
    bool streamReduction = target::lowering::isEnclosingStreamReductionAxis(
        planIndex, operation, reductionAxis.getNode());
    std::string result = makeResultName(operation, 0);
    std::string reductionOffset = "offs_" + reductionRole;
    line(result + " = tl.zeros(" + *resultShape + ", dtype=" +
         accumulatorDtype + ")");
    if (!streamReduction) {
      line("for reduction_block in range(0, tl.cdiv(" +
           roleDimensions.lookup(reductionRole) + ", " + reductionTile + ")):");
      ++indentation;
      line(reductionOffset + " = " + addressIndex("reduction_block") + " * " +
           reductionTile + " + " +
           addressIndex("tl.arange(0, " + reductionTile + ")"));
      axisIndices[reductionAxis.getNode()] = reductionOffset;
    } else if (axisIndices.lookup(reductionAxis.getNode()).empty()) {
      return operation.emitOpError(
          "has no active Triton stream-bound reduction range");
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
      lhsExpression = "tl.trans(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = "tl.trans(" + rhsExpression + ")";
    line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression +
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
          "Triton direct contraction unexpectedly owns deferred operands");
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    if (failed(lhs) || failed(rhs))
      return failure();
    std::string lhsExpression = lhs->str();
    std::string rhsExpression = rhs->str();
    if (orientation->lhsTranspose)
      lhsExpression = orientation->batched
                          ? "tl.trans(" + lhsExpression + ", 0, 2, 1)"
                          : "tl.trans(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = orientation->batched
                          ? "tl.trans(" + rhsExpression + ", 0, 2, 1)"
                          : "tl.trans(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    std::string accumulator = accumulatorInput
                                  ? ", " + *accumulatorInput
                                  : ", out_dtype=" + accumulatorDtype;
    line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression +
         accumulator + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (form == "deferred_one") {
    if (static_cast<bool>(lhsLoad) == static_cast<bool>(rhsLoad))
      return operation.emitOpError(
          "Triton one-sided deferred form does not own exactly one deferred operand");
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
          "one-sided deferred Triton contraction has inconsistent plan spaces");
    std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
    plan::AxisOp reductionAxis =
        reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
    if (!reductionAxis)
      return operation.emitOpError(
          "one-sided deferred Triton form has no selected reduction axis");
    if (!target::lowering::isEnclosingStreamReductionAxis(
            planIndex, operation, reductionAxis.getNode()) ||
        axisIndices.lookup(reductionAxis.getNode()).empty())
      return operation.emitOpError(
          "one-sided deferred Triton contraction requires an active "
          "stream-bound reduction range");
    FailureOr<ABIView *> view = lookupView(load->getOperand(0), *load);
    FailureOr<std::string> pointers =
        succeeded(view) ? emitPointerExpression(*load, **view, false)
                        : FailureOr<std::string>(failure());
    FailureOr<std::string> mask = emitMaskExpression(*load, false);
    FailureOr<StringRef> direct = lookupValue(operation, directOperand);
    FailureOr<int64_t> loadNode = target::getNodeID(*load, "deferred load");
    bool descriptorCandidate =
        succeeded(loadNode) && succeeded(view) &&
        planIndex.transferForms.lookup(*loadNode) == "pointer_or_descriptor";
    bool linearDescriptor =
        descriptorCandidate &&
        planIndex.descriptorLayouts.lookup(*loadNode) == "linear";
    FailureOr<std::string> descriptorOffset =
        descriptorCandidate ? descriptorOffsets(*load, **view)
                            : FailureOr<std::string>(std::string());
    FailureOr<std::string> descriptorShape =
        descriptorCandidate && !linearDescriptor
            ? emitTensorShape(*load, 0)
            : FailureOr<std::string>(std::string());
    if (failed(view) || failed(pointers) || failed(mask) || failed(direct) ||
        failed(loadNode) || failed(descriptorOffset) || failed(descriptorShape))
      return failure();
    std::string loaded = makeResultName(*load, 0);
    if (descriptorCandidate) {
      line("if USE_TMA:");
      ++indentation;
      if (linearDescriptor)
        line(loaded + " = " + descriptorName(*load) + ".load([" +
             *descriptorOffset + "])");
      else
        line(loaded + " = tl.reshape(" + descriptorName(*load) + ".load([" +
             *descriptorOffset + "]), " + *descriptorShape + ")");
      --indentation;
      line("else:");
      ++indentation;
      line(loaded + " = tl.load(" + *pointers + ", mask=" + *mask +
           ", other=0.0)");
      --indentation;
    } else {
      line(loaded + " = tl.load(" + *pointers + ", mask=" + *mask +
           ", other=0.0)");
    }
    std::string lhsExpression = loadOperand == 0 ? loaded : direct->str();
    std::string rhsExpression = loadOperand == 1 ? loaded : direct->str();
    if (orientation->lhsTranspose)
      lhsExpression = "tl.trans(" + lhsExpression + ")";
    if (orientation->rhsTranspose)
      rhsExpression = "tl.trans(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    std::string accumulator = accumulatorInput
                                  ? ", " + *accumulatorInput
                                  : ", out_dtype=" + accumulatorDtype;
    line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression +
         accumulator + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (form != "deferred_two" || !lhsLoad || !rhsLoad)
    return operation.emitOpError("has an inconsistent Triton contraction form");
  FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  std::optional<int64_t> lhsAxisNode = binding.getLhsResultAxisNode();
  std::optional<int64_t> rhsAxisNode = binding.getRhsResultAxisNode();
  std::optional<int64_t> reductionNode = binding.getReductionAxisNode();
  plan::AxisOp lhsResult =
      lhsAxisNode ? planIndex.axes.lookup(*lhsAxisNode) : plan::AxisOp();
  plan::AxisOp rhsResult =
      rhsAxisNode ? planIndex.axes.lookup(*rhsAxisNode) : plan::AxisOp();
  plan::AxisOp reductionAxis =
      reductionNode ? planIndex.axes.lookup(*reductionNode) : plan::AxisOp();
  if (failed(lhsView) || failed(rhsView) || !lhsResult || !rhsResult ||
      !reductionAxis)
    return failure();

  std::string reductionRole = reductionAxis.getRole().str();
  std::string reductionTile = reductionAxis.getTile().str();
  bool streamReduction = target::lowering::isEnclosingStreamReductionAxis(
      planIndex, operation, reductionAxis.getNode());
  FailureOr<std::string> lhsTile = physicalAxisTile(lhsResult);
  FailureOr<std::string> rhsTile = physicalAxisTile(rhsResult);
  if (failed(lhsTile) || failed(rhsTile))
    return failure();
  std::string reductionOffset = "offs_" + reductionRole;
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.zeros((" + *lhsTile + ", " + *rhsTile +
       "), dtype=" + accumulatorDtype + ")");
  if (!streamReduction) {
    line("for reduction_block in range(0, tl.cdiv(" +
         roleDimensions.lookup(reductionRole) + ", " + reductionTile + ")):");
    ++indentation;
    line(reductionOffset + " = " + addressIndex("reduction_block") + " * " +
         reductionTile + " + " +
         addressIndex("tl.arange(0, " + reductionTile + ")"));
    axisIndices[reductionAxis.getNode()] = reductionOffset;
  } else if (axisIndices.lookup(reductionAxis.getNode()).empty()) {
    return operation.emitOpError(
        "has no active Triton stream-bound reduction range");
  }
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
  auto emitDeferredLoad = [&](Operation &load, ABIView &view,
                              StringRef result, StringRef pointers,
                              StringRef mask) -> LogicalResult {
    FailureOr<int64_t> loadNode = target::getNodeID(load, "deferred load");
    bool descriptorCandidate =
        succeeded(loadNode) &&
        planIndex.transferForms.lookup(*loadNode) == "pointer_or_descriptor";
    bool linearDescriptor =
        descriptorCandidate &&
        planIndex.descriptorLayouts.lookup(*loadNode) == "linear";
    FailureOr<std::string> descriptorOffset =
        descriptorCandidate ? descriptorOffsets(load, view)
                            : FailureOr<std::string>(std::string());
    FailureOr<std::string> descriptorShape =
        descriptorCandidate && !linearDescriptor
            ? emitTensorShape(load, 0)
            : FailureOr<std::string>(std::string());
    if (failed(loadNode) || failed(descriptorOffset) || failed(descriptorShape))
      return failure();
    if (descriptorCandidate) {
      line("if USE_TMA:");
      ++indentation;
      if (linearDescriptor)
        line(result.str() + " = " + descriptorName(load) + ".load([" +
             *descriptorOffset + "])");
      else
        line(result.str() + " = tl.reshape(" + descriptorName(load) +
             ".load([" + *descriptorOffset + "]), " + *descriptorShape + ")");
      --indentation;
      line("else:");
      ++indentation;
      line(result.str() + " = tl.load(" + pointers.str() + ", mask=" +
           mask.str() + ", other=0.0)");
      --indentation;
    } else {
      line(result.str() + " = tl.load(" + pointers.str() + ", mask=" +
           mask.str() + ", other=0.0)");
    }
    return success();
  };
  if (failed(emitDeferredLoad(*lhsLoad, **lhsView, lhs, *lhsPointers, *lhsMask)) ||
      failed(emitDeferredLoad(*rhsLoad, **rhsView, rhs, *rhsPointers, *rhsMask)))
    return failure();
  std::string lhsExpression = lhs;
  std::string rhsExpression = rhs;
  if (orientation->lhsTranspose)
    lhsExpression = "tl.trans(" + lhsExpression + ")";
  if (orientation->rhsTranspose)
    rhsExpression = "tl.trans(" + rhsExpression + ")";
  line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression + ", " +
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
  if (failed(node) || !binding || binding.getLowering() != "tl.dot_scaled" ||
      operation.getNumOperands() != 4 || operation.getNumResults() != 1)
    return operation.emitOpError("lacks a Triton scaled-contraction binding");
  if (binding.getForm() != "scaled_direct")
    return operation.emitOpError(
        "has no realized direct Triton scaled-contraction form");
  for (Value operand : operation.getOperands())
    if (deferredLoads.count(operand))
      return operation.emitOpError(
          "Triton scaled contraction requires materialized operand tiles");
  auto format = [&](StringRef attribute) -> FailureOr<std::string> {
    auto value = operation.getAttrOfType<StringAttr>(attribute);
    if (!value)
      return failure();
    if (value.getValue() == "f8e4m3fn")
      return std::string("e4m3");
    if (value.getValue() == "f8e5m2")
      return std::string("e5m2");
    return operation.emitOpError("has no Triton microscaling format");
  };
  auto lhsGroup =
      operation.getAttrOfType<IntegerAttr>("intent.lhs_group_size");
  auto rhsGroup =
      operation.getAttrOfType<IntegerAttr>("intent.rhs_group_size");
  FailureOr<std::string> lhsFormat = format("intent.lhs_format");
  FailureOr<std::string> rhsFormat = format("intent.rhs_format");
  FailureOr<StringRef> lhs = lookupValue(operation, 0);
  FailureOr<StringRef> rhs = lookupValue(operation, 1);
  FailureOr<StringRef> lhsScale = lookupValue(operation, 2);
  FailureOr<StringRef> rhsScale = lookupValue(operation, 3);
  if (!lhsGroup || !rhsGroup || lhsGroup.getInt() != 32 ||
      rhsGroup.getInt() != 32 || failed(lhsFormat) || failed(rhsFormat) ||
      failed(lhs) || failed(rhs) || failed(lhsScale) || failed(rhsScale))
    return operation.emitOpError(
        "Triton scaled contraction requires matching K-group size 32");
  StringRef layout = binding.getScaledLayout();
  if (layout != "grouped_rank_two" && layout != "flattened_rank_three")
    return operation.emitOpError(
        "has no selected Triton scaled-contraction layout");
  std::string lhsExpression = lhs->str();
  std::string rhsExpression = rhs->str();
  if (layout == "flattened_rank_three") {
    lhsExpression += ".reshape((" + lhs->str() + ".shape[0], " + lhs->str() +
                     ".shape[1] * " + lhs->str() + ".shape[2]))";
    rhsExpression += ".reshape((" + rhs->str() + ".shape[0] * " + rhs->str() +
                     ".shape[1], " + rhs->str() + ".shape[2]))";
  }
  std::string result = makeResultName(operation, 0);
  line("if USE_NATIVE_SCALED:");
  ++indentation;
  line(result + " = tl.dot_scaled(" + lhsExpression + ", " + lhsScale->str() +
       ", \"" + *lhsFormat + "\", " + rhsExpression + ", tl.trans(" +
       rhsScale->str() + "), \"" + *rhsFormat +
       "\", out_dtype=tl.float32)");
  --indentation;
  line("else:");
  ++indentation;
  std::string groupedLhs = lhs->str();
  std::string groupedRhs = rhs->str();
  if (layout == "grouped_rank_two") {
    groupedLhs += ".reshape((" + lhs->str() + ".shape[0], " + lhsScale->str() +
                  ".shape[1], 32))";
    groupedRhs += ".reshape((" + rhsScale->str() + ".shape[0], 32, " +
                  rhs->str() + ".shape[1]))";
  }
  std::string decodedLhsScale =
      syntax::cast(lhsScale->str(), "tl.float32", true, true);
  std::string decodedRhsScale =
      syntax::cast(rhsScale->str(), "tl.float32", true, true);
  std::string expandedLhs = result + "_expanded_lhs";
  std::string expandedRhs = result + "_expanded_rhs";
  line(expandedLhs + " = tl.cast(" + groupedLhs +
       ", tl.float32) * (" + decodedLhsScale + ")[:, :, None]");
  line(expandedRhs + " = tl.cast(" + groupedRhs +
       ", tl.float32) * (" + decodedRhsScale + ")[:, None, :]");
  line(result + " = tl.dot(" + expandedLhs + ".reshape((" + expandedLhs +
       ".shape[0], " + expandedLhs + ".shape[1] * " + expandedLhs +
       ".shape[2])), " + expandedRhs + ".reshape((" + expandedRhs +
       ".shape[0] * " + expandedRhs + ".shape[1], " + expandedRhs +
       ".shape[2])), out_dtype=tl.float32)");
  --indentation;
  bindResult(operation, 0, result);
  return success();
}

LogicalResult ProgramMaterializer::emitStore(Operation &operation) {
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
  FailureOr<std::string> validity =
      boundary.getValidityDomainNodes().empty()
          ? FailureOr<std::string>(std::string("True"))
          : emitValidityExpression(boundary.getValidityTensorAxes(),
                                   boundary.getValidityDomainNodes(),
                                   operation.getOperand(valueIndex.getInt()),
                                   operation);
  bool descriptorCandidate =
      planIndex.transferForms.lookup(*node) == "pointer_or_descriptor";
  bool linearDescriptor =
      descriptorCandidate && planIndex.descriptorLayouts.lookup(*node) == "linear";
  FailureOr<std::string> descriptorOffset =
      descriptorCandidate ? descriptorOffsets(operation, **view)
                          : FailureOr<std::string>(std::string());
  FailureOr<SmallVector<std::string>> descriptorBlock =
      descriptorCandidate && !linearDescriptor
          ? descriptorBlockShape(operation, **view)
          : FailureOr<SmallVector<std::string>>(SmallVector<std::string>());
  if (failed(pointers) || failed(mask) || failed(validity) ||
      failed(descriptorOffset) || failed(descriptorBlock))
    return failure();
  if (*validity != "True" && *validity != *mask)
    *mask = *mask == "True" ? *validity
                            : "(" + *mask + ") & (" + *validity + ")";
  if (descriptorCandidate) {
    line("if USE_TMA:");
    ++indentation;
    if (linearDescriptor) {
      line(descriptorName(operation) + ".store([" + *descriptorOffset + "], " +
           stored->str() + ")");
    } else {
      std::string blockShape = "(";
      for (auto [position, extent] : llvm::enumerate(*descriptorBlock)) {
        if (position)
          blockShape += ", ";
        blockShape += extent;
      }
      if (descriptorBlock->size() == 1)
        blockShape += ",";
      blockShape += ")";
      line(descriptorName(operation) + ".store([" + *descriptorOffset +
           "], tl.reshape(" + stored->str() + ", " + blockShape + "))");
    }
    --indentation;
    line("else:");
    ++indentation;
    line("tl.store(" + *pointers + ", " + stored->str() + ", mask=" +
         *mask + ")");
    --indentation;
  } else {
    line("tl.store(" + *pointers + ", " + stored->str() + ", mask=" +
         *mask + ")");
  }
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
  if (failed(node) || !binding || binding.getDefer() ||
      binding.getStoreMask() != "predicate" || !valueIndex ||
      failed(relation) || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical Triton unique store");
  FailureOr<std::string> pointers =
      emitPointerExpression(operation, **view, true);
  FailureOr<std::string> mask = emitMaskExpression(operation, true);
  FailureOr<std::string> validity =
      binding.getValidityDomainNodes().empty()
          ? FailureOr<std::string>(std::string("True"))
          : emitValidityExpression(binding.getValidityTensorAxes(),
                                   binding.getValidityDomainNodes(),
                                   operation.getOperand(valueIndex.getInt()),
                                   operation);
  if (failed(pointers) || failed(mask) || failed(validity))
    return failure();
  if (*validity != "True" && *validity != *mask)
    *mask = *mask == "True" ? *validity
                            : "(" + *mask + ") & (" + *validity + ")";
  line("tl.store(" + *pointers + ", " + stored->str() + ", mask=" + *mask +
       ")");
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
    return operation.emitOpError("lacks a mechanical Triton atomic merge");
  FailureOr<std::string> pointer = emitPointerExpression(operation, **view, true);
  FailureOr<std::string> mask = emitMaskExpression(operation, true);
  if (failed(pointer) || failed(mask))
    return failure();
  std::string call = "tl.atomic_add(" + *pointer + ", " + stored->str();
  if (*mask != "True")
    call += ", mask=" + *mask;
  call += ", sem='relaxed', scope='gpu')";
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
      boundary.getCheckBounds() || !compareIndex ||
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

} // namespace intent::triton::lowering

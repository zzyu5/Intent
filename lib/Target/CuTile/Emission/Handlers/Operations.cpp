#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

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
  for (StringRef name : {"intent.dim", "intent.domain", "intent.partition",
                         "intent.yield", "intent.return", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();
  if (failed(addHandler(registry, "intent.constant",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitConstant(op);
                        })) ||
      failed(addHandler(
          registry, "intent.parallel",
          [&](Operation &op) { return emitter.enterParallel(op); },
          [&](Operation &op) { return emitter.leaveParallel(op); })) ||
      failed(addHandler(registry, "intent.view_load",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitLoad(op);
                        })) ||
      failed(addHandler(registry, "intent.reduce",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitReduction(op);
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
      failed(addHandler(registry, "intent.cast",
                        [&](Operation &op) {
                          if (!emitter.selectOperation(op))
                            return success();
                          return emitter.emitCast(op);
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
          [&](Operation &op) { return emitter.enterStateStream(op); },
          [&](Operation &op) { return emitter.leaveStateStream(op); })) ||
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

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (usesRaggedOrderedTraversal()) {
    if (&operation == programRoot) {
      line("query_block = ct.bid(" +
           std::to_string(planIndex.program.getWorkerAxes()[0]) + ")");
      line("sequence_index = ct.bid(" +
           std::to_string(planIndex.program.getWorkerAxes()[1]) + ")");
      line("sequence_begin = ct.gather(" + raggedOffsets->argument->name +
           ", sequence_index, padding_value=0)");
      line("sequence_end = ct.gather(" + raggedOffsets->argument->name +
           ", sequence_index + 1, padding_value=0)");
      line("sequence_length = sequence_end - sequence_begin");
      line("query_offsets = sequence_begin + query_block * TILE_SIZE_M + "
           "ct.arange(TILE_SIZE_M, dtype=ct.int32)");
      line("query_offsets = ct.where(query_offsets < sequence_end, "
           "query_offsets, " + raggedMembersView->argument->name +
           ".shape[0])");
    }
    if (operation.getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation.getRegion(0)) ||
        operation.getRegion(0).front().getNumArguments() != 1)
      return operation.emitOpError(
          "ragged ownership requires one parallel region argument");
    BlockArgument argument = operation.getRegion(0).front().getArgument(0);
    FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
    if (failed(axis))
      return failure();
    if (axis->getRole() == "program_0")
      valueNames[argument] = "sequence_index";
    else if (axis->getRole() == "program_1")
      valueNames[argument] = "query_offsets";
    else
      return operation.emitOpError(
          "parallel region has no ragged ownership role");
    return success();
  }
  if (usesStagedEmission()) {
    if (operation.getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation.getRegion(0)) ||
        operation.getRegion(0).front().getNumArguments() != 1)
      return operation.emitOpError(
          "ragged program ownership requires one region argument");
    BlockArgument argument = operation.getRegion(0).front().getArgument(0);
    FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
    if (failed(axis))
      return failure();
    if (axis->getRole() == "program_0")
      valueNames[argument] = "expert";
    else if (axis->getRole() == "program_1")
      valueNames[argument] = "member_offsets";
    else
      return operation.emitOpError("has no ragged program-axis role");
    return success();
  }
  if (hasTraversal("ordered_stream") &&
      planIndex.program.getOwnership() == "block_rows") {
    if (&operation != programRoot || operation.getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation.getRegion(0)) ||
        operation.getRegion(0).front().getNumArguments() != 1)
      return operation.emitOpError(
          "row stream requires one root ownership argument");
    int64_t workerAxis = planIndex.program.getWorkerAxes().front();
    line("program_index = ct.bid(" + std::to_string(workerAxis) + ")");
    valueNames[operation.getRegion(0).front().getArgument(0)] = "program_index";
    return success();
  }
  if (hasTraversal("ordered_stream") &&
      planIndex.program.getOwnership() == "block_tiles") {
    if (&operation == programRoot) {
      line("bid_program_2 = ct.bid(" +
           std::to_string(planIndex.program.getWorkerAxes()[0]) + ")");
      line("bid_program_01 = ct.bid(" +
           std::to_string(planIndex.program.getWorkerAxes()[1]) + ")");
      line("index_program_0 = bid_program_01 // " +
           roleDimensions.lookup("program_1"));
      line("index_program_1 = bid_program_01 % " +
           roleDimensions.lookup("program_1"));
    }
    if (operation.getNumRegions() != 1 ||
        !llvm::hasSingleElement(operation.getRegion(0)) ||
        operation.getRegion(0).front().getNumArguments() != 1)
      return operation.emitOpError(
          "multi-axis mapping requires one parallel region argument");
    FailureOr<plan::AxisOp> axis =
        resolveAxis(operation.getRegion(0).front().getArgument(0), operation);
    if (failed(axis))
      return failure();
    StringRef role = axis->getRole();
    if (role == "program_0")
      valueNames[operation.getRegion(0).front().getArgument(0)] =
          "index_program_0";
    else if (role == "program_1")
      valueNames[operation.getRegion(0).front().getArgument(0)] =
          "index_program_1";
    else if (role == "program_2")
      valueNames[operation.getRegion(0).front().getArgument(0)] =
          "bid_program_2";
    else
      return operation.emitOpError(
          "parallel region has no multi-axis cuTile role");
    return success();
  }
  if (&operation != programRoot)
    return success();
  int64_t workerAxis = planIndex.program.getWorkerAxes().front();
  if (hasTraversal("persistent")) {
    line("program_start = ct.bid(" + std::to_string(workerAxis) + ")");
    line("program_step = ct.num_blocks(" + std::to_string(workerAxis) + ")");
    line(vectorIndex + " = ct.arange(TILE_SIZE, dtype=ct.int32)");
    line("for " + programIndex +
         " in range(program_start, N_ROWS, program_step):");
    ++indentation;
    return success();
  }

  line("M = " + dimensionOwners.lookup(roleDimensions.lookup("program_0")));
  line("N = " + dimensionOwners.lookup(roleDimensions.lookup("program_1")));
  line("bid = ct.bid(" + std::to_string(workerAxis) + ")");
  line("num_bid_m = ct.cdiv(M, TILE_SIZE_M)");
  line("num_bid_n = ct.cdiv(N, TILE_SIZE_N)");
  line("num_bid_in_group = GROUP_SIZE_M * num_bid_n");
  line("group_id = bid // num_bid_in_group");
  line("first_bid_m = group_id * GROUP_SIZE_M");
  line("group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)");
  line("bid_m = first_bid_m + (bid % group_size_m)");
  line("bid_n = (bid % num_bid_in_group) // group_size_m");
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &operation) {
  if (&operation == programRoot && hasTraversal("persistent"))
    --indentation;
  return success();
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary)
    return operation.emitOpError("lacks a cuTile load boundary");
  if (boundary.getDomainNodes().empty() && boundary.getPadding() == "none") {
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
  FailureOr<std::string> indices = indexTuple(operation, false);
  if (failed(view) || failed(indices))
    return failure();
  StringRef padding = boundary.getPadding() == "negative_infinity"
                          ? "-math.inf"
                          : "0.0";
  std::string result = makeResultName(operation, 0);
  if (boundary.getAccess() == "gather") {
    line(result + " = ct.gather(" + (*view)->argument->name + ", " +
         *indices + ", check_bounds=True, padding_value=" + padding.str() +
         ")");
  } else if (boundary.getAccess() == "load") {
    FailureOr<std::string> shape = tileShape(operation);
    FailureOr<std::string> resultShape = emitTensorShape(operation, 0);
    if (failed(shape) || failed(resultShape))
      return failure();
    StringRef paddingMode = boundary.getPadding() == "negative_infinity"
                                ? "ct.PaddingMode.NEG_INF"
                                : "ct.PaddingMode.ZERO";
    line(result + " = ct.load(" + (*view)->argument->name + ", index=" +
         *indices + ", shape=" + *shape +
         ", padding_mode=" + paddingMode.str() + ").reshape(" + *resultShape +
         ")");
  } else {
    return boundary.emitOpError("is not a load-like cuTile access");
  }
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
  std::string result = makeResultName(operation, 0);
  line(result + " = " + binding.getLowering().str() + "(" + operand->str() +
       ", " + std::to_string(binding.getAxis()) + ", keepdims=" +
       (binding.getKeepDims() ? "True" : "False") + ")");
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
  std::string expression;
  if (binding.getLowering() == "ct.maximum")
    expression = "ct.maximum(" + lhs->str() + ", " + rhs->str() + ")";
  else {
    StringRef symbol;
    if (binding.getLowering() == "python_add")
      symbol = "+";
    else if (binding.getLowering() == "python_subtract")
      symbol = "-";
    else if (binding.getLowering() == "python_multiply")
      symbol = "*";
    else if (binding.getLowering() == "python_true_divide")
      symbol = "/";
    else
      return operation.emitOpError("uses an unsupported cuTile binary lowering");
    expression = lhs->str() + " " + symbol.str() + " " + rhs->str();
  }
  std::string result = makeResultName(operation, 0);
  line(result + " = " + expression);
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
    line(result + " = ct.full((1,), " + operand->str() + ", dtype=" +
         targetType + ")");
  else
    line(result + " = ct.astype(" + operand->str() + ", " + targetType + ")");
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
  FailureOr<StringRef> source = lookupValue(operation, 0);
  auto validIndex =
      operation.getAttrOfType<IntegerAttr>("intent.valid_operand_index");
  auto fillIndex =
      operation.getAttrOfType<IntegerAttr>("intent.fill_operand_index");
  FailureOr<StringRef> valid =
      validIndex ? lookupValue(operation, validIndex.getInt())
                 : FailureOr<StringRef>(failure());
  FailureOr<StringRef> fill =
      fillIndex ? lookupValue(operation, fillIndex.getInt())
                : FailureOr<StringRef>(failure());
  if (usesStagedEmission() && binding &&
      binding.getLowering() == "ct.indirect_gather") {
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view) || failed(relation) || failed(valid) || failed(fill))
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
    FailureOr<StringRef> index =
        lookupValue(operation, *(*relation)[0].operands.front());
    if (failed(index))
      return failure();
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.gather(" + (*view)->argument->name + ", " +
         index->str() + ", check_bounds=True, padding_value=" + fill->str() +
         ")");
    line(result + " = ct.where(member_mask & " + valid->str() + ", " +
         result + ", " + fill->str() + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (failed(node) || !binding ||
      (binding.getLowering() != "alias_column" &&
       binding.getLowering() != "expand_dims") ||
      failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "full_slice" ||
      (*relation)[1].kind != "new_axis" || failed(source) || failed(valid) ||
      failed(fill))
    return operation.emitOpError("lacks a mechanical cuTile gather binding");
  std::string result = makeResultName(operation, 0);
  std::string expanded =
      binding.getLowering() == "expand_dims" ? source->str() + "[:, None]"
                                              : source->str();
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
    return operation.emitOpError("lacks a staged cuTile members binding");
  bindResult(operation, 0, "routes");
  return success();
}

LogicalResult SourceEmitter::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || binding.getOrder() != "forward" ||
      binding.getCarrySpace() != "register" || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical cuTile stream binding");
  Block &body = operation.getRegion(0).front();
  if (operation.getNumOperands() != operation.getNumResults() + 1 ||
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
  std::string streamExtent =
      usesRaggedOrderedTraversal()
          ? "sequence_length"
          : dimensionOwners.lookup(roleDimensions.lookup("stream_0"));
  line("for stream_tile in range(ct.cdiv(" + streamExtent + ", " +
       binding.getTile().str() + ")):");
  ++indentation;
  if (usesRaggedOrderedTraversal()) {
    line("stream_offsets = sequence_begin + stream_tile * " +
         binding.getTile().str() + " + ct.arange(" + binding.getTile().str() +
         ", dtype=ct.int32)");
    line("stream_offsets = ct.where(stream_offsets < sequence_end, "
         "stream_offsets, " + raggedMembersView->argument->name +
         ".shape[0])");
  }
  valueNames[body.getArgument(0)] = "stream_tile";
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
  if (usesStagedEmission()) {
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
        binding.getLhsTranspose() || binding.getRhsTranspose())
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
    line("offs_reduction = k_tile * TILE_SIZE_K + "
         "ct.arange(TILE_SIZE_K, dtype=ct.int32)");

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
           rows->str() + "[:, None], offs_reduction[None, :]), "
           "check_bounds=True, padding_value=0.0)");
      line(lhs + " = ct.where(member_mask[:, None], " + lhs + ", 0.0)");
    } else {
      auto workspace = workspaceNames.find(operation.getOperand(0));
      if (workspace == workspaceNames.end())
        return operation.emitOpError(
            "staged contraction input has no materialized workspace");
      lhs = "stage_input";
      line(lhs + " = ct.gather(" + workspace->second +
           ", (safe_member_offsets[:, None], offs_reduction[None, :]), "
           "check_bounds=True, padding_value=0.0)");
    }
    std::string rhs = makeResultName(*rhsLoad, 0);
    line(rhs + " = ct.load(" + (*rhsView)->argument->name +
         ", index=(expert, k_tile, bid_feature), shape=(1, TILE_SIZE_K, "
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
    if (binding.getLhsTranspose())
      lhsExpression = "ct.transpose(" + lhsExpression + ")";
    if (binding.getRhsTranspose())
      rhsExpression = "ct.transpose(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = ct.full(" + *shape +
         ", 0.0, dtype=ct.float32)");
    line(result + " = ct.mma(" + lhsExpression + ", " + rhsExpression +
         ", " + result + ")");
    bindResult(operation, 0, result);
    return success();
  }
  if (!lhsLoad || !rhsLoad || binding.getLhsTranspose() ||
      binding.getRhsTranspose())
    return operation.emitOpError(
        "deferred cuTile contraction has inconsistent operand residency");
  FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  FailureOr<std::string> lhsIndex = indexTuple(*lhsLoad, true);
  FailureOr<std::string> rhsIndex = indexTuple(*rhsLoad, true);
  FailureOr<std::string> lhsShape = tileShape(*lhsLoad);
  FailureOr<std::string> rhsShape = tileShape(*rhsLoad);
  if (failed(lhsView) || failed(rhsView) || failed(lhsIndex) ||
      failed(rhsIndex) || failed(lhsShape) || failed(rhsShape))
    return failure();

  std::string result = makeResultName(operation, 0);
  line("num_tiles_k = ct.num_tiles(" + (*lhsView)->argument->name +
       ", axis=1, shape=" + *lhsShape + ")");
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
  line(lhs + " = ct.load(" + (*lhsView)->argument->name + ", index=" +
       *lhsIndex + ", shape=" + *lhsShape +
       ", padding_mode=ct.PaddingMode.ZERO).astype(" + operandDtype + ")");
  line(rhs + " = ct.load(" + (*rhsView)->argument->name + ", index=" +
       *rhsIndex + ", shape=" + *rhsShape +
       ", padding_mode=ct.PaddingMode.ZERO).astype(" + operandDtype + ")");
  line(result + " = ct.mma(" + lhs + ", " + rhs + ", " + result + ")");
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
  FailureOr<std::string> indices = indexTuple(operation, false);
  if (!valueIndex || failed(node) || !boundary || failed(stored) ||
      failed(view) || failed(indices))
    return operation.emitOpError("lacks a cuTile store binding");
  if (boundary.getAccess() == "scatter")
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
    line("ct.store(" + (*view)->argument->name + ", index=" + *indices +
         ", tile=" + tile + ")");
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
       ", (unique_rows[:, None], offs_feature[None, :]), " + stored->str() +
       ", check_bounds=True)");
  return success();
}

LogicalResult SourceEmitter::emitAtomic(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "atomic emission");
  plan::AtomicOp binding =
      succeeded(node) ? planIndex.atomics.lookup(*node) : plan::AtomicOp();
  auto valueIndex =
      operation.getAttrOfType<IntegerAttr>("intent.value_operand_index");
  FailureOr<SmallVector<target::IndexTerm>> relation =
      target::parseIndexRelation(operation);
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<StringRef> stored =
      valueIndex ? lookupValue(operation, valueIndex.getInt())
                 : FailureOr<StringRef>(failure());
  if (failed(node) || !binding || binding.getLowering() != "ct.atomic_add" ||
      binding.getMemoryOrder() != "relaxed" ||
      binding.getMemoryScope() != "device" || !valueIndex ||
      failed(relation) || relation->size() != 2 ||
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
       ", (atomic_rows[:, None], offs_feature[None, :]), " + stored->str() +
       ", check_bounds=True, memory_order=ct.MemoryOrder.RELAXED, "
       "memory_scope=ct.MemoryScope.DEVICE)");
  return success();
}

} // namespace intent::cutile::emission

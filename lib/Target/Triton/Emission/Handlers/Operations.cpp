#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
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

} // namespace

LogicalResult registerEmissionHandlers(target::OperationHandlerRegistry &registry,
                                       SourceEmitter &emitter) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.dim", "intent.domain", "intent.partition",
                         "intent.yield", "intent.return"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();
  if (failed(addHandler(registry, "intent.constant",
                        [&](Operation &op) { return emitter.emitConstant(op); })) ||
      failed(addHandler(
          registry, "intent.parallel",
          [&](Operation &op) { return emitter.enterParallel(op); },
          [&](Operation &op) { return emitter.leaveParallel(op); })) ||
      failed(addHandler(registry, "intent.view_load",
                        [&](Operation &op) { return emitter.emitLoad(op); })) ||
      failed(addHandler(registry, "intent.reduce",
                        [&](Operation &op) { return emitter.emitReduction(op); })) ||
      failed(addHandler(registry, "intent.broadcast",
                        [&](Operation &op) { return emitter.emitBroadcast(op); })) ||
      failed(addHandler(registry, "intent.unary",
                        [&](Operation &op) { return emitter.emitUnary(op); })) ||
      failed(addHandler(registry, "intent.binary",
                        [&](Operation &op) { return emitter.emitBinary(op); })) ||
      failed(addHandler(registry, "intent.cast",
                        [&](Operation &op) { return emitter.emitCast(op); })) ||
      failed(addHandler(registry, "intent.full",
                        [&](Operation &op) { return emitter.emitFull(op); })) ||
      failed(addHandler(registry, "intent.zeros",
                        [&](Operation &op) { return emitter.emitZeros(op); })) ||
      failed(addHandler(registry, "intent.gather",
                        [&](Operation &op) { return emitter.emitGather(op); })) ||
      failed(addHandler(
          registry, "intent.state_stream",
          [&](Operation &op) { return emitter.enterStateStream(op); },
          [&](Operation &op) { return emitter.leaveStateStream(op); })) ||
      failed(addHandler(registry, "intent.contract",
                        [&](Operation &op) { return emitter.emitContract(op); })) ||
      failed(addHandler(registry, "intent.view_store",
                        [&](Operation &op) { return emitter.emitStore(op); })))
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
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (planIndex.program.getMapping() == "grid_stride") {
    if (&operation != programRoot)
      return operation.emitOpError("is not owned by the resolved program");
    int64_t workerAxis = planIndex.program.getWorkerAxes().front();
    line("program_start = tl.program_id(" + std::to_string(workerAxis) + ")");
    line("program_step = tl.num_programs(" + std::to_string(workerAxis) + ")");
    line("for " + programIndex +
         " in tl.range(program_start, n_rows, program_step, "
         "num_stages=num_stages):");
    ++indentation;
    line(vectorIndex + " = tl.arange(0, BLOCK_SIZE)");
    return success();
  }
  if (planIndex.program.getMapping() == "multi_axis_stream") {
    if (&operation == programRoot) {
      line("pid_program_2 = tl.program_id(axis=" +
           std::to_string(planIndex.program.getWorkerAxes()[0]) + ")");
      line("pid_program_01 = tl.program_id(axis=" +
           std::to_string(planIndex.program.getWorkerAxes()[1]) + ")");
      line("index_program_0 = pid_program_01 // " +
           roleDimensions.lookup("program_1"));
      line("index_program_1 = pid_program_01 % " +
           roleDimensions.lookup("program_1"));
      line("offs_program_2 = pid_program_2 * BLOCK_SIZE_Q + "
           "tl.arange(0, BLOCK_SIZE_Q)");
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
          "offs_program_2";
    else
      return operation.emitOpError(
          "parallel region has no multi-axis program role");
    return success();
  }
  if (&operation != programRoot)
    return success();
  line("pid = tl.program_id(axis=" +
       std::to_string(planIndex.program.getWorkerAxes().front()) + ")");
  line("num_pid_m = tl.cdiv(" + roleDimensions.lookup("program_0") +
       ", BLOCK_SIZE_M)");
  line("num_pid_n = tl.cdiv(" + roleDimensions.lookup("program_1") +
       ", BLOCK_SIZE_N)");
  line("num_pid_in_group = GROUP_SIZE_M * num_pid_n");
  line("group_id = pid // num_pid_in_group");
  line("first_pid_m = group_id * GROUP_SIZE_M");
  line("group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)");
  line("pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)");
  line("pid_n = (pid % num_pid_in_group) // group_size_m");
  line("offs_program_0 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)");
  line("offs_program_1 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)");
  line("load_program_0 = offs_program_0 % " +
       roleDimensions.lookup("program_0"));
  line("load_program_1 = offs_program_1 % " +
       roleDimensions.lookup("program_1"));
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &operation) {
  if (planIndex.program.getMapping() == "grid_stride" &&
      &operation == programRoot)
    --indentation;
  return success();
}

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  bool feedsContract = llvm::any_of(operation.getResult(0).getUsers(),
                                   [](Operation *user) {
                                     return user->getName().getStringRef() ==
                                            "intent.contract";
                                   });
  if (feedsContract &&
      planIndex.program.getMapping() == "grouped_2d_tiles") {
    deferredLoads[operation.getResult(0)] = &operation;
    return success();
  }
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  if (failed(node) || !boundary || boundary.getLoadFill() == "none")
    return operation.emitOpError("lacks a semantics-preserving load boundary");
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  if (failed(view))
    return failure();
  FailureOr<std::string> pointers =
      emitPointerExpression(operation, **view, false);
  FailureOr<std::string> mask = emitMaskExpression(operation, false);
  if (failed(pointers) || failed(mask))
    return failure();
  StringRef fill = boundary.getLoadFill() == "negative_infinity"
                       ? "-float('inf')"
                       : "0.0";
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.load(" + *pointers + ", mask=" + *mask +
       ", other=" + fill.str() + ")");
  valueNames[operation.getResult(0)] = result;
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
  std::string result = makeResultName(operation, 0);
  line(result + " = " + binding.getLowering().str() + "(" + operand->str() +
       ", axis=" + std::to_string(binding.getAxis()) + ")");
  valueNames[operation.getResult(0)] = result;
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
  valueNames[operation.getResult(0)] = operand->str();
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
  valueNames[operation.getResult(0)] = result;
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
  if (binding.getLowering() == "tl.maximum")
    expression = "tl.maximum(" + lhs->str() + ", " + rhs->str() + ")";
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
      return operation.emitOpError("uses an unsupported binary lowering");
    expression = lhs->str() + " " + symbol.str() + " " + rhs->str();
  }
  std::string result = makeResultName(operation, 0);
  line(result + " = " + expression);
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::emitCast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "cast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  auto resultType = operation.getNumResults() == 1
                        ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                        : RankedTensorType();
  if (failed(node) || !binding || binding.getLowering() != "tl.cast" ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical Triton cast binding");
  StringRef targetType;
  if (resultType.getElementType().isF16())
    targetType = "tl.float16";
  else if (resultType.getElementType().isF32())
    targetType = "tl.float32";
  else
    return operation.emitOpError("casts to an unsupported Triton type");
  std::string result = makeResultName(operation, 0);
  line(result + " = " + operand->str() + ".to(" + targetType.str() + ")");
  valueNames[operation.getResult(0)] = result;
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
  StringRef dtype;
  if (resultType.getElementType().isF32())
    dtype = "tl.float32";
  else if (resultType.getElementType().isF16())
    dtype = "tl.float16";
  else
    return operation.emitOpError("uses an unsupported Triton full dtype");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.full(" + *shape + ", " + fill->str() +
       ", dtype=" + dtype.str() + ")");
  valueNames[operation.getResult(0)] = result;
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
  StringRef dtype;
  if (resultType.getElementType().isF32())
    dtype = "tl.float32";
  else if (resultType.getElementType().isF16())
    dtype = "tl.float16";
  else
    return operation.emitOpError("uses an unsupported Triton zeros dtype");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.zeros(" + *shape + ", dtype=" + dtype.str() + ")");
  valueNames[operation.getResult(0)] = result;
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
  if (failed(node) || !binding || binding.getLowering() != "expand_dims" ||
      failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "full_slice" ||
      (*relation)[1].kind != "new_axis" || failed(source) || failed(valid) ||
      failed(fill))
    return operation.emitOpError("lacks a mechanical Triton gather binding");
  std::string result = makeResultName(operation, 0);
  line(result + " = tl.where(" + valid->str() + ", " + source->str() +
       "[:, None], " + fill->str() + ")");
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || binding.getOrder() != "forward" ||
      binding.getCarrySpace() != "register" || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical Triton stream binding");
  Block &body = operation.getRegion(0).front();
  if (operation.getNumOperands() != operation.getNumResults() + 1 ||
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
  line("for stream_block in range(0, tl.cdiv(" +
       roleDimensions.lookup("stream_0") + ", " + binding.getTile().str() +
       ")):");
  ++indentation;
  line("offs_stream_0 = stream_block * " + binding.getTile().str() +
       " + tl.arange(0, " + binding.getTile().str() + ")");
  valueNames[body.getArgument(0)] = "offs_stream_0";
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
  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (!lhsLoad && !rhsLoad) {
    FailureOr<StringRef> lhs = lookupValue(operation, 0);
    FailureOr<StringRef> rhs = lookupValue(operation, 1);
    if (failed(lhs) || failed(rhs))
      return failure();
    std::string lhsExpression = lhs->str();
    std::string rhsExpression = rhs->str();
    if (binding.getLhsTranspose())
      lhsExpression = "tl.trans(" + lhsExpression + ")";
    if (binding.getRhsTranspose())
      rhsExpression = "tl.trans(" + rhsExpression + ")";
    std::string result = makeResultName(operation, 0);
    line(result + " = tl.dot(" + lhsExpression + ", " + rhsExpression +
         ", out_dtype=tl.float32)");
    valueNames[operation.getResult(0)] = result;
    return success();
  }
  if (!lhsLoad || !rhsLoad || binding.getLhsTranspose() ||
      binding.getRhsTranspose())
    return operation.emitOpError(
        "deferred Triton contraction has inconsistent operand residency");
  FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
  FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
  if (failed(lhsView) || failed(rhsView))
    return failure();

  std::string result = makeResultName(operation, 0);
  line(result + " = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)");
  line("for reduction_block in range(0, tl.cdiv(" +
       roleDimensions.lookup("reduction_0") + ", BLOCK_SIZE_K)):");
  ++indentation;
  line("offs_reduction_0 = reduction_block * BLOCK_SIZE_K + "
       "tl.arange(0, BLOCK_SIZE_K)");
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
  line(result + " = tl.dot(" + lhs + ", " + rhs + ", " + result + ")");
  --indentation;
  valueNames[operation.getResult(0)] = result;
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

} // namespace intent::triton::emission

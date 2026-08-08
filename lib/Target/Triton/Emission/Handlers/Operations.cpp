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
  if (feedsContract) {
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
  if (!lhsLoad || !rhsLoad)
    return operation.emitOpError(
        "Triton contraction operands must be mechanically deferred view loads");
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

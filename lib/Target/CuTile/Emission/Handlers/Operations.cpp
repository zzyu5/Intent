#include "Support/Model.h"

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
      expression = number < 0 ? "-math.inf" : "math.inf";
    else if (std::isnan(number))
      expression = "math.nan";
    else
      expression = std::to_string(number);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported cuTile constant value");
  }
  line(result + " = " + expression);
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (&operation != programRoot)
    return success();
  int64_t workerAxis = planIndex.program.getWorkerAxes().front();
  if (planIndex.program.getMapping() == "persistent_rows") {
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
  line("num_bid_in_group = " + std::to_string(planIndex.program.getGroupSize()) +
       " * num_bid_n");
  line("group_id = bid // num_bid_in_group");
  line("first_bid_m = group_id * " +
       std::to_string(planIndex.program.getGroupSize()));
  line("group_size_m = min(num_bid_m - first_bid_m, " +
       std::to_string(planIndex.program.getGroupSize()) + ")");
  line("bid_m = first_bid_m + (bid % group_size_m)");
  line("bid_n = (bid % num_bid_in_group) // group_size_m");
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &operation) {
  if (&operation == programRoot &&
      planIndex.program.getMapping() == "persistent_rows")
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
  if (failed(node) || !boundary || boundary.getAccess() != "gather")
    return operation.emitOpError("lacks a cuTile gather boundary");
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<std::string> indices = indexTuple(operation, false);
  if (failed(view) || failed(indices))
    return failure();
  StringRef padding = boundary.getPadding() == "negative_infinity"
                          ? "-math.inf"
                          : "0.0";
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.gather(" + (*view)->argument->name + ", " + *indices +
       ", check_bounds=True, padding_value=" + padding.str() + ")");
  valueNames[operation.getResult(0)] = result;
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
  valueNames[operation.getResult(0)] = result;
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
  if (failed(node) || !binding || binding.getLowering() != "ct.astype" ||
      failed(operand) || !resultType)
    return operation.emitOpError("lacks a mechanical cuTile cast binding");
  std::string targetType = dtypeName(resultType.getElementType(), operation);
  if (targetType.empty())
    return failure();
  std::string result = makeResultName(operation, 0);
  line(result + " = ct.astype(" + operand->str() + ", " + targetType + ")");
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::emitContract(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "contract emission");
  plan::ContractOp binding =
      succeeded(node) ? planIndex.contracts.lookup(*node) : plan::ContractOp();
  if (failed(node) || !binding || binding.getLowering() != "ct.mma" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a cuTile contraction binding");
  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (!lhsLoad || !rhsLoad)
    return operation.emitOpError(
        "cuTile contraction operands must be deferred view loads");
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
  line("operand_dtype = ct.tfloat32 if " + (*lhsView)->argument->name +
       ".dtype == ct.float32 else " + (*lhsView)->argument->name + ".dtype");
  line("for k_tile in range(num_tiles_k):");
  ++indentation;
  std::string lhs = makeResultName(*lhsLoad, 0);
  std::string rhs = makeResultName(*rhsLoad, 0);
  line(lhs + " = ct.load(" + (*lhsView)->argument->name + ", index=" +
       *lhsIndex + ", shape=" + *lhsShape +
       ", padding_mode=ct.PaddingMode.ZERO).astype(operand_dtype)");
  line(rhs + " = ct.load(" + (*rhsView)->argument->name + ", index=" +
       *rhsIndex + ", shape=" + *rhsShape +
       ", padding_mode=ct.PaddingMode.ZERO).astype(operand_dtype)");
  line(result + " = ct.mma(" + lhs + ", " + rhs + ", " + result + ")");
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
  else if (boundary.getAccess() == "store")
    line("ct.store(" + (*view)->argument->name + ", index=" + *indices +
         ", tile=" + stored->str() + ")");
  else
    return boundary.emitOpError("is not a store-like cuTile access");
  return success();
}

} // namespace intent::cutile::emission

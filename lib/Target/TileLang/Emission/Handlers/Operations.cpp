#include "Support/Model.h"

#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>

using namespace mlir;

namespace intent::tilelang::emission {
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
  if (failed(addHandler(registry, "intent.constant", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitConstant(op);
      })) ||
      failed(addHandler(
          registry, "intent.parallel",
          [&](Operation &op) { return emitter.enterParallel(op); },
          [&](Operation &op) { return emitter.leaveParallel(op); })) ||
      failed(addHandler(registry, "intent.view_load", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitLoad(op);
      })) ||
      failed(addHandler(registry, "intent.reduce", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitReduction(op);
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
      failed(addHandler(registry, "intent.cast", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitCast(op);
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
          [&](Operation &op) { return emitter.enterStateStream(op); },
          [&](Operation &op) { return emitter.leaveStateStream(op); })) ||
      failed(addHandler(registry, "intent.contract", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitContract(op);
      })) ||
      failed(addHandler(registry, "intent.view_store", [&](Operation &op) {
        if (!emitter.selectOperation(op))
          return success();
        return emitter.emitStore(op);
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
      expression = number < 0 ? "-T.infinity(T.float32)"
                              : "T.infinity(T.float32)";
    else if (std::isnan(number))
      return operation.emitOpError("TileLang constants cannot materialize NaN");
    else
      expression = std::to_string(number);
  } else if (auto integer = dyn_cast<IntegerAttr>(value)) {
    if (operation.getResult(0).getType().isInteger(1))
      expression = integer.getValue().isZero() ? "False" : "True";
    else
      expression = std::to_string(integer.getInt());
  } else {
    return operation.emitOpError("has an unsupported TileLang constant value");
  }
  line(result + " = " + expression);
  valueNames[operation.getResult(0)] = result;
  return success();
}

LogicalResult SourceEmitter::enterParallel(Operation &operation) {
  if (operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)) ||
      operation.getRegion(0).front().getNumArguments() != 1)
    return operation.emitOpError(
        "TileLang parallel ownership requires one region argument");
  BlockArgument argument = operation.getRegion(0).front().getArgument(0);
  FailureOr<plan::AxisOp> axis = resolveAxis(argument, operation);
  if (failed(axis))
    return failure();
  StringRef role = axis->getRole();
  StringRef mapping = programMapping;
  if (role == "program_0")
    valueNames[argument] = mapping == "persistent_rows" ? "program_index"
                           : mapping == "multi_axis_stream"
                               ? "index_program_0"
                           : mapping == "ragged_stages" ? "expert"
                                                         : "bid_m";
  else if (role == "program_1")
    valueNames[argument] = mapping == "multi_axis_stream"
                               ? "index_program_1"
                           : mapping == "ragged_stages" ? "member_start"
                                                         : "bid_n";
  else if (role == "program_2")
    valueNames[argument] = "bid_program_2";
  else
    return operation.emitOpError("has no TileLang program-axis role");
  return success();
}

LogicalResult SourceEmitter::leaveParallel(Operation &) { return success(); }

LogicalResult SourceEmitter::emitLoad(Operation &operation) {
  bool feedsContract = llvm::any_of(operation.getResult(0).getUsers(),
                                   [](Operation *user) {
                                     return user->getName().getStringRef() ==
                                            "intent.contract";
                                   });
  if (feedsContract &&
      (programMapping == "grouped_2d_tiles" ||
       isRaggedStages())) {
    deferredLoads[operation.getResult(0)] = &operation;
    return success();
  }
  FailureOr<int64_t> node = target::getNodeID(operation, "load emission");
  plan::BoundaryOp boundary =
      succeeded(node) ? planIndex.boundaries.lookup(*node) : plan::BoundaryOp();
  FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
  FailureOr<std::string> indices = accessIndices(operation, false);
  FailureOr<std::string> result =
      allocateResult(operation, 0, feedsContract ? "shared" : "fragment");
  if (failed(node) || !boundary || failed(view) || failed(indices) ||
      failed(result))
    return operation.emitOpError("lacks a mechanical TileLang load binding");
  if (boundary.getPadding() == "negative_infinity")
    line("T.fill(" + *result + ", -T.infinity(T.float32))");
  else
    line("T.clear(" + *result + ")");
  line("T.copy(" + (*view)->argument->name + "[" + *indices + "], " +
       *result + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitReduction(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "reduction emission");
  plan::ReductionOp binding =
      succeeded(node) ? planIndex.reductions.lookup(*node) : plan::ReductionOp();
  FailureOr<StringRef> operand = lookupValue(operation, 0);
  if (failed(node) || !binding || failed(operand) ||
      operation.getNumResults() != 1)
    return operation.emitOpError("lacks a TileLang reduction binding");
  if (isa<RankedTensorType>(operation.getResult(0).getType())) {
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
    if (failed(result))
      return failure();
    line(binding.getLowering().str() + "(" + operand->str() + ", " + *result +
         ", dim=" + std::to_string(binding.getAxis()) + ", clear=True)");
    bindResult(operation, 0, *result);
    return success();
  }
  std::string dtype = dtypeName(operation.getResult(0).getType(), operation);
  if (dtype.empty())
    return failure();
  std::string result = makeResultName(operation, 0) + "_fragment";
  line(result + " = T.alloc_fragment((1,), " + dtype + ")");
  line(binding.getLowering().str() + "(" + operand->str() + ", " + result +
       ", dim=" + std::to_string(binding.getAxis()) + ", clear=True)");
  valueNames[operation.getResult(0)] = result + "[0]";
  return success();
}

LogicalResult SourceEmitter::emitBroadcast(Operation &operation) {
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

LogicalResult SourceEmitter::emitUnary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "unary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(node) || !binding || failed(result) || failed(extents))
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
                               : binding.getLowering().str() + "(" + *operand + ")";
  std::string target = *result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + expression);
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitBinary(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "binary emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(node) || !binding || failed(result) || failed(extents))
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
  std::string expression;
  if (binding.getLowering() == "T.max")
    expression = "T.max(" + *lhs + ", " + *rhs + ")";
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
      return operation.emitOpError("uses an unsupported TileLang binary lowering");
    expression = *lhs + " " + symbol.str() + " " + *rhs;
  }
  std::string target = *result + "[";
  for (auto [axis, index] : llvm::enumerate(indices)) {
    if (axis)
      target += ", ";
    target += index;
  }
  line(target + "] = " + expression);
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitCast(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "cast emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  auto tensor = operation.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                    : RankedTensorType();
  if (failed(node) || !binding || binding.getLowering() != "T.cast" ||
      failed(result) || failed(extents) || !tensor)
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
  line(target + "] = T.cast(" + *operand + ", " + dtype + ")");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitFull(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "full emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  FailureOr<StringRef> fill = lookupValue(operation, 0);
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(node) || !binding || binding.getLowering() != "T.fill" ||
      failed(fill) || failed(result))
    return operation.emitOpError("lacks a TileLang full binding");
  line("T.fill(" + *result + ", " + fill->str() + ")");
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitZeros(Operation &operation) {
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
  if (failed(node) || !binding || failed(relation) || !validIndex || !fillIndex)
    return operation.emitOpError("lacks a TileLang gather binding");
  if (isRaggedStages() && binding.getLowering() == "T.indirect_gather") {
    bool feedsContract = llvm::any_of(operation.getResult(0).getUsers(),
                                     [](Operation *user) {
                                       return user->getName().getStringRef() ==
                                              "intent.contract";
                                     });
    FailureOr<ABIView *> view = lookupView(operation.getOperand(0), operation);
    if (failed(view))
      return failure();
    if (feedsContract && (*view)->tensor.getRank() == 2) {
      deferredLoads[operation.getResult(0)] = &operation;
      return success();
    }
    if ((*view)->tensor.getRank() != 1 || relation->size() != 1 ||
        (*relation)[0].kind != "value_index" ||
        (*relation)[0].operands.size() != 1 ||
        !(*relation)[0].operands.front())
      return operation.emitOpError(
          "staged indirect gather requires one indexed vector source");
    FailureOr<StringRef> indices =
        lookupValue(operation, *(*relation)[0].operands.front());
    FailureOr<StringRef> valid = lookupValue(operation, validIndex.getInt());
    FailureOr<StringRef> fill = lookupValue(operation, fillIndex.getInt());
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
    if (failed(indices) || failed(valid) || failed(fill) || failed(result))
      return failure();
    line("for gather_i in T.Parallel(TILE_SIZE_M):");
    ++indentation;
    line(*result + "[gather_i] = T.if_then_else(member_start + gather_i < "
         "route_end and " + valid->str() + ", " +
         (*view)->argument->name + "[" + indices->str() + "[gather_i]], " +
         fill->str() + ")");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }
  if ((binding.getLowering() != "expand_dims" &&
       binding.getLowering() != "alias_column") ||
      relation->size() != 2 || (*relation)[0].kind != "full_slice" ||
      (*relation)[1].kind != "new_axis")
    return operation.emitOpError("has no mechanical TileLang gather relation");
  FailureOr<StringRef> source = lookupValue(operation, 0);
  FailureOr<StringRef> valid = lookupValue(operation, validIndex.getInt());
  FailureOr<StringRef> fill = lookupValue(operation, fillIndex.getInt());
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  FailureOr<SmallVector<std::string>> extents = tensorExtents(operation, 0);
  if (failed(source) || failed(valid) || failed(fill) || failed(result) ||
      failed(extents) || extents->size() != 2)
    return failure();
  auto sourceTensor = dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
  line("for gather_i, gather_j in T.Parallel(" + (*extents)[0] + ", " +
       (*extents)[1] + "):");
  ++indentation;
  std::string sourceElement = sourceTensor && sourceTensor.getRank() == 1
                                  ? source->str() + "[gather_i]"
                                  : sourceTensor
                                        ? source->str() + "[gather_i, gather_j]"
                                        : source->str();
  line(*result + "[gather_i, gather_j] = T.if_then_else(" + valid->str() +
       ", " + sourceElement + ", " + fill->str() + ")");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::emitMembers(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "members emission");
  plan::PointwiseOp binding =
      succeeded(node) ? planIndex.pointwise.lookup(*node) : plan::PointwiseOp();
  if (failed(node) || !binding || binding.getLowering() != "T.members")
    return operation.emitOpError("lacks a staged TileLang members binding");
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(result))
    return failure();
  line("for member_i in T.Parallel(TILE_SIZE_M):");
  ++indentation;
  line(*result + "[member_i] = T.if_then_else(member_start + member_i < "
       "route_end, member_routes[member_start + member_i], 0)");
  --indentation;
  bindResult(operation, 0, *result);
  return success();
}

LogicalResult SourceEmitter::enterStateStream(Operation &operation) {
  FailureOr<int64_t> node = target::getNodeID(operation, "stream emission");
  plan::StreamOp binding =
      succeeded(node) ? planIndex.streams.lookup(*node) : plan::StreamOp();
  if (failed(node) || !binding || binding.getOrder() != "forward" ||
      binding.getCarrySpace() != "fragment" || operation.getNumRegions() != 1 ||
      !llvm::hasSingleElement(operation.getRegion(0)))
    return operation.emitOpError("lacks a mechanical TileLang stream binding");
  Block &body = operation.getRegion(0).front();
  if (operation.getNumOperands() != operation.getNumResults() + 1 ||
      body.getNumArguments() != operation.getNumResults() + 1)
    return operation.emitOpError("has inconsistent TileLang stream state");
  SmallVector<std::string> carriers;
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> initial = lookupValue(operation, index + 1);
    FailureOr<std::string> carrier = allocateResult(operation, index, "fragment");
    if (failed(initial) || failed(carrier))
      return failure();
    line("T.copy(" + initial->str() + ", " + *carrier + ")");
    carriers.push_back(*carrier);
    valueNames[body.getArgument(index + 1)] = *carrier;
  }
  streamCarriers[&operation] = carriers;
  line("for stream_tile in T.Pipelined(T.ceildiv(K, TILE_SIZE_N), "
       "num_stages=num_stages):");
  ++indentation;
  valueNames[body.getArgument(0)] = "stream_tile";
  return success();
}

LogicalResult SourceEmitter::leaveStateStream(Operation &operation) {
  auto carriers = streamCarriers.find(&operation);
  if (carriers == streamCarriers.end())
    return operation.emitOpError("has no active TileLang stream state");
  Operation &terminator = operation.getRegion(0).front().back();
  if (terminator.getName().getStringRef() != "intent.yield" ||
      terminator.getNumOperands() != operation.getNumResults())
    return operation.emitOpError("does not yield every TileLang stream state");
  for (unsigned index = 0; index < operation.getNumResults(); ++index) {
    FailureOr<StringRef> yielded = lookupValue(terminator, index);
    if (failed(yielded))
      return failure();
    line("T.copy(" + yielded->str() + ", " + carriers->second[index] + ")");
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
  if (failed(node) || !binding || binding.getLowering() != "T.gemm" ||
      operation.getNumOperands() != 2)
    return operation.emitOpError("lacks a TileLang contraction binding");
  if (isRaggedStages()) {
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
    std::string reduction = stageReductionDimensions.lookup(stage);
    std::string lhsDtype = lhsAccess ? "T.float16" : "T.float32";
    std::string rhsDtype = lhsAccess ? "T.float16" : "T.float32";
    std::string lhs = makeResultName(operation, 0) + "_lhs";
    std::string rhs = makeResultName(*rhsLoad, 0) + "_shared";
    std::string result = makeResultName(operation, 0);
    line(lhs + " = T.alloc_shared((TILE_SIZE_M, TILE_SIZE_K), " +
         lhsDtype + ")");
    line(rhs + " = T.alloc_shared((TILE_SIZE_K, TILE_SIZE_N), " +
         rhsDtype + ")");
    line(result +
         " = T.alloc_fragment((TILE_SIZE_M, TILE_SIZE_N), T.float32)");
    line("T.clear(" + result + ")");
    line("for k_tile in T.Pipelined(T.ceildiv(" + reduction +
         ", TILE_SIZE_K), num_stages=num_stages):");
    ++indentation;
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
      line("for load_i, load_k in T.Parallel(TILE_SIZE_M, TILE_SIZE_K):");
      ++indentation;
      line(lhs + "[load_i, load_k] = T.if_then_else(member_start + load_i < "
           "route_end and k_tile * TILE_SIZE_K + load_k < " + reduction +
           ", " + (*lhsView)->argument->name + "[" + rows->str() +
           "[load_i], k_tile * TILE_SIZE_K + load_k], 0.0)");
      --indentation;
    } else {
      auto workspace = workspaceNames.find(operation.getOperand(0));
      if (workspace == workspaceNames.end())
        return operation.emitOpError(
            "staged contraction input has no materialized workspace");
      line("for load_i, load_k in T.Parallel(TILE_SIZE_M, TILE_SIZE_K):");
      ++indentation;
      line(lhs + "[load_i, load_k] = T.if_then_else(member_start + load_i < "
           "route_end and k_tile * TILE_SIZE_K + load_k < " + reduction +
           ", " + workspace->second +
           "[member_start + load_i, k_tile * TILE_SIZE_K + load_k], 0.0)");
      --indentation;
    }
    if (lhsAccess) {
      line("T.copy(" + (*rhsView)->argument->name +
           "[expert, k_tile * TILE_SIZE_K, bid_feature * TILE_SIZE_N], " +
           rhs + ")");
    } else {
      line("for load_k, load_j in T.Parallel(TILE_SIZE_K, TILE_SIZE_N):");
      ++indentation;
      line(rhs + "[load_k, load_j] = T.if_then_else(k_tile * TILE_SIZE_K + "
           "load_k < " + reduction + " and bid_feature * TILE_SIZE_N + "
           "load_j < " + stageFeatureDimensions.lookup(stage) + ", T.cast(" +
           (*rhsView)->argument->name +
           "[expert, k_tile * TILE_SIZE_K + load_k, bid_feature * "
           "TILE_SIZE_N + load_j], T.float32), 0.0)");
      --indentation;
    }
    line("T.gemm(" + lhs + ", " + rhs + ", " + result + ")");
    --indentation;
    bindResult(operation, 0, result);
    return success();
  }

  Operation *lhsLoad = deferredLoads.lookup(operation.getOperand(0));
  Operation *rhsLoad = deferredLoads.lookup(operation.getOperand(1));
  if (lhsLoad || rhsLoad) {
    if (!lhsLoad || !rhsLoad)
      return operation.emitOpError(
          "deferred TileLang contraction has inconsistent operand residency");
    FailureOr<ABIView *> lhsView = lookupView(lhsLoad->getOperand(0), *lhsLoad);
    FailureOr<ABIView *> rhsView = lookupView(rhsLoad->getOperand(0), *rhsLoad);
    FailureOr<std::string> lhsIndices = accessIndices(*lhsLoad, true);
    FailureOr<std::string> rhsIndices = accessIndices(*rhsLoad, true);
    FailureOr<std::string> lhsShape = tensorShape(*lhsLoad, 0);
    FailureOr<std::string> rhsShape = tensorShape(*rhsLoad, 0);
    FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
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
    line("T.clear(" + *result + ")");
    line("for k_tile in T.Pipelined(T.ceildiv(K, TILE_SIZE_K), "
         "num_stages=num_stages):");
    ++indentation;
    line("T.copy(" + (*lhsView)->argument->name + "[" + *lhsIndices + "], " +
         lhs + ")");
    line("T.copy(" + (*rhsView)->argument->name + "[" + *rhsIndices + "], " +
         rhs + ")");
    std::string call = "T.gemm(" + lhs + ", " + rhs + ", " + *result;
    if (binding.getLhsTranspose())
      call += ", transpose_A=True";
    if (binding.getRhsTranspose())
      call += ", transpose_B=True";
    line(call + ")");
    --indentation;
    bindResult(operation, 0, *result);
    return success();
  }

  FailureOr<StringRef> lhs = lookupValue(operation, 0);
  FailureOr<StringRef> rhs = lookupValue(operation, 1);
  FailureOr<std::string> result = allocateResult(operation, 0, "fragment");
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  line("T.clear(" + *result + ")");
  std::string call = "T.gemm(" + lhs->str() + ", " + rhs->str() + ", " +
                     *result;
  if (binding.getLhsTranspose())
    call += ", transpose_A=True";
  if (binding.getRhsTranspose())
    call += ", transpose_B=True";
  line(call + ")");
  bindResult(operation, 0, *result);
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
  FailureOr<std::string> indices = accessIndices(operation, false);
  if (!valueIndex || failed(node) || !boundary || failed(stored) ||
      failed(view) || failed(indices))
    return operation.emitOpError("lacks a TileLang store binding");
  line("T.copy(" + stored->str() + ", " + (*view)->argument->name + "[" +
       *indices + "])");
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
  if (failed(node) || !binding || binding.getLowering() != "T.atomic_add" ||
      !valueIndex || failed(relation) || relation->size() != 2 ||
      (*relation)[0].kind != "value_index" ||
      (*relation)[0].operands.size() != 1 ||
      !(*relation)[0].operands.front() ||
      (*relation)[1].kind != "full_slice" || failed(view) || failed(stored))
    return operation.emitOpError("lacks a mechanical TileLang atomic merge");
  FailureOr<StringRef> rows =
      lookupValue(operation, *(*relation)[0].operands.front());
  if (failed(rows))
    return failure();
  line("for atomic_i, atomic_j in T.Parallel(TILE_SIZE_M, TILE_SIZE_N):");
  ++indentation;
  line("if member_start + atomic_i < route_end and bid_feature * "
       "TILE_SIZE_N + atomic_j < " + stageFeatureDimensions.lookup(activeStages.front()) +
       ":");
  ++indentation;
  line("T.atomic_add(" + (*view)->argument->name + "[" + rows->str() +
       "[atomic_i], bid_feature * TILE_SIZE_N + atomic_j], " + stored->str() +
       "[atomic_i, atomic_j])");
  --indentation;
  --indentation;
  return success();
}

} // namespace intent::tilelang::emission

#include "Support/Model.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu::realization {
namespace {

LogicalResult addHandler(target::OperationHandlerRegistry &registry,
                         StringRef name, target::OperationCallback enter) {
  return registry.add(name,
                      target::OperationHandler{std::move(enter), {}});
}

LogicalResult validateReduction(Operation &operation) {
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
  if (!combine || !axes || axes.size() != 1 ||
      !isa<IntegerAttr>(axes[0]))
    return operation.emitOpError("has no canonical single-axis reduction");
  if (operation.getName().getStringRef() == "intent.arg_reduce") {
    auto tie = operation.getAttrOfType<StringAttr>("intent.tie");
    if (operation.getNumResults() == 2 && combine.getValue() == "maximum" &&
        tie && tie.getValue() == "lowest_index")
      return success();
    return operation.emitOpError("has no supported GPU arg-reduction role");
  }
  if (combine.getValue() == "maximum" || combine.getValue() == "add")
    return success();
  return operation.emitOpError("has no supported GPU reduction role");
}

bool isLiteralZero(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition ||
      definition->getName().getStringRef() != "intent.constant")
    return false;
  Attribute literal = definition->getAttr("intent.value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(literal))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast_or_null<FloatAttr>(literal))
    return floating.getValue().isZero();
  return false;
}

LogicalResult validateScan(Operation &operation) {
  auto input = operation.getNumOperands() == 2
                   ? dyn_cast<RankedTensorType>(operation.getOperand(0).getType())
                   : RankedTensorType();
  auto result = operation.getNumResults() == 1
                    ? dyn_cast<RankedTensorType>(operation.getResult(0).getType())
                    : RankedTensorType();
  auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto inclusive = operation.getAttrOfType<BoolAttr>("intent.inclusive");
  if (!input || !result || input.getShape() != result.getShape() || !axis ||
      axis.getInt() < 0 || axis.getInt() >= input.getRank() || !combine ||
      combine.getValue() != "add" || !inclusive || !inclusive.getValue())
    return operation.emitOpError(
        "has no supported inclusive additive GPU scan schema");
  if (!isLiteralZero(operation.getOperand(1)))
    return operation.emitOpError(
        "requires a literal zero identity for inclusive additive GPU scan");
  return success();
}

LogicalResult validatePointwise(Operation &operation) {
  StringRef name = operation.getName().getStringRef();
  if (name == "intent.random") {
    auto algorithm =
        operation.getAttrOfType<StringAttr>("intent.algorithm");
    auto elementType = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        return tensor.getElementType();
      return type;
    };
    if (operation.getNumOperands() == 2 && operation.getNumResults() == 1 &&
        algorithm && algorithm.getValue() == "counter_xorshift32" &&
        isa<IntegerType, IndexType>(
            elementType(operation.getOperand(0).getType())) &&
        isa<IntegerType, IndexType, intent::LogicalIndexType>(
            elementType(operation.getOperand(1).getType())) &&
        elementType(operation.getResult(0).getType()).isF32())
      return success();
    return operation.emitOpError("has no supported GPU counter RNG schema");
  }
  if (name == "intent.indices" || name == "intent.broadcast" ||
      name == "intent.mask")
    return success();
  if (name == "intent.reshape") {
    if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
        !isa<RankedTensorType>(operation.getOperand(0).getType()) ||
        !isa<RankedTensorType>(operation.getResult(0).getType()))
      return operation.emitOpError("has no canonical tensor reshape schema");
    return success();
  }
  if (name == "intent.cast") {
    if (operation.getNumOperands() != 1 || operation.getNumResults() != 1)
      return operation.emitOpError("has no canonical cast schema");
    auto elementType = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        return tensor.getElementType();
      return type;
    };
    Type source = elementType(operation.getOperand(0).getType());
    Type result = elementType(operation.getResult(0).getType());
    auto rounding = operation.getAttrOfType<StringAttr>("intent.rounding");
    bool floatToInteger = isa<FloatType>(source) && isa<IntegerType>(result);
    if (floatToInteger &&
        (!rounding || rounding.getValue() != "toward_zero"))
      return operation.emitOpError(
          "float-to-integer cast requires toward-zero rounding semantics");
    if (!floatToInteger && rounding)
      return operation.emitOpError(
          "rounding semantics only apply to float-to-integer casts");
    return success();
  }
  if (name == "intent.compare") {
    auto predicate =
        operation.getAttrOfType<StringAttr>("intent.predicate");
    if (predicate &&
        llvm::is_contained({StringRef("eq"), StringRef("ne"), StringRef("lt"),
                            StringRef("le"), StringRef("gt"), StringRef("ge")},
                           predicate.getValue()))
      return success();
    return operation.emitOpError("has no supported GPU comparison predicate");
  }
  auto logical = operation.getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return operation.emitOpError("has no canonical pointwise operator");
  if (name == "intent.unary" &&
      llvm::is_contained({StringRef("exp"), StringRef("exp2"),
                          StringRef("log"), StringRef("rsqrt"),
                          StringRef("sigmoid"), StringRef("negate")},
                         logical.getValue()))
    return success();
  if (name == "intent.binary" &&
      llvm::is_contained({StringRef("add"), StringRef("subtract"),
                          StringRef("multiply"), StringRef("true_divide"),
                          StringRef("floor_divide"), StringRef("remainder"),
                          StringRef("maximum"), StringRef("minimum")},
                         logical.getValue()))
    return success();
  return operation.emitOpError("has no supported GPU pointwise role");
}

LogicalResult registerHandlers(target::OperationHandlerRegistry &registry) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.region_end",
                         "intent.assume_in_bounds",
                         "intent.partition", "intent.parallel",
                         "intent.view_load", "intent.view_store", "intent.yield",
                         "intent.return", "intent.state_stream", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member",
                         "intent.scatter_unique"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.reduce", validateReduction)))
    return failure();
  if (failed(addHandler(
          registry, "intent.arg_reduce", validateReduction)))
    return failure();
  if (failed(addHandler(registry, "intent.scan", validateScan)))
    return failure();

  for (StringRef name : {"intent.indices", "intent.broadcast", "intent.unary",
                         "intent.binary", "intent.compare", "intent.mask",
                         "intent.cast", "intent.reshape", "intent.random"})
    if (failed(addHandler(
            registry, name, validatePointwise)))
      return failure();

  for (StringRef name : {"intent.full", "intent.zeros", "intent.members"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.gather", [&](Operation &operation) -> LogicalResult {
            FailureOr<SmallVector<target::IndexTerm>> relation =
                target::parseIndexRelation(operation);
            if (failed(relation))
              return failure();
            bool expand =
                relation->size() == 2 &&
                (((*relation)[0].kind == "full_slice" &&
                  (*relation)[1].kind == "new_axis") ||
                 ((*relation)[0].kind == "new_axis" &&
                  (*relation)[1].kind == "full_slice"));
            bool indirect = llvm::any_of(
                *relation, [](const target::IndexTerm &term) {
                  return term.kind == "value_index";
                });
            if (!expand && !indirect)
              return operation.emitOpError(
                  "has no mechanical GPU gather realization");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.scatter_reduce",
          [&](Operation &operation) -> LogicalResult {
            auto combine =
                operation.getAttrOfType<StringAttr>("intent.combine");
            auto ordering =
                operation.getAttrOfType<StringAttr>("intent.ordering");
            auto scope = operation.getAttrOfType<StringAttr>("intent.scope");
            if (!combine || combine.getValue() != "add" ||
                (ordering && ordering.getValue() != "relaxed") ||
                (scope && scope.getValue() != "device"))
              return operation.emitOpError(
                  "has no relaxed device-scoped additive GPU merge");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.contract", [&](Operation &operation) -> LogicalResult {
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto accType =
                operation.getAttrOfType<StringAttr>("intent.acc_dtype");
            auto multiply =
                operation.getAttrOfType<StringAttr>("intent.multiply");
            auto combine =
                operation.getAttrOfType<StringAttr>("intent.combine");
            auto pair = reduce && reduce.size() == 1
                            ? dyn_cast<ArrayAttr>(reduce[0])
                            : ArrayAttr();
            auto lhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[0])
                               : IntegerAttr();
            auto rhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[1])
                               : IntegerAttr();
            auto lhsType = operation.getNumOperands() == 2
                               ? dyn_cast<RankedTensorType>(
                                     operation.getOperand(0).getType())
                               : RankedTensorType();
            auto rhsType = operation.getNumOperands() == 2
                               ? dyn_cast<RankedTensorType>(
                                     operation.getOperand(1).getType())
                               : RankedTensorType();
            auto resultType = operation.getNumResults() == 1
                                  ? dyn_cast<RankedTensorType>(
                                        operation.getResult(0).getType())
                                  : RankedTensorType();
            if (operation.getNumOperands() != 2 ||
                operation.getNumResults() != 1 || !accType ||
                accType.getValue() != "f32" || !multiply ||
                multiply.getValue() != "multiply" || !combine ||
                combine.getValue() != "add" || !lhsAxis || !rhsAxis ||
                !lhsType || lhsType.getRank() != 2 || !rhsType ||
                rhsType.getRank() != 2 || !resultType ||
                resultType.getRank() != 2 ||
                (lhsAxis.getInt() != 0 && lhsAxis.getInt() != 1) ||
                (rhsAxis.getInt() != 0 && rhsAxis.getInt() != 1))
              return operation.emitOpError(
                  "has no semantics-preserving GPU matrix-unit realization");
            return success();
          })))
    return failure();
  return success();
}

} // namespace

LogicalResult analyzeOperations(KernelFacts &facts) {
  if (failed(target::analyzeKernelFacts(facts)))
    return failure();
  target::OperationHandlerRegistry registry;
  if (failed(registerHandlers(registry)))
    return facts.kernel.entry.emitOpError(
        "failed to construct GPU operation handlers");
  return target::traverseKernel(facts.kernel.entry, registry,
                                "GPU realization analysis");
}

} // namespace intent::gpu::realization

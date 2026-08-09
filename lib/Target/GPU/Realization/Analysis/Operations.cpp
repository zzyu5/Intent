#include "Support/Model.h"

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
  if (combine.getValue() == "maximum" || combine.getValue() == "add")
    return success();
  return operation.emitOpError("has no supported GPU reduction role");
}

LogicalResult validatePointwise(Operation &operation) {
  StringRef name = operation.getName().getStringRef();
  if (name == "intent.indices" || name == "intent.broadcast" ||
      name == "intent.mask")
    return success();
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
    if (predicate && predicate.getValue() == "ge")
      return success();
    return operation.emitOpError("has no supported GPU comparison predicate");
  }
  auto logical = operation.getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return operation.emitOpError("has no canonical pointwise operator");
  if (name == "intent.unary" &&
      llvm::is_contained({StringRef("exp"), StringRef("exp2"),
                          StringRef("log"), StringRef("rsqrt"),
                          StringRef("negate")},
                         logical.getValue()))
    return success();
  if (name == "intent.binary" &&
      llvm::is_contained({StringRef("add"), StringRef("subtract"),
                          StringRef("multiply"), StringRef("true_divide"),
                          StringRef("maximum"), StringRef("minimum")},
                         logical.getValue()))
    return success();
  return operation.emitOpError("has no supported GPU pointwise role");
}

LogicalResult registerHandlers(target::OperationHandlerRegistry &registry) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.region_end",
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

  for (StringRef name : {"intent.indices", "intent.broadcast", "intent.unary",
                         "intent.binary", "intent.compare", "intent.mask",
                         "intent.cast"})
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
            if (operation.getNumOperands() != 2 ||
                operation.getNumResults() != 1 || !accType ||
                accType.getValue() != "f32" || !multiply ||
                multiply.getValue() != "multiply" || !combine ||
                combine.getValue() != "add" || !lhsAxis || !rhsAxis ||
                lhsAxis.getInt() != 1 ||
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

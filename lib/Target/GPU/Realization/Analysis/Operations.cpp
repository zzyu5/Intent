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

FailureOr<std::string> reductionRole(Operation &operation) {
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
  if (!combine || !axes || axes.size() != 1 ||
      !isa<IntegerAttr>(axes[0]))
    return operation.emitOpError("has no canonical single-axis reduction");
  if (combine.getValue() == "maximum")
    return std::string("reduce_maximum");
  if (combine.getValue() == "add")
    return std::string("reduce_add");
  return operation.emitOpError("has no supported GPU reduction role");
}

FailureOr<std::string> pointwiseRole(Operation &operation) {
  StringRef name = operation.getName().getStringRef();
  if (name == "intent.broadcast")
    return std::string("broadcast");
  if (name == "intent.cast")
    return std::string("cast");
  auto logical = operation.getAttrOfType<StringAttr>("intent.operator");
  if (!logical)
    return operation.emitOpError("has no canonical pointwise operator");
  if (name == "intent.unary" &&
      llvm::is_contained({StringRef("exp"), StringRef("exp2"),
                          StringRef("log"), StringRef("rsqrt"),
                          StringRef("negate")},
                         logical.getValue()))
    return ("unary_" + logical.getValue()).str();
  if (name == "intent.binary" &&
      llvm::is_contained({StringRef("add"), StringRef("subtract"),
                          StringRef("multiply"), StringRef("true_divide"),
                          StringRef("maximum")},
                         logical.getValue()))
    return ("binary_" + logical.getValue()).str();
  return operation.emitOpError("has no supported GPU pointwise role");
}

LogicalResult registerHandlers(target::OperationHandlerRegistry &registry,
                               OperationFacts &facts) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.partition", "intent.parallel",
                         "intent.view_load", "intent.view_store", "intent.yield",
                         "intent.return", "intent.state_stream", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member",
                         "intent.scatter_unique"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.reduce", [&](Operation &operation) -> LogicalResult {
            FailureOr<std::string> role = reductionRole(operation);
            if (failed(role))
              return failure();
            facts.primitiveRoles[&operation] = std::move(*role);
            return success();
          })))
    return failure();

  for (StringRef name : {"intent.broadcast", "intent.unary", "intent.binary",
                         "intent.cast"})
    if (failed(addHandler(
            registry, name, [&](Operation &operation) -> LogicalResult {
              FailureOr<std::string> role = pointwiseRole(operation);
              if (failed(role))
                return failure();
              facts.primitiveRoles[&operation] = std::move(*role);
              return success();
            })))
      return failure();

  for (auto [name, role] :
       {std::pair<StringRef, StringRef>{"intent.full", "full"},
        {"intent.zeros", "zeros"}, {"intent.members", "members"}})
    if (failed(addHandler(
            registry, name,
            [&, role](Operation &operation) -> LogicalResult {
              facts.primitiveRoles[&operation] = role.str();
              return success();
            })))
      return failure();

  if (failed(addHandler(
          registry, "intent.gather", [&](Operation &operation) -> LogicalResult {
            FailureOr<SmallVector<target::IndexTerm>> relation =
                target::parseIndexRelation(operation);
            if (failed(relation))
              return failure();
            bool expand = relation->size() == 2 &&
                          (*relation)[0].kind == "full_slice" &&
                          (*relation)[1].kind == "new_axis";
            bool indirect = llvm::any_of(
                *relation, [](const target::IndexTerm &term) {
                  return term.kind == "value_index";
                });
            if (!expand && !indirect)
              return operation.emitOpError(
                  "has no mechanical GPU gather realization");
            facts.primitiveRoles[&operation] =
                expand ? "expand_dims" : "indirect_gather";
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
            facts.primitiveRoles[&operation] = "atomic_add";
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
            facts.primitiveRoles[&operation] = "matrix_multiply";
            return success();
          })))
    return failure();
  return success();
}

} // namespace

LogicalResult analyzeOperations(OperationFacts &facts) {
  if (failed(target::analyzeKernelFacts(facts.semantics)))
    return failure();
  target::OperationHandlerRegistry registry;
  if (failed(registerHandlers(registry, facts)))
    return facts.semantics.kernel.entry.emitOpError(
        "failed to construct GPU operation handlers");
  return target::traverseKernel(facts.semantics.kernel.entry, registry,
                                "GPU realization analysis");
}

} // namespace intent::gpu::realization

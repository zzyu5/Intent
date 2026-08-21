#include "Support/Model.h"

#include "Intent/Target/GPU/Realization/Analysis.h"

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Target/Common/Analysis/IndexRelation.h"
#include "Intent/Target/Common/Analysis/Operation.h"
#include "Intent/Target/Common/Analysis/Record.h"
#include "Intent/Target/Common/Lowering/Combiner.h"
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

bool isLiteralBool(Value value, bool expected) {
  Operation *definition = value.getDefiningOp();
  if (!value.getType().isInteger(1) || !definition ||
      ::intent::target::semanticOperationName(*definition) != "intent.constant")
    return false;
  auto literal = definition->getAttrOfType<IntegerAttr>("intent.value");
  return literal && (!literal.getValue().isZero()) == expected;
}

LogicalResult validateReduction(Operation &operation) {
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto axes = operation.getAttrOfType<ArrayAttr>("intent.axes");
  if (!axes || axes.size() != 1 || !isa<IntegerAttr>(axes[0]))
    return operation.emitOpError("has no canonical single-axis reduction");
  if (target::lowering::hasGenericCombiner(operation))
    return success();
  if (!combine)
    return operation.emitOpError("has no canonical built-in reduction combiner");
  if (combine.getValue() == "maximum" || combine.getValue() == "add")
    return success();
  if (combine.getValue() == "logical_or" ||
      combine.getValue() == "logical_and") {
    auto input = operation.getNumOperands() == 2
                     ? dyn_cast<RankedTensorType>(
                           operation.getOperand(0).getType())
                     : RankedTensorType();
    Type result = operation.getNumResults() == 1
                      ? operation.getResult(0).getType()
                      : Type();
    Type resultElement = result;
    if (auto tensor = dyn_cast<RankedTensorType>(result))
      resultElement = tensor.getElementType();
    bool identity = combine.getValue() == "logical_and";
    if (input && input.getElementType().isInteger(1) && resultElement &&
        resultElement.isInteger(1) &&
        isLiteralBool(operation.getOperand(1), identity))
      return success();
    return operation.emitOpError(
        "logical GPU reduction requires bool values and its bool identity");
  }
  return operation.emitOpError("has no supported GPU reduction role");
}

bool isLiteralZero(Value value) {
  Operation *definition = value.getDefiningOp();
  while (definition) {
    if (::intent::target::semanticOperationName(*definition) == "intent.cast" &&
        definition->getNumOperands() == 1)
      value = definition->getOperand(0);
    else if (::intent::target::semanticOperationName(*definition) == "intent.extract") {
      FailureOr<Value> field = target::resolveRecordField(*definition);
      if (failed(field))
        return false;
      value = *field;
    } else
      break;
    definition = value.getDefiningOp();
  }
  if (!definition ||
      ::intent::target::semanticOperationName(*definition) != "intent.constant")
    return false;
  Attribute literal = definition->getAttr("intent.value");
  if (auto integer = dyn_cast_or_null<IntegerAttr>(literal))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast_or_null<FloatAttr>(literal))
    return floating.getValue().isZero();
  return false;
}

LogicalResult validateScan(Operation &operation) {
  if (target::lowering::hasGenericCombiner(operation)) {
    auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
    auto inclusive = operation.getAttrOfType<BoolAttr>("intent.inclusive");
    if (!axis || axis.getInt() < 0 || !inclusive || !inclusive.getValue())
      return operation.emitOpError(
          "generic GPU scan requires an inclusive logical prefix");
    return success();
  }
  auto components =
      operation.getAttrOfType<IntegerAttr>("intent.component_count");
  auto captures =
      operation.getAttrOfType<IntegerAttr>("intent.capture_count");
  auto axis = operation.getAttrOfType<IntegerAttr>("intent.axis");
  auto combine = operation.getAttrOfType<StringAttr>("intent.combine");
  auto inclusive = operation.getAttrOfType<BoolAttr>("intent.inclusive");
  if (!components || components.getInt() <= 0 || !captures ||
      captures.getInt() != 0 ||
      operation.getNumOperands() !=
          static_cast<unsigned>(2 * components.getInt()) ||
      operation.getNumResults() != static_cast<unsigned>(components.getInt()) ||
      !axis || !combine || combine.getValue() != "add" || !inclusive ||
      !inclusive.getValue())
    return operation.emitOpError(
        "has no supported inclusive additive GPU scan schema");
  for (unsigned component = 0;
       component < static_cast<unsigned>(components.getInt()); ++component) {
    auto input =
        dyn_cast<RankedTensorType>(operation.getOperand(component).getType());
    auto result = dyn_cast<RankedTensorType>(
        operation.getResult(component).getType());
    Value identity = operation.getOperand(components.getInt() + component);
    if (!input || !result || input != result || axis.getInt() < 0 ||
        axis.getInt() >= input.getRank() ||
        input.getElementType() != identity.getType() ||
        !isLiteralZero(identity))
      return operation.emitOpError(
          "requires shape-preserving components with literal zero identities for inclusive additive GPU scan");
  }
  return success();
}

LogicalResult validatePointwise(Operation &operation) {
  StringRef name = ::intent::target::semanticOperationName(operation);
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
  if (name == "intent.select") {
    if (operation.getNumOperands() != 3 || operation.getNumResults() != 1 ||
        operation.getOperand(1).getType() != operation.getOperand(2).getType() ||
        operation.getOperand(1).getType() != operation.getResult(0).getType())
      return operation.emitOpError("has no canonical select value schema");
    Type condition = operation.getOperand(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(condition)) {
      auto result = dyn_cast<RankedTensorType>(operation.getResult(0).getType());
      if (!result || !tensor.getElementType().isInteger(1) ||
          tensor.getShape() != result.getShape())
        return operation.emitOpError("has no canonical tensor select condition");
      return success();
    }
    if (!condition.isInteger(1) ||
        isa<RankedTensorType>(operation.getResult(0).getType()))
      return operation.emitOpError("has no canonical scalar select condition");
    return success();
  }
  if (name == "intent.reshape") {
    if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
        !isa<RankedTensorType>(operation.getOperand(0).getType()) ||
        !isa<RankedTensorType>(operation.getResult(0).getType()))
      return operation.emitOpError("has no canonical tensor reshape schema");
    return success();
  }
  if (name == "intent.transpose") {
    auto source = operation.getNumOperands() == 1
                      ? dyn_cast<RankedTensorType>(
                            operation.getOperand(0).getType())
                      : RankedTensorType();
    auto result = operation.getNumResults() == 1
                      ? dyn_cast<RankedTensorType>(
                            operation.getResult(0).getType())
                      : RankedTensorType();
    auto permutation =
        operation.getAttrOfType<ArrayAttr>("intent.permutation");
    if (!source || !result || source.getRank() != result.getRank() ||
        source.getElementType() != result.getElementType() || !permutation ||
        permutation.size() != static_cast<size_t>(source.getRank()))
      return operation.emitOpError("has no canonical tensor transpose schema");
    SmallVector<bool> covered(source.getRank(), false);
    for (Attribute attribute : permutation) {
      auto axis = dyn_cast<IntegerAttr>(attribute);
      if (!axis || axis.getInt() < 0 || axis.getInt() >= source.getRank() ||
          covered[axis.getInt()])
        return operation.emitOpError(
            "transpose permutation must cover each source axis exactly once");
      covered[axis.getInt()] = true;
    }
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
                          StringRef("log"), StringRef("sin"), StringRef("cos"),
                          StringRef("floor"),
                          StringRef("rsqrt"),
                          StringRef("sigmoid"), StringRef("negate"),
                          StringRef("not")},
                         logical.getValue()))
    return success();
  if (name == "intent.binary" &&
      logical.getValue() == "power") {
    auto elementType = [](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        return tensor.getElementType();
      return type;
    };
    auto supported = [](Type type) {
      return type.isF16() || type.isBF16() || type.isF32() ||
             isa<Float8E4M3FNType, Float8E5M2Type>(type);
    };
    if (operation.getNumOperands() == 2 && operation.getNumResults() == 1 &&
        supported(elementType(operation.getOperand(0).getType())) &&
        supported(elementType(operation.getOperand(1).getType())) &&
        supported(elementType(operation.getResult(0).getType())))
      return success();
    return operation.emitOpError(
        "GPU power projection requires f8e4m3fn, f8e5m2, f16, bf16, or f32 operands and result");
  }
  if (name == "intent.binary" &&
      llvm::is_contained({StringRef("add"), StringRef("subtract"),
                          StringRef("multiply"), StringRef("true_divide"),
                          StringRef("floor_divide"), StringRef("remainder"),
                          StringRef("bitwise_and"), StringRef("bitwise_or"),
                          StringRef("bitwise_xor"), StringRef("left_shift"),
                          StringRef("right_shift"),
                          StringRef("logical_and"), StringRef("logical_or"),
                          StringRef("maximum"), StringRef("minimum")},
                         logical.getValue()))
    return success();
  return operation.emitOpError("has no supported GPU pointwise role");
}

LogicalResult registerHandlers(target::OperationHandlerRegistry &registry) {
  auto noOp = [](Operation &) { return success(); };
  for (StringRef name : {"intent.constant", "intent.dim", "intent.domain",
                         "intent.domain_product",
                         "intent.region_end",
                         "intent.assume_in_bounds",
                         "intent.partition", "intent.parallel",
                         "intent.for", "intent.if", "intent.while",
                         "intent.condition", "intent.buffer",
                         "intent.make_record", "intent.extract",
                         "intent.buffer_load", "intent.buffer_store",
                         "intent.view_load", "intent.view_store", "intent.yield",
                         "intent.return", "intent.state_stream", "intent.ragged",
                         "intent.ragged_outer", "intent.ragged_member",
                         "intent.scatter_unique"})
    if (failed(addHandler(registry, name, noOp)))
      return failure();

  if (failed(addHandler(
          registry, "intent.sparse_contract",
          [&](Operation &operation) -> LogicalResult {
            auto format = operation.getAttrOfType<StringAttr>("intent.format");
            if (!format || format.getValue() != "two_of_four" ||
                operation.getNumOperands() != 3 || operation.getNumResults() != 1)
              return operation.emitOpError(
                  "has no mechanical 2:4 sparse contraction schema");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.reduce", validateReduction)))
    return failure();
  if (failed(addHandler(registry, "intent.scan", validateScan)))
    return failure();

  for (StringRef name : {"intent.indices", "intent.broadcast", "intent.unary",
                         "intent.binary", "intent.compare",
                         "intent.mask", "intent.select", "intent.cast",
                         "intent.reshape", "intent.transpose", "intent.random"})
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
            bool hasNewAxis =
                llvm::any_of(*relation, [](const target::IndexTerm &term) {
                  return term.kind == "new_axis";
                });
            bool expand = hasNewAxis && llvm::all_of(
                                              *relation,
                                              [](const target::IndexTerm &term) {
                                                return term.kind == "full_slice" ||
                                                       term.kind == "new_axis";
                                              });
            bool indirect = llvm::any_of(
                *relation, [](const target::IndexTerm &term) {
                  return term.kind == "value_index";
                });
            auto sourceType = dyn_cast<RankedTensorType>(
                operation.getNumOperands() > 0
                    ? operation.getOperand(0).getType()
                    : Type());
            bool extractFirstScalar =
                operation.getNumOperands() > 0 && operation.getNumResults() == 1 &&
                sourceType && sourceType.getRank() == 1 &&
                !isa<RankedTensorType>(operation.getResult(0).getType()) &&
                relation->size() == 1 && relation->front().kind == "static_index" &&
                relation->front().staticValues.size() == 1 &&
                relation->front().staticValues.front() &&
                *relation->front().staticValues.front() == 0;
            if (!expand && !indirect && !extractFirstScalar)
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
          registry, "intent.atomic_add",
          [&](Operation &operation) -> LogicalResult {
            auto valueIndex = operation.getAttrOfType<IntegerAttr>(
                "intent.value_operand_index");
            auto ordering =
                operation.getAttrOfType<StringAttr>("intent.ordering");
            auto scope = operation.getAttrOfType<StringAttr>("intent.scope");
            if (operation.getNumResults() != 1 || !valueIndex ||
                valueIndex.getInt() <= 0 ||
                static_cast<unsigned>(valueIndex.getInt()) >=
                    operation.getNumOperands() ||
                !isa<intent::ViewType>(operation.getOperand(0).getType()) ||
                !ordering || ordering.getValue() != "relaxed" || !scope ||
                scope.getValue() != "device" ||
                operation.getResult(0).getType() !=
                    operation.getOperand(valueIndex.getInt()).getType())
              return operation.emitOpError(
                  "has no relaxed device-scoped external GPU atomic-add schema");
            return success();
          })))
    return failure();

  if (failed(addHandler(
          registry, "intent.atomic_cas",
          [&](Operation &operation) -> LogicalResult {
            auto compareIndex = operation.getAttrOfType<IntegerAttr>(
                "intent.compare_operand_index");
            auto valueIndex = operation.getAttrOfType<IntegerAttr>(
                "intent.value_operand_index");
            auto ordering =
                operation.getAttrOfType<StringAttr>("intent.ordering");
            auto scope = operation.getAttrOfType<StringAttr>("intent.scope");
            auto view = operation.getNumOperands() > 0
                            ? dyn_cast<intent::ViewType>(
                                  operation.getOperand(0).getType())
                            : intent::ViewType();
            auto tensor = view ? dyn_cast<RankedTensorType>(view.getTensor())
                               : RankedTensorType();
            bool supportedOrdering =
                ordering && llvm::is_contained(
                                {StringRef("relaxed"), StringRef("acquire"),
                                 StringRef("release"), StringRef("acq_rel")},
                                ordering.getValue());
            bool supportedScope =
                scope && llvm::is_contained(
                             {StringRef("workgroup"), StringRef("device"),
                              StringRef("system")},
                             scope.getValue());
            if (operation.getNumResults() != 1 || !compareIndex || !valueIndex ||
                compareIndex.getInt() <= 0 ||
                valueIndex.getInt() != compareIndex.getInt() + 1 ||
                static_cast<unsigned>(valueIndex.getInt()) !=
                    operation.getNumOperands() - 1 ||
                !view || view.getAccess() != "inout" || !tensor ||
                !tensor.getElementType().isInteger(32) ||
                operation.getOperand(compareIndex.getInt()).getType() !=
                    tensor.getElementType() ||
                operation.getOperand(valueIndex.getInt()).getType() !=
                    tensor.getElementType() ||
                operation.getResult(0).getType() != tensor.getElementType() ||
                !supportedOrdering || !supportedScope)
              return operation.emitOpError(
                  "has no supported scalar i32 external GPU compare-and-swap schema");
            return success();
          })))
    return failure();

  auto analyzeContraction = [&](Operation &operation) -> LogicalResult {
            bool scaled =
                ::intent::target::semanticOperationName(operation) == "intent.scaled_contract";
            unsigned operandCount = scaled ? 4 : 2;
            auto reduce = operation.getAttrOfType<ArrayAttr>("intent.reduce");
            auto batch = operation.getAttrOfType<ArrayAttr>("intent.batch");
            auto accType =
                operation.getAttrOfType<StringAttr>("intent.acc_dtype");
            auto multiply =
                operation.getAttrOfType<StringAttr>("intent.multiply");
            auto combine =
                operation.getAttrOfType<StringAttr>("intent.combine");
            auto pair = reduce && !reduce.empty()
                            ? dyn_cast<ArrayAttr>(reduce[0])
                            : ArrayAttr();
            auto lhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[0])
                               : IntegerAttr();
            auto rhsAxis = pair && pair.size() == 2
                               ? dyn_cast<IntegerAttr>(pair[1])
                               : IntegerAttr();
            auto batchPair = batch && batch.size() == 1
                                 ? dyn_cast<ArrayAttr>(batch[0])
                                 : ArrayAttr();
            auto lhsBatch = batchPair && batchPair.size() == 2
                                ? dyn_cast<IntegerAttr>(batchPair[0])
                                : IntegerAttr();
            auto rhsBatch = batchPair && batchPair.size() == 2
                                ? dyn_cast<IntegerAttr>(batchPair[1])
                                : IntegerAttr();
            auto lhsType = operation.getNumOperands() == operandCount
                               ? dyn_cast<RankedTensorType>(
                                     operation.getOperand(0).getType())
                               : RankedTensorType();
            auto rhsType = operation.getNumOperands() == operandCount
                               ? dyn_cast<RankedTensorType>(
                                     operation.getOperand(1).getType())
                               : RankedTensorType();
            auto resultType = operation.getNumResults() == 1
                                  ? dyn_cast<RankedTensorType>(
                                        operation.getResult(0).getType())
                                  : RankedTensorType();
            bool supportedAccumulator =
                accType && (accType.getValue() == "f32" ||
                            accType.getValue() == "i32");
            bool ordinary = !scaled && batch && batch.empty() && lhsType && rhsType &&
                            resultType && lhsType.getRank() == 2 &&
                            rhsType.getRank() == 2 && resultType.getRank() == 2;
            bool scaledGrouped =
                scaled && batch && batch.empty() && reduce && reduce.size() == 2 &&
                lhsType && rhsType && resultType && lhsType.getRank() == 3 &&
                rhsType.getRank() == 3 && resultType.getRank() == 2;
            bool scaledFlat =
                scaled && batch && batch.empty() && reduce && reduce.size() == 1 &&
                lhsType && rhsType && resultType && lhsType.getRank() == 2 &&
                rhsType.getRank() == 2 && resultType.getRank() == 2;
            bool batched = !scaled && batch && batch.size() == 1 && lhsBatch && rhsBatch &&
                           lhsType && rhsType && resultType &&
                           lhsType.getRank() == 3 && rhsType.getRank() == 3 &&
                           resultType.getRank() == 3 && lhsBatch.getInt() == 0 &&
                           rhsBatch.getInt() == 0 &&
                           (lhsAxis.getInt() == 1 || lhsAxis.getInt() == 2) &&
                           (rhsAxis.getInt() == 1 || rhsAxis.getInt() == 2);
            if (operation.getNumOperands() != operandCount ||
                operation.getNumResults() != 1 || !supportedAccumulator ||
                !multiply ||
                multiply.getValue() != "multiply" || !combine ||
                combine.getValue() != "add" || !lhsAxis || !rhsAxis ||
                (!ordinary && !batched && !scaledFlat && !scaledGrouped) ||
                (scaled && accType.getValue() != "f32") ||
                (accType.getValue() == "f32" &&
                 !resultType.getElementType().isF32()) ||
                (accType.getValue() == "i32" &&
                 !resultType.getElementType().isInteger(32)) ||
                (lhsAxis.getInt() != 0 && lhsAxis.getInt() != 1 &&
                 lhsAxis.getInt() != 2) ||
                (rhsAxis.getInt() != 0 && rhsAxis.getInt() != 1 &&
                 rhsAxis.getInt() != 2))
              return operation.emitOpError(
                  "has no semantics-preserving GPU matrix-unit realization");
            if (scaled) {
              auto lhsScale = dyn_cast<RankedTensorType>(
                  operation.getOperand(2).getType());
              auto rhsScale = dyn_cast<RankedTensorType>(
                  operation.getOperand(3).getType());
              auto lhsGroup = operation.getAttrOfType<IntegerAttr>(
                  "intent.lhs_group_size");
              auto rhsGroup = operation.getAttrOfType<IntegerAttr>(
                  "intent.rhs_group_size");
              if (!batch.empty() || !lhsScale || !rhsScale ||
                  lhsScale.getRank() != 2 || rhsScale.getRank() != 2 ||
                  !lhsGroup || !rhsGroup || lhsGroup.getInt() <= 0 ||
                  rhsGroup.getInt() <= 0)
                return operation.emitOpError(
                    "has no supported GPU scaled-matrix realization");
            }
            return success();
          };
  if (failed(addHandler(registry, "intent.contract", analyzeContraction)) ||
      failed(addHandler(registry, "intent.scaled_contract", analyzeContraction)))
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

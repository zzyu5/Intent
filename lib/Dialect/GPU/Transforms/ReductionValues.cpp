#include "ReductionValues.h"
#include "ReductionAnalysis.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

#include <tuple>

using namespace mlir;

namespace intent::gpu::reduction {

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, sourceSubregionBoundAttr,
        worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value, StringRef symbol,
                            ArrayRef<Attribute> operands) {
  return PhysicalExprAttr::get(
      context, kind, value,
      kind == PhysicalExprKind::Parameter
          ? Attribute(ParameterRefAttr::get(context, StringAttr::get(context, symbol)))
          : Attribute(StringAttr::get(context, symbol)),
      ArrayAttr::get(context, operands));
}

PhysicalExprAttr nextPowerOfTwo(PhysicalExprAttr source) {
  if (source.getKind() == PhysicalExprKind::Constant) {
    uint64_t value = std::max<int64_t>(source.getValue(), 1);
    uint64_t result = 1;
    while (result < value)
      result <<= 1;
    return expression(source.getContext(), PhysicalExprKind::Constant, result);
  }
  return expression(source.getContext(), PhysicalExprKind::NextPowerOfTwo, 0,
                    {}, {source});
}

FragmentType replaceExtent(FragmentType source, unsigned axis,
                           PhysicalExprAttr extent, Type element) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[axis] = extent;
  return FragmentType::get(source.getContext(),
                           element ? element : source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

FailureOr<Value> predicateForReductionSource(OpBuilder &builder,
                                             Location location, Value predicate,
                                             FragmentType source,
                                             unsigned reductionAxis) {
  if (reductionAxis >= source.getAxisMaps().size())
    return failure();
  return projectPredicateToFragmentAxis(builder, location, predicate, source,
                                        reductionAxis);
}

FailureOr<SmallVector<Value>> inlinePureRegion(OpBuilder &builder, Region &region,
                                               ValueRange arguments,
                                               std::string &reason) {
  if (region.empty() || region.getBlocks().size() != 1 ||
      region.front().getNumArguments() != arguments.size()) {
    reason = ("combine argument schema mismatch: expected " +
              Twine(region.empty() ? 0 : region.front().getNumArguments()) +
              ", got " + Twine(arguments.size()))
                 .str();
    return failure();
  }
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield) {
    reason = ("combine region terminates with " +
              region.front().getTerminator()->getName().getStringRef())
                 .str();
    return failure();
  }
  IRMapping mapping;
  for (auto [argument, value] :
       llvm::zip(region.front().getArguments(), arguments))
    mapping.map(argument, value);
  for (Operation &operation : region.front().without_terminator()) {
    Operation *clone = builder.clone(operation, mapping);
    for (auto [source, result] :
         llvm::zip(operation.getResults(), clone->getResults()))
      if (!mapping.lookupOrNull(source))
        mapping.map(source, result);
  }
  SmallVector<Value> results;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      reason = "combine yield value was not mapped by pure-region cloning";
      return failure();
    }
    results.push_back(mapped);
  }
  return results;
}

Type dataElementType(Type type) {
  if (auto fragment = dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  return type;
}

FragmentType withElementType(FragmentType schema, Type elementType) {
  return FragmentType::get(schema.getContext(), elementType, schema.getShape(),
                           schema.getAxisMaps(), schema.getValidity(),
                           schema.getOwner());
}

namespace {

bool sameExecutionSchema(FragmentType lhs, FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

FailureOr<FragmentType> commonExecutionSchema(ValueRange values) {
  FragmentType result;
  for (Value value : values) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      continue;
    if (!result || result.getShape().size() < fragment.getShape().size()) {
      result = fragment;
      continue;
    }
    if (result.getShape().size() == fragment.getShape().size() &&
        !sameExecutionSchema(result, fragment))
      return failure();
  }
  return result ? FailureOr<FragmentType>(result)
                : FailureOr<FragmentType>(failure());
}

} // namespace

FailureOr<Value> alignToExecutionSchema(OpBuilder &builder, Location location,
                                        Value value,
                                        FragmentType executionSchema) {
  FragmentType target =
      withElementType(executionSchema, dataElementType(value.getType()));
  if (value.getType() == target)
    return value;
  if (!isa<FragmentType>(value.getType()))
    return Value(builder.create<SplatOp>(location, target, value));
  auto source = cast<FragmentType>(value.getType());
  if (source.getOwner() != target.getOwner() ||
      source.getShape().size() > target.getShape().size())
    return failure();
  return Value(builder.create<BroadcastOp>(location, target, value));
}

bool canLiftCombineOperation(Operation &operation) {
  return isa<arith::ConstantOp, UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
             BitcastOp, SplatOp, MakeRecordOp, ExtractOp>(operation);
}

LogicalResult cloneLiftedCombineRegion(Region &source, Region &target,
                                       TypeRange accumulatorTypes,
                                       std::string &reason) {
  if (source.empty() || !llvm::hasSingleElement(source)) {
    reason = "multi-axis combine is not a single typed block";
    return failure();
  }
  Block &sourceBlock = source.front();
  if (sourceBlock.getNumArguments() < accumulatorTypes.size() * 2) {
    reason = "multi-axis combine argument schema is incomplete";
    return failure();
  }
  for (Operation &operation : sourceBlock.without_terminator())
    if (!canLiftCombineOperation(operation)) {
      reason = ("multi-axis combine contains a non-elementwise operation: " +
                operation.getName().getStringRef())
                   .str();
      return failure();
    }
  auto sourceYield = dyn_cast<YieldOp>(sourceBlock.getTerminator());
  if (!sourceYield || sourceYield.getValues().size() != accumulatorTypes.size()) {
    reason = "multi-axis combine yield schema is incomplete";
    return failure();
  }

  auto *targetBlock = new Block();
  target.push_back(targetBlock);
  Location location = sourceYield.getLoc();
  MLIRContext *context = location.getContext();
  for (Type type : accumulatorTypes)
    targetBlock->addArgument(type, location);
  for (Type type : accumulatorTypes)
    targetBlock->addArgument(type, location);
  for (BlockArgument capture :
       sourceBlock.getArguments().drop_front(accumulatorTypes.size() * 2))
    targetBlock->addArgument(capture.getType(), location);

  IRMapping mapping;
  for (auto [original, replacement] :
       llvm::zip(sourceBlock.getArguments(), targetBlock->getArguments()))
    mapping.map(original, replacement);
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(targetBlock);

  auto mappedOperands = [&](Operation &operation) {
    SmallVector<Value> values;
    for (Value operand : operation.getOperands()) {
      Value mapped = mapping.lookupOrNull(operand);
      values.push_back(mapped ? mapped : operand);
    }
    return values;
  };
  auto createLike = [&](Operation &operation, ValueRange operands,
                        TypeRange results) {
    OperationState state(operation.getLoc(), operation.getName());
    state.addOperands(operands);
    state.addTypes(results);
    state.addAttributes(operation.getAttrs());
    return builder.create(state);
  };

  for (Operation &operation : sourceBlock.without_terminator()) {
    SmallVector<Value> operands = mappedOperands(operation);
    if (auto splat = dyn_cast<SplatOp>(operation)) {
      FragmentType schema;
      for (auto [index, type] : llvm::enumerate(accumulatorTypes)) {
        auto original = dyn_cast<FragmentType>(sourceBlock.getArgument(index).getType());
        if (!original || !sameExecutionSchema(original, splat.getResult().getType()))
          continue;
        auto lifted = dyn_cast<FragmentType>(type);
        if (!lifted || (schema && !sameExecutionSchema(schema, lifted))) {
          reason = "lifted splat has conflicting fragment execution schemas";
          return failure();
        }
        schema = lifted;
      }
      auto lifted = schema ? alignToExecutionSchema(
                                 builder, operation.getLoc(), operands.front(), schema)
                           : FailureOr<Value>(failure());
      if (failed(lifted)) {
        reason = "lifted splat has no compatible fragment execution schema";
        return failure();
      }
      mapping.map(splat.getResult(), *lifted);
      continue;
    }
    SmallVector<Type> resultTypes;
    bool changed = llvm::any_of(
        llvm::zip(operation.getOperands(), operands), [](auto pair) {
          return std::get<0>(pair).getType() != std::get<1>(pair).getType();
        });
    if (!changed) {
      Operation *clone = builder.clone(operation, mapping);
      for (auto [original, replacement] :
           llvm::zip(operation.getResults(), clone->getResults()))
        mapping.map(original, replacement);
      continue;
    }

    if (auto record = dyn_cast<MakeRecordOp>(operation)) {
      auto original = record.getResult().getType();
      SmallVector<Attribute> fields;
      for (Value field : operands)
        fields.push_back(TypeAttr::get(field.getType()));
      resultTypes.push_back(RecordType::get(
          context, original.getFieldNames(),
          ArrayAttr::get(context, fields), original.getOwner()));
    } else if (auto extract = dyn_cast<ExtractOp>(operation)) {
      auto record = dyn_cast<RecordType>(operands.front().getType());
      if (!record || extract.getField() >= record.getFieldTypes().size()) {
        reason = "lifted record projection lost its field schema";
        return failure();
      }
      resultTypes.push_back(
          cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
    } else if (isa<UnaryOp, CastOp, BitcastOp>(operation)) {
      auto schema = dyn_cast<FragmentType>(operands.front().getType());
      if (!schema) {
        reason = "lifted unary operation has no fragment execution schema";
        return failure();
      }
      resultTypes.push_back(withElementType(
          schema, dataElementType(operation.getResult(0).getType())));
    } else if (isa<BinaryOp, CompareOp>(operation)) {
      FailureOr<FragmentType> schema = commonExecutionSchema(operands);
      if (failed(schema)) {
        reason = "lifted binary operation has incompatible fragment schemas";
        return failure();
      }
      for (Value &operand : operands) {
        FailureOr<Value> aligned = alignToExecutionSchema(
            builder, operation.getLoc(), operand, *schema);
        if (failed(aligned)) {
          reason = "lifted binary operand cannot be broadcast to its fragment";
          return failure();
        }
        operand = *aligned;
      }
      Type element = isa<CompareOp>(operation)
                         ? Type(builder.getI1Type())
                         : dataElementType(operation.getResult(0).getType());
      resultTypes.push_back(withElementType(*schema, element));
    } else if (isa<SelectOp>(operation)) {
      FailureOr<FragmentType> schema =
          commonExecutionSchema(ValueRange(operands).drop_front());
      if (failed(schema)) {
        reason = "lifted select values have incompatible fragment schemas";
        return failure();
      }
      for (Value &operand : operands) {
        FailureOr<Value> aligned = alignToExecutionSchema(
            builder, operation.getLoc(), operand, *schema);
        if (failed(aligned)) {
          reason = "lifted select operand cannot be broadcast to its fragment";
          return failure();
        }
        operand = *aligned;
      }
      resultTypes.push_back(withElementType(
          *schema, dataElementType(operation.getResult(0).getType())));
    } else {
      reason = "multi-axis combine operation cannot be lifted to a fragment";
      return failure();
    }

    Operation *clone = createLike(operation, operands, resultTypes);
    for (auto [original, replacement] :
         llvm::zip(operation.getResults(), clone->getResults()))
      mapping.map(original, replacement);
  }

  SmallVector<Value> yields;
  for (Value value : sourceYield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      reason = "lifted multi-axis combine yield was not mapped";
      return failure();
    }
    yields.push_back(mapped);
  }
  for (auto [value, type] : llvm::zip(yields, accumulatorTypes))
    if (value.getType() != type) {
      reason = "lifted multi-axis combine result type disagrees with its accumulator";
      return failure();
    }
  builder.create<YieldOp>(sourceYield.getLoc(), yields);
  return success();
}

bool prepareVectorAccumulation(ReduceOp reduce,
                               ArrayRef<FragmentType> accumulatorTypes,
                               Region &combine) {
  ValueRange identities = reduce.getIdentities();
  ValueRange captures = reduce.getCaptures();
  if (!llvm::all_of(captures, [](Value capture) {
        return isa<IntegerType, FloatType, IndexType>(capture.getType());
      }) ||
      !llvm::all_of(llvm::enumerate(accumulatorTypes), [&](auto component) {
        FragmentType type = component.value();
        Type element = type.getElementType();
        auto identity = scalarSource(identities[component.index()]);
        return isa<IntegerType, FloatType, IndexType>(element) &&
               succeeded(identity) && (*identity).getType() == element &&
               dataElementType(reduce.getResult(component.index()).getType()) ==
                   element &&
               sameExecutionSchema(type, accumulatorTypes.front());
      }))
    return false;
  SmallVector<Type> types(accumulatorTypes.begin(), accumulatorTypes.end());
  std::string reason;
  return succeeded(
      cloneLiftedCombineRegion(reduce.getCombine(), combine, types, reason));
}

} // namespace intent::gpu::reduction

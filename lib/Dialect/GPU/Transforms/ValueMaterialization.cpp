#include "Intent/Dialect/GPU/IR/ProgramInterface.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {

namespace {

FailureOr<Value> projectPredicate(OpBuilder &builder, Location location,
                                  Value predicate, FragmentType target,
                                  unsigned axis) {
  auto base = dyn_cast<FragmentType>(predicate.getType());
  if (!base || base.getShape().size() != 1 || axis >= target.getShape().size())
    return failure();
  SmallVector<Attribute> shape(
      target.getShape().size(),
      PhysicalExprAttr::get(
          target.getContext(),
          PhysicalExprKind::Constant, 1,
          StringAttr::get(target.getContext()),
          ArrayAttr::get(target.getContext(), {})));
  shape[axis] = base.getShape()[0];
  auto reshaped = FragmentType::get(
      target.getContext(), builder.getI1Type(),
      ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
      target.getValidity(), target.getOwner());
  Value result = predicate;
  if (base != reshaped) {
    SmallVector<Attribute> groups;
    for (unsigned resultAxis = 0; resultAxis < shape.size(); ++resultAxis) {
      SmallVector<int64_t> sourceAxes;
      if (resultAxis == axis)
        sourceAxes.push_back(0);
      groups.push_back(ReshapeGroupAttr::get(
          target.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
          builder.getDenseI64ArrayAttr({static_cast<int64_t>(resultAxis)})));
    }
    result = builder.create<ReshapeOp>(location, reshaped, result,
                                      builder.getArrayAttr(groups));
  }
  auto projected = FragmentType::get(
      target.getContext(), builder.getI1Type(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  if (reshaped != projected)
    result = builder.create<BroadcastOp>(location, projected, result);
  return result;
}

FailureOr<Value> zeroValue(OpBuilder &builder, Location location, Type type) {
  auto fragment = dyn_cast<FragmentType>(type);
  Type element = fragment ? fragment.getElementType() : type;
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(element))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(element))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(element))
    zero = builder.getIndexAttr(0);
  if (!zero)
    return failure();
  FailureOr<Value> scalar =
      materializeScalarConstant(builder, location, zero, element);
  if (failed(scalar))
    return failure();
  if (!fragment)
    return *scalar;
  return Value(builder.create<BroadcastOp>(location, fragment, *scalar));
}


} // namespace

LogicalResult scalarizeElementwiseCallback(Region &source, Region &target) {
  if (!target.empty())
    return target.getParentOp()->emitOpError(
        "scalar callback target region must be empty");
  Block &body = source.front();
  for (Operation &nested : body) {
    for (Value operand : nested.getOperands())
      if (operand.getParentBlock() != &body)
        return nested.emitOpError(
            "native collective callback cannot capture enclosing values");
    if (!isa<arith::ConstantOp, SplatOp, BroadcastOp, ReshapeOp, UnaryOp, BinaryOp,
             CompareOp, SelectOp, CastOp, BitcastOp, MakeRecordOp, ExtractOp,
             YieldOp>(nested))
      return nested.emitOpError(
          "native collective requires an elementwise scalarizable combine");
    if (auto broadcast = dyn_cast<BroadcastOp>(nested)) {
      auto sourceType = dyn_cast<FragmentType>(broadcast.getValue().getType());
      if (sourceType) {
        auto targetType = broadcast.getResult().getType();
        auto projection = queryAxisProjection(sourceType, targetType);
        if (sourceType.getShape() != targetType.getShape() ||
            !projection.isExact() ||
            llvm::any_of(llvm::enumerate(projection.targetToSource),
                         [](auto pair) {
                           return !pair.value() ||
                                  *pair.value() != pair.index();
                         }))
          return broadcast.emitOpError(
              "non-identity fragment broadcast in a collective requires prior lane-wise legalization");
      }
    }
    if (auto reshape = dyn_cast<ReshapeOp>(nested))
      if (cast<FragmentType>(reshape.getValue().getType()).getShape() !=
          cast<FragmentType>(reshape.getResult().getType()).getShape())
        return reshape.emitOpError(
            "non-identity fragment reshape in a collective requires prior lane-wise legalization");
  }

  Block *scalarBody = new Block();
  target.push_back(scalarBody);
  IRMapping mapping;
  for (BlockArgument argument : body.getArguments())
    mapping.map(argument, scalarBody->addArgument(
                             scalarCallbackType(argument.getType()),
                             argument.getLoc()));
  OpBuilder builder(body.getTerminator()->getContext());
  builder.setInsertionPointToEnd(scalarBody);
  for (Operation &nested : body) {
    if (isa<SplatOp, BroadcastOp, ReshapeOp>(nested)) {
      mapping.map(nested.getResult(0), mapping.lookup(nested.getOperand(0)));
      continue;
    }
    Operation *cloned = builder.clone(nested, mapping);
    for (Value result : cloned->getResults())
      result.setType(scalarCallbackType(result.getType()));
  }
  return success();
}

FailureOr<Value> materializeNonOverlappingView(func::FuncOp kernel,
                                             Value resource) {
  auto view = cast<ViewType>(resource.getType());
  SmallVector<OpFoldResult> strides;
  for (Attribute attribute : view.getLayout().getStrides()) {
    auto expression = cast<PhysicalExprAttr>(attribute);
    if (expression.getKind() == PhysicalExprKind::Constant) {
      strides.push_back(IntegerAttr::get(IndexType::get(kernel.getContext()), expression.getValue()));
      continue;
    }
    Value stride = resolveArgument(kernel, expression.getArgumentReference());
    if (!stride || !stride.getType().isIndex())
      return failure();
    strides.push_back(stride);
  }
  if (strides.empty())
    return failure();
  OpBuilder builder(&kernel.front(), kernel.front().begin());
  Location location = resource.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  SmallVector<Value> values;
  for (OpFoldResult stride : strides)
    values.push_back(isa<Value>(stride) ? cast<Value>(stride) : Value(
        builder.create<arith::ConstantIndexOp>(
            location, cast<IntegerAttr>(cast<Attribute>(stride)).getInt())));
  Value valid = builder.create<CompareOp>(location, builder.getI1Type(),
                                         values.back(), zero, ComparePredicate::Gt);
  for (unsigned axis = values.size() - 1; axis > 0; --axis) {
    Value divisor = builder.create<BinaryOp>(location, builder.getIndexType(),
        values[axis], one, BinaryOperator::Maximum);
    Value slots = builder.create<BinaryOp>(location, builder.getIndexType(),
        values[axis - 1], divisor, BinaryOperator::FloorDivide);
    Value extent = builder.create<DimOp>(location, builder.getIndexType(),
                                         resource, axis);
    extent = builder.create<BinaryOp>(location, builder.getIndexType(), extent,
                                       one, BinaryOperator::Maximum);
    // Division avoids overflow in stride >= extent * next_stride. Every
    // accepted suffix has positive strides and fits inside its outer stride.
    Value separate = builder.create<CompareOp>(location, builder.getI1Type(),
                                               slots, extent, ComparePredicate::Ge);
    valid = builder.create<BinaryOp>(location, builder.getI1Type(), valid,
                                      separate, BinaryOperator::LogicalAnd);
  }
  return valid;
}

FailureOr<Value> materializeScalarConstant(OpBuilder &builder,
                                           Location location, Attribute value,
                                           Type resultType) {
  if (isa<IndexType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    return Value(
        builder.create<arith::ConstantIndexOp>(location, integer.getInt()));
  }
  if (auto integerType = dyn_cast<IntegerType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    auto signless = IntegerType::get(builder.getContext(), integerType.getWidth());
    llvm::APInt bits = integer.getValue().sextOrTrunc(integerType.getWidth());
    Value raw = builder.create<arith::ConstantOp>(
        location, signless, IntegerAttr::get(signless, bits));
    if (integerType.isSignless())
      return raw;
    return Value(builder.create<CastOp>(location, integerType, raw));
  }
  if (auto floatType = dyn_cast<FloatType>(resultType)) {
    auto floating = dyn_cast<FloatAttr>(value);
    if (!floating)
      return failure();
    return Value(builder.create<arith::ConstantOp>(
        location, floatType,
        FloatAttr::get(floatType, floating.getValueAsDouble())));
  }
  return failure();
}

static FailureOr<Value> projectFragmentValue(OpBuilder &builder,
                                             Location location, Value value,
                                             FragmentType target, ValueTypeChangeCallback changed) {
  if (value.getType() == target)
    return value;
  Type element = value.getType();
  auto source = dyn_cast<FragmentType>(element);
  if (source)
    element = source.getElementType();
  if (!isa<IntegerType, FloatType, IndexType>(element) ||
      element != target.getElementType())
    return failure();
  if (auto extract = value.getDefiningOp<ExtractOp>())
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>())
      return projectFragmentValue(builder, location,
                                  record.getFields()[extract.getField()], target, changed);
  Value scalar = value;
  while (isa<FragmentType>(scalar.getType())) {
    UniformExpression expression = describeUniformValue(scalar);
    if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
      break;
    scalar = expression.operands.front();
  }
  if (scalar.getType() == element)
    return Value(builder.create<SplatOp>(location, target, scalar));
  if (source)
    if (auto constant = dyn_cast_or_null<TypedAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(value));
        constant && constant.getType() == element) {
      Value scalar = builder.create<arith::ConstantOp>(location, element, constant);
      return Value(builder.create<SplatOp>(location, target, scalar));
    }
  if (source && source.getAxisMaps() != target.getAxisMaps()) {
    auto permutation = queryAxisPermutation(source, target);
    // An exact coordinate permutation is a transpose, not a broadcast. Match
    // source occurrences so equal-sized axes stay distinct.
    if (permutation &&
        llvm::any_of(llvm::enumerate(*permutation), [](auto item) {
          return item.value() != static_cast<int64_t>(item.index());
        })) {
      SmallVector<Attribute> shape;
      for (int64_t axis : *permutation)
        shape.push_back(source.getShape()[axis]);
      auto reordered = FragmentType::get(
          target.getContext(), element, builder.getArrayAttr(shape),
          target.getAxisMaps(), source.getValidity(), source.getOwner());
      auto transpose = builder.create<TransposeOp>(location, reordered, value,
                                                  *permutation);
      if (Operation *definition = value.getDefiningOp())
        if (Attribute origin = definition->getAttr(originAttr))
          transpose->setAttr(originAttr, origin);
      return projectFragmentValue(builder, location, transpose.getResult(),
                                  target, changed);
    }
  }
  Operation *projection = nullptr;
  if (!source) {
    projection = builder.create<SplatOp>(location, target, value);
  } else if (auto splat = value.getDefiningOp<SplatOp>()) {
    projection = builder.create<SplatOp>(location, target, splat.getValue());
  } else if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (input && input.getShape().size() < target.getShape().size()) {
      auto inputRelation = queryAxisProjection(input, source);
      auto resultRelation = queryAxisProjection(source, target);
      if (inputRelation.isExact() && resultRelation.isExact()) {
        SmallVector<Attribute> shape(input.getShape().getValue());
        for (auto [targetAxis, sourceAxis] :
             llvm::enumerate(resultRelation.targetToSource)) {
          if (!sourceAxis || !inputRelation.targetToSource[*sourceAxis])
            continue;
          unsigned inputAxis = *inputRelation.targetToSource[*sourceAxis];
          auto extent = cast<PhysicalExprAttr>(shape[inputAxis]);
          if (extent.getKind() !=
                  PhysicalExprKind::Constant ||
              extent.getValue() != 1)
            shape[inputAxis] = target.getShape()[targetAxis];
        }
        // Retile the producer in its own rank before expanding it. In
        // particular, a reduced row predicate must remain a row reduction.
        auto inputTarget = FragmentType::get(
            target.getContext(), input.getElementType(),
            builder.getArrayAttr(shape), input.getAxisMaps(),
            input.getValidity(), input.getOwner());
        FailureOr<Value> projected = projectFragmentValue(
            builder, location, broadcast.getValue(), inputTarget, changed);
        if (succeeded(projected) &&
            queryBroadcastProjection(inputTarget, target).isExact())
          projection = builder.create<BroadcastOp>(location, target, *projected);
      }
    } else {
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, broadcast.getValue(), target, changed);
      if (succeeded(projected))
        return *projected;
    }
  } else if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    auto input = cast<FragmentType>(reshape.getValue().getType());
    unsigned inputRank = 0, resultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = cast<ReshapeGroupAttr>(attribute);
      inputRank += group.getSourceAxes().size();
      resultRank += group.getResultAxes().size();
    }
    bool projects = inputRank <= input.getShape().size() &&
                    resultRank <= source.getShape().size() &&
                    source.getShape().size() == target.getShape().size();
    BroadcastProjection relation = queryAxisProjection(source, target);
    projects &= relation.isExact() && llvm::all_of(
        llvm::enumerate(relation.targetToSource), [](auto item) {
          return item.value() && *item.value() == item.index();
        });
    SmallVector<Attribute> inputShape(input.getShape().begin(),
                                       input.getShape().end());
    if (projects) {
      unsigned inputPrefix = input.getShape().size() - inputRank;
      unsigned resultPrefix = source.getShape().size() - resultRank;
      projects = inputPrefix == resultPrefix;
      if (projects)
        for (unsigned axis = 0; axis < inputPrefix; ++axis)
          inputShape[axis] = target.getShape()[axis];
      for (Attribute attribute : reshape.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        bool changed = llvm::any_of(group.getResultAxes().asArrayRef(),
                                    [&](int64_t axis) {
          return source.getShape()[resultPrefix + axis] !=
                 target.getShape()[resultPrefix + axis];
        });
        if (!changed)
          continue;
        if (group.getSourceAxes().size() != 1 ||
            group.getResultAxes().size() != 1) {
          projects = false;
          break;
        }
        inputShape[inputPrefix + group.getSourceAxes()[0]] =
            target.getShape()[resultPrefix + group.getResultAxes()[0]];
      }
    }
    if (projects) {
      auto inputTarget = FragmentType::get(
          target.getContext(), input.getElementType(), builder.getArrayAttr(inputShape),
          input.getAxisMaps(), target.getValidity(), target.getOwner());
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, reshape.getValue(), inputTarget, changed);
      if (succeeded(projected))
        projection = builder.create<ReshapeOp>(
            location, target, *projected, reshape.getReassociation());
    }
  } else if (auto reduce = value.getDefiningOp<ReduceOp>();
             reduce && reduce.getSources().size() == 1 &&
             reduce.getIdentities().size() == 1 && reduce.getCaptures().size() == 0 &&
             reduce.getNumResults() == 1 &&
             source.getOwner() == target.getOwner() &&
             queryBinaryCombineKind(reduce.getCombine())) {
    // A lane-wise combine preserves every non-reduced axis. Project the
    // source's free axes with the result, rather than resizing a finished
    // reduction or treating its construction extent as a broadcast scalar.
    auto relation = queryAxisProjection(source, target);
    bool projects =
        source.getShape().size() == target.getShape().size() &&
        relation.isExact() && llvm::all_of(
            llvm::enumerate(relation.targetToSource), [](auto item) {
              return item.value() && *item.value() == item.index();
            });
    if (projects) {
      auto input = cast<FragmentType>(reduce.getSources().front().getType());
      llvm::SmallDenseSet<int64_t> axes(reduce.getAxes().begin(),
                                       reduce.getAxes().end());
      SmallVector<Attribute> shape(input.getShape().getValue());
      unsigned resultAxis = 0;
      for (unsigned axis = 0; axis < shape.size(); ++axis)
        if (!axes.contains(axis))
          shape[axis] = target.getShape()[resultAxis++];
      auto inputTarget = FragmentType::get(
          target.getContext(), input.getElementType(), builder.getArrayAttr(shape),
          input.getAxisMaps(), input.getValidity(), input.getOwner());
      auto resultTarget = FragmentType::get(
          target.getContext(), target.getElementType(), target.getShape(),
          source.getAxisMaps(), target.getValidity(), target.getOwner());
      auto projectedSource = projectFragmentValue(
          builder, location, reduce.getSources().front(), inputTarget, changed);
      auto projectedIdentity = projectFragmentValue(
          builder, location, reduce.getIdentities().front(), resultTarget, changed);
      if (succeeded(projectedSource) && succeeded(projectedIdentity)) {
        IRMapping mapping;
        mapping.map(reduce.getSources().front(), *projectedSource);
        mapping.map(reduce.getIdentities().front(), *projectedIdentity);
        auto clone = cast<ReduceOp>(builder.clone(*reduce, mapping));
        for (BlockArgument argument : clone.getCombine().front().getArguments())
          for (unsigned axis = 0; axis < target.getShape().size(); ++axis)
            if (source.getShape()[axis] != target.getShape()[axis])
              retargetSourceExtent(
                  argument,
                  sourceAxisIdentity(
                      cast<AxisMapAttr>(source.getAxisMaps()[axis])),
                  cast<PhysicalExprAttr>(target.getShape()[axis]), std::nullopt, changed);
        clone.getResult(0).setType(resultTarget);
        projection = clone.getOperation();
        if (resultTarget != target)
          projection = builder.create<BroadcastOp>(location, target,
                                                   clone.getResult(0));
      }
    }
  }
  if (!projection && queryBroadcastProjection(source, target).isExact())
    projection = builder.create<BroadcastOp>(location, target, value);
  Operation *definition = value.getDefiningOp();
  if (!projection &&
      isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(
          definition)) {
    IRMapping mapping;
    for (Value operand : definition->getOperands()) {
      Type element = operand.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto operandTarget = FragmentType::get(
          target.getContext(), element, target.getShape(), target.getAxisMaps(),
          target.getValidity(), target.getOwner());
      FailureOr<Value> projected =
          projectFragmentValue(builder, location, operand, operandTarget, changed);
      if (failed(projected))
        return failure();
      mapping.map(operand, *projected);
    }
    projection = builder.clone(*definition, mapping);
    projection->getResult(0).setType(target);
  }
  if (!projection)
    return failure();
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projection->setAttr(originAttr, origin);
  return projection->getResult(0);
}

FailureOr<Value> projectPhysicalValueToSchema(OpBuilder &builder,
                                              Location location, Value value,
                                              Type target, ValueTypeChangeCallback changed) {
  if (value.getType() == target)
    return value;
  if (auto fragment = dyn_cast<FragmentType>(target))
    return projectFragmentValue(builder, location, value, fragment, changed);
  auto targetRecord = dyn_cast<RecordType>(target);
  auto sourceRecord = dyn_cast<RecordType>(value.getType());
  if (!targetRecord || !sourceRecord ||
      sourceRecord.getFieldNames() != targetRecord.getFieldNames() ||
      sourceRecord.getFieldTypes().size() !=
          targetRecord.getFieldTypes().size())
    return failure();
  auto record = value.getDefiningOp<MakeRecordOp>();
  SmallVector<Value> projectedFields;
  for (auto [index, targetField] :
       llvm::enumerate(targetRecord.getFieldTypes())) {
    Type sourceType =
        cast<TypeAttr>(sourceRecord.getFieldTypes()[index]).getValue();
    Value field = record
                      ? record.getFields()[index]
                      : Value(builder.create<ExtractOp>(location, sourceType,
                                                        value, index));
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, location, field, cast<TypeAttr>(targetField).getValue(), changed);
    if (failed(projected))
      return failure();
    projectedFields.push_back(*projected);
  }
  auto projected =
      builder.create<MakeRecordOp>(location, targetRecord, projectedFields);
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projected->setAttr(originAttr, origin);
  return projected.getResult();
}

FailureOr<Value> materializeZeroValue(OpBuilder &builder, Location location,
                                     Type target) {
  return zeroValue(builder, location, target);
}

FailureOr<Value> materializeZeroFragment(OpBuilder &builder,
                                         Location location,
                                         FragmentType target) {
  return materializeZeroValue(builder, location, target);
}

FailureOr<Value> materializeReplayedValue(
    OpBuilder &builder, Location location, Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    IRMapping &mapping, ReplayMaterializationOptions options) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  auto replayDimension = [&](PhysicalSourceAxis occurrence)
      -> std::optional<int64_t> {
    auto projections = queryFragmentAxes(value.getType(), occurrence);
    if (options.fragmentAxis)
      llvm::erase_if(projections, [&](PhysicalAxisProjection projection) {
        return projection.fragmentAxis != *options.fragmentAxis;
      });
    return projections.size() == 1
               ? std::optional<int64_t>(projections.front().dimensionId)
               : std::nullopt;
  };
  PhysicalReplayFact replay = analysis.replayability(
      value, source, options.scope, options.allowAccesses, nullptr,
      replayDimension(source));
  if (!replay.isReplayable())
    return failure();

  SmallVector<PhysicalSourceAxis> replaySources{source};
  for (MakeRangeOp range : options.traversalRanges) {
    PhysicalSourceAxis occurrence = sourceAxisIdentity(range);
    if (!llvm::is_contained(replaySources, occurrence))
      replaySources.push_back(occurrence);
    if (!analysis.replayability(value, occurrence, options.scope,
                               options.allowAccesses, nullptr,
                               replayDimension(occurrence)).isReplayable())
      return failure();
  }
  bool projectionFailed = false;
  auto replayProjection = [&](Type type,
                              std::optional<unsigned> axis = std::nullopt) {
    PhysicalAxisProjection result;
    for (PhysicalSourceAxis occurrence : replaySources) {
      auto projections = queryFragmentAxes(type, occurrence);
      auto dimension = replayDimension(occurrence);
      llvm::erase_if(projections, [&](PhysicalAxisProjection projection) {
        return (axis && projection.fragmentAxis != *axis) ||
               (dimension && projection.dimensionId != *dimension);
      });
      PhysicalAxisProjection current;
      if (projections.size() == 1)
        current = projections.front();
      if (projections.size() > 1 ||
          (result.isExact() && current.isExact() &&
           (result.fragmentAxis != current.fragmentAxis ||
            result.dimensionId != current.dimensionId))) {
        result.state = PhysicalFactState::Ambiguous;
        projectionFailed = true;
        return result;
      }
      if (!current.isExact())
        continue;
      result = current;
    }
    return result;
  };
  auto replaceReplayAxis = [&](FragmentType type, unsigned axis) {
    SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
    SmallVector<Attribute> axes(type.getAxisMaps().begin(),
                                type.getAxisMaps().end());
    shape[axis] = blockedExtent;
    if (options.segmentMapping)
      axes[axis] = AxisMapAttr::get(
          type.getContext(), options.segmentMapping.getSourceId(),
          options.segmentMapping.getSourceAxis(),
          options.segmentMapping.getDimensionId(), axis,
          options.segmentMapping.getDerived());
    return FragmentType::get(
        type.getContext(), type.getElementType(),
        ArrayAttr::get(type.getContext(), shape),
        ArrayAttr::get(type.getContext(), axes), type.getValidity(),
        type.getOwner());
  };
  std::function<Type(Type)> replaceReplayType = [&](Type type) -> Type {
    if (auto fragment = dyn_cast<FragmentType>(type)) {
      PhysicalAxisProjection projection = replayProjection(fragment);
      return projection.isExact()
                 ? Type(replaceReplayAxis(fragment, projection.fragmentAxis))
                 : type;
    }
    auto record = dyn_cast<RecordType>(type);
    if (!record)
      return type;
    SmallVector<Attribute> fields;
    bool changed = false;
    for (Attribute field : record.getFieldTypes()) {
      Type current = cast<TypeAttr>(field).getValue();
      Type replacement = replaceReplayType(current);
      fields.push_back(TypeAttr::get(replacement));
      changed |= replacement != current;
    }
    return changed ? Type(RecordType::get(
                         type.getContext(), record.getFieldNames(),
                         ArrayAttr::get(type.getContext(), fields),
                         record.getOwner()))
                   : type;
  };
  auto carriesReplaySource = [&](Type type) {
    return replaceReplayType(type) != type;
  };
  auto selectedReplayAxis = [&](Value current,
                                std::optional<unsigned> axis = std::nullopt)
      -> std::optional<unsigned> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    PhysicalAxisProjection projection =
        fragment ? replayProjection(fragment, axis)
                 : PhysicalAxisProjection{};
    if (fragment && !projection.isExact() &&
        projection.state != PhysicalFactState::Ambiguous &&
        !options.traversalRanges.empty()) {
      std::optional<unsigned> selected;
      for (unsigned candidate = 0; candidate < fragment.getShape().size(); ++candidate) {
        if (axis && candidate != *axis)
          continue;
        // Equal traversal bounds do not identify independent logical axes.
        auto candidateMap = cast<AxisMapAttr>(fragment.getAxisMaps()[candidate]);
        if (!llvm::any_of(options.traversalRanges, [&](MakeRangeOp range) {
              auto dimension = queryRangeDimension(range);
              return succeeded(dimension) &&
                     *dimension == candidateMap.getDimensionId();
            }))
          continue;
        PhysicalRangeFact fact = analysis.axisRanges(current, candidate);
        if (fact.state == PhysicalFactState::Unknown || !fact.blockers.empty() ||
            fact.roots.empty())
          continue;
        if (!llvm::any_of(fact.roots, [&](MakeRangeOp root) {
              return llvm::any_of(options.traversalRanges, [&](MakeRangeOp range) {
                return sameLogicalRange(root, range);
              });
            }))
          continue;
        SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
        combined.append(options.traversalRanges.begin(), options.traversalRanges.end());
        if (!analysis.lockstepRanges(combined).isExact())
          continue;
        if (selected)
          return std::nullopt;
        selected = candidate;
      }
      if (selected)
        return selected;
    }
    if (!fragment || !projection.isExact())
      return std::nullopt;
    if (options.traversalRanges.empty())
      return projection.fragmentAxis;
    PhysicalRangeFact fact =
        analysis.axisRanges(current, projection.fragmentAxis);
    if (fact.roots.empty()) {
      Operation *producer = current.getDefiningOp();
      bool neutralSchemaCarrier = isa_and_nonnull<SplatOp>(producer);
      neutralSchemaCarrier |= fact.isExact() && fact.blockers.empty() &&
          isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                         BitcastOp>(producer);
      if (auto broadcast = dyn_cast_or_null<BroadcastOp>(producer)) {
        auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
        neutralSchemaCarrier |=
            !isa<FragmentType, RecordType>(broadcast.getValue().getType());
        if (input) {
          BroadcastProjection relation = queryAxisProjection(input, fragment);
          if (relation.isExact()) {
            std::optional<unsigned> inputAxis =
                relation.targetToSource[projection.fragmentAxis];
            if (!inputAxis) {
              neutralSchemaCarrier = true;
            } else {
              auto extent = cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
              neutralSchemaCarrier |=
                  extent.getKind() ==
                      PhysicalExprKind::Constant &&
                  extent.getValue() == 1;
            }
          }
        }
      }
      return neutralSchemaCarrier
                 ? std::optional<unsigned>(projection.fragmentAxis)
                 : std::nullopt;
    }
    SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
    combined.append(options.traversalRanges.begin(),
                    options.traversalRanges.end());
    return analysis.lockstepRanges(combined).isExact()
               ? std::optional<unsigned>(projection.fragmentAxis)
               : std::nullopt;
  };
  auto retargetHelperSourceExtent = [&](Region &region,
                                        PhysicalExprAttr logicalExtent) {
    auto retarget = [&](Value current) {
      auto fragment = dyn_cast<FragmentType>(current.getType());
      PhysicalAxisProjection projection =
          fragment ? queryFragmentAxis(fragment, source, replayDimension(source))
                   : PhysicalAxisProjection{};
      if (!fragment || !projection.isExact() ||
          fragment.getShape()[projection.fragmentAxis] != logicalExtent)
        return;
      current.setType(replaceReplayAxis(fragment, projection.fragmentAxis));
    };
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments())
        retarget(argument);
      block.walk([&](Operation *operation) {
        for (Value result : operation->getResults())
          retarget(result);
      });
    }
  };

  llvm::DenseMap<std::pair<Value, unsigned>, Value> axisValues;
  auto hasMultipleReplayAxes = [&](Type type) {
    std::optional<unsigned> selected;
    for (PhysicalSourceAxis occurrence : replaySources)
      for (PhysicalAxisProjection projection :
           queryFragmentAxes(type, occurrence)) {
        if (selected && *selected != projection.fragmentAxis)
          return true;
        selected = projection.fragmentAxis;
      }
    return false;
  };
  auto remember = [&](Value original, Value replacement,
                       std::optional<unsigned> axis) {
    if (axis)
      axisValues[{original, *axis}] = replacement;
    if (!hasMultipleReplayAxes(original.getType()))
      mapping.map(original, replacement);
  };
  std::function<FailureOr<Value>(Value, std::optional<unsigned>)> materialize =
      [&](Value current,
          std::optional<unsigned> requestedAxis) -> FailureOr<Value> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    std::optional<unsigned> projection =
        selectedReplayAxis(current, requestedAxis);
    if (projection) {
      auto cached = axisValues.find({current, *projection});
      if (cached != axisValues.end())
        return cached->second;
    }
    if (Value mapped = mapping.lookupOrNull(current)) {
      // A prebound load must identify its occurrence by the changed axis alone.
      if (hasMultipleReplayAxes(current.getType()) &&
          (!projection || fragment.getShape()[*projection] == blockedExtent ||
           mapped.getType() != replaceReplayAxis(fragment, *projection)))
        return failure();
      return mapped;
    }
    if (auto extract = current.getDefiningOp<ExtractOp>()) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        uint64_t field = extract.getField();
        if (field >= record.getFields().size())
          return failure();
        FailureOr<Value> replayed =
            materialize(record.getFields()[field], projection);
        if (succeeded(replayed))
          remember(current, *replayed, projection);
        return replayed;
      }
      if (carriesReplaySource(extract.getRecord().getType())) {
        FailureOr<Value> replayedRecord =
            materialize(extract.getRecord(), std::nullopt);
        if (failed(replayedRecord))
          return failure();
        Type target = replaceReplayType(current.getType());
        Value replayed = builder.create<ExtractOp>(
            location, target, *replayedRecord, extract.getField());
        remember(current, replayed, projection);
        return replayed;
      }
    }

    if (!fragment && carriesReplaySource(current.getType())) {
      Operation *producer = current.getDefiningOp();
      if (!producer ||
          !isPhysicalReplayNode(producer, options.scope,
                                /*allowAccesses=*/false))
        return failure();
      for (Value operand : producer->getOperands()) {
        if (!carriesReplaySource(operand.getType()))
          continue;
        FailureOr<Value> replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        if (!mapping.lookupOrNull(operand) && *replayed != operand)
          mapping.map(operand, *replayed);
      }
      Operation *clone = builder.clone(*producer, mapping);
      for (auto [original, result] :
           llvm::zip(producer->getResults(), clone->getResults())) {
        result.setType(replaceReplayType(result.getType()));
        if (!mapping.lookupOrNull(original))
          mapping.map(original, result);
      }
      Region *combine = nullptr;
      if (auto reduce = dyn_cast<ReduceOp>(clone))
        combine = &reduce.getCombine();
      else if (auto scan = dyn_cast<ScanOp>(clone))
        combine = &scan.getCombine();
      if (combine)
        for (Block &block : *combine) {
          for (BlockArgument argument : block.getArguments())
            argument.setType(replaceReplayType(argument.getType()));
          block.walk([&](Operation *operation) {
            for (Value result : operation->getResults())
              result.setType(replaceReplayType(result.getType()));
          });
        }
      auto result = dyn_cast<OpResult>(current);
      if (!result || result.getResultNumber() >= clone->getNumResults())
        return failure();
      Value replayed = clone->getResult(result.getResultNumber());
      return replayed;
    }
    if (!fragment)
      return current;
    if (!projection) {
      Operation *producer = current.getDefiningOp();
      if (!producer || !isPhysicalReplayNode(
                           producer, PhysicalReplayScope::Coordinate,
                           /*allowAccesses=*/false))
        return current;
      IRMapping cloneMapping(mapping);
      bool changed = false;
      for (Value operand : producer->getOperands()) {
        FailureOr<Value> replayed = materialize(operand, std::nullopt);
        if (failed(replayed))
          return failure();
        cloneMapping.map(operand, *replayed);
        changed |= *replayed != operand;
      }
      if (!changed)
        return current;
      Operation *clone = builder.clone(*producer, cloneMapping);
      remember(current, clone->getResult(0), std::nullopt);
      return clone->getResult(0);
    }
    unsigned axis = *projection;
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return failure();

    auto replayOperand = [&](Value operand) -> FailureOr<Value> {
      std::optional<unsigned> operandAxis;
      auto input = dyn_cast<FragmentType>(operand.getType());
      if (input && (hasMultipleReplayAxes(input) ||
                    hasMultipleReplayAxes(fragment))) {
        auto reduction = dyn_cast<ReduceOp>(producer);
        bool reductionSource = reduction && llvm::is_contained(
            reduction.getSources(), operand);
        if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
                BroadcastOp, SplatOp>(producer) ||
            (reduction && !reductionSource)) {
          BroadcastProjection relation = queryAxisProjection(input, fragment);
          if (!relation.isExact())
            return failure();
          operandAxis = relation.targetToSource[axis];
          if (!operandAxis)
            return operand;
        } else if (auto transpose = dyn_cast<TransposeOp>(producer)) {
          operandAxis = transpose.getPermutation()[axis];
        } else if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
          if (isIntroducedReshapeUnitAxis(current, axis))
            return operand;
          operandAxis = reshapeInputAxis(reshape, axis);
        } else if (reduction) {
          SmallVector<unsigned> freeAxes;
          for (unsigned inputAxis = 0; inputAxis < input.getShape().size();
               ++inputAxis)
            if (!llvm::is_contained(reduction.getAxes(),
                                   static_cast<int64_t>(inputAxis)))
              freeAxes.push_back(inputAxis);
          if (axis < freeAxes.size())
            operandAxis = freeAxes[axis];
        }
        if (!operandAxis)
          return failure();
      }
      if (Value mapped = mapping.lookupOrNull(operand);
          mapped && !hasMultipleReplayAxes(operand.getType()))
        return mapped;
      return materialize(operand, operandAxis);
    };
    auto combineTail = [&](FragmentType target,
                           Value valid) -> FailureOr<Value> {
      if (!options.segmentTail)
        return valid ? FailureOr<Value>(valid)
                     : FailureOr<Value>(Value());
      FailureOr<Value> tail = projectPredicateToFragmentAxis(
          builder, location, options.segmentTail, target, axis);
      if (failed(tail))
        return failure();
      if (!valid)
        return *tail;
      return materializeValidityConjunction(builder, location, valid, *tail,
                                            target);
    };
    auto replayFill = [&](Value fill,
                          FragmentType target) -> FailureOr<Value> {
      if (fill) {
        FailureOr<Value> replayed = replayOperand(fill);
        if (failed(replayed))
          return failure();
        fill = *replayed;
        if (fill.getType() != target) {
          FailureOr<Value> projected =
              projectPhysicalValueToSchema(builder, location, fill, target);
          if (failed(projected))
            return failure();
          fill = *projected;
        }
        return fill;
      }
      if (!options.materializeZeroFill)
        return Value();
      return materializeZeroFragment(builder, location, target);
    };

    if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
      auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
      BroadcastProjection relation =
          input ? queryAxisProjection(input, fragment) : BroadcastProjection{};
      if (relation.isExact() && axis < relation.targetToSource.size()) {
        if (auto inputAxis = relation.targetToSource[axis]) {
          auto inputMap = cast<AxisMapAttr>(input.getAxisMaps()[*inputAxis]);
          auto resultMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
          auto inputExtent =
              cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
          bool sameLogicalAxis = inputMap.getDimensionId() > 0 &&
                                 inputMap.getDimensionId() ==
                                     resultMap.getDimensionId();
          bool nonUnitStaticExtent =
              inputExtent.getKind() ==
                  PhysicalExprKind::Constant &&
              inputExtent.getValue() > 1 &&
              queryBroadcastProjection(input, fragment).isExact();
          if (!(sourceAxisIdentity(inputMap) == sourceAxisIdentity(resultMap)) &&
              (sameLogicalAxis || nonUnitStaticExtent) &&
              input.getShape()[*inputAxis] == fragment.getShape()[axis]) {
            // An extent-preserving projection can rename an occurrence. Replay
            // its input with that input's identity, retaining the output map.
            ReplayMaterializationOptions inputOptions = options;
            inputOptions.fragmentAxis = *inputAxis;
            FailureOr<Value> replayed = materializeReplayedValue(
                builder, location, broadcast.getValue(),
                sourceAxisIdentity(inputMap), blockedExtent, mapping,
                inputOptions);
            if (failed(replayed))
              return failure();
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                builder, location, *replayed, replaceReplayAxis(fragment, axis));
            if (failed(projected))
              return failure();
            remember(current, *projected, projection);
            return *projected;
          }
        }
      }
    }

    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      auto input = cast<FragmentType>(reshape.getValue().getType());
      if (auto inputAxis = reshapeInputAxis(reshape, axis)) {
        auto inputMap = cast<AxisMapAttr>(input.getAxisMaps()[*inputAxis]);
        auto resultMap = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        if (!(sourceAxisIdentity(inputMap) == sourceAxisIdentity(resultMap)) &&
            input.getShape()[*inputAxis] == fragment.getShape()[axis]) {
          ReplayMaterializationOptions inputOptions = options;
          inputOptions.fragmentAxis = *inputAxis;
          FailureOr<Value> replayed = materializeReplayedValue(
              builder, location, reshape.getValue(), sourceAxisIdentity(inputMap),
              blockedExtent, mapping, inputOptions);
          if (failed(replayed))
            return failure();
          Value projected = builder.create<ReshapeOp>(
              location, replaceReplayAxis(fragment, axis), *replayed,
              reshape.getReassociation());
          remember(current, projected, projection);
          return projected;
        }
      }
    }

    if (auto contract = dyn_cast<ContractOp>(producer)) {
      SmallVector<std::pair<unsigned, unsigned>> freeAxes;
      auto lhsType = cast<FragmentType>(contract.getLhs().getType());
      auto rhsType = cast<FragmentType>(contract.getRhs().getType());
      for (unsigned inputAxis = 0; inputAxis < lhsType.getShape().size();
           ++inputAxis)
        if (!llvm::is_contained(contract.getLhsReductionAxes(),
                               static_cast<int64_t>(inputAxis)))
          freeAxes.emplace_back(0, inputAxis);
      for (unsigned inputAxis = 0; inputAxis < rhsType.getShape().size();
           ++inputAxis)
        if (!llvm::is_contained(contract.getRhsReductionAxes(),
                               static_cast<int64_t>(inputAxis)) &&
            !llvm::is_contained(contract.getRhsBatchAxes(),
                               static_cast<int64_t>(inputAxis)))
          freeAxes.emplace_back(1, inputAxis);
      if (axis >= freeAxes.size())
        return failure();
      // One SSA value can have different matrix roles at the two input uses.
      SmallVector<Value> operands{contract.getLhs(), contract.getRhs(),
                                   contract.getAccumulator()};
      auto replayAxis = [&](unsigned operand,
                            unsigned inputAxis) -> LogicalResult {
        auto type = cast<FragmentType>(operands[operand].getType());
        auto axisMap = cast<AxisMapAttr>(type.getAxisMaps()[inputAxis]);
        FailureOr<Value> replayed = failure();
        if (llvm::is_contained(replaySources, sourceAxisIdentity(axisMap))) {
          replayed = materialize(operands[operand], inputAxis);
        } else {
          ReplayMaterializationOptions inputOptions = options;
          inputOptions.fragmentAxis = inputAxis;
          replayed = materializeReplayedValue(
              builder, location, operands[operand], sourceAxisIdentity(axisMap),
              blockedExtent, mapping, inputOptions);
        }
        if (failed(replayed))
          return failure();
        operands[operand] = *replayed;
        return success();
      };
      auto [operand, inputAxis] = freeAxes[axis];
      if (failed(replayAxis(operand, inputAxis)) || failed(replayAxis(2, axis)))
        return failure();
      if (operand == 0)
        for (auto [batch, lhsAxis] : llvm::enumerate(contract.getLhsBatchAxes()))
          if (lhsAxis == static_cast<int64_t>(inputAxis) &&
              failed(replayAxis(1, contract.getRhsBatchAxes()[batch])))
            return failure();
      IRMapping cloneMapping(mapping);
      Operation *clone = builder.clone(*producer, cloneMapping);
      clone->setOperands(operands);
      clone->getResult(0).setType(replaceReplayAxis(fragment, axis));
      remember(current, clone->getResult(0), projection);
      return clone->getResult(0);
    }

    if (isa<BroadcastOp, SplatOp>(producer)) {
      FailureOr<Value> replayed = replayOperand(producer->getOperand(0));
      if (failed(replayed))
        return failure();
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, *replayed, replaceReplayAxis(fragment, axis));
      if (failed(projected))
        return failure();
      remember(current, *projected, projection);
      return *projected;
    }
    auto replayRead = [&](auto access, Value resource) -> FailureOr<Value> {
      SmallVector<Value> coordinates;
      for (Value coordinate : access.getCoordinates()) {
        auto replayed = replayOperand(coordinate);
        if (failed(replayed))
          return failure();
        coordinates.push_back(*replayed);
      }
      Value valid;
      if (access.getValid()) {
        auto replayed = replayOperand(access.getValid());
        if (failed(replayed))
          return failure();
        valid = *replayed;
      }
      FragmentType resultType = replaceReplayAxis(fragment, axis);
      auto combined = combineTail(resultType, valid);
      if (failed(combined))
        return failure();
      auto fill = replayFill(access.getFill(), resultType);
      if (failed(fill))
        return failure();
      auto clone = builder.create<decltype(access)>(
          location, resultType, resource, coordinates, *combined, *fill,
          access.getSourceAxes());
      if (Attribute origin = access->getAttr(originAttr))
        clone->setAttr(originAttr, origin);
      remember(current, clone.getResult(), projection);
      return clone.getResult();
    };
    if (auto load = dyn_cast<LoadOp>(producer))
      return replayRead(load, load.getResource());
    if (auto gather = dyn_cast<GatherOp>(producer)) {
      auto source = replayOperand(gather.getSource());
      if (failed(source))
        return failure();
      return replayRead(gather, *source);
    }
    bool pureBranch = isa<scf::IfOp>(producer) &&
                      options.scope == PhysicalReplayScope::ValueGraph &&
                      isMemoryEffectFree(producer);
    if (!pureBranch && !isPhysicalReplayNode(producer, options.scope,
                                            /*allowAccesses=*/false))
      return failure();
    IRMapping cloneMapping(mapping);
    llvm::SetVector<Value> replayOperands(producer->operand_begin(),
                                         producer->operand_end());
    if (pureBranch)
      for (Region &region : producer->getRegions())
        getUsedValuesDefinedAbove(region, replayOperands);
    for (Value operand : replayOperands) {
      FailureOr<Value> replayed = replayOperand(operand);
      if (failed(replayed))
        return failure();
      cloneMapping.map(operand, *replayed);
    }
    FragmentType pointwiseType;
    if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer)) {
      SmallVector<Value> operands;
      for (Value operand : producer->getOperands())
        operands.push_back(cloneMapping.lookupOrDefault(operand));
      auto refined = queryValueSchema(
          kernel, replaceReplayAxis(fragment, axis), operands);
      if (failed(refined))
        return failure();
      pointwiseType = *refined;
      for (Value operand : producer->getOperands()) {
        Value replayed = cloneMapping.lookupOrDefault(operand);
        Type element = replayed.getType();
        if (auto type = dyn_cast<FragmentType>(element))
          element = type.getElementType();
        auto target = FragmentType::get(
            kernel.getContext(), element, pointwiseType.getShape(),
            pointwiseType.getAxisMaps(), pointwiseType.getValidity(),
            pointwiseType.getOwner());
        auto projected = projectPhysicalValueToSchema(builder, location, replayed, target);
        if (failed(projected))
          return failure();
        cloneMapping.map(operand, *projected);
      }
    }
    Operation *clone = builder.clone(*producer, cloneMapping);
    bool structuredResults = pureBranch || isa<ReduceOp, ScanOp>(clone);
    for (auto [original, cloned] :
         llvm::zip(producer->getResults(), clone->getResults())) {
      if (structuredResults)
        cloned.setType(replaceReplayType(cloned.getType()));
      if (!hasMultipleReplayAxes(original.getType()))
        mapping.map(original, cloned);
    }
    auto result = dyn_cast<OpResult>(current);
    if (!result || result.getResultNumber() >= clone->getNumResults())
      return failure();
    Value clonedValue = clone->getResult(result.getResultNumber());
    if (pureBranch)
      for (Region &region : clone->getRegions())
        retargetHelperSourceExtent(
            region, cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    if (auto clonedReduce = dyn_cast<ReduceOp>(clone))
      retargetHelperSourceExtent(
          clonedReduce.getCombine(),
          cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    if (auto clonedScan = dyn_cast<ScanOp>(clone))
      retargetHelperSourceExtent(
          clonedScan.getCombine(),
          cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    auto clonedType = dyn_cast<FragmentType>(clonedValue.getType());
    if (!clonedType || axis >= clonedType.getShape().size())
      return failure();
    bool introducedUnitAxis = isIntroducedReshapeUnitAxis(current, axis);
    if (pointwiseType)
      clonedValue.setType(pointwiseType);
    else if (!introducedUnitAxis)
      clonedValue.setType(replaceReplayAxis(clonedType, axis));
    remember(current, clonedValue, projection);
    return clonedValue;
  };

  FailureOr<Value> result = materialize(value, options.fragmentAxis);
  return projectionFailed ? FailureOr<Value>(failure()) : result;
}

FailureOr<Value> projectPredicateToFragmentAxis(OpBuilder &builder,
                                                Location location,
                                                Value predicate,
                                                FragmentType target,
                                                unsigned fragmentAxis) {
  return projectPredicate(builder, location, predicate, target, fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection projection = queryFragmentAxis(target, source);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            int64_t dimensionId) {
  PhysicalDimensionProjection projection =
      queryFragmentDimension(target, dimensionId);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> materializeValidityConjunction(
    OpBuilder &builder, Location location, Value lhs, Value rhs,
    FragmentType valueType) {
  auto predicateType = FragmentType::get(
      valueType.getContext(), builder.getI1Type(), valueType.getShape(),
      valueType.getAxisMaps(), valueType.getValidity(), valueType.getOwner());
  Value result;
  if (lhs) {
    FailureOr<Value> projected =
        projectPhysicalValueToSchema(builder, location, lhs, predicateType);
    if (failed(projected))
      return failure();
    result = *projected;
  }
  if (rhs) {
    FailureOr<Value> projected =
        projectPhysicalValueToSchema(builder, location, rhs, predicateType);
    if (failed(projected))
      return failure();
    result = result ? Value(builder.create<BinaryOp>(
                          location, predicateType, result, *projected,
                          BinaryOperator::LogicalAnd))
                    : *projected;
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

FailureOr<Value> materializeRetargetedValidity(
    OpBuilder &builder, Location location, Value original,
    ArrayRef<std::pair<MakeRangeOp, Value>> originalTailRanges,
    Value physicalTail, FragmentType target) {
  func::FuncOp kernel;
  if (!originalTailRanges.empty())
    kernel = originalTailRanges.front().first->getParentOfType<func::FuncOp>();
  else if (original)
    if (Operation *definition = original.getDefiningOp())
      kernel = definition->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<std::pair<MakeRangeOp, Value>> equivalentTailRanges(
      originalTailRanges.begin(), originalTailRanges.end());
  if (original && isa<FragmentType>(original.getType())) {
    for (auto [expected, end] : originalTailRanges) {
      auto axis =
          queryFragmentAxis(original.getType(), sourceAxisIdentity(expected));
      auto dimension = queryRangeDimension(expected);
      if (!axis.isExact() || failed(dimension) || axis.dimensionId != *dimension)
        continue;
      auto ranges = analysis.axisRanges(original, axis.fragmentAxis);
      if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
        continue;
      SmallVector<MakeRangeOp> coincident{expected};
      llvm::append_range(coincident, ranges.roots);
      if (!analysis.lockstepRanges(coincident).isExact())
        continue;
      // A predicate may carry the output's coordinate identity on this input
      // axis. Keep the positional proof as well as equal physical traversal;
      // equal range bounds alone do not equate independent Cartesian axes.
      for (MakeRangeOp range : ranges.roots)
        if (!llvm::is_contained(equivalentTailRanges,
                                std::pair<MakeRangeOp, Value>{range, end}))
          equivalentTailRanges.emplace_back(range, end);
    }
  }

  std::function<FailureOr<Value>(Value)> residual =
      [&](Value value) -> FailureOr<Value> {
    if (!value || analysis.isTailPredicate(value, equivalentTailRanges))
      return Value();
    if (value.getType().isInteger(1))
      return value;
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return residual(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return residual(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return residual(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return residual(transpose.getValue());
    auto conjunction = value.getDefiningOp<BinaryOp>();
    if (!conjunction)
      return failure();
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    if (!logical && !bitwiseI1)
      return failure();
    FailureOr<Value> lhs = residual(conjunction.getLhs());
    FailureOr<Value> rhs = residual(conjunction.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!*lhs)
      return *rhs;
    if (!*rhs)
      return *lhs;
    if (!(*lhs).getType().isInteger(1) || !(*rhs).getType().isInteger(1))
      return failure();
    return Value(builder.create<BinaryOp>(location, builder.getI1Type(), *lhs,
                                          *rhs,
                                          BinaryOperator::LogicalAnd));
  };

  FailureOr<Value> authorResidual = residual(original);
  if (failed(authorResidual))
    return failure();
  if (!*authorResidual)
    return physicalTail;
  if (!physicalTail)
    return projectPhysicalValueToSchema(builder, location, *authorResidual,
                                          target);
  return materializeValidityConjunction(builder, location, physicalTail,
                                        *authorResidual, target);
}

} // namespace intent::gpu

#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/Traversal.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

#include <limits>

using namespace mlir;

namespace intent::gpu {
namespace {

bool isZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  return integer && integer.getValue().isZero();
}

FailureOr<Value> replayFragmentValue(OpBuilder &builder, Value value,
                                     FragmentType target, IRMapping &mapping,
                                     PhysicalProgramAnalysis &analysis) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    Value replacement;
    for (const auto &entry : mapping.getValueMap()) {
      auto known = entry.first.getDefiningOp<MakeRangeOp>();
      if (!known || !sameLogicalRange(range, known) ||
          !samePhysicalScalarExpression(range.getStart(), known.getStart()) ||
          !samePhysicalScalarExpression(range.getExtent(), known.getExtent()))
        continue;
      if (replacement && replacement != entry.second)
        return failure();
      replacement = entry.second;
    }
    if (!replacement)
      return failure();
    mapping.map(value, replacement);
    return replacement;
  }
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record)
      return failure();
    FailureOr<Value> field = replayFragmentValue(
        builder, record.getFields()[extract.getField()], target, mapping, analysis);
    if (failed(field))
      return failure();
    mapping.map(value, *field);
    return *field;
  }
  PhysicalReplayFact replay = analysis.replayability(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false);
  Operation *producer = value.getDefiningOp();
  if (!producer || !replay.isReplayable())
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayFragmentValue(builder, operand, target, mapping, analysis);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  if (isa<BroadcastOp, ReshapeOp, TransposeOp>(producer)) {
    auto resultType = FragmentType::get(
        target.getContext(), fragment.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    // Replayed coordinates already use the destination axes.  The old shape
    // relation must not introduce its singleton axes a second time.
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, producer->getLoc(),
        mapping.lookupOrDefault(producer->getOperand(0)), resultType);
    if (failed(projected))
      return failure();
    mapping.map(value, *projected);
    return *projected;
  }
  Operation *clone = builder.clone(*producer, mapping);
  for (Value result : clone->getResults()) {
    auto original = dyn_cast<FragmentType>(result.getType());
    if (!original)
      continue;
    result.setType(FragmentType::get(
        target.getContext(), original.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner()));
  }
  Value result = clone->getResult(0);
  mapping.map(value, result);
  return result;
}

FailureOr<Value> replayScalarValue(OpBuilder &builder, Value value,
                                   IRMapping &mapping,
                                   PhysicalProgramAnalysis &analysis) {
  if (!value)
    return Value();
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  if (Operation *projection = value.getDefiningOp();
      projection && isa<BroadcastOp, ReshapeOp, TransposeOp, SplatOp>(projection)) {
    FailureOr<Value> scalar =
        replayScalarValue(builder, projection->getOperand(0), mapping, analysis);
    if (succeeded(scalar))
      mapping.map(value, *scalar);
    return scalar;
  }
  PhysicalReplayFact replay = analysis.replayability(
      value, std::nullopt, PhysicalReplayScope::Coordinate,
      /*allowAccesses=*/false);
  Operation *producer = value.getDefiningOp();
  if (!producer || isa<MakeRangeOp>(producer) || !replay.isReplayable() ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(producer))
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement =
        replayScalarValue(builder, operand, mapping, analysis);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  Operation *clone = builder.clone(*producer, mapping);
  for (Value result : clone->getResults())
    if (auto resultType = dyn_cast<FragmentType>(result.getType()))
      result.setType(resultType.getElementType());
  Value result = clone->getResult(0);
  mapping.map(value, result);
  return result;
}

FailureOr<Value> combinePredicates(OpBuilder &builder, Location location,
                                   Type valueType, Value lhs,
                                   Value rhs) {
  Type predicate = builder.getI1Type();
  if (auto fragment = dyn_cast<FragmentType>(valueType))
    predicate = FragmentType::get(
        fragment.getContext(), builder.getI1Type(), fragment.getShape(),
        fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
  for (Value *value : {&lhs, &rhs}) {
    if (!*value)
      continue;
    if ((*value).getType() != predicate) {
      FailureOr<Value> projected =
          projectPhysicalValueToSchema(builder, location, *value, predicate);
      if (failed(projected))
        return failure();
      *value = *projected;
    }
  }
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return Value(builder.create<BinaryOp>(location, predicate, lhs, rhs,
                                        BinaryOperator::LogicalAnd));
}

FailureOr<Value> gatherBounds(OpBuilder &builder, Location location,
                              FragmentType source, ValueRange coordinates,
                              ArrayRef<int64_t> axes, Type indexType) {
  Type predicate = builder.getI1Type();
  if (auto fragment = dyn_cast<FragmentType>(indexType))
    predicate = FragmentType::get(builder.getContext(), builder.getI1Type(),
        fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  auto lower = projectPhysicalValueToSchema(builder, location, zero, indexType);
  if (failed(lower))
    return failure();
  Value valid;
  for (auto [coordinate, axis] : llvm::zip(coordinates, axes)) {
    auto projected = projectPhysicalValueToSchema(builder, location, coordinate, indexType);
    auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
    Value bound = extent.getKind() == PhysicalExprKind::Constant
        ? Value(builder.create<arith::ConstantIndexOp>(location, extent.getValue()))
        : Value(builder.create<PhysicalExprOp>(location, builder.getIndexType(), extent));
    auto upper = projectPhysicalValueToSchema(builder, location, bound, indexType);
    if (failed(projected) || failed(upper))
      return failure();
    Value nonnegative = builder.create<CompareOp>(location, predicate,
        *projected, *lower, ComparePredicate::Ge);
    Value below = builder.create<CompareOp>(location, predicate,
        *projected, *upper, ComparePredicate::Lt);
    Value bounded = builder.create<BinaryOp>(location, predicate,
        nonnegative, below, BinaryOperator::LogicalAnd);
    valid = valid ? Value(builder.create<BinaryOp>(location, predicate,
        valid, bounded, BinaryOperator::LogicalAnd)) : bounded;
  }
  return valid;
}

bool foldIndexRecompositions(func::FuncOp kernel) {
  bool changed = false;
  SmallVector<BinaryOp> quotients;
  kernel.walk([&](BinaryOp binary) {
    if (binary.getOperatorKind() == BinaryOperator::FloorDivide &&
        uniformElementType(binary.getResult().getType()).isIndex())
      quotients.push_back(binary);
  });
  auto unproject = [](Value value) {
    while (true) {
      if (auto broadcast = value.getDefiningOp<BroadcastOp>())
        value = broadcast.getValue();
      else if (auto reshape = value.getDefiningOp<ReshapeOp>())
        value = reshape.getValue();
      else
        return value;
    }
  };
  for (BinaryOp quotient : quotients) {
    auto range = unproject(quotient.getLhs()).getDefiningOp<MakeRangeOp>();
    Value divisor = unproject(quotient.getRhs());
    if (!range || !isUnitStepRange(range) ||
        !isZero(range.getLogicalStart()) ||
        !samePhysicalScalarExpression(range.getStart(),
                                      range.getLogicalStart()) ||
        !samePhysicalScalarExpression(range.getLogicalStop(), divisor))
      continue;
    // Every active logical member of [0, end) has quotient zero. An empty
    // range has no active members; physical padding retains its existing mask.
    OpBuilder builder(quotient);
    Value zero = builder.create<arith::ConstantIndexOp>(quotient.getLoc(), 0);
    if (auto fragment = dyn_cast<FragmentType>(quotient.getResult().getType()))
      zero = builder.create<SplatOp>(quotient.getLoc(), fragment, zero);
    quotient.getResult().replaceAllUsesWith(zero);
    quotient.erase();
    changed = true;
  }
  SmallVector<BinaryOp> sums;
  kernel.walk([&](BinaryOp op) {
    if (op.getOperatorKind() == BinaryOperator::Add ||
        op.getOperatorKind() == BinaryOperator::Subtract)
      sums.push_back(op);
  });
  for (BinaryOp sum : sums) {
    Type type = sum.getResult().getType();
    Type element = uniformElementType(type);
    if (!isa<IndexType, IntegerType>(element))
      continue;
    unsigned width =
        element.isIndex() ? 64 : cast<IntegerType>(element).getWidth();
    UniformValueAnalysis constants(describeUniformValue);
    auto constant = [&](Value value) -> std::optional<llvm::APInt> {
      auto attr = dyn_cast_or_null<IntegerAttr>(constants.evaluate(value));
      if (!attr || attr.getType() != element)
        return std::nullopt;
      return attr.getValue();
    };
    llvm::MapVector<Value, llvm::APInt> terms;
    auto addTerm = [&](Value value, const llvm::APInt &coefficient) {
      auto [it, inserted] = terms.insert({value, llvm::APInt(width, 0)});
      it->second += coefficient;
    };
    llvm::APInt offset(width, 0);
    SmallVector<std::pair<Value, llvm::APInt>> pending{
        {sum.getResult(), llvm::APInt(width, 1)}};
    bool compatible = true;
    while (!pending.empty()) {
      auto [value, coefficient] = pending.pop_back_val();
      if (value.getType() != type) {
        compatible = false;
        break;
      }
      if (auto literal = constant(value)) {
        offset += coefficient * *literal;
        continue;
      }
      auto binary = value.getDefiningOp<BinaryOp>();
      // Preserve shared arithmetic subexpressions and avoid expanding a DAG.
      if (binary && (value == sum.getResult() || value.hasOneUse())) {
        auto kind = binary.getOperatorKind();
        if (kind == BinaryOperator::Add || kind == BinaryOperator::Subtract) {
          pending.emplace_back(binary.getLhs(), coefficient);
          pending.emplace_back(
              binary.getRhs(),
              kind == BinaryOperator::Add ? coefficient : -coefficient);
          continue;
        }
        if (kind == BinaryOperator::Multiply) {
          if (auto scale = constant(binary.getRhs())) {
            pending.emplace_back(binary.getLhs(), coefficient * *scale);
            continue;
          }
          if (auto scale = constant(binary.getLhs())) {
            pending.emplace_back(binary.getRhs(), coefficient * *scale);
            continue;
          }
        }
      }
      addTerm(value, coefficient);
    }
    if (!compatible)
      continue;

    bool recomposed = false;
    bool merged;
    do {
      merged = false;
      for (auto [remainderValue, scale] : terms) {
        auto remainder = remainderValue.getDefiningOp<BinaryOp>();
        if (scale.isZero() || !remainder ||
            remainder.getLhs().getType() != type ||
            remainder.getOperatorKind() != BinaryOperator::Remainder)
          continue;
        auto divisor = constant(remainder.getRhs());
        if (!divisor || !divisor->isStrictlyPositive())
          continue;
        Value quotientValue;
        for (auto [value, coefficient] : terms) {
          auto quotient = value.getDefiningOp<BinaryOp>();
          if (quotient &&
              quotient.getOperatorKind() == BinaryOperator::FloorDivide &&
              quotient.getLhs() == remainder.getLhs() &&
              constant(quotient.getRhs()) == divisor &&
              coefficient == scale * *divisor) {
            quotientValue = value;
            break;
          }
        }
        if (!quotientValue)
          continue;
        // q*c+r=x also holds modulo 2^width, including negative dividends.
        // Coefficients retain that width; no address or bounds assumptions enter.
        Value dividend = remainder.getLhs();
        terms.erase(quotientValue);
        terms.erase(remainderValue);
        addTerm(dividend, scale);
        merged = recomposed = true;
        break;
      }
    } while (merged);
    if (!recomposed)
      continue;

    OpBuilder builder(sum);
    auto literal = [&](const llvm::APInt &bits) -> Value {
      Value value = builder.create<arith::ConstantOp>(
          sum.getLoc(), IntegerAttr::get(element, bits));
      if (auto fragment = dyn_cast<FragmentType>(type))
        value = builder.create<SplatOp>(sum.getLoc(), fragment, value);
      return value;
    };
    Value result;
    for (auto [value, coefficient] : terms) {
      if (coefficient.isZero())
        continue;
      Value term = value;
      if (!coefficient.isOne())
        term = builder.create<BinaryOp>(sum.getLoc(), type, value,
                                       literal(coefficient),
                                       BinaryOperator::Multiply);
      result = result ? Value(builder.create<BinaryOp>(
                            sum.getLoc(), type, result, term, BinaryOperator::Add))
                      : term;
    }
    if (!offset.isZero() || !result) {
      Value term = literal(offset);
      result = result ? Value(builder.create<BinaryOp>(
                            sum.getLoc(), type, result, term, BinaryOperator::Add))
                      : term;
    }
    sum.getResult().replaceAllUsesWith(result);
    sum.erase();
    changed = true;
  }
  return changed;
}

bool sameUniformValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  UniformValueAnalysis constants(describeUniformValue);
  if (equalUniformConstants(constants.evaluate(lhs), constants.evaluate(rhs)))
    return true;
  auto scalar = [](Value value) {
    while (value) {
      UniformExpression expression = describeUniformValue(value);
      if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
        break;
      value = expression.operands.front();
    }
    return value;
  };
  lhs = scalar(lhs);
  rhs = scalar(rhs);
  if (!lhs || !rhs || isa<FragmentType>(lhs.getType()) ||
      isa<FragmentType>(rhs.getType()))
    return false;
  if (lhs == rhs)
    return true;
  return false;
}

FailureOr<bool> composeSelectLoad(SelectOp select) {
  auto resultType = dyn_cast<FragmentType>(select.getResult().getType());
  Value loaded = select.getTrueValue();
  SmallVector<Operation *> projections;
  while (Operation *operation = loaded.getDefiningOp()) {
    if (!isa<BroadcastOp, TransposeOp, ReshapeOp>(operation))
      break;
    auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
    auto result = dyn_cast<FragmentType>(loaded.getType());
    if (!source || !result || !loaded.hasOneUse())
      return false;
    if (auto transpose = dyn_cast<TransposeOp>(operation)) {
      for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
        auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        auto transposed = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
        if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
            original.getDimensionId() != transposed.getDimensionId())
          return false;
      }
    } else if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
      unsigned sourceRank = 0, resultRank = 0;
      for (Attribute attribute : reshape.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        if (!group.getSourceAxes().empty())
          sourceRank = group.getSourceAxes().asArrayRef().back() + 1;
        if (!group.getResultAxes().empty())
          resultRank = group.getResultAxes().asArrayRef().back() + 1;
      }
      unsigned sourcePrefix = source.getShape().size() - sourceRank;
      unsigned resultPrefix = result.getShape().size() - resultRank;
      auto unitAxes = [](FragmentType type, unsigned prefix,
                         ArrayRef<int64_t> axes) {
        return llvm::all_of(axes, [&](int64_t axis) {
          auto extent = cast<PhysicalExprAttr>(type.getShape()[prefix + axis]);
          return extent.getKind() ==
                     PhysicalExprKind::Constant &&
                 extent.getValue() == 1;
        });
      };
      for (Attribute attribute : reshape.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        auto from = group.getSourceAxes().asArrayRef();
        auto to = group.getResultAxes().asArrayRef();
        if (from.empty() || to.empty()) {
          if (!unitAxes(source, sourcePrefix, from) ||
              !unitAxes(result, resultPrefix, to))
            return false;
          continue;
        }
        if (from.size() != 1 || to.size() != 1)
          return false;
        unsigned sourceAxis = sourcePrefix + from.front();
        unsigned resultAxis = resultPrefix + to.front();
        auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
        auto target = cast<AxisMapAttr>(result.getAxisMaps()[resultAxis]);
        if (!(sourceAxisIdentity(original) == sourceAxisIdentity(target)) ||
            original.getDimensionId() != target.getDimensionId() ||
            source.getShape()[sourceAxis] != result.getShape()[resultAxis])
          return false;
      }
    } else {
      auto projection = queryAxisProjection(source, result);
      if (source.getShape() != result.getShape() || !projection.isExact() ||
          llvm::any_of(llvm::enumerate(projection.targetToSource), [](auto item) {
            return !item.value() || *item.value() != item.index();
          }))
        return false;
    }
    projections.push_back(operation);
    loaded = operation->getOperand(0);
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  if (!resultType || !load || !load.getResult().hasOneUse() ||
      !canReplayReadAt(load, select))
    return false;

  OpBuilder builder(select);
  auto projectBack = [&](Value value) -> FailureOr<Value> {
    for (Operation *projection : projections) {
      auto source = cast<FragmentType>(projection->getOperand(0).getType());
      auto result = cast<FragmentType>(projection->getResult(0).getType());
      Type element = uniformElementType(value.getType());
      auto sourceSchema = FragmentType::get(
          source.getContext(), element, source.getShape(), source.getAxisMaps(),
          source.getValidity(), source.getOwner());
      if (auto transpose = dyn_cast<TransposeOp>(projection)) {
        auto resultSchema = FragmentType::get(
            result.getContext(), element, result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        auto projected = projectPhysicalValueToSchema(
            builder, select.getLoc(), value, resultSchema);
        if (failed(projected))
          return failure();
        SmallVector<int64_t> inverse(transpose.getPermutation().size());
        for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation()))
          inverse[sourceAxis] = axis;
        value = builder.create<TransposeOp>(select.getLoc(), sourceSchema,
                                            *projected, inverse);
      } else if (auto reshape = dyn_cast<ReshapeOp>(projection)) {
        auto resultSchema = FragmentType::get(
            result.getContext(), element, result.getShape(), result.getAxisMaps(),
            result.getValidity(), result.getOwner());
        auto projected = projectPhysicalValueToSchema(
            builder, select.getLoc(), value, resultSchema);
        if (failed(projected))
          return failure();
        SmallVector<Attribute> inverse;
        for (Attribute attribute : reshape.getReassociation()) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          inverse.push_back(ReshapeGroupAttr::get(
              select.getContext(), group.getResultAxes(), group.getSourceAxes()));
        }
        value = builder.create<ReshapeOp>(
            select.getLoc(), sourceSchema, *projected, builder.getArrayAttr(inverse));
      } else {
        auto projected = projectPhysicalValueToSchema(
            builder, select.getLoc(), value, sourceSchema);
        if (failed(projected))
          return failure();
        value = *projected;
      }
    }
    return value;
  };
  auto condition = projectBack(select.getCondition());
  auto projectedFill = projectBack(select.getFalseValue());
  if (failed(condition) || failed(projectedFill))
    return false;
  Value fill = *projectedFill;
  if (load.getValid()) {
    if (!load.getFill())
      return false;
  } else if (load.getFill()) {
    return false;
  }

  auto loadedType = projections.empty()
                        ? resultType
                        : cast<FragmentType>(load.getResult().getType());
  FailureOr<Value> valid = combinePredicates(
      builder, select.getLoc(), loadedType, load.getValid(), *condition);
  if (failed(valid)) {
    select.emitOpError(
        "masked load predicate cannot adopt the loaded value relation");
    return failure();
  }
  if (fill.getType() != loadedType) {
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, select.getLoc(), fill, loadedType);
    if (failed(projected)) {
      select.emitOpError(
          "masked load fill cannot adopt the loaded value relation");
      return failure();
    }
    fill = *projected;
  }
  if (load.getValid() && !sameUniformValue(load.getFill(), fill)) {
    auto sourceFill = projectPhysicalValueToSchema(
        builder, select.getLoc(), load.getFill(), loadedType);
    auto predicate = combinePredicates(
        builder, select.getLoc(), loadedType, Value(), *condition);
    if (failed(sourceFill) || failed(predicate))
      return false;
    fill = builder.create<SelectOp>(select.getLoc(), loadedType, *predicate,
                                    *sourceFill, fill);
  }
  auto replacement = builder.create<LoadOp>(
      select.getLoc(), loadedType, load.getResource(), load.getCoordinates(),
      *valid, fill, load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  // Keep the original load and value schemas. Only the predicate and fill
  // travel backwards through the inverse projection.
  IRMapping mapping;
  mapping.map(load.getResult(), replacement.getResult());
  for (Operation *projection : llvm::reverse(projections))
    builder.clone(*projection, mapping);
  select.getResult().replaceAllUsesWith(
      mapping.lookupOrDefault(select.getTrueValue()));
  select.erase();
  for (Operation *projection : projections)
    projection->erase();
  load.erase();
  return true;
}

FailureOr<bool> composeReshapedGather(GatherOp gather) {
  auto reshape = gather.getSource().getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!reshape)
    return false;
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto shaped = cast<FragmentType>(reshape.getResult().getType());
  // Scalar indexing is a rectangular extraction in the reshaped layout.
  // Flattening it can turn a native split into a strided fragment gather.
  if (result && !gather.getCoordinates().empty() &&
      llvm::all_of(llvm::zip(gather.getCoordinates(), gather.getSourceAxes()),
                   [&](auto entry) {
        Value coordinate = std::get<0>(entry);
        int64_t axis = std::get<1>(entry);
        while (isa<FragmentType>(coordinate.getType())) {
          UniformExpression expression = describeUniformValue(coordinate);
          if (expression.kind != UniformKind::Forward ||
              expression.operands.size() != 1)
            break;
          coordinate = expression.operands.front();
        }
        if (!isa<FragmentType>(coordinate.getType()))
          return true;
        auto range = coordinate.getDefiningOp<MakeRangeOp>();
        return range && isZero(range.getStart()) && isUnitStepRange(range) &&
               queryLaunchExpression(range.getExtent()) ==
                   shaped.getShape()[axis];
      }))
    return false;
  if (source.getShape() == shaped.getShape() &&
      llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        return group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1 &&
               group.getSourceAxes()[0] == group.getResultAxes()[0];
      }))
    return false;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  for (unsigned axis = 0; axis < source.getShape().size(); ++axis)
    if (analysis.axisRealization(reshape.getValue(), axis).constructionScalarSeed)
      return false;
  for (unsigned axis = 0; axis < shaped.getShape().size(); ++axis)
    if (analysis.axisRealization(reshape.getResult(), axis).constructionScalarSeed)
      return false;
  unsigned sourceRank = 0, resultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    sourceRank += group.getSourceAxes().size();
    resultRank += group.getResultAxes().size();
  }
  unsigned sourcePrefix = source.getShape().size() - sourceRank;
  unsigned resultPrefix = shaped.getShape().size() - resultRank;
  if (sourcePrefix != resultPrefix)
    return false;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1)
      continue;
    for (auto [type, axes] :
         {std::pair{source, group.getSourceAxes().asArrayRef()},
          std::pair{shaped, group.getResultAxes().asArrayRef()}})
      for (int64_t axis : axes) {
        auto extent = cast<PhysicalExprAttr>(type.getShape()[sourcePrefix + axis]);
        if (extent.getKind() != PhysicalExprKind::Constant ||
            extent.getValue() <= 0)
          return false;
      }
  }
  SmallVector<Value> coordinates(shaped.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes()))
    coordinates[axis] = coordinate;
  SmallVector<unsigned> retainedAxes(shaped.getShape().size());
  for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
    if (coordinates[axis])
      continue;
    if (!result)
      return false;
    auto mapping = cast<AxisMapAttr>(shaped.getAxisMaps()[axis]);
    auto projection = queryFragmentAxis(result, sourceAxisIdentity(mapping),
                                        mapping.getDimensionId());
    if (!projection.isExact() || projection.dimensionId != mapping.getDimensionId() ||
        result.getShape()[projection.fragmentAxis] != shaped.getShape()[axis])
      return false;
    retainedAxes[axis] = projection.fragmentAxis;
  }
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  Type indexType = builder.getIndexType();
  if (result)
    indexType = FragmentType::get(
        result.getContext(), builder.getIndexType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
  auto indexValue = [&](Value value) -> Value {
    return result ? Value(builder.create<SplatOp>(location, indexType, value))
                  : value;
  };
  auto extentValue = [&](Attribute attribute) -> Value {
    auto extent = cast<PhysicalExprAttr>(attribute);
    if (extent.getKind() == PhysicalExprKind::Constant)
      return builder.create<arith::ConstantIndexOp>(location, extent.getValue());
    return builder.create<PhysicalExprOp>(location, builder.getIndexType(), extent);
  };
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  for (unsigned axis = 0; axis < coordinates.size(); ++axis) {
    Value coordinate = coordinates[axis];
    if (!coordinate) {
      auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[retainedAxes[axis]]);
      auto type = FragmentType::get(
          result.getContext(), builder.getIndexType(),
          builder.getArrayAttr({shaped.getShape()[axis]}),
          builder.getArrayAttr({AxisMapAttr::get(
              result.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), 0, mapping.getDerived())}),
          result.getValidity(), result.getOwner());
      Value extent = extentValue(shaped.getShape()[axis]);
      coordinate = builder.create<MakeRangeOp>(
          location, type, zero, extent, one, zero, extent,
          mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
    }
    Type element = coordinate.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (!element.isIndex()) {
      Type target = builder.getIndexType();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        target = FragmentType::get(builder.getContext(), builder.getIndexType(),
            fragment.getShape(), fragment.getAxisMaps(), fragment.getValidity(),
            fragment.getOwner());
      coordinate = builder.create<CastOp>(location, target, coordinate);
    }
    auto projected = projectPhysicalValueToSchema(builder, location, coordinate, indexType);
    if (failed(projected))
      return gather.emitOpError("reshape gather lost an exact result coordinate projection");
    coordinates[axis] = *projected;
  }
  SmallVector<Value> selected(source.getShape().size());
  for (unsigned axis = 0; axis < sourcePrefix; ++axis)
    selected[axis] = coordinates[axis];
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty())
      continue;
    if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() == 1) {
      selected[sourcePrefix + group.getSourceAxes()[0]] =
          coordinates[resultPrefix + group.getResultAxes()[0]];
      continue;
    }
    Value ordinal = indexValue(zero);
    for (int64_t axis : group.getResultAxes().asArrayRef()) {
      Value extent = indexValue(extentValue(shaped.getShape()[resultPrefix + axis]));
      ordinal = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                        BinaryOperator::Multiply);
      ordinal = builder.create<BinaryOp>(location, indexType, ordinal,
          coordinates[resultPrefix + axis], BinaryOperator::Add);
    }
    auto axes = group.getSourceAxes().asArrayRef();
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned axis = sourcePrefix + axes[position];
      Value coordinate = ordinal;
      if (position != 0) {
        Value extent = indexValue(extentValue(source.getShape()[axis]));
        coordinate = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                              BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(location, indexType, ordinal, extent,
                                           BinaryOperator::FloorDivide);
      }
      selected[axis] = coordinate;
    }
  }
  SmallVector<int64_t> axes;
  Type predicate = builder.getI1Type();
  if (result)
    predicate = FragmentType::get(result.getContext(), builder.getI1Type(),
        result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
  Value valid = gather.getValid();
  Value lower = indexValue(zero);
  for (unsigned axis = 0; axis < selected.size(); ++axis)
  {
    axes.push_back(axis);
    Value upper = indexValue(extentValue(source.getShape()[axis]));
    Value nonnegative = builder.create<CompareOp>(location, predicate,
        selected[axis], lower, ComparePredicate::Ge);
    Value below = builder.create<CompareOp>(location, predicate,
        selected[axis], upper, ComparePredicate::Lt);
    Value bounded = builder.create<BinaryOp>(location, predicate, nonnegative,
                                            below, BinaryOperator::LogicalAnd);
    auto combined = combinePredicates(builder, location, gather.getResult().getType(), valid, bounded);
    if (failed(combined))
      return failure();
    valid = *combined;
  }
  Value fill = gather.getFill();
  if (!fill) {
    if (result) {
      auto zeroFill = materializeZeroFragment(builder, location, result);
      if (failed(zeroFill))
        return failure();
      fill = *zeroFill;
    } else {
      fill = builder.create<arith::ConstantOp>(
          location, builder.getZeroAttr(gather.getResult().getType()));
    }
  }
  auto replacement = builder.create<GatherOp>(location, gather.getResult().getType(), reshape.getValue(),
      selected, valid, fill, axes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

FailureOr<bool> composeRangeGather(GatherOp gather) {
  auto range = gather.getSource().getDefiningOp<MakeRangeOp>();
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!range || !result || gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
      !uniformElementType(gather.getCoordinates().front().getType()).isIndex())
    return false;
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto coordinate = projectPhysicalValueToSchema(
      builder, location, gather.getCoordinates().front(), result);
  if (failed(coordinate))
    return gather.emitOpError("range gather lost its result coordinate projection");
  Value start = builder.create<SplatOp>(location, result, range.getStart());
  Value step = builder.create<SplatOp>(location, result, range.getStep());
  Value offset = builder.create<BinaryOp>(location, result, *coordinate, step,
                                         BinaryOperator::Multiply);
  Value value = builder.create<BinaryOp>(location, result, start, offset,
                                        BinaryOperator::Add);
  if (gather.getValid())
    value = builder.create<SelectOp>(location, result, gather.getValid(), value,
                                     gather.getFill());
  gather.getResult().replaceAllUsesWith(value);
  gather.erase();
  return true;
}

FailureOr<bool> composePointwiseGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  Operation *producer = gather.getSource().getDefiningOp();
  if (!source || !producer ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
           BroadcastOp, SplatOp, ReshapeOp>(producer) ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;
  if (auto binary = dyn_cast<BinaryOp>(producer);
      binary && !isa<FloatType>(source.getElementType())) {
    // Masked operands below are filled with zero. Do not introduce undefined
    // integer arithmetic on those inactive lanes.
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
    case BinaryOperator::Multiply:
    case BinaryOperator::Maximum:
    case BinaryOperator::Minimum:
    case BinaryOperator::LogicalAnd:
    case BinaryOperator::LogicalOr:
    case BinaryOperator::BitwiseAnd:
    case BinaryOperator::BitwiseOr:
    case BinaryOperator::BitwiseXor:
      break;
    default:
      return false;
    }
  }
  bool positionalRebind = false;
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    auto input = cast<FragmentType>(reshape.getValue().getType());
    positionalRebind = input.getShape() == source.getShape() &&
        llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getSourceAxes().size() == 1 &&
                 group.getResultAxes().size() == 1 &&
                 group.getSourceAxes()[0] == group.getResultAxes()[0];
        });
    if (!positionalRebind)
      return false;
  }
  SmallVector<Value> coordinates(source.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (axis < 0 || axis >= static_cast<int64_t>(coordinates.size()) ||
        coordinates[axis])
      return false;
    coordinates[axis] = coordinate;
  }
  if (llvm::any_of(coordinates, [](Value value) { return !value; }))
    return false;
  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  auto broadcastUnit = [&](Value operand, unsigned axis) {
    auto input = cast<FragmentType>(operand.getType());
    auto extent = cast<PhysicalExprAttr>(input.getShape()[axis]);
    if (extent.getKind() != PhysicalExprKind::Constant ||
        extent.getValue() != 1)
      return false;
    PhysicalRangeFact ranges = analysis.axisRanges(operand, axis);
    return ranges.isExact() && llvm::all_of(ranges.roots, [](MakeRangeOp range) {
      return isProvablySingletonLogicalRange(range);
    });
  };
  SmallVector<SmallVector<unsigned>> operandAxes;
  for (Value operand : producer->getOperands()) {
    SmallVector<unsigned> axes;
    if (auto input = dyn_cast<FragmentType>(operand.getType())) {
      if (positionalRebind) {
        for (unsigned axis = 0; axis < input.getShape().size(); ++axis)
          axes.push_back(axis);
        operandAxes.push_back(std::move(axes));
        continue;
      }
      BroadcastProjection projection = queryAxisProjection(input, source);
      if (!projection.isExact())
        return false;
      for (unsigned inputAxis = 0; inputAxis < input.getShape().size(); ++inputAxis) {
        std::optional<unsigned> selected;
        for (auto [axis, mapped] : llvm::enumerate(projection.targetToSource))
          if (mapped && *mapped == inputAxis) {
            if (selected)
              return false;
            selected = axis;
          }
        if (!selected)
          return false;
        if (!broadcastUnit(operand, inputAxis) &&
            input.getShape()[inputAxis] != source.getShape()[*selected])
          return false;
        axes.push_back(*selected);
      }
    }
    operandAxes.push_back(std::move(axes));
  }
  OpBuilder builder(gather);
  IRMapping mapping;
  auto selectedTypeFor = [&](Type element) -> Type {
    return result ? Type(FragmentType::get(
                        result.getContext(), element, result.getShape(),
                        result.getAxisMaps(), result.getValidity(), result.getOwner()))
                  : element;
  };
  for (auto [operand, axes] : llvm::zip(producer->getOperands(), operandAxes)) {
    auto input = dyn_cast<FragmentType>(operand.getType());
    if (!input || mapping.contains(operand))
      continue;
    Type selectedType = selectedTypeFor(input.getElementType());
    SmallVector<Value> selectedCoordinates;
    SmallVector<int64_t> selectedAxes;
    for (auto [axis, sourceAxis] : llvm::enumerate(axes)) {
      Value coordinate = coordinates[sourceAxis];
      if (broadcastUnit(operand, axis))
        coordinate = builder.create<arith::ConstantIndexOp>(gather.getLoc(), 0);
      Type element = coordinate.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      Type indexType = selectedTypeFor(element);
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, indexType);
      if (failed(projected))
        return gather.emitOpError("pointwise gather index lost its result projection");
      selectedCoordinates.push_back(*projected);
      selectedAxes.push_back(axis);
    }
    Value fill;
    if (gather.getValid()) {
      if (auto fragment = dyn_cast<FragmentType>(selectedType)) {
        FailureOr<Value> zero = materializeZeroFragment(builder, gather.getLoc(), fragment);
        if (failed(zero))
          return failure();
        fill = *zero;
      } else {
        fill = builder.create<arith::ConstantOp>(gather.getLoc(),
                                                builder.getZeroAttr(selectedType));
      }
    }
    mapping.map(operand, builder.create<GatherOp>(
        gather.getLoc(), selectedType, operand, selectedCoordinates,
        gather.getValid(), fill, selectedAxes).getResult());
  }
  // Keep reductions and immutable reads as captured SSA producers. Only the
  // pure pointwise suffix moves to the selected coordinates. Inactive operands
  // use zero (also a valid cast input); restore the original gather fill after
  // evaluating the unchanged typed operation.
  Value value;
  Type resultType = gather.getResult().getType();
  if (positionalRebind || isa<BroadcastOp, SplatOp>(producer)) {
    auto projected = projectPhysicalValueToSchema(
        builder, gather.getLoc(), mapping.lookupOrDefault(producer->getOperand(0)),
        resultType);
    if (failed(projected))
      return failure();
    value = *projected;
  } else {
    Operation *replacement = builder.clone(*producer, mapping);
    replacement->getResult(0).setType(resultType);
    value = replacement->getResult(0);
  }
  if (gather.getValid())
    value = builder.create<SelectOp>(gather.getLoc(), resultType, gather.getValid(),
                                     value, gather.getFill());
  gather.getResult().replaceAllUsesWith(value);
  gather.erase();
  return true;
}

FailureOr<bool> composeReducedGather(GatherOp gather) {
  auto reduce = gather.getSource().getDefiningOp<ReduceOp>();
  auto output = dyn_cast<FragmentType>(gather.getSource().getType());
  if (!reduce || !output || isa<FragmentType>(gather.getResult().getType()) ||
      reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
      reduce.getCaptures().size() != 0 || reduce.getNumResults() != 1 ||
      gather.getSourceAxes().size() != output.getShape().size() ||
      !queryBinaryCombineKind(reduce.getCombine()) ||
      std::distance(reduce.getCombine().front().begin(),
                    reduce.getCombine().front().end()) != 2)
    return false;

  auto input = cast<FragmentType>(reduce.getSources().front().getType());
  auto kernel = gather->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<Value> freeCoordinates(output.getShape().size());
  for (auto [coordinate, axis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (axis < 0 || axis >= static_cast<int64_t>(freeCoordinates.size()) ||
        freeCoordinates[axis] || isa<FragmentType>(coordinate.getType()))
      return false;
    freeCoordinates[axis] = coordinate;
  }
  SmallVector<Attribute> shape;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
      continue;
    auto range = queryExactLogicalRange(analysis.axisRanges(reduce.getSources().front(), axis));
    auto extent = cast<PhysicalExprAttr>(input.getShape()[axis]);
    if (failed(range) ||
        extent.getKind() != PhysicalExprKind::Constant ||
        extent.getValue() <= 0 ||
        constantLogicalRangeCardinality(*range) != extent.getValue())
      return false;
    shape.push_back(extent);
  }

  // Select complete reduction fibers from immutable SSA. Loads remain behind
  // gathers until composeLoadGather proves that moving each read is legal.
  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  SmallVector<Attribute> mappings;
  SmallVector<Value> coordinates;
  SmallVector<int64_t> sourceAxes;
  SmallVector<int64_t> reductionAxes;
  unsigned freeAxis = 0, reductionAxis = 0;
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    sourceAxes.push_back(axis);
    if (!llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis))) {
      coordinates.push_back(freeCoordinates[freeAxis++]);
      continue;
    }
    auto extent = cast<PhysicalExprAttr>(shape[reductionAxis]);
    auto mapping = AxisMapAttr::get(builder.getContext(), sourceId,
                                    reductionAxis, dimensionId++, reductionAxis, true);
    mappings.push_back(mapping);
    auto rangeType = FragmentType::get(builder.getContext(), builder.getIndexType(),
        builder.getArrayAttr({extent}), builder.getArrayAttr({AxisMapAttr::get(
            builder.getContext(), sourceId, reductionAxis, mapping.getDimensionId(), 0, true)}),
        input.getValidity(), input.getOwner());
    Value stop = builder.create<arith::ConstantIndexOp>(location, extent.getValue());
    coordinates.push_back(builder.create<MakeRangeOp>(
        location, rangeType, zero, stop, one, zero, stop, sourceId, reductionAxis, true));
    reductionAxes.push_back(reductionAxis++);
  }
  auto selectedType = FragmentType::get(builder.getContext(), input.getElementType(),
      builder.getArrayAttr(shape), builder.getArrayAttr(mappings),
      input.getValidity(), input.getOwner());
  auto bounded = gatherBounds(builder, location, output, gather.getCoordinates(),
                              gather.getSourceAxes(), builder.getIndexType());
  if (failed(bounded))
    return failure();
  auto predicate = FragmentType::get(builder.getContext(), builder.getI1Type(),
      selectedType.getShape(), selectedType.getAxisMaps(),
      selectedType.getValidity(), selectedType.getOwner());
  Value valid = builder.create<BroadcastOp>(location, predicate, *bounded);
  auto zeroFill = materializeZeroFragment(builder, location, selectedType);
  if (failed(zeroFill))
    return failure();
  Value selected = builder.create<GatherOp>(location, selectedType,
      reduce.getSources().front(), coordinates, valid, *zeroFill, sourceAxes);
  Type resultType = gather.getResult().getType();
  Value identityFill = builder.create<arith::ConstantOp>(location, builder.getZeroAttr(resultType));
  Value identity = builder.create<GatherOp>(location, resultType,
      reduce.getIdentities().front(), gather.getCoordinates(), *bounded,
      identityFill, gather.getSourceAxes());
  auto projected = builder.create<ReduceOp>(location, ValueRange{selected},
      ValueRange{identity}, ValueRange{}, reductionAxes);
  if (failed(scalarizeElementwiseCallback(reduce.getCombine(), projected.getCombine())))
    return failure();
  Value replacement = projected.getResult(0);
  if (gather.getValid())
    replacement = builder.create<SelectOp>(location, resultType, gather.getValid(),
                                           replacement, gather.getFill());
  gather.getResult().replaceAllUsesWith(replacement);
  gather.erase();
  return true;
}

Value cancelIndexOffset(OpBuilder &builder, Value coordinate, Value offset) {
  if (!uniformElementType(coordinate.getType()).isIndex())
    return {};
  if (auto subtract = coordinate.getDefiningOp<BinaryOp>();
      subtract && subtract.getOperatorKind() == BinaryOperator::Subtract) {
    Value bound = subtract.getRhs();
    while (true) {
      UniformExpression expression = describeUniformValue(bound);
      if (expression.kind != UniformKind::Forward ||
          expression.operands.size() != 1)
        break;
      bound = expression.operands.front();
    }
    if (samePhysicalScalarExpression(bound, offset))
      return subtract.getLhs();
  }
  Operation *projection = coordinate.getDefiningOp();
  if (!isa_and_nonnull<BroadcastOp, ReshapeOp, TransposeOp>(projection))
    return {};
  Value translated = cancelIndexOffset(builder, projection->getOperand(0), offset);
  if (!translated)
    return {};
  IRMapping mapping;
  mapping.map(projection->getOperand(0), translated);
  return builder.clone(*projection, mapping)->getResult(0);
}

FailureOr<bool> composeLoadGather(GatherOp gather) {
  auto sourceType = dyn_cast<FragmentType>(gather.getSource().getType());
  Value loaded = gather.getSource();
  while (auto transpose = loaded.getDefiningOp<TransposeOp>()) {
    auto input = cast<FragmentType>(transpose.getValue().getType());
    auto output = transpose.getResult().getType();
    for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
      auto original = cast<AxisMapAttr>(input.getAxisMaps()[sourceAxis]);
      auto transposed = cast<AxisMapAttr>(output.getAxisMaps()[axis]);
      if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
          original.getDimensionId() != transposed.getDimensionId())
        return false;
    }
    loaded = transpose.getValue();
  }
  auto sourceLoad = loaded.getDefiningOp<LoadOp>();
  if (!sourceType || !sourceLoad ||
      !isa<ViewType, BufferType>(sourceLoad.getResource().getType()) ||
      !canReplayReadAt(sourceLoad, gather))
    return false;
  if (gather.getCoordinates().size() != gather.getSourceAxes().size()) {
    gather.emitOpError("gather coordinate/source-axis schema is incomplete");
    return failure();
  }

  PhysicalProgramAnalysis analysis(gather->getParentOfType<func::FuncOp>());
  // Earlier rewrites may have created gathers in the source mask or fill.
  // Let the existing composition worklist normalize those producers before
  // committing this rewrite, which must replay both at the selected positions.
  SmallVector<Value> replayInputs(sourceLoad.getCoordinates());
  replayInputs.append({sourceLoad.getValid(), sourceLoad.getFill()});
  for (Value value : replayInputs)
    if (value && isa<FragmentType>(value.getType()) &&
        !analysis.replayability(value, std::nullopt,
                                PhysicalReplayScope::Coordinate,
                                /*allowAccesses=*/false).isReplayable())
      return false;

  OpBuilder builder(gather);
  auto resultType = dyn_cast<FragmentType>(gather.getResult().getType());
  SmallVector<Value> originalCoordinates;
  for (Value coordinate : sourceLoad.getCoordinates()) {
    while (auto broadcast = coordinate.getDefiningOp<BroadcastOp>())
      coordinate = broadcast.getValue();
    originalCoordinates.push_back(coordinate);
  }
  SmallVector<Value> coordinates(originalCoordinates);
  IRMapping replay;
  SmallVector<MakeRangeOp> selectedRanges;
  llvm::SmallDenseSet<Value> selectedRoots;
  for (int64_t sourceAxis : gather.getSourceAxes()) {
    if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(sourceType.getShape().size()))
      return gather.emitOpError("gather source axis is outside its loaded value");
    auto root = queryExactLogicalRange(analysis.axisRanges(
        gather.getSource(), static_cast<unsigned>(sourceAxis)));
    if (failed(root) ||
        (*root).getResult().getType().getShape()[0] != sourceType.getShape()[sourceAxis] ||
        !selectedRoots.insert((*root).getResult()).second)
      return false;
    selectedRanges.push_back(*root);
  }
  for (auto [coordinate, range] :
       llvm::zip(gather.getCoordinates(), selectedRanges)) {
    // The ordinal of a retained slice often subtracts its original base.
    // Compose the inverse translation before rebuilding the access: keeping
    // start + (coordinate - start) would hide the original range's bounds.
    Value absolute = isUnitStepRange(range)
                         ? cancelIndexOffset(builder, coordinate, range.getStart())
                         : Value();
    if (absolute)
      coordinate = absolute;
    if (resultType) {
      Type element = coordinate.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      auto coordinateType = FragmentType::get(
          resultType.getContext(), element, resultType.getShape(),
          resultType.getAxisMaps(), resultType.getValidity(), resultType.getOwner());
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, gather.getLoc(), coordinate, coordinateType);
      if (failed(projected))
        return gather.emitOpError(
            "composed gather index cannot adopt its result coordinate relation");
      coordinate = *projected;
    }
    if (!absolute && (!isZero(range.getStart()) || !isUnitStepRange(range))) {
      Type indexType = range.getResult().getType().getElementType();
      if (auto fragment = dyn_cast<FragmentType>(coordinate.getType()))
        indexType = FragmentType::get(
            fragment.getContext(), indexType, fragment.getShape(),
            fragment.getAxisMaps(), fragment.getValidity(), fragment.getOwner());
      if (coordinate.getType() != indexType)
        coordinate = builder.create<CastOp>(gather.getLoc(), indexType, coordinate);
      auto projectedBound = [&](Value bound) -> Value {
        if (auto fragment = dyn_cast<FragmentType>(indexType))
          return builder.create<BroadcastOp>(gather.getLoc(), fragment, bound);
        return bound;
      };
      // Gather indexes positions in the loaded tensor. Its ordinal must be
      // composed with the load range before becoming a resource coordinate.
      if (!isUnitStepRange(range))
        coordinate = builder.create<BinaryOp>(
            gather.getLoc(), indexType, coordinate, projectedBound(range.getStep()),
            BinaryOperator::Multiply);
      if (!isZero(range.getStart()))
        coordinate = builder.create<BinaryOp>(
            gather.getLoc(), indexType, projectedBound(range.getStart()), coordinate,
            BinaryOperator::Add);
    }
    replay.map(range.getResult(), coordinate);
  }

  if (resultType) {
    for (Value predicateOrFill : replayInputs) {
      if (!predicateOrFill)
        continue;
      PhysicalRangeFact roots = analysis.sourceRanges(predicateOrFill);
      // Bounds may use the raw range while the address uses a guarded index.
      // Preserve each untouched range's values, not the address expression.
      for (MakeRangeOp range : roots.roots) {
        if (replay.lookupOrNull(range.getResult()))
          continue;
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        if (failed(dimension))
          continue;
        auto sourceAxis = queryFragmentAxis(sourceType, sourceAxisIdentity(range),
                                            *dimension);
        if (!sourceAxis.isExact()) {
          std::optional<unsigned> retainedAxis;
          bool ambiguous = false;
          for (unsigned axis = 0; axis < sourceType.getShape().size(); ++axis) {
            if (llvm::is_contained(gather.getSourceAxes(),
                                   static_cast<int64_t>(axis)))
              continue;
            auto retained = analysis.axisRanges(gather.getSource(), axis);
            if (!retained.isExact() || !retained.blockers.empty() ||
                !llvm::any_of(retained.roots, [&](MakeRangeOp root) {
                  auto rootDimension = queryRangeDimension(root);
                  return succeeded(rootDimension) && *rootDimension == *dimension &&
                         sameLogicalRange(root, range) &&
                         samePhysicalScalarExpression(root.getStart(),
                                                      range.getStart()) &&
                         samePhysicalScalarExpression(root.getExtent(),
                                                      range.getExtent());
                }))
              continue;
            if (retainedAxis) {
              ambiguous = true;
              break;
            }
            retainedAxis = axis;
          }
          if (retainedAxis && !ambiguous) {
            auto retained =
                cast<AxisMapAttr>(sourceType.getAxisMaps()[*retainedAxis]);
            sourceAxis = queryFragmentAxis(
                sourceType, sourceAxisIdentity(retained),
                retained.getDimensionId());
          }
        }
        if (!sourceAxis.isExact() ||
            llvm::is_contained(gather.getSourceAxes(),
                               static_cast<int64_t>(sourceAxis.fragmentAxis)))
          continue;
        auto retainedMapping =
            cast<AxisMapAttr>(sourceType.getAxisMaps()[sourceAxis.fragmentAxis]);
        auto axis = queryFragmentAxis(resultType, sourceAxisIdentity(retainedMapping),
                                      sourceAxis.dimensionId);
        auto rangeType = range.getResult().getType();
        if (!axis.isExact() ||
            axis.dimensionId != sourceAxis.dimensionId ||
            resultType.getShape()[axis.fragmentAxis] != rangeType.getShape()[0])
          continue;
        // Reshape can rename a retained coordinate axis. The exact range
        // dependency above supplies its result position without changing the
        // range's values or inventing an equality between unrelated axes.
        auto namedType = FragmentType::get(
            rangeType.getContext(), rangeType.getElementType(), rangeType.getShape(),
            builder.getArrayAttr({AxisMapAttr::get(
                rangeType.getContext(), retainedMapping.getSourceId(),
                retainedMapping.getSourceAxis(), retainedMapping.getDimensionId(),
                0, retainedMapping.getDerived())}),
            rangeType.getValidity(), rangeType.getOwner());
        Value retainedRange = range.getResult();
        if (namedType != rangeType) {
          auto relation = inferReshapeReassociation(rangeType, namedType);
          if (failed(relation))
            return gather.emitOpError(
                "retained coordinate has no exact renamed-axis relation");
          retainedRange = builder.create<ReshapeOp>(
              gather.getLoc(), namedType, retainedRange, *relation);
        }
        auto rangeTarget = FragmentType::get(
            resultType.getContext(), rangeType.getElementType(),
            resultType.getShape(), resultType.getAxisMaps(),
            resultType.getValidity(), resultType.getOwner());
        FailureOr<Value> projectedRange = projectPhysicalValueToSchema(
            builder, gather.getLoc(), retainedRange, rangeTarget);
        if (failed(projectedRange))
          return gather.emitOpError(
              "retained coordinate range cannot adopt the gather result relation");
        replay.map(range.getResult(), *projectedRange);
      }
    }
  }

  // A storage coordinate may combine several logical axes (for example a
  // reshaped row and column). Bind every selected range before replaying it.
  for (auto [slot, original] : llvm::enumerate(originalCoordinates)) {
    FailureOr<Value> selected = resultType
        ? replayFragmentValue(builder, original, resultType, replay, analysis)
        : replayScalarValue(builder, original, replay, analysis);
    if (failed(selected))
      return gather.emitOpError("indexed load coordinate cannot follow its source ranges");
    coordinates[slot] = *selected;
  }

  if (!resultType) {
    FailureOr<Value> sourceValid = replayScalarValue(
        builder, sourceLoad.getValid(), replay, analysis);
    FailureOr<Value> sourceFill = replayScalarValue(
        builder, sourceLoad.getFill(), replay, analysis);
    if (failed(sourceValid) || failed(sourceFill)) {
      gather.emitOpError(
          "source access validity/fill cannot follow scalar composed coordinates");
      return failure();
    }
    Value valid = *sourceValid;
    if (gather.getValid()) {
      if (!gather.getValid().getType().isInteger(1)) {
        gather.emitOpError("scalar gather validity is not scalar i1");
        return failure();
      }
      valid = valid ? Value(builder.create<BinaryOp>(
                            gather.getLoc(), builder.getI1Type(), valid,
                            gather.getValid(), BinaryOperator::LogicalAnd))
                    : gather.getValid();
    }
    Value fill = *sourceFill;
    if (sourceLoad.getValid() && gather.getValid()) {
      if (!fill || !gather.getFill() ||
          fill.getType() != gather.getResult().getType() ||
          gather.getFill().getType() != gather.getResult().getType()) {
        gather.emitOpError(
            "scalar composed conditional access has incompatible fills");
        return failure();
      }
      fill = builder.create<SelectOp>(
          gather.getLoc(), gather.getResult().getType(), gather.getValid(), fill,
          gather.getFill());
    } else if (gather.getValid()) {
      fill = gather.getFill();
    }
    if (static_cast<bool>(valid) != static_cast<bool>(fill)) {
      gather.emitOpError(
          "scalar composed access requires paired validity and fill");
      return failure();
    }
    auto replacement = builder.create<LoadOp>(
        gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
        coordinates, valid, fill, sourceLoad.getSourceAxes());
    if (Attribute origin = sourceLoad->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
    if (sourceLoad->getBlock() && sourceLoad.getResult().use_empty())
      sourceLoad.erase();
    return true;
  }
  FailureOr<Value> sourceValid = replayFragmentValue(
      builder, sourceLoad.getValid(), resultType, replay, analysis);
  FailureOr<Value> sourceFill = replayFragmentValue(
      builder, sourceLoad.getFill(), resultType, replay, analysis);
  if (failed(sourceValid) || failed(sourceFill)) {
    gather.emitOpError(
        "source access validity/fill cannot follow composed coordinates");
    return failure();
  }
  FailureOr<Value> valid = combinePredicates(
      builder, gather.getLoc(), resultType, *sourceValid, gather.getValid());
  if (failed(valid)) {
    gather.emitOpError(
        "source and gather validity cannot share the composed result relation");
    return failure();
  }
  Value fill = *sourceFill;
  if (sourceLoad.getValid() && gather.getValid()) {
    Value gatherFill = gather.getFill();
    if (!fill || !gatherFill) {
      gather.emitOpError(
          "composed conditional access requires both source and gather fill");
      return failure();
    }
    auto result = resultType;
    if (fill.getType() != result)
      fill = builder.create<BroadcastOp>(gather.getLoc(), result, fill);
    if (gatherFill.getType() != result)
      gatherFill =
          builder.create<BroadcastOp>(gather.getLoc(), result, gatherFill);
    Value condition = gather.getValid();
    auto predicate = FragmentType::get(
        result.getContext(), builder.getI1Type(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    if (condition.getType() != predicate)
      condition =
          builder.create<BroadcastOp>(gather.getLoc(), predicate, condition);
    fill = builder.create<SelectOp>(gather.getLoc(), result, condition, fill,
                                    gatherFill);
  } else if (gather.getValid()) {
    fill = gather.getFill();
  }
  auto replacement = builder.create<LoadOp>(
      gather.getLoc(), gather.getResult().getType(), sourceLoad.getResource(),
      coordinates, *valid, fill,
      sourceLoad.getSourceAxes());
  if (Attribute origin = sourceLoad->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  if (sourceLoad->getBlock() && sourceLoad.getResult().use_empty())
    sourceLoad.erase();
  return true;
}

bool scalarPredicateContains(Value predicate, Value required) {
  if (!predicate)
    return false;
  if (predicate == required)
    return true;
  auto lhs = predicate.getDefiningOp<CompareOp>();
  auto rhs = required.getDefiningOp<CompareOp>();
  if (lhs && rhs && lhs.getPredicate() == rhs.getPredicate() &&
      lhs.getLhs().getType().isIndex() && rhs.getLhs().getType().isIndex() &&
      samePhysicalScalarExpression(lhs.getLhs(), rhs.getLhs()) &&
      samePhysicalScalarExpression(lhs.getRhs(), rhs.getRhs()))
    return true;
  auto binary = predicate.getDefiningOp<BinaryOp>();
  return binary && binary.getOperatorKind() == BinaryOperator::LogicalAnd &&
         (scalarPredicateContains(binary.getLhs(), required) ||
          scalarPredicateContains(binary.getRhs(), required));
}

bool impliesUniformPredicate(Value predicate, Value required) {
  if (!required)
    return true;
  auto constant = dyn_cast_or_null<IntegerAttr>(
      UniformValueAnalysis(describeUniformValue).evaluate(required));
  if (constant && constant.getType().isInteger(1) &&
      constant.getValue().isOne())
    return true;
  if (auto splat = required.getDefiningOp<SplatOp>())
    return impliesUniformPredicate(predicate, splat.getValue());
  if (auto broadcast = required.getDefiningOp<BroadcastOp>())
    return impliesUniformPredicate(predicate, broadcast.getValue());
  if (auto binary = required.getDefiningOp<BinaryOp>();
      binary && binary.getOperatorKind() == BinaryOperator::LogicalAnd)
    return impliesUniformPredicate(predicate, binary.getLhs()) &&
           impliesUniformPredicate(predicate, binary.getRhs());
  // Only scalar conjuncts may be shared across retained fragment lanes.
  return required.getType().isInteger(1) &&
         scalarPredicateContains(predicate, required);
}

bool reuseFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  if (!source || gather.getResult().getType() != source.getElementType() ||
      gather.getSourceAxes().size() != source.getShape().size() ||
      llvm::any_of(gather.getCoordinates(),
                   [](Value coordinate) { return !coordinate.getType().isIndex(); }))
    return false;
  for (auto [axis, sourceAxis] : llvm::enumerate(gather.getSourceAxes()))
    if (sourceAxis != static_cast<int64_t>(axis))
      return false;

  // A scalar extraction can reuse an earlier immutable slice of the same SSA
  // tensor. Native layout/communication still belongs to the provider compiler.
  for (Operation *user : gather.getSource().getUsers()) {
    auto available = dyn_cast<GatherOp>(user);
    auto sliced = available
                      ? dyn_cast<FragmentType>(available.getResult().getType())
                      : FragmentType();
    if (!available || available == gather ||
        available->getBlock() != gather->getBlock() ||
        !available->isBeforeInBlock(gather) || !sliced ||
        sliced.getOwner() != source.getOwner() ||
        available.getSourceAxes().empty() ||
        sliced.getShape().size() + available.getSourceAxes().size() !=
            source.getShape().size() ||
        !impliesUniformPredicate(gather.getValid(), available.getValid()))
      continue;
    SmallVector<bool> selected(source.getShape().size(), false);
    bool matches = true;
    for (auto [coordinate, axis] :
         llvm::zip(available.getCoordinates(), available.getSourceAxes())) {
      if (axis < 0 || axis >= static_cast<int64_t>(selected.size()) ||
          selected[axis] || !coordinate.getType().isIndex() ||
          !samePhysicalScalarExpression(coordinate,
                                        gather.getCoordinates()[axis])) {
        matches = false;
        break;
      }
      selected[axis] = true;
    }
    if (!matches)
      continue;
    SmallVector<Value> coordinates;
    SmallVector<int64_t> axes;
    for (unsigned axis = 0; axis < selected.size(); ++axis) {
      if (selected[axis])
        continue;
      unsigned retained = coordinates.size();
      auto originalMap = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
      auto retainedMap = cast<AxisMapAttr>(sliced.getAxisMaps()[retained]);
      if (sliced.getShape()[retained] != source.getShape()[axis] ||
          !(sourceAxisIdentity(originalMap) == sourceAxisIdentity(retainedMap)) ||
          originalMap.getDimensionId() != retainedMap.getDimensionId()) {
        matches = false;
        break;
      }
      coordinates.push_back(gather.getCoordinates()[axis]);
      axes.push_back(retained);
    }
    if (!matches)
      continue;
    OpBuilder builder(gather);
    auto replacement = builder.create<GatherOp>(
        gather.getLoc(), gather.getResult().getType(), available.getResult(),
        coordinates, gather.getValid(), gather.getFill(), axes);
    replacement->setDiscardableAttrs(
        llvm::to_vector(gather->getDiscardableAttrs()));
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
    return true;
  }
  return false;
}

FailureOr<bool> composeIdentityFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result || source.getOwner() != result.getOwner() ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;
  // Gather indexes fragment positions. A complete one-dimensional ordinal
  // range is an identity even when storage and consumer use different axes.
  if (source.getShape().size() == 1 && source.getShape() == result.getShape() &&
      gather.getSourceAxes() == ArrayRef<int64_t>{0}) {
    auto range = gather.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    if (range && range.getResult().getType().getShape() == result.getShape() &&
        range.getResult().getType().getAxisMaps() == result.getAxisMaps() &&
        isZero(range.getStart()) &&
        isUnitStepRange(range) &&
        queryLaunchExpression(range.getExtent()) == source.getShape()[0]) {
      OpBuilder builder(gather);
      auto group = ReshapeGroupAttr::get(
          builder.getContext(), builder.getDenseI64ArrayAttr({0}),
          builder.getDenseI64ArrayAttr({0}));
      Value replacement = builder.create<ReshapeOp>(
          gather.getLoc(), result, gather.getSource(), builder.getArrayAttr({group}));
      if (gather.getValid())
        replacement = builder.create<SelectOp>(gather.getLoc(), result,
            gather.getValid(), replacement, gather.getFill());
      if (Attribute origin = gather->getAttr(originAttr))
        replacement.getDefiningOp()->setAttr(originAttr, origin);
      gather.getResult().replaceAllUsesWith(replacement);
      gather.erase();
      return true;
    }
  }
  SmallVector<bool> represented(result.getShape().size(), false);
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return false;
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    PhysicalSourceAxis physicalSource = sourceAxisIdentity(expected);
    PhysicalAxisProjection resultAxis =
        queryFragmentAxis(result, physicalSource, expected.getDimensionId());
    // Coordinate broadcasting does not change the ordinal of a full slice.
    // Inspect the range before its singleton axes were inserted.
    while (auto broadcast = coordinate.getDefiningOp<BroadcastOp>())
      coordinate = broadcast.getValue();
    PhysicalAxisProjection coordinateAxis =
        queryCoordinateIndex(ValueRange{coordinate}, physicalSource,
                             expected.getDimensionId());
    auto resultMapping =
        resultAxis.isExact()
            ? dyn_cast<AxisMapAttr>(result.getAxisMaps()[resultAxis.fragmentAxis])
            : AxisMapAttr();
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    auto coordinateRange = coordinate.getDefiningOp<MakeRangeOp>();
    if (!resultAxis.isExact() || !coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType || !coordinateRange ||
        !isZero(coordinateRange.getStart()) ||
        !isUnitStepRange(coordinateRange) ||
        source.getShape()[sourceAxis] !=
            result.getShape()[resultAxis.fragmentAxis] ||
        coordinateType.getShape().size() != 1 ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        coordinateAxis.fragmentAxis != 0)
      return false;
    represented[resultAxis.fragmentAxis] = true;
  }
  for (auto [axis, extent] : llvm::enumerate(result.getShape())) {
    if (represented[axis])
      continue;
    auto expression = dyn_cast<PhysicalExprAttr>(extent);
    if (!expression ||
        expression.getKind() !=
            PhysicalExprKind::Constant ||
        expression.getValue() != 1)
      return false;
  }
  OpBuilder builder(gather);
  Value replacement = builder.create<BroadcastOp>(gather.getLoc(), result,
                                                   gather.getSource());
  if (gather.getValid())
    replacement = builder.create<SelectOp>(gather.getLoc(), result,
                                           gather.getValid(), replacement,
                                           gather.getFill());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement.getDefiningOp()->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement);
  gather.erase();
  return true;
}

FailureOr<bool> composeReductionGathers(ReduceOp reduce) {
  if (reduce.getSources().size() != 1 || reduce.getIdentities().size() != 1 ||
      reduce.getCaptures().size() != 0 || reduce.getNumResults() != 1 ||
      reduce.getAxes() != ArrayRef<int64_t>{0})
    return false;
  Value input = reduce.getSources().front();
  auto schema = dyn_cast<FragmentType>(input.getType());
  auto result = dyn_cast<FragmentType>(reduce.getResult(0).getType());
  if (!schema || schema.getShape().size() != 1 ||
      (result && !result.getShape().empty()))
    return false;
  auto extent = cast<PhysicalExprAttr>(schema.getShape()[0]);
  int64_t size = extent.getValue();
  if (extent.getKind() != PhysicalExprKind::Constant ||
      size <= 0 || (size & (size - 1)) != 0 ||
      size > std::numeric_limits<int64_t>::max() / 2)
    return false;

  SmallVector<GatherOp> gathers;
  SmallVector<MakeRangeOp> ranges;
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> collect = [&](Value value) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment || !visited.insert(value).second)
      return true;
    if (fragment.getShape() != schema.getShape() ||
        fragment.getAxisMaps() != schema.getAxisMaps() ||
        fragment.getValidity() != schema.getValidity() ||
        fragment.getOwner() != schema.getOwner())
      return false;
    if (auto range = value.getDefiningOp<MakeRangeOp>()) {
      ranges.push_back(range);
      return true;
    }
    if (auto gather = value.getDefiningOp<GatherOp>()) {
      auto source = cast<FragmentType>(gather.getSource().getType());
      if (source.getShape() != schema.getShape() ||
          source.getOwner() != schema.getOwner() ||
          gather.getSourceAxes() != ArrayRef<int64_t>{0} ||
          gather.getCoordinates().size() != 1 || !gather.getValid() ||
          !gather.getFill() ||
          !gather.getCoordinates().front().getDefiningOp<MakeRangeOp>())
        return false;
      if (!collect(gather.getCoordinates().front()) ||
          !collect(gather.getValid()) || !collect(gather.getFill()))
        return false;
      gathers.push_back(gather);
      return true;
    }
    Operation *producer = value.getDefiningOp();
    if (!isa_and_nonnull<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                        BitcastOp, SplatOp, BroadcastOp, ReshapeOp>(producer) ||
        producer->getNumResults() != 1 || !isMemoryEffectFree(producer))
      return false;
    return llvm::all_of(producer->getOperands(), collect);
  };
  if (!collect(input) || gathers.empty() || ranges.empty())
    return false;
  MakeRangeOp range = ranges.front();
  if (isZero(range.getStart()) || !isUnitStepRange(range) ||
      !samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()))
    return false;
  for (MakeRangeOp other : ranges)
    if (!sameLogicalRange(range, other) ||
        !samePhysicalScalarExpression(range.getStart(), other.getStart()) ||
        !samePhysicalScalarExpression(range.getExtent(), other.getExtent()))
      return false;
  PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(range.getStart());
  if (!bound ||
      bound.getKind() != PhysicalExprKind::Constant ||
      bound.getValue() > size)
    return false;

  UniformValueAnalysis uniform(describeUniformValue);
  for (GatherOp gather : gathers) {
    Value coordinate = gather.getCoordinates().front();
    SmallVector<Value> predicates{gather.getValid()};
    bool bounded = false;
    while (!predicates.empty()) {
      Value predicate = predicates.pop_back_val();
      if (auto binary = predicate.getDefiningOp<BinaryOp>();
          binary &&
          (binary.getOperatorKind() == BinaryOperator::LogicalAnd ||
           binary.getOperatorKind() == BinaryOperator::BitwiseAnd)) {
        predicates.push_back(binary.getLhs());
        predicates.push_back(binary.getRhs());
      } else if (auto compare = predicate.getDefiningOp<CompareOp>();
                 compare && compare.getPredicate() == ComparePredicate::Lt &&
                 compare.getLhs() == coordinate) {
        auto upper = dyn_cast_or_null<IntegerAttr>(
            uniform.evaluate(compare.getRhs()));
        bounded |= upper && upper.getInt() <= size;
      }
    }
    if (!bounded)
      return false;
  }
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalSourceAxis source = sourceAxisIdentity(range);
  if (!analysis.replayability(input, source, PhysicalReplayScope::ValueGraph,
                             /*allowAccesses=*/true).isReplayable())
    return false;

  // Ordinary reduce permits a permutation of its inputs. For a shifted
  // [k, k + N) range, visit [N, N + k) then [k, N) instead. This retains
  // every original value, including padding/fill, while valid gathers become
  // identity reads from the immutable parent fragment. Other users keep their
  // original coordinates; no scan or ordered state is permuted.
  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  auto [sourceId, dimensionId] = nextPhysicalAxisIdentities(kernel);
  auto axis = AxisMapAttr::get(builder.getContext(), sourceId, 0, dimensionId,
                             0, true);
  auto typeFor = [&](Type element) {
    return FragmentType::get(builder.getContext(), element, schema.getShape(),
                            builder.getArrayAttr({axis}), schema.getValidity(),
                            schema.getOwner());
  };
  auto indexType = typeFor(builder.getIndexType());
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value count = builder.create<arith::ConstantIndexOp>(location, size);
  Value ordinal = builder.create<MakeRangeOp>(
      location, indexType, zero, count, one, zero, count, sourceId, 0, true);
  Value start = builder.create<SplatOp>(location, indexType, range.getStart());
  Value prefix = builder.create<CompareOp>(location, typeFor(builder.getI1Type()),
                                          ordinal, start, ComparePredicate::Lt);
  Value shifted = builder.create<BinaryOp>(
      location, indexType, ordinal,
      builder.create<SplatOp>(location, indexType, count), BinaryOperator::Add);
  Value coordinate = builder.create<SelectOp>(location, indexType, prefix,
                                              shifted, ordinal);
  IRMapping mapping;
  for (MakeRangeOp original : ranges)
    mapping.map(original.getResult(), coordinate);
  ReplayMaterializationOptions options;
  options.traversalRanges = ranges;
  options.fragmentAxis = 0;
  options.segmentMapping = axis;
  auto replay = [&](Value value) {
    return materializeReplayedValue(builder, location, value, source, extent,
                                    mapping, options);
  };
  auto group = ReshapeGroupAttr::get(builder.getContext(),
                                    builder.getDenseI64ArrayAttr({0}),
                                    builder.getDenseI64ArrayAttr({0}));
  for (GatherOp gather : gathers) {
    auto valid = replay(gather.getValid());
    auto fill = replay(gather.getFill());
    if (failed(valid) || failed(fill))
      return reduce.emitOpError("cannot permute gather validity/fill"), failure();
    auto type = typeFor(uniformElementType(gather.getResult().getType()));
    Value parent = builder.create<ReshapeOp>(
        location, type, gather.getSource(), builder.getArrayAttr({group}));
    Value replacement = builder.create<SelectOp>(location, type, *valid,
                                                 parent, *fill);
    mapping.map(gather.getResult(), replacement);
  }
  auto replacement = replay(input);
  if (failed(replacement))
    return reduce.emitOpError("cannot permute gather reduction inputs"), failure();
  reduce->setOperand(0, *replacement);
  return true;
}

FailureOr<bool> projectFragmentGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result ||
      gather.getCoordinates().size() != source.getShape().size() ||
      gather.getSourceAxes().size() != source.getShape().size())
    return false;

  SmallVector<Value> selectedCoordinates;
  SmallVector<int64_t> selectedAxes;
  bool projected = false;
  for (auto [coordinate, sourceAxis] :
       llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
    if (sourceAxis < 0 ||
        sourceAxis >= static_cast<int64_t>(source.getShape().size()))
      return false;
    auto expected = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    PhysicalSourceAxis physicalSource = sourceAxisIdentity(expected);
    PhysicalAxisProjection resultAxis =
        queryFragmentAxis(result, physicalSource, expected.getDimensionId());
    if (resultAxis.state == PhysicalFactState::Ambiguous)
      return false;
    if (!resultAxis.isExact()) {
      selectedCoordinates.push_back(coordinate);
      selectedAxes.push_back(sourceAxis);
      continue;
    }
    PhysicalAxisProjection coordinateAxis =
        queryCoordinateIndex(ValueRange{coordinate}, physicalSource,
                             expected.getDimensionId());
    auto resultMapping =
        dyn_cast<AxisMapAttr>(result.getAxisMaps()[resultAxis.fragmentAxis]);
    auto coordinateType = dyn_cast<FragmentType>(coordinate.getType());
    if (!coordinateAxis.isExact() || !resultMapping ||
        resultMapping.getDimensionId() != expected.getDimensionId() ||
        coordinateAxis.dimensionId != expected.getDimensionId() ||
        !coordinateType ||
        coordinateType.getShape().size() != 1 ||
        source.getShape()[sourceAxis] !=
            result.getShape()[resultAxis.fragmentAxis] ||
        coordinateType.getShape()[0] != source.getShape()[sourceAxis] ||
        coordinateAxis.fragmentAxis != 0)
      return false;
    projected = true;
  }
  if (!projected || selectedCoordinates.empty())
    return false;

  OpBuilder builder(gather);
  auto replacement = builder.create<GatherOp>(
      gather.getLoc(), result, gather.getSource(), selectedCoordinates,
      gather.getValid(), gather.getFill(), selectedAxes);
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

FailureOr<bool> composeBroadcastGather(GatherOp gather) {
  auto source = dyn_cast<FragmentType>(gather.getSource().getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!source || !result || !gather.getValid() ||
      source.getOwner() != result.getOwner() ||
      gather.getCoordinates().size() != source.getShape().size())
    return false;

  SmallVector<Value> coordinates;
  FragmentType indexSchema;
  for (Value coordinate : gather.getCoordinates()) {
    if (!uniformElementType(coordinate.getType()).isIndex())
      return false;
    Value scalar = coordinate;
    while (isa<FragmentType>(scalar.getType())) {
      UniformExpression expression = describeUniformValue(scalar);
      if (expression.kind != UniformKind::Forward || expression.operands.size() != 1)
        break;
      scalar = expression.operands.front();
    }
    if (!isa<FragmentType>(scalar.getType()))
      coordinate = scalar;
    if (auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        fragment && (!indexSchema || fragment.getShape().size() >
                                       indexSchema.getShape().size()))
      indexSchema = fragment;
    coordinates.push_back(coordinate);
  }
  if (indexSchema) {
    if (indexSchema.getOwner() != result.getOwner() ||
        indexSchema.getShape().size() >= result.getShape().size())
      return false;
    auto expansion = queryBroadcastProjection(indexSchema, result);
    if (!expansion.isExact())
      return false;
    for (Value coordinate : coordinates) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment)
        continue;
      auto projection = queryBroadcastProjection(fragment, indexSchema);
      auto original = queryBroadcastProjection(fragment, result);
      if (!projection.isExact() || !original.isExact())
        return false;
      for (auto [axis, indexAxis] : llvm::enumerate(expansion.targetToSource))
        if ((indexAxis ? projection.targetToSource[*indexAxis] : std::nullopt) !=
            original.targetToSource[axis])
          return false;
    }
  }

  OpBuilder builder(gather);
  Location location = gather.getLoc();
  auto selectedTypeFor = [&](Type element) -> Type {
    return indexSchema ? Type(FragmentType::get(
        result.getContext(), element, indexSchema.getShape(),
        indexSchema.getAxisMaps(), indexSchema.getValidity(), result.getOwner()))
        : element;
  };
  Type selectedType = selectedTypeFor(result.getElementType());
  Type indexType = selectedTypeFor(builder.getIndexType());
  auto valid = gatherBounds(builder, location, source, coordinates,
                            gather.getSourceAxes(), indexType);
  if (failed(valid))
    return gather.emitOpError("gather lost its exact index broadcast relation");
  Value fill;
  if (auto fragment = dyn_cast<FragmentType>(selectedType)) {
    auto zeroFill = materializeZeroFragment(builder, location, fragment);
    if (failed(zeroFill))
      return failure();
    fill = *zeroFill;
  } else {
    fill = builder.create<arith::ConstantOp>(location, builder.getZeroAttr(selectedType));
  }
  // Only indices determine the immutable SSA read. Keep its own bounds before
  // broadcasting; the original lane-dependent predicate and fill remain below.
  Value selected = builder.create<GatherOp>(
      location, selectedType, gather.getSource(), coordinates, *valid, fill,
      gather.getSourceAxes());
  Value expanded = builder.create<BroadcastOp>(location, result, selected);
  auto replacement = builder.create<SelectOp>(
      location, result, gather.getValid(), expanded, gather.getFill());
  if (Attribute origin = gather->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  gather.getResult().replaceAllUsesWith(replacement.getResult());
  gather.erase();
  return true;
}

bool hasNonUnitAxisSplit(ReshapeOp reshape) {
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto result = cast<FragmentType>(reshape.getResult().getType());
  if (source.getShape().size() >= result.getShape().size())
    return false;
  bool split = false;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty())
      return false;
    if (group.getSourceAxes().empty())
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1)
          return false;
      }
    if (group.getSourceAxes().size() == 1 && group.getResultAxes().size() > 1) {
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() <= 1)
          return false;
      }
      split = true;
    }
  }
  return split;
}

bool composeReshapedPointwise(ReshapeOp reshape) {
  auto source = cast<FragmentType>(reshape.getValue().getType());
  auto result = cast<FragmentType>(reshape.getResult().getType());
  if (auto inner = reshape.getValue().getDefiningOp<ReshapeOp>();
      inner && inner.getValue().getType() == result &&
      inner.getReassociation().size() == reshape.getReassociation().size() &&
      llvm::all_of(llvm::zip(inner.getReassociation(), reshape.getReassociation()),
                   [](auto pair) {
                     auto first = cast<ReshapeGroupAttr>(std::get<0>(pair));
                     auto second = cast<ReshapeGroupAttr>(std::get<1>(pair));
                     return first.getSourceAxes() == second.getResultAxes() &&
                            first.getResultAxes() == second.getSourceAxes();
                   })) {
    reshape.getResult().replaceAllUsesWith(inner.getValue());
    reshape.erase();
    return true;
  }
  if (!hasNonUnitAxisSplit(reshape) &&
      source.getShape().size() <= result.getShape().size())
    return false;
  Operation *producer = reshape.getValue().getDefiningOp();
  if (!producer ||
      !isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
           SplatOp, BroadcastOp>(producer))
    return false;
  for (Value operand : producer->getOperands()) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (fragment &&
        (fragment.getShape() != source.getShape() ||
         fragment.getAxisMaps() != source.getAxisMaps() ||
         fragment.getOwner() != source.getOwner()))
      return false;
  }
  OpBuilder builder(reshape);
  IRMapping mapping;
  for (Value operand : producer->getOperands()) {
    auto fragment = dyn_cast<FragmentType>(operand.getType());
    if (!fragment || mapping.contains(operand))
      continue;
    auto target = FragmentType::get(
        result.getContext(), fragment.getElementType(), result.getShape(),
        result.getAxisMaps(), result.getValidity(), result.getOwner());
    mapping.map(operand, builder.create<ReshapeOp>(
                             reshape.getLoc(), target, operand,
                             reshape.getReassociation()).getResult());
  }
  Operation *replacement = builder.clone(*producer, mapping);
  replacement->getResult(0).setType(result);
  reshape.getResult().replaceAllUsesWith(replacement->getResult(0));
  reshape.erase();
  return true;
}

FailureOr<bool> composeReshapedLoad(ReshapeOp reshape) {
  Value sourceValue = reshape.getValue();
  Value loaded = sourceValue;
  bool transposed = false;
  while (true) {
    if (auto broadcast = loaded.getDefiningOp<BroadcastOp>()) {
      auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
      auto output = dyn_cast<FragmentType>(broadcast.getResult().getType());
      if (!input || !output || input.getShape() != output.getShape())
        break;
      BroadcastProjection projection = queryAxisProjection(input, output);
      if (!projection.isExact() || llvm::any_of(
              llvm::enumerate(projection.targetToSource), [](auto item) {
                return !item.value() || *item.value() != item.index();
              }))
        break;
      loaded = broadcast.getValue();
      continue;
    }
    auto transpose = loaded.getDefiningOp<TransposeOp>();
    if (!transpose)
      break;
    auto source = cast<FragmentType>(transpose.getValue().getType());
    auto result = cast<FragmentType>(transpose.getResult().getType());
    for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
      auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
      auto transposed = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
      if (!(sourceAxisIdentity(original) == sourceAxisIdentity(transposed)) ||
          original.getDimensionId() != transposed.getDimensionId())
        return false;
    }
    transposed = true;
    loaded = transpose.getValue();
  }
  auto load = loaded.getDefiningOp<LoadOp>();
  auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
  if (!load || !result || !isa<ViewType>(load.getResource().getType()) ||
      !canReplayReadAt(load, reshape))
    return false;
  auto source = cast<FragmentType>(sourceValue.getType());
  if (source.getShape().size() <= result.getShape().size() &&
      !hasNonUnitAxisSplit(reshape) && !transposed)
    return false;
  auto kernel = reshape->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  unsigned sourceRank = 0;
  unsigned resultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty())
      for (int64_t axis : group.getSourceAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
        PhysicalRangeFact ranges = analysis.axisRanges(sourceValue, axis);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1 || !ranges.isExact() ||
            !llvm::all_of(ranges.roots, isProvablySingletonLogicalRange))
          return false;
      }
    if (group.getSourceAxes().empty())
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto extent = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (extent.getKind() !=
                PhysicalExprKind::Constant ||
            extent.getValue() != 1)
          return false;
      }
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      sourceRank = std::max(sourceRank, static_cast<unsigned>(axis + 1));
    for (int64_t axis : group.getResultAxes().asArrayRef())
      resultRank = std::max(resultRank, static_cast<unsigned>(axis + 1));
  }
  if (sourceRank != source.getShape().size() || resultRank != result.getShape().size())
    return false;
  SmallVector<bool> preservedSource(sourceRank, false);
  SmallVector<bool> preservedResult(resultRank, false);
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty()) {
      for (int64_t axis : group.getResultAxes().asArrayRef())
        preservedResult[axis] = true;
      continue;
    }
    if (group.getResultAxes().empty()) {
      for (int64_t axis : group.getSourceAxes().asArrayRef())
        preservedSource[axis] = true;
      continue;
    }
    if (group.getSourceAxes().size() != 1 ||
        group.getResultAxes().size() != 1)
      continue;
    unsigned sourceAxis = group.getSourceAxes()[0];
    unsigned resultAxis = group.getResultAxes()[0];
    auto original = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
    auto target = cast<AxisMapAttr>(result.getAxisMaps()[resultAxis]);
    if (sourceAxisIdentity(original) == sourceAxisIdentity(target) &&
        original.getDimensionId() == target.getDimensionId() &&
        source.getShape()[sourceAxis] == result.getShape()[resultAxis]) {
      preservedSource[sourceAxis] = true;
      preservedResult[resultAxis] = true;
    }
  }
  SmallVector<MakeRangeOp> sourceRanges(sourceRank);
  SmallVector<SmallVector<MakeRangeOp>> sourceRoots(sourceRank);
  SmallVector<Attribute> sourceExtents(sourceRank);
  llvm::DenseMap<Operation *, unsigned> rootAxes;
  for (unsigned axis = 0; axis < sourceRank; ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(sourceValue, axis);
    for (MakeRangeOp root : fact.roots) {
      auto [found, inserted] = rootAxes.try_emplace(root.getOperation(), axis);
      if (!inserted && found->second != axis)
        return false;
    }
    sourceRoots[axis].append(fact.roots.begin(), fact.roots.end());
    if (preservedSource[axis]) {
      auto realization = analysis.axisRealization(sourceValue, axis);
      bool introducedUnit = fact.roots.empty() && realization.isExact() &&
                            !realization.constructionScalarSeed &&
                            cast<PhysicalExprAttr>(source.getShape()[axis]).getKind() ==
                                PhysicalExprKind::Constant &&
                            cast<PhysicalExprAttr>(source.getShape()[axis]).getValue() == 1;
      if (fact.state != PhysicalFactState::Exact && !introducedUnit)
        return false;
      continue;
    }
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(fact);
    if (failed(authority))
      return false;
    for (MakeRangeOp root : fact.roots) {
      if (!samePhysicalScalarExpression(root.getStart(),
                                         root.getLogicalStart()) ||
          !isUnitStepRange(root))
        return false;
      auto count = constantLogicalRangeCardinality(root);
      auto physical =
          cast<PhysicalExprAttr>(root.getResult().getType().getShape()[0]);
      bool covered = count &&
                     physical.getKind() ==
                         PhysicalExprKind::Constant &&
                     physical.getValue() >= *count;
      auto realization = analysis.axisRealization(root.getResult(), 0);
      if (!covered &&
          (!isZero(root.getLogicalStart()) ||
           (!realization.constructionScalarSeed &&
            !samePhysicalScalarExpression(root.getExtent(),
                                           root.getLogicalStop()))))
        return false;
    }
    MakeRangeOp range = *authority;
    auto count = constantLogicalRangeCardinality(range);
    PhysicalExprAttr extent =
        count ? PhysicalExprAttr::get(
                    reshape.getContext(),
                    PhysicalExprKind::Constant, *count,
                    StringAttr::get(reshape.getContext(), ""),
                    ArrayAttr::get(reshape.getContext(), {}))
              : queryLaunchExpression(range.getLogicalStop());
    if (!extent)
      return false;
    if (!count && !queryNonNegativeIndexUpperBound(range.getLogicalStop())) {
      auto zero = PhysicalExprAttr::get(
          reshape.getContext(), PhysicalExprKind::Constant,
          0, StringAttr::get(reshape.getContext(), ""), ArrayAttr::get(reshape.getContext(), {}));
      extent = PhysicalExprAttr::get(
          reshape.getContext(), PhysicalExprKind::Maximum,
          0, StringAttr::get(reshape.getContext(), ""),
          ArrayAttr::get(reshape.getContext(), {extent, zero}));
    }
    sourceRanges[axis] = range;
    sourceExtents[axis] = extent;
  }
  SmallVector<PhysicalExprAttr> resultExtents(resultRank);
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getResultAxes().empty())
      continue;
    unsigned resultAxis = group.getResultAxes()[0];
    if (preservedResult[resultAxis])
      continue;
    auto axes = group.getSourceAxes().asArrayRef();
    auto extent = cast<PhysicalExprAttr>(sourceExtents[axes.front()]);
    if (group.getResultAxes().size() > 1) {
      // The declared row-major group may split a complete source axis.
      // Static result extents give each constituent its own exact range;
      // physical padding or a partial source tile cannot satisfy this proof.
      int64_t product = 1;
      for (int64_t axis : group.getResultAxes().asArrayRef()) {
        auto part = cast<PhysicalExprAttr>(result.getShape()[axis]);
        if (part.getKind() !=
                PhysicalExprKind::Constant ||
            part.getValue() <= 0 ||
            product > std::numeric_limits<int64_t>::max() / part.getValue())
          return false;
        product *= part.getValue();
        resultExtents[axis] = part;
      }
      int64_t sourceProduct = 1;
      for (int64_t axis : axes) {
        auto part = cast<PhysicalExprAttr>(sourceExtents[axis]);
        if (part.getKind() != PhysicalExprKind::Constant ||
            part.getValue() <= 0 ||
            sourceProduct > std::numeric_limits<int64_t>::max() / part.getValue())
          return false;
        sourceProduct *= part.getValue();
      }
      if (sourceProduct != product)
        return false;
      continue;
    }
    for (int64_t axis : axes.drop_front())
      extent = PhysicalExprAttr::get(reshape.getContext(),
          PhysicalExprKind::Multiply, 0,
          StringAttr::get(reshape.getContext(), ""),
          ArrayAttr::get(reshape.getContext(), {extent, sourceExtents[axis]}));
    resultExtents[resultAxis] = extent;
  }
  if (llvm::all_of(preservedResult, [](bool preserved) { return preserved; })) {
    bool exposesReductionPairs = llvm::any_of(
        reshape.getResult().getUsers(), [&](Operation *user) {
          auto contract = dyn_cast<ContractOp>(user);
          return contract && contract.getLhsReductionAxes().size() > 1 &&
                 (contract.getLhs() == reshape.getResult() ||
                  contract.getRhs() == reshape.getResult());
        });
    if (transposed || !exposesReductionPairs)
      return false;
    // Unit-axis insertion/removal changes the access schema, not its members.
    // Expose the load when this enables multi-pair contraction normalization;
    // a single-pair contraction can retain its existing load factorization.
    OpBuilder builder(reshape);
    auto project = [&](Value value) -> FailureOr<Value> {
      if (!value)
        return Value();
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment)
        return value;
      auto expandedType = FragmentType::get(
          source.getContext(), fragment.getElementType(), source.getShape(),
          source.getAxisMaps(), source.getValidity(), source.getOwner());
      auto expanded = projectPhysicalValueToSchema(
          builder, reshape.getLoc(), value, expandedType);
      if (failed(expanded))
        return failure();
      auto projectedType = FragmentType::get(
          result.getContext(), fragment.getElementType(), result.getShape(),
          result.getAxisMaps(), result.getValidity(), result.getOwner());
      return Value(builder.create<ReshapeOp>(
          reshape.getLoc(), projectedType, *expanded,
          reshape.getReassociation()));
    };
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates()) {
      auto projected = project(coordinate);
      if (failed(projected))
        return reshape.emitOpError("cannot project a unit-axis load coordinate"),
               failure();
      coordinates.push_back(*projected);
    }
    auto valid = project(load.getValid());
    auto fill = project(load.getFill());
    if (failed(valid) || failed(fill))
      return reshape.emitOpError("cannot project unit-axis load validity/fill"),
             failure();
    auto replacement = builder.create<LoadOp>(
        reshape.getLoc(), result, load.getResource(), coordinates, *valid, *fill,
        load.getSourceAxesAttr());
    replacement->setDiscardableAttrs(
        llvm::to_vector(load->getDiscardableAttrs()));
    reshape.getResult().replaceAllUsesWith(replacement.getResult());
    reshape.erase();
    return true;
  }
  OpBuilder builder(reshape);
  auto materializeExtent = [&](PhysicalExprAttr extent) -> Value {
    if (extent.getKind() ==
        PhysicalExprKind::Constant)
      return builder.create<arith::ConstantIndexOp>(reshape.getLoc(),
                                                    extent.getValue());
    return builder.create<PhysicalExprOp>(reshape.getLoc(), builder.getIndexType(),
                                         extent);
  };
  SmallVector<Value> resultStops(resultRank);
  for (unsigned axis = 0; axis < resultRank; ++axis)
    if (!preservedResult[axis])
      resultStops[axis] = materializeExtent(resultExtents[axis]);

  Value zero = builder.create<arith::ConstantIndexOp>(reshape.getLoc(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(reshape.getLoc(), 1);
  SmallVector<Value> flatCoordinates(resultRank);
  auto indexType = FragmentType::get(
      result.getContext(), builder.getIndexType(), result.getShape(),
      result.getAxisMaps(), result.getValidity(), result.getOwner());
  for (unsigned axis = 0; axis < resultRank; ++axis) {
    if (preservedResult[axis])
      continue;
    auto mapping = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
    Value stop = resultStops[axis];
    Value extent = materializeExtent(
        cast<PhysicalExprAttr>(result.getShape()[axis]));
    auto rangeType = FragmentType::get(
        result.getContext(), builder.getIndexType(),
        builder.getArrayAttr({result.getShape()[axis]}),
        builder.getArrayAttr({AxisMapAttr::get(
            result.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
            mapping.getDimensionId(), 0, mapping.getDerived())}),
        result.getValidity(), result.getOwner());
    Value range = builder.create<MakeRangeOp>(
        reshape.getLoc(), rangeType, zero, extent, one, zero, stop,
        mapping.getSourceId(), mapping.getSourceAxis(), mapping.getDerived());
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, reshape.getLoc(), range, indexType);
    if (failed(projected))
      return reshape.emitOpError("collapsed load has no flat coordinate projection");
    flatCoordinates[axis] = *projected;
  }
  SmallVector<Value> sourceCoordinates(sourceRank);
  IRMapping mapping;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty())
      continue;
    if (group.getResultAxes().empty()) {
      // A removed logical singleton still contributes its original address.
      // It becomes uniform in the destination shape, not an extra flat axis.
      for (int64_t axis : group.getSourceAxes().asArrayRef())
        for (MakeRangeOp root : sourceRoots[axis])
          mapping.map(root.getResult(), builder.create<SplatOp>(
              reshape.getLoc(), indexType, root.getStart()).getResult());
      continue;
    }
    if (preservedResult[group.getResultAxes()[0]]) {
      // Unmerged axes retain their current tile and coordinates, including
      // program-local batch coordinates and already blocked free dimensions.
      for (MakeRangeOp root : sourceRoots[group.getSourceAxes()[0]]) {
        unsigned axis = group.getResultAxes()[0];
        SmallVector<Attribute> shape(resultRank,
            PhysicalExprAttr::get(result.getContext(),
                PhysicalExprKind::Constant, 1,
                builder.getStringAttr(""), builder.getArrayAttr({})));
        shape[axis] = root.getResult().getType().getShape()[0];
        SmallVector<Attribute> groups;
        for (unsigned position = 0; position < resultRank; ++position) {
          SmallVector<int64_t> inputAxes;
          if (position == axis)
            inputAxes.push_back(0);
          groups.push_back(ReshapeGroupAttr::get(result.getContext(),
              builder.getDenseI64ArrayAttr(inputAxes),
              builder.getDenseI64ArrayAttr({position})));
        }
        auto shaped = FragmentType::get(result.getContext(), builder.getIndexType(),
            builder.getArrayAttr(shape), indexType.getAxisMaps(),
            indexType.getValidity(), indexType.getOwner());
        Value positioned = builder.create<ReshapeOp>(reshape.getLoc(), shaped,
            root.getResult(), builder.getArrayAttr(groups));
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, reshape.getLoc(), positioned, indexType);
        if (failed(projected))
          return reshape.emitOpError("collapsed load lost an unmerged axis projection");
        mapping.map(root.getResult(), *projected);
      }
      continue;
    }
    Value ordinal = flatCoordinates[group.getResultAxes()[0]];
    for (int64_t axis : group.getResultAxes().asArrayRef().drop_front()) {
      Value extent = builder.create<SplatOp>(reshape.getLoc(), indexType,
                                            resultStops[axis]);
      ordinal = builder.create<BinaryOp>(reshape.getLoc(), indexType, ordinal,
                                        extent, BinaryOperator::Multiply);
      ordinal = builder.create<BinaryOp>(reshape.getLoc(), indexType, ordinal,
                                        flatCoordinates[axis],
                                        BinaryOperator::Add);
    }
    auto axes = group.getSourceAxes().asArrayRef();
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned axis = axes[position];
      MakeRangeOp range = sourceRanges[axis];
      Value coordinate = ordinal;
      if (position != 0) {
        Value divisor = builder.create<BinaryOp>(
            reshape.getLoc(), builder.getIndexType(),
            materializeExtent(cast<PhysicalExprAttr>(sourceExtents[axis])), one,
            BinaryOperator::Maximum);
        Value extent = builder.create<SplatOp>(reshape.getLoc(), indexType, divisor);
        coordinate = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, ordinal, extent, BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, ordinal, extent, BinaryOperator::FloorDivide);
      }
      if (!isZero(range.getLogicalStart())) {
        Value start = builder.create<SplatOp>(
            reshape.getLoc(), indexType, range.getLogicalStart());
        coordinate = builder.create<BinaryOp>(
            reshape.getLoc(), indexType, start, coordinate, BinaryOperator::Add);
      }
      sourceCoordinates[axis] = coordinate;
      for (MakeRangeOp root : sourceRoots[axis])
        mapping.map(root.getResult(), coordinate);
    }
  }
  SmallVector<Value> coordinates;
  for (Value coordinate : load.getCoordinates()) {
    FailureOr<Value> replayed = replayFragmentValue(
        builder, coordinate, result, mapping, analysis);
    if (failed(replayed))
      return reshape.emitOpError("collapsed load could not preserve its coordinate graph");
    coordinates.push_back(*replayed);
  }
  FailureOr<Value> valid = replayFragmentValue(
      builder, load.getValid(), result, mapping, analysis);
  FailureOr<Value> fill = replayFragmentValue(
      builder, load.getFill(), result, mapping, analysis);
  if (failed(valid) || failed(fill))
    return reshape.emitOpError("collapsed load could not preserve validity and fill");
  auto predicate = FragmentType::get(
      result.getContext(), builder.getI1Type(), result.getShape(), result.getAxisMaps(),
      result.getValidity(), result.getOwner());
  Value active = *valid;
  for (unsigned axis = 0; axis < sourceRank; ++axis) {
    if (preservedSource[axis])
      continue;
    Value coordinate = sourceCoordinates[axis];
    Value lower = builder.create<SplatOp>(
        reshape.getLoc(), indexType, sourceRanges[axis].getLogicalStart());
    Value end = builder.create<SplatOp>(
        reshape.getLoc(), indexType, sourceRanges[axis].getLogicalStop());
    Value nonNegative = builder.create<CompareOp>(
        reshape.getLoc(), predicate, coordinate, lower, ComparePredicate::Ge);
    Value belowEnd = builder.create<CompareOp>(
        reshape.getLoc(), predicate, coordinate, end, ComparePredicate::Lt);
    Value within = builder.create<BinaryOp>(
        reshape.getLoc(), predicate, nonNegative, belowEnd, BinaryOperator::LogicalAnd);
    active = active ? Value(builder.create<BinaryOp>(
        reshape.getLoc(), predicate, active, within, BinaryOperator::LogicalAnd)) : within;
  }
  if (!*fill) {
    fill = materializeZeroFragment(builder, reshape.getLoc(), result);
    if (failed(fill))
      return failure();
  }
  auto replacement = builder.create<LoadOp>(
      reshape.getLoc(), result, load.getResource(), coordinates, active, *fill,
      load.getSourceAxes());
  if (Attribute origin = load->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  reshape.getResult().replaceAllUsesWith(replacement.getResult());
  reshape.erase();
  if (load.getResult().use_empty())
    load.erase();
  return true;
}

FailureOr<bool> composeReshapedStore(StoreOp store) {
  auto isPositionalRebinding = [](Operation *operation) {
      if (!isa_and_nonnull<BroadcastOp, ReshapeOp>(operation))
        return false;
      auto source = dyn_cast<FragmentType>(operation->getOperand(0).getType());
      auto result = dyn_cast<FragmentType>(operation->getResult(0).getType());
      if (!source || !result || source.getShape() != result.getShape() ||
          source.getElementType() != result.getElementType() ||
          source.getOwner() != result.getOwner() || source.getValidity() != result.getValidity())
        return false;
      if (auto reshape = dyn_cast<ReshapeOp>(operation))
        return llvm::all_of(reshape.getReassociation(), [](Attribute attribute) {
          auto group = cast<ReshapeGroupAttr>(attribute);
          return group.getSourceAxes().size() == 1 &&
                 group.getResultAxes().size() == 1 &&
                 group.getSourceAxes()[0] == group.getResultAxes()[0];
        });
      BroadcastProjection relation = queryAxisProjection(source, result);
      if (!relation.isExact())
        return false;
      for (auto [axis, inputAxis] : llvm::enumerate(relation.targetToSource)) {
        auto extent = cast<PhysicalExprAttr>(source.getShape()[axis]);
        bool unit = extent.getKind() ==
                        PhysicalExprKind::Constant &&
                    extent.getValue() == 1;
        if (inputAxis ? *inputAxis != axis : !unit)
          return false;
      }
      return true;
  };
  auto stripPositionalRebindings = [&](Value value) {
    while (isPositionalRebinding(value.getDefiningOp()))
      value = value.getDefiningOp()->getOperand(0);
    return value;
  };
  auto output = dyn_cast<FragmentType>(store.getValue().getType());
  if (!output || output.getShape().empty())
    return false;
  auto kernel = store->getParentOfType<func::FuncOp>();
  PhysicalProgramAnalysis analysis(kernel);
  unsigned outputRank = output.getShape().size();
  SmallVector<MakeRangeOp> outputRanges(outputRank);
  SmallVector<unsigned> coordinateSlots(outputRank);
  for (auto [slot, coordinate] : llvm::enumerate(store.getCoordinates())) {
    if (!isa<FragmentType>(coordinate.getType()))
      continue;
    auto range = coordinate.getDefiningOp<MakeRangeOp>();
    FailureOr<int64_t> dimension = range ? queryRangeDimension(range)
                                         : FailureOr<int64_t>(failure());
    if (!range || failed(dimension) || !isZero(range.getStart()) ||
        !isZero(range.getLogicalStart()) || !isUnitStepRange(range))
      return false;
    auto realization = analysis.axisRealization(range.getResult(), 0);
    if (!realization.constructionScalarSeed &&
        !samePhysicalScalarExpression(range.getExtent(), range.getLogicalStop()))
      return false;
    auto projection = queryFragmentDimension(output, *dimension);
    if (!projection.isExact() || outputRanges[projection.fragmentAxis])
      return false;
    outputRanges[projection.fragmentAxis] = range;
    coordinateSlots[projection.fragmentAxis] = slot;
  }
  if (llvm::any_of(outputRanges, [](MakeRangeOp range) { return !range; }))
    return false;

  SmallVector<unsigned> outerAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outerAxes.push_back(axis);
  auto isPointwise = [](Operation *operation) {
    return operation && isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
                            BitcastOp, SplatOp, BroadcastOp>(operation);
  };
  ReshapeOp reshape;
  llvm::SmallPtrSet<Operation *, 8> transposes;
  std::function<bool(Value, SmallVector<unsigned>)> findReshape =
      [&](Value value, SmallVector<unsigned> axes) {
        value = stripPositionalRebindings(value);
        if (auto transpose = value.getDefiningOp<TransposeOp>()) {
          if (transpose.getPermutation().size() != axes.size())
            return false;
          auto source = cast<FragmentType>(transpose.getValue().getType());
          auto result = cast<FragmentType>(transpose.getResult().getType());
          SmallVector<unsigned> inputAxes(axes.size());
          for (auto [axis, sourceAxis] : llvm::enumerate(transpose.getPermutation())) {
            auto sourceMap = cast<AxisMapAttr>(source.getAxisMaps()[sourceAxis]);
            auto resultMap = cast<AxisMapAttr>(result.getAxisMaps()[axis]);
            if (!(sourceAxisIdentity(sourceMap) == sourceAxisIdentity(resultMap)) ||
                sourceMap.getDimensionId() != resultMap.getDimensionId())
              return false;
            inputAxes[sourceAxis] = axes[axis];
          }
          if (!findReshape(transpose.getValue(), std::move(inputAxes)))
            return false;
          transposes.insert(transpose);
          return true;
        }
        if (auto candidate = value.getDefiningOp<ReshapeOp>()) {
          reshape = candidate;
          outerAxes = std::move(axes);
          return true;
        }
        Operation *producer = value.getDefiningOp();
        return isPointwise(producer) && !isa<BroadcastOp>(producer) &&
               llvm::any_of(producer->getOperands(), [&](Value operand) {
                 return findReshape(operand, axes);
               });
      };
  if (!findReshape(store.getValue(), outerAxes))
    return false;
  SmallVector<LoadOp> companions;
  SmallVector<ReshapeOp> sourceReshapes{reshape};
  llvm::DenseMap<Value, unsigned> companionAxes;
  using OutputAxes = SmallVector<std::optional<unsigned>>;
  llvm::DenseMap<Value, OutputAxes> checked;
  auto isSourceFrame = [&](ArrayRef<std::optional<unsigned>> axes) {
    return axes.size() == outerAxes.size() &&
           llvm::all_of(llvm::zip(axes, outerAxes), [](const auto &pair) {
             return std::get<0>(pair) == std::get<1>(pair);
           });
  };
  std::function<bool(Value, OutputAxes)> canReplayEpilogue =
      [&](Value value, OutputAxes axes) {
    auto result = dyn_cast<FragmentType>(value.getType());
    if (value == reshape.getResult())
      return isSourceFrame(axes);
    if (!result)
      return true;
    if (axes.size() != result.getShape().size())
      return false;
    auto [entry, inserted] = checked.try_emplace(value, axes);
    if (!inserted) {
      bool changed = false;
      for (auto [known, current] : llvm::zip(entry->second, axes)) {
        if (known && current && known != current)
          return false;
        if (!known && current) {
          known = current;
          changed = true;
        }
      }
      if (!changed)
        return true;
      axes = entry->second;
    }
    Operation *producer = value.getDefiningOp();
    if (auto transpose = dyn_cast_or_null<TransposeOp>(producer)) {
      if (!transposes.contains(producer))
        return false;
      OutputAxes inputAxes(axes.size());
      for (auto [axis, inputAxis] : llvm::enumerate(transpose.getPermutation()))
        inputAxes[inputAxis] = axes[axis];
      return canReplayEpilogue(transpose.getValue(), std::move(inputAxes));
    }
    if (auto load = dyn_cast_or_null<LoadOp>(producer)) {
      if (!canReplayReadAt(load, store))
        return false;
      for (auto [axis, outputAxis] : llvm::enumerate(axes)) {
        PhysicalRangeFact ranges = analysis.axisRanges(value, axis);
        if (ranges.state == PhysicalFactState::Unknown || !ranges.blockers.empty())
          return false;
        if (ranges.roots.empty())
          continue;
        if (!outputAxis || *outputAxis >= outputRanges.size())
          return false;
        for (MakeRangeOp range : ranges.roots) {
          if (!analysis.lockstepRanges({range, outputRanges[*outputAxis]}).isExact())
            return false;
          auto [found, inserted] = companionAxes.try_emplace(range.getResult(), *outputAxis);
          if (!inserted && found->second != *outputAxis)
            return false;
        }
      }
      companions.push_back(load);
      return true;
    }
    if (auto companion = dyn_cast_or_null<ReshapeOp>(producer)) {
      auto source = cast<FragmentType>(companion.getValue().getType());
      auto primarySource = cast<FragmentType>(reshape.getValue().getType());
      auto primaryResult = cast<FragmentType>(reshape.getResult().getType());
      if (companion.getReassociation() == reshape.getReassociation() &&
          source.getShape() == primarySource.getShape() &&
          source.getAxisMaps() == primarySource.getAxisMaps() &&
          source.getOwner() == primarySource.getOwner() &&
          source.getValidity() == primarySource.getValidity() &&
          result.getShape() == primaryResult.getShape() &&
          result.getAxisMaps() == primaryResult.getAxisMaps() &&
          result.getOwner() == primaryResult.getOwner() &&
          result.getValidity() == primaryResult.getValidity()) {
        if (!isSourceFrame(axes))
          return false;
        if (!llvm::is_contained(sourceReshapes, companion))
          sourceReshapes.push_back(companion);
        return true;
      }
      auto unitAxes = [](FragmentType type, ArrayRef<int64_t> axes, unsigned prefix) {
        return llvm::all_of(axes, [&](int64_t axis) {
          auto extent = cast<PhysicalExprAttr>(type.getShape()[prefix + axis]);
          return extent.getKind() ==
                     PhysicalExprKind::Constant &&
                 extent.getValue() == 1;
        });
      };
      unsigned sourceRank = 0, resultRank = 0;
      for (Attribute attribute : companion.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        sourceRank += group.getSourceAxes().size();
        resultRank += group.getResultAxes().size();
      }
      unsigned sourcePrefix = source.getShape().size() - sourceRank;
      unsigned resultPrefix = result.getShape().size() - resultRank;
      if (sourcePrefix != resultPrefix)
        return false;
      OutputAxes inputAxes(source.getShape().size());
      for (unsigned axis = 0; axis < sourcePrefix; ++axis)
        inputAxes[axis] = axes[axis];
      for (Attribute attribute : companion.getReassociation()) {
        auto group = cast<ReshapeGroupAttr>(attribute);
        if (group.getSourceAxes().empty()) {
          if (!unitAxes(result, group.getResultAxes().asArrayRef(), resultPrefix))
            return false;
        } else if (group.getResultAxes().empty()) {
          if (!unitAxes(source, group.getSourceAxes().asArrayRef(), sourcePrefix))
            return false;
        } else if (group.getSourceAxes().size() != 1 ||
                   group.getResultAxes().size() != 1) {
          return false;
        } else {
          inputAxes[sourcePrefix + group.getSourceAxes()[0]] =
              axes[resultPrefix + group.getResultAxes()[0]];
        }
      }
      return canReplayEpilogue(companion.getValue(), std::move(inputAxes));
    }
    if (!isPointwise(producer))
      return false;
    return llvm::all_of(producer->getOperands(), [&](Value operand) {
      auto input = dyn_cast<FragmentType>(operand.getType());
      if (!input)
        return true;
      auto projection = queryBroadcastProjection(input, result);
      if (!projection.isExact())
        return false;
      OutputAxes inputAxes(input.getShape().size());
      for (auto [axis, inputAxis] : llvm::enumerate(projection.targetToSource)) {
        if (!inputAxis || !axes[axis])
          continue;
        if (inputAxes[*inputAxis] && inputAxes[*inputAxis] != axes[axis])
          return false;
        inputAxes[*inputAxis] = axes[axis];
      }
      return canReplayEpilogue(operand, std::move(inputAxes));
    });
  };
  OutputAxes outputAxes;
  for (unsigned axis = 0; axis < outputRank; ++axis)
    outputAxes.push_back(axis);
  if (!canReplayEpilogue(store.getValue(), std::move(outputAxes)))
    return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  if (!input)
    return false;
  unsigned logicalSourceRank = 0;
  unsigned logicalResultRank = 0;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    for (int64_t axis : group.getSourceAxes().asArrayRef())
      logicalSourceRank = std::max(logicalSourceRank, static_cast<unsigned>(axis + 1));
    for (int64_t axis : group.getResultAxes().asArrayRef())
      logicalResultRank = std::max(logicalResultRank, static_cast<unsigned>(axis + 1));
  }
  if (logicalSourceRank != input.getShape().size() || logicalResultRank != outputRank)
    return false;
  SmallVector<MakeRangeOp> inputRanges;
  SmallVector<Attribute> inputExtents;
  for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
    PhysicalRangeFact ranges = analysis.axisRanges(reshape.getValue(), axis);
    FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
    if (failed(range) && ranges.blockers.empty()) {
      auto traversal = analysis.lockstepRanges(ranges.roots);
      if (traversal.isExact())
        range = traversal.authority;
    }
    if (failed(range) || !isZero((*range).getStart()) ||
        !isZero((*range).getLogicalStart()) || !isUnitStepRange(*range))
      return false;
    auto realization = analysis.axisRealization(reshape.getValue(), axis);
    if (!realization.constructionScalarSeed &&
        !samePhysicalScalarExpression((*range).getExtent(), (*range).getLogicalStop()))
      return false;
    PhysicalExprAttr extent = queryLaunchExpression((*range).getLogicalStop());
    if (!extent)
      return false;
    inputRanges.push_back(*range);
    inputExtents.push_back(extent);
  }
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    if (group.getSourceAxes().empty() && group.getResultAxes().empty())
      return false;
    SmallVector<Attribute> sourceExtents;
    for (int64_t axis : group.getSourceAxes().asArrayRef()) {
      if (axis < 0 || axis >= static_cast<int64_t>(inputRanges.size()))
        return false;
      sourceExtents.push_back(inputExtents[axis]);
    }
    SmallVector<Attribute> resultExtents;
    for (int64_t axis : group.getResultAxes().asArrayRef()) {
      if (axis < 0 || axis >= static_cast<int64_t>(outerAxes.size()))
        return false;
      PhysicalExprAttr extent = queryLaunchExpression(
          outputRanges[outerAxes[axis]].getLogicalStop());
      if (!extent)
        return false;
      resultExtents.push_back(extent);
    }
    // Check logical range extents, never provisional physical singletons.
    // The existing reassociation supplies the row-major axis order.
    if (failed(inferReshapeReassociation(
            store.getContext(), sourceExtents, resultExtents)))
      return false;
  }

  OpBuilder builder(store);
  SmallVector<Attribute> inputMappings;
  for (auto [axis, range] : llvm::enumerate(inputRanges)) {
    auto mapping = cast<AxisMapAttr>(range.getResult().getType().getAxisMaps()[0]);
    inputMappings.push_back(AxisMapAttr::get(
        store.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), axis, mapping.getDerived()));
  }
  input = FragmentType::get(
      store.getContext(), input.getElementType(), input.getShape(),
      builder.getArrayAttr(inputMappings), input.getValidity(), input.getOwner());
  auto indexType = FragmentType::get(
      input.getContext(), builder.getIndexType(), input.getShape(),
      input.getAxisMaps(), input.getValidity(), input.getOwner());
  SmallVector<Value> coordinates(store.getCoordinates());
  IRMapping mapping;
  for (Attribute attribute : reshape.getReassociation()) {
    auto group = cast<ReshapeGroupAttr>(attribute);
    Value ordinal;
    for (int64_t sourceAxis : group.getSourceAxes().asArrayRef()) {
      auto rangeType = inputRanges[sourceAxis].getResult().getType();
      SmallVector<Attribute> axisShape(input.getShape().size(),
          PhysicalExprAttr::get(store.getContext(),
              PhysicalExprKind::Constant, 1,
              builder.getStringAttr(""), builder.getArrayAttr({})));
      axisShape[sourceAxis] = rangeType.getShape()[0];
      auto axisType = FragmentType::get(
          store.getContext(), builder.getIndexType(), builder.getArrayAttr(axisShape),
          input.getAxisMaps(), input.getValidity(), input.getOwner());
      SmallVector<Attribute> relation;
      for (unsigned axis = 0; axis < input.getShape().size(); ++axis) {
        SmallVector<int64_t> sourceAxes;
        if (axis == sourceAxis)
          sourceAxes.push_back(0);
        relation.push_back(ReshapeGroupAttr::get(
            store.getContext(), builder.getDenseI64ArrayAttr(sourceAxes),
            builder.getDenseI64ArrayAttr({static_cast<int64_t>(axis)})));
      }
      Value projected = builder.create<ReshapeOp>(
          store.getLoc(), axisType, inputRanges[sourceAxis].getResult(),
          builder.getArrayAttr(relation));
      if (axisType != indexType)
        projected = builder.create<BroadcastOp>(store.getLoc(), indexType, projected);
      if (ordinal) {
        Value extent = builder.create<SplatOp>(
            store.getLoc(), indexType, inputRanges[sourceAxis].getLogicalStop());
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::Multiply);
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, projected, BinaryOperator::Add);
      } else {
        ordinal = projected;
      }
    }
    if (!ordinal) {
      Value zero = builder.create<arith::ConstantIndexOp>(store.getLoc(), 0);
      ordinal = builder.create<SplatOp>(store.getLoc(), indexType, zero);
    }
    auto axes = group.getResultAxes().asArrayRef();
    for (unsigned position = axes.size(); position-- > 0;) {
      unsigned outerAxis = outerAxes[axes[position]];
      MakeRangeOp range = outputRanges[outerAxis];
      Value coordinate = ordinal;
      if (position != 0) {
        Value one = builder.create<arith::ConstantIndexOp>(store.getLoc(), 1);
        // Empty logical domains have no active store.  A positive divisor
        // keeps their inactive physical lanes well-defined as well.
        Value divisor = builder.create<BinaryOp>(
            store.getLoc(), builder.getIndexType(), range.getLogicalStop(), one,
            BinaryOperator::Maximum);
        Value extent = builder.create<SplatOp>(store.getLoc(), indexType, divisor);
        coordinate = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::Remainder);
        ordinal = builder.create<BinaryOp>(
            store.getLoc(), indexType, ordinal, extent, BinaryOperator::FloorDivide);
      }
      coordinates[coordinateSlots[outerAxis]] = coordinate;
      mapping.map(range.getResult(), coordinate);
      for (auto [root, axis] : companionAxes)
        if (axis == outerAxis)
          mapping.map(root, coordinate);
      kernel.walk([&](MakeRangeOp occurrence) {
        FailureOr<int64_t> occurrenceDimension = queryRangeDimension(occurrence);
        FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
        if (succeeded(occurrenceDimension) && succeeded(rangeDimension) &&
            *occurrenceDimension == *rangeDimension &&
            analysis.lockstepRanges({occurrence, range}).isExact())
          mapping.map(occurrence.getResult(), coordinate);
      });
    }
  }
  FailureOr<Value> valid = replayFragmentValue(
      builder, store.getValid(), input, mapping, analysis);
  if (failed(valid))
    return store.emitOpError("flattened store could not preserve its access validity");
  auto predicate = FragmentType::get(
      input.getContext(), builder.getI1Type(), input.getShape(), input.getAxisMaps(),
      input.getValidity(), input.getOwner());
  Value zero = builder.create<arith::ConstantIndexOp>(store.getLoc(), 0);
  Value lower = builder.create<SplatOp>(store.getLoc(), indexType, zero);
  Value active = *valid;
  for (unsigned axis = 0; axis < outputRank; ++axis) {
    Value coordinate = coordinates[coordinateSlots[axis]];
    Value end = builder.create<SplatOp>(
        store.getLoc(), indexType, outputRanges[axis].getLogicalStop());
    Value nonNegative = builder.create<CompareOp>(
        store.getLoc(), predicate, coordinate, lower, ComparePredicate::Ge);
    Value belowEnd = builder.create<CompareOp>(
        store.getLoc(), predicate, coordinate, end, ComparePredicate::Lt);
    Value within = builder.create<BinaryOp>(
        store.getLoc(), predicate, nonNegative, belowEnd, BinaryOperator::LogicalAnd);
    active = active ? Value(builder.create<BinaryOp>(
        store.getLoc(), predicate, active, within, BinaryOperator::LogicalAnd)) : within;
  }
  Value source = reshape.getValue();
  if (source.getType() != input)
    source = builder.create<BroadcastOp>(store.getLoc(), input, source);
  mapping.map(reshape.getResult(), source);
  for (ReshapeOp companion : llvm::drop_begin(sourceReshapes)) {
    auto original = cast<FragmentType>(companion.getValue().getType());
    auto target = FragmentType::get(
        input.getContext(), original.getElementType(), input.getShape(),
        input.getAxisMaps(), input.getValidity(), input.getOwner());
    FailureOr<Value> value = projectPhysicalValueToSchema(
        builder, store.getLoc(), companion.getValue(), target);
    if (failed(value))
      return store.emitOpError("aligned reshapes have no common input relation");
    mapping.map(companion.getResult(), *value);
  }
  for (LoadOp companion : companions) {
    auto companionType = FragmentType::get(
        input.getContext(), cast<FragmentType>(companion.getResult().getType()).getElementType(),
        input.getShape(), input.getAxisMaps(), input.getValidity(), input.getOwner());
    SmallVector<Value> companionCoordinates;
    for (Value coordinate : companion.getCoordinates()) {
      if (!isa<FragmentType>(coordinate.getType())) {
        companionCoordinates.push_back(coordinate);
        continue;
      }
      FailureOr<Value> projected = replayFragmentValue(
          builder, coordinate, indexType, mapping, analysis);
      if (failed(projected))
        return store.emitOpError(
            "reshaped store companion has no exact coordinate mapping")
               << "; coordinate=" << coordinate;
      companionCoordinates.push_back(*projected);
    }
    FailureOr<Value> companionValid = replayFragmentValue(
        builder, companion.getValid(), input, mapping, analysis);
    FailureOr<Value> companionFill = replayFragmentValue(
        builder, companion.getFill(), input, mapping, analysis);
    if (failed(companionValid) || failed(companionFill))
      return store.emitOpError(
          "reshaped store companion could not preserve validity and fill");
    companionValid = combinePredicates(
        builder, store.getLoc(), input, *companionValid, active);
    if (!*companionFill)
      companionFill = materializeZeroFragment(builder, store.getLoc(), companionType);
    if (failed(companionValid) || failed(companionFill))
      return failure();
    auto loaded = builder.create<LoadOp>(
        companion.getLoc(), companionType, companion.getResource(), companionCoordinates,
        *companionValid, *companionFill, companion.getSourceAxes());
    if (Attribute origin = companion->getAttr(originAttr))
      loaded->setAttr(originAttr, origin);
    mapping.map(companion.getResult(), loaded.getResult());
  }
  std::function<FailureOr<Value>(Value)> replayEpilogue = [&](Value value) -> FailureOr<Value> {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return value;
    Operation *producer = value.getDefiningOp();
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> mapped = replayEpilogue(operand);
      if (failed(mapped))
        return failure();
      mapping.map(operand, *mapped);
    }
    auto target = FragmentType::get(
        input.getContext(), fragment.getElementType(), input.getShape(),
        input.getAxisMaps(), input.getValidity(), input.getOwner());
    if (isa<BroadcastOp, TransposeOp, ReshapeOp>(producer)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, producer->getLoc(), mapping.lookup(producer->getOperand(0)), target);
      if (succeeded(projected))
        mapping.map(value, *projected);
      return projected;
    }
    Operation *clone = builder.clone(*producer, mapping);
    clone->getResult(0).setType(target);
    mapping.map(value, clone->getResult(0));
    return clone->getResult(0);
  };
  FailureOr<Value> payload = replayEpilogue(store.getValue());
  if (failed(payload))
    return store.emitOpError("flattened store could not preserve its pointwise epilogue");
  auto replacement = builder.create<StoreOp>(
      store.getLoc(), store.getResource(), coordinates, *payload,
      active, store.getSourceAxes());
  if (Attribute origin = store->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  store.erase();
  return true;
}

bool sameImmutableLoad(LoadOp available, LoadOp current) {
  auto view = dyn_cast<ViewType>(current.getResource().getType());
  if (!view || view.getAccess() != 0 ||
      available.getResource() != current.getResource() ||
      available.getResult().getType() != current.getResult().getType() ||
      available.getValid() != current.getValid() ||
      available.getFill() != current.getFill() ||
      available.getSourceAxes() != current.getSourceAxes() ||
      available.getCoordinates().size() != current.getCoordinates().size())
    return false;
  return llvm::equal(available.getCoordinates(), current.getCoordinates());
}

bool deduplicateImmutableLoads(func::FuncOp kernel) {
  SmallVector<LoadOp> loads;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  bool changed = false;
  for (LoadOp current : loads) {
    if (!current->getBlock())
      continue;
    Block *block = current->getBlock();
    auto cursor = current->getIterator();
    while (cursor != block->begin()) {
      --cursor;
      Operation *candidate = &*cursor;
      if (candidate->getNumRegions() != 0)
        break;
      if (auto available = dyn_cast<LoadOp>(candidate)) {
        if (!sameImmutableLoad(available, current))
          continue;
        current.getResult().replaceAllUsesWith(available.getResult());
        current.erase();
        changed = true;
        break;
      }
      // Different ABI views may alias by default.  A write or another
      // side-effecting operation therefore ends the interval in which an
      // immutable-view load is known to retain its value.
      if (!isMemoryEffectFree(candidate))
        break;
    }
  }
  return changed;
}

void sinkImmutableLoadChains(func::FuncOp kernel) {
  llvm::SmallPtrSet<Operation *, 32> selected;
  SmallVector<Operation *> pending;
  kernel.walk([&](LoadOp load) {
    auto view = dyn_cast<ViewType>(load.getResource().getType());
    if (view && view.getAccess() == 0 && selected.insert(load).second)
      pending.push_back(load);
  });
  while (!pending.empty()) {
    Operation *producer = pending.pop_back_val();
    for (Operation *user : producer->getUsers()) {
      if (user->getBlock() != producer->getBlock() ||
          !isa<CastOp, BitcastOp, ReshapeOp, TransposeOp, BroadcastOp,
               SelectOp, BinaryOp, UnaryOp>(user) ||
          !isMemoryEffectFree(user) || !selected.insert(user).second)
        continue;
      pending.push_back(user);
    }
  }
  SmallVector<Operation *> ordered;
  kernel.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (selected.contains(operation))
      ordered.push_back(operation);
  });
  for (Operation *operation : llvm::reverse(ordered)) {
    Operation *firstUse = nullptr;
    for (Operation *user : operation->getUsers()) {
      Operation *ancestor = operation->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor) {
        firstUse = nullptr;
        break;
      }
      if (!firstUse || ancestor->isBeforeInBlock(firstUse))
        firstUse = ancestor;
    }
    if (!firstUse || operation->getNextNode() == firstUse)
      continue;
    bool crossesEffect = false;
    for (Operation *next = operation->getNextNode(); next != firstUse;
         next = next->getNextNode()) {
      // ABI views can alias. Do not cross writes, atomics, synchronization,
      // or an unknown effect while shortening an immutable load's live range.
      if (!isMemoryEffectFree(next) && !isa<LoadOp, GatherOp>(next)) {
        crossesEffect = true;
        break;
      }
    }
    if (!crossesEffect)
      operation->moveBefore(firstUse);
  }
}

LogicalResult materializeIndexedFragments(func::FuncOp kernel) {
  SmallVector<GatherOp> gathers;
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  for (GatherOp gather : gathers) {
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      PhysicalProgramAnalysis analysis(kernel);
      if (!analysis.axisRealization(gather.getSource(), sourceAxis)
               .constructionScalarSeed)
        continue;
      auto source = cast<FragmentType>(gather.getSource().getType());
      auto extent = cast<PhysicalExprAttr>(source.getShape()[sourceAxis]);
      PhysicalExprAttr bound = queryNonNegativeIndexUpperBound(coordinate);
      if (bound &&
          bound.getKind() == PhysicalExprKind::Constant &&
          extent.getKind() == PhysicalExprKind::Constant &&
          bound.getValue() < extent.getValue())
        continue;
      if (failed(realizeFullCoverageDimension(kernel, gather.getSource(),
                                               sourceAxis)))
        return gather.emitOpError(
            "indexed tensor source has no complete physical extent");
    }
  }
  return success();
}

} // namespace

static LogicalResult realizeAccessCompositionImpl(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  eraseDeadPhysicalValues(*physicalKernel);
  bool changed;
  do {
    changed = false;
    SmallVector<SelectOp> selects;
    physicalKernel->walk([&](SelectOp select) { selects.push_back(select); });
    for (SelectOp select : selects) {
      if (!select->getBlock())
        continue;
      FailureOr<bool> load = composeSelectLoad(select);
      if (failed(load))
        return failure();
      changed |= *load;
    }
    SmallVector<GatherOp> gathers;
    physicalKernel->walk([&](GatherOp gather) { gathers.push_back(gather); });
    for (GatherOp gather : gathers) {
      if (!gather->getBlock())
        continue;
      if (reuseFragmentGather(gather)) {
        changed = true;
        continue;
      }
      FailureOr<bool> identity = composeIdentityFragmentGather(gather);
      if (failed(identity))
        return failure();
      if (*identity) {
        changed = true;
        continue;
      }
      FailureOr<bool> range = composeRangeGather(gather);
      if (failed(range))
        return failure();
      if (*range) {
        changed = true;
        continue;
      }
      FailureOr<bool> reshaped = composeReshapedGather(gather);
      if (failed(reshaped))
        return failure();
      if (*reshaped) {
        changed = true;
        continue;
      }
      FailureOr<bool> pointwise = composePointwiseGather(gather);
      if (failed(pointwise))
        return failure();
      if (*pointwise) {
        changed = true;
        continue;
      }
      FailureOr<bool> reduced = composeReducedGather(gather);
      if (failed(reduced))
        return failure();
      if (*reduced) {
        changed = true;
        continue;
      }
      FailureOr<bool> projection = projectFragmentGather(gather);
      if (failed(projection))
        return failure();
      if (*projection) {
        changed = true;
        continue;
      }
      FailureOr<bool> broadcast = composeBroadcastGather(gather);
      if (failed(broadcast))
        return failure();
      if (*broadcast) {
        changed = true;
        continue;
      }
      FailureOr<bool> load = composeLoadGather(gather);
      if (failed(load))
        return failure();
      changed |= *load;
    }
    SmallVector<ReshapeOp> reshapes;
    physicalKernel->walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
    for (ReshapeOp reshape : reshapes) {
      if (composeReshapedPointwise(reshape)) {
        changed = true;
        continue;
      }
      FailureOr<bool> composed = composeReshapedLoad(reshape);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    SmallVector<StoreOp> stores;
    physicalKernel->walk([&](StoreOp store) { stores.push_back(store); });
    for (StoreOp store : stores) {
      FailureOr<bool> composed = composeReshapedStore(store);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    changed |= deduplicateImmutableLoads(*physicalKernel);
    changed |= foldIndexRecompositions(*physicalKernel);
    SmallVector<ReduceOp> reductions;
    physicalKernel->walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
    for (ReduceOp reduce : reductions) {
      FailureOr<bool> composed = composeReductionGathers(reduce);
      if (failed(composed))
        return failure();
      changed |= *composed;
    }
    eraseDeadPhysicalValues(*physicalKernel);
  } while (changed);
  if (failed(materializeIndexedFragments(*physicalKernel)))
    return failure();
  sinkImmutableLoadChains(*physicalKernel);
  return success();
}

LogicalResult simplifyMaskedAccessCoordinates(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  kernel->walk([&](LoadOp load) {
    Value valid = load.getValid();
    if (!valid)
      return;
    auto sameAxes = [&](Type type) {
      auto fragment = dyn_cast<FragmentType>(type);
      auto mask = dyn_cast<FragmentType>(valid.getType());
      if (!fragment || !mask)
        return !fragment && !mask;
      return fragment.getOwner() == mask.getOwner() &&
             fragment.getAxisMaps() == mask.getAxisMaps() &&
             queryBroadcastProjection(fragment, mask).isExact();
    };
    llvm::SmallDenseSet<Value, 16> conjuncts;
    SmallVector<Value> pending{valid};
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!sameAxes(value.getType()) || !conjuncts.insert(value).second)
        continue;
      if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
        if (sameAxes(broadcast.getValue().getType()))
          pending.push_back(broadcast.getValue());
        continue;
      }
      auto binary = value.getDefiningOp<BinaryOp>();
      if (!binary ||
          (binary.getOperatorKind() != BinaryOperator::LogicalAnd &&
           binary.getOperatorKind() != BinaryOperator::BitwiseAnd))
        continue;
      pending.push_back(binary.getLhs());
      pending.push_back(binary.getRhs());
    }
    OpBuilder builder(load);
    IRMapping simplified;
    std::function<Value(Value)> simplify = [&](Value value) -> Value {
      if (Value known = simplified.lookupOrNull(value))
        return known;
      if (!sameAxes(value.getType()))
        return value;
      Value result = value;
      if (auto select = value.getDefiningOp<SelectOp>();
          select && conjuncts.contains(select.getCondition())) {
        result = simplify(select.getTrueValue());
      } else if (auto broadcast = value.getDefiningOp<BroadcastOp>();
                 broadcast && sameAxes(broadcast.getValue().getType())) {
        Value operand = simplify(broadcast.getValue());
        if (operand != broadcast.getValue()) {
          IRMapping mapping;
          mapping.map(broadcast.getValue(), operand);
          result = builder.clone(*broadcast, mapping)->getResult(0);
        }
      }
      simplified.map(value, result);
      return result;
    };
    SmallVector<Value> coordinates(load.getCoordinates());
    bool changed = false;
    for (Value &coordinate : coordinates) {
      // Singleton broadcasts with unchanged axes retain the predicate's lane
      // relation. Other uses and the read's original mask stay unchanged.
      Value replacement = simplify(coordinate);
      changed |= replacement != coordinate;
      coordinate = replacement;
    }
    if (changed)
      load.getCoordinatesMutable().assign(coordinates);
  });
  eraseDeadPhysicalValues(*kernel);
  return success();
}

LogicalResult realizeAccessComposition(ModuleOp module) {
  if (failed(realizeAccessCompositionImpl(module))) return failure();
  auto kernel = getPhysicalKernel(module);
  return failed(kernel) ? failure() : closeValueRelations(*kernel);
}

} // namespace intent::gpu

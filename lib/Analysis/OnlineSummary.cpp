#include "Intent/Analysis/OnlineSummary.h"
#include "Intent/Analysis/UniformValues.h"
#include "mlir/IR/IRMapping.h"
#include <functional>

using namespace mlir;
namespace intent {
namespace {
Attribute constant(Value value) {
  return UniformValueAnalysis(describeCanonicalUniformValue).evaluate(value);
}
bool zero(Value value) {
  auto attr = constant(value);
  if (auto number = dyn_cast_or_null<FloatAttr>(attr)) return number.getValue().isZero();
  if (auto number = dyn_cast_or_null<IntegerAttr>(attr)) return number.getValue().isZero();
  return false;
}
bool truth(Value value, bool expected) {
  auto attr = dyn_cast_or_null<IntegerAttr>(constant(value));
  return attr && attr.getType().isInteger(1) && attr.getValue().getBoolValue() == expected;
}
bool negativeInfinity(Value value) {
  auto attr = dyn_cast_or_null<FloatAttr>(constant(value));
  return attr && attr.getValue().isInfinity() && attr.getValue().isNegative();
}
Value projected(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (isa<BroadcastOp>(op)) { value = op->getOperand(0); continue; }
    if (auto gather = dyn_cast<GatherOp>(op)) {
      bool insertion = gather.getValidOperandIndex() &&
          truth(gather.getInputs()[*gather.getValidOperandIndex()], true);
      unsigned retained = 0;
      for (Attribute attr : gather.getIndex().getTerms()) {
        auto term = cast<IndexTermAttr>(attr);
        insertion &= term.getKind() == 0 || term.getKind() == 1;
        retained += term.getKind() == 0;
      }
      insertion &= retained == gather.getIndex().getSourceRank();
      if (insertion) { value = gather.getInputs()[0]; continue; }
    }
    break;
  }
  return value;
}
std::optional<BinaryOperator> reductionKind(ReduceOp reduce) {
  if (!reduce || reduce.getNumResults() != 1 || reduce.getAxes().size() != 1 ||
      reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 || reduce.getCaptureCount())
    return std::nullopt;
  auto &body = reduce.getCombine().front();
  auto combine = body.getTerminator()->getOperand(0).getDefiningOp<BinaryOp>();
  if (!combine || body.getNumArguments() != 2 ||
      !((combine.getLhs() == body.getArgument(0) && combine.getRhs() == body.getArgument(1)) ||
        (combine.getLhs() == body.getArgument(1) && combine.getRhs() == body.getArgument(0)))) return std::nullopt;
  return combine.getOperatorKind();
}
bool field(Value value, Value record, unsigned index) {
  auto extract = projected(value).getDefiningOp<ExtractOp>();
  return extract && extract.getProduct() == record && extract.getField() == index;
}
bool fieldPair(Value value, BinaryOperator kind, Value left, Value right, unsigned index) {
  auto op = projected(value).getDefiningOp<BinaryOp>();
  return op && op.getOperatorKind() == kind &&
      ((field(op.getLhs(), left, index) && field(op.getRhs(), right, index)) ||
       (field(op.getRhs(), left, index) && field(op.getLhs(), right, index)));
}
bool scale(Value value, Value record, OnlineSummary plan) {
  auto select = projected(value).getDefiningOp<SelectOp>();
  if (!select || !field(select.getCondition(), record, plan.validField) || !zero(select.getFalseValue())) return false;
  auto exponential = select.getTrueValue().getDefiningOp<UnaryOp>();
  if (!exponential || exponential.getOperatorKind() != plan.exponential.getOperatorKind() ||
      exponential.getApproximate() != plan.exponential.getApproximate() ||
      exponential.getFlushToZero() != plan.exponential.getFlushToZero()) return false;
  auto subtract = exponential.getInput().getDefiningOp<BinaryOp>();
  if (!subtract || subtract.getOperatorKind() != BinaryOperator::Subtract ||
      projected(subtract.getRhs()) != projected(plan.combinedMaximum)) return false;
  auto valid = projected(subtract.getLhs()).getDefiningOp<SelectOp>();
  return valid && field(valid.getCondition(), record, plan.validField) &&
      field(valid.getTrueValue(), record, plan.maxField) &&
      projected(valid.getFalseValue()) == projected(plan.combinedMaximum);
}
bool scaled(Value value, Value factor, Value record, unsigned index) {
  auto multiply = projected(value).getDefiningOp<BinaryOp>();
  return multiply && multiply.getOperatorKind() == BinaryOperator::Multiply &&
      ((projected(multiply.getLhs()) == factor && field(multiply.getRhs(), record, index)) ||
       (projected(multiply.getRhs()) == factor && field(multiply.getLhs(), record, index)));
}
Value scaledSum(Value value, Value left, Value right, unsigned index, const OnlineSummary &plan) {
  auto add = projected(value).getDefiningOp<BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add) return {};
  if (scaled(add.getLhs(), plan.leftScale, left, index) && scaled(add.getRhs(), plan.rightScale, right, index)) return add.getLhs();
  if (scaled(add.getRhs(), plan.leftScale, left, index) && scaled(add.getLhs(), plan.rightScale, right, index)) return add.getRhs();
  return {};
}
}

std::optional<OnlineSummary> matchOnlineSummary(RegionFoldOp fold) {
  if (fold.getIdentityCount() != 1 || fold.getNumResults() != 1) return std::nullopt;
  auto &summary = fold.getSummarize().front();
  auto &merge = fold.getCombine().front();
  OnlineSummary plan{};
  plan.summary = summary.getTerminator()->getOperand(0).getDefiningOp<MakeRecordOp>();
  plan.merged = merge.getTerminator()->getOperand(0).getDefiningOp<MakeRecordOp>();
  if (!plan.summary || !plan.merged || plan.summary.getFields().size() != 4 ||
      plan.merged.getFields().size() != 4 || merge.getNumArguments() != 2) return std::nullopt;
  unsigned validCount = 0, maxCount = 0, massCount = 0, momentCount = 0;
  for (auto [i, value] : llvm::enumerate(plan.summary.getFields())) {
    value = projected(value);
    if (auto reduce = value.getDefiningOp<ReduceOp>()) {
      if (reductionKind(reduce) == BinaryOperator::LogicalOr) { plan.validity = reduce; plan.validField = i; ++validCount; }
      if (reductionKind(reduce) == BinaryOperator::Add) { plan.mass = reduce; plan.massField = i; ++massCount; }
    }
    if (auto select = value.getDefiningOp<SelectOp>()) {
      auto reduce = projected(select.getTrueValue()).getDefiningOp<ReduceOp>();
      auto kind = reductionKind(reduce);
      if ((kind == BinaryOperator::Maximum || kind == BinaryOperator::MaximumNum) && zero(select.getFalseValue())) {
        plan.maximum = reduce; plan.maximumOrEmpty = select; plan.maxField = i; ++maxCount;
      }
      if (auto moment = projected(select.getTrueValue()).getDefiningOp<ContractOp>(); moment && zero(select.getFalseValue())) {
        plan.moment = moment; plan.momentOrEmpty = select; plan.momentField = i; ++momentCount;
      }
    } else if (auto moment = value.getDefiningOp<ContractOp>()) {
      plan.moment = moment; plan.momentField = i; ++momentCount;
    }
  }
  if (validCount != 1 || maxCount != 1 || massCount != 1 || momentCount != 1 ||
      !truth(plan.validity.getInputs()[1], false) || !negativeInfinity(plan.maximum.getInputs()[1]) ||
      !zero(plan.mass.getInputs()[1]) || plan.maximumOrEmpty.getCondition() != plan.validity.getResult(0) ||
      (plan.momentOrEmpty && projected(plan.momentOrEmpty.getCondition()) != plan.validity.getResult(0))) return std::nullopt;
  auto masked = plan.maximum.getInputs()[0].getDefiningOp<SelectOp>();
  plan.probability = plan.mass.getInputs()[0].getDefiningOp<SelectOp>();
  if (!masked || !plan.probability || masked.getCondition() != plan.validity.getInputs()[0] ||
      !negativeInfinity(masked.getFalseValue()) || plan.probability.getCondition() != masked.getCondition() ||
      !zero(plan.probability.getFalseValue())) return std::nullopt;
  plan.exponential = plan.probability.getTrueValue().getDefiningOp<UnaryOp>();
  if (!plan.exponential || (plan.exponential.getOperatorKind() != UnaryOperator::Exp &&
      plan.exponential.getOperatorKind() != UnaryOperator::Exp2)) return std::nullopt;
  auto subtract = plan.exponential.getInput().getDefiningOp<BinaryOp>();
  plan.probabilityCast = plan.moment.getLhs().getDefiningOp<CastOp>();
  if (!subtract || subtract.getOperatorKind() != BinaryOperator::Subtract || subtract.getLhs() != masked.getResult() ||
      projected(subtract.getRhs()) != plan.maximumOrEmpty.getResult() || !plan.probabilityCast ||
      plan.probabilityCast.getInput() != plan.probability.getResult() || plan.probabilityCast.getRounding() ||
      plan.validity.getAxes() != plan.maximum.getAxes() || plan.mass.getAxes() != plan.maximum.getAxes()) return std::nullopt;
  auto pair = plan.moment.getReduce().size() == 1 ? dyn_cast<ArrayAttr>(plan.moment.getReduce()[0]) : ArrayAttr();
  if (!pair || pair.size() != 2 || !plan.moment.getBatch().empty() ||
      cast<IntegerAttr>(pair[0]).getInt() != cast<IntegerAttr>(plan.maximum.getAxes()[0]).getInt()) return std::nullopt;
  auto scoreType = cast<RankedTensorType>(masked.getType());
  auto valuesType = cast<RankedTensorType>(plan.moment.getRhs().getType());
  auto scoreAxes = dyn_cast<TensorShapeAttr>(scoreType.getEncoding());
  auto valueAxes = dyn_cast<TensorShapeAttr>(valuesType.getEncoding());
  if (!scoreAxes || !valueAxes || scoreAxes.getDimensions()[cast<IntegerAttr>(pair[0]).getInt()] !=
      valueAxes.getDimensions()[cast<IntegerAttr>(pair[1]).getInt()]) return std::nullopt;

  Value left = merge.getArgument(0), right = merge.getArgument(1);
  if (!fieldPair(plan.merged.getFields()[plan.validField], BinaryOperator::LogicalOr, left, right, plan.validField) &&
      !fieldPair(plan.merged.getFields()[plan.validField], BinaryOperator::BitwiseOr, left, right, plan.validField)) return std::nullopt;
  plan.combinedMaximum = plan.merged.getFields()[plan.maxField];
  auto finalMax = projected(plan.combinedMaximum).getDefiningOp<SelectOp>();
  if (!finalMax || !field(finalMax.getCondition(), right, plan.validField)) return std::nullopt;
  auto firstMax = projected(finalMax.getFalseValue()).getDefiningOp<SelectOp>();
  auto maximum = projected(finalMax.getTrueValue()).getDefiningOp<BinaryOp>();
  if (!firstMax || !field(firstMax.getCondition(), left, plan.validField) ||
      !field(firstMax.getTrueValue(), left, plan.maxField) || !field(firstMax.getFalseValue(), right, plan.maxField) ||
      !maximum || maximum.getOperatorKind() != reductionKind(plan.maximum) ||
      !((projected(maximum.getLhs()) == firstMax.getResult() && field(maximum.getRhs(), right, plan.maxField)) ||
        (projected(maximum.getRhs()) == firstMax.getResult() && field(maximum.getLhs(), right, plan.maxField)))) return std::nullopt;
  merge.walk([&](SelectOp select) {
    if (scale(select.getResult(), left, plan)) plan.leftScale = select.getResult();
    if (scale(select.getResult(), right, plan)) plan.rightScale = select.getResult();
  });
  if (!plan.leftScale || !plan.rightScale) return std::nullopt;
  plan.leftMassTerm = scaledSum(plan.merged.getFields()[plan.massField], left, right, plan.massField, plan);
  plan.leftMomentTerm = scaledSum(plan.merged.getFields()[plan.momentField], left, right, plan.momentField, plan);
  if (!plan.leftMassTerm || !plan.leftMomentTerm) return std::nullopt;
  return plan;
}

void buildOnlineSummaryUpdate(RegionFoldOp fold, OnlineSummary plan, Region &region) {
  auto &summary = fold.getSummarize().front();
  auto &merge = fold.getCombine().front();
  auto *body = new Block;
  region.push_back(body);
  OpBuilder b(fold.getContext());
  IRMapping mapping;
  for (BlockArgument argument : summary.getArguments())
    mapping.map(argument, body->addArgument(argument.getType(), argument.getLoc()));
  mapping.map(merge.getArgument(0), body->addArgument(merge.getArgument(0).getType(), fold.getLoc()));
  b.setInsertionPointToStart(body);
  SmallVector<Value> right(4);
  std::function<Value(Value)> clone = [&](Value value) -> Value {
    if (Value mapped = mapping.lookupOrNull(value)) return mapped;
    Operation *op = value.getDefiningOp();
    if (!op || (op->getBlock() != &summary && op->getBlock() != &merge)) return value;
    if (auto extract = dyn_cast<ExtractOp>(op); extract && extract.getProduct() == merge.getArgument(1)) {
      Value mapped = right[extract.getField()];
      assert(mapped && "only the available region maximum and validity may be read before normalization");
      mapping.map(value, mapped);
      return mapped;
    }
    for (Value operand : op->getOperands()) mapping.map(operand, clone(operand));
    return b.clone(*op, mapping)->getResult(cast<OpResult>(value).getResultNumber());
  };
  right[plan.validField] = clone(plan.validity.getResult(0));
  right[plan.maxField] = clone(plan.maximumOrEmpty.getResult());
  Value maximum = clone(plan.combinedMaximum);
  Value massSeed = clone(plan.leftMassTerm), momentSeed = clone(plan.leftMomentTerm);
  mapping.map(plan.maximumOrEmpty.getResult(), maximum);
  Value mass = clone(plan.mass.getResult(0));
  Value moment = clone(plan.moment.getResult());
  auto add = [&](Value lhs, Value rhs) -> Value {
    return b.create<BinaryOp>(fold.getLoc(), lhs.getType(), lhs, rhs,
        BinaryOperatorAttr::get(b.getContext(), BinaryOperator::Add), b.getBoolAttr(false), b.getBoolAttr(false));
  };
  Value accumulator = add(momentSeed, moment);
  if (plan.momentOrEmpty)
    accumulator = b.create<SelectOp>(fold.getLoc(), accumulator.getType(),
        clone(plan.momentOrEmpty.getCondition()), accumulator, momentSeed);
  SmallVector<Value> fields(4);
  fields[plan.validField] = clone(plan.merged.getFields()[plan.validField]);
  fields[plan.maxField] = maximum;
  fields[plan.massField] = add(massSeed, mass);
  fields[plan.momentField] = accumulator;
  Value result = b.create<MakeRecordOp>(fold.getLoc(), plan.merged.getResult().getType(), fields);
  b.create<YieldOp>(fold.getLoc(), result);
}
}

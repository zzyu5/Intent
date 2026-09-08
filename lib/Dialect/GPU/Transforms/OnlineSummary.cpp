#include "OnlineSummary.h"

#include "Intent/Dialect/GPU/IR/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

Value scalarValue(Value value) {
  while (value) {
    if (auto splat = value.getDefiningOp<SplatOp>()) {
      value = splat.getValue();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getValue();
      continue;
    }
    if (auto transpose = value.getDefiningOp<TransposeOp>()) {
      value = transpose.getValue();
      continue;
    }
    return value;
  }
  return {};
}

bool isBooleanConstant(Value value, bool expected) {
  auto constant = scalarValue(value).getDefiningOp<arith::ConstantOp>();
  auto attribute = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
  return attribute && attribute.getType().isInteger(1) &&
         attribute.getValue().getBoolValue() == expected;
}

bool isZero(Value value) {
  auto constant = scalarValue(value).getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
    return floating.getValue().isZero();
  return false;
}

bool isNegativeInfinity(Value value) {
  auto constant = scalarValue(value).getDefiningOp<arith::ConstantOp>();
  auto floating = constant ? dyn_cast<FloatAttr>(constant.getValue())
                           : FloatAttr();
  return floating && floating.getValue().isInfinity() &&
         floating.getValue().isNegative();
}

bool sameExecutionSchema(Type lhs, Type rhs) {
  auto left = dyn_cast<FragmentType>(lhs);
  auto right = dyn_cast<FragmentType>(rhs);
  return left && right && left.getShape() == right.getShape() &&
         left.getAxisMaps() == right.getAxisMaps() &&
         left.getValidity() == right.getValidity() &&
         left.getOwner() == right.getOwner();
}

bool isSingleBinaryReduction(ReduceOp reduce, BinaryOperator kind) {
  return reduce && reduce.getNumResults() == 1 &&
         reduce.getAxes().size() == 1 && reduce.getSourceCount() == 1 &&
         reduce.getIdentityCount() == 1 && reduce.getCaptureCount() == 0 &&
         queryBinaryCombineKind(reduce.getCombine()) == kind;
}

bool isProjectedFrom(Value value, Value source,
                     SmallPtrSetImpl<Operation *> &visited) {
  if (value == source)
    return true;
  Operation *producer = value.getDefiningOp();
  if (!producer || !visited.insert(producer).second)
    return false;
  if (auto broadcast = dyn_cast<BroadcastOp>(producer))
    return isProjectedFrom(broadcast.getValue(), source, visited);
  if (auto reshape = dyn_cast<ReshapeOp>(producer))
    return isProjectedFrom(reshape.getValue(), source, visited);
  if (auto select = dyn_cast<SelectOp>(producer))
    return isBooleanConstant(select.getCondition(), true) &&
           isZero(select.getFalseValue()) &&
           isProjectedFrom(select.getTrueValue(), source, visited);
  return false;
}

bool isProjectedFrom(Value value, Value source) {
  SmallPtrSet<Operation *, 8> visited;
  return isProjectedFrom(value, source, visited);
}

Value stripProjection(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
      value = broadcast.getValue();
      continue;
    }
    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      value = reshape.getValue();
      continue;
    }
    break;
  }
  return value;
}

bool isRecordField(Value value, BlockArgument record, unsigned field) {
  value = stripProjection(value);
  auto extract = value.getDefiningOp<ExtractOp>();
  return extract && extract.getRecord() == record && extract.getField() == field;
}

bool isCommutativeRecordFieldBinary(Value value, BinaryOperator kind,
                                    BlockArgument left, BlockArgument right,
                                    unsigned field) {
  auto binary = stripProjection(value).getDefiningOp<BinaryOp>();
  if (!binary || binary.getOperatorKind() != kind)
    return false;
  return (isRecordField(binary.getLhs(), left, field) &&
          isRecordField(binary.getRhs(), right, field)) ||
         (isRecordField(binary.getLhs(), right, field) &&
          isRecordField(binary.getRhs(), left, field));
}

FailureOr<Value> matchScale(Value value, BlockArgument record,
                            unsigned validityField, unsigned maximumField,
                            Value combinedMaximum,
                            UnaryOp summaryExponential) {
  auto scale = stripProjection(value).getDefiningOp<SelectOp>();
  if (!scale ||
      !isRecordField(scale.getCondition(), record, validityField) ||
      !isZero(scale.getFalseValue()))
    return failure();
  auto exponential = scale.getTrueValue().getDefiningOp<UnaryOp>();
  if (!exponential ||
      exponential.getOperatorKind() != summaryExponential.getOperatorKind() ||
      exponential.getApproximate() != summaryExponential.getApproximate() ||
      exponential.getFlushToZero() != summaryExponential.getFlushToZero())
    return failure();
  auto delta = exponential.getInput().getDefiningOp<BinaryOp>();
  if (!delta || delta.getOperatorKind() != BinaryOperator::Subtract ||
      !isProjectedFrom(delta.getRhs(), combinedMaximum))
    return failure();
  auto normalized = stripProjection(delta.getLhs()).getDefiningOp<SelectOp>();
  if (!normalized ||
      !isRecordField(normalized.getCondition(), record, validityField) ||
      !isRecordField(normalized.getTrueValue(), record, maximumField) ||
      !isProjectedFrom(normalized.getFalseValue(), combinedMaximum))
    return failure();
  return scale.getResult();
}

bool matchScaledField(Value value, Value scale, BlockArgument record,
                      unsigned field, Value &term) {
  auto multiply = stripProjection(value).getDefiningOp<BinaryOp>();
  if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
    return false;
  bool matched =
      (isProjectedFrom(multiply.getLhs(), scale) &&
       isRecordField(multiply.getRhs(), record, field)) ||
      (isProjectedFrom(multiply.getRhs(), scale) &&
       isRecordField(multiply.getLhs(), record, field));
  if (matched)
    term = value;
  return matched;
}

bool matchScaledSum(Value value, Value leftScale, BlockArgument left,
                    Value rightScale, BlockArgument right, unsigned field,
                    Value &leftTerm) {
  auto add = stripProjection(value).getDefiningOp<BinaryOp>();
  if (!add || add.getOperatorKind() != BinaryOperator::Add)
    return false;
  Value candidate;
  if (matchScaledField(add.getLhs(), leftScale, left, field, candidate) &&
      matchScaledField(add.getRhs(), rightScale, right, field, leftTerm)) {
    leftTerm = candidate;
    return true;
  }
  if (matchScaledField(add.getRhs(), leftScale, left, field, candidate) &&
      matchScaledField(add.getLhs(), rightScale, right, field, leftTerm)) {
    leftTerm = candidate;
    return true;
  }
  return false;
}

} // namespace

FailureOr<OnlineSummaryStructure>
matchOnlineSummaryStructure(MakeRecordOp record) {
  if (!record || record.getFields().size() != 4)
    return failure();

  SmallVector<std::pair<unsigned, ReduceOp>> validityCandidates;
  SmallVector<std::pair<unsigned, ReduceOp>> massCandidates;
  SmallVector<std::pair<unsigned, SelectOp>> maximumCandidates;
  SmallVector<std::pair<unsigned, ContractOp>> momentCandidates;
  for (auto [index, field] : llvm::enumerate(record.getFields())) {
    field = stripProjection(field);
    if (auto reduce = field.getDefiningOp<ReduceOp>()) {
      auto resultType = reduce.getNumResults() == 1
                            ? dyn_cast<FragmentType>(reduce.getResult(0).getType())
                            : FragmentType();
      if (isSingleBinaryReduction(reduce, BinaryOperator::LogicalOr) &&
          resultType && resultType.getElementType().isInteger(1))
        validityCandidates.emplace_back(index, reduce);
      if (isSingleBinaryReduction(reduce, BinaryOperator::Add))
        massCandidates.emplace_back(index, reduce);
      continue;
    }
    if (auto select = field.getDefiningOp<SelectOp>()) {
      auto reduce = select.getTrueValue().getDefiningOp<ReduceOp>();
      if (isSingleBinaryReduction(reduce, BinaryOperator::Maximum) ||
          isSingleBinaryReduction(reduce, BinaryOperator::MaximumNum))
        maximumCandidates.emplace_back(index, select);
      continue;
    }
    if (auto contract = field.getDefiningOp<ContractOp>())
      momentCandidates.emplace_back(index, contract);
  }
  if (validityCandidates.size() != 1 || massCandidates.size() != 1 ||
      maximumCandidates.size() != 1 || momentCandidates.size() != 1)
    return failure();

  auto [validityField, validity] = validityCandidates.front();
  auto [maximumField, maximumOrEmpty] = maximumCandidates.front();
  auto [massField, mass] = massCandidates.front();
  auto [momentField, moment] = momentCandidates.front();
  auto maximum = maximumOrEmpty.getTrueValue().getDefiningOp<ReduceOp>();
  Value memberValidity = validity.getInputs().front();
  if (maximumOrEmpty.getCondition() != validity.getResult(0) ||
      !isZero(maximumOrEmpty.getFalseValue()) ||
      !isBooleanConstant(validity.getInputs()[1], false) ||
      !isNegativeInfinity(maximum.getInputs()[1]))
    return failure();

  auto maskedScore = maximum.getInputs().front().getDefiningOp<SelectOp>();
  if (!maskedScore || maskedScore.getCondition() != memberValidity ||
      !isNegativeInfinity(maskedScore.getFalseValue()))
    return failure();
  Value score = maskedScore.getTrueValue();

  Value probabilityValue = mass.getInputs().front();
  auto probability = probabilityValue.getDefiningOp<SelectOp>();
  if (!probability || probability.getCondition() != memberValidity ||
      !isZero(probability.getFalseValue()))
    return failure();
  auto exponential = probability.getTrueValue().getDefiningOp<UnaryOp>();
  if (!exponential ||
      (exponential.getOperatorKind() != UnaryOperator::Exp &&
       exponential.getOperatorKind() != UnaryOperator::Exp2))
    return failure();
  auto shift = exponential.getInput().getDefiningOp<BinaryOp>();
  if (!shift || shift.getOperatorKind() != BinaryOperator::Subtract ||
      shift.getLhs() != maskedScore.getResult() ||
      !isProjectedFrom(shift.getRhs(), maximumOrEmpty.getResult()))
    return failure();

  auto probabilityCast = moment.getLhs().getDefiningOp<CastOp>();
  if (!probabilityCast || probabilityCast.getValue() != probabilityValue ||
      !isZero(moment.getAccumulator()) ||
      moment.getLhsReductionAxes().size() != 1 ||
      moment.getRhsReductionAxes().size() != 1)
    return failure();
  Value values = moment.getRhs();

  int64_t rawReductionAxis = validity.getAxes().front();
  int64_t rawValueReductionAxis = moment.getRhsReductionAxes().front();
  if (rawReductionAxis < 0 || rawValueReductionAxis < 0)
    return failure();
  unsigned reductionAxis = static_cast<unsigned>(rawReductionAxis);
  if (validity.getAxes() != maximum.getAxes() ||
      validity.getAxes() != mass.getAxes() ||
      moment.getLhsReductionAxes().front() !=
          static_cast<int64_t>(reductionAxis) ||
      !sameExecutionSchema(validity.getResult(0).getType(),
                           maximum.getResult(0).getType()) ||
      !sameExecutionSchema(maximum.getResult(0).getType(),
                           mass.getResult(0).getType()) ||
      !sameExecutionSchema(memberValidity.getType(), score.getType()) ||
      !sameExecutionSchema(score.getType(), probabilityValue.getType()))
    return failure();
  unsigned valueReductionAxis = static_cast<unsigned>(rawValueReductionAxis);

  auto scoreType = dyn_cast<FragmentType>(score.getType());
  auto validityType = dyn_cast<FragmentType>(memberValidity.getType());
  auto valueType = dyn_cast<FragmentType>(values.getType());
  if (!scoreType || !validityType || !valueType ||
      reductionAxis >= scoreType.getShape().size() ||
      reductionAxis >= validityType.getShape().size() ||
      valueReductionAxis >= valueType.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> scoreMap = queryAxisMap(scoreType, reductionAxis);
  FailureOr<AxisMapAttr> validityMap =
      queryAxisMap(validityType, reductionAxis);
  FailureOr<AxisMapAttr> valueMap =
      queryAxisMap(valueType, valueReductionAxis);
  if (failed(scoreMap) || failed(validityMap) || failed(valueMap))
    return failure();
  PhysicalSourceAxis traversal = sourceAxisIdentity(*scoreMap);
  if (!(sourceAxisIdentity(*validityMap) == traversal) ||
      !(sourceAxisIdentity(*valueMap) == traversal))
    return failure();

  return OnlineSummaryStructure{
      record,          validity,       maximum,
      mass,            moment,         maximumOrEmpty,
      maskedScore,     probability,    probabilityCast,
      exponential,     memberValidity, score,
      values,          validityField,  maximumField,
      massField,       momentField,    reductionAxis,
      valueReductionAxis, traversal};
}

FailureOr<OnlineSummaryMerge>
matchOnlineSummaryMerge(Region &region,
                        const OnlineSummaryStructure &summary) {
  if (region.empty() || !llvm::hasSingleElement(region) ||
      region.front().getNumArguments() != 2)
    return failure();
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  auto record = yield && yield.getValues().size() == 1
                    ? yield.getValues().front().getDefiningOp<MakeRecordOp>()
                    : MakeRecordOp();
  if (!record || record.getFields().size() != 4)
    return failure();
  BlockArgument left = region.front().getArgument(0);
  BlockArgument right = region.front().getArgument(1);
  UnaryOp summaryExponential = summary.exponential;

  bool mergedValidity =
      isCommutativeRecordFieldBinary(
          record.getFields()[summary.validityField],
          BinaryOperator::LogicalOr, left, right, summary.validityField) ||
      isCommutativeRecordFieldBinary(
          record.getFields()[summary.validityField],
          BinaryOperator::BitwiseOr, left, right, summary.validityField);
  if (!mergedValidity)
    return failure();

  Value combinedMaximum = record.getFields()[summary.maximumField];
  auto finalMaximum =
      stripProjection(combinedMaximum).getDefiningOp<SelectOp>();
  if (!finalMaximum ||
      !isRecordField(finalMaximum.getCondition(), right,
                     summary.validityField))
    return failure();
  auto initialMaximum =
      stripProjection(finalMaximum.getFalseValue()).getDefiningOp<SelectOp>();
  if (!initialMaximum ||
      !isRecordField(initialMaximum.getCondition(), left,
                     summary.validityField) ||
      !isRecordField(initialMaximum.getTrueValue(), left,
                     summary.maximumField) ||
      !isRecordField(initialMaximum.getFalseValue(), right,
                     summary.maximumField))
    return failure();
  auto maximumOfBoth =
      stripProjection(finalMaximum.getTrueValue()).getDefiningOp<BinaryOp>();
  ReduceOp summaryMaximum = summary.maximum;
  if (!maximumOfBoth ||
      maximumOfBoth.getOperatorKind() !=
          queryBinaryCombineKind(summaryMaximum.getCombine()) ||
      !((isProjectedFrom(maximumOfBoth.getLhs(), initialMaximum.getResult()) &&
         isRecordField(maximumOfBoth.getRhs(), right,
                       summary.maximumField)) ||
        (isProjectedFrom(maximumOfBoth.getRhs(), initialMaximum.getResult()) &&
         isRecordField(maximumOfBoth.getLhs(), right,
                       summary.maximumField))))
    return failure();

  auto massAdd = stripProjection(record.getFields()[summary.massField])
                     .getDefiningOp<BinaryOp>();
  if (!massAdd || massAdd.getOperatorKind() != BinaryOperator::Add)
    return failure();
  SmallVector<Value> candidates;
  for (Value term : {massAdd.getLhs(), massAdd.getRhs()}) {
    auto multiply = stripProjection(term).getDefiningOp<BinaryOp>();
    if (!multiply || multiply.getOperatorKind() != BinaryOperator::Multiply)
      return failure();
    for (Value operand : {multiply.getLhs(), multiply.getRhs()})
      if (stripProjection(operand).getDefiningOp<SelectOp>())
        candidates.push_back(stripProjection(operand));
  }
  FailureOr<Value> leftScale = failure();
  FailureOr<Value> rightScale = failure();
  for (Value candidate : candidates) {
    if (failed(leftScale))
      leftScale = matchScale(candidate, left, summary.validityField,
                             summary.maximumField, combinedMaximum,
                             summaryExponential);
    if (failed(rightScale))
      rightScale = matchScale(candidate, right, summary.validityField,
                              summary.maximumField, combinedMaximum,
                              summaryExponential);
  }
  if (failed(leftScale) || failed(rightScale))
    return failure();

  Value leftMassTerm;
  Value leftMomentTerm;
  if (!matchScaledSum(record.getFields()[summary.massField], *leftScale, left,
                      *rightScale, right, summary.massField, leftMassTerm) ||
      !matchScaledSum(record.getFields()[summary.momentField], *leftScale, left,
                      *rightScale, right, summary.momentField,
                      leftMomentTerm))
    return failure();

  return OnlineSummaryMerge{record, combinedMaximum, *leftScale, *rightScale,
                            leftMassTerm, leftMomentTerm};
}

ReduceOp cloneReductionWithSource(OpBuilder &builder, Location location,
                                  ReduceOp source, Value value) {
  SmallVector<Value> inputs{value};
  inputs.append(source.getInputs().drop_front(source.getSourceCount()).begin(),
                source.getInputs().drop_front(source.getSourceCount()).end());
  OperationState state(location, ReduceOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(source.getResultTypes());
  state.addAttribute("axes", source->getAttr("axes"));
  state.addAttribute("source_count", source->getAttr("source_count"));
  state.addAttribute("identity_count", source->getAttr("identity_count"));
  state.addAttribute("capture_count", source->getAttr("capture_count"));
  state.addRegion();
  auto result = cast<ReduceOp>(builder.create(state));
  IRMapping mapping;
  source.getCombine().cloneInto(&result.getCombine(), mapping);
  if (Attribute origin = source->getAttr(originAttr))
    result->setAttr(originAttr, origin);
  return result;
}

} // namespace intent::gpu

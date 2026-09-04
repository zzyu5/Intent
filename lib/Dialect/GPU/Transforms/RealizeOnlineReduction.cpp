#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

struct OnlineSummaryPattern {
  MakeRecordOp record;
  ReduceOp validity;
  ReduceOp maximum;
  ReduceOp mass;
  ContractOp moment;
  SelectOp maximumOrEmpty;
  SelectOp maskedScore;
  SelectOp probability;
  UnaryOp exponential;
  Value memberValidity;
  Value score;
  Value values;
  unsigned validityField;
  unsigned maximumField;
  unsigned massField;
  unsigned momentField;
  unsigned reductionAxis;
  unsigned valueReductionAxis;
  PhysicalSourceAxis traversal;
  MakeRangeOp authority;
  SmallVector<MakeRangeOp> ranges;
};

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      StringAttr::get(context, name), ArrayAttr::get(context, {}));
}

FragmentType withElementType(FragmentType type, Type elementType) {
  return FragmentType::get(type.getContext(), elementType, type.getShape(),
                           type.getAxisMaps(), type.getValidity(),
                           type.getOwner());
}

FragmentType replaceExtent(FragmentType type, unsigned axis,
                           PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
  shape[axis] = extent;
  return FragmentType::get(type.getContext(), type.getElementType(),
                           ArrayAttr::get(type.getContext(), shape),
                           type.getAxisMaps(), type.getValidity(),
                           type.getOwner());
}

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, sourceSubregionBoundAttr,
        worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

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
         reduce.getAxes().size() == 1 &&
         reduce.getSourceCount() == 1 && reduce.getIdentityCount() == 1 &&
         reduce.getCaptureCount() == 0 &&
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

FailureOr<OnlineSummaryPattern> matchOnlineSummary(MakeRecordOp record,
                                                   func::FuncOp kernel) {
  if (record->getBlock() != &kernel.getBody().front())
    return failure();
  if (record.getFields().size() != 4)
    return failure();

  SmallVector<std::pair<unsigned, ReduceOp>> validityCandidates;
  SmallVector<std::pair<unsigned, ReduceOp>> massCandidates;
  SmallVector<std::pair<unsigned, SelectOp>> maximumCandidates;
  SmallVector<std::pair<unsigned, ContractOp>> momentCandidates;
  for (auto [index, field] : llvm::enumerate(record.getFields())) {
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
      if (isSingleBinaryReduction(reduce, BinaryOperator::Maximum))
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

  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<MakeRangeOp> ranges;
  for (auto [value, axis] :
       {std::pair<Value, unsigned>{memberValidity, reductionAxis},
        std::pair<Value, unsigned>{score, reductionAxis},
        std::pair<Value, unsigned>{values, valueReductionAxis}}) {
    PhysicalRangeFact fact = analysis.axisRanges(value, axis);
    if (fact.roots.empty())
      return failure();
    for (MakeRangeOp range : fact.roots)
      if (!llvm::is_contained(ranges, range))
        ranges.push_back(range);
  }
  PhysicalLockstepTraversalFact lockstep = analysis.lockstepRanges(ranges);
  if (!lockstep.isExact() || !lockstep.authority ||
      !llvm::all_of(ranges, [&](MakeRangeOp range) {
        return sourceAxisIdentity(range) == traversal &&
               isUnitStepRange(range);
      }))
    return failure();
  for (Value value : {memberValidity, score, values})
    if (!analysis
             .replayability(value, traversal,
                            PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/true)
             .isReplayable())
      return failure();

  return OnlineSummaryPattern{
      record,
      validity,
      maximum,
      mass,
      moment,
      maximumOrEmpty,
      maskedScore,
      probability,
      exponential,
      memberValidity,
      score,
      values,
      validityField,
      maximumField,
      massField,
      momentField,
      reductionAxis,
      valueReductionAxis,
      traversal,
      lockstep.authority,
      std::move(ranges)};
}

ReduceOp cloneReduction(OpBuilder &builder, Location location, ReduceOp source,
                        Value value) {
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

LogicalResult realizeOnlineSummary(OnlineSummaryPattern pattern,
                                   func::FuncOp kernel) {
  auto scoreType = cast<FragmentType>(pattern.score.getType());
  std::string parameterName =
      ("REDUCE_CHUNK_" + Twine(pattern.traversal.sourceId) + "_A" +
       Twine(pattern.traversal.sourceAxis) +
       (pattern.traversal.derived ? "_DERIVED" : ""))
          .str();
  SmallVector<int64_t> candidates{8, 16, 32, 64, 128,
                                  256, 512, 1024, 2048, 4096};
  ParameterOp chunk = getOrCreatePhysicalParameter(
      kernel, parameterName, ParameterRole::Reduction,
      ParameterCategory::Reduction,
      scoreType.getElementType().getIntOrFloatBitWidth(), candidates);
  if (!chunk)
    return pattern.record.emitOpError(
        "online reduction has no physical traversal parameter");
  if (FailureOr<int64_t> dimension = queryRangeDimension(pattern.authority);
      succeeded(dimension))
    chunk->setAttr(dimensionAttr,
                   IntegerAttr::get(IntegerType::get(chunk.getContext(), 64),
                                    *dimension));
  PhysicalExprAttr chunkExtent = parameterExpression(
      pattern.record.getContext(), chunk.getParameter().getName().getValue());

  OpBuilder builder(pattern.record);
  Location location = pattern.record.getLoc();
  Value validIdentity = pattern.validity.getInputs()[1];
  Value maximumIdentity = builder.create<SplatOp>(
      location, pattern.maximum.getResult(0).getType(),
      scalarValue(pattern.maximumOrEmpty.getFalseValue()));
  Value maximumReductionIdentity = pattern.maximum.getInputs()[1];
  Value massIdentity = pattern.mass.getInputs()[1];
  Value momentIdentity = pattern.moment.getAccumulator();
  SmallVector<Value> initials{validIdentity, maximumReductionIdentity,
                              massIdentity, momentIdentity};
  bool bodyFailed = false;
  std::string bodyFailure;
  auto loop = builder.create<scf::ForOp>(
      location, pattern.authority.getStart(), pattern.authority.getLogicalStop(),
      chunk.getResult(), initials,
      [&](OpBuilder &nested, Location nestedLocation, Value chunkStart,
          ValueRange carries) {
        IRMapping mapping;
        ReplayMaterializationOptions options;
        options.traversalRanges = pattern.ranges;
        options.materializeZeroFill = true;
        for (MakeRangeOp range : pattern.ranges) {
          auto original = cast<FragmentType>(range.getResult().getType());
          FragmentType blocked = replaceExtent(original, 0, chunkExtent);
          auto coordinate = nested.create<MakeRangeOp>(
              nestedLocation, blocked, chunkStart, chunk.getResult(),
              range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
              range.getSourceId(), range.getSourceAxis(), range.getDerived());
          inheritRangeAuthority(coordinate, range);
          mapping.map(range.getResult(), coordinate.getResult());
        }
        Value authority = mapping.lookupOrNull(pattern.authority.getResult());
        if (!authority) {
          bodyFailed = true;
          bodyFailure = "lockstep traversal authority was not rematerialized";
          return;
        }
        auto authorityType = cast<FragmentType>(authority.getType());
        Value end = nested.create<BroadcastOp>(
            nestedLocation, authorityType, pattern.authority.getLogicalStop());
        Value tail = nested.create<CompareOp>(
            nestedLocation,
            withElementType(authorityType, nested.getI1Type()), authority, end,
            ComparePredicate::Lt);
        options.segmentTail = tail;

        auto replay = [&](Value value) -> FailureOr<Value> {
          return materializeReplayedValue(
              nested, nestedLocation, value, pattern.traversal, chunkExtent,
              mapping, options);
        };
        FailureOr<Value> replayedValidity = replay(pattern.memberValidity);
        FailureOr<Value> replayedScore = replay(pattern.score);
        FailureOr<Value> replayedValues = replay(pattern.values);
        if (failed(replayedValidity) || failed(replayedScore) ||
            failed(replayedValues)) {
          bodyFailed = true;
          bodyFailure = "typed member graph could not be replayed once";
          return;
        }
        auto blockedValidityType =
            cast<FragmentType>((*replayedValidity).getType());
        FailureOr<Value> projectedTail = projectPredicateToFragment(
            nested, nestedLocation, tail, blockedValidityType,
            pattern.traversal);
        if (failed(projectedTail)) {
          bodyFailed = true;
          bodyFailure = "traversal tail has no member-predicate projection";
          return;
        }
        Value blockedValidity = nested.create<BinaryOp>(
            nestedLocation, blockedValidityType, *replayedValidity,
            *projectedTail, BinaryOperator::LogicalAnd);
        mapping.map(pattern.memberValidity, blockedValidity);

        auto blockedScoreType = cast<FragmentType>((*replayedScore).getType());
        Value negativeInfinity = nested.create<SplatOp>(
            nestedLocation, blockedScoreType,
            scalarValue(pattern.maskedScore.getFalseValue()));
        Value replayedMasked = nested.create<SelectOp>(
            nestedLocation, blockedScoreType, blockedValidity, *replayedScore,
            negativeInfinity);
        ReduceOp chunkValidity = cloneReduction(
            nested, nestedLocation, pattern.validity, blockedValidity);
        ReduceOp chunkMaximum = cloneReduction(
            nested, nestedLocation, pattern.maximum, replayedMasked);
        Value currentMaximum = nested.create<SelectOp>(
            nestedLocation, pattern.maximumOrEmpty.getResult().getType(),
            chunkValidity.getResult(0), chunkMaximum.getResult(0),
            maximumIdentity);
        Value broadcastMaximum = nested.create<BroadcastOp>(
            nestedLocation, blockedScoreType, currentMaximum);
        Value shiftedScore = nested.create<BinaryOp>(
            nestedLocation, blockedScoreType, replayedMasked,
            broadcastMaximum, BinaryOperator::Subtract);
        Value unmaskedProbability = nested.create<UnaryOp>(
            nestedLocation, blockedScoreType, shiftedScore,
            pattern.exponential.getOperatorKind());
        Value probabilityZero = nested.create<SplatOp>(
            nestedLocation, blockedScoreType,
            scalarValue(pattern.probability.getFalseValue()));
        Value replayedProbability = nested.create<SelectOp>(
            nestedLocation, blockedScoreType, blockedValidity,
            unmaskedProbability, probabilityZero);
        ReduceOp chunkMass = cloneReduction(
            nested, nestedLocation, pattern.mass, replayedProbability);
        auto originalWeightType =
            cast<FragmentType>(pattern.moment.getLhs().getType());
        FragmentType blockedWeightType = withElementType(
            blockedScoreType, originalWeightType.getElementType());
        Value replayedProbabilityCast = nested.create<CastOp>(
            nestedLocation, blockedWeightType, replayedProbability);
        Value chunkMoment = nested.create<ContractOp>(
            nestedLocation, pattern.moment.getResult().getType(),
            replayedProbabilityCast, *replayedValues, momentIdentity,
            pattern.moment.getLhsReductionAxes(),
            pattern.moment.getRhsReductionAxes(),
            pattern.moment.getLhsBatchAxes(),
            pattern.moment.getRhsBatchAxes());
        if (Attribute origin = pattern.moment->getAttr(originAttr))
          chunkMoment.getDefiningOp()->setAttr(originAttr, origin);

        Value bothValid = nested.create<BinaryOp>(
            nestedLocation, carries[0].getType(), carries[0],
            chunkValidity.getResult(0), BinaryOperator::LogicalOr);
        Value maximumOfBoth = nested.create<BinaryOp>(
            nestedLocation, carries[1].getType(), carries[1], currentMaximum,
            BinaryOperator::Maximum);
        Value maximumWithRight = nested.create<SelectOp>(
            nestedLocation, carries[1].getType(), chunkValidity.getResult(0),
            maximumOfBoth, carries[1]);
        Value combinedMaximum = nested.create<SelectOp>(
            nestedLocation, carries[1].getType(), carries[0], maximumWithRight,
            currentMaximum);

        auto scale = [&](Value present, Value previousMaximum) {
          Value delta = nested.create<BinaryOp>(
              nestedLocation, carries[1].getType(), previousMaximum,
              combinedMaximum, BinaryOperator::Subtract);
          Value exponential = nested.create<UnaryOp>(
              nestedLocation, carries[1].getType(), delta,
              pattern.exponential.getOperatorKind());
          return Value(nested.create<SelectOp>(
              nestedLocation, carries[1].getType(), present, exponential,
              massIdentity));
        };
        Value carryScale = scale(carries[0], carries[1]);
        Value chunkScale =
            scale(chunkValidity.getResult(0), currentMaximum);
        Value scaledCarryMass = nested.create<BinaryOp>(
            nestedLocation, carries[2].getType(), carries[2], carryScale,
            BinaryOperator::Multiply);
        Value scaledChunkMass = nested.create<BinaryOp>(
            nestedLocation, carries[2].getType(), chunkMass.getResult(0),
            chunkScale, BinaryOperator::Multiply);
        Value combinedMass = nested.create<BinaryOp>(
            nestedLocation, carries[2].getType(), scaledCarryMass,
            scaledChunkMass, BinaryOperator::Add);

        auto momentType = cast<FragmentType>(carries[3].getType());
        Value carryMomentScale = nested.create<BroadcastOp>(
            nestedLocation, momentType, carryScale);
        Value chunkMomentScale = nested.create<BroadcastOp>(
            nestedLocation, momentType, chunkScale);
        Value scaledCarryMoment = nested.create<BinaryOp>(
            nestedLocation, momentType, carries[3], carryMomentScale,
            BinaryOperator::Multiply);
        Value scaledChunkMoment = nested.create<BinaryOp>(
            nestedLocation, momentType, chunkMoment, chunkMomentScale,
            BinaryOperator::Multiply);
        Value combinedMoment = nested.create<BinaryOp>(
            nestedLocation, momentType, scaledCarryMoment, scaledChunkMoment,
            BinaryOperator::Add);
        nested.create<scf::YieldOp>(
            nestedLocation,
            ValueRange{bothValid, combinedMaximum, combinedMass,
                       combinedMoment});
      });
  if (bodyFailed) {
    loop.erase();
    return pattern.record.emitOpError(
               "online reduction could not form one physical traversal: ")
           << bodyFailure;
  }
  if (Attribute origin = pattern.record->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  loop->setAttr(
      reductionSourcesAttr,
      builder.getArrayAttr({PhysicalSourceAttr::get(
          pattern.record.getContext(), pattern.traversal.sourceId,
          pattern.traversal.sourceAxis, pattern.traversal.derived)}));
  Value finalMaximum = builder.create<SelectOp>(
      location, pattern.maximumOrEmpty.getResult().getType(),
      loop.getResult(0), loop.getResult(1), maximumIdentity);
  pattern.record->setOperand(pattern.validityField, loop.getResult(0));
  pattern.record->setOperand(pattern.maximumField, finalMaximum);
  pattern.record->setOperand(pattern.massField, loop.getResult(2));
  pattern.record->setOperand(pattern.momentField, loop.getResult(3));
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace

LogicalResult realizeOnlineReductions(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<MakeRecordOp> records;
  kernel.walk([&](MakeRecordOp record) { records.push_back(record); });
  for (MakeRecordOp record : records) {
    if (!record->getBlock())
      continue;
    FailureOr<OnlineSummaryPattern> pattern =
        matchOnlineSummary(record, kernel);
    if (failed(pattern))
      continue;
    if (failed(realizeOnlineSummary(*pattern, kernel)))
      return failure();
  }
  return success();
}

} // namespace intent::gpu

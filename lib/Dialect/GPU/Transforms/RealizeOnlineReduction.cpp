#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "OnlineSummary.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::gpu {
namespace {

struct OnlineSummaryPattern : OnlineSummaryStructure {
  OnlineSummaryPattern(OnlineSummaryStructure structure, MakeRangeOp authority,
                       SmallVector<MakeRangeOp> ranges)
      : OnlineSummaryStructure(std::move(structure)), authority(authority),
        ranges(std::move(ranges)) {}

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

FailureOr<OnlineSummaryPattern> matchOnlineSummary(MakeRecordOp record,
                                                   func::FuncOp kernel) {
  if (record->getBlock() != &kernel.getBody().front())
    return failure();
  FailureOr<OnlineSummaryStructure> summary =
      matchOnlineSummaryStructure(record);
  if (failed(summary))
    return failure();

  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<MakeRangeOp> ranges;
  for (auto [value, axis] :
       {std::pair<Value, unsigned>{summary->memberValidity,
                                   summary->reductionAxis},
        std::pair<Value, unsigned>{summary->score, summary->reductionAxis},
        std::pair<Value, unsigned>{summary->values,
                                   summary->valueReductionAxis}}) {
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
        return sourceAxisIdentity(range) == summary->traversal &&
               isUnitStepRange(range);
      }))
    return failure();
  for (Value value :
       {summary->memberValidity, summary->score, summary->values})
    if (!analysis
             .replayability(value, summary->traversal,
                            PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/true)
             .isReplayable())
      return failure();

  return OnlineSummaryPattern(std::move(*summary), lockstep.authority,
                              std::move(ranges));
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
        ReduceOp chunkValidity = cloneReductionWithSource(
            nested, nestedLocation, pattern.validity, blockedValidity);
        ReduceOp chunkMaximum = cloneReductionWithSource(
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
        ReduceOp chunkMass = cloneReductionWithSource(
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
            *queryBinaryCombineKind(pattern.maximum.getCombine()));
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

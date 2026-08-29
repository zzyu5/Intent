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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <tuple>

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {},
                            ArrayRef<Attribute> operands = {}) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(kind), value,
      StringAttr::get(context, symbol), ArrayAttr::get(context, operands));
}

PhysicalExprAttr nextPowerOfTwo(PhysicalExprAttr source) {
  if (source.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant)) {
    uint64_t value = std::max<int64_t>(source.getValue(), 1);
    uint64_t result = 1;
    while (result < value)
      result <<= 1;
    return expression(source.getContext(), PhysicalExprKind::Constant, result);
  }
  return expression(source.getContext(), PhysicalExprKind::NextPowerOfTwo, 0,
                    {}, {source});
}

bool isCompileTimeExtent(PhysicalExprAttr expression) {
  auto kind = static_cast<PhysicalExprKind>(expression.getKind());
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExtent(cast<PhysicalExprAttr>(operand));
  });
}

bool isCompileTimeValue(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() ||
         value.getDefiningOp<ParameterOp>() ||
         value.getDefiningOp<PhysicalExprOp>();
}

bool requiresPhysicalRealization(ReduceOp reduce) {
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    if (llvm::any_of(fragment.getShape(), [](Attribute extent) {
          return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
        }))
      return true;
    for (int64_t axis : reduce.getAxes()) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
        continue;
      auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
      if (extent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
        return true;
    }
  }
  return false;
}

FailureOr<AxisMapAttr> axisMap(FragmentType fragment, unsigned axis) {
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == axis)
      return mapping;
  }
  return failure();
}

FailureOr<int64_t> rangeDimension(MakeRangeOp range) {
  return querySourceDimension(
      range.getResult().getType(),
      PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis()});
}

FailureOr<unsigned> coordinateForSource(ValueRange coordinates,
                                        uint64_t sourceId) {
  return queryCoordinateIndex(coordinates, sourceId);
}

MakeRangeOp sourceRange(Value value, std::optional<uint64_t> sourceId = std::nullopt) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return {};
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact;
  if (sourceId) {
    PhysicalAxisProjection source =
        queryUniqueSourceAxis(value.getType(), *sourceId);
    if (!source.isExact())
      return {};
    fact = analysis.sourceRanges(value, source.source);
  } else {
    fact = analysis.sourceRanges(value);
  }
  return fact.isUnique() ? fact.roots.front() : MakeRangeOp();
}

FailureOr<Value> scalarSource(Value value) {
  if (!value)
    return failure();
  if (!isa<FragmentType>(value.getType()))
    return value;
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    if (!isa<FragmentType>(broadcast.getValue().getType()))
      return broadcast.getValue();
  if (auto splat = value.getDefiningOp<SplatOp>())
    return splat.getValue();
  return failure();
}

bool sameScalarValue(Value lhs, Value rhs) {
  FailureOr<Value> left = scalarSource(lhs);
  FailureOr<Value> right = scalarSource(rhs);
  if (failed(left) || failed(right))
    return false;
  if (*left == *right)
    return true;
  auto leftConstant = (*left).getDefiningOp<arith::ConstantOp>();
  auto rightConstant = (*right).getDefiningOp<arith::ConstantOp>();
  return leftConstant && rightConstant &&
         leftConstant.getValue() == rightConstant.getValue();
}

bool sameScalarExpression(Value lhs, Value rhs, unsigned depth = 0) {
  if (lhs == rhs)
    return true;
  if (depth >= 32 || lhs.getType() != rhs.getType())
    return false;
  FailureOr<Value> left = scalarSource(lhs);
  FailureOr<Value> right = scalarSource(rhs);
  if (succeeded(left) && succeeded(right) && (*left != lhs || *right != rhs))
    return sameScalarExpression(*left, *right, depth + 1);
  auto leftConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rightConstant = rhs.getDefiningOp<arith::ConstantOp>();
  if (leftConstant || rightConstant)
    return leftConstant && rightConstant &&
           leftConstant.getValue() == rightConstant.getValue();
  auto leftBinary = lhs.getDefiningOp<BinaryOp>();
  auto rightBinary = rhs.getDefiningOp<BinaryOp>();
  if (leftBinary || rightBinary)
    return leftBinary && rightBinary &&
           leftBinary.getOperatorKind() == rightBinary.getOperatorKind() &&
           sameScalarExpression(leftBinary.getLhs(), rightBinary.getLhs(),
                                depth + 1) &&
           sameScalarExpression(leftBinary.getRhs(), rightBinary.getRhs(),
                                depth + 1);
  auto leftCast = lhs.getDefiningOp<CastOp>();
  auto rightCast = rhs.getDefiningOp<CastOp>();
  if (leftCast || rightCast)
    return leftCast && rightCast &&
           sameScalarExpression(leftCast.getValue(), rightCast.getValue(),
                                depth + 1);
  return false;
}

FragmentType replaceExtent(FragmentType source, unsigned axis,
                           PhysicalExprAttr extent, Type element = {}) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[axis] = extent;
  return FragmentType::get(source.getContext(),
                           element ? element : source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

void collectRangesAndLoads(Value value, PhysicalExprAttr logicalExtent,
                           SmallVectorImpl<MakeRangeOp> &ranges,
                           SmallVectorImpl<LoadOp> &loads,
                           llvm::SmallPtrSetImpl<Operation *> &visited) {
  (void)visited;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return;
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value);
  for (MakeRangeOp range : fact.roots) {
    auto fragment = range.getResult().getType();
    auto constant = range.getExtent().getDefiningOp<arith::ConstantOp>();
    auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                            : IntegerAttr();
    if (fragment.getShape().size() == 1 &&
        (fragment.getShape()[0] == logicalExtent ||
         (logicalExtent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          integer && integer.getInt() == logicalExtent.getValue())) &&
        !llvm::is_contained(ranges, range))
      ranges.push_back(range);
  }
  for (Operation *access : fact.accesses)
    if (auto load = dyn_cast<LoadOp>(access);
        load && !llvm::is_contained(loads, load))
      loads.push_back(load);
}

FailureOr<Value> predicateForReductionSource(OpBuilder &builder,
                                             Location location, Value predicate,
                                             FragmentType source,
                                             unsigned reductionAxis) {
  if (reductionAxis >= source.getAxisMaps().size())
    return failure();
  auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[reductionAxis]);
  return projectPredicateToFragment(
      builder, location, predicate, source,
      PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()});
}

void retargetHelperSourceExtent(Region &region, PhysicalSourceAxis source,
                                PhysicalExprAttr logicalExtent,
                                PhysicalExprAttr physicalExtent) {
  auto retarget = [&](Value value) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return;
    PhysicalAxisProjection projected = queryFragmentAxis(fragment, source);
    if (!projected.isExact() ||
        fragment.getShape()[projected.fragmentAxis] != logicalExtent)
      return;
    value.setType(
        replaceExtent(fragment, projected.fragmentAxis, physicalExtent));
  };
  for (Block &block : region) {
    for (BlockArgument argument : block.getArguments())
      retarget(argument);
    block.walk([&](Operation *operation) {
      for (Value result : operation->getResults())
        retarget(result);
    });
  }
}

FailureOr<Value> clonePaddedProducer(
    OpBuilder &builder, Location location, Value value,
    PhysicalSourceAxis reductionSource,
    PhysicalExprAttr logicalExtent, PhysicalExprAttr physicalExtent,
    Value physicalExtentValue, IRMapping &mapping,
    SmallVectorImpl<Value> &tailPredicates) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  PhysicalAxisProjection projected =
      queryFragmentAxis(fragment, reductionSource);
  if (!projected.isExact())
    return value;
  unsigned reductionAxis = projected.fragmentAxis;
  if (fragment.getShape()[reductionAxis] != logicalExtent)
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getNumResults() != 1 ||
      (producer->getNumRegions() != 0 && !isa<ReduceOp>(producer)))
    return producer
               ? (producer->emitOpError(
                      "padded reduction producer is not a replayable single-result operation"),
                  FailureOr<Value>(failure()))
               : FailureOr<Value>(failure());

  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
    auto padded = builder.create<MakeRangeOp>(
        location, paddedType, range.getStart(), physicalExtentValue,
        range.getStep(), range.getSourceId(), range.getSourceAxis());
    if (Attribute origin = range->getAttr(originAttr))
      padded->setAttr(originAttr, origin);
    Value logicalLength = builder.create<arith::ConstantIndexOp>(
        location, logicalExtent.getValue());
    Value distance = builder.create<BinaryOp>(
        location, builder.getIndexType(), logicalLength, range.getStep(),
        BinaryOperator::Multiply);
    Value stop = builder.create<BinaryOp>(location, builder.getIndexType(),
                                          range.getStart(), distance,
                                          BinaryOperator::Add);
    Value stopFragment =
        builder.create<BroadcastOp>(location, paddedType, stop);
    auto predicateType = FragmentType::get(
        builder.getContext(), builder.getI1Type(), paddedType.getShape(),
        paddedType.getAxisMaps(), paddedType.getValidity(),
        paddedType.getOwner());
    Value predicate = builder.create<CompareOp>(
        location, predicateType, padded.getResult(), stopFragment,
        ComparePredicate::Lt);
    tailPredicates.push_back(predicate);
    mapping.map(value, padded.getResult());
    return padded.getResult();
  }

  if (auto broadcast = dyn_cast<BroadcastOp>(producer)) {
    auto input = dyn_cast<FragmentType>(broadcast.getValue().getType());
    if (input && input.getShape().size() <= fragment.getShape().size()) {
      unsigned offset = fragment.getShape().size() - input.getShape().size();
      if (reductionAxis >= offset) {
        unsigned inputAxis = reductionAxis - offset;
        if (input.getShape()[inputAxis] == logicalExtent) {
          auto inputMapping =
              cast<AxisMapAttr>(input.getAxisMaps()[inputAxis]);
          FailureOr<Value> replayed = clonePaddedProducer(
              builder, location, broadcast.getValue(),
              PhysicalSourceAxis{inputMapping.getSourceId(),
                                 inputMapping.getSourceAxis()},
              logicalExtent, physicalExtent, physicalExtentValue, mapping,
              tailPredicates);
          if (failed(replayed))
            return failure();
          auto paddedType =
              replaceExtent(fragment, reductionAxis, physicalExtent);
          auto padded =
              builder.create<BroadcastOp>(location, paddedType, *replayed);
          if (Attribute origin = broadcast->getAttr(originAttr))
            padded->setAttr(originAttr, origin);
          mapping.map(value, padded.getResult());
          return padded.getResult();
        }
      }
    }
  }

  if (auto load = dyn_cast<LoadOp>(producer)) {
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, coordinate, reductionSource, logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      coordinates.push_back(*replayed);
    }
    Value valid;
    if (load.getValid()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, load.getValid(), reductionSource, logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      valid = *replayed;
    }
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, load.getFill(), reductionSource, logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      fill = *replayed;
    }
    auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
    if (tailPredicates.empty())
      return load.emitOpError(
                 "padded reduction load has no logical tail predicate"),
             failure();
    Value tail;
    for (Value base : tailPredicates) {
      FailureOr<Value> current = predicateForReductionSource(
          builder, location, base, paddedType, reductionAxis);
      if (failed(current))
        return failure();
      tail = tail ? Value(builder.create<BinaryOp>(
                        location, current->getType(), tail, *current,
                        BinaryOperator::LogicalAnd))
                  : *current;
    }
    auto predicateType = cast<FragmentType>(tail.getType());
    if (valid) {
      Type element = valid.getType();
      if (auto validFragment = dyn_cast<FragmentType>(element))
        element = validFragment.getElementType();
      if (!element.isInteger(1))
        return load.emitOpError(
                   "padded reduction replay produced non-predicate validity"),
               failure();
      if (valid.getType() != predicateType)
        valid = builder.create<BroadcastOp>(location, predicateType, valid);
      valid = builder.create<BinaryOp>(location, predicateType, valid, tail,
                                       BinaryOperator::LogicalAnd);
    } else {
      valid = tail;
    }
    if (!fill) {
      FailureOr<Value> zero =
          materializeZeroFragment(builder, location, paddedType);
      if (failed(zero))
        return failure();
      fill = *zero;
    } else if (fill.getType() != paddedType)
      fill = builder.create<BroadcastOp>(location, paddedType, fill);
    auto padded = builder.create<LoadOp>(
        location, paddedType, load.getResource(), coordinates, valid, fill,
        load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      padded->setAttr(originAttr, origin);
    mapping.map(value, padded.getResult());
    return padded.getResult();
  }

  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed = clonePaddedProducer(
        builder, location, operand, reductionSource, logicalExtent,
        physicalExtent,
        physicalExtentValue, mapping, tailPredicates);
    if (failed(replayed))
      return failure();
    if (!mapping.lookupOrNull(operand) && *replayed != operand)
      mapping.map(operand, *replayed);
  }
  Operation *clone = builder.clone(*producer, mapping);
  if (auto clonedReduce = dyn_cast<ReduceOp>(clone))
    retargetHelperSourceExtent(clonedReduce.getCombine(), reductionSource,
                               logicalExtent, physicalExtent);
  auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
  clone->getResult(0).setType(paddedType);
  mapping.map(value, clone->getResult(0));
  return clone->getResult(0);
}

FailureOr<unsigned> axisForSource(FragmentType fragment, uint64_t sourceId) {
  return queryFragmentAxis(fragment, sourceId);
}

bool isReplayableWithoutLoad(Value value, uint64_t sourceId,
                             llvm::SmallPtrSetImpl<Operation *> &visited) {
  (void)visited;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return false;
  PhysicalAxisProjection source = queryUniqueSourceAxis(value.getType(), sourceId);
  if (!source.isExact())
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  return analysis
      .replayability(value, source.source, PhysicalReplayScope::ValueGraph,
                     /*allowAccesses=*/false)
      .isReplayable();
}

struct SourcePlan {
  Value source;
  uint64_t sourceId;
  unsigned reductionAxis;
  MakeRangeOp reductionRange;
  SmallVector<LoadOp> roots;
  SmallVector<MakeRangeOp> ranges;
};

struct RootAccess {
  LoadOp load;
  MakeRangeOp range;
  unsigned coordinateIndex;
  unsigned fragmentAxis;
};

bool isUnitStep(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value() == 1;
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    return range && bound.getBound() == 2 && isUnitStep(range.getStep());
  }
  return false;
}

bool sameLogicalSourceRange(MakeRangeOp lhs, MakeRangeOp rhs);

FailureOr<MakeRangeOp> exactLogicalRange(const PhysicalRangeFact &fact) {
  if (fact.state == PhysicalFactState::Unknown || fact.roots.empty())
    return failure();
  MakeRangeOp first = fact.roots.front();
  return llvm::all_of(fact.roots, [&](MakeRangeOp range) {
           return sameLogicalSourceRange(first, range);
         })
             ? FailureOr<MakeRangeOp>(first)
             : FailureOr<MakeRangeOp>(failure());
}

FailureOr<RootAccess> analyzeRoot(LoadOp load, MakeRangeOp reductionRange) {
  auto fragment = dyn_cast<FragmentType>(load.getResult().getType());
  if (!fragment)
    return load.emitOpError("reduction producer root is not a fragment load");
  auto kernel = load->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  std::optional<unsigned> fragmentAxis;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(load.getResult(), axis);
    if (!fact.isUnique() || fact.roots.front() != reductionRange)
      continue;
    if (fragmentAxis)
      return load.emitOpError(
          "reduction range drives multiple root fragment axes");
    fragmentAxis = axis;
  }
  if (!fragmentAxis) {
    FailureOr<unsigned> legacy =
        axisForSource(fragment, reductionRange.getSourceId());
    if (succeeded(legacy))
      fragmentAxis = *legacy;
  }
  if (!fragmentAxis)
    return load.emitOpError(
        "reduction source provenance is absent from its root load");
  std::optional<unsigned> coordinate;
  for (auto [index, value] : llvm::enumerate(load.getCoordinates())) {
    auto type = dyn_cast<FragmentType>(value.getType());
    if (!type || *fragmentAxis >= type.getShape().size())
      continue;
    PhysicalRangeFact fact = analysis.axisRanges(value, *fragmentAxis);
    if (!fact.isUnique() || fact.roots.front() != reductionRange)
      continue;
    if (coordinate)
      return load.emitOpError(
          "reduction range drives multiple load coordinates");
    coordinate = index;
  }
  if (!coordinate) {
    FailureOr<unsigned> legacy =
        coordinateForSource(load.getCoordinates(), reductionRange.getSourceId());
    if (succeeded(legacy)) {
      MakeRangeOp range =
          sourceRange(load.getCoordinates()[*legacy],
                      reductionRange.getSourceId());
      if (range && sameLogicalSourceRange(range, reductionRange))
        coordinate = *legacy;
    }
  }
  if (!coordinate)
    return load.emitOpError(
        "reduction source provenance is absent from load coordinates");
  if (!isUnitStep(reductionRange.getStep()))
    return load.emitOpError("reduction load range is not unit-step");
  return RootAccess{load, reductionRange, *coordinate, *fragmentAxis};
}

FailureOr<SourcePlan> analyzeSource(Value source, unsigned reductionAxis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || reductionAxis >= fragment.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> mapping = axisMap(fragment, reductionAxis);
  if (failed(mapping))
    return failure();
  SourcePlan plan{source, mapping->getSourceId(), reductionAxis, {}, {}, {}};
  auto kernel = source.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalSourceAxis physicalSource{mapping->getSourceId(),
                                    mapping->getSourceAxis()};
  const bool repeatedOccurrence =
      queryFragmentAxes(fragment, physicalSource).size() > 1;
  PhysicalRangeFact fact =
      repeatedOccurrence
          ? analysis.axisRanges(source, reductionAxis)
          : analysis.sourceRanges(source, physicalSource);
  FailureOr<MakeRangeOp> authority = exactLogicalRange(fact);
  if (failed(authority))
    return failure();
  plan.reductionRange = *authority;
  for (MakeRangeOp range : fact.roots)
    if (sameLogicalSourceRange(plan.reductionRange, range) &&
        !llvm::is_contained(plan.ranges, range))
      plan.ranges.push_back(range);
  for (Operation *access : fact.accesses)
    if (auto load = dyn_cast<LoadOp>(access);
        load && !llvm::is_contained(plan.roots, load))
      plan.roots.push_back(load);
  if (plan.roots.empty() && plan.ranges.empty())
    return failure();
  return plan;
}

FailureOr<Value> replayValue(OpBuilder &builder, Location location, Value value,
                             uint64_t sourceId,
                             PhysicalExprAttr blockedExtent,
                             IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  if (auto extract = value.getDefiningOp<ExtractOp>()) {
    if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
      uint64_t field = extract.getField();
      if (field >= record.getFields().size())
        return failure();
      FailureOr<Value> replayed = replayValue(
          builder, location, record.getFields()[field], sourceId,
          blockedExtent, mapping);
      if (succeeded(replayed) && !mapping.lookupOrNull(value))
        mapping.map(value, *replayed);
      return replayed;
    }
  }
  auto fragment = dyn_cast<FragmentType>(value.getType());
  FailureOr<unsigned> axis = fragment ? axisForSource(fragment, sourceId)
                                      : FailureOr<unsigned>(failure());
  if (!fragment || failed(axis))
    return value;
  Operation *producer = value.getDefiningOp();
  if (auto load = dyn_cast_or_null<LoadOp>(producer)) {
    SmallVector<Value> coordinates;
    coordinates.reserve(load.getCoordinates().size());
    for (Value coordinate : load.getCoordinates()) {
      FailureOr<Value> replayed = replayValue(
          builder, location, coordinate, sourceId, blockedExtent, mapping);
      if (failed(replayed))
        return failure();
      coordinates.push_back(*replayed);
    }
    Value valid;
    if (load.getValid()) {
      FailureOr<Value> replayed = replayValue(
          builder, location, load.getValid(), sourceId, blockedExtent, mapping);
      if (failed(replayed))
        return failure();
      valid = *replayed;
    }
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> replayed = replayValue(
          builder, location, load.getFill(), sourceId, blockedExtent, mapping);
      if (failed(replayed))
        return failure();
      fill = *replayed;
    }
    auto blockedType = replaceExtent(fragment, *axis, blockedExtent);
    auto replayed = builder.create<LoadOp>(
        location, blockedType, load.getResource(), coordinates, valid, fill,
        load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replayed->setAttr(originAttr, origin);
    mapping.map(value, replayed.getResult());
    return replayed.getResult();
  }
  if (!isPhysicalReplayNode(producer, PhysicalReplayScope::ValueGraph,
                            /*allowAccesses=*/false))
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed =
        replayValue(builder, location, operand, sourceId, blockedExtent, mapping);
    if (failed(replayed))
      return failure();
    if (!mapping.lookupOrNull(operand))
      mapping.map(operand, *replayed);
  }
  Operation *clone = builder.clone(*producer, mapping);
  if (auto clonedReduce = dyn_cast<ReduceOp>(clone))
    retargetHelperSourceExtent(
        clonedReduce.getCombine(),
        PhysicalSourceAxis{sourceId,
                           cast<AxisMapAttr>(fragment.getAxisMaps()[*axis])
                               .getSourceAxis()},
        cast<PhysicalExprAttr>(fragment.getShape()[*axis]), blockedExtent);
  auto clonedType = cast<FragmentType>(clone->getResult(0).getType());
  FailureOr<unsigned> clonedAxis = axisForSource(clonedType, sourceId);
  if (failed(clonedAxis))
    return failure();
  bool introducedUnitAxis = false;
  if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
    auto extent = cast<PhysicalExprAttr>(clonedType.getShape()[*clonedAxis]);
    auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
    introducedUnitAxis =
        extent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        extent.getValue() == 1 &&
        (!input || failed(axisForSource(input, sourceId)));
  }
  if (!introducedUnitAxis)
    clone->getResult(0).setType(
        replaceExtent(clonedType, *clonedAxis, blockedExtent));
  if (!mapping.lookupOrNull(value))
    mapping.map(value, clone->getResult(0));
  return clone->getResult(0);
}

FragmentType eraseFragmentAxis(FragmentType source, unsigned erasedAxis) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
    if (axis == erasedAxis)
      continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size()));
  }
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           ArrayAttr::get(source.getContext(), mappings),
                           source.getValidity(), source.getOwner());
}

ParameterOp getOrCreateParameter(func::FuncOp kernel, StringRef name,
                                 ParameterRole role,
                                 ArrayRef<int64_t> candidates) {
  ParameterOp existing;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() == name)
      existing = parameter;
  });
  if (existing) {
    ParameterAttr schema = existing.getParameter();
    auto expectedCandidates =
        DenseI64ArrayAttr::get(kernel.getContext(), candidates);
    if (schema.getRole() != static_cast<uint32_t>(role) ||
        schema.getCandidates() != expectedCandidates) {
      existing.emitOpError(
          "physical parameter name is reused with a different role or candidate domain");
      return ParameterOp();
    }
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr(name),
      static_cast<uint32_t>(role),
      DenseI64ArrayAttr::get(kernel.getContext(), candidates));
  return builder.create<ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
}

FailureOr<ParameterOp> fullCoverageParameter(func::FuncOp kernel,
                                             PhysicalExprAttr extent) {
  if (extent.getKind() !=
      static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  StringRef name = extent.getSymbol().getValue();
  StringRef identity = name;
  uint64_t dimension = 0;
  if (!identity.consume_front("FRAGMENT_D") ||
      identity.getAsInteger(10, dimension))
    return failure();
  bool launchVisible = false;
  for (BlockArgument argument : kernel.getArguments()) {
    DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    launchVisible |= kind && kind.getValue() == "dimension" && identity &&
                     identity.getInt() == static_cast<int64_t>(dimension);
  }
  if (!launchVisible)
    return failure();
  ParameterOp parameter;
  kernel.walk([&](ParameterOp candidate) {
    if (candidate.getParameter().getName().getValue() == name)
      parameter = candidate;
  });
  if (!parameter)
    return failure();
  static constexpr int64_t candidates[] = {
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
      4096, 8192, 16384, 32768, 65536};
  ParameterAttr schema = parameter.getParameter();
  parameter->setAttr(
      "parameter",
      ParameterAttr::get(kernel.getContext(), schema.getName(), schema.getRole(),
                         DenseI64ArrayAttr::get(kernel.getContext(), candidates)));
  parameter->setAttr(coverageDimensionAttr,
                     IntegerAttr::get(IntegerType::get(kernel.getContext(), 64),
                                      dimension));
  return parameter;
}

FailureOr<ParameterOp> parameterForExtent(func::FuncOp kernel,
                                          PhysicalExprAttr extent) {
  if (extent.getKind() !=
      static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  ParameterOp result;
  kernel.walk([&](ParameterOp parameter) {
    if (!result && parameter.getParameter().getName() == extent.getSymbol())
      result = parameter;
  });
  return result ? FailureOr<ParameterOp>(result)
                : FailureOr<ParameterOp>(failure());
}

FailureOr<Value> dimensionArgument(func::FuncOp kernel, int64_t dimension);

FailureOr<ParameterOp> parameterForDimension(func::FuncOp kernel,
                                             int64_t dimension) {
  ParameterOp result;
  bool ambiguous = false;
  kernel.walk([&](ParameterOp parameter) {
    auto bound = parameter->getAttrOfType<IntegerAttr>(dimensionAttr);
    auto coverage =
        parameter->getAttrOfType<IntegerAttr>(coverageDimensionAttr);
    bool matches = (bound && bound.getInt() == dimension) ||
                   (coverage && coverage.getInt() == dimension);
    if (!matches)
      return;
    uint32_t role = parameter.getParameter().getRole();
    bool ownership =
        role == static_cast<uint32_t>(ParameterRole::OwnershipM) ||
        role == static_cast<uint32_t>(ParameterRole::OwnershipN);
    if (!ownership && !coverage)
      return;
    if (result && result != parameter) {
      ambiguous = true;
      return;
    }
    result = parameter;
  });
  return result && !ambiguous ? FailureOr<ParameterOp>(result)
                              : FailureOr<ParameterOp>(failure());
}

PhysicalExprAttr selectedParameterExtent(ParameterOp parameter) {
  ParameterAttr schema = parameter.getParameter();
  StringRef name = schema.getName().getValue();
  if (name.starts_with("FRAGMENT_S") && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    name);
}

FailureOr<ParameterOp> fullCoverageParameterForDimension(func::FuncOp kernel,
                                                         int64_t dimension) {
  if (dimension <= 0 || failed(dimensionArgument(kernel, dimension)))
    return failure();
  static constexpr int64_t candidates[] = {
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
      4096, 8192, 16384, 32768, 65536};
  ParameterOp parameter = getOrCreateParameter(
      kernel, ("REDUCE_FULL_D" + Twine(dimension)).str(),
      ParameterRole::OwnershipN, candidates);
  if (!parameter)
    return failure();
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
  return parameter;
}

bool hasNonUnitFreeAxis(ReduceOp reduce) {
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
      if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
        continue;
      auto extent = cast<PhysicalExprAttr>(attribute);
      if (extent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          extent.getValue() == 1)
        continue;
      return true;
    }
  }
  return false;
}

bool hasSelectedSegmentExtent(ReduceOp reduce, func::FuncOp kernel) {
  ParameterOp segment;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    int64_t axis = reduce.getAxes().empty() ? -1 : reduce.getAxes().front();
    if (!fragment || axis < 0 ||
        axis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[axis]);
    FailureOr<ParameterOp> parameter = parameterForExtent(kernel, extent);
    if (failed(parameter) ||
        (*parameter).getParameter().getRole() !=
            static_cast<uint32_t>(ParameterRole::ScanChunk))
      return false;
    if (segment && segment != *parameter)
      return false;
    segment = *parameter;
  }
  return static_cast<bool>(segment);
}

FailureOr<Value> dimensionArgument(func::FuncOp kernel, int64_t dimension) {
  for (BlockArgument argument : kernel.getArguments()) {
    DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == dimension)
      return Value(argument);
  }
  return failure();
}

void collectSourceRanges(Value value, uint64_t sourceId,
                         SmallVectorImpl<MakeRangeOp> &ranges,
                         llvm::SmallPtrSetImpl<Operation *> &visited) {
  (void)visited;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return;
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value);
  if (fact.state == PhysicalFactState::Unknown)
    return;
  for (MakeRangeOp range : fact.roots)
    if (range.getSourceId() == sourceId && !llvm::is_contained(ranges, range))
      ranges.push_back(range);
}

bool sameLogicalSourceRange(MakeRangeOp lhs, MakeRangeOp rhs) {
  if (lhs.getSourceId() != rhs.getSourceId() ||
      lhs.getSourceAxis() != rhs.getSourceAxis())
    return false;
  FailureOr<int64_t> leftDimension = rangeDimension(lhs);
  FailureOr<int64_t> rightDimension = rangeDimension(rhs);
  if (!lhs->hasAttr(sourceSubregionAttr) &&
      !rhs->hasAttr(sourceSubregionAttr) && succeeded(leftDimension) &&
      succeeded(rightDimension) && *leftDimension == *rightDimension)
    return true;
  return sameScalarExpression(lhs.getStart(), rhs.getStart()) &&
         sameScalarExpression(lhs.getExtent(), rhs.getExtent()) &&
         sameScalarExpression(lhs.getStep(), rhs.getStep());
}

FailureOr<MakeRangeOp> uniqueSourceRange(func::FuncOp kernel,
                                         uint64_t sourceId) {
  MakeRangeOp result;
  bool ambiguous = false;
  kernel.walk([&](MakeRangeOp candidate) {
    if (candidate.getSourceId() != sourceId)
      return;
    if (!result) {
      result = candidate;
      return;
    }
    if (!sameLogicalSourceRange(result, candidate)) {
      ambiguous = true;
      return;
    }
  });
  return result && !ambiguous ? FailureOr<MakeRangeOp>(result)
                              : FailureOr<MakeRangeOp>(failure());
}

FailureOr<bool> realizeStaticPaddingReduce(ReduceOp reduce,
                                           func::FuncOp kernel) {
  if (reduce.getAxes().size() != 1 || reduce.getSourceCount() == 0)
    return false;
  int64_t reductionAxis = reduce.getAxes().front();
  if (reductionAxis < 0)
    return false;

  PhysicalExprAttr logicalExtent;
  SmallVector<SmallVector<MakeRangeOp>> componentRanges;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    auto extent =
        dyn_cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (!extent || extent.getKind() !=
                       static_cast<uint32_t>(PhysicalExprKind::Constant))
      return false;
    if (logicalExtent && logicalExtent != extent)
      return false;
    logicalExtent = extent;
    SmallVector<MakeRangeOp> ranges;
    SmallVector<LoadOp> loads;
    llvm::SmallPtrSet<Operation *, 32> visited;
    collectRangesAndLoads(source, extent, ranges, loads, visited);
    llvm::sort(ranges, [](MakeRangeOp lhs, MakeRangeOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    llvm::sort(loads, [](LoadOp lhs, LoadOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    loads.erase(std::unique(loads.begin(), loads.end()), loads.end());
    componentRanges.push_back(std::move(ranges));
  }
  if (!logicalExtent || logicalExtent.getValue() <= 0)
    return false;
  PhysicalExprAttr physicalExtent = nextPowerOfTwo(logicalExtent);
  if (physicalExtent == logicalExtent)
    return false;
  if (llvm::any_of(componentRanges,
                   [](ArrayRef<MakeRangeOp> ranges) { return ranges.empty(); }))
    return reduce.emitOpError(
               "static non-power-of-two reduction source has no exact logical range"),
           failure();

  OpBuilder builder(reduce);
  Value physicalExtentValue = builder.create<arith::ConstantIndexOp>(
      reduce.getLoc(), physicalExtent.getValue());
  for (unsigned component = 0; component < reduce.getSourceCount(); ++component) {
    auto originalType =
        cast<FragmentType>(reduce.getInputs()[component].getType());
    auto sourceMap = cast<AxisMapAttr>(
        originalType.getAxisMaps()[static_cast<unsigned>(reductionAxis)]);
    PhysicalSourceAxis reductionSource{sourceMap.getSourceId(),
                                       sourceMap.getSourceAxis()};
    IRMapping mapping;
    SmallVector<Value> tailPredicates;
    FailureOr<Value> source = clonePaddedProducer(
        builder, reduce.getLoc(), reduce.getInputs()[component], reductionSource,
        logicalExtent, physicalExtent, physicalExtentValue, mapping,
        tailPredicates);
    if (failed(source) || tailPredicates.empty())
      return reduce.emitOpError(
                 "static reduction producer cannot be replayed over its padded extent"),
             failure();
    auto sourceType = cast<FragmentType>(source->getType());
    Value tail;
    for (Value base : tailPredicates) {
      FailureOr<Value> current = predicateForReductionSource(
          builder, reduce.getLoc(), base, sourceType,
          static_cast<unsigned>(reductionAxis));
      if (failed(current))
        return failure();
      tail = tail ? Value(builder.create<BinaryOp>(
                        reduce.getLoc(), current->getType(), tail, *current,
                        BinaryOperator::LogicalAnd))
                  : *current;
    }
    Value identity =
        reduce.getInputs()[reduce.getSourceCount() + component];
    if (identity.getType() != sourceType)
      identity = builder.create<BroadcastOp>(reduce.getLoc(), sourceType, identity);
    Value selected = builder.create<SelectOp>(reduce.getLoc(), sourceType, tail,
                                              *source, identity);
    reduce->setOperand(component, selected);
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}

FailureOr<bool> realizeFullCoverageReduce(ReduceOp reduce,
                                          func::FuncOp kernel) {
  if (reduce.getAxes().size() != 1 || reduce.getSourceCount() == 0)
    return false;
  int64_t reductionAxis = reduce.getAxes().front();
  if (reductionAxis < 0)
    return false;

  PhysicalExprAttr extent;
  SmallVector<uint64_t> sourceIds;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    FailureOr<AxisMapAttr> mapping = axisMap(fragment, reductionAxis);
    if (failed(mapping))
      return false;
    PhysicalExprAttr current =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (extent && extent != current)
      return false;
    extent = current;
    sourceIds.push_back(mapping->getSourceId());
  }
  if (!extent || extent.getKind() !=
                     static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return false;
  llvm::DenseMap<uint64_t, MakeRangeOp> ranges;
  for (auto [source, sourceId] : llvm::zip(
           reduce.getInputs().take_front(reduce.getSourceCount()), sourceIds)) {
    SmallVector<MakeRangeOp> candidates;
    llvm::SmallPtrSet<Operation *, 32> visited;
    collectSourceRanges(source, sourceId, candidates, visited);
    for (MakeRangeOp candidate : candidates) {
      auto found = ranges.find(sourceId);
      if (found != ranges.end() &&
          !sameLogicalSourceRange(found->second, candidate))
        return false;
      ranges[sourceId] = candidate;
    }
  }
  if (ranges.empty())
    return false;
  MakeRangeOp range = ranges.begin()->second;
  if (range->hasAttr(sourceSubregionAttr))
    return false;
  FailureOr<ParameterOp> parameter = fullCoverageParameter(kernel, extent);
  if (failed(parameter))
    return false;
  Value logicalExtent = range.getExtent();
  if (isCompileTimeValue(logicalExtent)) {
    FailureOr<int64_t> sourceDimension = rangeDimension(range);
    auto coverageDimension =
        (*parameter)->getAttrOfType<IntegerAttr>(coverageDimensionAttr);
    if (range->hasAttr(sourceSubregionAttr) || failed(sourceDimension) ||
        !coverageDimension ||
        *sourceDimension != coverageDimension.getInt())
      return false;
    FailureOr<Value> launchExtent =
        dimensionArgument(kernel, *sourceDimension);
    if (failed(launchExtent))
      return false;
    logicalExtent = *launchExtent;
  }

  auto coverageDimension =
      (*parameter)->getAttrOfType<IntegerAttr>(coverageDimensionAttr);
  if (!coverageDimension ||
      failed(bindFullCoverageDimension(
          kernel, coverageDimension.getInt(), parameter->getResult())))
    return failure();

  OpBuilder builder(reduce);
  for (unsigned component = 0; component < reduce.getSourceCount(); ++component) {
    Value source = reduce.getInputs()[component];
    auto found = ranges.find(sourceIds[component]);
    if (found == ranges.end())
      return false;
    MakeRangeOp componentRange = found->second;
    auto coordinateType =
        cast<FragmentType>(componentRange.getResult().getType());
    auto sourceType = cast<FragmentType>(source.getType());
    Value stop = builder.create<BinaryOp>(
        reduce.getLoc(), builder.getIndexType(), componentRange.getStart(),
        logicalExtent, BinaryOperator::Add);
    Value stopFragment =
        builder.create<BroadcastOp>(reduce.getLoc(), coordinateType, stop);
    auto coordinatePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), coordinateType.getShape(),
        coordinateType.getAxisMaps(), coordinateType.getValidity(),
        coordinateType.getOwner());
    Value coordinateValid = builder.create<CompareOp>(
        reduce.getLoc(), coordinatePredicate, componentRange.getResult(),
        stopFragment, ComparePredicate::Lt);
    auto predicateType = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), sourceType.getShape(),
        sourceType.getAxisMaps(), sourceType.getValidity(), sourceType.getOwner());
    Value valid = coordinateValid;
    if (valid.getType() != predicateType)
      valid = builder.create<BroadcastOp>(reduce.getLoc(), predicateType, valid);
    Value identity = reduce.getInputs()[reduce.getSourceCount() + component];
    if (identity.getType() != sourceType)
      identity = builder.create<BroadcastOp>(reduce.getLoc(), sourceType, identity);
    Value selected = builder.create<SelectOp>(reduce.getLoc(), sourceType, valid,
                                              source, identity);
    reduce->setOperand(component, selected);
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}

LogicalResult bindReductionFreeAxes(ReduceOp reduce, func::FuncOp kernel) {
  SmallVector<std::tuple<Value, PhysicalSourceAxis, int64_t>> pending;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (auto [axis, attribute] : llvm::enumerate(fragment.getShape())) {
      if (llvm::is_contained(reduce.getAxes(), static_cast<int64_t>(axis)))
        continue;
      auto extent = cast<PhysicalExprAttr>(attribute);
      if (isCompileTimeExtent(extent))
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      if (extent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Dimension)) {
        if (mapping.getDimensionId() <= 0)
          return reduce.emitOpError(
              "reduction free axis has no logical dimension authority");
        pending.emplace_back(
            source,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis()},
            mapping.getDimensionId());
        continue;
      }
      if (extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
        return reduce.emitOpError(
            "reduction free axis has no physical parameter authority");
      FailureOr<ParameterOp> parameter = parameterForExtent(kernel, extent);
      if (failed(parameter))
        return reduce.emitOpError(
            "reduction free axis has no physical parameter authority");
      uint32_t role = (*parameter).getParameter().getRole();
      bool ownership =
          role == static_cast<uint32_t>(ParameterRole::OwnershipM) ||
          role == static_cast<uint32_t>(ParameterRole::OwnershipN);
      if (!ownership && !(*parameter)->hasAttr(coverageDimensionAttr))
        return reduce.emitOpError(
            "reduction free axis is neither ownership-blocked nor exact full coverage");
    }
  }

  for (auto [source, sourceAxis, dimension] : pending) {
    FailureOr<ParameterOp> parameter = parameterForDimension(kernel, dimension);
    bool fullCoverage = false;
    if (failed(parameter)) {
      parameter = fullCoverageParameterForDimension(kernel, dimension);
      fullCoverage = succeeded(parameter);
    }
    if (failed(parameter))
      return reduce.emitOpError(
                 "reduction free axis has neither prior ownership nor exact full-coverage authority")
             << "; dimension=" << dimension;
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact ranges = analysis.sourceRanges(source, sourceAxis);
    if (ranges.roots.empty()) {
      // A free axis introduced by a typed broadcast has no coordinate range of
      // its own.  Its dimension identity is nevertheless exact, so project
      // only this value flow onto the already selected ownership extent.
      retargetDimensionExtent(source, dimension,
                              selectedParameterExtent(*parameter));
      if (fullCoverage)
        return reduce.emitOpError(
                   "reduction full-coverage free axis has no coordinate range for tail validity")
               << "; source_id=" << sourceAxis.sourceId
               << ", source_axis=" << sourceAxis.sourceAxis
               << ", dimension=" << dimension;
      continue;
    }
    if (ranges.state == PhysicalFactState::Unknown)
      return reduce.emitOpError(
                 "reduction free axis has no physical range projection")
             << "; source_id=" << sourceAxis.sourceId
             << ", source_axis=" << sourceAxis.sourceAxis
             << ", dimension=" << dimension;
    MakeRangeOp authority = ranges.roots.front();
    if (!llvm::all_of(ranges.roots, [&](MakeRangeOp range) {
          return sameLogicalSourceRange(authority, range);
        }))
      return reduce.emitOpError(
                 "reduction free axis has conflicting physical range projections")
             << "; source_id=" << sourceAxis.sourceId
             << ", source_axis=" << sourceAxis.sourceAxis
             << ", dimension=" << dimension;
    for (MakeRangeOp range : ranges.roots)
      retargetDimensionExtent(range.getResult(), dimension,
                              selectedParameterExtent(*parameter));
    if (fullCoverage &&
        failed(bindFullCoverageDimension(kernel, dimension,
                                         parameter->getResult())))
      return reduce.emitOpError(
                 "reduction free axis full-coverage binding failed")
             << "; dimension=" << dimension;
  }
  return success();
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
             BitcastOp, MakeRecordOp, ExtractOp>(operation);
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
  Location location = source.getLoc();
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
  OpBuilder builder(targetBlock, targetBlock->end());

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
          source.getContext(), original.getFieldNames(),
          ArrayAttr::get(source.getContext(), fields), original.getOwner()));
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

FragmentType eraseFragmentAxes(FragmentType source,
                               ArrayRef<int64_t> erasedAxes) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  for (auto [axis, extent] : llvm::enumerate(source.getShape())) {
    if (llvm::is_contained(erasedAxes, static_cast<int64_t>(axis)))
      continue;
    shape.push_back(extent);
    auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
    mappings.push_back(AxisMapAttr::get(
        source.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
        mapping.getDimensionId(), mappings.size()));
  }
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           ArrayAttr::get(source.getContext(), mappings),
                           source.getValidity(), source.getOwner());
}

bool isSingleComponentAddReduce(ReduceOp reduce) {
  if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
      reduce.getCaptureCount() != 0 || reduce.getCombine().empty() ||
      reduce.getCombine().getBlocks().size() != 1)
    return false;
  Block &block = reduce.getCombine().front();
  if (block.getNumArguments() != 2)
    return false;
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;
  auto combine = yield.getValues().front().getDefiningOp<BinaryOp>();
  return combine && combine->getBlock() == &block &&
         combine.getOperatorKind() == BinaryOperator::Add &&
         combine.getLhs() == block.getArgument(0) &&
         combine.getRhs() == block.getArgument(1) &&
         std::distance(block.begin(), block.end()) == 2;
}

FailureOr<bool> decomposeFullCoverageMultiAxisReduce(
    ReduceOp reduce, func::FuncOp kernel, unsigned outerAxis,
    MakeRangeOp traversalRange) {
  if (traversalRange->hasAttr(sourceSubregionAttr))
    return false;
  for (Operation &operation : reduce.getCombine().front().without_terminator())
    if (!canLiftCombineOperation(operation))
      return false;

  auto firstSource =
      dyn_cast<FragmentType>(reduce.getInputs().front().getType());
  if (!firstSource || outerAxis >= firstSource.getShape().size())
    return false;
  auto outerExtent =
      dyn_cast<PhysicalExprAttr>(firstSource.getShape()[outerAxis]);
  FailureOr<ParameterOp> coverage =
      outerExtent ? fullCoverageParameter(kernel, outerExtent)
                  : FailureOr<ParameterOp>(failure());
  if (failed(coverage))
    return false;
  auto dimension =
      (*coverage)->getAttrOfType<IntegerAttr>(coverageDimensionAttr);
  if (!dimension ||
      failed(bindFullCoverageDimension(kernel, dimension.getInt(),
                                       coverage->getResult())))
    return reduce.emitOpError(
               "multi-axis full-coverage fragment could not bind its logical dimension"),
           failure();

  SmallVector<int64_t> innerAxes;
  for (int64_t axis : reduce.getAxes())
    if (axis != static_cast<int64_t>(outerAxis))
      innerAxes.push_back(axis);
  if (innerAxes.empty())
    return false;

  SmallVector<Value> identities(
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .begin(),
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .end());
  ValueRange captures = reduce.getInputs().drop_front(
      reduce.getSourceCount() + reduce.getIdentityCount());
  SmallVector<Type> innerResultTypes;
  SmallVector<Value> innerIdentities;
  OpBuilder builder(reduce);
  for (auto [component, source] : llvm::enumerate(
           reduce.getInputs().take_front(reduce.getSourceCount()))) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || outerAxis >= fragment.getShape().size())
      return reduce.emitOpError(
                 "multi-axis full-coverage source lost its fragment schema"),
             failure();
    FragmentType result = eraseFragmentAxes(fragment, innerAxes);
    if (result.getShape().empty())
      return reduce.emitOpError(
                 "multi-axis inner reduction did not retain its outer axis"),
             failure();
    innerResultTypes.push_back(result);
    FailureOr<Value> identity = alignToExecutionSchema(
        builder, reduce.getLoc(), identities[component], result);
    if (failed(identity))
      return reduce.emitOpError(
                 "multi-axis identity cannot be projected over the retained fragment"),
             failure();
    innerIdentities.push_back(*identity);
  }

  SmallVector<Value> innerInputs(
      reduce.getInputs().take_front(reduce.getSourceCount()).begin(),
      reduce.getInputs().take_front(reduce.getSourceCount()).end());
  innerInputs.append(innerIdentities.begin(), innerIdentities.end());
  innerInputs.append(captures.begin(), captures.end());
  OperationState innerState(reduce.getLoc(), ReduceOp::getOperationName());
  innerState.addOperands(innerInputs);
  innerState.addTypes(innerResultTypes);
  innerState.addAttribute(
      "axes", DenseI64ArrayAttr::get(reduce.getContext(), innerAxes));
  innerState.addAttribute("source_count", reduce->getAttr("source_count"));
  innerState.addAttribute("identity_count", reduce->getAttr("identity_count"));
  innerState.addAttribute("capture_count", reduce->getAttr("capture_count"));
  innerState.addRegion();
  auto innerReduce = cast<ReduceOp>(builder.create(innerState));
  if (Attribute origin = reduce->getAttr(originAttr))
    innerReduce->setAttr(originAttr, origin);
  std::string reason;
  if (failed(cloneLiftedCombineRegion(reduce.getCombine(),
                                      innerReduce.getCombine(),
                                      innerResultTypes, reason)))
    return reduce.emitOpError(
               "multi-axis combine cannot execute over its retained fragment: ")
           << reason;

  unsigned outerResultAxis = 0;
  for (unsigned axis = 0; axis < outerAxis; ++axis)
    if (!llvm::is_contained(innerAxes, static_cast<int64_t>(axis)))
      ++outerResultAxis;
  SmallVector<Value> outerInputs(innerReduce.getResults().begin(),
                                 innerReduce.getResults().end());
  outerInputs.append(identities.begin(), identities.end());
  outerInputs.append(captures.begin(), captures.end());
  OperationState outerState(reduce.getLoc(), ReduceOp::getOperationName());
  outerState.addOperands(outerInputs);
  outerState.addTypes(reduce.getResultTypes());
  outerState.addAttribute(
      "axes", DenseI64ArrayAttr::get(
                  reduce.getContext(),
                  ArrayRef<int64_t>{static_cast<int64_t>(outerResultAxis)}));
  outerState.addAttribute("source_count", reduce->getAttr("source_count"));
  outerState.addAttribute("identity_count", reduce->getAttr("identity_count"));
  outerState.addAttribute("capture_count", reduce->getAttr("capture_count"));
  outerState.addRegion();
  auto outerReduce = cast<ReduceOp>(builder.create(outerState));
  if (Attribute origin = reduce->getAttr(originAttr))
    outerReduce->setAttr(originAttr, origin);
  IRMapping regionMapping;
  reduce.getCombine().cloneInto(&outerReduce.getCombine(), regionMapping);

  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), outerReduce.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return true;
}

LogicalResult decomposeMultiAxisReduce(ReduceOp reduce, func::FuncOp kernel) {
  if (!reduce->getBlock() || reduce.getAxes().size() <= 1)
    return success();
  if (reduce.getSourceCount() == 0)
    return reduce.emitOpError(
        "multi-axis reduction decomposition requires at least one source");

  unsigned outerAxis = static_cast<unsigned>(reduce.getAxes().front());
  SmallVector<SourcePlan> plans;
  SmallVector<SmallVector<RootAccess>> accesses;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    FailureOr<SourcePlan> plan = analyzeSource(source, outerAxis);
    if (failed(plan)) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      FailureOr<AxisMapAttr> mapping =
          fragment ? axisMap(fragment, outerAxis)
                   : FailureOr<AxisMapAttr>(failure());
      llvm::SmallPtrSet<Operation *, 16> visited;
      if (!fragment || failed(mapping) ||
          !isReplayableWithoutLoad(source, mapping->getSourceId(), visited))
        return reduce.emitOpError()
               << "multi-axis reduction outer axis is neither load-rooted nor a replayable pure source; source type="
               << source.getType() << ", producer="
               << (source.getDefiningOp()
                       ? source.getDefiningOp()->getName().getStringRef()
                       : StringRef("block argument"));
      plan = SourcePlan{source, mapping->getSourceId(), outerAxis, {}, {}, {}};
    }
    SmallVector<RootAccess> roots;
    for (LoadOp load : plan->roots) {
      FailureOr<RootAccess> access = analyzeRoot(load, plan->reductionRange);
      if (failed(access))
        return reduce.emitOpError(
            "multi-axis reduction outer axis has no unit-step load coordinate");
      roots.push_back(*access);
    }
    plans.push_back(*plan);
    accesses.push_back(std::move(roots));
  }

  std::optional<RootAccess> master;
  for (const auto &component : accesses)
    if (!component.empty()) {
      master = component.front();
      break;
    }
  if (!master)
    return reduce.emitOpError(
        "multi-axis reduction has no load-rooted traversal authority");
  for (const auto &component : accesses)
    for (RootAccess access : component) {
      if (access.range.getSourceId() != master->range.getSourceId() ||
          access.range.getSourceAxis() != master->range.getSourceAxis() ||
          !sameScalarValue(access.range.getStart(), master->range.getStart()) ||
          !sameScalarValue(access.range.getExtent(), master->range.getExtent()) ||
          !sameScalarValue(access.range.getStep(), master->range.getStep()))
        return reduce.emitOpError(
            "multi-axis reduction components do not share one exact outer traversal");
    }

  FailureOr<bool> fullCoverage = decomposeFullCoverageMultiAxisReduce(
      reduce, kernel, outerAxis, master->range);
  if (failed(fullCoverage))
    return failure();
  if (*fullCoverage)
    return success();

  SmallVector<int64_t> innerAxes;
  for (int64_t axis : reduce.getAxes()) {
    if (axis == static_cast<int64_t>(outerAxis))
      continue;
    innerAxes.push_back(axis > static_cast<int64_t>(outerAxis) ? axis - 1
                                                               : axis);
  }
  if (innerAxes.empty())
    return reduce.emitOpError(
        "multi-axis reduction decomposition lost every inner axis");

  SmallVector<Value> identities(
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .begin(),
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .end());
  ValueRange captures = reduce.getInputs().drop_front(
      reduce.getSourceCount() + reduce.getIdentityCount());
  PhysicalExprAttr unitExtent =
      expression(reduce.getContext(), PhysicalExprKind::Constant, 1);
  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  Value stop = builder.create<BinaryOp>(
      location, builder.getIndexType(), master->range.getStart(),
      master->range.getExtent(), BinaryOperator::Add);
  bool bodyFailed = false;
  std::string failureReason;
  auto loop = builder.create<scf::ForOp>(
      location, master->range.getStart(), stop, master->range.getStep(), identities,
      [&](OpBuilder &nested, Location nestedLocation, Value coordinate,
          ValueRange carries) {
        SmallVector<Value> innerSources;
        for (auto [component, plan] : llvm::enumerate(plans)) {
          IRMapping mapping;
          for (RootAccess access : accesses[component]) {
            mapping.map(access.range.getResult(), coordinate);
            LoadOp load = access.load;
            auto sourceType = cast<FragmentType>(load.getResult().getType());
            FragmentType slicedType = replaceExtent(
                sourceType, access.fragmentAxis, unitExtent);
            SmallVector<Value> coordinates(load.getCoordinates());
            FailureOr<Value> reducedCoordinate = replayValue(
                nested, nestedLocation,
                load.getCoordinates()[access.coordinateIndex], plan.sourceId,
                unitExtent, mapping);
            if (failed(reducedCoordinate)) {
              bodyFailed = true;
              failureReason =
                  "reduced source coordinate could not be cloned into the outer loop";
              return;
            }
            coordinates[access.coordinateIndex] = *reducedCoordinate;
            using ReplayAxis =
                std::tuple<uint64_t, PhysicalExprAttr, Value, Value>;
            SmallVector<ReplayAxis> replayAxes{{
                plan.sourceId, unitExtent, access.range.getResult(), coordinate}};
            for (auto [coordinateIndex, original] :
                 llvm::enumerate(load.getCoordinates())) {
              if (coordinateIndex == access.coordinateIndex)
                continue;
              MakeRangeOp range = sourceRange(original);
              if (!range)
                continue;
              if (!mapping.lookupOrNull(range.getResult())) {
                auto rangeType = cast<FragmentType>(range.getResult().getType());
                auto clone = nested.create<MakeRangeOp>(
                    nestedLocation, rangeType, range.getStart(),
                    range.getExtent(), range.getStep(), range.getSourceId(),
                    range.getSourceAxis());
                if (Attribute value = range->getAttr(sourceSubregionAttr))
                  clone->setAttr(sourceSubregionAttr, value);
                mapping.map(range.getResult(), clone.getResult());
              }
              auto rangeType = cast<FragmentType>(range.getResult().getType());
              replayAxes.emplace_back(
                  range.getSourceId(),
                  cast<PhysicalExprAttr>(rangeType.getShape()[0]),
                  range.getResult(), mapping.lookupOrNull(range.getResult()));
              FailureOr<Value> replayedCoordinate = replayValue(
                  nested, nestedLocation, original, range.getSourceId(),
                  cast<PhysicalExprAttr>(rangeType.getShape()[0]), mapping);
              if (failed(replayedCoordinate)) {
                bodyFailed = true;
                failureReason =
                    "non-reduced source coordinate could not be cloned into the outer loop";
                return;
              }
              coordinates[coordinateIndex] = *replayedCoordinate;
            }
            Value valid;
            if (load.getValid()) {
              valid = load.getValid();
              for (auto [sourceId, extent, originalRange, replacementRange] :
                   replayAxes) {
                IRMapping axisMapping;
                axisMapping.map(originalRange, replacementRange);
                FailureOr<Value> replayed = replayValue(
                    nested, nestedLocation, valid, sourceId, extent,
                    axisMapping);
                if (failed(replayed)) {
                  bodyFailed = true;
                  failureReason =
                      "source validity could not be cloned into the outer loop";
                  return;
                }
                valid = *replayed;
              }
            }
            Value fill;
            if (load.getFill()) {
              fill = load.getFill();
              for (auto [sourceId, extent, originalRange, replacementRange] :
                   replayAxes) {
                IRMapping axisMapping;
                axisMapping.map(originalRange, replacementRange);
                FailureOr<Value> replayed = replayValue(
                    nested, nestedLocation, fill, sourceId, extent,
                    axisMapping);
                if (failed(replayed)) {
                  bodyFailed = true;
                  failureReason =
                      "source fill could not be cloned into the outer loop";
                  return;
                }
                fill = *replayed;
              }
            }
            auto slicedLoad = nested.create<LoadOp>(
                nestedLocation, slicedType, load.getResource(), coordinates,
                valid, fill, load.getSourceAxes());
            mapping.map(load.getResult(), slicedLoad.getResult());
          }
          FailureOr<Value> replayed = replayValue(
              nested, nestedLocation, plan.source, plan.sourceId, unitExtent,
              mapping);
          if (failed(replayed)) {
            bodyFailed = true;
            failureReason = "outer-axis source graph could not be sliced";
            return;
          }
          auto sliced = dyn_cast<FragmentType>((*replayed).getType());
          if (!sliced || outerAxis >= sliced.getShape().size()) {
            bodyFailed = true;
            failureReason = "outer-axis source lost its fragment schema";
            return;
          }
          FragmentType squeezed = eraseFragmentAxis(sliced, outerAxis);
          FailureOr<ArrayAttr> reassociation =
              inferReshapeReassociation(sliced, squeezed);
          if (failed(reassociation)) {
            bodyFailed = true;
            failureReason =
                "outer-axis source has no exact row-major reassociation";
            return;
          }
          innerSources.push_back(nested.create<ReshapeOp>(
              nestedLocation, squeezed, *replayed,
              *reassociation));
        }
        if (bodyFailed)
          return;

        SmallVector<Value> innerInputs(innerSources);
        innerInputs.append(identities.begin(), identities.end());
        innerInputs.append(captures.begin(), captures.end());
        OperationState state(nestedLocation, ReduceOp::getOperationName());
        state.addOperands(innerInputs);
        state.addTypes(reduce.getResultTypes());
        state.addAttribute("axes",
                           DenseI64ArrayAttr::get(reduce.getContext(), innerAxes));
        state.addAttribute("source_count", reduce->getAttr("source_count"));
        state.addAttribute("identity_count", reduce->getAttr("identity_count"));
        state.addAttribute("capture_count", reduce->getAttr("capture_count"));
        state.addRegion();
        auto innerReduce = cast<ReduceOp>(nested.create(state));
        if (Attribute origin = reduce->getAttr(originAttr))
          innerReduce->setAttr(originAttr, origin);
        IRMapping regionMapping;
        reduce.getCombine().cloneInto(&innerReduce.getCombine(), regionMapping);

        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(innerReduce.getResults().begin(),
                                innerReduce.getResults().end());
        combineArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, innerReduce.getCombine(), combineArguments, failureReason);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      });
  if (bodyFailed) {
    loop.erase();
    return reduce.emitOpError(
               "multi-axis reduction decomposition failed: ")
           << failureReason;
  }
  if (Attribute origin = reduce->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  loop->setAttr(reductionTraversalSourceAttr,
                builder.getI64IntegerAttr(master->range.getSourceId()));
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), loop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult realizeRuntimeReduce(ReduceOp reduce,
                                   ArrayRef<SourcePlan> sourcePlans,
                                   func::FuncOp kernel) {
  if (sourcePlans.empty())
    return reduce.emitOpError("runtime reduction has no physical sources");
  SmallVector<SmallVector<RootAccess>> accesses;
  SmallVector<MakeRangeOp> traversalRanges;
  for (const SourcePlan &plan : sourcePlans) {
    auto source = cast<FragmentType>(plan.source.getType());
    for (auto [axis, extent] : llvm::enumerate(source.getShape()))
      if (axis != plan.reductionAxis &&
          !isCompileTimeExtent(cast<PhysicalExprAttr>(extent))) {
        auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[axis]);
        return reduce.emitOpError(
                   "runtime reduction free axes must be physicalized before chunking")
               << "; free_axis=" << axis << ", reduction_axis="
               << plan.reductionAxis << ", extent=" << extent
               << ", source_id=" << mapping.getSourceId()
               << ", source_axis=" << mapping.getSourceAxis()
               << ", dimension=" << mapping.getDimensionId()
               << ", source=" << source;
      }
    SmallVector<RootAccess> roots;
    for (LoadOp load : plan.roots) {
      FailureOr<RootAccess> access = analyzeRoot(load, plan.reductionRange);
      if (failed(access))
        return reduce.emitOpError(
            "load-rooted producer has no unit-step reduction coordinate");
      roots.push_back(*access);
    }
    MakeRangeOp traversal;
    for (RootAccess access : roots) {
      if (traversal && !sameLogicalSourceRange(traversal, access.range))
        return reduce.emitOpError(
            "one reduction component has multiple physical source ranges");
      traversal = access.range;
    }
    for (MakeRangeOp range : plan.ranges) {
      if (traversal && !sameLogicalSourceRange(traversal, range))
        return reduce.emitOpError(
            "one reduction component has ambiguous range provenance");
      traversal = range;
    }
    if (!traversal)
      return reduce.emitOpError(
          "runtime reduction has no exact physical range authority");
    accesses.push_back(std::move(roots));
    traversalRanges.push_back(traversal);
  }
  for (Type result : reduce.getResultTypes())
    if (auto fragment = dyn_cast<FragmentType>(result))
      if (llvm::any_of(fragment.getShape(), [](Attribute extent) {
            return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
          }))
        return reduce.emitOpError(
            "runtime reduction result still has an unphysicalized free axis");

  MakeRangeOp firstRange = traversalRanges.front();
  FailureOr<Value> firstEnd = resolveLogicalRangeEnd(kernel, firstRange);
  if (failed(firstEnd))
    return reduce.emitOpError(
        "runtime reduction source range has no exact logical end");
  for (MakeRangeOp range : traversalRanges) {
    FailureOr<Value> end = resolveLogicalRangeEnd(kernel, range);
    if (!sameLogicalSourceRange(firstRange, range) || failed(end) ||
        !sameScalarExpression(*firstEnd, *end))
      return reduce.emitOpError(
          "runtime reduction components require one lockstep logical range");
  }
  auto firstSource = cast<FragmentType>(sourcePlans.front().source.getType());
  PhysicalExprAttr sourceExtent = cast<PhysicalExprAttr>(
      firstSource.getShape()[sourcePlans.front().reductionAxis]);
  FailureOr<ParameterOp> fullCoverage = failure();
  if (!hasNonUnitFreeAxis(reduce))
    fullCoverage = fullCoverageParameter(kernel, sourceExtent);
  FailureOr<ParameterOp> selectedChunk =
      parameterForExtent(kernel, sourceExtent);
  ParameterOp chunk;
  if (!firstRange->hasAttr(sourceSubregionAttr) && succeeded(fullCoverage)) {
    chunk = *fullCoverage;
  } else if (succeeded(selectedChunk) &&
             !(*selectedChunk)->hasAttr(coverageDimensionAttr) &&
             (*selectedChunk).getParameter().getRole() ==
                 static_cast<uint32_t>(ParameterRole::Reduction)) {
    chunk = *selectedChunk;
  } else {
    std::string name =
        ("REDUCE_CHUNK_" + Twine(sourcePlans.front().sourceId)).str();
    SmallVector<int64_t> candidates{8, 16, 32, 64, 128,
                                    256, 512, 1024, 2048, 4096};
    if (sourceExtent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        sourceExtent.getValue() > 0) {
      PhysicalExprAttr upper = nextPowerOfTwo(sourceExtent);
      llvm::erase_if(candidates, [&](int64_t candidate) {
        return candidate > upper.getValue();
      });
      if (candidates.empty())
        candidates.push_back(upper.getValue());
    }
    chunk = getOrCreateParameter(kernel, name, ParameterRole::Reduction,
                                 candidates);
    if (chunk)
      if (FailureOr<int64_t> dimension = rangeDimension(firstRange);
          succeeded(dimension))
        chunk->setAttr(dimensionAttr,
                       IntegerAttr::get(IntegerType::get(chunk.getContext(), 64),
                                        *dimension));
  }
  if (!chunk)
    return failure();
  PhysicalExprAttr chunkExtent = expression(
      reduce.getContext(), PhysicalExprKind::Parameter, 0,
      chunk.getParameter().getName().getValue());

  const bool vectorAccumulation =
      isSingleComponentAddReduce(reduce) &&
      succeeded(scalarSource(reduce.getInputs()[reduce.getSourceCount()]));
  SmallVector<FragmentType> blockedSourceTypes;
  for (const SourcePlan &plan : sourcePlans) {
    auto originalSource = cast<FragmentType>(plan.source.getType());
    blockedSourceTypes.push_back(
        replaceExtent(originalSource, plan.reductionAxis, chunkExtent));
  }

  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  Value stop = *firstEnd;
  SmallVector<Value> identities(
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .begin(),
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .end());
  ValueRange captures = reduce.getInputs().drop_front(
      reduce.getSourceCount() + reduce.getIdentityCount());

  SmallVector<Value> loopInitials(identities);
  if (vectorAccumulation) {
    loopInitials.clear();
    for (auto [identity, blockedType] :
         llvm::zip(identities, blockedSourceTypes)) {
      FailureOr<Value> scalar = scalarSource(identity);
      if (failed(scalar))
        return reduce.emitOpError(
            "vector reduction accumulation lost its scalar identity");
      Value initial = builder.create<SplatOp>(location, blockedType, *scalar);
      loopInitials.push_back(initial);
    }
  }

  bool bodyFailed = false;
  std::string bodyFailure = "unknown producer replay failure";
  auto loop = builder.create<scf::ForOp>(
      location, firstRange.getStart(), stop, chunk.getResult(), loopInitials,
      [&](OpBuilder &nested, Location nestedLocation, Value chunkStart,
          ValueRange carries) {
        auto masterType = cast<FragmentType>(firstRange.getResult().getType());
        SmallVector<Attribute> masterShape(masterType.getShape().begin(),
                                           masterType.getShape().end());
        masterShape[0] = chunkExtent;
        auto blockedMaster = FragmentType::get(
            reduce.getContext(), masterType.getElementType(),
            ArrayAttr::get(reduce.getContext(), masterShape),
            masterType.getAxisMaps(), 2, masterType.getOwner());
        Value masterCoordinate = nested.create<MakeRangeOp>(
            nestedLocation, blockedMaster, chunkStart, chunk.getResult(),
            firstRange.getStep(), firstRange.getSourceId(),
            firstRange.getSourceAxis());
        Value masterEnd = nested.create<BroadcastOp>(nestedLocation,
                                                     blockedMaster, stop);
        auto masterPredicate = FragmentType::get(
            reduce.getContext(), nested.getI1Type(), blockedMaster.getShape(),
            blockedMaster.getAxisMaps(), 2, blockedMaster.getOwner());
        Value sharedTail = nested.create<CompareOp>(
            nestedLocation, masterPredicate, masterCoordinate, masterEnd,
            ComparePredicate::Lt);
        SmallVector<Value> blockedSources;
        for (auto [component, plan] : llvm::enumerate(sourcePlans)) {
          FragmentType blockedSource = blockedSourceTypes[component];
          auto blockedPredicate = FragmentType::get(
              reduce.getContext(), nested.getI1Type(), blockedSource.getShape(),
              blockedSource.getAxisMaps(), blockedSource.getValidity(),
              blockedSource.getOwner());
          IRMapping mapping;
          Value sourceTail;
          for (MakeRangeOp range : plan.ranges) {
            if (mapping.lookupOrNull(range.getResult()))
              continue;
            auto originalCoordinate = range.getResult().getType();
            SmallVector<Attribute> coordinateShape(
                originalCoordinate.getShape().begin(),
                originalCoordinate.getShape().end());
            coordinateShape[0] = chunkExtent;
            auto blockedCoordinate = FragmentType::get(
                reduce.getContext(), originalCoordinate.getElementType(),
                ArrayAttr::get(reduce.getContext(), coordinateShape),
                originalCoordinate.getAxisMaps(), originalCoordinate.getValidity(),
                originalCoordinate.getOwner());
            auto coordinate = nested.create<MakeRangeOp>(
                nestedLocation, blockedCoordinate, chunkStart,
                chunk.getResult(), range.getStep(), range.getSourceId(),
                range.getSourceAxis());
            if (Attribute value = range->getAttr(sourceSubregionAttr))
              coordinate->setAttr(sourceSubregionAttr, value);
            mapping.map(range.getResult(), coordinate.getResult());
            Value end = nested.create<BroadcastOp>(nestedLocation,
                                                   blockedCoordinate, stop);
            auto coordinatePredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(),
                blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(),
                blockedCoordinate.getValidity(),
                blockedCoordinate.getOwner());
            Value valid = nested.create<CompareOp>(
                nestedLocation, coordinatePredicate, coordinate.getResult(), end,
                ComparePredicate::Lt);
            Value projected = nested.create<BroadcastOp>(
                nestedLocation, blockedPredicate, valid);
            sourceTail = sourceTail
                             ? Value(nested.create<BinaryOp>(
                                   nestedLocation, blockedPredicate, sourceTail,
                                   projected, BinaryOperator::LogicalAnd))
                             : projected;
          }
          for (RootAccess access : accesses[component]) {
            LoadOp load = access.load;
            MakeRangeOp range = access.range;
            auto rootType = cast<FragmentType>(load.getResult().getType());
            FragmentType blockedRoot =
                replaceExtent(rootType, access.fragmentAxis, chunkExtent);
            SmallVector<Attribute> coordinateShape(
                range.getResult().getType().getShape().begin(),
                range.getResult().getType().getShape().end());
            coordinateShape[0] = chunkExtent;
            auto blockedCoordinate = FragmentType::get(
                reduce.getContext(), range.getResult().getType().getElementType(),
                ArrayAttr::get(reduce.getContext(), coordinateShape),
                range.getResult().getType().getAxisMaps(),
                range.getResult().getType().getValidity(),
                range.getResult().getType().getOwner());
            Value coordinate = nested.create<MakeRangeOp>(
                nestedLocation, blockedCoordinate, chunkStart, chunk.getResult(),
                range.getStep(), range.getSourceId(), range.getSourceAxis());
            if (!mapping.lookupOrNull(range.getResult())) {
              auto originalCoordinate = range.getResult().getType();
              auto replayCoordinate = FragmentType::get(
                  reduce.getContext(), originalCoordinate.getElementType(),
                  blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(),
                  originalCoordinate.getValidity(),
                  originalCoordinate.getOwner());
              Value mappedCoordinate = coordinate;
              if (replayCoordinate != blockedCoordinate) {
                FailureOr<ArrayAttr> reassociation =
                    inferReshapeReassociation(blockedCoordinate,
                                              replayCoordinate);
                if (failed(reassociation)) {
                  bodyFailed = true;
                  bodyFailure =
                      "reduction coordinate has no exact row-major reassociation";
                  return;
                }
                mappedCoordinate = nested.create<ReshapeOp>(
                    nestedLocation, replayCoordinate, coordinate,
                    *reassociation);
              }
              mapping.map(range.getResult(), mappedCoordinate);
            }
            Value end = nested.create<BroadcastOp>(nestedLocation,
                                                   blockedCoordinate, stop);
            auto coordinatePredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(),
                blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(),
                blockedCoordinate.getValidity(),
                blockedCoordinate.getOwner());
            Value coordinateValid = nested.create<CompareOp>(
                nestedLocation, coordinatePredicate, coordinate, end,
                ComparePredicate::Lt);
            auto rootPredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(), blockedRoot.getShape(),
                blockedRoot.getAxisMaps(), blockedRoot.getValidity(),
                blockedRoot.getOwner());
            Value valid = nested.create<BroadcastOp>(nestedLocation,
                                                     rootPredicate,
                                                     coordinateValid);
            if (load.getValid()) {
              FailureOr<Value> original = replayValue(
                  nested, nestedLocation, load.getValid(), plan.sourceId,
                  chunkExtent, mapping);
              if (failed(original)) {
                bodyFailed = true;
                bodyFailure = "could not replay source validity";
                return;
              }
              Value originalValid = *original;
              if (originalValid.getType() != rootPredicate)
                originalValid = nested.create<BroadcastOp>(
                    nestedLocation, rootPredicate, originalValid);
              valid = nested.create<BinaryOp>(nestedLocation, rootPredicate,
                                              valid, originalValid,
                                              BinaryOperator::LogicalAnd);
            }
            Value fill;
            if (load.getFill()) {
              FailureOr<Value> replayedFill = replayValue(
                  nested, nestedLocation, load.getFill(), plan.sourceId,
                  chunkExtent, mapping);
              if (failed(replayedFill)) {
                bodyFailed = true;
                bodyFailure = "could not replay source fill";
                return;
              }
              fill = *replayedFill;
              if (fill.getType() != blockedRoot)
                fill = nested.create<BroadcastOp>(nestedLocation, blockedRoot,
                                                  fill);
            } else {
              FailureOr<Value> zero = materializeZeroFragment(
                  nested, nestedLocation, blockedRoot);
              if (failed(zero)) {
                bodyFailed = true;
                bodyFailure = "source element type has no zero fill";
                return;
              }
              fill = *zero;
            }
            SmallVector<Value> coordinates(load.getCoordinates());
            FailureOr<Value> reducedCoordinate = replayValue(
                nested, nestedLocation,
                load.getCoordinates()[access.coordinateIndex], plan.sourceId,
                chunkExtent, mapping);
            if (failed(reducedCoordinate)) {
              bodyFailed = true;
              bodyFailure = "could not replay reduced source coordinate";
              return;
            }
            coordinates[access.coordinateIndex] = *reducedCoordinate;
            Value blockedLoad = nested.create<LoadOp>(
                nestedLocation, blockedRoot, load.getResource(), coordinates,
                valid, fill, load.getSourceAxes());
            mapping.map(load.getResult(), blockedLoad);
            if (!sourceTail)
              sourceTail = nested.create<BroadcastOp>(
                  nestedLocation, blockedPredicate, coordinateValid);
          }
          FailureOr<Value> replayed = replayValue(
              nested, nestedLocation, plan.source, plan.sourceId, chunkExtent,
              mapping);
          if (failed(replayed) || !sourceTail) {
            if (failed(replayed)) {
              bodyFailed = true;
              bodyFailure = "could not replay load-rooted pure producer graph";
              return;
            }
            sourceTail = nested.create<BroadcastOp>(
                nestedLocation, blockedPredicate, sharedTail);
          }
          Value identity = identities[component];
          if (identity.getType() != blockedSource)
            identity = nested.create<BroadcastOp>(nestedLocation, blockedSource,
                                                  identity);
          blockedSources.push_back(nested.create<SelectOp>(
              nestedLocation, blockedSource, sourceTail, *replayed, identity));
        }
        if (bodyFailed)
          return;

        if (vectorAccumulation) {
          Value combined = nested.create<BinaryOp>(
              nestedLocation, blockedSourceTypes.front(), carries.front(),
              blockedSources.front(), BinaryOperator::Add);
          nested.create<scf::YieldOp>(nestedLocation, combined);
          return;
        }

        SmallVector<Value> chunkInputs(blockedSources);
        chunkInputs.append(identities.begin(), identities.end());
        chunkInputs.append(captures.begin(), captures.end());
        OperationState state(nestedLocation, ReduceOp::getOperationName());
        state.addOperands(chunkInputs);
        state.addTypes(reduce.getResultTypes());
        state.addAttribute("axes", reduce->getAttr("axes"));
        state.addAttribute("source_count", reduce->getAttr("source_count"));
        state.addAttribute("identity_count", reduce->getAttr("identity_count"));
        state.addAttribute("capture_count", reduce->getAttr("capture_count"));
        state.addRegion();
        Operation *raw = nested.create(state);
        auto chunkReduce = cast<ReduceOp>(raw);
        if (Attribute origin = reduce->getAttr(originAttr))
          chunkReduce->setAttr(originAttr, origin);
        IRMapping regionMapping;
        reduce.getCombine().cloneInto(&chunkReduce.getCombine(), regionMapping);

        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(chunkReduce.getResults().begin(),
                                chunkReduce.getResults().end());
        combineArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, chunkReduce.getCombine(), combineArguments, bodyFailure);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      });
  if (Attribute origin = reduce->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  if (bodyFailed) {
    loop.erase();
    return reduce.emitOpError(
        "runtime reduction could not be materialized in the chunk loop: ")
           << bodyFailure;
  }

  SmallVector<Value> realizedResults(loop.getResults().begin(),
                                     loop.getResults().end());
  if (vectorAccumulation) {
    SmallVector<Value> finalInputs(realizedResults);
    finalInputs.append(identities.begin(), identities.end());
    OperationState state(location, ReduceOp::getOperationName());
    state.addOperands(finalInputs);
    state.addTypes(reduce.getResultTypes());
    state.addAttribute("axes", reduce->getAttr("axes"));
    state.addAttribute("source_count", reduce->getAttr("source_count"));
    state.addAttribute("identity_count", reduce->getAttr("identity_count"));
    state.addAttribute("capture_count", reduce->getAttr("capture_count"));
    state.addRegion();
    auto finalReduce = cast<ReduceOp>(builder.create(state));
    if (Attribute origin = reduce->getAttr(originAttr))
      finalReduce->setAttr(originAttr, origin);
    IRMapping regionMapping;
    reduce.getCombine().cloneInto(&finalReduce.getCombine(), regionMapping);
    realizedResults.assign(finalReduce.getResults().begin(),
                           finalReduce.getResults().end());
  }
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), realizedResults))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult realizeReduce(ReduceOp reduce, func::FuncOp kernel) {
  const bool required = requiresPhysicalRealization(reduce);
  auto unhandled = [&](const Twine &reason) -> LogicalResult {
    return required ? reduce.emitOpError()
                          << "cannot form a complete physical reduction: "
                          << reason
                    : success();
  };
  if (reduce.getAxes().size() != 1 || reduce.getSourceCount() == 0)
    return unhandled("requires one reduction axis and at least one source");
  if (hasSelectedSegmentExtent(reduce, kernel))
    return success();
  FailureOr<bool> staticPadding = realizeStaticPaddingReduce(reduce, kernel);
  if (failed(staticPadding))
    return failure();
  if (*staticPadding) {
    eraseDeadPhysicalValues(kernel);
    return success();
  }
  FailureOr<bool> fullCoverage = realizeFullCoverageReduce(reduce, kernel);
  if (failed(fullCoverage))
    return failure();
  if (*fullCoverage)
    return success();
  if (hasNonUnitFreeAxis(reduce) &&
      failed(bindReductionFreeAxes(reduce, kernel)))
    return failure();
  int64_t reductionAxis = reduce.getAxes().front();
  SmallVector<SourcePlan> sourcePlans;
  SmallVector<LoadOp> sourceLoads;
  SmallVector<MakeRangeOp> sourceRanges;
  SmallVector<unsigned> coordinateIndices;
  PhysicalExprAttr sourceExtent;
  bool hasDerivedSource = false;
  bool hasRuntimeSourceRange = false;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || reductionAxis < 0 ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return unhandled("source has no physical reduction axis");
    FailureOr<SourcePlan> plan = analyzeSource(source, reductionAxis);
    if (failed(plan)) {
      FailureOr<AxisMapAttr> mapping = axisMap(fragment, reductionAxis);
      llvm::SmallPtrSet<Operation *, 16> visited;
      if (failed(mapping) ||
          !isReplayableWithoutLoad(source, mapping->getSourceId(), visited))
        return unhandled(
            Twine("source is neither load-rooted nor replayable pure data on the reduction axis; producer=") +
            (source.getDefiningOp()
                 ? source.getDefiningOp()->getName().getStringRef()
                 : StringRef("block argument")));
      plan = SourcePlan{source, mapping->getSourceId(),
                        static_cast<unsigned>(reductionAxis), {}, {}, {}};
    }
    if (plan->ranges.empty()) {
      FailureOr<MakeRangeOp> authority =
          uniqueSourceRange(kernel, plan->sourceId);
      if (succeeded(authority))
        plan->ranges.push_back(*authority);
    }
    if (plan->roots.empty() && plan->ranges.empty())
      return reduce.emitOpError(
          "pure reduction source has no exact physical range authority");
    for (LoadOp root : plan->roots) {
      FailureOr<RootAccess> rootAccess =
          analyzeRoot(root, plan->reductionRange);
      if (failed(rootAccess))
        return unhandled(
            "load-rooted producer has no unit-step reduction coordinate");
      hasRuntimeSourceRange |=
          !isCompileTimeValue(rootAccess->range.getExtent());
    }
    PhysicalExprAttr extent =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (sourceExtent && sourceExtent != extent)
      return unhandled("reduction components disagree on physical extent");
    sourceExtent = extent;
    hasDerivedSource |=
        plan->roots.size() != 1 || plan->source != plan->roots.front().getResult();
    if (!hasDerivedSource) {
      RootAccess access =
          *analyzeRoot(plan->roots.front(), plan->reductionRange);
      Value identity = reduce.getInputs()[reduce.getSourceCount() +
                                          sourcePlans.size()];
      if (access.load.getValid() &&
          (!sameScalarValue(access.load.getFill(), identity) ||
           failed(scalarSource(access.load.getValid()))))
        hasDerivedSource = true;
      sourceLoads.push_back(access.load);
      sourceRanges.push_back(access.range);
      coordinateIndices.push_back(access.coordinateIndex);
    }
    sourcePlans.push_back(*plan);
  }
  if (!sourceExtent)
    return unhandled("reduction has no physical source extent");
  if (!isCompileTimeExtent(sourceExtent) || hasDerivedSource ||
      hasRuntimeSourceRange)
    return realizeRuntimeReduce(reduce, sourcePlans, kernel);

  PhysicalExprAttr blockExtent = nextPowerOfTwo(sourceExtent);
  OpBuilder builder(reduce);
  Value physicalExtent;
  if (blockExtent.getKind() ==
      static_cast<uint32_t>(PhysicalExprKind::Constant))
    physicalExtent = builder.create<arith::ConstantIndexOp>(
        reduce.getLoc(), blockExtent.getValue());
  else
    physicalExtent = builder.create<PhysicalExprOp>(
        reduce.getLoc(), builder.getIndexType(), blockExtent);

  SmallVector<Value> blockedSources;
  for (unsigned component = 0; component < reduce.getSourceCount(); ++component) {
    LoadOp load = sourceLoads[component];
    MakeRangeOp range = sourceRanges[component];
    auto sourceType = cast<FragmentType>(load.getResult().getType());
    FragmentType blockedSource =
        replaceExtent(sourceType, reductionAxis, blockExtent);
    SmallVector<Attribute> coordinateShape(
        range.getResult().getType().getShape().begin(),
        range.getResult().getType().getShape().end());
    coordinateShape[0] = blockExtent;
    auto blockedCoordinate = FragmentType::get(
        reduce.getContext(), range.getResult().getType().getElementType(),
        ArrayAttr::get(reduce.getContext(), coordinateShape),
        range.getResult().getType().getAxisMaps(),
        range.getResult().getType().getValidity(),
        range.getResult().getType().getOwner());
    Value coordinate = builder.create<MakeRangeOp>(
        reduce.getLoc(), blockedCoordinate, range.getStart(), physicalExtent,
        range.getStep(), range.getSourceId(), range.getSourceAxis());
    Value stop = builder.create<BinaryOp>(
        reduce.getLoc(), builder.getIndexType(), range.getStart(),
        range.getExtent(), BinaryOperator::Add);
    auto coordinatePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), blockedCoordinate.getShape(),
        blockedCoordinate.getAxisMaps(), blockedCoordinate.getValidity(),
        blockedCoordinate.getOwner());
    Value stopFragment =
        builder.create<BroadcastOp>(reduce.getLoc(), blockedCoordinate, stop);
    Value valid = builder.create<CompareOp>(
        reduce.getLoc(), coordinatePredicate, coordinate, stopFragment,
        ComparePredicate::Lt);
    auto sourcePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), blockedSource.getShape(),
        blockedSource.getAxisMaps(), blockedSource.getValidity(),
        blockedSource.getOwner());
    valid = builder.create<BroadcastOp>(reduce.getLoc(), sourcePredicate, valid);
    if (load.getValid()) {
      FailureOr<Value> scalar = scalarSource(load.getValid());
      Value original = builder.create<BroadcastOp>(reduce.getLoc(),
                                                    sourcePredicate, *scalar);
      valid = builder.create<BinaryOp>(reduce.getLoc(), sourcePredicate, valid,
                                       original, BinaryOperator::LogicalAnd);
    }
    Value identity =
        reduce.getInputs()[reduce.getSourceCount() + component];
    Value fill = identity;
    if (identity.getType() != blockedSource)
      fill = builder.create<BroadcastOp>(reduce.getLoc(), blockedSource, identity);
    SmallVector<Value> coordinates(load.getCoordinates());
    IRMapping coordinateMapping;
    coordinateMapping.map(range.getResult(), coordinate);
    FailureOr<Value> reducedCoordinate = replayValue(
        builder, reduce.getLoc(),
        load.getCoordinates()[coordinateIndices[component]],
        range.getSourceId(), blockExtent, coordinateMapping);
    if (failed(reducedCoordinate))
      return reduce.emitOpError(
          "could not replay translated full-coverage reduction coordinate");
    coordinates[coordinateIndices[component]] = *reducedCoordinate;
    blockedSources.push_back(builder.create<LoadOp>(
        reduce.getLoc(), blockedSource, load.getResource(), coordinates, valid,
        fill, load.getSourceAxes()));
  }

  SmallVector<Value> inputs(blockedSources);
  inputs.append(reduce.getInputs().drop_front(reduce.getSourceCount()).begin(),
                reduce.getInputs().drop_front(reduce.getSourceCount()).end());
  OperationState state(reduce.getLoc(), ReduceOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(reduce.getResultTypes());
  state.addAttribute("axes", reduce->getAttr("axes"));
  state.addAttribute("source_count", reduce->getAttr("source_count"));
  state.addAttribute("identity_count", reduce->getAttr("identity_count"));
  state.addAttribute("capture_count", reduce->getAttr("capture_count"));
  state.addRegion();
  Operation *raw = builder.create(state);
  auto replacement = cast<ReduceOp>(raw);
  replacement.getCombine().takeBody(reduce.getCombine());
  if (Attribute origin = reduce->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  for (LoadOp load : sourceLoads)
    if (load->getBlock() && load.getResult().use_empty())
      load.erase();
  return success();
}

} // namespace

LogicalResult decomposeMultiAxisReductions(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<ReduceOp> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions)
    if (reduce->getBlock() && reduce.getAxes().size() > 1 &&
        failed(decomposeMultiAxisReduce(reduce, kernel)))
      return failure();
  return success();
}

LogicalResult realizeReductionBlocking(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<ReduceOp> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions) {
    if (reduce.getAxes().size() > 1)
      return reduce.emitOpError(
          "reduction blocking requires prior multi-axis normalization");
    if (reduce->getBlock() && failed(realizeReduce(reduce, kernel)))
      return failure();
  }
  return success();
}

} // namespace intent::gpu

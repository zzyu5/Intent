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

void inheritRangeAuthority(Operation *target, MakeRangeOp source) {
  for (StringRef name :
       {originAttr, sourceSubregionAttr, worksetCoordinateRangeAttr})
    if (Attribute value = source->getAttr(name))
      target->setAttr(name, value);
}

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
  auto kernel = reduce->getParentOfType<func::FuncOp>();
  if (!kernel)
    return true;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (int64_t axis : reduce.getAxes()) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
        continue;
      PhysicalAxisRealizationFact fact =
          analysis.axisRealization(source, static_cast<unsigned>(axis));
      if (fact.constructionScalarSeed ||
          (fact.isExact() && !fact.physicalized))
        return true;
    }
  }
  return false;
}

bool hasNonUnitFreeAxis(ReduceOp reduce);

LogicalResult realizeConstructionScalarReductionAxes(ReduceOp reduce,
                                                      func::FuncOp kernel) {
  if (hasNonUnitFreeAxis(reduce))
    return success();
  SmallVector<std::pair<Value, unsigned>> pending;
  PhysicalProgramAnalysis analysis(kernel);
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (int64_t rawAxis : reduce.getAxes()) {
      if (rawAxis < 0 ||
          rawAxis >= static_cast<int64_t>(fragment.getShape().size()))
        continue;
      unsigned axis = static_cast<unsigned>(rawAxis);
      PhysicalAxisRealizationFact fact = analysis.axisRealization(source, axis);
      if (fact.constructionScalarSeed &&
          llvm::none_of(fact.roots, [](MakeRangeOp range) {
            return range->hasAttr(sourceSubregionAttr);
          }))
        pending.emplace_back(source, axis);
    }
  }
  for (auto [source, axis] : pending)
    if (failed(realizeFullCoverageDimension(kernel, source, axis)))
      return reduce.emitOpError(
          "construction scalar reduction axis has no exact full-coverage realization");
  return success();
}

MakeRangeOp sourceRange(Value value) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return {};
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value);
  FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
  return succeeded(range) ? *range : MakeRangeOp();
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

FailureOr<Value> predicateForReductionSource(OpBuilder &builder,
                                             Location location, Value predicate,
                                             FragmentType source,
                                             unsigned reductionAxis) {
  if (reductionAxis >= source.getAxisMaps().size())
    return failure();
  auto mapping = cast<AxisMapAttr>(source.getAxisMaps()[reductionAxis]);
  return projectPredicateToFragment(
      builder, location, predicate, source,
      PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()});
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
    std::optional<unsigned> projectedAxis,
    PhysicalSourceAxis reductionSource,
    ArrayRef<MakeRangeOp> selectedRanges,
    PhysicalExprAttr logicalExtent, PhysicalExprAttr physicalExtent,
    Value physicalExtentValue, IRMapping &mapping,
    SmallVectorImpl<Value> &tailPredicates) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment)
    return value;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalRangeAxisFact selected =
      PhysicalProgramAnalysis(kernel).rangeAxes(value, selectedRanges);
  if (!selected.isExact())
    return failure();
  if (selected.fragmentAxes.size() > 1)
    return failure();
  if (!selected.fragmentAxes.empty() && projectedAxis &&
      selected.fragmentAxes.front() != *projectedAxis)
    return failure();
  if (!projectedAxis && !selected.fragmentAxes.empty())
    projectedAxis = selected.fragmentAxes.front();
  if (!projectedAxis)
    return value;
  unsigned reductionAxis = *projectedAxis;
  if (reductionAxis >= fragment.getShape().size())
    return failure();
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
        range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(padded, range);
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
    auto paddedType = replaceExtent(fragment, reductionAxis, physicalExtent);
    if (!input) {
      auto padded = builder.create<BroadcastOp>(location, paddedType,
                                                broadcast.getValue());
      if (Attribute origin = broadcast->getAttr(originAttr))
        padded->setAttr(originAttr, origin);
      mapping.map(value, padded.getResult());
      return padded.getResult();
    }
    BroadcastProjection projection =
        queryAxisProjection(input, fragment);
    if (!projection.isExact() ||
        reductionAxis >= projection.targetToSource.size()) {
      InFlightDiagnostic diagnostic = broadcast.emitOpError(
          "padded broadcast has no exact source-axis projection");
      diagnostic << "; input=" << broadcast.getValue().getType()
                 << "; result=" << fragment
                 << "; reduction_axis=" << reductionAxis;
      return failure();
    }
    Value replayed = broadcast.getValue();
    if (std::optional<unsigned> inputAxis =
            projection.targetToSource[reductionAxis]) {
      auto inputExtent =
          cast<PhysicalExprAttr>(input.getShape()[*inputAxis]);
      bool expandsSingleton =
          inputExtent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant) &&
          inputExtent.getValue() == 1 &&
          input.getShape()[*inputAxis] != fragment.getShape()[reductionAxis];
      if (!expandsSingleton) {
        auto inputMapping =
            cast<AxisMapAttr>(input.getAxisMaps()[*inputAxis]);
        FailureOr<Value> replacement = clonePaddedProducer(
            builder, location, broadcast.getValue(), *inputAxis,
            PhysicalSourceAxis{inputMapping.getSourceId(),
                               inputMapping.getSourceAxis(),
                               inputMapping.getDerived()},
            selectedRanges, logicalExtent, physicalExtent,
            physicalExtentValue, mapping, tailPredicates);
        if (failed(replacement))
          return failure();
        replayed = *replacement;
      }
    }
    auto padded =
        builder.create<BroadcastOp>(location, paddedType, replayed);
    if (Attribute origin = broadcast->getAttr(originAttr))
      padded->setAttr(originAttr, origin);
    mapping.map(value, padded.getResult());
    return padded.getResult();
  }

  if (auto load = dyn_cast<LoadOp>(producer)) {
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, coordinate, std::nullopt, reductionSource,
          selectedRanges,
          logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      coordinates.push_back(*replayed);
    }
    Value valid;
    if (load.getValid()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, load.getValid(), reductionAxis, reductionSource,
          selectedRanges,
          logicalExtent,
          physicalExtent,
          physicalExtentValue, mapping, tailPredicates);
      if (failed(replayed))
        return failure();
      valid = *replayed;
    }
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> replayed = clonePaddedProducer(
          builder, location, load.getFill(), reductionAxis, reductionSource,
          selectedRanges,
          logicalExtent,
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
    std::optional<unsigned> operandAxis;
    if (auto operandType = dyn_cast<FragmentType>(operand.getType())) {
      BroadcastProjection projection = queryAxisProjection(operandType, fragment);
      if (projection.isExact() &&
          reductionAxis < projection.targetToSource.size())
        operandAxis = projection.targetToSource[reductionAxis];
      if (operandAxis) {
        auto operandExtent =
            cast<PhysicalExprAttr>(operandType.getShape()[*operandAxis]);
        if (operandExtent.getKind() ==
                static_cast<uint32_t>(PhysicalExprKind::Constant) &&
            operandExtent.getValue() == 1 &&
            operandType.getShape()[*operandAxis] !=
                fragment.getShape()[reductionAxis])
          operandAxis.reset();
      }
    }
    FailureOr<Value> replayed = clonePaddedProducer(
        builder, location, operand, operandAxis, reductionSource, selectedRanges,
        logicalExtent,
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

bool isReplayableWithoutLoad(Value value, PhysicalSourceAxis source) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return false;
  if (!queryFragmentAxis(value.getType(), source).isExact())
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  return analysis
      .replayability(value, source, PhysicalReplayScope::ValueGraph,
                     /*allowAccesses=*/false)
      .isReplayable();
}

struct SourcePlan {
  Value source;
  PhysicalSourceAxis sourceIdentity;
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

FailureOr<std::optional<RootAccess>>
analyzeRoot(LoadOp load, ArrayRef<MakeRangeOp> reductionRanges,
            unsigned preferredAxis) {
  auto fragment = dyn_cast<FragmentType>(load.getResult().getType());
  if (!fragment || reductionRanges.empty())
    return std::optional<RootAccess>();
  auto kernel = load->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  struct AxisRelation {
    unsigned axis;
    MakeRangeOp authority;
  };
  SmallVector<AxisRelation> resultAxes;
  for (unsigned axis = 0; axis < fragment.getShape().size(); ++axis) {
    PhysicalRangeFact fact = analysis.axisRanges(load.getResult(), axis);
    if (fact.roots.empty())
      continue;
    SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
    combined.append(reductionRanges.begin(), reductionRanges.end());
    PhysicalLockstepTraversalFact relation = analysis.lockstepRanges(combined);
    if (relation.isExact())
      resultAxes.push_back({axis, fact.roots.front()});
  }
  if (resultAxes.empty())
    return std::optional<RootAccess>();
  if (resultAxes.size() > 1) {
    llvm::erase_if(resultAxes,
                   [&](const AxisRelation &relation) {
                     return relation.axis != preferredAxis;
                   });
  }
  if (resultAxes.size() != 1)
    return load.emitOpError(
               "reduction traversal has no unique root-load fragment axis"),
           failure();
  unsigned reductionAxis = resultAxes.front().axis;
  MakeRangeOp resultAuthority = resultAxes.front().authority;
  SmallVector<std::pair<unsigned, unsigned>, 2> occurrences;
  for (auto [index, value] : llvm::enumerate(load.getCoordinates())) {
    auto type = dyn_cast<FragmentType>(value.getType());
    if (!type)
      continue;
    for (unsigned axis = 0; axis < type.getShape().size(); ++axis) {
      PhysicalRangeFact fact = analysis.axisRanges(value, axis);
      if (fact.roots.empty())
        continue;
      SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
      combined.push_back(resultAuthority);
      if (analysis.lockstepRanges(combined).isExact())
        occurrences.emplace_back(index, axis);
    }
  }
  SmallVector<std::pair<unsigned, unsigned>, 2> alignedOccurrences;
  llvm::copy_if(occurrences, std::back_inserter(alignedOccurrences),
                [&](const auto &occurrence) {
                  return occurrence.second == reductionAxis;
                });
  if (alignedOccurrences.size() == 1)
    occurrences = std::move(alignedOccurrences);
  if (occurrences.size() != 1) {
    InFlightDiagnostic diagnostic = load.emitOpError(
        "reduction source provenance is absent from load coordinates");
    diagnostic << "; reduction_range=" << resultAuthority.getResult().getType()
               << ", fragment_axis=" << reductionAxis
               << ", matching_occurrences=" << occurrences.size();
    for (Value value : load.getCoordinates())
      diagnostic << ", coordinate=" << value.getType();
    return failure();
  }
  if (!isUnitStep(resultAuthority.getStep()))
    return load.emitOpError("reduction load range is not unit-step");
  return std::optional<RootAccess>(RootAccess{
      load, resultAuthority, occurrences.front().first, reductionAxis});
}

FailureOr<SourcePlan> analyzeSource(Value source, unsigned reductionAxis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || reductionAxis >= fragment.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, reductionAxis);
  if (failed(mapping))
    return failure();
  SourcePlan plan{source, sourceAxisIdentity(*mapping), reductionAxis, {}, {}, {}};
  auto kernel = source.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.axisRanges(source, reductionAxis);
  PhysicalLockstepTraversalFact relation = analysis.lockstepRanges(fact.roots);
  if (!relation.isExact())
    return failure();
  plan.reductionRange = relation.authority;
  for (MakeRangeOp range : fact.roots)
    if (!llvm::is_contained(plan.ranges, range))
      plan.ranges.push_back(range);
  // Predicates, gathers and address arithmetic can carry another occurrence of
  // the same traversal without defining the result fragment axis.  Replay must
  // map those coordinates from the same lockstep authority instead of meeting
  // an unmapped make_range inside the chunk loop.
  PhysicalRangeFact graphRanges = analysis.sourceRanges(source);
  for (MakeRangeOp range : graphRanges.roots) {
    if (llvm::is_contained(plan.ranges, range))
      continue;
    SmallVector<MakeRangeOp> combined(plan.ranges.begin(), plan.ranges.end());
    combined.push_back(range);
    if (analysis.lockstepRanges(combined).isExact())
      plan.ranges.push_back(range);
  }
  for (Operation *access : fact.accesses) {
    auto load = dyn_cast<LoadOp>(access);
    if (!load || llvm::is_contained(plan.roots, load))
      continue;
    FailureOr<std::optional<RootAccess>> root =
        analyzeRoot(load, plan.ranges, reductionAxis);
    if (failed(root))
      return failure();
    if (*root)
      plan.roots.push_back(load);
  }
  if (plan.roots.empty() && plan.ranges.empty())
    return failure();
  return plan;
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
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
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
          "physical parameter name is reused with a different role or candidate domain")
          << "; name=" << name << "; existing_role=" << schema.getRole()
          << "; requested_role=" << static_cast<uint32_t>(role)
          << "; existing_candidates=" << schema.getCandidates()
          << "; requested_candidates=" << expectedCandidates;
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
  ParameterOp parameter;
  kernel.walk([&](ParameterOp candidate) {
    if (candidate.getParameter().getName().getValue() == name)
      parameter = candidate;
  });
  auto coverage = parameter
                      ? parameter->getAttrOfType<IntegerAttr>(
                            coverageDimensionAttr)
                      : IntegerAttr();
  if (!parameter || !coverage || coverage.getInt() <= 0)
    return failure();
  uint64_t dimension = static_cast<uint64_t>(coverage.getInt());
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
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (binding.source && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    schema.getName().getValue());
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

FailureOr<bool> realizeStaticPaddingReduce(ReduceOp reduce,
                                           func::FuncOp kernel) {
  if (reduce.getAxes().size() != 1 || reduce.getSourceCount() == 0)
    return false;
  int64_t reductionAxis = reduce.getAxes().front();
  if (reductionAxis < 0)
    return false;

  PhysicalExprAttr logicalExtent;
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
  }
  if (!logicalExtent || logicalExtent.getValue() <= 0)
    return false;
  PhysicalExprAttr physicalExtent = nextPowerOfTwo(logicalExtent);
  if (physicalExtent == logicalExtent)
    return false;

  SmallVector<SmallVector<MakeRangeOp>> componentRanges;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = cast<FragmentType>(source.getType());
    PhysicalProgramAnalysis analysis(kernel);
    PhysicalRangeFact reductionRanges =
        analysis.axisRanges(source, static_cast<unsigned>(reductionAxis));
    FailureOr<MakeRangeOp> authority =
        queryExactLogicalRange(reductionRanges);
    if (failed(authority) && reductionRanges.roots.empty()) {
      auto mapping = cast<AxisMapAttr>(
          fragment.getAxisMaps()[static_cast<unsigned>(reductionAxis)]);
      PhysicalSourceAxis reductionSource{mapping.getSourceId(),
                                         mapping.getSourceAxis(),
                                         mapping.getDerived()};
      PhysicalRangeFact programRanges =
          analysis.programRanges(reductionSource);
      PhysicalRangeFact dimensionRanges;
      dimensionRanges.state = PhysicalFactState::Exact;
      for (MakeRangeOp range : programRanges.roots) {
        FailureOr<int64_t> dimension = queryRangeDimension(range);
        if (succeeded(dimension) &&
            *dimension == mapping.getDimensionId())
          dimensionRanges.roots.push_back(range);
      }
      if (dimensionRanges.roots.empty())
        dimensionRanges.state = PhysicalFactState::Unknown;
      else if (dimensionRanges.roots.size() > 1)
        dimensionRanges.state = PhysicalFactState::Ambiguous;
      dimensionRanges.unitStep =
          !dimensionRanges.roots.empty() &&
          llvm::all_of(dimensionRanges.roots, isUnitStepRange);
      authority = queryExactLogicalRange(dimensionRanges);
      if (succeeded(authority))
        reductionRanges = std::move(dimensionRanges);
    }
    SmallVector<MakeRangeOp> ranges(reductionRanges.roots.begin(),
                                    reductionRanges.roots.end());
    llvm::sort(ranges, [](MakeRangeOp lhs, MakeRangeOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    if (failed(authority)) {
      InFlightDiagnostic diagnostic = reduce.emitOpError(
          "static non-power-of-two reduction axis has no exact range authority");
      diagnostic << "; range_state="
                 << static_cast<unsigned>(reductionRanges.state)
                 << ", roots=" << reductionRanges.roots.size()
                 << ", blockers=" << reductionRanges.blockers.size()
                 << ", source_type=" << source.getType();
      if (Operation *definition = source.getDefiningOp()) {
        diagnostic << ", source_def=" << definition->getName();
        for (Value operand : definition->getOperands())
          if (Operation *operandDefinition = operand.getDefiningOp())
            diagnostic << ", operand_def=" << operandDefinition->getName();
      }
      for (MakeRangeOp root : reductionRanges.roots) {
        FailureOr<int64_t> dimension = queryRangeDimension(root);
        diagnostic << ", root=(source_id=" << root.getSourceId()
                   << ",source_axis=" << root.getSourceAxis()
                   << ",dimension="
                   << (succeeded(dimension) ? *dimension : -1) << ")";
      }
      for (Operation *blocker : reductionRanges.blockers)
        diagnostic << ", blocker=" << blocker->getName();
      return failure();
    }
    componentRanges.push_back(std::move(ranges));
  }
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
                                       sourceMap.getSourceAxis(),
                                       sourceMap.getDerived()};
    PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
        reduce.getInputs()[component], reductionSource,
        PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true);
    if (!replay.isReplayable())
      return reduce.emitOpError(
                 "static reduction producer has no exact shared replay fact"),
             failure();
    IRMapping mapping;
    SmallVector<Value> tailPredicates;
    FailureOr<Value> source = clonePaddedProducer(
        builder, reduce.getLoc(), reduce.getInputs()[component],
        static_cast<unsigned>(reductionAxis), reductionSource,
        componentRanges[component], logicalExtent, physicalExtent,
        physicalExtentValue, mapping,
        tailPredicates);
    if (failed(source) || tailPredicates.empty()) {
      PhysicalRangeAxisFact selected =
          PhysicalProgramAnalysis(kernel).rangeAxes(
              reduce.getInputs()[component], componentRanges[component]);
      InFlightDiagnostic diagnostic = reduce.emitOpError(
          "static reduction producer cannot be replayed over its padded extent");
      diagnostic << "; selected_state=" << static_cast<unsigned>(selected.state)
                 << ", selected_axes=" << selected.fragmentAxes.size()
                 << ", selected_ranges=" << componentRanges[component].size()
                 << ", replay_failed=" << failed(source)
                 << ", tail_count=" << tailPredicates.size();
      return failure();
    }
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
  SmallVector<PhysicalSourceAxis> sourceAxes;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return false;
    FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, reductionAxis);
    if (failed(mapping))
      return false;
    PhysicalExprAttr current =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (extent && extent != current)
      return false;
    extent = current;
    sourceAxes.push_back(sourceAxisIdentity(*mapping));
  }
  if (!extent || extent.getKind() !=
                     static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return false;
  llvm::DenseMap<PhysicalSourceAxis, MakeRangeOp> ranges;
  PhysicalProgramAnalysis analysis(kernel);
  for (auto [source, sourceAxis] : llvm::zip(
           reduce.getInputs().take_front(reduce.getSourceCount()), sourceAxes)) {
    PhysicalRangeFact fact = analysis.axisRanges(source, reductionAxis);
    for (MakeRangeOp candidate : fact.roots) {
      auto found = ranges.find(sourceAxis);
      if (found != ranges.end() &&
          !sameLogicalRange(found->second, candidate))
        return false;
      ranges[sourceAxis] = candidate;
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
    FailureOr<int64_t> sourceDimension = queryRangeDimension(range);
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
    auto found = ranges.find(sourceAxes[component]);
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
      PhysicalAxisRealizationFact realization =
          PhysicalProgramAnalysis(kernel).axisRealization(source, axis);
      if (isCompileTimeExtent(extent) &&
          !realization.constructionScalarSeed)
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      if (realization.constructionScalarSeed ||
          extent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Dimension)) {
        if (mapping.getDimensionId() <= 0)
          return reduce.emitOpError(
              "reduction free axis has no logical dimension authority");
        pending.emplace_back(
            source,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
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
             << "; dimension=" << dimension << "; source=" << source.getType()
             << "; source_id=" << sourceAxis.sourceId
             << "; source_axis=" << sourceAxis.sourceAxis;
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
          return sameLogicalRange(authority, range);
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

  if (reduce.getSourceCount() != reduce.getResults().size())
    return reduce.emitOpError(
        "physical reduction needs one result schema per source component");
  llvm::SmallDenseSet<int64_t> reducedAxes(reduce.getAxes().begin(),
                                           reduce.getAxes().end());
  for (auto [source, result] : llvm::zip_equal(
           reduce.getInputs().take_front(reduce.getSourceCount()),
           reduce.getResults())) {
    auto sourceType = dyn_cast<FragmentType>(source.getType());
    if (!sourceType)
      continue;
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    for (auto [axis, extent] : llvm::enumerate(sourceType.getShape())) {
      if (reducedAxes.contains(static_cast<int64_t>(axis)))
        continue;
      shape.push_back(extent);
      auto mapping = cast<AxisMapAttr>(sourceType.getAxisMaps()[axis]);
      mappings.push_back(AxisMapAttr::get(
          reduce.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), static_cast<uint32_t>(mappings.size()),
          mapping.getDerived()));
    }
    Type element = result.getType();
    if (auto current = dyn_cast<FragmentType>(element))
      element = current.getElementType();
    if (shape.empty()) {
      result.setType(element);
      continue;
    }
    result.setType(FragmentType::get(
        reduce.getContext(), element, ArrayAttr::get(reduce.getContext(), shape),
        ArrayAttr::get(reduce.getContext(), mappings), sourceType.getValidity(),
        sourceType.getOwner()));
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
        mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
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
          fragment ? queryAxisMap(fragment, outerAxis)
                   : FailureOr<AxisMapAttr>(failure());
      if (!fragment || failed(mapping) ||
          !isReplayableWithoutLoad(source, sourceAxisIdentity(*mapping)))
        return reduce.emitOpError()
               << "multi-axis reduction outer axis is neither load-rooted nor a replayable pure source; source type="
               << source.getType() << ", producer="
               << (source.getDefiningOp()
                       ? source.getDefiningOp()->getName().getStringRef()
                       : StringRef("block argument"));
      plan = SourcePlan{source, sourceAxisIdentity(*mapping), outerAxis, {}, {},
                        {}};
    }
    SmallVector<RootAccess> roots;
    for (LoadOp load : plan->roots) {
      FailureOr<std::optional<RootAccess>> access =
          analyzeRoot(load, plan->ranges, plan->reductionAxis);
      if (failed(access) || !*access)
        return reduce.emitOpError(
            "multi-axis reduction outer axis has no unit-step load coordinate");
      roots.push_back(**access);
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
          !sameScalarValue(access.range.getLogicalStart(),
                           master->range.getLogicalStart()) ||
          !sameScalarValue(access.range.getLogicalStop(),
                           master->range.getLogicalStop()) ||
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
  bool bodyFailed = false;
  std::string failureReason;
  auto loop = builder.create<scf::ForOp>(
      location, master->range.getLogicalStart(),
      master->range.getLogicalStop(), master->range.getStep(), identities,
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
            FailureOr<Value> reducedCoordinate = materializeReplayedValue(
                nested, nestedLocation,
                load.getCoordinates()[access.coordinateIndex],
                plan.sourceIdentity, unitExtent, mapping);
            if (failed(reducedCoordinate)) {
              bodyFailed = true;
              failureReason =
                  "reduced source coordinate could not be cloned into the outer loop";
              return;
            }
            coordinates[access.coordinateIndex] = *reducedCoordinate;
            using ReplayAxis =
                std::tuple<PhysicalSourceAxis, PhysicalExprAttr, Value, Value>;
            SmallVector<ReplayAxis> replayAxes{{
                plan.sourceIdentity, unitExtent, access.range.getResult(),
                coordinate}};
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
                    range.getExtent(), range.getStep(), range.getLogicalStart(),
                    range.getLogicalStop(), range.getSourceId(),
                    range.getSourceAxis(), range.getDerived());
                inheritRangeAuthority(clone, range);
                mapping.map(range.getResult(), clone.getResult());
              }
              auto rangeType = cast<FragmentType>(range.getResult().getType());
              replayAxes.emplace_back(
                  sourceAxisIdentity(range),
                  cast<PhysicalExprAttr>(rangeType.getShape()[0]),
                  range.getResult(), mapping.lookupOrNull(range.getResult()));
              FailureOr<Value> replayedCoordinate = materializeReplayedValue(
                  nested, nestedLocation, original, sourceAxisIdentity(range),
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
              for (auto [source, extent, originalRange, replacementRange] :
                   replayAxes) {
                IRMapping axisMapping;
                axisMapping.map(originalRange, replacementRange);
                FailureOr<Value> replayed = materializeReplayedValue(
                    nested, nestedLocation, valid, source, extent,
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
              for (auto [source, extent, originalRange, replacementRange] :
                   replayAxes) {
                IRMapping axisMapping;
                axisMapping.map(originalRange, replacementRange);
                FailureOr<Value> replayed = materializeReplayedValue(
                    nested, nestedLocation, fill, source, extent,
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
          FailureOr<Value> replayed = materializeReplayedValue(
              nested, nestedLocation, plan.source, plan.sourceIdentity,
              unitExtent, mapping);
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
          SmallVector<Attribute> reassociation;
          unsigned resultAxis = 0;
          for (unsigned sourceAxis = 0;
               sourceAxis < sliced.getShape().size(); ++sourceAxis) {
            SmallVector<int64_t> resultAxes;
            if (sourceAxis != outerAxis)
              resultAxes.push_back(resultAxis++);
            reassociation.push_back(ReshapeGroupAttr::get(
                reduce.getContext(),
                DenseI64ArrayAttr::get(
                    reduce.getContext(),
                    {static_cast<int64_t>(sourceAxis)}),
                DenseI64ArrayAttr::get(reduce.getContext(), resultAxes)));
          }
          innerSources.push_back(nested.create<ReshapeOp>(
              nestedLocation, squeezed, *replayed,
              ArrayAttr::get(reduce.getContext(), reassociation)));
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
                PhysicalSourceAttr::get(builder.getContext(),
                                        master->range.getSourceId(),
                                        master->range.getSourceAxis(),
                                        master->range.getDerived()));
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
      FailureOr<std::optional<RootAccess>> access =
          analyzeRoot(load, plan.ranges, plan.reductionAxis);
      if (failed(access) || !*access)
        return reduce.emitOpError(
            "load-rooted producer has no unit-step reduction coordinate");
      roots.push_back(**access);
    }
    SmallVector<MakeRangeOp> componentRanges;
    for (RootAccess access : roots)
      if (!llvm::is_contained(componentRanges, access.range))
        componentRanges.push_back(access.range);
    for (MakeRangeOp range : plan.ranges)
      if (!llvm::is_contained(componentRanges, range))
        componentRanges.push_back(range);
    PhysicalLockstepTraversalFact traversal =
        PhysicalProgramAnalysis(kernel).lockstepRanges(componentRanges);
    if (!traversal.isExact())
      return reduce.emitOpError(
          "one reduction component has no exact lockstep range authority");
    accesses.push_back(std::move(roots));
    traversalRanges.push_back(traversal.authority);
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
    if (!sameLogicalRange(firstRange, range) || failed(end) ||
        !samePhysicalScalarExpression(*firstEnd, *end))
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
        ("REDUCE_CHUNK_" + Twine(sourcePlans.front().sourceIdentity.sourceId) +
         "_A" + Twine(sourcePlans.front().sourceIdentity.sourceAxis) +
         (sourcePlans.front().sourceIdentity.derived ? "_DERIVED" : ""))
            .str();
    SmallVector<int64_t> candidates{8, 16, 32, 64, 128,
                                    256, 512, 1024, 2048, 4096};
    chunk = getOrCreateParameter(kernel, name, ParameterRole::Reduction,
                                 candidates);
    if (chunk)
      if (FailureOr<int64_t> dimension = queryRangeDimension(firstRange);
          succeeded(dimension))
        chunk->setAttr(dimensionAttr,
                       IntegerAttr::get(IntegerType::get(chunk.getContext(), 64),
                                        *dimension));
  }
  if (!chunk)
    return reduce.emitOpError(
               "reduction blocking has no unique physical parameter relation")
           << "; source=" << sourcePlans.front().source.getType()
           << "; axis=" << sourcePlans.front().reductionAxis;
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
            firstRange.getStep(), firstRange.getLogicalStart(),
            firstRange.getLogicalStop(), firstRange.getSourceId(),
            firstRange.getSourceAxis(), firstRange.getDerived());
        inheritRangeAuthority(masterCoordinate.getDefiningOp(), firstRange);
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
          ReplayMaterializationOptions replayOptions;
          replayOptions.traversalRanges = plan.ranges;
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
                chunk.getResult(), range.getStep(), range.getLogicalStart(),
                range.getLogicalStop(), range.getSourceId(),
                range.getSourceAxis(), range.getDerived());
            inheritRangeAuthority(coordinate, range);
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
                range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
                range.getSourceId(), range.getSourceAxis(), range.getDerived());
            inheritRangeAuthority(coordinate.getDefiningOp(), range);
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
              FailureOr<Value> original = materializeReplayedValue(
                  nested, nestedLocation, load.getValid(), plan.sourceIdentity,
                  chunkExtent, mapping, replayOptions);
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
              FailureOr<Value> replayedFill = materializeReplayedValue(
                  nested, nestedLocation, load.getFill(), plan.sourceIdentity,
                  chunkExtent, mapping, replayOptions);
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
            FailureOr<Value> reducedCoordinate = materializeReplayedValue(
                nested, nestedLocation,
                load.getCoordinates()[access.coordinateIndex],
                plan.sourceIdentity, chunkExtent, mapping, replayOptions);
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
          FailureOr<Value> replayed = materializeReplayedValue(
              nested, nestedLocation, plan.source, plan.sourceIdentity,
              chunkExtent, mapping, replayOptions);
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
  if (failed(bindReductionFreeAxes(reduce, kernel)))
    return failure();
  if (failed(realizeConstructionScalarReductionAxes(reduce, kernel)))
    return failure();
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
  // A fixed fragment already gives the first-class reduction a complete
  // physical axis. Replaying its producer graph into another chunk loop would
  // duplicate structured loop carries without adding a physical decision.
  if (!required)
    return success();
  FailureOr<bool> fullCoverage = realizeFullCoverageReduce(reduce, kernel);
  if (failed(fullCoverage))
    return failure();
  if (*fullCoverage)
    return success();
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
      FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, reductionAxis);
      bool replayable = succeeded(mapping) &&
                        isReplayableWithoutLoad(
                            source, sourceAxisIdentity(*mapping));
      if (failed(mapping) || !replayable) {
        PhysicalRangeFact ranges =
            PhysicalProgramAnalysis(kernel).axisRanges(source, reductionAxis);
        InFlightDiagnostic diagnostic = reduce.emitOpError(
            "cannot form a complete physical reduction: source is neither load-rooted nor replayable pure data on the reduction axis");
        diagnostic << "; producer="
                   << (source.getDefiningOp()
                           ? source.getDefiningOp()->getName().getStringRef()
                           : StringRef("block argument"))
                   << ", range_state="
                   << static_cast<unsigned>(ranges.state)
                   << ", roots=" << ranges.roots.size()
                   << ", accesses=" << ranges.accesses.size()
                   << ", blockers=" << ranges.blockers.size();
        for (Operation *blocker : ranges.blockers)
          diagnostic << ", blocker=" << blocker->getName();
        return failure();
      }
      plan = SourcePlan{source, sourceAxisIdentity(*mapping),
                        static_cast<unsigned>(reductionAxis), {}, {}, {}};
    }
    if (plan->ranges.empty()) {
      PhysicalProgramAnalysis analysis(kernel);
      FailureOr<MakeRangeOp> authority = queryExactLogicalRange(
          analysis.programRanges(plan->sourceIdentity));
      if (succeeded(authority))
        plan->ranges.push_back(*authority);
    }
    if (plan->roots.empty() && plan->ranges.empty())
      return reduce.emitOpError(
          "pure reduction source has no exact physical range authority");
    for (LoadOp root : plan->roots) {
      FailureOr<std::optional<RootAccess>> rootAccess =
          analyzeRoot(root, plan->ranges, plan->reductionAxis);
      if (failed(rootAccess) || !*rootAccess)
        return unhandled(
            "load-rooted producer has no unit-step reduction coordinate");
      hasRuntimeSourceRange |=
          !isCompileTimeValue((**rootAccess).range.getExtent());
    }
    PhysicalExprAttr extent =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (sourceExtent && sourceExtent != extent)
      return unhandled("reduction components disagree on physical extent");
    sourceExtent = extent;
    hasDerivedSource |=
        plan->roots.size() != 1 || plan->source != plan->roots.front().getResult();
    if (!hasDerivedSource) {
      FailureOr<std::optional<RootAccess>> analyzed = analyzeRoot(
          plan->roots.front(), plan->ranges, plan->reductionAxis);
      if (failed(analyzed) || !*analyzed)
        return unhandled(
            "load-rooted producer has no unit-step reduction coordinate");
      RootAccess access = **analyzed;
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
        range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    inheritRangeAuthority(coordinate.getDefiningOp(), range);
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
    FailureOr<Value> reducedCoordinate = materializeReplayedValue(
        builder, reduce.getLoc(),
        load.getCoordinates()[coordinateIndices[component]],
        sourceAxisIdentity(range), blockExtent, coordinateMapping);
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
  auto remainingExcessAxes = [&]() {
    uint64_t result = 0;
    kernel.walk([&](ReduceOp reduce) {
      if (reduce.getAxes().size() > 1)
        result += reduce.getAxes().size() - 1;
    });
    return result;
  };
  uint64_t previous = remainingExcessAxes();
  while (previous != 0) {
    SmallVector<ReduceOp> reductions;
    kernel.walk([&](ReduceOp reduce) {
      if (reduce.getAxes().size() > 1)
        reductions.push_back(reduce);
    });
    for (ReduceOp reduce : reductions)
      if (reduce->getBlock() &&
          failed(decomposeMultiAxisReduce(reduce, kernel)))
        return failure();
    uint64_t current = remainingExcessAxes();
    if (current >= previous)
      return kernel.emitError(
          "multi-axis reduction normalization made no structural progress");
    previous = current;
  }
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

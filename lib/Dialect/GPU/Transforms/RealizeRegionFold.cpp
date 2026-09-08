#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "OnlineSummary.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"

#include <functional>
#include <memory>

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr parameterExtent(ParameterAttr parameter) {
  return PhysicalExprAttr::get(
      parameter.getContext(),
      static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      parameter.getName(), ArrayAttr::get(parameter.getContext(), {}));
}

FragmentType replaceSliceAxis(FragmentType source, unsigned axis,
                              PhysicalExprAttr extent,
                              AxisMapAttr segmentMapping) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  SmallVector<Attribute> mappings(source.getAxisMaps().begin(),
                                  source.getAxisMaps().end());
  shape[axis] = extent;
  mappings[axis] = AxisMapAttr::get(
      source.getContext(), segmentMapping.getSourceId(),
      segmentMapping.getSourceAxis(), segmentMapping.getDimensionId(), axis,
      segmentMapping.getDerived());
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape),
      ArrayAttr::get(source.getContext(), mappings), source.getValidity(),
      source.getOwner());
}

FragmentType predicateType(FragmentType source) {
  return FragmentType::get(
      source.getContext(), IntegerType::get(source.getContext(), 1),
      source.getShape(), source.getAxisMaps(), source.getValidity(),
      source.getOwner());
}

bool isUnitExtent(Attribute attribute) {
  auto expression = dyn_cast<PhysicalExprAttr>(attribute);
  return expression &&
         expression.getKind() ==
             static_cast<uint32_t>(PhysicalExprKind::Constant) &&
         expression.getValue() == 1;
}

struct SourcePlan {
  Value source;
  PhysicalSourceAxis sourceIdentity;
  unsigned sourceAxis;
  bool hasLoads;
  SmallVector<MakeRangeOp> ranges;
};

struct SliceRelation {
  PhysicalSourceAxis source;
  AxisMapAttr segmentMapping;
};

FailureOr<SourcePlan> analyzeSource(Value source, unsigned sourceAxis,
                                    PhysicalProgramAnalysis &analysis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || sourceAxis >= fragment.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> mapping = queryAxisMap(fragment, sourceAxis);
  if (failed(mapping))
    return failure();
  SourcePlan plan{source, sourceAxisIdentity(*mapping), sourceAxis, false, {}};
  PhysicalRangeFact fact = analysis.axisRanges(source, sourceAxis);
  if (failed(queryExactLogicalRange(fact)))
    return failure();
  plan.ranges.assign(fact.roots.begin(), fact.roots.end());
  for (Operation *access : fact.accesses) {
    if (auto load = dyn_cast<LoadOp>(access)) {
      if (!isa<ViewType>(load.getResource().getType()))
        return load.emitOpError(
            "region-fold source replay requires an immutable external-view load");
      plan.hasLoads = true;
      continue;
    }
    if (isa<GatherOp>(access)) {
      plan.hasLoads = true;
      continue;
    }
    return access->emitOpError(
               "region-fold source replay encountered an unsupported access"),
           failure();
  }
  if (!fact.unitStep)
    return plan.ranges.front().emitOpError(
        "region-fold source traversal requires a unit-step physical range");
  if (plan.ranges.empty())
    return failure();
  return plan;
}

LogicalResult buildSourceSlices(OpBuilder &builder, Location location,
                                ArrayRef<SourcePlan> plans,
                                ArrayRef<FragmentType> sliceTypes, Value offset,
                                Value segment, PhysicalExprAttr sliceExtent,
                                bool fullSegment,
                                SmallVectorImpl<Value> &slices,
                                Value &segmentTail, IRMapping &sliceMapping,
                                SmallVectorImpl<std::shared_ptr<IRMapping>>
                                    &sourceMappings,
                                std::string &reason) {
  for (auto [planIndex, plan] : llvm::enumerate(plans)) {
    if (planIndex >= sliceTypes.size() ||
        plan.sourceAxis >= sliceTypes[planIndex].getAxisMaps().size()) {
      reason = "source slice has no helper-local segment coordinate mapping";
      return failure();
    }
    // Repeated occurrences of the same author value share one replay graph.
    // Equal provenance is not enough: two distinct values may carry different
    // validity, fill, or producer relations even when they traverse the same
    // logical source axis.
    std::optional<unsigned> sharedRelation;
    for (unsigned previous = 0; previous < planIndex; ++previous) {
      const SourcePlan &candidate = plans[previous];
      bool sameSliceSchema =
          candidate.source == plan.source &&
          sliceTypes[previous] == sliceTypes[planIndex];
      if (sameSliceSchema &&
          candidate.sourceIdentity == plan.sourceIdentity &&
          candidate.sourceAxis == plan.sourceAxis &&
          candidate.ranges == plan.ranges) {
        sharedRelation = previous;
        break;
      }
    }
    std::shared_ptr<IRMapping> ownedMapping =
        sharedRelation ? sourceMappings[*sharedRelation]
                       : std::make_shared<IRMapping>();
    IRMapping &mapping = *ownedMapping;
    auto segmentMapping = cast<AxisMapAttr>(
        sliceTypes[planIndex].getAxisMaps()[plan.sourceAxis]);
    Value tail = sharedRelation ? segmentTail : Value();
    auto buildRange = [&](MakeRangeOp range) -> Value {
      Value start = builder.create<BinaryOp>(
          location, builder.getIndexType(), range.getStart(), offset,
          BinaryOperator::Add);
      auto rangeType = cast<FragmentType>(range.getResult().getType());
      auto blockedRange =
          replaceSliceAxis(rangeType, 0, sliceExtent, segmentMapping);
      Value value = builder.create<MakeRangeOp>(
          location, blockedRange, start, segment, range.getStep(),
          range.getLogicalStart(), range.getLogicalStop(),
          segmentMapping.getSourceId(), segmentMapping.getSourceAxis(),
          segmentMapping.getDerived());
      for (StringRef name :
           {sourceSubregionAttr, sourceSubregionBoundAttr})
        if (Attribute inherited = range->getAttr(name))
          value.getDefiningOp()->setAttr(name, inherited);
      Value valid;
      if (fullSegment) {
        Value truth = builder.create<arith::ConstantOp>(
            location, builder.getI1Type(), builder.getBoolAttr(true));
        valid = builder.create<SplatOp>(location, predicateType(blockedRange),
                                        truth);
      } else {
        Value stopFragment = builder.create<BroadcastOp>(
            location, blockedRange, range.getLogicalStop());
        auto validComparison = builder.create<CompareOp>(
            location, predicateType(blockedRange), value, stopFragment,
            ComparePredicate::Lt);
        validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
        valid = validComparison.getResult();
      }
      if (!tail)
        tail = valid;
      if (!segmentTail)
        segmentTail = valid;
      if (!mapping.lookupOrNull(range.getResult()))
        mapping.map(range.getResult(), value);
      if (!sliceMapping.lookupOrNull(range.getResult()))
        sliceMapping.map(range.getResult(), value);
      return value;
    };
    if (!sharedRelation)
      for (MakeRangeOp range : plan.ranges)
        buildRange(range);
    if (!tail) {
      reason = "source slice has no tail predicate";
      return failure();
    }
    ReplayMaterializationOptions replayOptions;
    replayOptions.scope = PhysicalReplayScope::Coordinate;
    replayOptions.segmentTail = tail;
    replayOptions.segmentMapping = segmentMapping;
    replayOptions.materializeZeroFill = true;
    FailureOr<Value> replayed = materializeReplayedValue(
        builder, location, plan.source, plan.sourceIdentity, sliceExtent,
        mapping, replayOptions);
    if (failed(replayed)) {
      reason = "source pure producer graph cannot be replayed";
      return failure();
    }
    slices.push_back(*replayed);
    if (!sliceMapping.lookupOrNull(plan.source))
      sliceMapping.map(plan.source, *replayed);
    sourceMappings.push_back(std::move(ownedMapping));
  }
  if (!segmentTail) {
    reason = "source slices have no shared physical tail predicate";
    return failure();
  }
  return success();
}

struct ExtentBinding {
  uint64_t sourceId;
  uint64_t sourceAxis;
  int64_t dimensionId;
  bool derived;
  Attribute extent;
  uint64_t actualSourceId;
  uint64_t actualSourceAxis;
  int64_t actualDimensionId;
  bool actualDerived;
};

const ExtentBinding *findBinding(ArrayRef<ExtentBinding> bindings,
                                 uint64_t sourceId, uint64_t sourceAxis,
                                 int64_t dimensionId, bool derived) {
  auto found = llvm::find_if(bindings, [&](const ExtentBinding &binding) {
    return binding.sourceId == sourceId &&
           binding.sourceAxis == sourceAxis &&
           binding.dimensionId == dimensionId && binding.derived == derived;
  });
  return found == bindings.end() ? nullptr : &*found;
}

LogicalResult collectExtentBindings(Type expected, Type actual,
                                    SmallVectorImpl<ExtentBinding> &bindings,
                                    std::string &reason) {
  if (auto expectedFragment = dyn_cast<FragmentType>(expected)) {
    auto actualFragment = dyn_cast<FragmentType>(actual);
    if (!actualFragment)
      return success();
    for (auto [expectedAxis, attribute] :
         llvm::enumerate(expectedFragment.getAxisMaps())) {
      auto expectedMap = cast<AxisMapAttr>(attribute);
      std::optional<unsigned> actualAxis;
      for (auto [axis, candidate] :
           llvm::enumerate(actualFragment.getAxisMaps())) {
        auto actualMap = cast<AxisMapAttr>(candidate);
        if (actualMap.getSourceId() == expectedMap.getSourceId() &&
            actualMap.getSourceAxis() == expectedMap.getSourceAxis() &&
            actualMap.getDerived() == expectedMap.getDerived()) {
          actualAxis = axis;
          break;
        }
      }
      // A structured helper argument is positionally paired with its source
      // component.  Canonical KIR gives the helper-local tensor dimensions
      // fresh provenance IDs, so the sliced source axis is not required to
      // retain the caller's ID across this explicit region boundary.
      if (!actualAxis && expectedFragment.getShape().size() ==
                             actualFragment.getShape().size())
        actualAxis = expectedAxis;
      if (!actualAxis)
        continue;
      Attribute extent = actualFragment.getShape()[*actualAxis];
      auto actualMap = cast<AxisMapAttr>(
          actualFragment.getAxisMaps()[*actualAxis]);
      if (const ExtentBinding *existing =
              findBinding(bindings, expectedMap.getSourceId(),
                          expectedMap.getSourceAxis(),
                          expectedMap.getDimensionId(),
                          expectedMap.getDerived())) {
        if (existing->extent != extent ||
            existing->actualSourceId != actualMap.getSourceId() ||
            existing->actualSourceAxis != actualMap.getSourceAxis() ||
            existing->actualDimensionId != actualMap.getDimensionId() ||
            existing->actualDerived != actualMap.getDerived()) {
          reason = "helper arguments bind one logical axis to incompatible physical extents";
          return failure();
        }
        continue;
      }
      bindings.push_back({expectedMap.getSourceId(),
                          expectedMap.getSourceAxis(),
                          expectedMap.getDimensionId(), expectedMap.getDerived(),
                          extent,
                          actualMap.getSourceId(),
                          actualMap.getSourceAxis(),
                          actualMap.getDimensionId(), actualMap.getDerived()});
    }
    return success();
  }
  auto expectedRecord = dyn_cast<RecordType>(expected);
  auto actualRecord = dyn_cast<RecordType>(actual);
  if (!expectedRecord || !actualRecord ||
      expectedRecord.getFieldNames() != actualRecord.getFieldNames() ||
      expectedRecord.getFieldTypes().size() !=
          actualRecord.getFieldTypes().size())
    return success();
  for (auto [expectedField, actualField] :
       llvm::zip(expectedRecord.getFieldTypes(),
                 actualRecord.getFieldTypes()))
    if (failed(collectExtentBindings(cast<TypeAttr>(expectedField).getValue(),
                                     cast<TypeAttr>(actualField).getValue(),
                                     bindings, reason)))
      return failure();
  return success();
}

bool carriesAxis(Type type, AxisMapAttr expected) {
  auto fragment = dyn_cast<FragmentType>(type);
  return fragment && llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
           auto mapping = cast<AxisMapAttr>(attribute);
           return mapping.getSourceId() == expected.getSourceId() &&
                  mapping.getSourceAxis() == expected.getSourceAxis() &&
                  mapping.getDerived() == expected.getDerived();
         });
}

Type bindPhysicalExtents(Type type, ArrayRef<ExtentBinding> bindings,
                         Operation *producer = nullptr) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    SmallVector<Attribute> shape(fragment.getShape().begin(),
                                 fragment.getShape().end());
    SmallVector<Attribute> mappings(fragment.getAxisMaps().begin(),
                                    fragment.getAxisMaps().end());
    bool changed = false;
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      const ExtentBinding *binding =
          findBinding(bindings, mapping.getSourceId(),
                      mapping.getSourceAxis(), mapping.getDimensionId(),
                      mapping.getDerived());
      if (!binding)
        continue;
      bool introducedUnitAxis = false;
      if (auto reshape = dyn_cast_or_null<ReshapeOp>(producer)) {
        introducedUnitAxis =
            isUnitExtent(shape[axis]) &&
            !carriesAxis(reshape.getValue().getType(), mapping);
      }
      if (introducedUnitAxis)
        continue;
      if (shape[axis] != binding->extent) {
        shape[axis] = binding->extent;
        changed = true;
      }
      if (mapping.getSourceId() != binding->actualSourceId ||
          mapping.getSourceAxis() != binding->actualSourceAxis ||
          mapping.getDerived() != binding->actualDerived) {
        mappings[axis] = AxisMapAttr::get(
            type.getContext(), binding->actualSourceId,
            binding->actualSourceAxis, binding->actualDimensionId, axis,
            binding->actualDerived);
        changed = true;
      }
    }
    if (!changed)
      return type;
    return FragmentType::get(
        type.getContext(), fragment.getElementType(),
        ArrayAttr::get(type.getContext(), shape),
        ArrayAttr::get(type.getContext(), mappings),
        fragment.getValidity(), fragment.getOwner());
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return type;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute attribute : record.getFieldTypes()) {
    Type field = cast<TypeAttr>(attribute).getValue();
    Type replacement = bindPhysicalExtents(field, bindings);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != field;
  }
  return changed ? Type(RecordType::get(type.getContext(), record.getFieldNames(),
                                        ArrayAttr::get(type.getContext(), fields),
                                        record.getOwner()))
                 : type;
}

void bindClonedOperationTypes(Operation *root,
                              ArrayRef<ExtentBinding> bindings) {
  std::function<void(Operation *)> bind = [&](Operation *operation) {
    for (Value result : operation->getResults()) {
      Type replacement =
          bindPhysicalExtents(result.getType(), bindings, operation);
      result.setType(replacement);
    }
    if (auto range = dyn_cast<MakeRangeOp>(operation)) {
      auto originalExtent =
          range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
      bool coveredIntroducedUnitDomain =
          originalExtent && originalExtent.value() == 1 &&
          samePhysicalScalarExpression(range.getStart(),
                                       range.getLogicalStart()) &&
          samePhysicalScalarExpression(range.getExtent(),
                                       range.getLogicalStop());
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      OpBuilder builder(range);
      Value physicalExtent;
      if (extent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Constant))
        physicalExtent = builder.create<arith::ConstantIndexOp>(
            range.getLoc(), extent.getValue());
      else
        physicalExtent = builder.create<PhysicalExprOp>(
            range.getLoc(), builder.getIndexType(), extent);
      range.getExtentMutable().assign(physicalExtent);
      if (coveredIntroducedUnitDomain)
        range->setOperand(4, physicalExtent);
    }
    for (Region &region : operation->getRegions())
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments())
          argument.setType(bindPhysicalExtents(argument.getType(), bindings));
        for (Operation &nested : block)
          bind(&nested);
      }
  };
  bind(root);
}

FailureOr<SmallVector<Value>> inlinePureRegion(OpBuilder &builder, Region &region,
                                               ValueRange arguments,
                                               std::string &reason,
                                               Value substituteSource = {},
                                               Value substituteTarget = {},
                                               Value conjunctSource = {},
                                               Value conjunctPredicate = {},
                                               Value additionalSource = {},
                                               Value additionalTarget = {},
                                               IRMapping *resultMapping = nullptr) {
  if (region.empty() || !llvm::hasSingleElement(region) ||
      region.front().getNumArguments() != arguments.size()) {
    reason = "helper argument schema mismatch";
    return failure();
  }
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield) {
    reason = "helper has no physical yield";
    return failure();
  }
  IRMapping localMapping;
  IRMapping &mapping = resultMapping ? *resultMapping : localMapping;
  SmallVector<ExtentBinding> extentBindings;
  for (auto [argument, value] :
       llvm::zip(region.front().getArguments(), arguments)) {
    if (failed(collectExtentBindings(argument.getType(), value.getType(),
                                     extentBindings, reason)))
      return failure();
    mapping.map(argument, value);
  }
  auto conjoin = [&](Value value) -> FailureOr<Value> {
    auto target = dyn_cast<FragmentType>(value.getType());
    if (!target || !target.getElementType().isInteger(1)) {
      reason = "summary membership predicate is not a boolean fragment";
      return failure();
    }
    Value predicate = conjunctPredicate;
    auto source = dyn_cast<FragmentType>(predicate.getType());
    if (source && source.getShape().size() == 1 &&
        target.getShape().size() > 1) {
      std::optional<unsigned> targetAxis;
      for (auto [axis, extent] : llvm::enumerate(target.getShape())) {
        if (extent != source.getShape()[0])
          continue;
        if (targetAxis) {
          targetAxis.reset();
          break;
        }
        targetAxis = axis;
      }
      if (!targetAxis) {
        reason = "physical tail extent does not identify one summary membership axis";
        return failure();
      }
      auto targetMapping =
          cast<AxisMapAttr>(target.getAxisMaps()[*targetAxis]);
      auto projected = FragmentType::get(
          target.getContext(), source.getElementType(), source.getShape(),
          ArrayAttr::get(
              target.getContext(),
              {AxisMapAttr::get(target.getContext(),
                                targetMapping.getSourceId(),
                                targetMapping.getSourceAxis(),
                                targetMapping.getDimensionId(), 0,
                                targetMapping.getDerived())}),
          target.getValidity(), target.getOwner());
      if (predicate.getType() != projected)
        predicate = builder.create<BroadcastOp>(value.getLoc(), projected,
                                                predicate);
    }
    if (predicate.getType() != target)
      predicate = builder.create<BroadcastOp>(value.getLoc(), target, predicate);
    return Value(builder.create<BinaryOp>(value.getLoc(), target, value,
                                          predicate,
                                          BinaryOperator::LogicalAnd));
  };
  if (substituteSource && substituteTarget) {
    if (substituteSource == conjunctSource) {
      FailureOr<Value> combined = conjoin(substituteTarget);
      if (failed(combined))
        return failure();
      substituteTarget = *combined;
    }
    mapping.map(substituteSource, substituteTarget);
  }
  if (additionalSource && additionalTarget)
    mapping.map(additionalSource, additionalTarget);
  for (Operation &operation : region.front().without_terminator()) {
    if (operation.getNumResults() == 1 &&
        mapping.lookupOrNull(operation.getResult(0)))
      continue;
    Operation *clone = builder.clone(operation, mapping);
    bindClonedOperationTypes(clone, extentBindings);
    for (auto [source, result] :
         llvm::zip(operation.getResults(), clone->getResults())) {
      if (source == conjunctSource) {
        Value mapped = mapping.lookupOrNull(source);
        FailureOr<Value> combined = conjoin(mapped ? mapped : result);
        if (failed(combined))
          return failure();
        mapping.map(source, *combined);
      } else if (!mapping.lookupOrNull(source)) {
        mapping.map(source, result);
      }
    }
  }
  SmallVector<Value> results;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      reason = "helper yield is outside the cloned value graph";
      return failure();
    }
    results.push_back(mapped);
  }
  return results;
}

ParameterOp findParameter(func::FuncOp kernel, ParameterAttr schema) {
  ParameterOp result;
  kernel.walk([&](ParameterOp parameter) {
    if (!result && parameter.getParameter().getName() == schema.getName())
      result = parameter;
  });
  return result;
}

std::optional<bool> booleanConstant(
    Value value, Value falsePredicate,
    llvm::SmallPtrSetImpl<Operation *> &visiting);

Value stripHelperForwarding(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (auto broadcast = dyn_cast<BroadcastOp>(definition)) {
      value = broadcast.getValue();
      continue;
    }
    if (auto cast = dyn_cast<CastOp>(definition)) {
      value = cast.getValue();
      continue;
    }
    if (auto reshape = dyn_cast<ReshapeOp>(definition)) {
      value = reshape.getValue();
      continue;
    }
    if (auto transpose = dyn_cast<TransposeOp>(definition)) {
      value = transpose.getValue();
      continue;
    }
    if (auto splat = dyn_cast<SplatOp>(definition)) {
      value = splat.getValue();
      continue;
    }
    if (auto select = dyn_cast<SelectOp>(definition)) {
      llvm::SmallPtrSet<Operation *, 16> visiting;
      std::optional<bool> condition =
          booleanConstant(select.getCondition(), Value(), visiting);
      if (!condition)
        return {};
      value = *condition ? select.getTrueValue() : select.getFalseValue();
      continue;
    }
    return value;
  }
  return value;
}

BlockArgument rootHelperArgument(Value value) {
  return dyn_cast_or_null<BlockArgument>(stripHelperForwarding(value));
}

struct HelperCoordinateExpression {
  BlockArgument coordinate;
  Value scalar;
  bool subtractScalar = false;
};

bool isHelperScalar(Value value) {
  value = stripHelperForwarding(value);
  if (!value)
    return false;
  if (isa<BlockArgument>(value))
    return value.getType().isIntOrIndex();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  return constant && isa<IntegerAttr>(constant.getValue());
}

FailureOr<HelperCoordinateExpression>
helperCoordinateExpression(Value value) {
  value = stripHelperForwarding(value);
  if (!value)
    return failure();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (!isa<FragmentType>(argument.getType()))
      return failure();
    return HelperCoordinateExpression{argument, Value(), false};
  }
  auto binary = value.getDefiningOp<BinaryOp>();
  if (!binary)
    return failure();
  BinaryOperator kind = binary.getOperatorKind();
  if (kind != BinaryOperator::Add && kind != BinaryOperator::Subtract)
    return failure();
  BlockArgument lhs = rootHelperArgument(binary.getLhs());
  if (lhs && isa<FragmentType>(lhs.getType()) &&
      isHelperScalar(binary.getRhs()))
    return HelperCoordinateExpression{lhs,
                                      stripHelperForwarding(binary.getRhs()),
                                      kind == BinaryOperator::Subtract};
  if (kind == BinaryOperator::Add) {
    BlockArgument rhs = rootHelperArgument(binary.getRhs());
    if (rhs && isa<FragmentType>(rhs.getType()) &&
        isHelperScalar(binary.getLhs()))
      return HelperCoordinateExpression{rhs,
                                        stripHelperForwarding(binary.getLhs()),
                                        false};
  }
  return failure();
}

bool isZeroConstant(Value value, Value falsePredicate,
                    llvm::SmallPtrSetImpl<Operation *> &visiting);

std::optional<bool> booleanConstant(
    Value value, Value falsePredicate,
    llvm::SmallPtrSetImpl<Operation *> &visiting) {
  if (value == falsePredicate)
    return false;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      if (integer.getType().isInteger(1))
        return integer.getInt() != 0;
    return std::nullopt;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition || !visiting.insert(definition).second)
    return std::nullopt;
  auto finish = [&](std::optional<bool> result) {
    visiting.erase(definition);
    return result;
  };
  if (auto broadcast = dyn_cast<BroadcastOp>(definition))
    return finish(booleanConstant(broadcast.getValue(), falsePredicate, visiting));
  if (auto splat = dyn_cast<SplatOp>(definition))
    return finish(booleanConstant(splat.getValue(), falsePredicate, visiting));
  if (auto reshape = dyn_cast<ReshapeOp>(definition))
    return finish(booleanConstant(reshape.getValue(), falsePredicate, visiting));
  if (auto transpose = dyn_cast<TransposeOp>(definition))
    return finish(booleanConstant(transpose.getValue(), falsePredicate, visiting));
  if (auto cast = dyn_cast<CastOp>(definition))
    return finish(booleanConstant(cast.getValue(), falsePredicate, visiting));
  if (auto extract = dyn_cast<ExtractOp>(definition)) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record || extract.getField() >= record.getFields().size())
      return finish(std::nullopt);
    return finish(booleanConstant(record.getFields()[extract.getField()],
                                  falsePredicate, visiting));
  }
  if (auto select = dyn_cast<SelectOp>(definition)) {
    std::optional<bool> condition =
        booleanConstant(select.getCondition(), falsePredicate, visiting);
    if (!condition)
      return finish(std::nullopt);
    return finish(booleanConstant(*condition ? select.getTrueValue()
                                             : select.getFalseValue(),
                                  falsePredicate, visiting));
  }
  if (auto compare = dyn_cast<CompareOp>(definition)) {
    llvm::SmallPtrSet<Operation *, 16> zeroVisiting;
    bool lhsZero =
        isZeroConstant(compare.getLhs(), falsePredicate, zeroVisiting);
    zeroVisiting.clear();
    bool rhsZero =
        isZeroConstant(compare.getRhs(), falsePredicate, zeroVisiting);
    if (!lhsZero || !rhsZero)
      return finish(std::nullopt);
    switch (compare.getPredicate()) {
    case ComparePredicate::Eq:
    case ComparePredicate::Le:
    case ComparePredicate::Ge:
      return finish(true);
    case ComparePredicate::Ne:
    case ComparePredicate::Lt:
    case ComparePredicate::Gt:
      return finish(false);
    default:
      return finish(std::nullopt);
    }
  }
  if (auto binary = dyn_cast<BinaryOp>(definition)) {
    std::optional<bool> lhs =
        booleanConstant(binary.getLhs(), falsePredicate, visiting);
    std::optional<bool> rhs =
        booleanConstant(binary.getRhs(), falsePredicate, visiting);
    BinaryOperator kind = binary.getOperatorKind();
    if (kind == BinaryOperator::LogicalAnd ||
        kind == BinaryOperator::BitwiseAnd) {
      if ((lhs && !*lhs) || (rhs && !*rhs))
        return finish(false);
      if (lhs && rhs)
        return finish(*lhs && *rhs);
    }
    if (kind == BinaryOperator::LogicalOr ||
        kind == BinaryOperator::BitwiseOr) {
      if ((lhs && *lhs) || (rhs && *rhs))
        return finish(true);
      if (lhs && rhs)
        return finish(*lhs || *rhs);
    }
    return finish(std::nullopt);
  }
  if (auto reduce = dyn_cast<ReduceOp>(definition)) {
    auto result = cast<OpResult>(value).getResultNumber();
    if (result >= reduce.getSourceCount() ||
        result >= reduce.getIdentityCount())
      return finish(std::nullopt);
    std::optional<bool> source = booleanConstant(
        reduce.getInputs()[result], falsePredicate, visiting);
    std::optional<bool> identity = booleanConstant(
        reduce.getInputs()[reduce.getSourceCount() + result], falsePredicate,
        visiting);
    if (source && identity && *source == *identity)
      return finish(source);
  }
  return finish(std::nullopt);
}

bool isZeroConstant(Value value, Value falsePredicate,
                    llvm::SmallPtrSetImpl<Operation *> &visiting) {
  if (value == falsePredicate)
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt() == 0;
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
      return floating.getValue().isZero();
    return false;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition || !visiting.insert(definition).second)
    return false;
  auto finish = [&](bool result) {
    visiting.erase(definition);
    return result;
  };
  if (auto broadcast = dyn_cast<BroadcastOp>(definition))
    return finish(isZeroConstant(broadcast.getValue(), falsePredicate, visiting));
  if (auto splat = dyn_cast<SplatOp>(definition))
    return finish(isZeroConstant(splat.getValue(), falsePredicate, visiting));
  if (auto reshape = dyn_cast<ReshapeOp>(definition))
    return finish(isZeroConstant(reshape.getValue(), falsePredicate, visiting));
  if (auto transpose = dyn_cast<TransposeOp>(definition))
    return finish(isZeroConstant(transpose.getValue(), falsePredicate, visiting));
  if (auto cast = dyn_cast<CastOp>(definition))
    return finish(isZeroConstant(cast.getValue(), falsePredicate, visiting));
  if (auto extract = dyn_cast<ExtractOp>(definition)) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record || extract.getField() >= record.getFields().size())
      return finish(false);
    return finish(isZeroConstant(record.getFields()[extract.getField()],
                                 falsePredicate, visiting));
  }
  if (auto binary = dyn_cast<BinaryOp>(definition)) {
    bool lhs = isZeroConstant(binary.getLhs(), falsePredicate, visiting);
    bool rhs = isZeroConstant(binary.getRhs(), falsePredicate, visiting);
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
      return finish(lhs && rhs);
    case BinaryOperator::Multiply:
      return finish(lhs || rhs);
    default:
      return finish(false);
    }
  }
  if (auto select = dyn_cast<SelectOp>(definition)) {
    std::optional<bool> condition =
        booleanConstant(select.getCondition(), falsePredicate, visiting);
    if (!condition)
      return finish(false);
    return finish(isZeroConstant(*condition ? select.getTrueValue()
                                            : select.getFalseValue(),
                                 falsePredicate, visiting));
  }
  if (auto reduce = dyn_cast<ReduceOp>(definition)) {
    auto result = cast<OpResult>(value).getResultNumber();
    if (result >= reduce.getSourceCount() ||
        result >= reduce.getIdentityCount())
      return finish(false);
    return finish(isZeroConstant(reduce.getInputs()[result], falsePredicate,
                                 visiting) &&
                  isZeroConstant(
                      reduce.getInputs()[reduce.getSourceCount() + result],
                      falsePredicate, visiting));
  }
  if (auto contract = dyn_cast<ContractOp>(definition))
    return finish(isZeroConstant(contract.getAccumulator(), falsePredicate,
                                 visiting) &&
                  (isZeroConstant(contract.getLhs(), falsePredicate, visiting) ||
                   isZeroConstant(contract.getRhs(), falsePredicate, visiting)));
  return finish(false);
}

bool equalWhenPredicateIsFalse(Value summary, Value identity,
                               Value falsePredicate) {
  auto summaryRecord = summary.getDefiningOp<MakeRecordOp>();
  auto identityRecord = identity.getDefiningOp<MakeRecordOp>();
  if (summaryRecord || identityRecord) {
    if (!summaryRecord || !identityRecord ||
        summaryRecord.getFields().size() != identityRecord.getFields().size())
      return false;
    for (auto [summaryField, identityField] :
         llvm::zip(summaryRecord.getFields(), identityRecord.getFields()))
      if (!equalWhenPredicateIsFalse(summaryField, identityField,
                                     falsePredicate))
        return false;
    return true;
  }
  llvm::SmallPtrSet<Operation *, 16> booleanVisiting;
  std::optional<bool> summaryBoolean =
      booleanConstant(summary, falsePredicate, booleanVisiting);
  booleanVisiting.clear();
  std::optional<bool> identityBoolean =
      booleanConstant(identity, falsePredicate, booleanVisiting);
  if (summaryBoolean && identityBoolean)
    return *summaryBoolean == *identityBoolean;
  llvm::SmallPtrSet<Operation *, 16> zeroVisiting;
  if (!isZeroConstant(summary, falsePredicate, zeroVisiting))
    return false;
  zeroVisiting.clear();
  return isZeroConstant(identity, falsePredicate, zeroVisiting);
}

Value summaryMembershipPredicate(RegionFoldOp fold, ValueRange identities,
                                 ParameterAttr segment) {
  auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  if (!yield || yield.getValues().size() != identities.size())
    return {};
  Value candidate;
  for (Operation &operation : fold.getSummarize().front().without_terminator()) {
    for (Value result : operation.getResults()) {
      auto fragment = dyn_cast<FragmentType>(result.getType());
      bool carriesSegment = fragment && llvm::any_of(
          fragment.getShape(), [&](Attribute extent) {
            auto expression = dyn_cast<PhysicalExprAttr>(extent);
            return expression &&
                   expression.getKind() == static_cast<uint32_t>(
                                               PhysicalExprKind::Parameter) &&
                   expression.getSymbol() == segment.getName();
          });
      if (!fragment || !fragment.getElementType().isInteger(1) ||
          !carriesSegment)
        continue;
      bool identityOnly = true;
      for (auto [summary, identity] : llvm::zip(yield.getValues(), identities))
        identityOnly &=
            equalWhenPredicateIsFalse(summary, identity, result);
      if (identityOnly)
        candidate = result;
    }
  }
  return candidate;
}

struct SummaryEmptinessPlan {
  unsigned optionalField;
  RecordType fullType;
  RecordType payloadType;
  Value summarizeValidity;
};

bool isRecordField(Value value, BlockArgument record, unsigned field) {
  auto extract = value.getDefiningOp<ExtractOp>();
  return extract && extract.getRecord() == record && extract.getField() == field;
}

bool isMembershipReduction(Value value, Value membershipPredicate) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  auto reduce = value.getDefiningOp<ReduceOp>();
  auto result = dyn_cast<OpResult>(value);
  if (!reduce || !result || reduce.getSourceCount() != 1 ||
      reduce.getIdentityCount() != 1 || reduce.getCaptureCount() != 0 ||
      reduce.getNumResults() != 1 || result.getResultNumber() != 0 ||
      reduce.getInputs().size() != 2 ||
      reduce.getInputs().front() != membershipPredicate ||
      !llvm::hasSingleElement(reduce.getCombine()))
    return false;
  llvm::SmallPtrSet<Operation *, 16> visiting;
  std::optional<bool> identity =
      booleanConstant(reduce.getInputs().back(), Value(), visiting);
  auto yield = dyn_cast<YieldOp>(reduce.getCombine().front().getTerminator());
  auto merged = yield && yield.getValues().size() == 1
                    ? yield.getValues().front().getDefiningOp<BinaryOp>()
                    : BinaryOp();
  if (!identity || *identity || !merged ||
      (merged.getOperatorKind() != BinaryOperator::LogicalOr &&
       merged.getOperatorKind() != BinaryOperator::BitwiseOr) ||
      reduce.getCombine().front().getNumArguments() != 2)
    return false;
  Block &combine = reduce.getCombine().front();
  return (merged.getLhs() == combine.getArgument(0) &&
          merged.getRhs() == combine.getArgument(1)) ||
         (merged.getRhs() == combine.getArgument(0) &&
          merged.getLhs() == combine.getArgument(1));
}

std::optional<SummaryEmptinessPlan>
summaryEmptinessPlan(RegionFoldOp fold, ValueRange identities,
                     Value membershipPredicate) {
  if (identities.size() != 1 || fold.getSummarize().empty() ||
      fold.getCombine().empty())
    return std::nullopt;
  auto identity = identities.front().getDefiningOp<MakeRecordOp>();
  auto summarizeYield =
      dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  auto combineYield =
      dyn_cast<YieldOp>(fold.getCombine().front().getTerminator());
  auto summary = summarizeYield && summarizeYield.getValues().size() == 1
                     ? summarizeYield.getValues().front().getDefiningOp<MakeRecordOp>()
                     : MakeRecordOp();
  auto combined = combineYield && combineYield.getValues().size() == 1
                      ? combineYield.getValues().front().getDefiningOp<MakeRecordOp>()
                      : MakeRecordOp();
  Block &combine = fold.getCombine().front();
  if (!identity || !summary || !combined || combine.getNumArguments() != 2 ||
      identity.getFields().size() != summary.getFields().size() ||
      identity.getFields().size() != combined.getFields().size())
    return std::nullopt;
  auto fullType = dyn_cast<RecordType>(identities.front().getType());
  if (!fullType || fullType.getFieldTypes().size() != identity.getFields().size())
    return std::nullopt;

  std::optional<unsigned> selected;
  for (unsigned field = 0; field < identity.getFields().size(); ++field) {
    llvm::SmallPtrSet<Operation *, 16> visiting;
    std::optional<bool> identityValue =
        booleanConstant(identity.getFields()[field], Value(), visiting);
    auto summaryType = dyn_cast<FragmentType>(summary.getFields()[field].getType());
    auto merged = combined.getFields()[field].getDefiningOp<BinaryOp>();
    if (!identityValue || *identityValue || !summaryType ||
        !summaryType.getElementType().isInteger(1) || !merged ||
        !isMembershipReduction(summary.getFields()[field],
                               membershipPredicate) ||
        (merged.getOperatorKind() != BinaryOperator::LogicalOr &&
         merged.getOperatorKind() != BinaryOperator::BitwiseOr))
      continue;
    BlockArgument lhs = combine.getArgument(0);
    BlockArgument rhs = combine.getArgument(1);
    bool fieldsMatch =
        (isRecordField(merged.getLhs(), lhs, field) &&
         isRecordField(merged.getRhs(), rhs, field)) ||
        (isRecordField(merged.getRhs(), lhs, field) &&
         isRecordField(merged.getLhs(), rhs, field));
    if (!fieldsMatch || selected)
      return std::nullopt;
    selected = field;
  }
  if (!selected)
    return std::nullopt;

  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  for (unsigned field = 0; field < fullType.getFieldTypes().size(); ++field) {
    if (field == *selected)
      continue;
    names.push_back(fullType.getFieldNames()[field]);
    types.push_back(fullType.getFieldTypes()[field]);
  }
  auto payload = RecordType::get(
      fold.getContext(), ArrayAttr::get(fold.getContext(), names),
      ArrayAttr::get(fold.getContext(), types), fullType.getOwner());
  return SummaryEmptinessPlan{*selected, fullType, payload,
                              summary.getFields()[*selected]};
}

struct OnlineRegionPlan {
  OnlineSummaryStructure summary;
  OnlineSummaryMerge merge;
};

std::optional<OnlineRegionPlan>
onlineRegionPlan(RegionFoldOp fold,
                 const std::optional<SummaryEmptinessPlan> &emptiness) {
  if (!emptiness || fold.getSummarize().empty())
    return std::nullopt;
  auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  auto record = yield && yield.getValues().size() == 1
                    ? yield.getValues().front().getDefiningOp<MakeRecordOp>()
                    : MakeRecordOp();
  FailureOr<OnlineSummaryStructure> summary =
      matchOnlineSummaryStructure(record);
  if (failed(summary))
    return std::nullopt;
  if (summary->validityField != emptiness->optionalField ||
      summary->record.getResult().getType() != emptiness->fullType)
    return std::nullopt;
  FailureOr<OnlineSummaryMerge> merge =
      matchOnlineSummaryMerge(fold.getCombine(), *summary);
  if (failed(merge))
    return std::nullopt;
  return OnlineRegionPlan{std::move(*summary), std::move(*merge)};
}

FailureOr<Value> coRealizeOnlineRegion(
    OpBuilder &builder, Location location, OnlineRegionPlan &plan,
    IRMapping &summaryMapping, IRMapping &mergeMapping) {
  auto summaryValue = [&](Value value) {
    return summaryMapping.lookupOrNull(value);
  };
  auto mergeValue = [&](Value value) {
    return mergeMapping.lookupOrNull(value);
  };

  Value memberValidity = summaryValue(plan.summary.memberValidity);
  Value maskedScore = summaryValue(plan.summary.maskedScore.getResult());
  Value probability = summaryValue(plan.summary.probability.getResult());
  Value probabilityZero =
      summaryValue(plan.summary.probability.getFalseValue());
  Value probabilityCast =
      summaryValue(plan.summary.probabilityCast.getResult());
  Value mass = summaryValue(plan.summary.mass.getResult(0));
  Value moment = summaryValue(plan.summary.moment.getResult());
  Value combinedMaximum = mergeValue(plan.merge.combinedMaximum);
  Value leftMassTerm = mergeValue(plan.merge.leftMassTerm);
  Value leftMomentTerm = mergeValue(plan.merge.leftMomentTerm);
  Value mergedRecord = mergeValue(plan.merge.record.getResult());
  if (!memberValidity || !maskedScore || !probability || !probabilityZero ||
      !probabilityCast || !mass || !moment || !combinedMaximum ||
      !leftMassTerm || !leftMomentTerm || !mergedRecord)
    return failure();

  auto mappedMass = mass.getDefiningOp<ReduceOp>();
  auto mappedMoment = moment.getDefiningOp<ContractOp>();
  if (!mappedMass || !mappedMoment)
    return failure();
  FailureOr<Value> projectedMaximum = projectPhysicalValueToSchema(
      builder, location, combinedMaximum, maskedScore.getType());
  FailureOr<Value> projectedZero = projectPhysicalValueToSchema(
      builder, location, probabilityZero, probability.getType());
  if (failed(projectedMaximum) || failed(projectedZero))
    return failure();

  // region_fold declares summarize/combine as an ordered homomorphism.  Keep
  // the same maximum and left carry scale, but associate the right scale with
  // each member before its mass/moment reductions.
  auto shifted = builder.create<BinaryOp>(
      location, maskedScore.getType(), maskedScore, *projectedMaximum,
      BinaryOperator::Subtract);
  auto directExponential = builder.create<UnaryOp>(
      location, probability.getType(), shifted,
      plan.summary.exponential.getOperatorKind(),
      plan.summary.exponential.getApproximate(), plan.summary.exponential.getFlushToZero());
  auto directProbability = builder.create<SelectOp>(
      location, probability.getType(), memberValidity, directExponential,
      *projectedZero);
  if (Attribute origin = plan.summary.exponential->getAttr(originAttr)) {
    shifted->setAttr(originAttr, origin);
    directExponential->setAttr(originAttr, origin);
    directProbability->setAttr(originAttr, origin);
  }

  ReduceOp directMass = cloneReductionWithSource(
      builder, location, mappedMass, directProbability.getResult());
  auto directProbabilityCast = builder.create<CastOp>(
      location, probabilityCast.getType(), directProbability.getResult());
  if (Attribute origin = plan.summary.probabilityCast->getAttr(originAttr))
    directProbabilityCast->setAttr(originAttr, origin);
  auto mappedRecord = mergedRecord.getDefiningOp<MakeRecordOp>();
  if (!mappedRecord)
    return failure();
  Type massType = mappedRecord.getFields()[plan.summary.massField].getType();
  Type momentType =
      mappedRecord.getFields()[plan.summary.momentField].getType();
  FailureOr<Value> projectedLeftMass = projectPhysicalValueToSchema(
      builder, location, leftMassTerm, massType);
  FailureOr<Value> projectedMass = projectPhysicalValueToSchema(
      builder, location, directMass.getResult(0), massType);
  FailureOr<Value> projectedLeftMoment = projectPhysicalValueToSchema(
      builder, location, leftMomentTerm, mappedMoment.getResult().getType());
  if (failed(projectedLeftMass) || failed(projectedMass) ||
      failed(projectedLeftMoment))
    return failure();
  // The matched moment has a zero accumulator. Keep the ordered left carry
  // inside the contraction instead of materializing a second matrix and add.
  auto directMoment = builder.create<ContractOp>(
      location, mappedMoment.getResult().getType(), directProbabilityCast,
      mappedMoment.getRhs(), *projectedLeftMoment,
      mappedMoment.getLhsReductionAxes(), mappedMoment.getRhsReductionAxes(),
      mappedMoment.getLhsBatchAxes(), mappedMoment.getRhsBatchAxes());
  if (Attribute origin = mappedMoment->getAttr(originAttr))
    directMoment->setAttr(originAttr, origin);
  FailureOr<Value> projectedMoment = projectPhysicalValueToSchema(
      builder, location, directMoment.getResult(), momentType);
  if (failed(projectedMoment))
    return failure();
  Value combinedMass = builder.create<BinaryOp>(
      location, massType, *projectedLeftMass, *projectedMass,
      BinaryOperator::Add);

  SmallVector<Value> fields;
  fields.reserve(plan.merge.record.getFields().size());
  for (auto [field, original] :
       llvm::enumerate(plan.merge.record.getFields())) {
    if (field == plan.summary.massField) {
      fields.push_back(combinedMass);
      continue;
    }
    if (field == plan.summary.momentField) {
      fields.push_back(*projectedMoment);
      continue;
    }
    Value mapped = mergeValue(original);
    if (!mapped)
      return failure();
    fields.push_back(mapped);
  }
  auto replacement = builder.create<MakeRecordOp>(
      location, mappedRecord.getResult().getType(), fields);
  if (Attribute origin = plan.merge.record->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  return replacement.getResult();
}

Value allTrueValue(OpBuilder &builder, Location location, Type type) {
  Value truth = builder.create<arith::ConstantOp>(
      location, builder.getI1Type(), builder.getBoolAttr(true));
  if (auto fragment = dyn_cast<FragmentType>(type))
    return builder.create<SplatOp>(location, fragment, truth);
  return truth;
}

FailureOr<Value> stripOptionalRecord(OpBuilder &builder, Location location,
                                     Value value,
                                     const SummaryEmptinessPlan &plan) {
  auto record = value.getDefiningOp<MakeRecordOp>();
  SmallVector<Value> fields;
  unsigned payloadField = 0;
  for (unsigned field = 0; field < plan.fullType.getFieldTypes().size(); ++field) {
    if (field == plan.optionalField)
      continue;
    Type type = cast<TypeAttr>(plan.payloadType.getFieldTypes()[payloadField++])
                    .getValue();
    fields.push_back(record ? record.getFields()[field]
                            : Value(builder.create<ExtractOp>(location, type,
                                                              value, field)));
  }
  return Value(builder.create<MakeRecordOp>(location, plan.payloadType, fields));
}

Value restoreOptionalRecord(OpBuilder &builder, Location location, Value payload,
                            const SummaryEmptinessPlan &plan,
                            Value optionalValidity = {}) {
  SmallVector<Value> fields;
  unsigned payloadField = 0;
  for (unsigned field = 0; field < plan.fullType.getFieldTypes().size(); ++field) {
    Type type = cast<TypeAttr>(plan.fullType.getFieldTypes()[field]).getValue();
    if (field == plan.optionalField) {
      fields.push_back(optionalValidity
                           ? optionalValidity
                           : allTrueValue(builder, location, type));
      continue;
    }
    fields.push_back(builder.create<ExtractOp>(location, type, payload,
                                               payloadField++));
  }
  return builder.create<MakeRecordOp>(location, plan.fullType, fields);
}

void simplifyKnownRecordValues(func::FuncOp kernel) {
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> candidates;
    kernel.walk([&](Operation *operation) {
      if (isa<ExtractOp, SelectOp, BinaryOp>(operation))
        candidates.push_back(operation);
    });
    for (Operation *operation : candidates) {
      if (!operation->getBlock())
        continue;
      if (auto extract = dyn_cast<ExtractOp>(operation)) {
        auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
        if (!record || extract.getField() >= record.getFields().size())
          continue;
        extract.getResult().replaceAllUsesWith(
            record.getFields()[extract.getField()]);
        extract.erase();
        changed = true;
        continue;
      }
      if (auto select = dyn_cast<SelectOp>(operation)) {
        llvm::SmallPtrSet<Operation *, 16> visiting;
        std::optional<bool> condition =
            booleanConstant(select.getCondition(), Value(), visiting);
        if (!condition)
          continue;
        select.getResult().replaceAllUsesWith(
            *condition ? select.getTrueValue() : select.getFalseValue());
        select.erase();
        changed = true;
        continue;
      }
      auto binary = cast<BinaryOp>(operation);
      BinaryOperator kind = binary.getOperatorKind();
      if (kind != BinaryOperator::LogicalAnd &&
          kind != BinaryOperator::BitwiseAnd &&
          kind != BinaryOperator::LogicalOr &&
          kind != BinaryOperator::BitwiseOr)
        continue;
      llvm::SmallPtrSet<Operation *, 16> visiting;
      std::optional<bool> lhs =
          booleanConstant(binary.getLhs(), Value(), visiting);
      visiting.clear();
      std::optional<bool> rhs =
          booleanConstant(binary.getRhs(), Value(), visiting);
      Value replacement;
      bool conjunction = kind == BinaryOperator::LogicalAnd ||
                         kind == BinaryOperator::BitwiseAnd;
      if (conjunction) {
        if (lhs && !*lhs)
          replacement = binary.getLhs();
        else if (rhs && !*rhs)
          replacement = binary.getRhs();
        else if (lhs && *lhs)
          replacement = binary.getRhs();
        else if (rhs && *rhs)
          replacement = binary.getLhs();
      } else {
        if (lhs && *lhs)
          replacement = binary.getLhs();
        else if (rhs && *rhs)
          replacement = binary.getRhs();
        else if (lhs && !*lhs)
          replacement = binary.getRhs();
        else if (rhs && !*rhs)
          replacement = binary.getLhs();
      }
      if (!replacement)
        continue;
      binary.getResult().replaceAllUsesWith(replacement);
      binary.erase();
      changed = true;
    }
  } while (changed);
}

bool isZeroWithSources(Value value, const llvm::SmallDenseSet<Value> &sources,
                       llvm::SmallPtrSetImpl<Operation *> &visiting) {
  if (sources.contains(value))
    return true;
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt() == 0;
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
      return floating.getValue().isZero();
    return false;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition || !visiting.insert(definition).second)
    return false;
  auto finish = [&](bool result) {
    visiting.erase(definition);
    return result;
  };
  if (auto broadcast = dyn_cast<BroadcastOp>(definition))
    return finish(isZeroWithSources(broadcast.getValue(), sources, visiting));
  if (auto splat = dyn_cast<SplatOp>(definition))
    return finish(isZeroWithSources(splat.getValue(), sources, visiting));
  if (auto cast = dyn_cast<CastOp>(definition))
    return finish(isZeroWithSources(cast.getValue(), sources, visiting));
  if (auto extract = dyn_cast<ExtractOp>(definition)) {
    auto record = extract.getRecord().getDefiningOp<MakeRecordOp>();
    if (!record || extract.getField() >= record.getFields().size())
      return finish(false);
    return finish(isZeroWithSources(record.getFields()[extract.getField()],
                                    sources, visiting));
  }
  if (auto unary = dyn_cast<UnaryOp>(definition))
    return finish(unary.getOperatorKind() == UnaryOperator::Negate &&
                  isZeroWithSources(unary.getInput(), sources, visiting));
  if (auto binary = dyn_cast<BinaryOp>(definition)) {
    bool lhs = isZeroWithSources(binary.getLhs(), sources, visiting);
    bool rhs = isZeroWithSources(binary.getRhs(), sources, visiting);
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add:
    case BinaryOperator::Subtract:
      return finish(lhs && rhs);
    case BinaryOperator::Multiply:
      return finish(lhs || rhs);
    default:
      return finish(false);
    }
  }
  if (auto reduce = dyn_cast<ReduceOp>(definition)) {
    auto result = cast<OpResult>(value).getResultNumber();
    if (result >= reduce.getSourceCount() ||
        result >= reduce.getIdentityCount())
      return finish(false);
    return finish(isZeroWithSources(reduce.getInputs()[result], sources,
                                    visiting) &&
                  isZeroWithSources(
                      reduce.getInputs()[reduce.getSourceCount() + result],
                      sources, visiting));
  }
  if (auto contract = dyn_cast<ContractOp>(definition))
    return finish(isZeroWithSources(contract.getAccumulator(), sources,
                                    visiting) &&
                  (isZeroWithSources(contract.getLhs(), sources, visiting) ||
                   isZeroWithSources(contract.getRhs(), sources, visiting)));
  if (auto record = dyn_cast<MakeRecordOp>(definition)) {
    for (Value field : record.getFields())
      if (!isZeroWithSources(field, sources, visiting))
        return finish(false);
    return finish(true);
  }
  return finish(false);
}

bool scanTailIsIdentity(RegionScanOp scan, ArrayRef<SourcePlan> plans,
                        ValueRange identities) {
  auto yield = dyn_cast<YieldOp>(scan.getSummarize().front().getTerminator());
  if (!yield || yield.getValues().size() != identities.size())
    return false;
  llvm::SmallDenseSet<Value> zeroSources;
  for (auto [index, argument] : llvm::enumerate(
           scan.getSummarize().front().getArguments().take_front(
               scan.getSourceCount())))
    if (plans[index].hasLoads)
      zeroSources.insert(argument);
  llvm::SmallDenseSet<Value> noSources;
  for (auto [summary, identity] : llvm::zip(yield.getValues(), identities)) {
    llvm::SmallPtrSet<Operation *, 16> visiting;
    if (!isZeroWithSources(summary, zeroSources, visiting))
      return false;
    visiting.clear();
    if (!isZeroWithSources(identity, noSources, visiting))
      return false;
  }
  return true;
}

FailureOr<Type> slicedType(Type type,
                           ArrayRef<SliceRelation> relations,
                           PhysicalExprAttr extent,
                           Operation *producer = nullptr) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    std::optional<PhysicalAxisProjection> selected;
    AxisMapAttr segmentMapping;
    for (const SliceRelation &relation : relations) {
      PhysicalAxisProjection axis =
          queryFragmentAxis(fragment, relation.source);
      if (axis.state == PhysicalFactState::Ambiguous)
        return failure();
      if (!axis.isExact())
        continue;
      if (selected &&
          (selected->fragmentAxis != axis.fragmentAxis ||
           segmentMapping != relation.segmentMapping))
        return failure();
      selected = axis;
      segmentMapping = relation.segmentMapping;
    }
    if (!selected)
      return type;
    if (auto reshape = dyn_cast_or_null<ReshapeOp>(producer)) {
      bool introducedUnitAxis =
          isUnitExtent(fragment.getShape()[selected->fragmentAxis]) &&
          llvm::none_of(relations, [&](const SliceRelation &relation) {
            return !queryFragmentAxes(reshape.getValue().getType(),
                                      relation.source)
                        .empty();
          });
      if (introducedUnitAxis)
        return type;
    }
    return Type(replaceSliceAxis(fragment, selected->fragmentAxis, extent,
                                 segmentMapping));
  }
  if (auto record = dyn_cast<RecordType>(type)) {
    SmallVector<Attribute> fields;
    for (Attribute field : record.getFieldTypes()) {
      FailureOr<Type> converted =
          slicedType(cast<TypeAttr>(field).getValue(), relations, extent);
      if (failed(converted))
        return failure();
      fields.push_back(TypeAttr::get(*converted));
    }
    return Type(RecordType::get(type.getContext(), record.getFieldNames(),
                                ArrayAttr::get(type.getContext(), fields),
                                record.getOwner()));
  }
  return type;
}

bool isScanOutputPureConsumer(Operation *operation) {
  return isPhysicalReplayNode(operation, PhysicalReplayScope::Coordinate,
                              /*allowAccesses=*/false);
}

LogicalResult collectScanOutputConsumers(RegionScanOp scan,
                                         SmallVectorImpl<Operation *> &ordered,
                                         std::string &reason) {
  llvm::SetVector<Operation *> closure;
  SmallVector<Operation *> worklist;
  for (Value output : scan.getResults().take_front(scan.getOutputCount()))
    llvm::append_range(worklist, output.getUsers());
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (!closure.insert(operation))
      continue;
    if (operation->getBlock() != scan->getBlock()) {
      reason = "region-scan output escapes its physical execution block";
      return failure();
    }
    if (isa<StoreOp>(operation))
      continue;
    if (!isScanOutputPureConsumer(operation)) {
      reason = "region-scan output consumer is not a slice-preserving pure graph ending in a store";
      return failure();
    }
    for (Value result : operation->getResults())
      llvm::append_range(worklist, result.getUsers());
  }
  if (closure.empty()) {
    reason = "region-scan output has no source-aligned physical assembly";
    return failure();
  }
  for (Operation &operation : *scan->getBlock())
    if (closure.contains(&operation))
      ordered.push_back(&operation);
  if (!llvm::any_of(ordered, [](Operation *operation) {
        return isa<StoreOp>(operation);
      })) {
    reason = "region-scan output assembly has no observable store";
    return failure();
  }
  return success();
}

Value mappedValue(IRMapping &mapping, Value value) {
  if (!value)
    return {};
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  return value;
}

FailureOr<Value> materializeScanConsumerValue(
    OpBuilder &builder, Location location, RegionScanOp scan, Value value,
    IRMapping &mapping, ArrayRef<SliceRelation> relations,
    PhysicalExprAttr sliceExtent,
    Value offset, Value segment, std::string &reason) {
  if (!value)
    return Value();
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getBlock() != scan->getBlock())
    return value;
  bool precedesScan = definition->isBeforeInBlock(scan);
  if (auto range = dyn_cast<MakeRangeOp>(definition)) {
    FailureOr<Value> start = materializeScanConsumerValue(
        builder, location, scan, range.getStart(), mapping, relations,
        sliceExtent, offset, segment, reason);
    FailureOr<Value> extent = materializeScanConsumerValue(
        builder, location, scan, range.getExtent(), mapping, relations,
        sliceExtent, offset, segment, reason);
    FailureOr<Value> step = materializeScanConsumerValue(
        builder, location, scan, range.getStep(), mapping, relations,
        sliceExtent, offset, segment, reason);
    if (failed(start) || failed(extent) || failed(step))
      return failure();
    auto type = cast<FragmentType>(range.getResult().getType());
    auto relation = llvm::find_if(relations, [&](const SliceRelation &candidate) {
      return sourceAxisIdentity(range) == candidate.source;
    });
    bool isSourceAxis = relation != relations.end();
    if (precedesScan && !isSourceAxis && *start == range.getStart() &&
        *extent == range.getExtent() && *step == range.getStep())
      return value;
    Value physicalStart = *start;
    Value physicalExtent = *extent;
    FragmentType physicalType = type;
    if (isSourceAxis) {
      physicalStart = builder.create<BinaryOp>(location, builder.getIndexType(),
                                               physicalStart, offset,
                                               BinaryOperator::Add);
      physicalExtent = segment;
      physicalType = replaceSliceAxis(type, 0, sliceExtent,
                                      relation->segmentMapping);
    }
    Value result = builder.create<MakeRangeOp>(
        location, physicalType, physicalStart, physicalExtent, *step,
        range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    mapping.map(value, result);
    return result;
  }
  bool changed = false;
  for (Value operand : definition->getOperands()) {
    FailureOr<Value> materialized = materializeScanConsumerValue(
        builder, location, scan, operand, mapping, relations, sliceExtent,
        offset, segment, reason);
    if (failed(materialized))
      return failure();
    changed |= *materialized != operand;
    if (!mapping.lookupOrNull(operand) && *materialized != operand)
      mapping.map(operand, *materialized);
  }
  FailureOr<Type> sliced =
      slicedType(value.getType(), relations, sliceExtent, definition);
  if (failed(sliced)) {
    reason = "region-scan output address has no unique segment relation";
    return failure();
  }
  if (precedesScan && !changed && *sliced == value.getType())
    return value;
  if (!isa<arith::ConstantOp>(definition) &&
      !isScanOutputPureConsumer(definition)) {
    reason =
        "region-scan output address depends on an operation that cannot be moved into the segment loop: " +
        definition->getName().getStringRef().str();
    return failure();
  }
  Operation *clone = builder.clone(*definition, mapping);
  for (auto [original, result] :
       llvm::zip(definition->getResults(), clone->getResults())) {
    FailureOr<Type> type =
        slicedType(result.getType(), relations, sliceExtent, clone);
    if (failed(type))
      return failure();
    result.setType(*type);
    if (!mapping.lookupOrNull(original))
      mapping.map(original, result);
  }
  Value result = mapping.lookupOrNull(value);
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

LogicalResult cloneScanOutputConsumers(
    OpBuilder &builder, Location location, ArrayRef<Operation *> consumers,
    IRMapping &mapping, ArrayRef<SliceRelation> relations,
    PhysicalExprAttr sliceExtent,
    Value segmentTail, RegionScanOp scan, Value offset, Value segment,
    std::string &reason) {
  for (Operation *operation : consumers) {
    for (Value operand : operation->getOperands()) {
      if (mapping.lookupOrNull(operand))
        continue;
      FailureOr<Value> materialized = materializeScanConsumerValue(
          builder, location, scan, operand, mapping, relations, sliceExtent,
          offset, segment, reason);
      if (failed(materialized))
        return failure();
      if (*materialized != operand && !mapping.lookupOrNull(operand))
        mapping.map(operand, *materialized);
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      Value value = mappedValue(mapping, store.getValue());
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment) {
        reason = "region-scan slice store value is not a fragment";
        return failure();
      }
      Value valid = mappedValue(mapping, store.getValid());
      Value tail = segmentTail;
      FragmentType predicate = predicateType(fragment);
      if (tail.getType() != predicate)
        tail = builder.create<BroadcastOp>(location, predicate, tail);
      if (valid) {
        if (valid.getType() != predicate)
          valid = builder.create<BroadcastOp>(location, predicate, valid);
        valid = builder.create<BinaryOp>(location, predicate, valid, tail,
                                         BinaryOperator::LogicalAnd);
      } else {
        valid = tail;
      }
      SmallVector<Value> coordinates;
      for (Value coordinate : store.getCoordinates())
        coordinates.push_back(mappedValue(mapping, coordinate));
      auto replacement = builder.create<StoreOp>(
          location, mappedValue(mapping, store.getResource()), coordinates,
          value, valid, store.getSourceAxes());
      if (Attribute origin = store->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
      continue;
    }
    Operation *clone = builder.clone(*operation, mapping);
    for (auto [original, result] :
         llvm::zip(operation->getResults(), clone->getResults())) {
      FailureOr<Type> type =
          slicedType(result.getType(), relations, sliceExtent, clone);
      if (failed(type)) {
        reason = "region-scan output consumer has no sliced physical type";
        return failure();
      }
      result.setType(*type);
      if (!mapping.lookupOrNull(original))
        mapping.map(original, result);
    }
  }
  return success();
}

struct PredicatePartition {
  Value allTrueStart;
  Value allTrueStop;
  Value effectiveStart;
  Value effectiveStop;
  SmallVector<Value> allTruePredicates;
  bool firstMemberIsActive = false;
  bool prefixSpecializable = false;
};

FailureOr<PredicatePartition>
predicatePartition(OpBuilder &builder, RegionFoldOp fold,
                   ArrayRef<SourcePlan> plans, MakeRangeOp master,
                   Value masterExtent, ValueRange identities, Value segment) {
  auto yield = dyn_cast<YieldOp>(fold.getSummarize().front().getTerminator());
  if (!yield || yield.getValues().size() != identities.size())
    return failure();
  auto kernel = fold->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  auto uniqueRange = [&](Value value) -> MakeRangeOp {
    PhysicalRangeFact fact = analysis.sourceRanges(value);
    FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
    return succeeded(range) ? *range : MakeRangeOp();
  };
  unsigned sourceCount = fold.getSourceCount();
  ValueRange captures = fold.getInputs().drop_front(
      fold.getSourceCount() + fold.getIdentityCount());
  auto boundRanges = [&](BlockArgument source,
                         BlockArgument capture)
      -> std::optional<std::pair<MakeRangeOp, MakeRangeOp>> {
    if (!source || !capture || source.getArgNumber() >= sourceCount ||
        capture.getArgNumber() < sourceCount ||
        source.getArgNumber() >= plans.size())
      return std::nullopt;
    unsigned captureIndex = capture.getArgNumber() - sourceCount;
    if (captureIndex >= captures.size())
      return std::nullopt;
    Value sourceCoordinate =
        stripHelperForwarding(plans[source.getArgNumber()].source);
    Value captureCoordinate = stripHelperForwarding(captures[captureIndex]);
    if (!sourceCoordinate.getDefiningOp<MakeRangeOp>() ||
        !captureCoordinate.getDefiningOp<MakeRangeOp>())
      return std::nullopt;
    MakeRangeOp captureRange = uniqueRange(captures[captureIndex]);
    MakeRangeOp comparedSource =
        uniqueRange(plans[source.getArgNumber()].source);
    auto comparedType = comparedSource
                            ? dyn_cast<FragmentType>(
                                  comparedSource.getResult().getType())
                            : FragmentType();
    auto masterType = dyn_cast<FragmentType>(master.getResult().getType());
    if (!captureRange || !comparedSource || !isUnitStepRange(captureRange) ||
        !isUnitStepRange(comparedSource) || !comparedType || !masterType ||
        comparedType.getShape().size() != 1 ||
        masterType.getShape().size() != 1 ||
        comparedType.getShape()[0] != masterType.getShape()[0])
      return std::nullopt;
    return std::make_pair(captureRange, comparedSource);
  };
  auto materializeHelperScalar = [&](Value helperScalar) -> FailureOr<Value> {
    helperScalar = stripHelperForwarding(helperScalar);
    if (!helperScalar)
      return failure();
    Value scalar;
    if (auto argument = dyn_cast<BlockArgument>(helperScalar)) {
      if (argument.getArgNumber() < sourceCount)
        return failure();
      unsigned captureIndex = argument.getArgNumber() - sourceCount;
      if (captureIndex >= captures.size())
        return failure();
      scalar = captures[captureIndex];
    } else if (auto constant =
                   helperScalar.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer)
        return failure();
      scalar = builder.create<arith::ConstantIndexOp>(fold.getLoc(),
                                                       integer.getInt());
    } else {
      return failure();
    }
    if (!scalar.getType().isIntOrIndex())
      return failure();
    if (!isa<IndexType>(scalar.getType()))
      scalar = builder.create<CastOp>(fold.getLoc(), builder.getIndexType(),
                                      scalar);
    return scalar;
  };

  Location location = fold.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value effectiveStart = zero;
  Value effectiveStop = masterExtent;
  Value allTrueStart = zero;
  Value allTrueStop = masterExtent;
  SmallVector<Value> allTruePredicates;
  bool firstMemberIsActive = false;
  bool hasLowerBound = false;
  unsigned upperBoundCount = 0;
  bool foundBound = false;
  for (CompareOp compare : fold.getSummarize().front().getOps<CompareOp>()) {
    BlockArgument lhs = rootHelperArgument(compare.getLhs());
    BlockArgument rhs = rootHelperArgument(compare.getRhs());
    bool identityOnly = true;
    for (auto [summary, identity] : llvm::zip(yield.getValues(), identities))
      identityOnly &=
          equalWhenPredicateIsFalse(summary, identity, compare.getResult());
    if (!identityOnly)
      continue;

    BlockArgument upperSource;
    BlockArgument upperCapture;
    bool strictUpper = false;
    switch (compare.getPredicate()) {
    case ComparePredicate::Le:
    case ComparePredicate::Lt:
      upperSource = lhs;
      upperCapture = rhs;
      strictUpper = compare.getPredicate() == ComparePredicate::Lt;
      break;
    case ComparePredicate::Ge:
    case ComparePredicate::Gt:
      upperSource = rhs;
      upperCapture = lhs;
      strictUpper = compare.getPredicate() == ComparePredicate::Gt;
      break;
    default:
      break;
    }
    if (auto ranges = boundRanges(upperSource, upperCapture)) {
      MakeRangeOp captureRange = ranges->first;
      MakeRangeOp comparedSource = ranges->second;
      Value captureWidth = builder.create<BinaryOp>(
          location, builder.getIndexType(), captureRange.getExtent(),
          captureRange.getStep(), BinaryOperator::Multiply);
      Value captureUpper = builder.create<BinaryOp>(
          location, builder.getIndexType(), captureRange.getStart(),
          captureWidth, BinaryOperator::Add);
      if (strictUpper)
        captureUpper = builder.create<BinaryOp>(
            location, builder.getIndexType(), captureUpper,
            captureRange.getStep(), BinaryOperator::Subtract);
      Value relativeUpper = builder.create<BinaryOp>(
          location, builder.getIndexType(), captureUpper, master.getStart(),
          BinaryOperator::Subtract);
      Value nonNegative = builder.create<BinaryOp>(
          location, builder.getIndexType(), relativeUpper, zero,
          BinaryOperator::Maximum);
      Value candidateStop = builder.create<BinaryOp>(
          location, builder.getIndexType(), masterExtent, nonNegative,
          BinaryOperator::Minimum);
      effectiveStop = builder.create<BinaryOp>(
          location, builder.getIndexType(), effectiveStop, candidateStop,
          BinaryOperator::Minimum);
      {
        Value relativeStart = builder.create<BinaryOp>(
            location, builder.getIndexType(), captureRange.getStart(),
            master.getStart(), BinaryOperator::Subtract);
        Value nonNegativeStart = builder.create<BinaryOp>(
            location, builder.getIndexType(), relativeStart, zero,
            BinaryOperator::Maximum);
        Value boundedStart = builder.create<BinaryOp>(
            location, builder.getIndexType(), masterExtent, nonNegativeStart,
            BinaryOperator::Minimum);
        Value wholeSegments = builder.create<BinaryOp>(
            location, builder.getIndexType(), boundedStart, segment,
            BinaryOperator::FloorDivide);
        Value candidateAllTrueStop = builder.create<BinaryOp>(
            location, builder.getIndexType(), wholeSegments, segment,
            BinaryOperator::Multiply);
        allTrueStop = builder.create<BinaryOp>(
            location, builder.getIndexType(), allTrueStop, candidateAllTrueStop,
            BinaryOperator::Minimum);
        allTruePredicates.push_back(compare.getResult());
        if (upperBoundCount++ == 0) {
          firstMemberIsActive =
              samePhysicalScalarExpression(master.getStart(),
                                           master.getLogicalStart()) &&
              samePhysicalScalarExpression(captureRange.getLogicalStart(),
                                           master.getStart()) &&
              samePhysicalScalarExpression(comparedSource.getStart(),
                                           master.getStart());
        }
      }
      foundBound = true;
    }

    BlockArgument lowerSource;
    FailureOr<HelperCoordinateExpression> lowerCapture = failure();
    bool strictLower = false;
    switch (compare.getPredicate()) {
    case ComparePredicate::Ge:
    case ComparePredicate::Gt:
      lowerSource = lhs;
      lowerCapture = helperCoordinateExpression(compare.getRhs());
      strictLower = compare.getPredicate() == ComparePredicate::Gt;
      break;
    case ComparePredicate::Le:
    case ComparePredicate::Lt:
      lowerSource = rhs;
      lowerCapture = helperCoordinateExpression(compare.getLhs());
      strictLower = compare.getPredicate() == ComparePredicate::Lt;
      break;
    default:
      break;
    }
    if (failed(lowerCapture))
      continue;
    auto ranges = boundRanges(lowerSource, lowerCapture->coordinate);
    if (!ranges)
      continue;
    MakeRangeOp captureRange = ranges->first;
    MakeRangeOp comparedSource = ranges->second;
    Value lower = captureRange.getStart();
    if (lowerCapture->scalar) {
      FailureOr<Value> scalar =
          materializeHelperScalar(lowerCapture->scalar);
      if (failed(scalar))
        continue;
      lower = builder.create<BinaryOp>(
          location, builder.getIndexType(), lower, *scalar,
          lowerCapture->subtractScalar ? BinaryOperator::Subtract
                                       : BinaryOperator::Add);
    }
    if (strictLower)
      lower = builder.create<BinaryOp>(
          location, builder.getIndexType(), lower, comparedSource.getStep(),
          BinaryOperator::Add);
    Value relativeLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), lower, master.getStart(),
        BinaryOperator::Subtract);
    Value nonNegativeLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), relativeLower, zero,
        BinaryOperator::Maximum);
    Value boundedLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), masterExtent, nonNegativeLower,
        BinaryOperator::Minimum);
    Value wholeSegments = builder.create<BinaryOp>(
        location, builder.getIndexType(), boundedLower, segment,
        BinaryOperator::FloorDivide);
    Value alignedLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), wholeSegments, segment,
        BinaryOperator::Multiply);
    effectiveStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), effectiveStart, alignedLower,
        BinaryOperator::Maximum);
    Value captureSpan = builder.create<BinaryOp>(
        location, builder.getIndexType(), captureRange.getExtent(),
        captureRange.getStep(), BinaryOperator::Multiply);
    Value lastCaptureOffset = builder.create<BinaryOp>(
        location, builder.getIndexType(), captureSpan, captureRange.getStep(),
        BinaryOperator::Subtract);
    Value lastLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), relativeLower, lastCaptureOffset,
        BinaryOperator::Add);
    Value nonNegativeLastLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), lastLower, zero,
        BinaryOperator::Maximum);
    Value boundedLastLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), masterExtent, nonNegativeLastLower,
        BinaryOperator::Minimum);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value adjustment = builder.create<BinaryOp>(
        location, builder.getIndexType(), segment, one,
        BinaryOperator::Subtract);
    Value roundedLastLower = builder.create<BinaryOp>(
        location, builder.getIndexType(), boundedLastLower, adjustment,
        BinaryOperator::Add);
    Value firstWholeSegment = builder.create<BinaryOp>(
        location, builder.getIndexType(), roundedLastLower, segment,
        BinaryOperator::FloorDivide);
    Value candidateAllTrueStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), firstWholeSegment, segment,
        BinaryOperator::Multiply);
    allTrueStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), allTrueStart, candidateAllTrueStart,
        BinaryOperator::Maximum);
    allTruePredicates.push_back(compare.getResult());
    hasLowerBound = true;
    foundBound = true;
  }
  if (!foundBound)
    return failure();
  return PredicatePartition{allTrueStart, allTrueStop, effectiveStart,
                            effectiveStop,
                            std::move(allTruePredicates), firstMemberIsActive,
                            upperBoundCount == 1 && !hasLowerBound};
}

void foldKnownRecordProjections(Operation *structured) {
  for (Region &region : structured->getRegions())
    region.walk([&](ExtractOp extract) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        extract.getResult().replaceAllUsesWith(
            record.getFields()[extract.getField()]);
        extract.erase();
      }
    });
}

LogicalResult realizeFold(RegionFoldOp fold, func::FuncOp kernel) {
  foldKnownRecordProjections(fold);
  ParameterOp segment = findParameter(kernel, fold.getSegment());
  if (!segment)
    return fold.emitOpError("region-fold segment parameter is not declared");
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  SmallVector<SourcePlan> plans;
  SmallVector<Value> sources;
  SmallVector<unsigned> sourceAxes;
  for (Value source : fold.getInputs().take_front(fold.getSourceCount())) {
    FailureOr<SourcePlan> plan =
        analyzeSource(source, fold.getAxis(), physicalAnalysis);
    if (failed(plan))
      return fold.emitOpError(
          "region-fold source is not a sliceable unit-step physical value graph");
    plans.push_back(*plan);
    sources.push_back(source);
    sourceAxes.push_back(fold.getAxis());
  }
  PhysicalLockstepTraversalFact traversal =
      physicalAnalysis.lockstepTraversal(sources, sourceAxes);
  if (!traversal.isExact()) {
    InFlightDiagnostic diagnostic = fold.emitOpError(
        traversal.state == PhysicalLockstepState::Inconsistent
            ? "region sources have inconsistent physical traversals"
            : "region source traversal is not exactly known");
    for (Operation *blocker : traversal.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  SmallVector<FragmentType> sliceTypes;
  for (BlockArgument argument :
       fold.getSummarize().front().getArguments().take_front(
           fold.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(argument.getType());
    if (!fragment)
      return fold.emitOpError(
          "region-fold source slice has no physical fragment schema");
    sliceTypes.push_back(fragment);
  }

  SmallVector<AssumeInBoundsOp> sourceAssumptions;
  DominanceInfo dominance(kernel);
  kernel.walk([&](AssumeInBoundsOp assumption) {
    if (!dominance.properlyDominates(assumption.getOperation(),
                                    fold.getOperation()))
      return;
    for (const SourcePlan &plan : plans) {
      PhysicalRangeFact fact = physicalAnalysis.sourceRanges(
          assumption.getIndex(), plan.sourceIdentity);
      FailureOr<MakeRangeOp> range = queryExactLogicalRange(fact);
      if (failed(range) ||
          llvm::none_of(fact.roots, [&](MakeRangeOp root) {
            return llvm::any_of(plan.ranges, [&](MakeRangeOp planned) {
              return sameLogicalRange(planned, root);
            });
          }))
        continue;
      sourceAssumptions.push_back(assumption);
      break;
    }
  });

  MakeRangeOp master = traversal.authority;
  OpBuilder builder(fold);
  Location location = fold.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  FailureOr<Value> logicalEnd = resolveLogicalRangeEnd(kernel, master);
  if (failed(logicalEnd))
    return fold.emitOpError(
        "region-fold traversal has no exact logical upper bound");
  Value stop = builder.create<BinaryOp>(
      location, builder.getIndexType(), *logicalEnd, master.getStart(),
      BinaryOperator::Subtract);
  PhysicalExprAttr sliceExtent = parameterExtent(fold.getSegment());
  SmallVector<Value> identities(
      fold.getInputs()
          .slice(fold.getSourceCount(), fold.getIdentityCount())
          .begin(),
      fold.getInputs()
          .slice(fold.getSourceCount(), fold.getIdentityCount())
          .end());
  ValueRange captures = fold.getInputs().drop_front(
      fold.getSourceCount() + fold.getIdentityCount());
  FailureOr<PredicatePartition> partition = predicatePartition(
      builder, fold, plans, master, stop, identities, segment.getResult());
  bool specializePredicatePrefix =
      succeeded(partition) && partition->prefixSpecializable;
  Value memberPredicate =
      summaryMembershipPredicate(fold, identities, fold.getSegment());
  if (!memberPredicate)
    return fold.emitOpError(
        "region-fold summarizer has no typed membership predicate that makes a physical tail equal to identity");
  std::optional<SummaryEmptinessPlan> emptiness =
      summaryEmptinessPlan(fold, identities, memberPredicate);
  std::optional<OnlineRegionPlan> online = onlineRegionPlan(fold, emptiness);

  bool bodyFailed = false;
  std::string failureReason;
  auto emitSummary = [&](OpBuilder &nested, Location nestedLocation,
                         Value offset,
                         bool fullSegment,
                         bool predicateIsTrue,
                         bool summaryIsNonempty,
                         IRMapping *summaryMapping)
      -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> slices;
    Value segmentTail;
    IRMapping sliceMapping;
    SmallVector<std::shared_ptr<IRMapping>> sourceMappings;
    if (failed(buildSourceSlices(nested, nestedLocation, plans, sliceTypes,
                                 offset,
                                 segment.getResult(), sliceExtent,
                                 fullSegment, slices,
                                 segmentTail, sliceMapping, sourceMappings,
                                 failureReason)))
      return failure();
    for (AssumeInBoundsOp assumption : sourceAssumptions) {
      SmallVector<Value> assertedIndices;
      for (const std::shared_ptr<IRMapping> &sourceMapping : sourceMappings) {
        Value index = sourceMapping->lookupOrNull(assumption.getIndex());
        if (!index || llvm::is_contained(assertedIndices, index))
          continue;
        assertedIndices.push_back(index);
        OpBuilder::InsertionGuard guard(nested);
        nested.setInsertionPointAfter(index.getDefiningOp());
        auto replacement = nested.create<AssumeInBoundsOp>(
            nestedLocation, index, assumption.getResource(),
            assumption.getAxis());
        if (Attribute origin = assumption->getAttr(originAttr))
          replacement->setAttr(originAttr, origin);
      }
    }
    SmallVector<Value> summarizeArguments(slices);
    summarizeArguments.append(captures.begin(), captures.end());
    IRMapping localSummaryMapping;
    IRMapping &mapping = summaryMapping ? *summaryMapping : localSummaryMapping;
    if (predicateIsTrue) {
      for (Value predicate : partition->allTruePredicates) {
        auto predicateType = dyn_cast<FragmentType>(predicate.getType());
        if (!predicateType) {
          failureReason = "range predicate is not a physical fragment";
          return failure();
        }
        Value truth = nested.create<arith::ConstantOp>(
            nestedLocation, nested.getI1Type(), nested.getBoolAttr(true));
        mapping.map(predicate,
                    nested.create<SplatOp>(nestedLocation, predicateType, truth));
      }
    }
    Value nonemptySource;
    Value nonemptyTarget;
    if (summaryIsNonempty && emptiness) {
      nonemptySource = emptiness->summarizeValidity;
      Type validityType =
          cast<TypeAttr>(emptiness->fullType.getFieldTypes()[
              emptiness->optionalField])
              .getValue();
      nonemptyTarget = allTrueValue(nested, nestedLocation, validityType);
    }
    return inlinePureRegion(nested, fold.getSummarize(), summarizeArguments,
                            failureReason, {}, {},
                            fullSegment ? Value() : memberPredicate,
                            fullSegment ? Value() : segmentTail,
                            nonemptySource, nonemptyTarget, &mapping);
  };
  auto emitLoop = [&](Value lower, Value upper, ValueRange initial,
                      bool fullSegment, bool predicateIsTrue) {
    auto loop = builder.create<scf::ForOp>(
      location, lower, upper, segment.getResult(), initial,
      [&](OpBuilder &nested, Location nestedLocation, Value offset,
          ValueRange carries) {
        IRMapping summaryMapping;
        FailureOr<SmallVector<Value>> summary = emitSummary(
            nested, nestedLocation, offset, fullSegment, predicateIsTrue,
            /*summaryIsNonempty=*/false, online ? &summaryMapping : nullptr);
        if (failed(summary)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(summary->begin(), summary->end());
        IRMapping mergeMapping;
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, fold.getCombine(), combineArguments, failureReason,
            {}, {}, {}, {}, {}, {}, online ? &mergeMapping : nullptr);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        if (online) {
          FailureOr<Value> direct = coRealizeOnlineRegion(
              nested, nestedLocation, *online, summaryMapping, mergeMapping);
          if (succeeded(direct))
            combined->front() = *direct;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      });
    if (Attribute origin = fold->getAttr(originAttr))
      loop->setAttr(originAttr, origin);
    return loop;
  };

  auto finish = [&]() -> LogicalResult {
    for (AssumeInBoundsOp assumption : sourceAssumptions)
      assumption.erase();
    simplifyKnownRecordValues(kernel);
    eraseDeadPhysicalValues(kernel);
    return success();
  };

  if (segment->hasAttr(coverageDimensionAttr)) {
    FailureOr<SmallVector<Value>> summary =
        emitSummary(builder, location, zero, /*fullSegment=*/false,
                    /*predicateIsTrue=*/false,
                    /*summaryIsNonempty=*/false,
                    /*summaryMapping=*/nullptr);
    if (failed(summary))
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    for (auto [oldResult, newResult] :
         llvm::zip(fold.getResults(), *summary))
      oldResult.replaceAllUsesWith(newResult);
    fold.erase();
    return finish();
  }

  if (emptiness && specializePredicatePrefix &&
      partition->firstMemberIsActive) {
    stop = partition->effectiveStop;
    Value nonempty = builder.create<CompareOp>(
        location, builder.getI1Type(), stop, zero, ComparePredicate::Gt);
    auto conditional = builder.create<scf::IfOp>(
        location, TypeRange{emptiness->payloadType}, nonempty,
        /*withElseRegion=*/true);
    auto prepareBranch = [](Region &region) {
      Block &block = region.front();
      if (!block.empty() && isa<scf::YieldOp>(block.back()))
        block.back().erase();
      return OpBuilder(&block, block.end());
    };

    OpBuilder nonemptyBuilder = prepareBranch(conditional.getThenRegion());
    FailureOr<SmallVector<Value>> first = emitSummary(
        nonemptyBuilder, location, zero, /*fullSegment=*/false,
        /*predicateIsTrue=*/false, /*summaryIsNonempty=*/true,
        /*summaryMapping=*/nullptr);
    if (succeeded(first) && first->size() != 1) {
      failureReason =
          "summary-emptiness realization requires one summary record";
      bodyFailed = true;
    }
    FailureOr<Value> payload = failure();
    if (succeeded(first) && !bodyFailed)
      payload = stripOptionalRecord(nonemptyBuilder, location, first->front(),
                                    *emptiness);
    if (failed(first) || failed(payload))
      bodyFailed = true;

    if (!bodyFailed)
      nonemptyBuilder.create<scf::YieldOp>(location, *payload);
    OpBuilder emptyBuilder = prepareBranch(conditional.getElseRegion());
    FailureOr<Value> emptyPayload = stripOptionalRecord(
        emptyBuilder, location, identities.front(), *emptiness);
    if (failed(emptyPayload))
      bodyFailed = true;
    else
      emptyBuilder.create<scf::YieldOp>(location, *emptyPayload);
    if (bodyFailed) {
      conditional.erase();
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    }

    auto emitPayloadLoop = [&](Value lower, Value upper, Value initial,
                               bool fullSegment,
                               bool predicateIsTrue,
                               bool summaryIsNonempty) -> scf::ForOp {
      auto loop = builder.create<scf::ForOp>(
          location, lower, upper, segment.getResult(), ValueRange{initial},
          [&](OpBuilder &nested, Location nestedLocation, Value offset,
              ValueRange carries) {
            IRMapping summaryMapping;
            FailureOr<SmallVector<Value>> summary = emitSummary(
                nested, nestedLocation, offset, fullSegment, predicateIsTrue,
                summaryIsNonempty,
                online ? &summaryMapping : nullptr);
            if (failed(summary) || summary->size() != 1 ||
                carries.size() != 1) {
              if (succeeded(summary))
                failureReason =
                    "summary-emptiness loop requires one summary and one payload carry";
              bodyFailed = true;
              return;
            }
            Value fullCarry = restoreOptionalRecord(
                nested, nestedLocation, carries.front(), *emptiness);
            IRMapping mergeMapping;
            FailureOr<SmallVector<Value>> combined = inlinePureRegion(
                nested, fold.getCombine(),
                ValueRange{fullCarry, summary->front()}, failureReason,
                {}, {}, {}, {}, {}, {}, online ? &mergeMapping : nullptr);
            if (failed(combined) || combined->size() != 1) {
              if (succeeded(combined))
                failureReason =
                    "summary-emptiness combine requires one summary record";
              bodyFailed = true;
              return;
            }
            Value combinedValue = combined->front();
            if (online) {
              FailureOr<Value> direct = coRealizeOnlineRegion(
                  nested, nestedLocation, *online, summaryMapping,
                  mergeMapping);
              if (succeeded(direct))
                combinedValue = *direct;
            }
            FailureOr<Value> next = stripOptionalRecord(
                nested, nestedLocation, combinedValue, *emptiness);
            if (failed(next)) {
              bodyFailed = true;
              return;
            }
            nested.create<scf::YieldOp>(nestedLocation, *next);
          });
      if (Attribute origin = fold->getAttr(originAttr))
        loop->setAttr(originAttr, origin);
      return loop;
    };

    scf::ForOp allTrueLoop;
    scf::ForOp fullMixedLoop;
    scf::ForOp tailLoop;
    Value current = conditional.getResult(0);
    allTrueLoop = emitPayloadLoop(
        segment.getResult(), partition->allTrueStop, current,
        /*fullSegment=*/true, /*predicateIsTrue=*/true,
        /*summaryIsNonempty=*/true);
    current = allTrueLoop.getResult(0);
    Value mixedStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), partition->allTrueStop,
        segment.getResult(), BinaryOperator::Maximum);
    Value fullMixedSegments = builder.create<BinaryOp>(
        location, builder.getIndexType(), stop, segment.getResult(),
        BinaryOperator::FloorDivide);
    Value fullMixedStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedSegments,
        segment.getResult(), BinaryOperator::Multiply);
    if (!bodyFailed) {
      fullMixedLoop = emitPayloadLoop(
          mixedStart, fullMixedStop, current, /*fullSegment=*/true,
          /*predicateIsTrue=*/false,
          /*summaryIsNonempty=*/false);
      current = fullMixedLoop.getResult(0);
    }
    Value tailStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedStop,
        segment.getResult(), BinaryOperator::Maximum);
    if (!bodyFailed) {
      tailLoop = emitPayloadLoop(
          tailStart, stop, current, /*fullSegment=*/false,
          /*predicateIsTrue=*/false,
          /*summaryIsNonempty=*/false);
      current = tailLoop.getResult(0);
    }
    if (bodyFailed) {
      if (allTrueLoop && allTrueLoop->getBlock())
        allTrueLoop.erase();
      if (fullMixedLoop && fullMixedLoop->getBlock())
        fullMixedLoop.erase();
      if (tailLoop && tailLoop->getBlock())
        tailLoop.erase();
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    }
    Type validityType = cast<TypeAttr>(emptiness->fullType.getFieldTypes()[
                                           emptiness->optionalField])
                            .getValue();
    Value validity = nonempty;
    if (isa<FragmentType>(validityType))
      validity = builder.create<SplatOp>(location, validityType, validity);
    Value result = restoreOptionalRecord(builder, location, current, *emptiness,
                                         validity);
    fold.getResult(0).replaceAllUsesWith(result);
    fold.erase();
    return finish();
  }

  SmallVector<Value> current(identities.begin(), identities.end());
  scf::ForOp allTrueLoop;
  if (specializePredicatePrefix) {
    allTrueLoop = emitLoop(zero, partition->allTrueStop, current,
                           /*fullSegment=*/true,
                           /*predicateIsTrue=*/true);
    if (!bodyFailed)
      current.assign(allTrueLoop.getResults().begin(),
                     allTrueLoop.getResults().end());
  }
  Value mixedStart = specializePredicatePrefix
                         ? partition->allTrueStop
                         : succeeded(partition) ? partition->effectiveStart
                                                : zero;
  stop = succeeded(partition) ? partition->effectiveStop : stop;
  Value fullMixedSegments = builder.create<BinaryOp>(
      location, builder.getIndexType(), stop, segment.getResult(),
      BinaryOperator::FloorDivide);
  Value fullMixedStop = builder.create<BinaryOp>(
      location, builder.getIndexType(), fullMixedSegments, segment.getResult(),
      BinaryOperator::Multiply);
  scf::ForOp leadingMixedLoop;
  if (!specializePredicatePrefix && succeeded(partition)) {
    Value boundedStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), mixedStart, partition->allTrueStart,
        BinaryOperator::Maximum);
    Value interiorStart = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedStop, boundedStart,
        BinaryOperator::Minimum);
    Value boundedStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), interiorStart, partition->allTrueStop,
        BinaryOperator::Maximum);
    Value interiorStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), fullMixedStop, boundedStop,
        BinaryOperator::Minimum);
    leadingMixedLoop = emitLoop(mixedStart, interiorStart, current,
                                /*fullSegment=*/true,
                                /*predicateIsTrue=*/false);
    if (!bodyFailed) {
      current.assign(leadingMixedLoop.getResults().begin(),
                     leadingMixedLoop.getResults().end());
      allTrueLoop = emitLoop(interiorStart, interiorStop, current,
                             /*fullSegment=*/true,
                             /*predicateIsTrue=*/true);
    }
    if (!bodyFailed)
      current.assign(allTrueLoop.getResults().begin(),
                     allTrueLoop.getResults().end());
    mixedStart = interiorStop;
  }
  scf::ForOp fullMixedLoop;
  if (!bodyFailed)
    fullMixedLoop = emitLoop(mixedStart, fullMixedStop, current,
                             /*fullSegment=*/true,
                             /*predicateIsTrue=*/false);
  if (!bodyFailed)
    current.assign(fullMixedLoop.getResults().begin(),
                   fullMixedLoop.getResults().end());
  scf::ForOp tailLoop;
  if (!bodyFailed)
    tailLoop = emitLoop(fullMixedStop, stop, current,
                        /*fullSegment=*/false,
                        /*predicateIsTrue=*/false);
  if (bodyFailed) {
    if (leadingMixedLoop && leadingMixedLoop->getBlock())
      leadingMixedLoop.erase();
    if (allTrueLoop && allTrueLoop->getBlock())
      allTrueLoop.erase();
    if (fullMixedLoop && fullMixedLoop->getBlock())
      fullMixedLoop.erase();
    if (tailLoop && tailLoop->getBlock())
      tailLoop.erase();
    return fold.emitOpError("region-fold physicalization failed: ")
           << failureReason;
  }
  for (auto [oldResult, newResult] :
       llvm::zip(fold.getResults(), tailLoop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  fold.erase();
  return finish();
}

LogicalResult realizeScan(RegionScanOp scan, func::FuncOp kernel) {
  foldKnownRecordProjections(scan);
  ParameterOp segment = findParameter(kernel, scan.getSegment());
  if (!segment)
    return scan.emitOpError("region-scan segment parameter is not declared");
  PhysicalProgramAnalysis physicalAnalysis(kernel);
  SmallVector<SourcePlan> plans;
  SmallVector<Value> sources;
  SmallVector<unsigned> sourceAxes;
  for (Value source : scan.getInputs().take_front(scan.getSourceCount())) {
    FailureOr<SourcePlan> plan =
        analyzeSource(source, scan.getAxis(), physicalAnalysis);
    if (failed(plan))
      return scan.emitOpError(
          "region-scan source is not a sliceable unit-step physical value graph");
    plans.push_back(*plan);
    sources.push_back(source);
    sourceAxes.push_back(scan.getAxis());
  }
  PhysicalLockstepTraversalFact traversal =
      physicalAnalysis.lockstepTraversal(sources, sourceAxes);
  if (!traversal.isExact()) {
    InFlightDiagnostic diagnostic = scan.emitOpError(
        traversal.state == PhysicalLockstepState::Inconsistent
            ? "region sources have inconsistent physical traversals"
            : "region source traversal is not exactly known");
    for (Operation *blocker : traversal.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  SmallVector<FragmentType> sliceTypes;
  for (BlockArgument argument :
       scan.getSummarize().front().getArguments().take_front(
           scan.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(argument.getType());
    if (!fragment)
      return scan.emitOpError(
          "region-scan source slice has no physical fragment schema");
    sliceTypes.push_back(fragment);
  }
  SmallVector<SliceRelation> outputRelations;
  for (auto [plan, sliceType] : llvm::zip(plans, sliceTypes)) {
    auto mapping = cast<AxisMapAttr>(
        sliceType.getAxisMaps()[plan.sourceAxis]);
    auto relation = llvm::find_if(
        outputRelations, [&](const SliceRelation &candidate) {
          return candidate.source == plan.sourceIdentity;
        });
    if (relation == outputRelations.end()) {
      outputRelations.push_back({plan.sourceIdentity, mapping});
      continue;
    }
    if (relation->segmentMapping != mapping)
      return scan.emitOpError(
          "region-scan source has inconsistent helper-local slice relations");
  }
  MakeRangeOp master = traversal.authority;
  ValueRange identities = scan.getInputs().slice(scan.getSourceCount(),
                                                 scan.getIdentityCount());
  if (!scanTailIsIdentity(scan, plans, identities))
    return scan.emitOpError(
        "region-scan summarizer does not prove that zero-filled physical tail members produce its transition identity");

  SmallVector<Operation *> outputConsumers;
  std::string failureReason;
  if (failed(collectScanOutputConsumers(scan, outputConsumers, failureReason)))
    return scan.emitOpError("region-scan physicalization failed: ")
           << failureReason;

  unsigned stateOffset = scan.getSourceCount() + scan.getIdentityCount();
  ValueRange initialStates =
      scan.getInputs().slice(stateOffset, scan.getStateCount());
  ValueRange captures = scan.getInputs().drop_front(stateOffset +
                                                    scan.getStateCount());
  OpBuilder builder(scan);
  Location location = scan.getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  FailureOr<Value> logicalEnd = resolveLogicalRangeEnd(kernel, master);
  if (failed(logicalEnd))
    return scan.emitOpError(
        "region-scan traversal has no exact logical upper bound");
  Value traversalExtent = builder.create<BinaryOp>(
      location, builder.getIndexType(), *logicalEnd, master.getStart(),
      BinaryOperator::Subtract);
  PhysicalExprAttr sliceExtent = parameterExtent(scan.getSegment());
  bool bodyFailed = false;
  auto loop = builder.create<scf::ForOp>(
      location, zero, traversalExtent, segment.getResult(), identities,
      [&](OpBuilder &nested, Location nestedLocation, Value offset,
          ValueRange prefix) {
        SmallVector<Value> slices;
        Value segmentTail;
        IRMapping sliceMapping;
        SmallVector<std::shared_ptr<IRMapping>> sourceMappings;
        if (failed(buildSourceSlices(
                nested, nestedLocation, plans, sliceTypes, offset,
                segment.getResult(),
                sliceExtent, /*fullSegment=*/false, slices, segmentTail,
                sliceMapping, sourceMappings,
                failureReason))) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> summarizeArguments(slices);
        summarizeArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> summary = inlinePureRegion(
            nested, scan.getSummarize(), summarizeArguments, failureReason);
        if (failed(summary)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> applyArguments(prefix.begin(), prefix.end());
        applyArguments.append(initialStates.begin(), initialStates.end());
        FailureOr<SmallVector<Value>> incoming = inlinePureRegion(
            nested, scan.getApply(), applyArguments, failureReason);
        if (failed(incoming)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> emitArguments(slices);
        emitArguments.append(incoming->begin(), incoming->end());
        emitArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> emitted = inlinePureRegion(
            nested, scan.getEmit(), emitArguments, failureReason);
        if (failed(emitted)) {
          bodyFailed = true;
          return;
        }
        for (auto [output, slice] : llvm::zip(
                 scan.getResults().take_front(scan.getOutputCount()), *emitted))
          sliceMapping.map(output, slice);
        if (failed(cloneScanOutputConsumers(
                nested, nestedLocation, outputConsumers, sliceMapping,
                outputRelations, sliceExtent, segmentTail,
                scan, offset, segment.getResult(), failureReason))) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> combineArguments(prefix.begin(), prefix.end());
        combineArguments.append(summary->begin(), summary->end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, scan.getCombine(), combineArguments, failureReason);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      });
  if (bodyFailed) {
    if (loop->getBlock())
      loop.erase();
    return scan.emitOpError("region-scan physicalization failed: ")
           << failureReason;
  }
  if (Attribute origin = scan->getAttr(originAttr))
    loop->setAttr(originAttr, origin);

  SmallVector<Value> finalArguments(loop.getResults().begin(),
                                    loop.getResults().end());
  finalArguments.append(initialStates.begin(), initialStates.end());
  FailureOr<SmallVector<Value>> finalStates = inlinePureRegion(
      builder, scan.getApply(), finalArguments, failureReason);
  if (failed(finalStates)) {
    loop.erase();
    return scan.emitOpError("region-scan final-state physicalization failed: ")
           << failureReason;
  }
  for (auto [oldResult, newResult] : llvm::zip(
           scan.getResults().drop_front(scan.getOutputCount()), *finalStates))
    oldResult.replaceAllUsesWith(newResult);
  for (Operation *consumer : llvm::reverse(outputConsumers))
    consumer->erase();
  scan.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace

LogicalResult realizeRegionFolds(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<RegionFoldOp> folds;
  kernel.walk([&](RegionFoldOp fold) { folds.push_back(fold); });
  for (RegionFoldOp fold : folds)
    if (fold->getBlock() && failed(realizeFold(fold, kernel)))
      return failure();
  return success();
}

LogicalResult realizeRegionScans(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<RegionScanOp> scans;
  kernel.walk([&](RegionScanOp scan) { scans.push_back(scan); });
  for (RegionScanOp scan : scans)
    if (scan->getBlock() && failed(realizeScan(scan, kernel)))
      return failure();
  // Helper inlining substitutes segment-local physical extents throughout the
  // cloned graph.  Close every affected value relation here: a realized scan
  // is a complete physical program transformation, not an invalid intermediate
  // that a later, unrelated pipeline stage is expected to repair.
  if (failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)))
    return failure();
  return success();
}

} // namespace intent::gpu

#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
      Value stopFragment =
          builder.create<BroadcastOp>(location, blockedRange,
                                      range.getLogicalStop());
      auto validComparison = builder.create<CompareOp>(
          location, predicateType(blockedRange), value, stopFragment,
          ComparePredicate::Lt);
      validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
      Value valid = validComparison.getResult();
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
                                               Value conjunctPredicate = {}) {
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
  IRMapping mapping;
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
  for (Operation &operation : region.front().without_terminator()) {
    if (operation.getNumResults() == 1 &&
        mapping.lookupOrNull(operation.getResult(0)))
      continue;
    Operation *clone = builder.clone(operation, mapping);
    bindClonedOperationTypes(clone, extentBindings);
    for (auto [source, result] :
         llvm::zip(operation.getResults(), clone->getResults())) {
      if (mapping.lookupOrNull(source))
        continue;
      if (source == conjunctSource) {
        FailureOr<Value> combined = conjoin(result);
        if (failed(combined))
          return failure();
        mapping.map(source, *combined);
      } else {
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

BlockArgument rootHelperArgument(Value value) {
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
    return {};
  }
  return dyn_cast<BlockArgument>(value);
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
  Value allTrueStop;
  Value effectiveStop;
  Value predicate;
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
  for (CompareOp compare : fold.getSummarize().front().getOps<CompareOp>()) {
    BlockArgument lhs = rootHelperArgument(compare.getLhs());
    BlockArgument rhs = rootHelperArgument(compare.getRhs());
    if (!lhs || !rhs)
      continue;
    unsigned sourceCount = fold.getSourceCount();
    bool lhsCapture = lhs.getArgNumber() >= sourceCount;
    bool rhsCapture = rhs.getArgNumber() >= sourceCount;
    if (lhsCapture == rhsCapture)
      continue;
    unsigned sourceArgument =
        lhsCapture ? rhs.getArgNumber() : lhs.getArgNumber();
    unsigned captureArgument =
        lhsCapture ? lhs.getArgNumber() : rhs.getArgNumber();
    bool sourceHasUpperBound =
        (lhsCapture && compare.getPredicate() == ComparePredicate::Ge) ||
        (rhsCapture && compare.getPredicate() == ComparePredicate::Le);
    if (!sourceHasUpperBound || sourceArgument >= plans.size())
      continue;
    unsigned captureIndex = captureArgument - sourceCount;
    ValueRange captures = fold.getInputs().drop_front(
        fold.getSourceCount() + fold.getIdentityCount());
    if (captureIndex >= captures.size())
      continue;
    MakeRangeOp captureRange = uniqueRange(captures[captureIndex]);
    MakeRangeOp comparedSource = uniqueRange(plans[sourceArgument].source);
    auto comparedType = comparedSource
                            ? dyn_cast<FragmentType>(
                                  comparedSource.getResult().getType())
                            : FragmentType();
    auto masterType = dyn_cast<FragmentType>(master.getResult().getType());
    if (!captureRange || !comparedSource || !isUnitStepRange(captureRange) ||
        !isUnitStepRange(comparedSource) ||
        !comparedType || !masterType || comparedType.getShape().size() != 1 ||
        masterType.getShape().size() != 1 ||
        comparedType.getShape()[0] != masterType.getShape()[0])
      continue;
    bool identityOnly = true;
    for (auto [summary, identity] : llvm::zip(yield.getValues(), identities))
      identityOnly &=
          equalWhenPredicateIsFalse(summary, identity, compare.getResult());
    if (!identityOnly)
      continue;
    Location location = fold.getLoc();
    Value captureWidth = builder.create<BinaryOp>(
        location, builder.getIndexType(), captureRange.getExtent(),
        captureRange.getStep(), BinaryOperator::Multiply);
    Value captureUpper = builder.create<BinaryOp>(
        location, builder.getIndexType(), captureRange.getStart(), captureWidth,
        BinaryOperator::Add);
    Value relativeUpper = builder.create<BinaryOp>(
        location, builder.getIndexType(), captureUpper, master.getStart(),
        BinaryOperator::Subtract);
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value nonNegative = builder.create<BinaryOp>(
        location, builder.getIndexType(), relativeUpper, zero,
        BinaryOperator::Maximum);
    Value effectiveStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), masterExtent, nonNegative,
        BinaryOperator::Minimum);
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
    Value allTrueStop = builder.create<BinaryOp>(
        location, builder.getIndexType(), wholeSegments, segment,
        BinaryOperator::Multiply);
    return PredicatePartition{allTrueStop, effectiveStop,
                              compare.getResult()};
  }
  return failure();
}

LogicalResult realizeFold(RegionFoldOp fold, func::FuncOp kernel) {
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

  struct SourceAssumption {
    AssumeInBoundsOp operation;
    PhysicalSourceAxis source;
    unsigned planIndex;
  };
  SmallVector<SourceAssumption> sourceAssumptions;
  kernel.walk([&](AssumeInBoundsOp assumption) {
    for (auto [planIndex, plan] : llvm::enumerate(plans)) {
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
      sourceAssumptions.push_back(
          {assumption, plan.sourceIdentity, static_cast<unsigned>(planIndex)});
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
  Value memberPredicate =
      summaryMembershipPredicate(fold, identities, fold.getSegment());
  if (!memberPredicate)
    return fold.emitOpError(
        "region-fold summarizer has no typed membership predicate that makes a physical tail equal to identity");

  bool bodyFailed = false;
  std::string failureReason;
  auto emitSummary = [&](OpBuilder &nested, Location nestedLocation,
                         Value offset,
                         bool predicateIsTrue)
      -> FailureOr<SmallVector<Value>> {
    SmallVector<Value> slices;
    Value segmentTail;
    IRMapping sliceMapping;
    SmallVector<std::shared_ptr<IRMapping>> sourceMappings;
    if (failed(buildSourceSlices(nested, nestedLocation, plans, sliceTypes,
                                 offset,
                                 segment.getResult(), sliceExtent, slices,
                                 segmentTail, sliceMapping, sourceMappings,
                                 failureReason)))
      return failure();
    for (SourceAssumption &sourceAssumption : sourceAssumptions) {
      if (sourceAssumption.planIndex >= sourceMappings.size()) {
        failureReason =
            "region-fold source assertion has no matching slice mapping";
        return failure();
      }
      ReplayMaterializationOptions replayOptions;
      replayOptions.scope = PhysicalReplayScope::Coordinate;
      replayOptions.segmentTail = segmentTail;
      replayOptions.materializeZeroFill = true;
      FailureOr<Value> index = materializeReplayedValue(
          nested, nestedLocation, sourceAssumption.operation.getIndex(),
          sourceAssumption.source, sliceExtent,
          *sourceMappings[sourceAssumption.planIndex], replayOptions);
      if (failed(index)) {
        failureReason =
            "region-fold source assertion could not be replayed for a physical slice";
        return failure();
      }
      auto replacement = nested.create<AssumeInBoundsOp>(
          nestedLocation, *index, sourceAssumption.operation.getResource(),
          sourceAssumption.operation.getAxis());
      if (Attribute origin =
              sourceAssumption.operation->getAttr(originAttr))
        replacement->setAttr(originAttr, origin);
    }
    SmallVector<Value> summarizeArguments(slices);
    summarizeArguments.append(captures.begin(), captures.end());
    Value substituteSource;
    Value substituteTarget;
    if (predicateIsTrue) {
      auto predicateType = dyn_cast<FragmentType>(partition->predicate.getType());
      if (!predicateType) {
        failureReason = "range predicate is not a physical fragment";
        return failure();
      }
      Value truth = nested.create<arith::ConstantOp>(
          nestedLocation, nested.getI1Type(), nested.getBoolAttr(true));
      substituteSource = partition->predicate;
      substituteTarget =
          nested.create<SplatOp>(nestedLocation, predicateType, truth);
    }
    return inlinePureRegion(nested, fold.getSummarize(), summarizeArguments,
                            failureReason, substituteSource, substituteTarget,
                            memberPredicate, segmentTail);
  };
  auto emitLoop = [&](Value lower, Value upper, ValueRange initial,
                      bool predicateIsTrue) {
    auto loop = builder.create<scf::ForOp>(
      location, lower, upper, segment.getResult(), initial,
      [&](OpBuilder &nested, Location nestedLocation, Value offset,
          ValueRange carries) {
        FailureOr<SmallVector<Value>> summary = emitSummary(
            nested, nestedLocation, offset, predicateIsTrue);
        if (failed(summary)) {
          bodyFailed = true;
          return;
        }
        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(summary->begin(), summary->end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, fold.getCombine(), combineArguments, failureReason);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      });
    if (Attribute origin = fold->getAttr(originAttr))
      loop->setAttr(originAttr, origin);
    return loop;
  };

  auto finish = [&]() -> LogicalResult {
    for (SourceAssumption &sourceAssumption : sourceAssumptions)
      if (sourceAssumption.operation->getBlock())
        sourceAssumption.operation.erase();
    eraseDeadPhysicalValues(kernel);
    return success();
  };

  if (segment->hasAttr(coverageDimensionAttr)) {
    FailureOr<SmallVector<Value>> summary =
        emitSummary(builder, location, zero, /*predicateIsTrue=*/false);
    if (failed(summary))
      return fold.emitOpError("region-fold physicalization failed: ")
             << failureReason;
    for (auto [oldResult, newResult] :
         llvm::zip(fold.getResults(), *summary))
      oldResult.replaceAllUsesWith(newResult);
    fold.erase();
    return finish();
  }

  SmallVector<Value> current(identities.begin(), identities.end());
  scf::ForOp allTrueLoop;
  if (succeeded(partition)) {
    allTrueLoop = emitLoop(zero, partition->allTrueStop, current,
                           /*predicateIsTrue=*/true);
    if (!bodyFailed)
      current.assign(allTrueLoop.getResults().begin(),
                     allTrueLoop.getResults().end());
  }
  Value mixedStart = succeeded(partition) ? partition->allTrueStop : zero;
  stop = succeeded(partition) ? partition->effectiveStop : stop;
  scf::ForOp mixedLoop;
  if (!bodyFailed)
    mixedLoop = emitLoop(mixedStart, stop, current,
                         /*predicateIsTrue=*/false);
  if (bodyFailed) {
    if (allTrueLoop && allTrueLoop->getBlock())
      allTrueLoop.erase();
    if (mixedLoop && mixedLoop->getBlock())
      mixedLoop.erase();
    return fold.emitOpError("region-fold physicalization failed: ")
           << failureReason;
  }
  for (auto [oldResult, newResult] :
       llvm::zip(fold.getResults(), mixedLoop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  fold.erase();
  return finish();
}

LogicalResult realizeScan(RegionScanOp scan, func::FuncOp kernel) {
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
                sliceExtent, slices, segmentTail, sliceMapping, sourceMappings,
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

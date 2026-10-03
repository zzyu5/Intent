#include "RegionCloning.h"
#include "RegionSources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace intent::gpu::region {
namespace {

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
                                    std::string &reason,
                                    ArrayRef<unsigned> selectedAxes = {}) {
  if (auto expectedFragment = dyn_cast<FragmentType>(expected)) {
    auto actualFragment = dyn_cast<FragmentType>(actual);
    if (!actualFragment)
      return success();
    for (auto [expectedAxis, attribute] :
         llvm::enumerate(expectedFragment.getAxisMaps())) {
      if (!selectedAxes.empty() &&
          !llvm::is_contained(selectedAxes, expectedAxis))
        continue;
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

void bindClonedRanges(Operation *root) {
  root->walk([&](MakeRangeOp range) {
    auto originalExtent =
        range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
    bool coveredIntroducedUnitDomain =
        originalExtent && originalExtent.value() == 1 &&
        samePhysicalScalarExpression(range.getStart(), range.getLogicalStart()) &&
        samePhysicalScalarExpression(range.getExtent(), range.getLogicalStop());
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    OpBuilder builder(range);
    Value physicalExtent;
    if (extent.getKind() == PhysicalExprKind::Constant)
      physicalExtent = builder.create<arith::ConstantIndexOp>(
          range.getLoc(), extent.getValue());
    else
      physicalExtent = builder.create<PhysicalExprOp>(
          range.getLoc(), builder.getIndexType(), extent);
    range.getExtentMutable().assign(physicalExtent);
    if (coveredIntroducedUnitDomain)
      range->setOperand(4, physicalExtent);
  });
}

} // namespace

FailureOr<SmallVector<Value>> inlinePureRegion(OpBuilder &builder, Region &region,
                                               ValueRange arguments,
                                               std::string &reason,
                                               Value substituteSource,
                                               Value substituteTarget,
                                               Value conjunctSource,
                                               Value conjunctPredicate,
                                               Value additionalSource,
                                               Value additionalTarget,
                                               IRMapping *resultMapping) {
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
    SmallVector<Type> selectedTypes;
    for (Value result : operation.getResults())
      selectedTypes.push_back(bindPhysicalExtents(result.getType(), extentBindings,
                                                 &operation));
    Operation *clone = builder.clone(operation, mapping);
    if (failed(rewriteClonedPhysicalTypes(&operation, clone, [&](Value original) {
          return bindPhysicalExtents(original.getType(), extentBindings,
                                     original.getDefiningOp());
        }))) {
      reason = "helper operands cannot transport the declared fragment schema";
      return failure();
    }
    bindClonedRanges(clone);
    for (unsigned index = 0; index < operation.getNumResults(); ++index) {
      auto original = dyn_cast<FragmentType>(operation.getResult(index).getType());
      auto selected = dyn_cast<FragmentType>(selectedTypes[index]);
      auto type = dyn_cast<FragmentType>(clone->getResult(index).getType());
      if (!original || !selected || !type || type == selected)
        continue;
      SmallVector<unsigned> changedAxes;
      bool changedRank = selected.getShape().size() != type.getShape().size();
      if (!changedRank)
        for (auto [axis, extent] : llvm::enumerate(selected.getShape()))
          if (extent != type.getShape()[axis] ||
              selected.getAxisMaps()[axis] != type.getAxisMaps()[axis])
            changedAxes.push_back(axis);
      if (changedRank || !changedAxes.empty()) {
        if (failed(collectExtentBindings(original, type, extentBindings, reason,
                                        changedAxes)))
          return failure();
      }
    }
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

} // namespace intent::gpu::region

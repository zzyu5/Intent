#include "RegionCloning.h"
#include "RegionSources.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Value/SchemaMutation.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::gpu::region {
namespace {

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

} // namespace

LogicalResult collectScanOutputConsumers(RegionScanOp scan,
                                         SmallVectorImpl<Operation *> &ordered,
                                         std::string &reason) {
  llvm::SetVector<Operation *> closure;
  SmallVector<Operation *> worklist;
  for (Value output : scan.getEmittedResults())
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

namespace {

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
  if (failed(rewriteClonedPhysicalTypes(
          definition, clone, [&](Value original) -> Type {
            auto selected = slicedType(original.getType(), relations,
                                       sliceExtent, original.getDefiningOp());
            return succeeded(selected) ? *selected : Type{};
          }))) {
    reason = "region-scan output address cannot transport its sliced schema";
    return failure();
  }
  for (auto [original, result] :
       llvm::zip(definition->getResults(), clone->getResults())) {
    if (!mapping.lookupOrNull(original))
      mapping.map(original, result);
  }
  Value result = mapping.lookupOrNull(value);
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

} // namespace

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
    if (failed(rewriteClonedPhysicalTypes(
            operation, clone, [&](Value original) -> Type {
              auto selected = slicedType(original.getType(), relations,
                                         sliceExtent, original.getDefiningOp());
              return succeeded(selected) ? *selected : Type{};
            }))) {
      reason = "region-scan output consumer cannot transport its sliced schema";
      return failure();
    }
    for (auto [original, result] :
         llvm::zip(operation->getResults(), clone->getResults())) {
      if (!mapping.lookupOrNull(original))
        mapping.map(original, result);
    }
  }
  return success();
}

} // namespace intent::gpu::region

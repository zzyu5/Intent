#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

void collectParameterSymbols(Attribute attribute, llvm::StringSet<> &symbols);

using AxisSelector = llvm::function_ref<bool(AxisMapAttr)>;

FragmentType replaceExtent(FragmentType source, AxisSelector selects,
                           ArrayRef<Attribute> previousExtents,
                           PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  bool changed = false;
  for (auto [axis, attribute] : llvm::enumerate(source.getAxisMaps())) {
    if (!selects(cast<AxisMapAttr>(attribute)) ||
        !llvm::is_contained(previousExtents, source.getShape()[axis]))
      continue;
    shape[axis] = extent;
    changed = true;
  }
  if (!changed)
    return source;
  return FragmentType::get(
      source.getContext(), source.getElementType(),
      ArrayAttr::get(source.getContext(), shape), source.getAxisMaps(),
      source.getValidity(), source.getOwner());
}

Type replaceExtent(Type source, AxisSelector selects,
                   ArrayRef<Attribute> previousExtents,
                   PhysicalExprAttr extent) {
  if (auto fragment = dyn_cast<FragmentType>(source))
    return replaceExtent(fragment, selects, previousExtents, extent);
  auto record = dyn_cast<RecordType>(source);
  if (!record)
    return source;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute attribute : record.getFieldTypes()) {
    Type field = cast<TypeAttr>(attribute).getValue();
    Type replacement = replaceExtent(field, selects, previousExtents, extent);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != field;
  }
  if (!changed)
    return source;
  return RecordType::get(source.getContext(), record.getFieldNames(),
                         ArrayAttr::get(source.getContext(), fields),
                         record.getOwner());
}

bool carriesExtent(Type type, AxisSelector selects,
                   ArrayRef<Attribute> extents) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (selects(cast<AxisMapAttr>(attribute)) &&
          llvm::is_contained(extents, fragment.getShape()[axis]))
        return true;
    return false;
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return false;
  return llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
    return carriesExtent(cast<TypeAttr>(attribute).getValue(), selects, extents);
  });
}

bool preservesIntroducedUnitAxis(Value value, AxisSelector selects) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(value.getType());
  if (!reshape || !result)
    return false;
  std::optional<unsigned> axis;
  for (Attribute attribute : result.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (!selects(mapping))
      continue;
    if (axis)
      return false;
    axis = mapping.getFragmentAxis();
  }
  if (!axis)
    return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  return !input || !llvm::any_of(input.getAxisMaps(), [&](Attribute attribute) {
           return selects(cast<AxisMapAttr>(attribute));
         });
}

bool selectsSegmentAxis(Type type, uint64_t axis, AxisSelector selects) {
  auto fragment = dyn_cast<FragmentType>(type);
  return fragment && axis < fragment.getAxisMaps().size() &&
         selects(cast<AxisMapAttr>(fragment.getAxisMaps()[axis]));
}

void appendStructuredParentRelations(BlockArgument argument,
                                     SmallVectorImpl<Value> &worklist,
                                     AxisSelector selects) {
  Block *block = argument.getOwner();
  Region *region = block->getParent();
  Operation *parent = region ? region->getParentOp() : nullptr;
  unsigned index = argument.getArgNumber();
  if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent)) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    if (region == &fold.getSummarize()) {
      if (index < sources) {
        if (!selectsSegmentAxis(argument.getType(), fold.getAxis(), selects))
          worklist.push_back(fold.getInputs()[index]);
        return;
      }
      unsigned capture = index - sources;
      worklist.push_back(fold.getInputs()[sources + identities + capture]);
      return;
    }
    if (region == &fold.getCombine()) {
      unsigned component = index % identities;
      worklist.push_back(fold.getInputs()[sources + component]);
      worklist.push_back(fold.getResults()[component]);
    }
    return;
  }
  auto scan = dyn_cast_or_null<RegionScanOp>(parent);
  if (!scan)
    return;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned states = scan.getStateCount();
  unsigned outputs = scan.getOutputCount();
  unsigned stateOffset = sources + identities;
  unsigned captureOffset = stateOffset + states;
  if (region == &scan.getSummarize()) {
    if (index < sources) {
      if (!selectsSegmentAxis(argument.getType(), scan.getAxis(), selects))
        worklist.push_back(scan.getInputs()[index]);
      return;
    }
    worklist.push_back(scan.getInputs()[captureOffset + index - sources]);
    return;
  }
  if (region == &scan.getCombine()) {
    worklist.push_back(scan.getInputs()[sources + index % identities]);
    return;
  }
  if (region == &scan.getApply()) {
    if (index < identities) {
      worklist.push_back(scan.getInputs()[sources + index]);
      return;
    }
    unsigned state = index - identities;
    worklist.push_back(scan.getInputs()[stateOffset + state]);
    worklist.push_back(scan.getResults()[outputs + state]);
    return;
  }
  if (region != &scan.getEmit())
    return;
  if (index < sources) {
    if (!selectsSegmentAxis(argument.getType(), scan.getAxis(), selects))
      worklist.push_back(scan.getInputs()[index]);
    return;
  }
  if (index < sources + states) {
    unsigned state = index - sources;
    worklist.push_back(scan.getInputs()[stateOffset + state]);
    worklist.push_back(scan.getResults()[outputs + state]);
    return;
  }
  worklist.push_back(
      scan.getInputs()[captureOffset + index - sources - states]);
}

void appendStructuredChildRelations(Operation *user, Value value,
                                    SmallVectorImpl<Value> &worklist,
                                    AxisSelector selects) {
  if (auto fold = dyn_cast<RegionFoldOp>(user)) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    for (auto [index, operand] : llvm::enumerate(fold.getInputs())) {
      if (operand != value)
        continue;
      if (index < sources) {
        BlockArgument slice =
            fold.getSummarize().front().getArgument(index);
        if (!selectsSegmentAxis(slice.getType(), fold.getAxis(), selects))
          worklist.push_back(slice);
        continue;
      }
      if (index < sources + identities) {
        unsigned component = index - sources;
        worklist.push_back(fold.getResults()[component]);
        Block &combine = fold.getCombine().front();
        worklist.push_back(combine.getArgument(component));
        worklist.push_back(combine.getArgument(identities + component));
        continue;
      }
      Block &summarize = fold.getSummarize().front();
      unsigned capture = index - sources - identities;
      worklist.push_back(summarize.getArgument(sources + capture));
    }
    return;
  }
  auto scan = dyn_cast<RegionScanOp>(user);
  if (!scan)
    return;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned states = scan.getStateCount();
  unsigned outputs = scan.getOutputCount();
  unsigned stateOffset = sources + identities;
  unsigned captureOffset = stateOffset + states;
  for (auto [index, operand] : llvm::enumerate(scan.getInputs())) {
    if (operand != value)
      continue;
    if (index < sources) {
      BlockArgument summarize =
          scan.getSummarize().front().getArgument(index);
      BlockArgument emit = scan.getEmit().front().getArgument(index);
      if (!selectsSegmentAxis(summarize.getType(), scan.getAxis(), selects)) {
        worklist.push_back(summarize);
        worklist.push_back(emit);
      }
      continue;
    }
    if (index < stateOffset) {
      unsigned component = index - sources;
      Block &combine = scan.getCombine().front();
      Block &apply = scan.getApply().front();
      worklist.push_back(combine.getArgument(component));
      worklist.push_back(combine.getArgument(identities + component));
      worklist.push_back(apply.getArgument(component));
      continue;
    }
    if (index < captureOffset) {
      unsigned state = index - stateOffset;
      worklist.push_back(scan.getResults()[outputs + state]);
      worklist.push_back(
          scan.getApply().front().getArgument(identities + state));
      worklist.push_back(scan.getEmit().front().getArgument(sources + state));
      continue;
    }
    unsigned capture = index - captureOffset;
    worklist.push_back(
        scan.getSummarize().front().getArgument(sources + capture));
    worklist.push_back(
        scan.getEmit().front().getArgument(sources + states + capture));
  }
}

void appendStructuredResultRelations(Value value,
                                     SmallVectorImpl<Value> &worklist) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return;
  unsigned index = result.getResultNumber();
  if (auto fold = dyn_cast<RegionFoldOp>(result.getOwner())) {
    unsigned sources = fold.getSourceCount();
    unsigned identities = fold.getIdentityCount();
    worklist.push_back(fold.getInputs()[sources + index]);
    worklist.push_back(fold.getCombine().front().getArgument(index));
    worklist.push_back(
        fold.getCombine().front().getArgument(identities + index));
    worklist.push_back(
        cast<YieldOp>(fold.getSummarize().front().getTerminator())
            .getValues()[index]);
    worklist.push_back(cast<YieldOp>(fold.getCombine().front().getTerminator())
                           .getValues()[index]);
    return;
  }
  auto scan = dyn_cast<RegionScanOp>(result.getOwner());
  if (!scan)
    return;
  unsigned outputs = scan.getOutputCount();
  if (index < outputs) {
    worklist.push_back(
        cast<YieldOp>(scan.getEmit().front().getTerminator()).getValues()[index]);
    return;
  }
  unsigned state = index - outputs;
  unsigned sources = scan.getSourceCount();
  unsigned identities = scan.getIdentityCount();
  unsigned stateOffset = sources + identities;
  worklist.push_back(scan.getInputs()[stateOffset + state]);
  worklist.push_back(
      scan.getApply().front().getArgument(identities + state));
  worklist.push_back(scan.getEmit().front().getArgument(sources + state));
  worklist.push_back(
      cast<YieldOp>(scan.getApply().front().getTerminator()).getValues()[state]);
}

FailureOr<Value> projectPredicate(OpBuilder &builder, Location location,
                                  Value predicate, FragmentType target,
                                  unsigned axis) {
  auto base = dyn_cast<FragmentType>(predicate.getType());
  if (!base || base.getShape().size() != 1 || axis >= target.getShape().size())
    return failure();
  SmallVector<Attribute> shape(
      target.getShape().size(),
      PhysicalExprAttr::get(
          target.getContext(),
          static_cast<uint32_t>(PhysicalExprKind::Constant), 1,
          StringAttr::get(target.getContext()),
          ArrayAttr::get(target.getContext(), {})));
  shape[axis] = base.getShape()[0];
  auto reshaped = FragmentType::get(
      target.getContext(), builder.getI1Type(),
      ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
      target.getValidity(), target.getOwner());
  Value result = predicate;
  if (base != reshaped) {
    FailureOr<ArrayAttr> reassociation =
        inferReshapeReassociation(base, reshaped);
    if (failed(reassociation))
      return failure();
    result = builder.create<ReshapeOp>(location, reshaped, result,
                                      *reassociation);
  }
  auto projected = FragmentType::get(
      target.getContext(), builder.getI1Type(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  if (reshaped != projected)
    result = builder.create<BroadcastOp>(location, projected, result);
  return result;
}

FailureOr<Value> zeroFill(OpBuilder &builder, Location location,
                          FragmentType type) {
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(type.getElementType()))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(type.getElementType()))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(type.getElementType()))
    zero = builder.getIndexAttr(0);
  if (!zero)
    return failure();
  FailureOr<Value> scalar =
      materializeScalarConstant(builder, location, zero, type.getElementType());
  if (failed(scalar))
    return failure();
  return Value(builder.create<BroadcastOp>(location, type, *scalar));
}

void collectParameterSymbols(Type type, llvm::StringSet<> &symbols) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    collectParameterSymbols(fragment.getShape(), symbols);
    return;
  }
  if (auto buffer = dyn_cast<BufferType>(type)) {
    collectParameterSymbols(buffer.getShape(), symbols);
    return;
  }
  if (auto view = dyn_cast<ViewType>(type)) {
    collectParameterSymbols(view.getLayout().getExtents(), symbols);
    return;
  }
  if (auto record = dyn_cast<RecordType>(type))
    collectParameterSymbols(record.getFieldTypes(), symbols);
}

void collectParameterSymbols(Attribute attribute, llvm::StringSet<> &symbols) {
  if (!attribute)
    return;
  if (auto expression = dyn_cast<PhysicalExprAttr>(attribute)) {
    if (expression.getKind() ==
        static_cast<uint32_t>(PhysicalExprKind::Parameter))
      symbols.insert(expression.getSymbol().getValue());
    for (Attribute operand : expression.getOperands())
      collectParameterSymbols(operand, symbols);
    return;
  }
  if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    for (Attribute element : array)
      collectParameterSymbols(element, symbols);
    return;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
    for (NamedAttribute element : dictionary)
      collectParameterSymbols(element.getValue(), symbols);
    return;
  }
  if (auto typed = dyn_cast<TypeAttr>(attribute))
    collectParameterSymbols(typed.getValue(), symbols);
}

} // namespace

FailureOr<Value> materializeScalarConstant(OpBuilder &builder,
                                           Location location, Attribute value,
                                           Type resultType) {
  if (isa<IndexType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    return Value(
        builder.create<arith::ConstantIndexOp>(location, integer.getInt()));
  }
  if (auto integerType = dyn_cast<IntegerType>(resultType)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer)
      return failure();
    auto signless = IntegerType::get(builder.getContext(), integerType.getWidth());
    llvm::APInt bits = integer.getValue().sextOrTrunc(integerType.getWidth());
    Value raw = builder.create<arith::ConstantOp>(
        location, signless, IntegerAttr::get(signless, bits));
    if (integerType.isSignless())
      return raw;
    return Value(builder.create<CastOp>(location, integerType, raw));
  }
  if (auto floatType = dyn_cast<FloatType>(resultType)) {
    auto floating = dyn_cast<FloatAttr>(value);
    if (!floating)
      return failure();
    return Value(builder.create<arith::ConstantOp>(
        location, floatType,
        FloatAttr::get(floatType, floating.getValueAsDouble())));
  }
  return failure();
}

FailureOr<Value> materializeBroadcastToFragment(OpBuilder &builder,
                                                Location location, Value value,
                                                FragmentType target) {
  if (value.getType() == target)
    return value;
  Type element = value.getType();
  if (auto fragment = dyn_cast<FragmentType>(element))
    element = fragment.getElementType();
  if (!isa<IntegerType, FloatType, IndexType>(element) ||
      element != target.getElementType())
    return failure();
  return Value(builder.create<BroadcastOp>(location, target, value));
}

FailureOr<Value> projectPhysicalValueToSchema(OpBuilder &builder,
                                              Location location, Value value,
                                              Type target) {
  if (value.getType() == target)
    return value;
  if (auto fragment = dyn_cast<FragmentType>(target)) {
    Type sourceElement = value.getType();
    if (auto source = dyn_cast<FragmentType>(sourceElement))
      sourceElement = source.getElementType();
    if (sourceElement != fragment.getElementType())
      return failure();
    Operation *projection = nullptr;
    if (isa<IntegerType, FloatType, IndexType>(value.getType()))
      projection = builder.create<SplatOp>(location, fragment, value);
    else if (isa<FragmentType>(value.getType()))
      projection = builder.create<BroadcastOp>(location, fragment, value);
    if (!projection)
      return failure();
    if (Operation *definition = value.getDefiningOp())
      if (Attribute origin = definition->getAttr(originAttr))
        projection->setAttr(originAttr, origin);
    return projection->getResult(0);
  }
  auto targetRecord = dyn_cast<RecordType>(target);
  auto sourceRecord = dyn_cast<RecordType>(value.getType());
  if (!targetRecord || !sourceRecord ||
      sourceRecord.getFieldNames() != targetRecord.getFieldNames() ||
      sourceRecord.getFieldTypes().size() !=
          targetRecord.getFieldTypes().size())
    return failure();
  auto record = value.getDefiningOp<MakeRecordOp>();
  SmallVector<Value> projectedFields;
  for (auto [index, targetField] :
       llvm::enumerate(targetRecord.getFieldTypes())) {
    Type sourceType =
        cast<TypeAttr>(sourceRecord.getFieldTypes()[index]).getValue();
    Value field = record
                      ? record.getFields()[index]
                      : Value(builder.create<ExtractOp>(location, sourceType,
                                                        value, index));
    FailureOr<Value> projected = projectPhysicalValueToSchema(
        builder, location, field, cast<TypeAttr>(targetField).getValue());
    if (failed(projected))
      return failure();
    projectedFields.push_back(*projected);
  }
  auto projected =
      builder.create<MakeRecordOp>(location, targetRecord, projectedFields);
  if (Operation *definition = value.getDefiningOp())
    if (Attribute origin = definition->getAttr(originAttr))
      projected->setAttr(originAttr, origin);
  return projected.getResult();
}

LogicalResult alignReductionIdentityRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](Operation *operation) {
    auto align = [&](ValueRange inputs, ValueRange results, Region &combine,
                     uint64_t sourceCount,
                     uint64_t identityCount) -> LogicalResult {
      if (identityCount != results.size())
        return failure();
      OpBuilder builder(operation);
      for (unsigned index = 0; index < identityCount; ++index) {
        unsigned operand = sourceCount + index;
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, operation->getLoc(), inputs[operand],
            results[index].getType());
        if (failed(projected))
          return operation->emitOpError(
              "physical reduction identity cannot adopt its result relation");
        operation->setOperand(operand, *projected);
      }
      if (!llvm::hasSingleElement(combine) ||
          combine.front().getNumArguments() < 2 * identityCount)
        return operation->emitOpError(
            "physical reduction helper has no complete accumulator schema");
      for (unsigned index = 0; index < identityCount; ++index) {
        Type target = results[index].getType();
        combine.front().getArgument(index).setType(target);
        combine.front().getArgument(identityCount + index).setType(target);
      }
      return success();
    };
    if (auto reduce = dyn_cast<ReduceOp>(operation))
      return failed(align(reduce.getInputs(), reduce.getResults(),
                          reduce.getCombine(),
                          reduce.getSourceCount(), reduce.getIdentityCount()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    if (auto scan = dyn_cast<ScanOp>(operation))
      return failed(align(scan.getInputs(), scan.getResults(), scan.getCombine(),
                          scan.getSourceCount(), scan.getIdentityCount()))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignReductionYieldRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](Operation *operation) {
    Region *combine = nullptr;
    ValueRange results;
    if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      combine = &reduce.getCombine();
      results = reduce.getResults();
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      combine = &scan.getCombine();
      results = scan.getResults();
    } else {
      return WalkResult::advance();
    }
    if (!combine || !llvm::hasSingleElement(*combine))
      return WalkResult::interrupt();
    auto yield = dyn_cast<YieldOp>(combine->front().getTerminator());
    if (!yield || yield.getValues().size() != results.size())
      return WalkResult::interrupt();
    OpBuilder builder(yield);
    for (auto [index, target] : llvm::enumerate(results)) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, operation->getLoc(), yield.getValues()[index],
          target.getType());
      if (failed(projected))
        return WalkResult::interrupt();
      yield->setOperand(index, *projected);
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

FailureOr<Value> materializeZeroFragment(OpBuilder &builder,
                                         Location location,
                                         FragmentType target) {
  return zeroFill(builder, location, target);
}

FailureOr<Value> materializeReplayedValue(
    OpBuilder &builder, Location location, Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    IRMapping &mapping, ReplayMaterializationOptions options) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalReplayFact replay = analysis.replayability(
      value, source, options.scope, options.allowAccesses);
  if (!replay.isReplayable())
    return failure();

  auto replaceReplayAxis = [&](FragmentType type, unsigned axis) {
    SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
    SmallVector<Attribute> axes(type.getAxisMaps().begin(),
                                type.getAxisMaps().end());
    shape[axis] = blockedExtent;
    if (options.segmentMapping)
      axes[axis] = AxisMapAttr::get(
          type.getContext(), options.segmentMapping.getSourceId(),
          options.segmentMapping.getSourceAxis(),
          options.segmentMapping.getDimensionId(), axis,
          options.segmentMapping.getDerived());
    return FragmentType::get(
        type.getContext(), type.getElementType(),
        ArrayAttr::get(type.getContext(), shape),
        ArrayAttr::get(type.getContext(), axes), type.getValidity(),
        type.getOwner());
  };
  std::function<Type(Type)> replaceReplayType = [&](Type type) -> Type {
    if (auto fragment = dyn_cast<FragmentType>(type)) {
      PhysicalAxisProjection projection = queryFragmentAxis(fragment, source);
      return projection.isExact()
                 ? Type(replaceReplayAxis(fragment, projection.fragmentAxis))
                 : type;
    }
    auto record = dyn_cast<RecordType>(type);
    if (!record)
      return type;
    SmallVector<Attribute> fields;
    bool changed = false;
    for (Attribute field : record.getFieldTypes()) {
      Type current = cast<TypeAttr>(field).getValue();
      Type replacement = replaceReplayType(current);
      fields.push_back(TypeAttr::get(replacement));
      changed |= replacement != current;
    }
    return changed ? Type(RecordType::get(
                         type.getContext(), record.getFieldNames(),
                         ArrayAttr::get(type.getContext(), fields),
                         record.getOwner()))
                   : type;
  };
  auto carriesReplaySource = [&](Type type) {
    return replaceReplayType(type) != type;
  };
  auto selectedReplayAxis = [&](Value current)
      -> std::optional<unsigned> {
    auto fragment = dyn_cast<FragmentType>(current.getType());
    PhysicalAxisProjection projection =
        fragment ? queryFragmentAxis(fragment, source)
                 : PhysicalAxisProjection{};
    if (!fragment || !projection.isExact())
      return std::nullopt;
    if (options.traversalRanges.empty())
      return projection.fragmentAxis;
    PhysicalRangeFact fact =
        analysis.axisRanges(current, projection.fragmentAxis);
    if (fact.roots.empty()) {
      Operation *producer = current.getDefiningOp();
      bool neutralSchemaCarrier = isa_and_nonnull<SplatOp>(producer);
      if (auto broadcast = dyn_cast_or_null<BroadcastOp>(producer))
        neutralSchemaCarrier |=
            !isa<FragmentType, RecordType>(broadcast.getValue().getType());
      return neutralSchemaCarrier
                 ? std::optional<unsigned>(projection.fragmentAxis)
                 : std::nullopt;
    }
    SmallVector<MakeRangeOp> combined(fact.roots.begin(), fact.roots.end());
    combined.append(options.traversalRanges.begin(),
                    options.traversalRanges.end());
    return analysis.lockstepRanges(combined).isExact()
               ? std::optional<unsigned>(projection.fragmentAxis)
               : std::nullopt;
  };
  auto isUnitExtent = [](Attribute attribute) {
    auto extent = dyn_cast<PhysicalExprAttr>(attribute);
    return extent &&
           extent.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           extent.getValue() == 1;
  };
  auto retargetHelperSourceExtent = [&](Region &region,
                                        PhysicalExprAttr logicalExtent) {
    auto retarget = [&](Value current) {
      auto fragment = dyn_cast<FragmentType>(current.getType());
      PhysicalAxisProjection projection =
          fragment ? queryFragmentAxis(fragment, source)
                   : PhysicalAxisProjection{};
      if (!fragment || !projection.isExact() ||
          fragment.getShape()[projection.fragmentAxis] != logicalExtent)
        return;
      current.setType(replaceReplayAxis(fragment, projection.fragmentAxis));
    };
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments())
        retarget(argument);
      block.walk([&](Operation *operation) {
        for (Value result : operation->getResults())
          retarget(result);
      });
    }
  };

  std::function<FailureOr<Value>(Value)> materialize =
      [&](Value current) -> FailureOr<Value> {
    if (Value mapped = mapping.lookupOrNull(current))
      return mapped;
    if (auto extract = current.getDefiningOp<ExtractOp>()) {
      if (auto record = extract.getRecord().getDefiningOp<MakeRecordOp>()) {
        uint64_t field = extract.getField();
        if (field >= record.getFields().size())
          return failure();
        FailureOr<Value> replayed = materialize(record.getFields()[field]);
        if (succeeded(replayed) && !mapping.lookupOrNull(current))
          mapping.map(current, *replayed);
        return replayed;
      }
      if (carriesReplaySource(extract.getRecord().getType())) {
        FailureOr<Value> replayedRecord = materialize(extract.getRecord());
        if (failed(replayedRecord))
          return failure();
        Type target = replaceReplayType(current.getType());
        Value replayed = builder.create<ExtractOp>(
            location, target, *replayedRecord, extract.getField());
        mapping.map(current, replayed);
        return replayed;
      }
    }

    auto fragment = dyn_cast<FragmentType>(current.getType());
    if (!fragment && carriesReplaySource(current.getType())) {
      Operation *producer = current.getDefiningOp();
      if (!producer ||
          !isPhysicalReplayNode(producer, options.scope,
                                /*allowAccesses=*/false))
        return failure();
      for (Value operand : producer->getOperands()) {
        if (!carriesReplaySource(operand.getType()))
          continue;
        FailureOr<Value> replayed = materialize(operand);
        if (failed(replayed))
          return failure();
        if (!mapping.lookupOrNull(operand) && *replayed != operand)
          mapping.map(operand, *replayed);
      }
      Operation *clone = builder.clone(*producer, mapping);
      for (auto [original, result] :
           llvm::zip(producer->getResults(), clone->getResults())) {
        result.setType(replaceReplayType(result.getType()));
        if (!mapping.lookupOrNull(original))
          mapping.map(original, result);
      }
      Region *combine = nullptr;
      if (auto reduce = dyn_cast<ReduceOp>(clone))
        combine = &reduce.getCombine();
      else if (auto scan = dyn_cast<ScanOp>(clone))
        combine = &scan.getCombine();
      if (combine)
        for (Block &block : *combine) {
          for (BlockArgument argument : block.getArguments())
            argument.setType(replaceReplayType(argument.getType()));
          block.walk([&](Operation *operation) {
            for (Value result : operation->getResults())
              result.setType(replaceReplayType(result.getType()));
          });
        }
      auto result = dyn_cast<OpResult>(current);
      if (!result || result.getResultNumber() >= clone->getNumResults())
        return failure();
      Value replayed = clone->getResult(result.getResultNumber());
      return replayed;
    }
    std::optional<unsigned> projection = selectedReplayAxis(current);
    if (!fragment || !projection)
      return current;
    unsigned axis = *projection;
    Operation *producer = current.getDefiningOp();
    if (!producer)
      return failure();

    auto replayOperand = [&](Value operand) -> FailureOr<Value> {
      return selectedReplayAxis(operand) ? materialize(operand)
                                         : FailureOr<Value>(operand);
    };
    auto combineTail = [&](FragmentType target,
                           Value valid) -> FailureOr<Value> {
      if (!options.segmentTail)
        return valid ? FailureOr<Value>(valid)
                     : FailureOr<Value>(Value());
      auto targetAxis = cast<AxisMapAttr>(target.getAxisMaps()[axis]);
      FailureOr<Value> tail = projectPredicateToFragment(
          builder, location, options.segmentTail, target,
          sourceAxisIdentity(targetAxis));
      if (failed(tail))
        return failure();
      if (!valid)
        return *tail;
      return materializeValidityConjunction(builder, location, valid, *tail,
                                            target);
    };
    auto replayFill = [&](Value fill,
                          FragmentType target) -> FailureOr<Value> {
      if (fill) {
        FailureOr<Value> replayed = replayOperand(fill);
        if (failed(replayed))
          return failure();
        fill = *replayed;
        if (fill.getType() != target)
          fill = builder.create<BroadcastOp>(location, target, fill);
        return fill;
      }
      if (!options.materializeZeroFill)
        return Value();
      return materializeZeroFragment(builder, location, target);
    };

    if (auto load = dyn_cast<LoadOp>(producer)) {
      SmallVector<Value> coordinates;
      for (Value coordinate : load.getCoordinates()) {
        FailureOr<Value> replayed = replayOperand(coordinate);
        if (failed(replayed))
          return failure();
        coordinates.push_back(*replayed);
      }
      Value valid;
      if (load.getValid()) {
        FailureOr<Value> replayed = replayOperand(load.getValid());
        if (failed(replayed))
          return failure();
        valid = *replayed;
      }
      FragmentType resultType = replaceReplayAxis(fragment, axis);
      FailureOr<Value> combined = combineTail(resultType, valid);
      if (failed(combined))
        return failure();
      FailureOr<Value> fill = replayFill(load.getFill(), resultType);
      if (failed(fill))
        return failure();
      auto clone = builder.create<LoadOp>(
          location, resultType, load.getResource(), coordinates, *combined,
          *fill, load.getSourceAxes());
      if (Attribute origin = load->getAttr(originAttr))
        clone->setAttr(originAttr, origin);
      mapping.map(current, clone.getResult());
      return clone.getResult();
    }
    if (auto gather = dyn_cast<GatherOp>(producer)) {
      FailureOr<Value> replayedSource = replayOperand(gather.getSource());
      if (failed(replayedSource))
        return failure();
      SmallVector<Value> coordinates;
      for (Value coordinate : gather.getCoordinates()) {
        FailureOr<Value> replayed = replayOperand(coordinate);
        if (failed(replayed))
          return failure();
        coordinates.push_back(*replayed);
      }
      Value valid;
      if (gather.getValid()) {
        FailureOr<Value> replayed = replayOperand(gather.getValid());
        if (failed(replayed))
          return failure();
        valid = *replayed;
      }
      FragmentType resultType = replaceReplayAxis(fragment, axis);
      FailureOr<Value> combined = combineTail(resultType, valid);
      if (failed(combined))
        return failure();
      FailureOr<Value> fill = replayFill(gather.getFill(), resultType);
      if (failed(fill))
        return failure();
      auto clone = builder.create<GatherOp>(
          location, resultType, *replayedSource, coordinates, *combined, *fill,
          gather.getSourceAxes());
      if (Attribute origin = gather->getAttr(originAttr))
        clone->setAttr(originAttr, origin);
      mapping.map(current, clone.getResult());
      return clone.getResult();
    }
    if (!isPhysicalReplayNode(producer, options.scope,
                              /*allowAccesses=*/false))
      return failure();
    for (Value operand : producer->getOperands()) {
      FailureOr<Value> replayed = replayOperand(operand);
      if (failed(replayed))
        return failure();
      if (!mapping.lookupOrNull(operand) && *replayed != operand)
        mapping.map(operand, *replayed);
    }
    Operation *clone = builder.clone(*producer, mapping);
    bool structuredResults = isa<ReduceOp, ScanOp>(clone);
    for (auto [original, cloned] :
         llvm::zip(producer->getResults(), clone->getResults())) {
      if (structuredResults)
        cloned.setType(replaceReplayType(cloned.getType()));
      if (!mapping.lookupOrNull(original))
        mapping.map(original, cloned);
    }
    auto result = dyn_cast<OpResult>(current);
    if (!result || result.getResultNumber() >= clone->getNumResults())
      return failure();
    Value clonedValue = clone->getResult(result.getResultNumber());
    if (auto clonedReduce = dyn_cast<ReduceOp>(clone))
      retargetHelperSourceExtent(
          clonedReduce.getCombine(),
          cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    if (auto clonedScan = dyn_cast<ScanOp>(clone))
      retargetHelperSourceExtent(
          clonedScan.getCombine(),
          cast<PhysicalExprAttr>(fragment.getShape()[axis]));
    auto clonedType = dyn_cast<FragmentType>(clonedValue.getType());
    if (!clonedType || axis >= clonedType.getShape().size())
      return failure();
    bool introducedUnitAxis = false;
    if (auto reshape = dyn_cast<ReshapeOp>(producer)) {
      auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
      introducedUnitAxis =
          isUnitExtent(clonedType.getShape()[axis]) &&
          (!input || !queryFragmentAxis(input, source).isExact());
    }
    if (!introducedUnitAxis)
      clonedValue.setType(replaceReplayAxis(clonedType, axis));
    return clonedValue;
  };

  return materialize(value);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            PhysicalSourceAxis source) {
  PhysicalAxisProjection projection = queryFragmentAxis(target, source);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> projectPredicateToFragment(OpBuilder &builder,
                                            Location location, Value predicate,
                                            FragmentType target,
                                            int64_t dimensionId) {
  PhysicalDimensionProjection projection =
      queryFragmentDimension(target, dimensionId);
  if (!projection.isExact())
    return failure();
  return projectPredicate(builder, location, predicate, target,
                          projection.fragmentAxis);
}

FailureOr<Value> materializeValidityConjunction(
    OpBuilder &builder, Location location, Value lhs, Value rhs,
    FragmentType valueType) {
  auto predicateType = FragmentType::get(
      valueType.getContext(), builder.getI1Type(), valueType.getShape(),
      valueType.getAxisMaps(), valueType.getValidity(), valueType.getOwner());
  Value result;
  if (lhs) {
    FailureOr<Value> projected =
        materializeBroadcastToFragment(builder, location, lhs, predicateType);
    if (failed(projected))
      return failure();
    result = *projected;
  }
  if (rhs) {
    FailureOr<Value> projected =
        materializeBroadcastToFragment(builder, location, rhs, predicateType);
    if (failed(projected))
      return failure();
    result = result ? Value(builder.create<BinaryOp>(
                          location, predicateType, result, *projected,
                          BinaryOperator::LogicalAnd))
                    : *projected;
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

FailureOr<Value> materializeRetargetedValidity(
    OpBuilder &builder, Location location, Value original,
    ArrayRef<std::pair<MakeRangeOp, Value>> originalTailRanges,
    Value physicalTail, FragmentType target) {
  func::FuncOp kernel;
  if (!originalTailRanges.empty())
    kernel = originalTailRanges.front().first->getParentOfType<func::FuncOp>();
  else if (original)
    if (Operation *definition = original.getDefiningOp())
      kernel = definition->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  PhysicalProgramAnalysis analysis(kernel);

  std::function<FailureOr<Value>(Value)> residual =
      [&](Value value) -> FailureOr<Value> {
    if (!value || analysis.isTailPredicate(value, originalTailRanges))
      return Value();
    if (value.getType().isInteger(1))
      return value;
    if (auto broadcast = value.getDefiningOp<BroadcastOp>())
      return residual(broadcast.getValue());
    if (auto splat = value.getDefiningOp<SplatOp>())
      return residual(splat.getValue());
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return residual(reshape.getValue());
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return residual(transpose.getValue());
    auto conjunction = value.getDefiningOp<BinaryOp>();
    if (!conjunction)
      return failure();
    Type element = conjunction.getResult().getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    bool logical =
        conjunction.getOperatorKind() == BinaryOperator::LogicalAnd;
    bool bitwiseI1 =
        conjunction.getOperatorKind() == BinaryOperator::BitwiseAnd &&
        element.isInteger(1);
    if (!logical && !bitwiseI1)
      return failure();
    FailureOr<Value> lhs = residual(conjunction.getLhs());
    FailureOr<Value> rhs = residual(conjunction.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!*lhs)
      return *rhs;
    if (!*rhs)
      return *lhs;
    if (!(*lhs).getType().isInteger(1) || !(*rhs).getType().isInteger(1))
      return failure();
    return Value(builder.create<BinaryOp>(location, builder.getI1Type(), *lhs,
                                          *rhs,
                                          BinaryOperator::LogicalAnd));
  };

  FailureOr<Value> authorResidual = residual(original);
  if (failed(authorResidual))
    return failure();
  if (!*authorResidual)
    return physicalTail;
  if (!physicalTail)
    return materializeBroadcastToFragment(builder, location, *authorResidual,
                                          target);
  return materializeValidityConjunction(builder, location, physicalTail,
                                        *authorResidual, target);
}

LogicalResult alignPointwiseValueRelations(func::FuncOp kernel) {
  auto sameSchema = [](FragmentType lhs, FragmentType rhs) {
    return lhs.getShape() == rhs.getShape() &&
           lhs.getAxisMaps() == rhs.getAxisMaps() &&
           lhs.getValidity() == rhs.getValidity() &&
           lhs.getOwner() == rhs.getOwner();
  };
  auto align = [&](Operation *operation, unsigned operandIndex,
                   FragmentType targetShape) -> LogicalResult {
    Value value = operation->getOperand(operandIndex);
    auto source = dyn_cast<FragmentType>(value.getType());
    if (source && sameSchema(source, targetShape))
      return success();
    if (!source) {
      if (!isa<IntegerType, FloatType, IndexType>(value.getType()))
        return failure();
      auto target = FragmentType::get(
          kernel.getContext(), value.getType(), targetShape.getShape(),
          targetShape.getAxisMaps(), targetShape.getValidity(),
          targetShape.getOwner());
      OpBuilder builder(operation);
      Value replacement =
          builder.create<SplatOp>(operation->getLoc(), target, value);
      if (Operation *definition = value.getDefiningOp())
        if (Attribute origin = definition->getAttr(originAttr))
          replacement.getDefiningOp()->setAttr(originAttr, origin);
      operation->setOperand(operandIndex, replacement);
      return success();
    }
    if (source.getShape().size() > targetShape.getShape().size())
      return failure();
    for (Attribute sourceMapping : source.getAxisMaps()) {
      auto sourceAxis = cast<AxisMapAttr>(sourceMapping);
      if (!llvm::any_of(targetShape.getAxisMaps(), [&](Attribute targetMapping) {
            auto targetAxis = cast<AxisMapAttr>(targetMapping);
            return sourceAxisIdentity(sourceAxis) ==
                   sourceAxisIdentity(targetAxis);
          }))
        return failure();
    }
    Operation *definition = value.getDefiningOp();
    auto target = FragmentType::get(
        kernel.getContext(), source.getElementType(), targetShape.getShape(),
        targetShape.getAxisMaps(), targetShape.getValidity(),
        targetShape.getOwner());
    OpBuilder builder(operation);
    Value replacement;
    if (auto broadcast = dyn_cast_or_null<BroadcastOp>(definition))
      replacement = builder.create<BroadcastOp>(operation->getLoc(), target,
                                                broadcast.getValue());
    else if (auto splat = dyn_cast_or_null<SplatOp>(definition))
      replacement = builder.create<SplatOp>(operation->getLoc(), target,
                                            splat.getValue());
    else
      replacement =
          builder.create<BroadcastOp>(operation->getLoc(), target, value);
    if (definition)
      if (Attribute origin = definition->getAttr(originAttr))
        replacement.getDefiningOp()->setAttr(originAttr, origin);
    operation->setOperand(operandIndex, replacement);
    return success();
  };

  WalkResult result = kernel.walk([&](Operation *operation) {
    FragmentType target;
    if (operation->getNumResults() == 1)
      target = dyn_cast<FragmentType>(operation->getResult(0).getType());
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(
            operation))
      return WalkResult::advance();
    if (!target && operation->getNumResults() == 1 &&
        isa<IntegerType, FloatType, IndexType>(
            operation->getResult(0).getType())) {
      FragmentType prototype;
      for (Value operand : operation->getOperands())
        if ((prototype = dyn_cast<FragmentType>(operand.getType())))
          break;
      if (prototype) {
        target = FragmentType::get(
            kernel.getContext(), operation->getResult(0).getType(),
            prototype.getShape(), prototype.getAxisMaps(),
            prototype.getValidity(), prototype.getOwner());
        operation->getResult(0).setType(target);
      }
    }
    if (!target)
      return WalkResult::advance();
    for (unsigned index = 0; index < operation->getNumOperands(); ++index)
      if (failed(align(operation, index, target)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignAccessValueRelations(func::FuncOp kernel) {
  auto predicateType = [&](FragmentType value) {
    return FragmentType::get(
        kernel.getContext(), IntegerType::get(kernel.getContext(), 1),
        value.getShape(), value.getAxisMaps(), value.getValidity(),
        value.getOwner());
  };
  auto project = [&](OpBuilder &builder, Location location, Value value,
                     Type target) -> FailureOr<Value> {
    if (!value)
      return failure();
    return projectPhysicalValueToSchema(builder, location, value, target);
  };

  SmallVector<LoadOp> loads;
  SmallVector<GatherOp> gathers;
  SmallVector<StoreOp> stores;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });

  for (LoadOp load : loads) {
    if (!load.getValid() && !load.getFill())
      continue;
    if (!load.getValid())
      return load.emitOpError("load fill has no validity authority");
    OpBuilder builder(load);
    Type valueType = load.getResult().getType();
    Type validType = builder.getI1Type();
    if (auto fragment = dyn_cast<FragmentType>(valueType))
      validType = predicateType(fragment);
    FailureOr<Value> valid = project(builder, load.getLoc(), load.getValid(),
                                     validType);
    if (failed(valid))
      return load.emitOpError("cannot align load validity with its value schema");
    FailureOr<Value> fill = failure();
    if (load.getFill())
      fill = project(builder, load.getLoc(), load.getFill(), valueType);
    else if (auto fragment = dyn_cast<FragmentType>(valueType))
      fill = materializeZeroFragment(builder, load.getLoc(), fragment);
    else {
      TypedAttr zero;
      if (auto integer = dyn_cast<IntegerType>(valueType))
        zero = builder.getIntegerAttr(integer, 0);
      else if (auto floating = dyn_cast<FloatType>(valueType))
        zero = builder.getFloatAttr(floating, 0.0);
      else if (isa<IndexType>(valueType))
        zero = builder.getIndexAttr(0);
      if (zero)
        fill = materializeScalarConstant(builder, load.getLoc(), zero, valueType);
    }
    if (failed(fill))
      return load.emitOpError("cannot align load fill with its value schema");
    if (*valid == load.getValid() && *fill == load.getFill())
      continue;
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), valueType, load.getResource(), load.getCoordinates(),
        *valid, *fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }

  for (GatherOp gather : gathers) {
    if (!gather.getValid() && !gather.getFill())
      continue;
    if (!gather.getValid())
      return gather.emitOpError("gather fill has no validity authority");
    OpBuilder builder(gather);
    Type valueType = gather.getResult().getType();
    auto fragment = dyn_cast<FragmentType>(valueType);
    if (!fragment) {
      if (gather.getValid() && gather.getFill())
        continue;
      return gather.emitOpError(
          "scalar gather validity and fill must remain paired");
    }
    FailureOr<Value> valid = project(
        builder, gather.getLoc(), gather.getValid(), predicateType(fragment));
    if (failed(valid))
      return gather.emitOpError(
          "cannot align gather validity with its value schema");
    FailureOr<Value> fill =
        gather.getFill()
            ? project(builder, gather.getLoc(), gather.getFill(), valueType)
            : materializeZeroFragment(builder, gather.getLoc(), fragment);
    if (failed(fill))
      return gather.emitOpError("cannot align gather fill with its value schema");
    if (*valid == gather.getValid() && *fill == gather.getFill())
      continue;
    auto replacement = builder.create<GatherOp>(
        gather.getLoc(), valueType, gather.getSource(), gather.getCoordinates(),
        *valid, *fill, gather.getSourceAxes());
    if (Attribute origin = gather->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
  }

  for (StoreOp store : stores) {
    if (!store.getValid())
      continue;
    auto valueType = dyn_cast<FragmentType>(store.getValue().getType());
    if (!valueType)
      continue;
    OpBuilder builder(store);
    FailureOr<Value> valid = project(
        builder, store.getLoc(), store.getValid(), predicateType(valueType));
    if (failed(valid))
      return store.emitOpError(
          "cannot align store validity with its value schema");
    if (*valid == store.getValid())
      continue;
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        store.getValue(), *valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

LogicalResult refreshReshapeRelations(func::FuncOp kernel) {
  WalkResult result = kernel.walk([&](ReshapeOp reshape) {
    auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
    auto target = dyn_cast<FragmentType>(reshape.getResult().getType());
    if (!source || !target) {
      reshape.emitOpError(
          "reshape relation requires physical fragment operands and result");
      return WalkResult::interrupt();
    }
    unsigned logicalSourceRank = 0;
    unsigned logicalResultRank = 0;
    for (Attribute attribute : reshape.getReassociation()) {
      auto group = dyn_cast<ReshapeGroupAttr>(attribute);
      if (!group) {
        reshape.emitOpError("reshape relation contains an untyped group");
        return WalkResult::interrupt();
      }
      if (!group.getSourceAxes().empty())
        logicalSourceRank =
            std::max(logicalSourceRank,
                     static_cast<unsigned>(
                         group.getSourceAxes().asArrayRef().back() + 1));
      if (!group.getResultAxes().empty())
        logicalResultRank =
            std::max(logicalResultRank,
                     static_cast<unsigned>(
                         group.getResultAxes().asArrayRef().back() + 1));
    }
    if (logicalSourceRank > source.getShape().size() ||
        logicalResultRank > target.getShape().size()) {
      reshape.emitOpError(
          "reshape relation rank exceeds the current physical fragments");
      return WalkResult::interrupt();
    }
    unsigned sourcePrefix = source.getShape().size() - logicalSourceRank;
    unsigned resultPrefix = target.getShape().size() - logicalResultRank;
    FailureOr<ArrayAttr> reassociation = inferReshapeReassociation(
        source, target, sourcePrefix, resultPrefix);
    if (failed(reassociation)) {
      reshape.emitOpError(
          "current physical extents have no exact row-major reshape relation")
          << "; source=" << source << "; target=" << target;
      return WalkResult::interrupt();
    }
    reshape->setAttr("reassociation", *reassociation);
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult alignAggregateValueRelations(func::FuncOp kernel) {
  kernel.walk([&](MakeRecordOp record) {
    RecordType current = record.getResult().getType();
    SmallVector<Attribute> fields;
    fields.reserve(record.getFields().size());
    for (Value field : record.getFields())
      fields.push_back(TypeAttr::get(field.getType()));
    record.getResult().setType(RecordType::get(
        kernel.getContext(), current.getFieldNames(),
        ArrayAttr::get(kernel.getContext(), fields), current.getOwner()));
  });
  kernel.walk([&](ExtractOp extract) {
    RecordType record = extract.getRecord().getType();
    if (extract.getField() < record.getFieldTypes().size())
      extract.getResult().setType(
          cast<TypeAttr>(record.getFieldTypes()[extract.getField()]).getValue());
  });
  return success();
}

FailureOr<Value> resolveLogicalRangeEnd(func::FuncOp kernel,
                                        MakeRangeOp range) {
  (void)kernel;
  return range.getLogicalStop();
}

static void retargetExtent(Value root, AxisSelector selects,
                           PhysicalExprAttr extent) {
  auto rootFragment = dyn_cast<FragmentType>(root.getType());
  if (!rootFragment)
    return;
  SmallVector<Attribute> previousExtents;
  for (auto [axis, attribute] : llvm::enumerate(rootFragment.getAxisMaps()))
    if (selects(cast<AxisMapAttr>(attribute)) &&
        !llvm::is_contained(previousExtents, rootFragment.getShape()[axis]))
      previousExtents.push_back(rootFragment.getShape()[axis]);
  if (previousExtents.empty())
    return;
  SmallVector<Attribute> connectedExtents(previousExtents.begin(),
                                          previousExtents.end());
  if (!llvm::is_contained(connectedExtents, Attribute(extent)))
    connectedExtents.push_back(extent);
  SmallVector<Value> worklist{root};
  llvm::DenseMap<Value, Type> visitedTypes;
  auto isSegmentSourceSlice = [&](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return false;
    Operation *parent = argument.getOwner()->getParentOp();
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent))
      return argument.getOwner()->getParent() == &fold.getSummarize() &&
             argument.getArgNumber() < fold.getSourceCount() &&
             selectsSegmentAxis(argument.getType(), fold.getAxis(), selects);
    if (auto scan = dyn_cast_or_null<RegionScanOp>(parent))
      return (argument.getOwner()->getParent() == &scan.getSummarize() ||
              argument.getOwner()->getParent() == &scan.getEmit()) &&
             argument.getArgNumber() < scan.getSourceCount() &&
             selectsSegmentAxis(argument.getType(), scan.getAxis(), selects);
    return false;
  };
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto [visited, inserted] = visitedTypes.try_emplace(value, value.getType());
    if (!inserted && visited->second == value.getType())
      continue;
    visited->second = value.getType();
    // A structured segment slice has one exact extent authority: the segment
    // parameter owned by its region_fold/region_scan operation.  Retargeting
    // stops only for that segmented axis; every non-segment axis must preserve
    // the source schema across the helper boundary.
    if (isSegmentSourceSlice(value))
      continue;
    // Each make_range is an independent physical traversal authority.  A
    // source-axis extent selected for one range may flow through its users,
    // but must not cross a shared consumer and rewrite a different range that
    // happens to carry the same logical provenance.
    if (value != root && value.getDefiningOp<MakeRangeOp>())
      continue;
    // A reduction or other structured operation can consume an axis and
    // produce a scalar/record that no longer carries it.  The extent decision
    // ends there; following the scalar into a later broadcast would conflate a
    // new traversal with the one being retargeted.
    if (!carriesExtent(value.getType(), selects, connectedExtents))
      continue;
    if (!preservesIntroducedUnitAxis(value, selects)) {
      Type replacement =
          replaceExtent(value.getType(), selects, previousExtents, extent);
      if (replacement != value.getType()) {
        value.setType(replacement);
        // A make_range owns both the fragment schema and the SSA extent used
        // to materialize that schema.  Retarget them from the same physical
        // decision; leaving the operand behind creates two executable
        // authorities for one traversal.
        if (auto range = value.getDefiningOp<MakeRangeOp>()) {
          OpBuilder builder(range);
          Value physicalExtent;
          if (extent.getKind() ==
              static_cast<uint32_t>(PhysicalExprKind::Constant))
            physicalExtent = builder.create<arith::ConstantIndexOp>(
                range.getLoc(), extent.getValue());
          else
            physicalExtent = builder.create<PhysicalExprOp>(
                range.getLoc(), builder.getIndexType(), extent);
          range->setOperand(1, physicalExtent);
        }
      }
    }
    // Product fields and structured helper arguments are part of the same
    // physical value flow even though MLIR does not connect them with ordinary
    // result uses.  A blocking decision for one provenance axis must cross
    // those boundaries; otherwise an operation can retain two physical
    // schemas for one summary/carry value.
    if (auto record = value.getDefiningOp<MakeRecordOp>())
      worklist.append(record.getFields().begin(), record.getFields().end());
    if (auto extract = value.getDefiningOp<ExtractOp>())
      worklist.push_back(extract.getRecord());
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      appendStructuredParentRelations(argument, worklist, selects);
      auto loop = dyn_cast_or_null<scf::ForOp>(
          argument.getOwner()->getParentOp());
      if (loop)
        for (auto [index, iterArgument] :
             llvm::enumerate(loop.getRegionIterArgs()))
          if (argument == iterArgument) {
            worklist.push_back(loop.getInitArgs()[index]);
            worklist.push_back(loop.getResult(index));
          }
    }
    if (auto loop = value.getDefiningOp<scf::ForOp>())
      for (auto [index, result] : llvm::enumerate(loop.getResults()))
        if (value == result) {
          worklist.push_back(loop.getInitArgs()[index]);
          worklist.push_back(loop.getRegionIterArgs()[index]);
          worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
        }
    appendStructuredResultRelations(value, worklist);
    for (Operation *user : value.getUsers()) {
      if (isa<RegionFoldOp, RegionScanOp>(user)) {
        appendStructuredChildRelations(user, value, worklist, selects);
        for (Value result : user->getResults())
          worklist.push_back(result);
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(user))
        for (auto [index, init] : llvm::enumerate(loop.getInitArgs()))
          if (value == init) {
            worklist.push_back(loop.getRegionIterArgs()[index]);
            worklist.push_back(loop.getResult(index));
          }
      if (auto yield = dyn_cast<scf::YieldOp>(user))
        if (auto loop = dyn_cast_or_null<scf::ForOp>(yield->getParentOp()))
          for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
            if (value == yielded) {
              worklist.push_back(loop.getInitArgs()[index]);
              worklist.push_back(loop.getRegionIterArgs()[index]);
              worklist.push_back(loop.getResult(index));
            }
      if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
              ContractOp, ScaledContractOp, SparseContractOp, ReduceOp,
              ScanOp, RandomBitsOp,
              ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
              AtomicCompareExchangeOp>(user)) {
        worklist.append(user->getOperands().begin(), user->getOperands().end());
        for (Region &region : user->getRegions())
          for (Block &block : region)
            for (BlockArgument argument : block.getArguments())
              if (!isSegmentSourceSlice(argument))
                worklist.push_back(argument);
      }
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
}

void retargetSourceExtent(Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent) {
  retargetExtent(
      root,
      [=](AxisMapAttr mapping) {
        return mapping.getSourceId() == source.sourceId &&
               mapping.getSourceAxis() == source.sourceAxis &&
               mapping.getDerived() == source.derived;
      },
      extent);
}

void retargetDimensionExtent(Value root, int64_t dimensionId,
                             PhysicalExprAttr extent) {
  if (dimensionId <= 0)
    return;
  retargetExtent(root,
                 [=](AxisMapAttr mapping) {
                   return mapping.getDimensionId() == dimensionId;
                 },
                 extent);
}

LogicalResult alignStructuredCaptureRelations(func::FuncOp kernel) {
  auto align = [&](Operation *owner, Value capture,
                   BlockArgument argument) -> LogicalResult {
    auto authority = dyn_cast<FragmentType>(capture.getType());
    auto target = dyn_cast<FragmentType>(argument.getType());
    if (!authority || !target)
      return capture.getType() == argument.getType()
                 ? success()
                 : owner->emitOpError(
                       "physical structured capture lost its parent schema");
    if (authority.getElementType() != target.getElementType() ||
        authority.getShape().size() != target.getShape().size() ||
        authority.getAxisMaps() != target.getAxisMaps() ||
        authority.getOwner() != target.getOwner())
      return owner->emitOpError(
          "physical structured capture changed its coordinate relation");
    for (auto [axis, attribute] : llvm::enumerate(authority.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getDimensionId() <= 0)
        continue;
      retargetDimensionExtent(argument, mapping.getDimensionId(),
                              cast<PhysicalExprAttr>(authority.getShape()[axis]));
    }
    return capture.getType() == argument.getType()
               ? success()
               : owner->emitOpError(
                     "physical structured capture extent is inconsistent");
  };

  WalkResult result = kernel.walk([&](Operation *operation) {
    if (auto fold = dyn_cast<RegionFoldOp>(operation)) {
      unsigned sourceCount = fold.getSourceCount();
      unsigned captureOffset = sourceCount + fold.getIdentityCount();
      for (unsigned index = 0; index < fold.getCaptureCount(); ++index)
        if (failed(align(operation, fold.getInputs()[captureOffset + index],
                         fold.getSummarize().front().getArgument(sourceCount +
                                                                 index))))
          return WalkResult::interrupt();
      return WalkResult::advance();
    }
    auto scan = dyn_cast<RegionScanOp>(operation);
    if (!scan)
      return WalkResult::advance();
    unsigned sourceCount = scan.getSourceCount();
    unsigned stateCount = scan.getStateCount();
    unsigned captureOffset =
        sourceCount + scan.getIdentityCount() + stateCount;
    for (unsigned index = 0; index < scan.getCaptureCount(); ++index) {
      Value capture = scan.getInputs()[captureOffset + index];
      if (failed(align(operation, capture,
                       scan.getSummarize().front().getArgument(sourceCount +
                                                               index))) ||
          failed(align(operation, capture,
                       scan.getEmit().front().getArgument(sourceCount +
                                                          stateCount + index))))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

FailureOr<uint64_t> blockedDimension(Attribute attribute) {
  auto extent = dyn_cast<PhysicalExprAttr>(attribute);
  if (!extent ||
      extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::CeilDiv) ||
      extent.getOperands().size() != 2)
    return failure();
  auto logical = dyn_cast<PhysicalExprAttr>(extent.getOperands()[0]);
  auto block = dyn_cast<PhysicalExprAttr>(extent.getOperands()[1]);
  if (!logical || !block ||
      logical.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Dimension) ||
      block.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  return logical.getValue() > 0
             ? FailureOr<uint64_t>(logical.getValue())
             : FailureOr<uint64_t>(failure());
}

bool hasBlockedDimension(func::FuncOp kernel, uint64_t dimension) {
  bool found = false;
  kernel.walk([&](DelinearizeOp mapping) {
    for (Attribute extent : mapping.getLaunchExtents()) {
      FailureOr<uint64_t> blocked = blockedDimension(extent);
      found |= succeeded(blocked) && *blocked == dimension;
    }
  });
  return found;
}

LogicalResult realizeFullCoverageDimension(func::FuncOp kernel, Value source,
                                           unsigned fragmentAxis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || fragmentAxis >= fragment.getShape().size())
    return failure();
  auto mapping =
      dyn_cast<AxisMapAttr>(fragment.getAxisMaps()[fragmentAxis]);
  if (!mapping || mapping.getDimensionId() <= 0)
    return failure();
  const int64_t dimension = mapping.getDimensionId();
  auto currentExtent =
      cast<PhysicalExprAttr>(fragment.getShape()[fragmentAxis]);

  Value runtimeDimension;
  for (BlockArgument argument : kernel.getArguments()) {
    DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == dimension) {
      if (runtimeDimension && runtimeDimension != argument)
        return kernel.emitError(
            "logical dimension has multiple runtime ABI authorities");
      runtimeDimension = argument;
    }
  }

  std::optional<int64_t> staticDimension;
  for (BlockArgument argument : kernel.getArguments()) {
    auto view = dyn_cast<ViewType>(argument.getType());
    if (!view)
      continue;
    for (auto [axis, identity] :
         llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
      if (identity != dimension)
        continue;
      auto extent = cast<PhysicalExprAttr>(
          view.getLayout().getExtents()[axis]);
      if (extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant))
        continue;
      if (staticDimension && *staticDimension != extent.getValue())
        return kernel.emitError(
            "logical dimension has conflicting static ABI extents");
      staticDimension = extent.getValue();
    }
  }
  if (!runtimeDimension && staticDimension &&
      currentExtent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Constant) &&
      currentExtent.getValue() == *staticDimension)
    return success();
  if (!runtimeDimension)
    return kernel.emitError(
        "full-coverage physicalization has no runtime or static dimension authority");

  ParameterOp parameter;
  if (currentExtent.getKind() ==
      static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
    FailureOr<ParameterOp> declaration =
        queryParameterBySymbol(kernel, currentExtent.getSymbol());
    if (succeeded(declaration)) {
      auto covered = (*declaration)->getAttrOfType<IntegerAttr>(
          coverageDimensionAttr);
      if (covered && covered.getInt() == dimension)
        parameter = *declaration;
    }
  }
  if (!parameter) {
    kernel.walk([&](ParameterOp candidate) {
      auto covered = candidate->getAttrOfType<IntegerAttr>(
          coverageDimensionAttr);
      if (!covered || covered.getInt() != dimension)
        return;
      if (parameter && parameter != candidate) {
        parameter = ParameterOp();
        return;
      }
      parameter = candidate;
    });
  }
  static constexpr int64_t candidates[] = {
      1,    2,    4,     8,     16,    32,    64,    128,   256,
      512,  1024, 2048,  4096,  8192,  16384, 32768, 65536};
  if (!parameter) {
    std::string name = ("FULL_D" + Twine(dimension)).str();
    bool nameCollision = false;
    kernel.walk([&](ParameterOp candidate) {
      nameCollision |=
          candidate.getParameter().getName().getValue() == name;
    });
    if (nameCollision)
      return kernel.emitError(
          "full-coverage parameter name is already owned by another decision");
    OpBuilder builder(&kernel.getBody().front(),
                      kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(name),
        static_cast<uint32_t>(ParameterRole::FullCoverage),
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    parameter = builder.create<ParameterOp>(source.getLoc(),
                                            builder.getIndexType(), schema);
  } else {
    ParameterAttr schema = parameter.getParameter();
    parameter->setAttr(
        "parameter",
        ParameterAttr::get(
            kernel.getContext(), schema.getName(), schema.getRole(),
            DenseI64ArrayAttr::get(kernel.getContext(), candidates)));
  }
  parameter->setAttr(
      dimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));

  auto covered = PhysicalExprAttr::get(
      kernel.getContext(),
      static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      parameter.getParameter().getName(),
      ArrayAttr::get(kernel.getContext(), {}));
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, fragmentAxis);
  if (!ranges.roots.empty() && failed(queryExactLogicalRange(ranges)))
    return kernel.emitError(
        "full-coverage source has ambiguous physical range authority");
  if (ranges.state == PhysicalFactState::Unknown || ranges.roots.empty()) {
    PhysicalReplayFact replay = analysis.replayability(
        source, sourceAxisIdentity(mapping), PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/false);
    if (!ranges.roots.empty() || !replay.isReplayable()) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "full-coverage source has no exact physical range authority");
      for (Operation *blocker : ranges.blockers)
        diagnostic << "; blocker=" << blocker->getName();
      return failure();
    }
  }
  for (MakeRangeOp range : ranges.roots) {
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    if (failed(rangeDimension) || *rangeDimension != dimension ||
        range->hasAttr(sourceSubregionAttr))
      return range.emitOpError(
          "full-coverage range does not cover the selected logical dimension");
    retargetDimensionExtent(range.getResult(), dimension, covered);
  }
  retargetDimensionExtent(source, dimension, covered);
  if (failed(bindFullCoverageDimension(kernel, dimension,
                                       parameter.getResult())))
    return kernel.emitError(
        "full-coverage decision could not preserve access validity");
  return success();
}

LogicalResult bindFullCoverageDimension(func::FuncOp kernel, uint64_t dimension,
                                        Value physicalExtent) {
  auto parameter = physicalExtent.getDefiningOp<ParameterOp>();
  if (!parameter)
    return failure();
  PhysicalExprAttr parameterExtent = PhysicalExprAttr::get(
      kernel.getContext(),
      static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      parameter.getParameter().getName(),
      ArrayAttr::get(kernel.getContext(), {}));
  SmallVector<MakeRangeOp> ranges;
  kernel.walk([&](MakeRangeOp range) {
    FailureOr<int64_t> sourceDimension = querySourceDimension(
        range.getResult().getType(), sourceAxisIdentity(range));
    auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension) ||
        range->hasAttr(sourceSubregionAttr) || !fragment ||
        fragment.getShape().size() != 1)
      return;
    ranges.push_back(range);
  });
  bool alreadyBound = !ranges.empty() &&
                      llvm::all_of(ranges, [&](MakeRangeOp range) {
                        return range.getExtent() == physicalExtent;
                      });
  for (MakeRangeOp range : ranges)
    retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                         parameterExtent);
  if (ranges.empty() || alreadyBound)
    return success();

  llvm::SmallPtrSet<Operation *, 32> rangeSet;
  for (MakeRangeOp range : ranges)
    rangeSet.insert(range.getOperation());

  SmallVector<LoadOp> loads;
  SmallVector<GatherOp> gathers;
  SmallVector<StoreOp> stores;
  SmallVector<ScatterReduceOp> scatters;
  SmallVector<AtomicLoadOp> atomicLoads;
  SmallVector<AtomicStoreOp> atomicStores;
  SmallVector<AtomicRMWOp> atomicRMWs;
  SmallVector<AtomicCompareExchangeOp> atomicCAS;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  kernel.walk([&](ScatterReduceOp scatter) { scatters.push_back(scatter); });
  kernel.walk([&](AtomicLoadOp atomic) { atomicLoads.push_back(atomic); });
  kernel.walk([&](AtomicStoreOp atomic) { atomicStores.push_back(atomic); });
  kernel.walk([&](AtomicRMWOp atomic) { atomicRMWs.push_back(atomic); });
  kernel.walk(
      [&](AtomicCompareExchangeOp atomic) { atomicCAS.push_back(atomic); });
  llvm::DenseMap<Operation *, SmallVector<MakeRangeOp>> accessRanges;
  PhysicalProgramAnalysis accessAnalysis(kernel);
  auto recordAccessRanges = [&](Operation *access) -> LogicalResult {
    PhysicalAccessFootprint footprint = accessAnalysis.footprint(access);
    if (footprint.state != PhysicalFactState::Exact ||
        footprint.rangeState != PhysicalFactState::Exact)
      return access->emitOpError(
          "full-coverage rewrite requires one exact physical access footprint");
    auto &relevant = accessRanges[access];
    for (MakeRangeOp range : footprint.ranges)
      if (rangeSet.contains(range.getOperation()) &&
          !llvm::is_contained(relevant, range))
        relevant.push_back(range);
    return success();
  };
  for (LoadOp load : loads)
    if (failed(recordAccessRanges(load)))
      return failure();
  for (GatherOp gather : gathers)
    if (failed(recordAccessRanges(gather)))
      return failure();
  for (StoreOp store : stores)
    if (failed(recordAccessRanges(store)))
      return failure();
  for (ScatterReduceOp scatter : scatters)
    if (failed(recordAccessRanges(scatter)))
      return failure();
  for (AtomicLoadOp atomic : atomicLoads)
    if (failed(recordAccessRanges(atomic)))
      return failure();
  for (AtomicStoreOp atomic : atomicStores)
    if (failed(recordAccessRanges(atomic)))
      return failure();
  for (AtomicRMWOp atomic : atomicRMWs)
    if (failed(recordAccessRanges(atomic)))
      return failure();
  for (AtomicCompareExchangeOp atomic : atomicCAS)
    if (failed(recordAccessRanges(atomic)))
      return failure();

  llvm::DenseMap<Operation *, Value> predicates;
  for (MakeRangeOp range : ranges) {
    OpBuilder builder(range);
    range->setOperand(1, physicalExtent);
    builder.setInsertionPointAfter(range);
    Value distance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getLogicalStop(),
        range.getLogicalStart(), BinaryOperator::Subtract);
    Value one = builder.create<arith::ConstantIndexOp>(range.getLoc(), 1);
    Value adjusted = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), distance,
        builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                 range.getStep(), one,
                                 BinaryOperator::Subtract),
        BinaryOperator::Add);
    Value logicalExtent = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), adjusted, range.getStep(),
        BinaryOperator::FloorDivide);
    Value logicalDistance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), logicalExtent, range.getStep(),
        BinaryOperator::Multiply);
    Value stop = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getStart(), logicalDistance,
        BinaryOperator::Add);
    auto coordinate = cast<FragmentType>(range.getResult().getType());
    Value stopFragment =
        builder.create<BroadcastOp>(range.getLoc(), coordinate, stop);
    auto predicate = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinate.getShape(),
        coordinate.getAxisMaps(), coordinate.getValidity(),
        coordinate.getOwner());
    predicates[range.getOperation()] = builder.create<CompareOp>(
        range.getLoc(), predicate, range.getResult(), stopFragment,
        ComparePredicate::Lt);
  }

  auto materializeTail = [&](OpBuilder &builder, Location location,
                             FragmentType target,
                             ArrayRef<MakeRangeOp> sources) -> FailureOr<Value> {
    Value result;
    for (MakeRangeOp range : sources) {
      SmallVector<PhysicalAxisProjection, 2> projections =
          queryRangeProjections(target, range);
      if (FailureOr<int64_t> dimension = queryRangeDimension(range);
          succeeded(dimension))
        for (PhysicalDimensionProjection occurrence :
             queryFragmentDimensions(target, *dimension))
          if (!llvm::any_of(projections,
                            [&](const PhysicalAxisProjection &projection) {
                              return projection.fragmentAxis ==
                                     occurrence.fragmentAxis;
                            }))
            projections.push_back(PhysicalAxisProjection{
                PhysicalFactState::Exact, sourceAxisIdentity(range), *dimension,
                occurrence.fragmentAxis});
      if (projections.empty())
        return failure();
      for (PhysicalAxisProjection projection : projections) {
        FailureOr<Value> current = projectPredicate(
            builder, location, predicates.lookup(range.getOperation()), target,
            projection.fragmentAxis);
        if (failed(current))
          return failure();
        result = result ? Value(builder.create<BinaryOp>(
                              location, current->getType(), result, *current,
                              BinaryOperator::LogicalAnd))
                        : *current;
      }
    }
    return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
  };

  auto combineValidity = [&](OpBuilder &builder, Location location,
                             FragmentType type, ArrayRef<MakeRangeOp> sources,
                             Value existing) -> FailureOr<Value> {
    FailureOr<Value> tail = materializeTail(builder, location, type, sources);
    if (failed(tail))
      return failure();
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (!existing)
      return valid;
    Type element = existing.getType();
    if (auto fragment = dyn_cast<FragmentType>(element))
      element = fragment.getElementType();
    if (!element.isInteger(1))
      return failure();
    if (existing.getType() != predicate)
      existing = builder.create<BroadcastOp>(location, predicate, existing);
    return Value(builder.create<BinaryOp>(location, predicate, existing, valid,
                                          BinaryOperator::LogicalAnd));
  };

  for (LoadOp load : loads) {
    auto sources = accessRanges.lookup(load.getOperation());
    auto type = dyn_cast<FragmentType>(load.getResult().getType());
    if (!load->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(load);
    FailureOr<Value> tail =
        materializeTail(builder, load.getLoc(), type, sources);
    if (failed(tail))
      return load.emitOpError(
          "cannot project full-coverage dimension to load validity");
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (load.getValid()) {
      Value existing = load.getValid();
      Type element = existing.getType();
      if (auto fragment = dyn_cast<FragmentType>(element))
        element = fragment.getElementType();
      if (!element.isInteger(1))
        return load.emitOpError(
            "full-coverage load carried non-predicate validity");
      if (existing.getType() != predicate)
        existing = builder.create<BroadcastOp>(load.getLoc(), predicate, existing);
      valid = builder.create<BinaryOp>(load.getLoc(), predicate, existing, valid,
                                       BinaryOperator::LogicalAnd);
    }
    Value fill = load.getFill();
    if (!fill) {
      FailureOr<Value> zero = zeroFill(builder, load.getLoc(), type);
      if (failed(zero))
        return load.emitOpError("full-coverage load has no neutral fill");
      fill = *zero;
    } else if (fill.getType() != type)
      fill = builder.create<BroadcastOp>(load.getLoc(), type, fill);
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), type, load.getResource(), load.getCoordinates(), valid,
        fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }

  for (GatherOp gather : gathers) {
    auto sources = accessRanges.lookup(gather.getOperation());
    auto type = dyn_cast<FragmentType>(gather.getResult().getType());
    if (!gather->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(gather);
    FailureOr<Value> valid = combineValidity(
        builder, gather.getLoc(), type, sources, gather.getValid());
    if (failed(valid))
      return gather.emitOpError(
          "cannot project full-coverage dimension to gather validity");
    Value fill = gather.getFill();
    if (!fill) {
      FailureOr<Value> zero = zeroFill(builder, gather.getLoc(), type);
      if (failed(zero))
        return gather.emitOpError("full-coverage gather has no neutral fill");
      fill = *zero;
    } else if (fill.getType() != type)
      fill = builder.create<BroadcastOp>(gather.getLoc(), type, fill);
    auto replacement = builder.create<GatherOp>(
        gather.getLoc(), type, gather.getSource(), gather.getCoordinates(),
        *valid, fill, gather.getSourceAxes());
    if (Attribute origin = gather->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
  }

  for (StoreOp store : stores) {
    auto sources = accessRanges.lookup(store.getOperation());
    auto type = dyn_cast<FragmentType>(store.getValue().getType());
    if (!store->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(store);
    FailureOr<Value> tail =
        materializeTail(builder, store.getLoc(), type, sources);
    if (failed(tail))
      return store.emitOpError(
          "cannot project full-coverage dimension to store validity");
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (store.getValid()) {
      Value existing = store.getValid();
      if (existing.getType() != predicate)
        existing =
            builder.create<BroadcastOp>(store.getLoc(), predicate, existing);
      valid = builder.create<BinaryOp>(store.getLoc(), predicate, existing,
                                       valid, BinaryOperator::LogicalAnd);
    }
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        store.getValue(), valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }

  for (ScatterReduceOp scatter : scatters) {
    auto sources = accessRanges.lookup(scatter.getOperation());
    auto type = dyn_cast<FragmentType>(scatter.getValue().getType());
    if (!scatter->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(scatter);
    FailureOr<Value> valid = combineValidity(
        builder, scatter.getLoc(), type, sources, scatter.getValid());
    if (failed(valid))
      return scatter.emitOpError(
          "cannot project full-coverage dimension to scatter validity");
    OperationState state(scatter.getLoc(), ScatterReduceOp::getOperationName());
    state.addOperands(scatter.getResource());
    state.addOperands(scatter.getCoordinates());
    state.addOperands(scatter.getValue());
    state.addOperands(*valid);
    for (NamedAttribute attribute : scatter->getAttrs())
      if (attribute.getName() != "operandSegmentSizes")
        state.addAttribute(attribute.getName(), attribute.getValue());
    state.addAttribute(
        "operandSegmentSizes",
        builder.getDenseI32ArrayAttr(
            {1, static_cast<int32_t>(scatter.getCoordinates().size()), 1, 1}));
    state.addRegion();
    auto replacement = cast<ScatterReduceOp>(builder.create(state));
    replacement.getCombine().takeBody(scatter.getCombine());
    scatter.erase();
  }

  for (AtomicLoadOp atomic : atomicLoads) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getResult().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to atomic-load validity");
    auto replacement = builder.create<AtomicLoadOp>(
        atomic.getLoc(), type, atomic.getResource(), atomic.getCoordinates(),
        *valid, atomic.getOrdering(), atomic.getSharing(), atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }

  for (AtomicStoreOp atomic : atomicStores) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getValue().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to atomic-store validity");
    auto replacement = builder.create<AtomicStoreOp>(
        atomic.getLoc(), atomic.getResource(), atomic.getCoordinates(),
        atomic.getValue(), *valid, atomic.getOrdering(), atomic.getSharing(),
        atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.erase();
  }

  for (AtomicRMWOp atomic : atomicRMWs) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getValue().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to atomic-RMW validity");
    auto replacement = builder.create<AtomicRMWOp>(
        atomic.getLoc(), atomic.getResult().getType(), atomic.getResource(),
        atomic.getCoordinates(), atomic.getValue(), *valid, atomic.getKind(),
        atomic.getOrdering(), atomic.getSharing(), atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }

  for (AtomicCompareExchangeOp atomic : atomicCAS) {
    auto sources = accessRanges.lookup(atomic.getOperation());
    auto type = dyn_cast<FragmentType>(atomic.getExpected().getType());
    if (!atomic->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(atomic);
    FailureOr<Value> valid = combineValidity(
        builder, atomic.getLoc(), type, sources, atomic.getValid());
    if (failed(valid))
      return atomic.emitOpError(
          "cannot project full-coverage dimension to compare-exchange validity");
    auto replacement = builder.create<AtomicCompareExchangeOp>(
        atomic.getLoc(), atomic.getResult().getType(), atomic.getResource(),
        atomic.getCoordinates(), atomic.getExpected(), atomic.getDesired(),
        *valid, atomic.getOrdering(), atomic.getSharing(),
        atomic.getSourceAxes());
    if (Attribute origin = atomic->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }
  return success();
}

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  SmallVector<Operation *> operations;
  bool changed = false;
  do {
    changed = false;
    operations.clear();
    kernel.walk([&](Operation *operation) { operations.push_back(operation); });
    for (Operation *operation : llvm::reverse(operations)) {
      if (!operation->getBlock() || isa<DelinearizeOp, ParameterOp>(operation) ||
          operation == kernel.getOperation() || !operation->getNumResults() ||
          !llvm::all_of(operation->getResults(),
                        [](Value value) { return value.use_empty(); }))
        continue;
      if (isMemoryEffectFree(operation) || isa<LoadOp, GatherOp>(operation)) {
        operation->erase();
        changed = true;
      }
    }
  } while (changed);

}

void eraseUnusedPhysicalParameters(func::FuncOp kernel) {
  llvm::StringSet<> referencedParameters;
  kernel.walk([&](Operation *operation) {
    for (Type type : operation->getResultTypes())
      collectParameterSymbols(type, referencedParameters);
    for (NamedAttribute attribute : operation->getAttrs())
      collectParameterSymbols(attribute.getValue(), referencedParameters);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectParameterSymbols(argument.getType(), referencedParameters);
  });
  SmallVector<Operation *> operations;
  kernel.walk([&](Operation *operation) { operations.push_back(operation); });
  for (Operation *operation : llvm::reverse(operations)) {
    // Delinearize is the explicit execution-workset authority.  A singleton
    // workset may not consume its coordinate until a later blocking pass, so
    // ordinary SSA liveness cannot remove it between shared realizations.
    if (isa<DelinearizeOp>(operation))
      continue;
    auto parameter = dyn_cast<ParameterOp>(operation);
    if (!parameter ||
        referencedParameters.contains(
            parameter.getParameter().getName().getValue()))
      continue;
    if (operation == kernel.getOperation() || !operation->getNumResults() ||
        !llvm::all_of(operation->getResults(),
                      [](Value value) { return value.use_empty(); }))
      continue;
    if (isMemoryEffectFree(operation) || isa<LoadOp, GatherOp>(operation))
      operation->erase();
  }
}

} // namespace intent::gpu

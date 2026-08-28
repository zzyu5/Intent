#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

void collectParameterSymbols(Attribute attribute, llvm::StringSet<> &symbols);

Value stripBroadcast(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  return value;
}

bool isPhysicalRangeEnd(MakeRangeOp range, Value value) {
  auto addition = stripBroadcast(value).getDefiningOp<BinaryOp>();
  if (!addition || addition.getOperatorKind() != 0)
    return false;
  Value lhs = stripBroadcast(addition.getLhs());
  Value rhs = stripBroadcast(addition.getRhs());
  Value start = stripBroadcast(range.getStart());
  Value extent = stripBroadcast(range.getExtent());
  return (lhs == start && rhs == extent) ||
         (lhs == extent && rhs == start);
}

FragmentType replaceSourceExtent(FragmentType source, uint64_t sourceId,
                                 ArrayRef<Attribute> previousExtents,
                                 PhysicalExprAttr extent) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  bool changed = false;
  for (auto [axis, attribute] : llvm::enumerate(source.getAxisMaps())) {
    if (cast<AxisMapAttr>(attribute).getSourceId() != sourceId ||
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

Type replaceSourceExtent(Type source, uint64_t sourceId,
                         ArrayRef<Attribute> previousExtents,
                         PhysicalExprAttr extent) {
  if (auto fragment = dyn_cast<FragmentType>(source))
    return replaceSourceExtent(fragment, sourceId, previousExtents, extent);
  auto record = dyn_cast<RecordType>(source);
  if (!record)
    return source;
  SmallVector<Attribute> fields;
  bool changed = false;
  for (Attribute attribute : record.getFieldTypes()) {
    Type field = cast<TypeAttr>(attribute).getValue();
    Type replacement =
        replaceSourceExtent(field, sourceId, previousExtents, extent);
    fields.push_back(TypeAttr::get(replacement));
    changed |= replacement != field;
  }
  if (!changed)
    return source;
  return RecordType::get(source.getContext(), record.getFieldNames(),
                         ArrayAttr::get(source.getContext(), fields),
                         record.getOwner());
}

bool carriesSourceExtent(Type type, uint64_t sourceId,
                         ArrayRef<Attribute> extents) {
  if (auto fragment = dyn_cast<FragmentType>(type)) {
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (cast<AxisMapAttr>(attribute).getSourceId() == sourceId &&
          llvm::is_contained(extents, fragment.getShape()[axis]))
        return true;
    return false;
  }
  auto record = dyn_cast<RecordType>(type);
  if (!record)
    return false;
  return llvm::any_of(record.getFieldTypes(), [&](Attribute attribute) {
    return carriesSourceExtent(cast<TypeAttr>(attribute).getValue(), sourceId,
                               extents);
  });
}

void collectRanges(Value value, const llvm::SmallPtrSetImpl<Operation *> &eligible,
                   SmallVectorImpl<MakeRangeOp> &ranges,
                   llvm::SmallPtrSetImpl<Operation *> &visited) {
  Operation *producer = value.getDefiningOp();
  if (!producer || !visited.insert(producer).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    if (eligible.contains(producer) && !llvm::is_contained(ranges, range))
      ranges.push_back(range);
    return;
  }
  for (Value operand : producer->getOperands())
    collectRanges(operand, eligible, ranges, visited);
}

FailureOr<unsigned> sourceAxis(FragmentType fragment, uint64_t sourceId) {
  std::optional<unsigned> result;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != sourceId)
      continue;
    if (result)
      return failure();
    result = mapping.getFragmentAxis();
  }
  return result ? FailureOr<unsigned>(*result)
                : FailureOr<unsigned>(failure());
}

bool preservesIntroducedUnitSourceAxis(Value value, uint64_t sourceId) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  auto result = dyn_cast<FragmentType>(value.getType());
  if (!reshape || !result)
    return false;
  FailureOr<unsigned> axis = sourceAxis(result, sourceId);
  if (failed(axis))
    return false;
  auto extent = dyn_cast<PhysicalExprAttr>(result.getShape()[*axis]);
  if (!extent ||
      extent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      extent.getValue() != 1)
    return false;
  auto input = dyn_cast<FragmentType>(reshape.getValue().getType());
  return !input || failed(sourceAxis(input, sourceId));
}

FailureOr<Value> projectPredicate(OpBuilder &builder, Location location,
                                  Value predicate, FragmentType target,
                                  uint64_t sourceId) {
  auto base = dyn_cast<FragmentType>(predicate.getType());
  FailureOr<unsigned> axis = sourceAxis(target, sourceId);
  if (!base || base.getShape().size() != 1 || failed(axis))
    return failure();
  SmallVector<Attribute> shape(
      target.getShape().size(),
      PhysicalExprAttr::get(
          target.getContext(),
          static_cast<uint32_t>(PhysicalExprKind::Constant), 1,
          StringAttr::get(target.getContext()),
          ArrayAttr::get(target.getContext(), {})));
  shape[*axis] = base.getShape()[0];
  auto reshaped = FragmentType::get(
      target.getContext(), builder.getI1Type(),
      ArrayAttr::get(target.getContext(), shape), target.getAxisMaps(),
      target.getValidity(), target.getOwner());
  Value result = predicate;
  if (base != reshaped)
    result = builder.create<ReshapeOp>(location, reshaped, result,
                                      builder.getArrayAttr({}));
  auto projected = FragmentType::get(
      target.getContext(), builder.getI1Type(), target.getShape(),
      target.getAxisMaps(), target.getValidity(), target.getOwner());
  if (reshaped != projected)
    result = builder.create<BroadcastOp>(location, projected, result);
  return result;
}

Value zeroFill(OpBuilder &builder, Location location, FragmentType type) {
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(type.getElementType()))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(type.getElementType()))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(type.getElementType()))
    zero = builder.getIndexAttr(0);
  if (!zero)
    return {};
  FailureOr<Value> scalar =
      materializeScalarConstant(builder, location, zero, type.getElementType());
  if (failed(scalar))
    return {};
  return builder.create<BroadcastOp>(location, type, *scalar);
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

FailureOr<Value> resolveLogicalRangeEnd(func::FuncOp kernel,
                                        MakeRangeOp range) {
  auto dimension = range->getAttrOfType<IntegerAttr>(sourceDimensionAttr);
  if (dimension && !range->hasAttr(sourceSubregionAttr)) {
    for (BlockArgument argument : kernel.getArguments()) {
      DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
      auto kind = attributes.getAs<StringAttr>(abiKindAttr);
      auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
      if (kind && kind.getValue() == "dimension" && identity &&
          identity.getInt() == dimension.getInt())
        return Value(argument);
    }
    std::optional<int64_t> staticExtent;
    for (BlockArgument argument : kernel.getArguments()) {
      auto view = dyn_cast<ViewType>(argument.getType());
      if (!view)
        continue;
      auto identities = view.getLayout().getDimensionIds();
      auto extents = view.getLayout().getExtents();
      for (auto [axis, identity] : llvm::enumerate(identities.asArrayRef())) {
        if (identity != dimension.getInt())
          continue;
        auto extent = cast<PhysicalExprAttr>(extents[axis]);
        if (extent.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Constant))
          return failure();
        if (staticExtent && *staticExtent != extent.getValue())
          return failure();
        staticExtent = extent.getValue();
      }
    }
    if (staticExtent) {
      OpBuilder builder(range);
      return Value(builder.create<arith::ConstantIndexOp>(range.getLoc(),
                                                          *staticExtent));
    }
  }

  Value result;
  SmallVector<Value> worklist{range.getResult()};
  llvm::SmallDenseSet<Value> visited;
  while (!worklist.empty()) {
    Value coordinate = worklist.pop_back_val();
    if (!visited.insert(coordinate).second)
      continue;
    for (Operation *user : coordinate.getUsers()) {
      if (isa<BroadcastOp, ReshapeOp, TransposeOp>(user) &&
          user->getNumResults() == 1) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      auto comparison = dyn_cast<CompareOp>(user);
      if (!comparison || comparison.getPredicate() != 2 ||
          comparison.getLhs() != coordinate)
        continue;
      Value candidate = stripBroadcast(comparison.getRhs());
      if (range->hasAttr(sourceSubregionAttr) &&
          isPhysicalRangeEnd(range, candidate))
        continue;
      if (result && result != candidate)
        return failure();
      result = candidate;
    }
  }
  if (result)
    return result;

  OpBuilder builder(range);
  return Value(builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), range.getStart(),
      range.getExtent(), /*add=*/0));
}

void retargetSourceExtent(Value root, uint64_t sourceId,
                          PhysicalExprAttr extent) {
  auto rootFragment = dyn_cast<FragmentType>(root.getType());
  if (!rootFragment)
    return;
  SmallVector<Attribute> previousExtents;
  for (auto [axis, attribute] : llvm::enumerate(rootFragment.getAxisMaps()))
    if (cast<AxisMapAttr>(attribute).getSourceId() == sourceId &&
        !llvm::is_contained(previousExtents, rootFragment.getShape()[axis]))
      previousExtents.push_back(rootFragment.getShape()[axis]);
  if (previousExtents.empty())
    return;
  SmallVector<Value> worklist{root};
  llvm::SmallDenseSet<Value> visited;
  auto isSegmentSourceSlice = [](Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return false;
    Operation *parent = argument.getOwner()->getParentOp();
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent))
      return argument.getOwner()->getParent() == &fold.getSummarize() &&
             argument.getArgNumber() < fold.getSourceCount();
    if (auto scan = dyn_cast_or_null<RegionScanOp>(parent))
      return argument.getOwner()->getParent() == &scan.getSummarize() &&
             argument.getArgNumber() < scan.getSourceCount();
    return false;
  };
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    // A structured segment slice has one exact extent authority: the segment
    // parameter owned by its region_fold/region_scan operation.  Pointwise
    // ownership retargeting must stop at that block argument regardless of
    // which logical provenance reached the structured boundary.
    if (isSegmentSourceSlice(value))
      continue;
    // Each make_range is an independent physical traversal authority.  A
    // source-axis extent selected for one range may flow through its users,
    // but must not cross a shared consumer and rewrite a different range that
    // happens to carry the same logical provenance.
    if (value != root && value.getDefiningOp<MakeRangeOp>())
      continue;
    // A reduction or other structured operation can consume a source axis and
    // produce a scalar/record that no longer carries that physical axis.  The
    // extent decision ends there; following the scalar into a later broadcast
    // would conflate a new traversal with the one being retargeted.
    if (!carriesSourceExtent(value.getType(), sourceId, previousExtents))
      continue;
    if (!preservesIntroducedUnitSourceAxis(value, sourceId))
      value.setType(replaceSourceExtent(value.getType(), sourceId,
                                        previousExtents, extent));
    // Product fields and structured helper arguments are part of the same
    // physical value flow even though MLIR does not connect them with ordinary
    // result uses.  A blocking decision for one provenance axis must cross
    // those boundaries; otherwise an operation can retain two physical
    // schemas for one summary/carry value.
    if (auto record = value.getDefiningOp<MakeRecordOp>())
      worklist.append(record.getFields().begin(), record.getFields().end());
    if (auto extract = value.getDefiningOp<ExtractOp>())
      worklist.push_back(extract.getRecord());
    for (Operation *user : value.getUsers()) {
      if (isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp,
              ContractOp, ScaledContractOp, SparseContractOp, ReduceOp,
              ScanOp, RegionFoldOp, RegionScanOp, RandomBitsOp,
              ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
              AtomicCompareExchangeOp>(user)) {
        worklist.append(user->getOperands().begin(), user->getOperands().end());
        for (Region &region : user->getRegions()) {
          for (Block &block : region) {
            for (BlockArgument argument : block.getArguments()) {
              if (!isSegmentSourceSlice(argument))
                worklist.push_back(argument);
            }
          }
        }
      }
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
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
  StringRef name = logical.getSymbol().getValue();
  if (!name.consume_front("D"))
    return failure();
  uint64_t dimension = 0;
  return name.getAsInteger(10, dimension)
             ? FailureOr<uint64_t>(failure())
             : FailureOr<uint64_t>(dimension);
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

LogicalResult bindFullCoverageDimension(func::FuncOp kernel, uint64_t dimension,
                                        Value physicalExtent) {
  auto parameter = physicalExtent.getDefiningOp<ParameterOp>();
  if (!parameter)
    return failure();
  Value logicalExtent;
  for (BlockArgument argument : kernel.getArguments()) {
    DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == static_cast<int64_t>(dimension)) {
      logicalExtent = argument;
      break;
    }
  }
  if (!logicalExtent)
    return failure();

  PhysicalExprAttr parameterExtent = PhysicalExprAttr::get(
      kernel.getContext(),
      static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      parameter.getParameter().getName(),
      ArrayAttr::get(kernel.getContext(), {}));
  SmallVector<MakeRangeOp> ranges;
  kernel.walk([&](MakeRangeOp range) {
    auto sourceDimension =
        range->getAttrOfType<IntegerAttr>(sourceDimensionAttr);
    auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
    if (!sourceDimension ||
        sourceDimension.getInt() != static_cast<int64_t>(dimension) ||
        range->hasAttr(sourceSubregionAttr) || !fragment ||
        fragment.getShape().size() != 1 ||
        fragment.getShape()[0] != parameterExtent)
      return;
    ranges.push_back(range);
  });
  for (MakeRangeOp range : ranges)
    retargetSourceExtent(range.getResult(), range.getSourceId(), parameterExtent);
  if (ranges.empty() || llvm::all_of(ranges, [&](MakeRangeOp range) {
        return range.getExtent() == physicalExtent;
      }))
    return success();

  llvm::SmallPtrSet<Operation *, 32> rangeSet;
  for (MakeRangeOp range : ranges)
    rangeSet.insert(range.getOperation());
  auto relevantRanges = [&](ValueRange values) {
    SmallVector<MakeRangeOp> result;
    llvm::SmallPtrSet<Operation *, 32> visited;
    for (Value value : values)
      collectRanges(value, rangeSet, result, visited);
    return result;
  };

  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  llvm::DenseMap<Operation *, SmallVector<MakeRangeOp>> accessRanges;
  for (LoadOp load : loads)
    accessRanges[load.getOperation()] = relevantRanges(load.getCoordinates());
  for (StoreOp store : stores)
    accessRanges[store.getOperation()] = relevantRanges(store.getCoordinates());

  llvm::DenseMap<Operation *, Value> predicates;
  for (MakeRangeOp range : ranges) {
    OpBuilder builder(range);
    range->setOperand(1, physicalExtent);
    builder.setInsertionPointAfter(range);
    Value distance = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), logicalExtent, range.getStep(),
        /*multiply=*/2);
    Value stop = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), range.getStart(), distance,
        /*add=*/0);
    auto coordinate = cast<FragmentType>(range.getResult().getType());
    Value stopFragment =
        builder.create<BroadcastOp>(range.getLoc(), coordinate, stop);
    auto predicate = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinate.getShape(),
        coordinate.getAxisMaps(), coordinate.getValidity(),
        coordinate.getOwner());
    predicates[range.getOperation()] = builder.create<CompareOp>(
        range.getLoc(), predicate, range.getResult(), stopFragment,
        /*less-than=*/2);
  }

  auto materializeTail = [&](OpBuilder &builder, Location location,
                             FragmentType target,
                             ArrayRef<MakeRangeOp> sources) -> FailureOr<Value> {
    Value result;
    for (MakeRangeOp range : sources) {
      FailureOr<Value> current = projectPredicate(
          builder, location, predicates.lookup(range.getOperation()), target,
          range.getSourceId());
      if (failed(current))
        return failure();
      result = result ? Value(builder.create<BinaryOp>(
                            location, current->getType(), result, *current,
                            /*and=*/11))
                      : *current;
    }
    return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
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
                                       /*and=*/11);
    }
    Value fill = load.getFill();
    if (!fill)
      fill = zeroFill(builder, load.getLoc(), type);
    else if (fill.getType() != type)
      fill = builder.create<BroadcastOp>(load.getLoc(), type, fill);
    if (!fill)
      return load.emitOpError("full-coverage load has no neutral fill");
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), type, load.getResource(), load.getCoordinates(), valid,
        fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
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
                                       valid, /*and=*/11);
    }
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        store.getValue(), valid, store.getSourceAxes(), store.getCollision());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
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

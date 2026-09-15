#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool isZero(Value value) {
  PhysicalExprAttr expression = queryNonNegativeIndexUpperBound(value);
  return expression &&
         expression.getKind() ==
             static_cast<uint32_t>(PhysicalExprKind::Constant) &&
         expression.getValue() == 0;
}

} // namespace

Value createInvocationWorkspace(func::FuncOp kernel, Location location,
                                FragmentType payload, ArrayAttr shape) {
  uint64_t instance = 1;
  llvm::StringSet<> names;
  for (BlockArgument argument : kernel.getArguments()) {
    names.insert(kernel.getArgAttrDict(argument.getArgNumber())
                     .getAs<StringAttr>(abiNameAttr).getValue());
    if (auto buffer = dyn_cast<BufferType>(argument.getType()))
      instance = std::max(instance, buffer.getInstance() + 1);
  }
  kernel.walk([&](BufferOp buffer) {
    instance = std::max(instance, buffer.getResult().getType().getInstance() + 1);
  });
  std::string name = ("_workspace_" + Twine(instance)).str();
  while (!names.insert(name).second)
    name += "_";
  OpBuilder builder(kernel.getContext());
  auto type = BufferType::get(
      kernel.getContext(), payload.getElementType(), shape,
      BufferScopeAttr::get(kernel.getContext(), BufferScope::InvocationWorkspace),
      instance, payload.getOwner(),
      BufferInitializationAttr::get(kernel.getContext(), BufferInitialization::FirstWrite),
      BufferLifetimeAttr::get(kernel.getContext(), BufferLifetime::Invocation),
      /*visibility=*/1, /*workspace=*/true);
  unsigned argument = kernel.getNumArguments();
  kernel.insertArgument(
      argument, type,
      builder.getDictionaryAttr({
          builder.getNamedAttr(abiKindAttr, builder.getStringAttr("workspace")),
          builder.getNamedAttr(abiNameAttr, builder.getStringAttr(name)),
      }), location);
  return kernel.getArgument(argument);
}

namespace {

template <typename Emit>
LogicalResult buildStoreTraversal(OpBuilder &builder, Location location,
                                  ArrayRef<MakeRangeOp> ranges,
                                  ArrayRef<std::pair<MakeRangeOp, unsigned>> roots, Value rowChunk,
                                  FragmentType payload, Emit emit) {
  OpBuilder::InsertionGuard guard(builder);
  IRMapping mapping;
  SmallVector<Value> coordinates;
  SmallVector<Value> predicates;
  for (unsigned axis = 0; axis < ranges.size(); ++axis) {
    MakeRangeOp range = ranges[axis];
    Value chunk = rowChunk;
    if (axis != 0)
      chunk = builder.create<PhysicalExprOp>(
          location, builder.getIndexType(),
          cast<PhysicalExprAttr>(payload.getShape()[axis]));
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value stop = builder.create<PhysicalExprOp>(
        location, builder.getIndexType(),
        queryLaunchExpression(range.getLogicalStop()));
    Value start = zero;
    if (axis == 0) {
      auto loop = builder.create<scf::ForOp>(location, zero, stop, chunk);
      builder.setInsertionPointToStart(loop.getBody());
      start = loop.getInductionVar();
    }
    auto original = range.getResult().getType();
    auto type = FragmentType::get(
        builder.getContext(), original.getElementType(),
        builder.getArrayAttr({queryLaunchExpression(chunk)}),
        original.getAxisMaps(), original.getValidity(), original.getOwner());
    auto current = builder.create<MakeRangeOp>(
        location, type, start, chunk, one, zero, stop, range.getSourceId(),
        range.getSourceAxis(), range.getDerived());
    for (auto [root, rootAxis] : roots) {
      if (rootAxis != axis)
        continue;
      if (sameLogicalRange(root, range)) {
        mapping.map(root.getResult(), current.getResult());
        continue;
      }
      auto original = root.getResult().getType();
      auto rebound = FragmentType::get(
          builder.getContext(), original.getElementType(), type.getShape(),
          original.getAxisMaps(), original.getValidity(), original.getOwner());
      Value equivalent = builder.create<MakeRangeOp>(
          location, rebound, start, chunk, one, root.getLogicalStart(),
          root.getLogicalStop(), root.getSourceId(), root.getSourceAxis(), root.getDerived());
      mapping.map(root.getResult(), equivalent);
    }
    mapping.map(range.getResult(), current.getResult());
    coordinates.push_back(current);
    auto boolean = FragmentType::get(
        builder.getContext(), builder.getI1Type(), type.getShape(),
        type.getAxisMaps(), type.getValidity(), type.getOwner());
    Value end = builder.create<BroadcastOp>(location, type, stop);
    predicates.push_back(builder.create<CompareOp>(
        location, boolean, current, end, ComparePredicate::Lt));
  }
  auto boolean = FragmentType::get(
      builder.getContext(), builder.getI1Type(), payload.getShape(),
      payload.getAxisMaps(), payload.getValidity(), payload.getOwner());
  Value valid;
  for (Value predicate : predicates) {
    FailureOr<Value> projected =
        materializeBroadcastToFragment(builder, location, predicate, boolean);
    if (failed(projected))
      return failure();
    valid = valid ? Value(builder.create<BinaryOp>(
                        location, boolean, valid, *projected,
                        BinaryOperator::LogicalAnd))
                  : *projected;
  }
  return emit(builder, mapping, coordinates, valid);
}

FailureOr<bool> materializeRetainedStore(StoreOp store, func::FuncOp kernel) {
  auto payload = dyn_cast<FragmentType>(store.getValue().getType());
  auto output = dyn_cast<ViewType>(store.getResource().getType());
  if (store->getBlock() != &kernel.front() || !payload || !output ||
      payload.getShape().size() != 2 || output.getRank() != 2 ||
      store.getSourceAxes() != ArrayRef<int64_t>{0, 1})
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  if (!analysis.boundaryValidity(store).isExact())
    return false;
  SmallVector<MakeRangeOp> ranges;
  SmallVector<Attribute> shape;
  for (auto [axis, coordinate] : llvm::enumerate(store.getCoordinates())) {
    auto range = coordinate.getDefiningOp<MakeRangeOp>();
    auto extent = cast<PhysicalExprAttr>(payload.getShape()[axis]);
    if (!range || !isZero(range.getStart()) || !isZero(range.getLogicalStart()) ||
        !isUnitStepRange(range))
      return false;
    PhysicalExprAttr end = queryLaunchExpression(range.getLogicalStop());
    if (!end || end != output.getLayout().getExtents()[axis])
      return false;
    if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      FailureOr<ParameterOp> parameter = queryParameterBySymbol(kernel, extent.getSymbol());
      if (failed(parameter) || !(*parameter)->hasAttr(coverageDimensionAttr))
        return false;
    } else if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
               extent != end) {
      return false;
    }
    ranges.push_back(range);
    shape.push_back(end);
  }
  if (ranges.size() != 2)
    return false;
  PhysicalSourceAxis source = sourceAxisIdentity(ranges.front());
  // Preserve the read snapshot before the first clobber, then perform the
  // external write at its original position using the private saved value.
  StoreOp clobber;
  for (Operation &operation : kernel.front().without_terminator()) {
    if (&operation == store)
      break;
    auto write = dyn_cast<StoreOp>(operation);
    if (!write || !isa<ViewType>(write.getResource().getType()))
      continue;
    PhysicalReplayFact replay = analysis.replayability(
        store.getValue(), source, PhysicalReplayScope::ValueGraph,
        /*allowAccesses=*/true, write);
    if (!replay.isReplayable() || replay.accesses.empty())
      continue;
    bool blocked = false;
    bool available = llvm::all_of(replay.accesses, [&](Operation *access) {
      auto load = dyn_cast<LoadOp>(access);
      if (!load || !canReplayReadAt(load, write))
        return false;
      blocked |= !canReplayReadAt(load, store);
      return true;
    });
    if (available && blocked) {
      clobber = write;
      break;
    }
  }
  if (!clobber) {
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    __int128 footprint = std::max(1u,
        (payload.getElementType().getIntOrFloatBitWidth() + 31) / 32);
    bool fixed = true;
    for (Attribute attribute : shape) {
      auto extent = cast<PhysicalExprAttr>(attribute);
      fixed &= extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant);
      if (fixed)
        footprint *= extent.getValue();
    }
    // A launch-dependent full matrix has no proven register bound. Use the
    // same bounded traversal as an oversized static matrix; its workspace
    // still follows the original runtime shape and store order.
    if (!fixed || footprint > capabilities.getRegistersPerUnit()) {
      auto replay = analysis.replayability(store.getValue(), source,
          PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true, store);
      if (replay.isReplayable() && !replay.accesses.empty() &&
          llvm::all_of(replay.accesses, [&](Operation *access) {
            auto load = dyn_cast<LoadOp>(access);
            return load && canReplayReadAt(load, store);
          }))
        clobber = store;
    }
  }
  if (!clobber)
    return false;
  SmallVector<Value> dependencies{store.getValue()};
  llvm::DenseSet<Operation *> visited;
  for (unsigned index = 0; index < dependencies.size(); ++index) {
    Operation *producer = dependencies[index].getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (isa<ScanOp>(producer))
      return false;
    dependencies.append(producer->getOperands().begin(),
                        producer->getOperands().end());
  }
  PhysicalRangeFact facts = analysis.sourceRanges(store.getValue());
  SmallVector<MakeRangeOp> roots(facts.roots.begin(), facts.roots.end());
  roots.append(ranges.begin(), ranges.end());
  SmallVector<std::pair<MakeRangeOp, unsigned>> rootBindings;
  SmallVector<MakeRangeOp> rowRoots;
  for (MakeRangeOp root : roots) {
    std::optional<unsigned> selected;
    bool exactSource = false;
    for (auto [axis, range] : llvm::enumerate(ranges)) {
      if (!isZero(root.getStart()) ||
          !analysis.lockstepRanges({root, range}).isExact())
        continue;
      bool sameSource = sameLogicalRange(root, range);
      if (selected && sameSource == exactSource)
        return false;
      if (!selected || sameSource) {
        selected = axis;
        exactSource = sameSource;
      }
    }
    if (!selected)
      return false;
    rootBindings.emplace_back(root, *selected);
    if (*selected == 0 && !llvm::is_contained(rowRoots, root))
      rowRoots.push_back(root);
  }
  for (Operation *producer : visited)
    if (auto reduce = dyn_cast<ReduceOp>(producer))
      for (Value input : reduce.getInputs().take_front(reduce.getSourceCount())) {
        PhysicalRangeAxisFact axes = analysis.rangeAxes(input, rowRoots);
        if (!axes.isExact() ||
            llvm::any_of(axes.fragmentAxes, [&](unsigned axis) {
              return llvm::is_contained(reduce.getAxes(), axis);
            }))
          return false;
      }
  DominanceInfo dominance(kernel);
  SmallVector<std::pair<Value, TypedAttr>> constants;
  UniformValueAnalysis uniform(describeUniformValue);
  for (Value dependency : dependencies) {
    if (dominance.dominates(dependency, clobber))
      continue;
    if (!isa<FragmentType>(dependency.getType())) {
      auto constant = dyn_cast_or_null<TypedAttr>(uniform.evaluate(dependency));
      if (!constant || constant.getType() != dependency.getType())
        return false;
      constants.emplace_back(dependency, constant);
      continue;
    }
    PhysicalRangeAxisFact axes = analysis.rangeAxes(dependency, rowRoots);
    // The mapped ranges also reconstruct pure column coordinates and their
    // predicates. These do not carry a row axis; data reads still require it.
    if (axes.isExact() && axes.fragmentAxes.empty() &&
        analysis.replayability(dependency, std::nullopt,
                               PhysicalReplayScope::Coordinate,
                               /*allowAccesses=*/false).isReplayable())
      continue;
    bool scalarBroadcast = dependency.getDefiningOp<SplatOp>() != nullptr;
    if (auto broadcast = dependency.getDefiningOp<BroadcastOp>())
      scalarBroadcast |=
          !isa<FragmentType, RecordType>(broadcast.getValue().getType());
    bool projectedBroadcast = scalarBroadcast && llvm::any_of(
        rowRoots, [&](MakeRangeOp root) {
          return queryFragmentAxis(dependency.getType(), sourceAxisIdentity(root))
              .isExact();
        });
    if (!projectedBroadcast &&
        (!axes.isExact() || axes.fragmentAxes.size() != 1))
      return false;
  }

  OpBuilder builder(clobber);
  IRMapping constantValues;
  for (auto [original, constant] : constants)
    constantValues.map(original, builder.create<arith::ConstantOp>(
        store.getLoc(), original.getType(), constant));
  Value workspace = createInvocationWorkspace(
      kernel, store.getLoc(), payload, builder.getArrayAttr(shape));
  uint64_t instance = cast<BufferType>(workspace.getType()).getInstance();
  ParameterOp chunk = getOrCreatePhysicalParameter(
      kernel, ("MATERIALIZE_ROWS_" + Twine(instance)).str(),
      ParameterRole::ReductionOuter, ParameterCategory::Reduction,
      payload.getElementType().getIntOrFloatBitWidth(),
      {1, 2, 4, 8, 16, 32, 64});
  SmallVector<Attribute> blockedShape(payload.getShape().begin(),
                                      payload.getShape().end());
  blockedShape[0] = queryLaunchExpression(chunk);
  auto blocked = FragmentType::get(
      kernel.getContext(), payload.getElementType(),
      builder.getArrayAttr(blockedShape), payload.getAxisMaps(),
      payload.getValidity(), payload.getOwner());
  ReplayMaterializationOptions options;
  options.fragmentAxis = 0;
  options.traversalRanges = rowRoots;
  LogicalResult initialized = buildStoreTraversal(
      builder, store.getLoc(), ranges, rootBindings, chunk, blocked,
      [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
          Value valid) -> LogicalResult {
        for (const auto &entry : constantValues.getValueMap())
          mapping.map(entry.first, entry.second);
        FailureOr<Value> value = materializeReplayedValue(
            nested, store.getLoc(), store.getValue(), source,
            queryLaunchExpression(chunk), mapping, options);
        if (failed(value))
          return store.emitOpError("retained value has no bounded producer replay");
        nested.create<StoreOp>(store.getLoc(), workspace, coordinates, *value,
                               valid, store.getSourceAxes());
        return success();
      });
  if (failed(initialized))
    return failure();
  builder.setInsertionPoint(store);
  LogicalResult copied = buildStoreTraversal(
      builder, store.getLoc(), ranges, rootBindings, chunk, blocked,
      [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
          Value valid) -> LogicalResult {
        if (store.getValid()) {
          FailureOr<Value> predicate = materializeReplayedValue(
              nested, store.getLoc(), store.getValid(), source,
              queryLaunchExpression(chunk), mapping, options);
          if (failed(predicate))
            return store.emitOpError("retained output validity cannot be replayed");
          auto type = cast<FragmentType>(valid.getType());
          valid = nested.create<BinaryOp>(store.getLoc(), type, valid, *predicate,
                                          BinaryOperator::LogicalAnd);
        }
        FailureOr<Value> zero =
            materializeZeroFragment(nested, store.getLoc(), blocked);
        if (failed(zero))
          return failure();
        Value value = nested.create<LoadOp>(
            store.getLoc(), blocked, workspace, coordinates, valid, *zero,
            store.getSourceAxes());
        auto replacement = nested.create<StoreOp>(
            store.getLoc(), store.getResource(), coordinates, value, valid,
            store.getSourceAxes());
        replacement->setAttrs(store->getAttrs());
        return success();
      });
  if (failed(copied))
    return failure();
  store.erase();
  eraseDeadPhysicalValues(kernel);
  return true;
}

FailureOr<bool> materializeRetainedGather(GatherOp gather, func::FuncOp kernel) {
  Value source = gather.getSource();
  auto payload = dyn_cast<FragmentType>(source.getType());
  auto result = dyn_cast<FragmentType>(gather.getResult().getType());
  if (!payload || !result || payload.getShape().size() != 1 ||
      result.getShape().size() != 1 || gather.getCoordinates().size() != 1 ||
      gather.getSourceAxes() != ArrayRef<int64_t>{0})
    return false;
  auto full = cast<PhysicalExprAttr>(payload.getShape()[0]);
  auto chunkExtent = cast<PhysicalExprAttr>(result.getShape()[0]);
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (full.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) ||
      full.getValue() <= capabilities.getRegistersPerUnit() || full == chunkExtent ||
      (chunkExtent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
       chunkExtent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant)))
    return false;
  Operation *definition = source.getDefiningOp();
  if (!definition)
    return false;
  SmallVector<GatherOp> readers;
  for (Operation *user : source.getUsers()) {
    auto reader = dyn_cast<GatherOp>(user);
    if (!reader || reader.getSource() != source ||
        reader.getSourceAxes() != ArrayRef<int64_t>{0})
      return false;
    readers.push_back(reader);
  }
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, 0);
  FailureOr<MakeRangeOp> range = queryExactLogicalRange(ranges);
  if (failed(range) || ranges.roots.empty() || !isZero((*range).getStart()) ||
      !isZero((*range).getLogicalStart()) || !isUnitStepRange(*range) ||
      queryLaunchExpression((*range).getLogicalStop()) != full)
    return false;
  PhysicalSourceAxis axis = sourceAxisIdentity(*range);
  auto replay = analysis.replayability(source, axis, PhysicalReplayScope::ValueGraph,
                                      /*allowAccesses=*/true, definition);
  if (!replay.isReplayable() || llvm::any_of(replay.accesses, [&](Operation *access) {
        auto load = dyn_cast<LoadOp>(access);
        return !load || !canReplayReadAt(load, definition);
      }))
    return false;
  SmallVector<Value> dependencies{source};
  llvm::DenseSet<Operation *> visited;
  for (unsigned index = 0; index < dependencies.size(); ++index) {
    Operation *producer = dependencies[index].getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer->getNumRegions() || isa<GatherOp>(producer))
      return false;
    dependencies.append(producer->getOperands().begin(), producer->getOperands().end());
  }
  OpBuilder builder(definition);
  Value chunk;
  if (chunkExtent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
    FailureOr<ParameterOp> parameter = queryParameterBySymbol(kernel, chunkExtent.getSymbol());
    if (failed(parameter))
      return failure();
    chunk = parameter->getResult();
  } else {
    chunk = builder.create<arith::ConstantIndexOp>(gather.getLoc(), chunkExtent.getValue());
  }
  Value workspace = createInvocationWorkspace(kernel, gather.getLoc(), payload,
                                              builder.getArrayAttr({full}));
  auto blocked = FragmentType::get(kernel.getContext(), payload.getElementType(),
      builder.getArrayAttr({chunkExtent}), payload.getAxisMaps(),
      payload.getValidity(), payload.getOwner());
  ReplayMaterializationOptions options;
  options.fragmentAxis = 0;
  options.traversalRanges = ranges.roots;
  options.materializeZeroFill = true;
  SmallVector<std::pair<MakeRangeOp, unsigned>> rootBindings;
  for (MakeRangeOp root : ranges.roots)
    rootBindings.emplace_back(root, 0);
  if (failed(buildStoreTraversal(
          builder, gather.getLoc(), ArrayRef<MakeRangeOp>(*range), rootBindings,
          chunk, blocked,
          [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
              Value valid) -> LogicalResult {
            options.segmentTail = valid;
            FailureOr<Value> value = materializeReplayedValue(
                nested, gather.getLoc(), source, axis, chunkExtent, mapping, options);
            if (failed(value))
              return gather.emitOpError("retained gather source has no bounded producer replay");
            nested.create<StoreOp>(gather.getLoc(), workspace, coordinates, *value,
                                   valid, gather.getSourceAxes());
            return success();
          })))
    return failure();
  for (GatherOp reader : readers) {
    builder.setInsertionPoint(reader);
    auto load = builder.create<LoadOp>(reader.getLoc(), reader.getResult().getType(),
        workspace, reader.getCoordinates(), reader.getValid(), reader.getFill(),
        reader.getSourceAxes());
    if (Attribute origin = reader->getAttr(originAttr))
      load->setAttr(originAttr, origin);
    reader.getResult().replaceAllUsesWith(load.getResult());
    reader.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}

} // namespace

LogicalResult materializeRetainedValues(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() ==
                   static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      }))
    return success();
  while (true) {
    SmallVector<GatherOp> gathers;
    kernel.walk([&](GatherOp gather) { gathers.push_back(gather); });
    bool changed = false;
    for (GatherOp gather : gathers) {
      FailureOr<bool> result = materializeRetainedGather(gather, kernel);
      if (failed(result))
        return failure();
      if (*result) {
        changed = true;
        break;
      }
    }
    if (changed)
      continue;
    SmallVector<StoreOp> stores;
    kernel.walk([&](StoreOp store) { stores.push_back(store); });
    for (StoreOp store : stores) {
      FailureOr<bool> result = materializeRetainedStore(store, kernel);
      if (failed(result))
        return failure();
      if (*result) {
        changed = true;
        break;
      }
    }
    if (!changed)
      return success();
  }
}

} // namespace intent::gpu

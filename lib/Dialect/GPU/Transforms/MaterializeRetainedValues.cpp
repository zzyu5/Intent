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
                                  FragmentType payload, unsigned chunkAxis, Emit emit) {
  OpBuilder::InsertionGuard guard(builder);
  IRMapping mapping;
  SmallVector<Value> coordinates;
  SmallVector<Value> predicates;
  for (unsigned axis = 0; axis < ranges.size(); ++axis) {
    MakeRangeOp range = ranges[axis];
    Value chunk = rowChunk;
    if (axis != chunkAxis)
      chunk = builder.create<PhysicalExprOp>(
          location, builder.getIndexType(),
          cast<PhysicalExprAttr>(payload.getShape()[axis]));
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value stop;
    if (auto constant = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(range.getLogicalStop())))
      stop = builder.create<arith::ConstantIndexOp>(location, constant.getInt());
    else
      stop = builder.create<PhysicalExprOp>(location, builder.getIndexType(),
                                           queryLaunchExpression(range.getLogicalStop()));
    Value start = isZero(range.getStart()) ? zero : range.getStart();
    if (axis == chunkAxis) {
      auto loop = builder.create<scf::ForOp>(location, zero, stop, chunk);
      builder.setInsertionPointToStart(loop.getBody());
      start = loop.getInductionVar();
    }
    auto original = range.getResult().getType();
    auto axisMap = cast<AxisMapAttr>(payload.getAxisMaps()[axis]);
    auto type = FragmentType::get(
        builder.getContext(), original.getElementType(),
        builder.getArrayAttr({queryLaunchExpression(chunk)}),
        builder.getArrayAttr({AxisMapAttr::get(builder.getContext(),
            axisMap.getSourceId(), axisMap.getSourceAxis(), axisMap.getDimensionId(),
            0, axisMap.getDerived())}), original.getValidity(), original.getOwner());
    auto current = builder.create<MakeRangeOp>(
        location, type, start, chunk, one, zero, stop, axisMap.getSourceId(),
        axisMap.getSourceAxis(), axisMap.getDerived());
    for (auto [root, rootAxis] : roots) {
      if (rootAxis != axis)
        continue;
      if (sameLogicalRange(root, range) &&
          root.getResult().getType().getAxisMaps() == type.getAxisMaps()) {
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
    if (!mapping.contains(range.getResult()))
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
      !((payload.getShape().size() == 2 && output.getRank() == 2 &&
         store.getSourceAxes() == ArrayRef<int64_t>{0, 1}) ||
        (payload.getShape().size() == 1 && output.getRank() == 1 &&
         store.getSourceAxes() == ArrayRef<int64_t>{0})))
    return false;
  PhysicalProgramAnalysis analysis(kernel);
  bool indirect = payload.getShape().size() == 1 &&
                  !store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
  if (indirect ? !analysis.accessBounds(store).isExact()
               : !analysis.boundaryValidity(store).isExact())
    return false;
  SmallVector<Value> retained{store.getValue()};
  if (indirect) {
    retained.push_back(store.getCoordinates().front());
    if (store.getValid())
      retained.push_back(store.getValid());
    if (!llvm::all_of(retained, [&](Value value) {
          auto type = dyn_cast<FragmentType>(value.getType());
          return type && type.getShape().size() == 1 &&
                 type.getShape() == payload.getShape();
        }))
      return false;
  }
  SmallVector<MakeRangeOp> ranges;
  SmallVector<Attribute> shape;
  for (auto [axis, coordinate] : llvm::enumerate(store.getCoordinates())) {
    auto range = coordinate.getDefiningOp<MakeRangeOp>();
    if (indirect) {
      auto facts = analysis.axisRanges(store.getValue(), axis);
      if (facts.state == PhysicalFactState::Unknown || facts.roots.empty() || !facts.blockers.empty() ||
          !analysis.lockstepRanges(facts.roots).isExact())
        return false;
      range = facts.roots.front();
    }
    auto extent = cast<PhysicalExprAttr>(payload.getShape()[axis]);
    if (!range || !isZero(range.getStart()) || !isZero(range.getLogicalStart()) ||
        !isUnitStepRange(range))
      return false;
    PhysicalExprAttr end = queryLaunchExpression(range.getLogicalStop());
    if (!end || (!indirect && end != output.getLayout().getExtents()[axis]))
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
  PhysicalSourceAxis source = sourceAxisIdentity(ranges.front());
  SmallVector<std::pair<PhysicalSourceAxis, int64_t>> replayAxes;
  for (Value value : retained) {
    auto mapping = cast<AxisMapAttr>(cast<FragmentType>(value.getType()).getAxisMaps()[0]);
    std::pair axis{sourceAxisIdentity(mapping), mapping.getDimensionId()};
    if (!llvm::is_contained(replayAxes, axis))
      replayAxes.push_back(axis);
    auto facts = analysis.axisRanges(value, 0);
    for (MakeRangeOp root : facts.roots) {
      auto mapping = cast<AxisMapAttr>(root.getResult().getType().getAxisMaps()[0]);
      std::pair axis{sourceAxisIdentity(root), mapping.getDimensionId()};
      if (!llvm::is_contained(replayAxes, axis))
        replayAxes.push_back(axis);
    }
  }
  auto replayAt = [&](Operation *anchor, bool &blocked) {
    bool reads = false;
    for (Value value : retained)
      for (auto [axis, dimension] : replayAxes) {
        auto replay = analysis.replayability(value, axis,
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true, anchor, dimension);
        if (!replay.isReplayable())
          return false;
        for (Operation *access : replay.accesses) {
          auto load = dyn_cast<LoadOp>(access);
          if (!load || !canReplayReadAt(load, anchor))
            return false;
          reads = true;
          blocked |= !canReplayReadAt(load, store);
        }
      }
    return reads;
  };
  // Preserve the read snapshot before the first clobber, then perform the
  // external write at its original position using the private saved value.
  StoreOp clobber;
  for (Operation &operation : kernel.front().without_terminator()) {
    if (&operation == store)
      break;
    auto write = dyn_cast<StoreOp>(operation);
    if (!write || !isa<ViewType>(write.getResource().getType()))
      continue;
    bool blocked = false;
    bool available = replayAt(write, blocked);
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
    // A launch-dependent full tensor has no proven register bound. Use the
    // same bounded traversal as an oversized static tensor; its workspace
    // still follows the original runtime shape and store order.
    if (!fixed || footprint > capabilities.getRegistersPerUnit()) {
      bool blocked = false;
      if (replayAt(store, blocked))
        clobber = store;
    }
  }
  if (!clobber)
    return false;
  SmallVector<Value> dependencies(retained);
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
  SmallVector<MakeRangeOp> roots;
  for (Value value : retained) {
    PhysicalRangeFact facts = analysis.sourceRanges(value);
    for (MakeRangeOp root : facts.roots)
      if (!llvm::is_contained(roots, root))
        roots.push_back(root);
  }
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
  SmallVector<Value> workspaces;
  for (Value value : retained)
    workspaces.push_back(createInvocationWorkspace(kernel, store.getLoc(),
        cast<FragmentType>(value.getType()), builder.getArrayAttr(shape)));
  Value workspace = workspaces.front();
  uint64_t instance = cast<BufferType>(workspace.getType()).getInstance();
  bool linearTraversal = payload.getShape().size() == 1;
  ParameterOp chunk = getOrCreatePhysicalParameter(
      kernel, ((linearTraversal ? "MATERIALIZE_ELEMENTS_" : "MATERIALIZE_ROWS_") +
               Twine(instance)).str(),
      linearTraversal ? ParameterRole::OwnershipN : ParameterRole::ReductionOuter,
      linearTraversal ? ParameterCategory::Pointwise : ParameterCategory::Reduction,
      payload.getElementType().getIntOrFloatBitWidth(),
      linearTraversal
          ? ArrayRef<int64_t>{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
          : ArrayRef<int64_t>{1, 2, 4, 8, 16, 32, 64});
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
      builder, store.getLoc(), ranges, rootBindings, chunk, blocked, 0,
      [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
          Value valid) -> LogicalResult {
        for (const auto &entry : constantValues.getValueMap())
          mapping.map(entry.first, entry.second);
        for (auto [original, snapshot] : llvm::zip(retained, workspaces)) {
          FailureOr<Value> value = materializeReplayedValue(
              nested, store.getLoc(), original, source,
              queryLaunchExpression(chunk), mapping, options);
          if (failed(value))
            return store.emitOpError("retained value has no bounded producer replay");
          nested.create<StoreOp>(store.getLoc(), snapshot, coordinates, *value,
                                 valid, store.getSourceAxes());
        }
        return success();
      });
  if (failed(initialized))
    return failure();
  builder.setInsertionPoint(store);
  LogicalResult copied = buildStoreTraversal(
      builder, store.getLoc(), ranges, rootBindings, chunk, blocked, 0,
      [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
          Value valid) -> LogicalResult {
        if (store.getValid() && !indirect) {
          FailureOr<Value> predicate = materializeReplayedValue(
              nested, store.getLoc(), store.getValid(), source,
              queryLaunchExpression(chunk), mapping, options);
          if (failed(predicate))
            return store.emitOpError("retained output validity cannot be replayed");
          auto type = cast<FragmentType>(valid.getType());
          valid = nested.create<BinaryOp>(store.getLoc(), type, valid, *predicate,
                                          BinaryOperator::LogicalAnd);
        }
        SmallVector<Value> restored;
        for (auto [original, snapshot] : llvm::zip(retained, workspaces)) {
          auto type = cast<FragmentType>(original.getType());
          auto current = FragmentType::get(kernel.getContext(), type.getElementType(),
              blocked.getShape(), type.getAxisMaps(), type.getValidity(), type.getOwner());
          auto zero = materializeZeroFragment(nested, store.getLoc(), current);
          if (failed(zero))
            return failure();
          restored.push_back(nested.create<LoadOp>(store.getLoc(), current, snapshot,
              coordinates, valid, *zero, store.getSourceAxes()));
        }
        SmallVector<Value> destinations(coordinates);
        if (indirect) {
          destinations[0] = restored[1];
          if (store.getValid()) {
            auto predicate = materializeBroadcastToFragment(nested, store.getLoc(),
                restored[2], cast<FragmentType>(valid.getType()));
            if (failed(predicate))
              return failure();
            valid = nested.create<BinaryOp>(store.getLoc(), valid.getType(), valid,
                                            *predicate, BinaryOperator::LogicalAnd);
          }
          // The original access was in bounds. Retain that executable proof
          // after its address and predicate have become separate saved values.
          auto originalIndex = cast<FragmentType>(destinations[0].getType());
          auto indexType = FragmentType::get(kernel.getContext(), nested.getIndexType(),
              originalIndex.getShape(), originalIndex.getAxisMaps(),
              originalIndex.getValidity(), originalIndex.getOwner());
          if (originalIndex != indexType)
            destinations[0] = nested.create<CastOp>(store.getLoc(), indexType, destinations[0]);
          Value zero = nested.create<arith::ConstantIndexOp>(store.getLoc(), 0);
          Value end = nested.create<DimOp>(store.getLoc(), nested.getIndexType(),
                                          store.getResource(), store.getSourceAxes()[0]);
          for (auto [bound, predicate] :
               {std::pair{zero, ComparePredicate::Ge}, std::pair{end, ComparePredicate::Lt}}) {
            Value limit = nested.create<BroadcastOp>(store.getLoc(), indexType, bound);
            Value inBounds = nested.create<CompareOp>(store.getLoc(), valid.getType(),
                                                      destinations[0], limit, predicate);
            valid = nested.create<BinaryOp>(store.getLoc(), valid.getType(), valid,
                                            inBounds, BinaryOperator::LogicalAnd);
          }
        }
        auto replacement = nested.create<StoreOp>(
            store.getLoc(), store.getResource(), destinations, restored.front(), valid,
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
  if (!payload || payload.getShape().empty() ||
      gather.getCoordinates().size() != payload.getShape().size() ||
      !isa<FloatType, IntegerType>(payload.getElementType()))
    return false;
  unsigned rank = payload.getShape().size();
  auto coverageBound = [&](unsigned axis) -> PhysicalExprAttr {
    auto extent = cast<PhysicalExprAttr>(payload.getShape()[axis]);
    if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Parameter))
      return {};
    auto parameter = queryParameterBySymbol(kernel, extent.getSymbol());
    if (failed(parameter))
      return {};
    auto dimension = (*parameter)->getAttrOfType<IntegerAttr>(coverageDimensionAttr);
    auto mapping = cast<AxisMapAttr>(payload.getAxisMaps()[axis]);
    if (!dimension || dimension.getInt() != mapping.getDimensionId())
      return {};
    return (*parameter)->getAttrOfType<PhysicalExprAttr>(coverageBoundAttr);
  };
  unsigned chunkAxis = 0;
  int64_t largest = 0;
  bool dynamicCoverage = false;
  for (auto [axis, attribute] : llvm::enumerate(payload.getShape())) {
    auto bound = coverageBound(axis);
    if (bound && bound.getKind() == static_cast<uint32_t>(PhysicalExprKind::Dimension)) {
      chunkAxis = axis;
      dynamicCoverage = true;
      break;
    }
    auto extent = cast<PhysicalExprAttr>(attribute);
    if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        extent.getValue() > largest) {
      largest = extent.getValue();
      chunkAxis = axis;
    }
  }
  auto full = cast<PhysicalExprAttr>(payload.getShape()[chunkAxis]);
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  if (!dynamicCoverage &&
      full.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant))
    return false;
  int64_t footprint = std::max(1u,
      (payload.getElementType().getIntOrFloatBitWidth() + 31) / 32);
  for (Attribute attribute : payload.getShape()) {
    auto extent = cast<PhysicalExprAttr>(attribute);
    int64_t minimum;
    if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant)) {
      minimum = extent.getValue();
    } else if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      auto parameter = queryParameterBySymbol(kernel, extent.getSymbol());
      if (failed(parameter))
        return false;
      minimum = *llvm::min_element(parameter->getParameter().getCandidates().asArrayRef());
    } else {
      return false;
    }
    if (minimum <= 0)
      return false;
    footprint = std::min<__int128>(
        static_cast<__int128>(footprint) * minimum,
        static_cast<__int128>(capabilities.getRegistersPerUnit()) + 1);
  }
  if (!dynamicCoverage && footprint <= capabilities.getRegistersPerUnit())
    return false;
  Operation *definition = source.getDefiningOp();
  if (!definition)
    return false;
  SmallVector<GatherOp> readers;
  for (Operation *user : source.getUsers()) {
    auto reader = dyn_cast<GatherOp>(user);
    if (!reader || reader.getSource() != source ||
        reader.getCoordinates().size() != rank)
      return false;
    readers.push_back(reader);
  }
  PhysicalProgramAnalysis analysis(kernel);
  SmallVector<MakeRangeOp> ranges;
  SmallVector<std::pair<MakeRangeOp, unsigned>> rootBindings;
  SmallVector<MakeRangeOp> traversalRoots;
  SmallVector<Attribute> shape;
  SmallVector<unsigned> ownedAxes;
  for (unsigned axis = 0; axis < rank; ++axis) {
    auto facts = analysis.axisRanges(source, axis);
    if (facts.roots.empty() || !facts.blockers.empty() ||
        !analysis.lockstepRanges(facts.roots).isExact())
      return false;
    MakeRangeOp range = facts.roots.front();
    if (!isZero(range.getLogicalStart()) || !isUnitStepRange(range))
      return false;
    auto extent = queryLaunchExpression(range.getLogicalStop());
    if (auto constant = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(range.getLogicalStop())))
      extent = PhysicalExprAttr::get(kernel.getContext(),
          static_cast<uint32_t>(PhysicalExprKind::Constant), constant.getInt(),
          StringAttr::get(kernel.getContext(), ""), ArrayAttr::get(kernel.getContext(), {}));
    if (!extent ||
        (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Constant) &&
         extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Dimension)))
      return false;
    if (analysis.isProgramOwnedRange(range))
      ownedAxes.push_back(axis);
    else if (!isZero(range.getStart()) ||
             (extent != payload.getShape()[axis] && extent != coverageBound(axis)))
      return false;
    ranges.push_back(range);
    shape.push_back(extent);
    for (MakeRangeOp root : facts.roots) {
      rootBindings.emplace_back(root, axis);
      if (axis == chunkAxis)
        traversalRoots.push_back(root);
    }
  }
  if (llvm::any_of(ownedAxes, [&](unsigned axis) { return axis >= chunkAxis; }))
    return false;
  if (!result && !ownedAxes.empty())
    return false;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (ownedAxes.empty() && !llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      }))
    return false;
  auto ordinalRange = [](Value value) {
    while (value) {
      if (auto broadcast = value.getDefiningOp<BroadcastOp>()) {
        value = broadcast.getValue();
        continue;
      }
      if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
        if (llvm::any_of(reshape.getReassociation(), [](Attribute attribute) {
              auto group = cast<ReshapeGroupAttr>(attribute);
              return group.getSourceAxes().size() > 1 || group.getResultAxes().size() > 1;
            }))
          return MakeRangeOp();
        value = reshape.getValue();
        continue;
      }
      break;
    }
    return value.getDefiningOp<MakeRangeOp>();
  };
  for (GatherOp reader : readers)
    for (unsigned axis : ownedAxes) {
      auto position = llvm::find(reader.getSourceAxes(), axis);
      if (position == reader.getSourceAxes().end())
        return false;
      auto ordinal = ordinalRange(reader.getCoordinates()[position - reader.getSourceAxes().begin()]);
      if (!ordinal || !isZero(ordinal.getStart()) ||
          !isZero(ordinal.getLogicalStart()) || !isUnitStepRange(ordinal) ||
          queryLaunchExpression(ordinal.getExtent()) != queryLaunchExpression(ranges[axis].getExtent()) ||
          queryLaunchExpression(ordinal.getLogicalStop()) != queryLaunchExpression(ranges[axis].getExtent()))
        return false;
    }
  Operation *insertionAnchor = definition->getNextNode();
  if (!insertionAnchor)
    return false;
  PhysicalSourceAxis axis = sourceAxisIdentity(ranges[chunkAxis]);
  auto replay = analysis.replayability(source, axis, PhysicalReplayScope::ValueGraph,
                                      /*allowAccesses=*/true, insertionAnchor);
  if (!replay.isReplayable() || llvm::any_of(replay.accesses, [&](Operation *access) {
        auto load = dyn_cast<LoadOp>(access);
        return !load || !canReplayReadAt(load, insertionAnchor);
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
  OpBuilder builder(insertionAnchor);
  PhysicalExprAttr chunkExtent;
  Value chunk;
  bool readerChunk = rank == 1 && result && result.getShape().size() == 1;
  if (readerChunk) {
    chunkExtent = cast<PhysicalExprAttr>(result.getShape()[0]);
    if (chunkExtent == full)
      return false;
  }
  Value workspace = createInvocationWorkspace(kernel, gather.getLoc(), payload,
                                              builder.getArrayAttr(shape));
  if (!readerChunk) {
    auto instance = cast<BufferType>(workspace.getType()).getInstance();
    auto parameter = getOrCreatePhysicalParameter(kernel,
        ("MATERIALIZE_AXIS_" + Twine(instance)).str(),
        ParameterRole::ReductionOuter, ParameterCategory::Reduction,
        payload.getElementType().getIntOrFloatBitWidth(),
        {64, 128, 256, 512, 1024, 2048});
    chunkExtent = queryLaunchExpression(parameter);
  }
  if (chunkExtent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
    FailureOr<ParameterOp> parameter = queryParameterBySymbol(kernel, chunkExtent.getSymbol());
    if (failed(parameter))
      return failure();
    chunk = parameter->getResult();
  } else {
    chunk = builder.create<arith::ConstantIndexOp>(gather.getLoc(), chunkExtent.getValue());
  }
  SmallVector<Attribute> blockedShape(payload.getShape().begin(), payload.getShape().end());
  blockedShape[chunkAxis] = chunkExtent;
  auto blocked = FragmentType::get(kernel.getContext(), payload.getElementType(),
      builder.getArrayAttr(blockedShape), payload.getAxisMaps(),
      payload.getValidity(), payload.getOwner());
  ReplayMaterializationOptions options;
  options.fragmentAxis = chunkAxis;
  options.traversalRanges = traversalRoots;
  options.materializeZeroFill = true;
  SmallVector<int64_t> sourceAxes;
  for (unsigned axis = 0; axis < rank; ++axis)
    sourceAxes.push_back(axis);
  SmallVector<Value> ownedCoordinates(rank);
  if (failed(buildStoreTraversal(
          builder, gather.getLoc(), ranges, rootBindings,
          chunk, blocked, chunkAxis,
          [&](OpBuilder &nested, IRMapping &mapping, ValueRange coordinates,
              Value valid) -> LogicalResult {
            for (unsigned axis : ownedAxes)
              ownedCoordinates[axis] = coordinates[axis];
            auto coordinateType = cast<FragmentType>(coordinates[chunkAxis].getType());
            auto tailType = FragmentType::get(kernel.getContext(), nested.getI1Type(),
                coordinateType.getShape(), coordinateType.getAxisMaps(),
                coordinateType.getValidity(), coordinateType.getOwner());
            Value end = nested.create<BroadcastOp>(gather.getLoc(), coordinateType,
                                                   ranges[chunkAxis].getLogicalStop());
            options.segmentTail = nested.create<CompareOp>(gather.getLoc(), tailType,
                coordinates[chunkAxis], end, ComparePredicate::Lt);
            FailureOr<Value> value = materializeReplayedValue(
                nested, gather.getLoc(), source, axis, chunkExtent, mapping, options);
            if (failed(value))
              return gather.emitOpError("retained gather source has no bounded producer replay");
            nested.create<StoreOp>(gather.getLoc(), workspace, coordinates, *value,
                                   valid, sourceAxes);
            return success();
          })))
    return failure();
  for (GatherOp reader : readers) {
    builder.setInsertionPoint(reader);
    auto result = dyn_cast<FragmentType>(reader.getResult().getType());
    SmallVector<Value> coordinates(reader.getCoordinates());
    Value valid = reader.getValid();
    for (unsigned axis : ownedAxes) {
      auto indexType = FragmentType::get(result.getContext(), builder.getIndexType(),
          result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
      auto predicate = FragmentType::get(result.getContext(), builder.getI1Type(),
          result.getShape(), result.getAxisMaps(), result.getValidity(), result.getOwner());
      unsigned position = llvm::find(reader.getSourceAxes(), axis) - reader.getSourceAxes().begin();
      auto ordinal = ordinalRange(reader.getCoordinates()[position]);
      auto identity = builder.getArrayAttr({ReshapeGroupAttr::get(
          kernel.getContext(), builder.getDenseI64ArrayAttr({0}),
          builder.getDenseI64ArrayAttr({0}))});
      Value rebound = builder.create<ReshapeOp>(reader.getLoc(),
          ordinal.getResult().getType(), ownedCoordinates[axis], identity);
      auto projected = projectPhysicalValueToSchema(builder, reader.getLoc(), rebound, indexType);
      if (failed(projected))
        return reader.emitOpError("workspace reader lost its owned range projection");
      coordinates[position] = *projected;
      Value end = builder.create<SplatOp>(reader.getLoc(), indexType, ranges[axis].getLogicalStop());
      Value tail = builder.create<CompareOp>(reader.getLoc(), predicate, *projected, end,
                                             ComparePredicate::Lt);
      valid = valid ? Value(builder.create<BinaryOp>(reader.getLoc(), predicate, valid,
                         tail, BinaryOperator::LogicalAnd)) : tail;
    }
    Value fill = reader.getFill();
    if (valid && !fill) {
      if (result) {
        auto zero = materializeZeroFragment(builder, reader.getLoc(), result);
        if (failed(zero))
          return failure();
        fill = *zero;
      } else {
        fill = builder.create<arith::ConstantOp>(reader.getLoc(),
            builder.getZeroAttr(reader.getResult().getType()));
      }
    }
    auto load = builder.create<LoadOp>(reader.getLoc(), reader.getResult().getType(),
        workspace, coordinates, valid, fill,
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
    auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
    if (!llvm::all_of(space, [](Attribute attribute) {
          auto extent = cast<PhysicalExprAttr>(attribute);
          return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
                 extent.getValue() == 1;
        }))
      return success();
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

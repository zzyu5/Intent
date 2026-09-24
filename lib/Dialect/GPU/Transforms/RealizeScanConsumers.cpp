#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSet.h"

#include <functional>

using namespace mlir;

namespace intent::gpu {
namespace {

struct ScanConsumerMatch {
  ScanOp scan;
  scf::ForOp loop;
  Value identity;
  SmallVector<MakeRangeOp> ranges;
};

bool isScanProjection(CastOp cast) {
  if (!cast || !canPredicateValueOperation(cast))
    return false;
  auto source = dyn_cast<FragmentType>(cast.getValue().getType());
  auto result = dyn_cast<FragmentType>(cast.getType());
  return source && result && source.getShape() == result.getShape() &&
         source.getAxisMaps() == result.getAxisMaps() &&
         source.getOwner() == result.getOwner() &&
         source.getValidity() == result.getValidity();
}

std::optional<ScanConsumerMatch>
matchScanConsumer(ScanOp scan, PhysicalProgramAnalysis &analysis) {
  if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
      scan.getCaptureCount() || scan.getAxis() != 0 ||
      scan.getReverse())
    return std::nullopt;
  auto type = dyn_cast<FragmentType>(scan.getResult(0).getType());
  if (!type || type.getShape().size() != 1)
    return std::nullopt;
  ScanConsumerMatch match{scan, {}, scan.getInputs()[1], {}};
  while (true) {
    if (auto splat = match.identity.getDefiningOp<SplatOp>())
      match.identity = splat.getValue();
    else if (auto broadcast = match.identity.getDefiningOp<BroadcastOp>())
      match.identity = broadcast.getValue();
    else
      break;
  }
  if (match.identity.getType() != type.getElementType())
    return std::nullopt;
  SmallVector<Value> projections{scan.getResult(0)};
  llvm::DenseSet<Value> visitedProjections;
  for (unsigned position = 0; position < projections.size(); ++position) {
    if (!visitedProjections.insert(projections[position]).second)
      continue;
    for (Operation *user : projections[position].getUsers()) {
      if (auto cast = dyn_cast<CastOp>(user);
          isScanProjection(cast) && cast->getBlock() == scan->getBlock()) {
        projections.push_back(cast.getResult());
        continue;
      }
    auto gather = dyn_cast<GatherOp>(user);
    auto loop = user->getParentOfType<scf::ForOp>();
    if (!gather || !loop || !loop->hasAttr(independentIterationAttr) ||
        loop.getNumRegionIterArgs() || gather->getBlock() != loop.getBody() ||
        loop->getBlock() != scan->getBlock() || !scan->isBeforeInBlock(loop) ||
        (match.loop && match.loop != loop) || gather.getCoordinates().size() != 1 ||
        gather.getCoordinates().front() != loop.getInductionVar() ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0})
      return std::nullopt;
    match.loop = loop;
    }
  }
  if (!match.loop || !canPredicateScalarBlock(*match.loop.getBody()) ||
      !match.loop.getInductionVar().getType().isIndex())
    return std::nullopt;
  auto isConstant = [&](Value value, int64_t expected) {
    PhysicalExprAttr expression = queryLaunchExpression(value);
    return expression &&
           expression.getKind() ==
               static_cast<uint32_t>(PhysicalExprKind::Constant) &&
           expression.getValue() == expected;
  };
  if (!isConstant(match.loop.getLowerBound(), 0) ||
      !isConstant(match.loop.getStep(), 1))
    return std::nullopt;
  PhysicalRangeFact ranges = analysis.axisRanges(scan.getInputs()[0], 0);
  if (failed(queryExactLogicalRange(ranges)) || ranges.roots.empty() ||
      !analysis.lockstepRanges(ranges.roots).isExact())
    return std::nullopt;
  match.ranges.assign(ranges.roots.begin(), ranges.roots.end());
  for (MakeRangeOp range : ranges.roots)
    if (!isConstant(range.getStart(), 0) ||
        !isConstant(range.getLogicalStart(), 0) ||
        !isConstant(range.getStep(), 1) ||
        !samePhysicalScalarExpression(range.getLogicalStop(),
                                     match.loop.getUpperBound()))
      return std::nullopt;

  // Every replayed source read has a mandatory matching read in its independent
  // consumer iteration. Cross-iteration writes cannot alter those locations in
  // a legal parallel program; the current iteration still reads before writing.
  llvm::DenseSet<Operation *> sourceLoads;
  for (Operation *access : ranges.accesses) {
    auto load = dyn_cast<LoadOp>(access);
    auto view =
        load ? dyn_cast<ViewType>(load.getResource().getType()) : ViewType();
    if (!load || !view || view.getRank() != 1 ||
        load.getCoordinates().size() != 1 ||
        !llvm::is_contained(
            ranges.roots,
            load.getCoordinates().front().getDefiningOp<MakeRangeOp>()) ||
        !canReplayReadAt(load, match.loop) ||
        !analysis.boundaryValidity(load, /*allowRangeGuards=*/true).isExact() ||
        queryLaunchExpression(match.loop.getUpperBound()) !=
            view.getLayout().getExtents()[0])
      return std::nullopt;
    bool matched = false;
    for (Operation &operation : match.loop.getBody()->without_terminator()) {
      if (isa<StoreOp>(operation))
        break;
      auto reader = dyn_cast<LoadOp>(operation);
      if (reader && reader.getResource() == load.getResource() &&
          reader.getCoordinates() == ValueRange{match.loop.getInductionVar()} &&
          reader.getSourceAxes() == ArrayRef<int64_t>{0} &&
          analysis.boundaryValidity(reader).isExact()) {
        matched = true;
        break;
      }
    }
    bool privateWrites = llvm::all_of(
        match.loop.getBody()->without_terminator(), [](Operation &operation) {
          auto store = dyn_cast<StoreOp>(operation);
          if (!store)
            return true;
          auto buffer = dyn_cast<BufferType>(store.getResource().getType());
          return buffer && buffer.getScope().getValue() == BufferScope::InvocationWorkspace;
        });
    if (!matched && !privateWrites)
      return std::nullopt;
    sourceLoads.insert(load);
  }
  llvm::DenseSet<Value> visited;
  std::function<bool(Value)> safeSource = [&](Value value) {
    if (!visited.insert(value).second)
      return true;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      return isa<BlockArgument>(value);
    if (isa<LoadOp>(producer))
      return sourceLoads.contains(producer);
    if (isa<MakeRangeOp>(producer))
      return llvm::is_contained(ranges.roots, cast<MakeRangeOp>(producer));
    if (isa<GatherOp>(producer) || !canPredicateValueOperation(producer))
      return false;
    return llvm::all_of(producer->getOperands(), safeSource);
  };
  if (!safeSource(scan.getInputs()[0]))
    return std::nullopt;
  return match;
}

FailureOr<ParameterOp> realizeScanConsumerMatch(func::FuncOp kernel,
                                               ScanConsumerMatch &match) {
  ScanOp scan = match.scan;
  auto original = cast<FragmentType>(scan.getResult(0).getType());
  auto mapping = cast<AxisMapAttr>(original.getAxisMaps()[0]);
  PhysicalSourceAxis source = sourceAxisIdentity(mapping);
  llvm::StringSet<> names;
  kernel.walk([&](ParameterOp parameter) {
    names.insert(parameter.getParameter().getName().getValue());
  });
  std::string name = ("SCAN_CHUNK_S" + Twine(source.sourceId) + "_A" +
                      Twine(source.sourceAxis))
                         .str();
  while (names.contains(name))
    name += "_";
  Type element = original.getElementType();
  auto integer = dyn_cast<IntegerType>(element);
  auto identity = match.identity.getDefiningOp<arith::ConstantOp>();
  auto zero = identity ? dyn_cast<IntegerAttr>(identity.getValue()) : IntegerAttr();
  bool invertIntegerSum = integer && integer.getWidth() > 1 && zero &&
      zero.getValue().isZero() &&
      queryBinaryCombineKind(scan.getCombine()) == BinaryOperator::Add;
  auto chunk = getOrCreatePhysicalParameter(
      kernel, name, ParameterRole::ScanChunk, ParameterCategory::Scan,
      element.isIndex() ? 64 : element.getIntOrFloatBitWidth(),
      {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192});
  chunk->setAttr(parameterSourceAttr,
                 PhysicalSourceAttr::get(kernel.getContext(), source.sourceId,
                                         source.sourceAxis, source.derived));
  PhysicalExprAttr extent = queryLaunchExpression(chunk);
  auto fragment = [&](Type element) {
    return FragmentType::get(kernel.getContext(), element,
                             ArrayAttr::get(kernel.getContext(), {extent}),
                             original.getAxisMaps(), original.getValidity(),
                             original.getOwner());
  };
  OpBuilder builder(match.loop);
  bool failedBody = false;
  builder.create<scf::ForOp>(
      scan.getLoc(), match.loop.getLowerBound(), match.loop.getUpperBound(), chunk,
      ValueRange{match.identity},
      [&](OpBuilder &nested, Location location, Value offset, ValueRange carry) {
        auto yield = nested.create<scf::YieldOp>(location, carry);
        nested.setInsertionPoint(yield);
        Value one = nested.create<arith::ConstantIndexOp>(location, 1);
        auto range = nested.create<MakeRangeOp>(
            location, fragment(nested.getIndexType()), offset, chunk, one,
            match.loop.getLowerBound(), match.loop.getUpperBound(),
            source.sourceId, source.sourceAxis, source.derived);
        auto lift = [&](Value value) -> Value {
          auto existing = dyn_cast<FragmentType>(value.getType());
          Type element = existing ? existing.getElementType() : value.getType();
          Type target = fragment(element);
          return value.getType() == target
                     ? value
                     : Value(nested.create<BroadcastOp>(location, target, value));
        };
        Value tail = nested.create<CompareOp>(
            location, fragment(nested.getI1Type()), range,
            lift(match.loop.getUpperBound()), ComparePredicate::Lt);
        Value identity = lift(match.identity);
        IRMapping replay;
        replay.map(scan.getInputs()[1], identity);
        for (MakeRangeOp root : match.ranges)
          replay.map(root.getResult(), range.getResult());
        ReplayMaterializationOptions options;
        options.scope = PhysicalReplayScope::Coordinate;
        options.traversalRanges = match.ranges;
        options.fragmentAxis = 0;
        options.segmentTail = tail;
        options.segmentMapping = mapping;
        options.materializeZeroFill = true;
        FailureOr<Value> replayed = materializeReplayedValue(
            nested, location, scan.getInputs()[0], source, extent, replay, options);
        if (failed(replayed)) {
          failedBody = true;
          return;
        }
        Value sourceSlice = nested.create<SelectOp>(
            location, (*replayed).getType(), tail, *replayed, identity);
        replay.map(scan.getInputs()[0], sourceSlice);
        auto local = cast<ScanOp>(nested.clone(*scan, replay));
        local.setInclusive(true);
        retargetSourceExtent(local.getResult(0), source, extent);
        IRMapping combine;
        Block &body = local.getCombine().front();
        for (BlockArgument argument : body.getArguments())
          retargetSourceExtent(argument, source, extent);
        combine.map(body.getArgument(0), lift(carry.front()));
        combine.map(body.getArgument(1), local.getResult(0));
        for (Operation &operation : body.without_terminator())
          nested.clone(operation, combine);
        Value prefix = combine.lookup(
            cast<YieldOp>(body.getTerminator()).getValues().front());
        Value consumerPrefix = prefix;
        if (!scan.getInclusive() && invertIntegerSum) {
          // Integer addition/subtraction are modular. Remove this lane's input
          // exactly, including the previous chunk's carry, without a lane shift.
          consumerPrefix = nested.create<BinaryOp>(
              location, prefix.getType(), prefix, sourceSlice,
              BinaryOperator::Subtract);
        } else if (!scan.getInclusive()) {
          Value ordinal = nested.create<BinaryOp>(
              location, fragment(nested.getIndexType()), range, lift(offset),
              BinaryOperator::Subtract);
          Value shifted = nested.create<BinaryOp>(
              location, fragment(nested.getIndexType()), ordinal, lift(one),
              BinaryOperator::Subtract);
          Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
          Value lower = nested.create<CompareOp>(
              location, fragment(nested.getI1Type()), shifted, lift(zero), ComparePredicate::Ge);
          Value upper = nested.create<CompareOp>(
              location, fragment(nested.getI1Type()), shifted, lift(chunk), ComparePredicate::Lt);
          Value valid = nested.create<BinaryOp>(
              location, fragment(nested.getI1Type()), lower, upper, BinaryOperator::LogicalAnd);
          consumerPrefix = nested.create<GatherOp>(
              location, prefix.getType(), prefix, ValueRange{shifted}, valid,
              lift(carry.front()), nested.getDenseI64ArrayAttr({0}));
        }
        IRMapping consumers;
        consumers.map(match.loop.getInductionVar(), range);
        consumers.map(scan.getResult(0), consumerPrefix);
        auto mapped = [&](Value value) { return consumers.lookupOrDefault(value); };
        auto validity = [&](Value valid) -> Value {
          if (!valid)
            return tail;
          return nested.create<BinaryOp>(
              location, fragment(nested.getI1Type()), tail, lift(mapped(valid)),
              BinaryOperator::LogicalAnd);
        };
        for (Operation &operation : match.loop.getBody()->without_terminator()) {
          auto gather = dyn_cast<GatherOp>(operation);
          Value root = gather ? gather.getSource() : Value();
          SmallVector<CastOp> projections;
          while (root) {
            auto cast = root.getDefiningOp<CastOp>();
            if (!isScanProjection(cast))
              break;
            projections.push_back(cast);
            root = cast.getValue();
          }
          if (gather && root == scan.getResult(0)) {
            for (CastOp cast : llvm::reverse(projections))
              if (!consumers.lookupOrNull(cast.getResult())) {
                auto result = llvm::cast<FragmentType>(cast.getResult().getType());
                Value projected = nested.create<CastOp>(
                    cast.getLoc(), fragment(result.getElementType()), mapped(cast.getValue()));
                consumers.map(cast.getResult(), projected);
              }
            Value selectedPrefix = mapped(gather.getSource());
            Value fill =
                gather.getFill() ? lift(mapped(gather.getFill())) : Value();
            if (!fill) {
              Value zero = nested.create<arith::ConstantOp>(
                  gather.getLoc(), nested.getZeroAttr(gather.getType()));
              fill = lift(zero);
            }
            Value result = nested.create<SelectOp>(
                gather.getLoc(), selectedPrefix.getType(), validity(gather.getValid()),
                selectedPrefix, fill);
            consumers.map(gather.getResult(), result);
            continue;
          }
          clonePredicatedScalarOperation(nested, &operation, consumers, tail,
                                         fragment(nested.getIndexType()));
        }
        Value last = nested.create<BinaryOp>(location, nested.getIndexType(), chunk,
                                            one, BinaryOperator::Subtract);
        Value zero = nested.create<arith::ConstantIndexOp>(location, 0);
        Value lower = nested.create<CompareOp>(location, nested.getI1Type(), last,
                                              zero, ComparePredicate::Ge);
        Value upper = nested.create<CompareOp>(location, nested.getI1Type(), last,
                                              chunk, ComparePredicate::Lt);
        Value valid = nested.create<BinaryOp>(location, nested.getI1Type(), lower,
                                             upper, BinaryOperator::LogicalAnd);
        Value next = nested.create<GatherOp>(
            location, original.getElementType(), prefix, ValueRange{last}, valid,
            match.identity, nested.getDenseI64ArrayAttr({0}));
        yield->setOperand(0, next);
      });
  if (failedBody)
    return failure();
  match.loop.erase();
  eraseDeadPhysicalValues(kernel);
  return chunk;
}

FailureOr<bool> materializeScanSnapshot(ScanOp scan, func::FuncOp kernel,
                                       PhysicalProgramAnalysis &analysis) {
  auto type = dyn_cast<FragmentType>(scan.getResult(0).getType());
  if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
      scan.getCaptureCount() || scan.getAxis() != 0 || scan.getReverse() ||
      scan->getBlock() != &kernel.front() || !type || type.getShape().size() != 1)
    return false;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      }))
    return false;
  SmallVector<GatherOp> readers;
  SmallVector<StoreOp> stores;
  llvm::SmallPtrSet<Operation *, 8> pointwiseUsers;
  for (Operation *user : scan.getResult(0).getUsers()) {
    if (auto store = dyn_cast<StoreOp>(user);
        store && store.getValue() == scan.getResult(0) &&
        store->getBlock() == scan->getBlock() &&
        scan->isBeforeInBlock(store) && store.getCoordinates().size() == 1 &&
        store.getSourceAxes() == ArrayRef<int64_t>{0}) {
      stores.push_back(store);
      continue;
    }
    if (user->getBlock() == scan->getBlock() && scan->isBeforeInBlock(user) &&
        canPredicateValueOperation(user)) {
      pointwiseUsers.insert(user);
      continue;
    }
    auto gather = dyn_cast<GatherOp>(user);
    if (!gather || gather.getSource() != scan.getResult(0) ||
        gather.getCoordinates().size() != 1 ||
        gather.getSourceAxes() != ArrayRef<int64_t>{0})
      return false;
    readers.push_back(gather);
  }
  if (readers.empty() && stores.empty() && pointwiseUsers.empty())
    return false;
  PhysicalRangeFact ranges = analysis.axisRanges(scan.getInputs()[0], 0);
  FailureOr<MakeRangeOp> root = queryExactLogicalRange(ranges);
  if (failed(root))
    return false;
  for (MakeRangeOp range : ranges.roots) {
    auto begin = queryLaunchExpression(range.getStart());
    auto logicalBegin = queryLaunchExpression(range.getLogicalStart());
    auto zero = [](PhysicalExprAttr value) {
      return value && value.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
             value.getValue() == 0;
    };
    if (!zero(begin) || !zero(logicalBegin) || !isUnitStepRange(range))
      return false;
  }
  PhysicalExprAttr stop = queryLaunchExpression((*root).getLogicalStop());
  if (!stop)
    return false;
  auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  Type element = type.getElementType();
  unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
  if (stop.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
      static_cast<__int128>(stop.getValue()) * ((bits + 31) / 32) <=
      capabilities.getRegistersPerUnit())
    return false;
  for (StoreOp store : stores) {
    auto range = store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    if (!range || !sameLogicalRange(range, *root) ||
        !analysis.lockstepRanges({range, *root}).isExact() ||
        (store.getValid() &&
         !analysis.isTailPredicate(store.getValid(), {{range, range.getLogicalStop()}})))
      return false;
  }
  for (Operation *access : ranges.accesses) {
    auto load = dyn_cast<LoadOp>(access);
    if (!load || !canReplayReadAt(load, scan))
      return false;
  }

  OpBuilder builder(scan);
  Value workspace = createInvocationWorkspace(
      kernel, scan.getLoc(), type.getElementType(),
      builder.getArrayAttr({stop}), type.getOwner());
  builder.setInsertionPointAfter(scan);
  Value zero = builder.create<arith::ConstantIndexOp>(scan.getLoc(), 0);
  Value one = builder.create<arith::ConstantIndexOp>(scan.getLoc(), 1);
  auto copy = builder.create<scf::ForOp>(scan.getLoc(), zero,
                                        (*root).getLogicalStop(), one);
  copy->setAttr(independentIterationAttr, builder.getUnitAttr());
  builder.setInsertionPointToStart(copy.getBody());
  Value coordinate = copy.getInductionVar();
  Value value = builder.create<GatherOp>(scan.getLoc(), element, scan.getResult(0),
                                        ValueRange{coordinate}, Value(), Value(),
                                        builder.getDenseI64ArrayAttr({0}));
  builder.create<StoreOp>(scan.getLoc(), workspace, ValueRange{coordinate}, value,
                           Value(), builder.getDenseI64ArrayAttr({0}));
  if (!pointwiseUsers.empty()) {
    builder.setInsertionPointAfter(copy);
    Value coordinate = (*root).getResult();
    auto coordinateType = cast<FragmentType>(coordinate.getType());
    auto boolean = FragmentType::get(kernel.getContext(), builder.getI1Type(),
        type.getShape(), type.getAxisMaps(), type.getValidity(), type.getOwner());
    Value stop = builder.create<BroadcastOp>(scan.getLoc(), coordinateType,
                                            (*root).getLogicalStop());
    Value valid = builder.create<CompareOp>(scan.getLoc(), boolean, coordinate,
                                            stop, ComparePredicate::Lt);
    auto fill = materializeZeroFragment(builder, scan.getLoc(), type);
    if (failed(fill))
      return failure();
    Value snapshot = builder.create<LoadOp>(scan.getLoc(), type, workspace,
        ValueRange{coordinate}, valid, *fill, builder.getDenseI64ArrayAttr({0}));
    scan.getResult(0).replaceUsesWithIf(snapshot, [&](OpOperand &use) {
      return pointwiseUsers.contains(use.getOwner());
    });
  }
  for (GatherOp reader : readers) {
    builder.setInsertionPoint(reader);
    auto load = builder.create<LoadOp>(
        reader.getLoc(), reader.getResult().getType(), workspace,
        reader.getCoordinates(), reader.getValid(), reader.getFill(), reader.getSourceAxes());
    if (Attribute origin = reader->getAttr(originAttr))
      load->setAttr(originAttr, origin);
    reader.getResult().replaceAllUsesWith(load.getResult());
    reader.erase();
  }
  // Preserve the scan's immutable snapshot before any external output write.
  // Direct vector stores use the same private prefix as indexed consumers;
  // their copy is tiled after the bounded scan has been formed.
  for (StoreOp store : stores) {
    builder.setInsertionPoint(store);
    auto fill = materializeZeroFragment(builder, store.getLoc(), type);
    if (failed(fill))
      return failure();
    Value value = builder.create<LoadOp>(
        store.getLoc(), type, workspace, store.getCoordinates(),
        store.getValid(), store.getValid() ? *fill : Value(), store.getSourceAxes());
    store.getValueMutable().assign(value);
  }
  // The copy is an independent consumer of an immutable scan. Reuse the same
  // bounded prefix traversal; the original ordered consumers stay after it.
  PhysicalProgramAnalysis currentAnalysis(kernel);
  auto match = matchScanConsumer(scan, currentAnalysis);
  if (!match)
    return scan.emitOpError("scan snapshot has no legal bounded prefix traversal");
  auto chunk = realizeScanConsumerMatch(kernel, *match);
  if (failed(chunk))
    return failure();
  for (StoreOp store : stores) {
    builder.setInsertionPoint(store);
    Location location = store.getLoc();
    auto outputRange = store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    auto coordinateType = outputRange.getResult().getType();
    PhysicalExprAttr extent = queryLaunchExpression(*chunk);
    auto fragment = [&](Type element) {
      return FragmentType::get(
          kernel.getContext(), element, builder.getArrayAttr({extent}),
          coordinateType.getAxisMaps(), coordinateType.getValidity(),
          coordinateType.getOwner());
    };
    Value begin = builder.create<arith::ConstantIndexOp>(location, 0);
    Value step = builder.create<arith::ConstantIndexOp>(location, 1);
    auto loop = builder.create<scf::ForOp>(
        location, begin, outputRange.getLogicalStop(), *chunk);
    builder.setInsertionPointToStart(loop.getBody());
    Value coordinate = builder.create<MakeRangeOp>(
        location, fragment(builder.getIndexType()), loop.getInductionVar(),
        *chunk, step, begin, outputRange.getLogicalStop(),
        outputRange.getSourceId(), outputRange.getSourceAxis(), outputRange.getDerived());
    Value end = builder.create<BroadcastOp>(
        location, fragment(builder.getIndexType()), outputRange.getLogicalStop());
    Value valid = builder.create<CompareOp>(
        location, fragment(builder.getI1Type()), coordinate, end, ComparePredicate::Lt);
    auto outputType = fragment(element);
    auto fill = materializeZeroFragment(builder, location, outputType);
    if (failed(fill))
      return failure();
    Value value = builder.create<LoadOp>(
        location, outputType, workspace, ValueRange{coordinate}, valid, *fill,
        store.getSourceAxes());
    auto replacement = builder.create<StoreOp>(
        location, store.getResource(), ValueRange{coordinate}, value, valid,
        store.getSourceAxes());
    replacement->setAttrs(store->getAttrs());
    store.erase();
  }
  eraseDeadPhysicalValues(kernel);
  return true;
}
} // namespace

LogicalResult realizeScanConsumerTraversals(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  while (true) {
    PhysicalProgramAnalysis analysis(*kernel);
    std::optional<ScanConsumerMatch> match;
    kernel->walk([&](ScanOp scan) {
      match = matchScanConsumer(scan, analysis);
      return match ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (match) {
      if (failed(realizeScanConsumerMatch(*kernel, *match)))
        return failure();
      continue;
    }
    SmallVector<ScanOp> scans;
    kernel->walk([&](ScanOp scan) { scans.push_back(scan); });
    bool materialized = false;
    for (ScanOp scan : scans) {
      FailureOr<bool> result = materializeScanSnapshot(scan, *kernel, analysis);
      if (failed(result))
        return failure();
      if (*result) {
        materialized = true;
        break;
      }
    }
    if (!materialized)
      break;
  }
  return success();
}
} // namespace intent::gpu

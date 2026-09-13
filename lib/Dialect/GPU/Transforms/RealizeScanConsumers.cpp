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

std::optional<ScanConsumerMatch>
matchScanConsumer(ScanOp scan, PhysicalProgramAnalysis &analysis) {
  if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
      scan.getCaptureCount() || scan.getAxis() != 0 || !scan.getInclusive() ||
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
  for (Operation *user : scan.getResult(0).getUsers()) {
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
    if (!matched)
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

LogicalResult realizeScanConsumerMatch(func::FuncOp kernel,
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
  auto chunk = getOrCreatePhysicalParameter(
      kernel, name, ParameterRole::ScanChunk, ParameterCategory::Scan,
      element.isIndex() ? 64 : element.getIntOrFloatBitWidth(),
      {32, 64, 128, 256, 512, 1024, 2048});
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
        IRMapping consumers;
        consumers.map(match.loop.getInductionVar(), range);
        auto mapped = [&](Value value) { return consumers.lookupOrDefault(value); };
        auto validity = [&](Value valid) -> Value {
          if (!valid)
            return tail;
          return nested.create<BinaryOp>(
              location, fragment(nested.getI1Type()), tail, lift(mapped(valid)),
              BinaryOperator::LogicalAnd);
        };
        for (Operation &operation : match.loop.getBody()->without_terminator()) {
          if (auto gather = dyn_cast<GatherOp>(operation);
              gather && gather.getSource() == scan.getResult(0)) {
            Value fill =
                gather.getFill() ? lift(mapped(gather.getFill())) : Value();
            if (!fill) {
              Value zero = nested.create<arith::ConstantOp>(
                  gather.getLoc(), nested.getZeroAttr(gather.getType()));
              fill = lift(zero);
            }
            Value result = nested.create<SelectOp>(
                gather.getLoc(), prefix.getType(), validity(gather.getValid()),
                prefix, fill);
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
  scan.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
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
    if (!match)
      break;
    if (failed(realizeScanConsumerMatch(*kernel, *match)))
      return failure();
  }
  return success();
}
} // namespace intent::gpu

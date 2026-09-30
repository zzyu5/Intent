#include "Intent/Dialect/GPU/Transforms/Traversal.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;

namespace intent::gpu {

PhysicalExprAttr boundedTraversalChunk(ParameterOp chunk, MakeRangeOp range) {
  auto expression = [&](PhysicalExprKind kind, int64_t value = 0,
                        StringRef symbol = {}, ArrayRef<Attribute> operands = {}) {
    return PhysicalExprAttr::get(chunk.getContext(), static_cast<uint32_t>(kind),
                                 value, StringAttr::get(chunk.getContext(), symbol),
                                 ArrayAttr::get(chunk.getContext(), operands));
  };
  auto extent = expression(PhysicalExprKind::Parameter, 0,
                           chunk.getParameter().getName().getValue());
  auto capacity = queryLogicalRangeCapacity(range);
  if (!capacity || !isShapeBound(capacity) || chunk->hasAttr(coverageDimensionAttr))
    return extent;
  int64_t maximum =
      *llvm::max_element(chunk.getParameter().getCandidates().asArrayRef());
  if (maximum > (int64_t{1} << 62))
    return extent;
  if (auto bound = constantPhysicalExpression(capacity);
      bound && *bound >= maximum)
    return extent;
  auto positive = expression(PhysicalExprKind::Maximum, 0, {},
                             {capacity, expression(PhysicalExprKind::Constant, 1)});
  auto bounded = expression(PhysicalExprKind::Minimum, 0, {},
                            {positive, expression(PhysicalExprKind::Constant, maximum)});
  auto padded = expression(PhysicalExprKind::NextPowerOfTwo, 0, {}, {bounded});
  return expression(PhysicalExprKind::Minimum, 0, {}, {extent, padded});
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
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact ranges = analysis.axisRanges(source, fragmentAxis);
  if (ranges.roots.empty())
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps()))
      if (cast<AxisMapAttr>(attribute).getDimensionId() == dimension &&
          fragment.getShape()[axis] != currentExtent)
        return emitError(source.getLoc(),
                         "full coverage has no range authority for distinct physical occurrences");
  int64_t coverageDimension = dimension;
  bool subregion = llvm::any_of(ranges.roots, [](MakeRangeOp range) {
    return range->hasAttr(sourceSubregionAttr);
  });
  PhysicalExprAttr rangeCapacity;
  if (subregion) {
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(ranges);
    if (failed(authority))
      return kernel.emitError(
          "subregion full coverage has no exact logical range authority");
    rangeCapacity = queryLogicalRangeCapacity(*authority);
    std::optional<int64_t> parent;
    for (MakeRangeOp range : ranges.roots) {
      auto bound = range->getAttrOfType<IntegerAttr>(sourceSubregionAttr);
      if (!bound || bound.getInt() <= 0 ||
          (parent && *parent != bound.getInt()) ||
          (!rangeCapacity &&
           !queryNonNegativeIndexUpperBound(range.getLogicalStart())) ||
          !isUnitStepRange(range))
        return range.emitOpError(
            "subregion full coverage has no proven nonnegative parent-bounded traversal");
      parent = bound.getInt();
    }
    if (!rangeCapacity)
      coverageDimension = *parent;
  }

  Value runtimeDimension;
  bool coverageIsRangeCapacity = static_cast<bool>(rangeCapacity);
  if (rangeCapacity) {
    OpBuilder builder(&kernel.front(), kernel.front().begin());
    runtimeDimension = builder.create<PhysicalExprOp>(
        source.getLoc(), builder.getIndexType(), rangeCapacity);
  }
  for (BlockArgument argument : kernel.getArguments()) {
    if (coverageIsRangeCapacity)
      break;
    DictionaryAttr attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == coverageDimension) {
      if (runtimeDimension && runtimeDimension != argument)
        return kernel.emitError(
            "logical dimension has multiple runtime ABI authorities");
      runtimeDimension = argument;
    }
  }

  std::optional<int64_t> staticDimension;
  for (BlockArgument argument : kernel.getArguments()) {
    if (coverageIsRangeCapacity)
      break;
    auto view = dyn_cast<ViewType>(argument.getType());
    if (!view)
      continue;
    for (auto [axis, identity] :
         llvm::enumerate(view.getLayout().getDimensionIds().asArrayRef())) {
      if (identity != coverageDimension)
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
      currentExtent.getValue() == *staticDimension &&
      llvm::isPowerOf2_64(*staticDimension))
    return success();
  if (!runtimeDimension) {
    FailureOr<MakeRangeOp> authority = queryExactLogicalRange(ranges);
    PhysicalExprAttr capacity = succeeded(authority)
                                   ? queryLogicalRangeCapacity(*authority)
                                   : PhysicalExprAttr();
    if (capacity) {
      OpBuilder builder(&kernel.front(), kernel.front().begin());
      runtimeDimension = builder.create<PhysicalExprOp>(
          source.getLoc(), builder.getIndexType(), capacity);
      coverageIsRangeCapacity = true;
    }
  }
  if (!runtimeDimension)
    return emitError(source.getLoc(),
                     "full-coverage physicalization has no runtime or static dimension authority")
           << "; dimension=" << dimension << "; source=" << source;
  PhysicalExprAttr coverageBound = queryLaunchExpression(runtimeDimension);
  if (!coverageBound)
    return kernel.emitError(
        "full-coverage physicalization has no launch-visible extent expression");
  // A range capacity already bounds its member count. Unlike a parent extent,
  // it is not an absolute coordinate against which to compare logicalStop.
  if (subregion && !coverageIsRangeCapacity)
    for (MakeRangeOp range : ranges.roots) {
      auto end = queryNonNegativeIndexUpperBound(range.getLogicalStop());
      auto endConstant = end ? constantPhysicalExpression(end) : std::nullopt;
      auto coverageConstant = constantPhysicalExpression(coverageBound);
      bool bounded = end &&
          (end == coverageBound ||
           (endConstant && coverageConstant && *endConstant <= *coverageConstant));
      if (!samePhysicalScalarExpression(range.getLogicalStop(), runtimeDimension) &&
          !bounded)
        return range.emitOpError(
            "subregion full coverage has no proven stop bound");
    }

  if (!staticDimension)
    if (auto value = dyn_cast_or_null<IntegerAttr>(
            UniformValueAnalysis(describeUniformValue).evaluate(runtimeDimension)))
      staticDimension = value.getInt();
  if (!staticDimension && coverageBound.getKind() ==
                              static_cast<uint32_t>(PhysicalExprKind::Constant))
    staticDimension = coverageBound.getValue();
  if (staticDimension && *staticDimension >= 0) {
    uint64_t size = llvm::PowerOf2Ceil(
        static_cast<uint64_t>(std::max<int64_t>(*staticDimension, 1)));
    if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return kernel.emitError("full-coverage extent exceeds the index range");
    auto covered = PhysicalExprAttr::get(kernel.getContext(),
        static_cast<uint32_t>(PhysicalExprKind::Constant), size,
        StringAttr::get(kernel.getContext(), ""), ArrayAttr::get(kernel.getContext(), {}));
    if (currentExtent == covered &&
        analysis.axisRealization(source, fragmentAxis).physicalized)
      return success();
    OpBuilder builder(&kernel.front(), kernel.front().begin());
    Value physicalExtent = builder.create<arith::ConstantIndexOp>(source.getLoc(), size);
    llvm::SmallDenseSet<int64_t> rangeDimensions;
    for (MakeRangeOp range : ranges.roots) {
      FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
      if (failed(rangeDimension))
        return range.emitOpError("full coverage has no range dimension authority");
      rangeDimensions.insert(*rangeDimension);
      retargetSourceExtent(range.getResult(), sourceAxisIdentity(range), covered,
                           *rangeDimension);
    }
    if (ranges.roots.empty())
      retargetDimensionExtent(source, dimension, covered);
    if (rangeDimensions.empty())
      rangeDimensions.insert(dimension);
    for (int64_t rangeDimension : rangeDimensions)
      if (failed(bindFullCoverageDimension(kernel, rangeDimension, physicalExtent)))
        return failure();
    return success();
  }

  ParameterOp parameter;
  if (currentExtent.getKind() ==
      static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
    FailureOr<ParameterOp> declaration =
        queryParameterBySymbol(kernel, currentExtent.getSymbol());
    if (succeeded(declaration)) {
      auto covered = (*declaration)->getAttrOfType<IntegerAttr>(
          coverageDimensionAttr);
      if (covered && covered.getInt() == coverageDimension &&
          (*declaration).getParameter().getRole() ==
              static_cast<uint32_t>(ParameterRole::FullCoverage))
        parameter = *declaration;
    }
  }
  if (!parameter) {
    kernel.walk([&](ParameterOp candidate) {
      auto covered = candidate->getAttrOfType<IntegerAttr>(
          coverageDimensionAttr);
      if (!covered || covered.getInt() != coverageDimension ||
          candidate.getParameter().getRole() !=
              static_cast<uint32_t>(ParameterRole::FullCoverage))
        return;
      if (parameter && parameter != candidate) {
        parameter = ParameterOp();
        return;
      }
      parameter = candidate;
    });
  }
  if (parameter &&
      currentExtent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Parameter) &&
      currentExtent.getSymbol() == parameter.getParameter().getName() &&
      parameter.getParameter().getRole() ==
          static_cast<uint32_t>(ParameterRole::FullCoverage)) {
    parameter->setAttr(coverageBoundAttr, coverageBound);
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.dimension ||
        *binding.dimension != coverageDimension)
      return parameter.emitOpError(
          "full-coverage parameter lost its typed dimension authority");
    if (failed(bindFullCoverageDimension(kernel, dimension,
                                         parameter.getResult())))
      return kernel.emitError(
          "existing full-coverage decision could not preserve access validity");
    return success();
  }
  static constexpr int64_t candidates[] = {
      1,    2,    4,     8,     16,    32,    64,    128,   256,
      512,  1024, 2048,  4096,  8192,  16384, 32768, 65536};
  if (!parameter) {
    std::string name = ("FULL_D" + Twine(coverageDimension)).str();
    bool nameCollision = false;
    kernel.walk([&](ParameterOp candidate) {
      nameCollision |=
          candidate.getParameter().getName().getValue() == name;
    });
    if (nameCollision) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "full-coverage parameter name is already owned by another decision");
      kernel.walk([&](ParameterOp candidate) {
        if (candidate.getParameter().getName().getValue() != name)
          return;
        diagnostic << "; role=" << candidate.getParameter().getRole();
        if (auto covered = candidate->getAttrOfType<IntegerAttr>(
                coverageDimensionAttr))
          diagnostic << ", coverage_dimension=" << covered.getInt();
        if (auto bound = candidate->getAttrOfType<IntegerAttr>(dimensionAttr))
          diagnostic << ", dimension=" << bound.getInt();
      });
      return failure();
    }
    OpBuilder builder(&kernel.getBody().front(),
                      kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(name),
        static_cast<uint32_t>(ParameterRole::FullCoverage),
        static_cast<uint32_t>(ParameterCategory::Coverage),
        /*elementBitWidth=*/0,
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    parameter = builder.create<ParameterOp>(source.getLoc(),
                                            builder.getIndexType(), schema);
  } else {
    ParameterAttr schema = parameter.getParameter();
    parameter->setAttr(
        "parameter",
        ParameterAttr::get(
            kernel.getContext(), schema.getName(), schema.getRole(),
            schema.getCategory(), schema.getElementBitWidth(),
            DenseI64ArrayAttr::get(kernel.getContext(), candidates)));
  }
  parameter->setAttr(
      dimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), coverageDimension));
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), coverageDimension));
  parameter->setAttr(coverageBoundAttr, coverageBound);

  auto covered = PhysicalExprAttr::get(
      kernel.getContext(),
      static_cast<uint32_t>(PhysicalExprKind::Parameter), 0,
      parameter.getParameter().getName(),
      ArrayAttr::get(kernel.getContext(), {}));
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
      for (Operation *blocker : replay.blockers)
        diagnostic << "; replay_blocker=" << blocker->getName();
      return failure();
    }
  }
  for (MakeRangeOp range : ranges.roots) {
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    if (failed(rangeDimension) || *rangeDimension != dimension)
      return range.emitOpError(
          "full-coverage range does not cover the selected logical dimension");
    retargetSourceExtent(range.getResult(), sourceAxisIdentity(range), covered,
                         *rangeDimension);
  }
  if (ranges.roots.empty())
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
  PhysicalExprAttr parameterExtent = queryLaunchExpression(physicalExtent);
  if (!parameterExtent ||
      (!parameter && parameterExtent.getKind() !=
                         static_cast<uint32_t>(PhysicalExprKind::Constant)))
    return failure();
  std::function<bool(PhysicalExprAttr)> hasBlockedExtent =
      [&](PhysicalExprAttr extent) {
    if (extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Parameter)) {
      auto declaration = queryParameterBySymbol(kernel, extent.getSymbol());
      return succeeded(declaration) && *declaration != parameter &&
             declaration->getParameter().getRole() !=
                 static_cast<uint32_t>(ParameterRole::FullCoverage);
    }
    return llvm::any_of(extent.getOperands(), [&](Attribute operand) {
      return hasBlockedExtent(cast<PhysicalExprAttr>(operand));
    });
  };
  SmallVector<MakeRangeOp> ranges;
  kernel.walk([&](MakeRangeOp range) {
    FailureOr<int64_t> sourceDimension = querySourceDimension(
        range.getResult().getType(), sourceAxisIdentity(range));
    auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension) ||
        !fragment ||
        fragment.getShape().size() != 1)
      return;
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    if (hasBlockedExtent(extent))
      return;
    ranges.push_back(range);
  });
  // Validity can carry an independent occurrence of the same coordinates.
  SmallVector<MakeRangeOp> authorities(ranges.begin(), ranges.end());
  kernel.walk([&](MakeRangeOp range) {
    if (llvm::is_contained(ranges, range) ||
        range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>())
      return;
    auto sourceDimension = queryRangeDimension(range);
    if (failed(sourceDimension) ||
        *sourceDimension != static_cast<int64_t>(dimension))
      return;
    if (llvm::any_of(authorities, [&](MakeRangeOp authority) {
          return sameLogicalRange(range, authority) &&
                 samePhysicalScalarExpression(range.getStart(),
                                              authority.getStart());
        }))
      ranges.push_back(range);
  });
  bool alreadyBound = !ranges.empty() &&
                      llvm::all_of(ranges, [&](MakeRangeOp range) {
                        return range.getExtent() == physicalExtent;
                      });
  for (MakeRangeOp range : ranges)
    retargetSourceExtent(range.getResult(), sourceAxisIdentity(range),
                         parameterExtent, static_cast<int64_t>(dimension));
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
    auto coordinate = cast<FragmentType>(range.getResult().getType());
    Value member = range.getResult();
    Value stop;
    if (isUnitStepRange(range) &&
        (!queryNonNegativeIndexUpperBound(range.getLogicalStart()) ||
         !queryNonNegativeIndexUpperBound(range.getLogicalStop()))) {
      Value nonempty = builder.create<CompareOp>(
          range.getLoc(), builder.getI1Type(), range.getLogicalStart(),
          range.getLogicalStop(), ComparePredicate::Lt);
      Value zero = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
      stop = builder.create<SelectOp>(range.getLoc(), builder.getIndexType(),
                                       nonempty, logicalDistance, zero);
      Value base = builder.create<BroadcastOp>(range.getLoc(), coordinate,
                                               range.getStart());
      // Unit-step offsets remain [0, extent) even when an inactive padded
      // absolute coordinate wraps. Preserve the range's physical origin.
      member = builder.create<BinaryOp>(range.getLoc(), coordinate, member,
                                         base, BinaryOperator::Subtract);
    } else {
      stop = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getStart(), logicalDistance,
          BinaryOperator::Add);
    }
    Value stopFragment =
        builder.create<BroadcastOp>(range.getLoc(), coordinate, stop);
    auto predicate = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), coordinate.getShape(),
        coordinate.getAxisMaps(), coordinate.getValidity(),
        coordinate.getOwner());
    auto tail = builder.create<CompareOp>(range.getLoc(), predicate,
                                          member, stopFragment,
                                          ComparePredicate::Lt);
    tail->setAttr(physicalTailAttr, builder.getUnitAttr());
    predicates[range.getOperation()] = tail.getResult();
  }

  auto materializeTail = [&](OpBuilder &builder, Location location,
                             FragmentType target,
                             ArrayRef<MakeRangeOp> sources,
                             Value data = Value()) -> FailureOr<Value> {
    Value result;
    for (MakeRangeOp range : sources) {
      SmallVector<unsigned, 2> axes;
      for (PhysicalAxisProjection projection : queryRangeProjections(target, range))
        axes.push_back(projection.fragmentAxis);
      if (axes.empty() && data) {
        PhysicalRangeAxisFact relation =
            PhysicalProgramAnalysis(kernel).rangeAxes(data, {range});
        if (relation.isExact())
          axes.append(relation.fragmentAxes.begin(), relation.fragmentAxes.end());
      }
      if (axes.empty())
        return failure();
      for (unsigned axis : axes) {
        FailureOr<Value> current = projectPredicateToFragmentAxis(
            builder, location, predicates.lookup(range.getOperation()), target,
            axis);
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
    if (existing.getType() != predicate) {
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, location, existing, predicate);
      if (failed(projected))
        return failure();
      existing = *projected;
    }
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
        materializeTail(builder, load.getLoc(), type, sources, load.getResult());
    if (failed(tail))
      return load.emitOpError(
          "cannot project full-coverage dimension to load validity")
             << "; dimension=" << dimension << "; result=" << type
             << "; range_count=" << sources.size();
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
      if (existing.getType() != predicate) {
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, load.getLoc(), existing, predicate);
        if (failed(projected))
          return load.emitOpError(
              "full-coverage validity has no exact physical projection")
                 << "; existing=" << existing.getType()
                 << "; required=" << predicate;
        existing = *projected;
      }
      valid = builder.create<BinaryOp>(load.getLoc(), predicate, existing, valid,
                                       BinaryOperator::LogicalAnd);
    }
    Value fill = load.getFill();
    if (!fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, load.getLoc(), type);
      if (failed(zero))
        return load.emitOpError("full-coverage load has no neutral fill");
      fill = *zero;
    } else if (fill.getType() != type) {
      FailureOr<Value> projected =
          projectPhysicalValueToSchema(builder, load.getLoc(), fill, type);
      if (failed(projected))
        return load.emitOpError(
            "full-coverage fill has no exact physical projection");
      fill = *projected;
    }
    // Callers may be realizing coverage for this exact SSA result.  Updating
    // its access operands keeps that value live across physicalization.
    load.getValidMutable().assign(ValueRange{valid});
    load.getFillMutable().assign(ValueRange{fill});
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
      FailureOr<Value> zero = materializeZeroFragment(builder, gather.getLoc(), type);
      if (failed(zero))
        return gather.emitOpError("full-coverage gather has no neutral fill");
      fill = *zero;
    } else if (fill.getType() != type) {
      FailureOr<Value> projected =
          projectPhysicalValueToSchema(builder, gather.getLoc(), fill, type);
      if (failed(projected))
        return gather.emitOpError(
            "full-coverage fill has no exact physical projection");
      fill = *projected;
    }
    gather.getValidMutable().assign(ValueRange{*valid});
    gather.getFillMutable().assign(ValueRange{fill});
  }

  for (StoreOp store : stores) {
    auto sources = accessRanges.lookup(store.getOperation());
    auto type = dyn_cast<FragmentType>(store.getValue().getType());
    if (!store->getBlock() || !type || sources.empty())
      continue;
    OpBuilder builder(store);
    SmallVector<Attribute> shape(type.getShape().begin(), type.getShape().end());
    for (MakeRangeOp range : sources) {
      auto coordinate = cast<FragmentType>(range.getResult().getType());
      for (PhysicalAxisProjection projection : queryRangeProjections(type, range))
        shape[projection.fragmentAxis] = coordinate.getShape()[0];
    }
    type = FragmentType::get(kernel.getContext(), type.getElementType(),
                             builder.getArrayAttr(shape), type.getAxisMaps(),
                             type.getValidity(), type.getOwner());
    // The access coordinates own the physical extent.  Project a uniform
    // payload to that schema without retargeting its independent value graph.
    FailureOr<Value> value = projectPhysicalValueToSchema(
        builder, store.getLoc(), store.getValue(), type);
    if (failed(value))
      return store.emitOpError(
          "full-coverage stored value has no exact physical projection");
    FailureOr<Value> tail =
        materializeTail(builder, store.getLoc(), type, sources);
    if (failed(tail))
      return store.emitOpError(
          "cannot project full-coverage dimension to store validity");
    Value valid = *tail;
    auto predicate = cast<FragmentType>(valid.getType());
    if (store.getValid()) {
      Value existing = store.getValid();
      if (existing.getType() != predicate) {
        FailureOr<Value> projected = projectPhysicalValueToSchema(
            builder, store.getLoc(), existing, predicate);
        if (failed(projected))
          return store.emitOpError(
              "full-coverage validity has no exact physical projection")
                 << "; existing=" << existing.getType()
                 << "; required=" << predicate;
        existing = *projected;
      }
      valid = builder.create<BinaryOp>(store.getLoc(), predicate, existing,
                                       valid, BinaryOperator::LogicalAnd);
    }
    store.getValueMutable().assign(*value);
    store.getValidMutable().assign(ValueRange{valid});
  }

  auto updateValidity = [&](auto access, Type payload,
                            StringRef kind) -> LogicalResult {
    auto sources = accessRanges.lookup(access.getOperation());
    auto type = dyn_cast<FragmentType>(payload);
    if (!access->getBlock() || !type || sources.empty())
      return success();
    OpBuilder builder(access);
    auto valid = combineValidity(builder, access.getLoc(), type, sources,
                                 access.getValid());
    if (failed(valid))
      return access.emitOpError()
             << "cannot project full-coverage dimension to " << kind << " validity";
    // The operand-segment interface updates only validity. Atomic ordering,
    // sharing, returned old values and scatter combine regions stay attached.
    access.getValidMutable().assign(ValueRange{*valid});
    return success();
  };
  for (ScatterReduceOp scatter : scatters)
    if (failed(updateValidity(scatter, scatter.getValue().getType(), "scatter")))
      return failure();
  for (AtomicLoadOp atomic : atomicLoads)
    if (failed(updateValidity(atomic, atomic.getResult().getType(), "atomic-load")))
      return failure();
  for (AtomicStoreOp atomic : atomicStores)
    if (failed(updateValidity(atomic, atomic.getValue().getType(), "atomic-store")))
      return failure();
  for (AtomicRMWOp atomic : atomicRMWs)
    if (failed(updateValidity(atomic, atomic.getValue().getType(), "atomic-RMW")))
      return failure();
  for (AtomicCompareExchangeOp atomic : atomicCAS)
    if (failed(updateValidity(atomic, atomic.getExpected().getType(), "compare-exchange")))
      return failure();
  return success();
}

} // namespace intent::gpu

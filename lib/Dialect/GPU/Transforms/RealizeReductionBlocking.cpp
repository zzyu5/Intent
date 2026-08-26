#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {},
                            ArrayRef<Attribute> operands = {}) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(kind), value,
      StringAttr::get(context, symbol), ArrayAttr::get(context, operands));
}

PhysicalExprAttr nextPowerOfTwo(PhysicalExprAttr source) {
  if (source.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant)) {
    uint64_t value = std::max<int64_t>(source.getValue(), 1);
    uint64_t result = 1;
    while (result < value)
      result <<= 1;
    return expression(source.getContext(), PhysicalExprKind::Constant, result);
  }
  return expression(source.getContext(), PhysicalExprKind::NextPowerOfTwo, 0,
                    {}, {source});
}

bool isCompileTimeExtent(PhysicalExprAttr expression) {
  auto kind = static_cast<PhysicalExprKind>(expression.getKind());
  if (kind == PhysicalExprKind::Constant || kind == PhysicalExprKind::Parameter)
    return true;
  if (kind == PhysicalExprKind::Dimension ||
      kind == PhysicalExprKind::ScalarABI)
    return false;
  return llvm::all_of(expression.getOperands(), [](Attribute operand) {
    return isCompileTimeExtent(cast<PhysicalExprAttr>(operand));
  });
}

bool requiresPhysicalRealization(ReduceOp reduce) {
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    if (llvm::any_of(fragment.getShape(), [](Attribute extent) {
          return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
        }))
      return true;
  }
  return false;
}

FailureOr<AxisMapAttr> axisMap(FragmentType fragment, unsigned axis) {
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getFragmentAxis() == axis)
      return mapping;
  }
  return failure();
}

FailureOr<unsigned> coordinateForSource(ValueRange coordinates,
                                        uint64_t sourceId) {
  std::optional<unsigned> result;
  for (auto [index, coordinate] : llvm::enumerate(coordinates)) {
    auto fragment = dyn_cast<FragmentType>(coordinate.getType());
    if (!fragment)
      continue;
    bool matches = llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      return cast<AxisMapAttr>(attribute).getSourceId() == sourceId;
    });
    if (!matches)
      continue;
    if (result)
      return failure();
    result = index;
  }
  return result ? FailureOr<unsigned>(*result) : FailureOr<unsigned>(failure());
}

MakeRangeOp sourceRange(Value value) {
  while (auto broadcast = value.getDefiningOp<BroadcastOp>())
    value = broadcast.getValue();
  return value.getDefiningOp<MakeRangeOp>();
}

FailureOr<Value> scalarSource(Value value) {
  if (!value)
    return failure();
  if (!isa<FragmentType>(value.getType()))
    return value;
  if (auto broadcast = value.getDefiningOp<BroadcastOp>())
    if (!isa<FragmentType>(broadcast.getValue().getType()))
      return broadcast.getValue();
  if (auto splat = value.getDefiningOp<SplatOp>())
    return splat.getValue();
  return failure();
}

bool sameScalarValue(Value lhs, Value rhs) {
  FailureOr<Value> left = scalarSource(lhs);
  FailureOr<Value> right = scalarSource(rhs);
  if (failed(left) || failed(right))
    return false;
  if (*left == *right)
    return true;
  auto leftConstant = (*left).getDefiningOp<arith::ConstantOp>();
  auto rightConstant = (*right).getDefiningOp<arith::ConstantOp>();
  return leftConstant && rightConstant &&
         leftConstant.getValue() == rightConstant.getValue();
}

FragmentType replaceExtent(FragmentType source, unsigned axis,
                           PhysicalExprAttr extent, Type element = {}) {
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[axis] = extent;
  return FragmentType::get(source.getContext(),
                           element ? element : source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), 2, source.getOwner());
}

FailureOr<unsigned> axisForSource(FragmentType fragment, uint64_t sourceId) {
  std::optional<unsigned> result;
  for (Attribute attribute : fragment.getAxisMaps()) {
    auto mapping = cast<AxisMapAttr>(attribute);
    if (mapping.getSourceId() != sourceId)
      continue;
    if (result)
      return failure();
    result = mapping.getFragmentAxis();
  }
  return result ? FailureOr<unsigned>(*result) : FailureOr<unsigned>(failure());
}

bool isReplayableProducer(Operation *operation) {
  return isa<CastOp, UnaryOp, BinaryOp, BroadcastOp, SelectOp, CompareOp,
             SplatOp>(operation);
}

struct SourcePlan {
  Value source;
  uint64_t sourceId;
  unsigned reductionAxis;
  SmallVector<LoadOp> roots;
};

struct RootAccess {
  LoadOp load;
  MakeRangeOp range;
  unsigned coordinateIndex;
  unsigned fragmentAxis;
};

bool isUnitStep(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value() == 1;
  if (auto bound = value.getDefiningOp<RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<RangeOp>();
    return range && bound.getBound() == 2 && isUnitStep(range.getStep());
  }
  return false;
}

FailureOr<RootAccess> analyzeRoot(LoadOp load, uint64_t sourceId) {
  auto fragment = dyn_cast<FragmentType>(load.getResult().getType());
  if (!fragment)
    return load.emitOpError("reduction producer root is not a fragment load");
  FailureOr<unsigned> fragmentAxis = axisForSource(fragment, sourceId);
  FailureOr<unsigned> coordinate =
      coordinateForSource(load.getCoordinates(), sourceId);
  if (failed(fragmentAxis))
    return load.emitOpError("reduction source provenance is absent from its root load");
  if (failed(coordinate))
    return load.emitOpError("reduction source provenance is absent from load coordinates");
  auto range = sourceRange(load.getCoordinates()[*coordinate]);
  if (!range)
    return load.emitOpError("reduction load coordinate is not a physical range");
  if (!isUnitStep(range.getStep()))
    return load.emitOpError("reduction load range is not unit-step");
  return RootAccess{load, range, *coordinate, *fragmentAxis};
}

LogicalResult collectLoadRoots(Value value, uint64_t sourceId,
                               SmallVectorImpl<LoadOp> &roots,
                               llvm::SmallPtrSetImpl<Operation *> &visited) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  if (!fragment || failed(axisForSource(fragment, sourceId)))
    return success();
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return failure();
  if (auto load = dyn_cast<LoadOp>(producer)) {
    if (!llvm::is_contained(roots, load))
      roots.push_back(load);
    return success();
  }
  if (!isReplayableProducer(producer) || producer->getNumRegions() != 0 ||
      producer->getNumResults() != 1)
    return failure();
  if (!visited.insert(producer).second)
    return success();
  for (Value operand : producer->getOperands())
    if (failed(collectLoadRoots(operand, sourceId, roots, visited)))
      return failure();
  return success();
}

FailureOr<SourcePlan> analyzeSource(Value source, unsigned reductionAxis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || reductionAxis >= fragment.getShape().size())
    return failure();
  FailureOr<AxisMapAttr> mapping = axisMap(fragment, reductionAxis);
  if (failed(mapping))
    return failure();
  SourcePlan plan{source, mapping->getSourceId(), reductionAxis, {}};
  llvm::SmallPtrSet<Operation *, 16> visited;
  if (failed(collectLoadRoots(source, plan.sourceId, plan.roots, visited)) ||
      plan.roots.empty())
    return failure();
  return plan;
}

FailureOr<Value> replayValue(OpBuilder &builder, Location location, Value value,
                             uint64_t sourceId,
                             PhysicalExprAttr blockedExtent,
                             IRMapping &mapping) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  auto fragment = dyn_cast<FragmentType>(value.getType());
  FailureOr<unsigned> axis = fragment ? axisForSource(fragment, sourceId)
                                      : FailureOr<unsigned>(failure());
  if (!fragment || failed(axis))
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer || !isReplayableProducer(producer) ||
      producer->getNumRegions() != 0 || producer->getNumResults() != 1)
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replayed =
        replayValue(builder, location, operand, sourceId, blockedExtent, mapping);
    if (failed(replayed))
      return failure();
    if (!mapping.lookupOrNull(operand))
      mapping.map(operand, *replayed);
  }
  Operation *clone = builder.clone(*producer, mapping);
  auto clonedType = cast<FragmentType>(clone->getResult(0).getType());
  FailureOr<unsigned> clonedAxis = axisForSource(clonedType, sourceId);
  if (failed(clonedAxis))
    return failure();
  clone->getResult(0).setType(
      replaceExtent(clonedType, *clonedAxis, blockedExtent));
  if (!mapping.lookupOrNull(value))
    mapping.map(value, clone->getResult(0));
  return clone->getResult(0);
}

Value zeroFill(OpBuilder &builder, Location location, FragmentType type) {
  Value zero;
  if (auto floating = dyn_cast<FloatType>(type.getElementType()))
    zero = builder.create<arith::ConstantOp>(
        location, floating, builder.getFloatAttr(floating, 0.0));
  else if (auto integer = dyn_cast<IntegerType>(type.getElementType()))
    zero = builder.create<arith::ConstantOp>(
        location, integer, builder.getIntegerAttr(integer, 0));
  return zero ? Value(builder.create<SplatOp>(location, type, zero)) : Value();
}

ParameterOp getOrCreateParameter(func::FuncOp kernel, StringRef name,
                                 ParameterRole role,
                                 ArrayRef<int64_t> candidates) {
  ParameterOp existing;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() == name)
      existing = parameter;
  });
  if (existing) {
    ParameterAttr schema = existing.getParameter();
    auto expectedCandidates =
        DenseI64ArrayAttr::get(kernel.getContext(), candidates);
    if (schema.getRole() != static_cast<uint32_t>(role) ||
        schema.getCandidates() != expectedCandidates) {
      existing.emitOpError(
          "physical parameter name is reused with a different role or candidate domain");
      return ParameterOp();
    }
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr(name),
      static_cast<uint32_t>(role),
      DenseI64ArrayAttr::get(kernel.getContext(), candidates));
  return builder.create<ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                     schema);
}

FailureOr<SmallVector<Value>> inlinePureRegion(OpBuilder &builder, Region &region,
                                               ValueRange arguments,
                                               std::string &reason) {
  if (region.empty() || region.getBlocks().size() != 1 ||
      region.front().getNumArguments() != arguments.size()) {
    reason = ("combine argument schema mismatch: expected " +
              Twine(region.empty() ? 0 : region.front().getNumArguments()) +
              ", got " + Twine(arguments.size()))
                 .str();
    return failure();
  }
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield) {
    reason = ("combine region terminates with " +
              region.front().getTerminator()->getName().getStringRef())
                 .str();
    return failure();
  }
  IRMapping mapping;
  for (auto [argument, value] :
       llvm::zip(region.front().getArguments(), arguments))
    mapping.map(argument, value);
  for (Operation &operation : region.front().without_terminator()) {
    Operation *clone = builder.clone(operation, mapping);
    for (auto [source, result] :
         llvm::zip(operation.getResults(), clone->getResults()))
      if (!mapping.lookupOrNull(source))
        mapping.map(source, result);
  }
  SmallVector<Value> results;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      reason = "combine yield value was not mapped by pure-region cloning";
      return failure();
    }
    results.push_back(mapped);
  }
  return results;
}

LogicalResult realizeRuntimeReduce(ReduceOp reduce,
                                   ArrayRef<SourcePlan> sourcePlans,
                                   func::FuncOp kernel) {
  if (sourcePlans.empty())
    return reduce.emitOpError("runtime reduction has no physical sources");
  SmallVector<SmallVector<RootAccess>> accesses;
  for (const SourcePlan &plan : sourcePlans) {
    auto source = cast<FragmentType>(plan.source.getType());
    for (auto [axis, extent] : llvm::enumerate(source.getShape()))
      if (axis != plan.reductionAxis &&
          !isCompileTimeExtent(cast<PhysicalExprAttr>(extent)))
        return reduce.emitOpError(
            "runtime reduction free axes must be physicalized before chunking");
    SmallVector<RootAccess> roots;
    for (LoadOp load : plan.roots) {
      FailureOr<RootAccess> access = analyzeRoot(load, plan.sourceId);
      if (failed(access))
        return reduce.emitOpError(
            "load-rooted producer has no unit-step reduction coordinate");
      roots.push_back(*access);
    }
    accesses.push_back(std::move(roots));
  }
  for (Type result : reduce.getResultTypes())
    if (auto fragment = dyn_cast<FragmentType>(result))
      if (llvm::any_of(fragment.getShape(), [](Attribute extent) {
            return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
          }))
        return reduce.emitOpError(
            "runtime reduction result still has an unphysicalized free axis");

  MakeRangeOp firstRange = accesses.front().front().range;
  for (const auto &component : accesses)
    for (RootAccess access : component)
      if (access.range.getStart() != firstRange.getStart() ||
          access.range.getExtent() != firstRange.getExtent() ||
          access.range.getStep() != firstRange.getStep())
        return reduce.emitOpError(
            "runtime reduction producer roots require one lockstep source range");
  std::string name =
      ("REDUCE_CHUNK_" + Twine(sourcePlans.front().sourceId)).str();
  ParameterOp chunk = getOrCreateParameter(
      kernel, name, ParameterRole::Reduction, {32, 64, 128, 256});
  if (!chunk)
    return failure();
  PhysicalExprAttr chunkExtent = expression(
      reduce.getContext(), PhysicalExprKind::Parameter, 0,
      chunk.getParameter().getName().getValue());

  OpBuilder builder(reduce);
  Location location = reduce.getLoc();
  Value stop = builder.create<BinaryOp>(
      location, builder.getIndexType(), firstRange.getStart(),
      firstRange.getExtent(), 0);
  SmallVector<Value> identities(
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .begin(),
      reduce.getInputs()
          .slice(reduce.getSourceCount(), reduce.getIdentityCount())
          .end());
  ValueRange captures = reduce.getInputs().drop_front(
      reduce.getSourceCount() + reduce.getIdentityCount());

  bool bodyFailed = false;
  std::string bodyFailure = "unknown producer replay failure";
  auto loop = builder.create<scf::ForOp>(
      location, firstRange.getStart(), stop, chunk.getResult(), identities,
      [&](OpBuilder &nested, Location nestedLocation, Value chunkStart,
          ValueRange carries) {
        SmallVector<Value> blockedSources;
        for (auto [component, plan] : llvm::enumerate(sourcePlans)) {
          auto originalSource = cast<FragmentType>(plan.source.getType());
          FragmentType blockedSource = replaceExtent(
              originalSource, plan.reductionAxis, chunkExtent);
          auto blockedPredicate = FragmentType::get(
              reduce.getContext(), nested.getI1Type(), blockedSource.getShape(),
              blockedSource.getAxisMaps(), 2, blockedSource.getOwner());
          IRMapping mapping;
          Value sourceTail;
          for (RootAccess access : accesses[component]) {
            LoadOp load = access.load;
            MakeRangeOp range = access.range;
            auto rootType = cast<FragmentType>(load.getResult().getType());
            FragmentType blockedRoot =
                replaceExtent(rootType, access.fragmentAxis, chunkExtent);
            SmallVector<Attribute> coordinateShape(
                range.getResult().getType().getShape().begin(),
                range.getResult().getType().getShape().end());
            coordinateShape[0] = chunkExtent;
            auto blockedCoordinate = FragmentType::get(
                reduce.getContext(), range.getResult().getType().getElementType(),
                ArrayAttr::get(reduce.getContext(), coordinateShape),
                range.getResult().getType().getAxisMaps(), 2,
                range.getResult().getType().getOwner());
            Value coordinate = nested.create<MakeRangeOp>(
                nestedLocation, blockedCoordinate, chunkStart, chunk.getResult(),
                range.getStep(), range.getSourceId(), range.getSourceAxis());
            if (!mapping.lookupOrNull(range.getResult()))
              mapping.map(range.getResult(), coordinate);
            Value end = nested.create<BroadcastOp>(nestedLocation,
                                                   blockedCoordinate, stop);
            auto coordinatePredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(),
                blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(), 2,
                blockedCoordinate.getOwner());
            Value coordinateValid = nested.create<CompareOp>(
                nestedLocation, coordinatePredicate, coordinate, end, 2);
            auto rootPredicate = FragmentType::get(
                reduce.getContext(), nested.getI1Type(), blockedRoot.getShape(),
                blockedRoot.getAxisMaps(), 2, blockedRoot.getOwner());
            Value valid = nested.create<BroadcastOp>(nestedLocation,
                                                     rootPredicate,
                                                     coordinateValid);
            if (load.getValid()) {
              FailureOr<Value> original = replayValue(
                  nested, nestedLocation, load.getValid(), plan.sourceId,
                  chunkExtent, mapping);
              if (failed(original)) {
                bodyFailed = true;
                bodyFailure = "could not replay source validity";
                return;
              }
              Value originalValid = *original;
              if (originalValid.getType() != rootPredicate)
                originalValid = nested.create<BroadcastOp>(
                    nestedLocation, rootPredicate, originalValid);
              valid = nested.create<BinaryOp>(nestedLocation, rootPredicate,
                                              valid, originalValid, 11);
            }
            Value fill;
            if (load.getFill()) {
              FailureOr<Value> replayedFill = replayValue(
                  nested, nestedLocation, load.getFill(), plan.sourceId,
                  chunkExtent, mapping);
              if (failed(replayedFill)) {
                bodyFailed = true;
                bodyFailure = "could not replay source fill";
                return;
              }
              fill = *replayedFill;
              if (fill.getType() != blockedRoot)
                fill = nested.create<BroadcastOp>(nestedLocation, blockedRoot,
                                                  fill);
            } else {
              fill = zeroFill(nested, nestedLocation, blockedRoot);
              if (!fill) {
                bodyFailed = true;
                bodyFailure = "source element type has no zero fill";
                return;
              }
            }
            SmallVector<Value> coordinates(load.getCoordinates());
            coordinates[access.coordinateIndex] = coordinate;
            Value blockedLoad = nested.create<LoadOp>(
                nestedLocation, blockedRoot, load.getResource(), coordinates,
                valid, fill, load.getSourceAxes());
            mapping.map(load.getResult(), blockedLoad);
            if (!sourceTail)
              sourceTail = nested.create<BroadcastOp>(
                  nestedLocation, blockedPredicate, coordinateValid);
          }
          FailureOr<Value> replayed = replayValue(
              nested, nestedLocation, plan.source, plan.sourceId, chunkExtent,
              mapping);
          if (failed(replayed) || !sourceTail) {
            bodyFailed = true;
            bodyFailure = "could not replay load-rooted pure producer graph";
            return;
          }
          Value identity = identities[component];
          if (identity.getType() != blockedSource)
            identity = nested.create<BroadcastOp>(nestedLocation, blockedSource,
                                                  identity);
          blockedSources.push_back(nested.create<SelectOp>(
              nestedLocation, blockedSource, sourceTail, *replayed, identity));
        }
        if (bodyFailed)
          return;

        SmallVector<Value> chunkInputs(blockedSources);
        chunkInputs.append(identities.begin(), identities.end());
        chunkInputs.append(captures.begin(), captures.end());
        OperationState state(nestedLocation, ReduceOp::getOperationName());
        state.addOperands(chunkInputs);
        state.addTypes(reduce.getResultTypes());
        state.addAttribute("axes", reduce->getAttr("axes"));
        state.addAttribute("source_count", reduce->getAttr("source_count"));
        state.addAttribute("identity_count", reduce->getAttr("identity_count"));
        state.addAttribute("capture_count", reduce->getAttr("capture_count"));
        state.addRegion();
        Operation *raw = nested.create(state);
        auto chunkReduce = cast<ReduceOp>(raw);
        if (Attribute origin = reduce->getAttr(originAttr))
          chunkReduce->setAttr(originAttr, origin);
        IRMapping regionMapping;
        reduce.getCombine().cloneInto(&chunkReduce.getCombine(), regionMapping);

        SmallVector<Value> combineArguments(carries.begin(), carries.end());
        combineArguments.append(chunkReduce.getResults().begin(),
                                chunkReduce.getResults().end());
        combineArguments.append(captures.begin(), captures.end());
        FailureOr<SmallVector<Value>> combined = inlinePureRegion(
            nested, chunkReduce.getCombine(), combineArguments, bodyFailure);
        if (failed(combined)) {
          bodyFailed = true;
          return;
        }
        nested.create<scf::YieldOp>(nestedLocation, *combined);
      });
  if (Attribute origin = reduce->getAttr(originAttr))
    loop->setAttr(originAttr, origin);
  if (bodyFailed) {
    loop.erase();
    return reduce.emitOpError(
        "runtime reduction could not be materialized in the chunk loop: ")
           << bodyFailure;
  }

  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), loop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult realizeReduce(ReduceOp reduce, func::FuncOp kernel) {
  const bool required = requiresPhysicalRealization(reduce);
  auto unhandled = [&](const Twine &reason) -> LogicalResult {
    return required ? reduce.emitOpError()
                          << "cannot form a complete physical reduction: "
                          << reason
                    : success();
  };
  if (reduce.getAxes().size() != 1 || reduce.getSourceCount() == 0)
    return unhandled("requires one reduction axis and at least one source");
  int64_t reductionAxis = reduce.getAxes().front();
  SmallVector<SourcePlan> sourcePlans;
  SmallVector<LoadOp> sourceLoads;
  SmallVector<MakeRangeOp> sourceRanges;
  SmallVector<unsigned> coordinateIndices;
  PhysicalExprAttr sourceExtent;
  bool hasDerivedSource = false;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || reductionAxis < 0 ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()))
      return unhandled("source has no physical reduction axis");
    FailureOr<SourcePlan> plan = analyzeSource(source, reductionAxis);
    if (failed(plan))
      return unhandled(
          "source is not a load-rooted pure producer on the reduction axis");
    for (LoadOp root : plan->roots)
      if (failed(analyzeRoot(root, plan->sourceId)))
        return unhandled(
            "load-rooted producer has no unit-step reduction coordinate");
    PhysicalExprAttr extent =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (sourceExtent && sourceExtent != extent)
      return unhandled("reduction components disagree on physical extent");
    sourceExtent = extent;
    hasDerivedSource |=
        plan->roots.size() != 1 || plan->source != plan->roots.front().getResult();
    if (!hasDerivedSource) {
      RootAccess access = *analyzeRoot(plan->roots.front(), plan->sourceId);
      Value identity = reduce.getInputs()[reduce.getSourceCount() +
                                          sourcePlans.size()];
      if (access.load.getValid() &&
          (!sameScalarValue(access.load.getFill(), identity) ||
           failed(scalarSource(access.load.getValid()))))
        hasDerivedSource = true;
      sourceLoads.push_back(access.load);
      sourceRanges.push_back(access.range);
      coordinateIndices.push_back(access.coordinateIndex);
    }
    sourcePlans.push_back(*plan);
  }
  if (!sourceExtent)
    return unhandled("reduction has no physical source extent");
  if (!isCompileTimeExtent(sourceExtent) || hasDerivedSource)
    return realizeRuntimeReduce(reduce, sourcePlans, kernel);

  PhysicalExprAttr blockExtent = nextPowerOfTwo(sourceExtent);
  OpBuilder builder(reduce);
  Value physicalExtent;
  if (blockExtent.getKind() ==
      static_cast<uint32_t>(PhysicalExprKind::Constant))
    physicalExtent = builder.create<arith::ConstantIndexOp>(
        reduce.getLoc(), blockExtent.getValue());
  else
    physicalExtent = builder.create<PhysicalExprOp>(
        reduce.getLoc(), builder.getIndexType(), blockExtent);

  SmallVector<Value> blockedSources;
  for (unsigned component = 0; component < reduce.getSourceCount(); ++component) {
    LoadOp load = sourceLoads[component];
    MakeRangeOp range = sourceRanges[component];
    auto sourceType = cast<FragmentType>(load.getResult().getType());
    FragmentType blockedSource =
        replaceExtent(sourceType, reductionAxis, blockExtent);
    SmallVector<Attribute> coordinateShape(
        range.getResult().getType().getShape().begin(),
        range.getResult().getType().getShape().end());
    coordinateShape[0] = blockExtent;
    auto blockedCoordinate = FragmentType::get(
        reduce.getContext(), range.getResult().getType().getElementType(),
        ArrayAttr::get(reduce.getContext(), coordinateShape),
        range.getResult().getType().getAxisMaps(), 2,
        range.getResult().getType().getOwner());
    Value coordinate = builder.create<MakeRangeOp>(
        reduce.getLoc(), blockedCoordinate, range.getStart(), physicalExtent,
        range.getStep(), range.getSourceId(), range.getSourceAxis());
    Value stop = builder.create<BinaryOp>(
        reduce.getLoc(), builder.getIndexType(), range.getStart(),
        range.getExtent(), 0);
    auto coordinatePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), blockedCoordinate.getShape(),
        blockedCoordinate.getAxisMaps(), 2, blockedCoordinate.getOwner());
    Value stopFragment =
        builder.create<BroadcastOp>(reduce.getLoc(), blockedCoordinate, stop);
    Value valid = builder.create<CompareOp>(
        reduce.getLoc(), coordinatePredicate, coordinate, stopFragment, 2);
    auto sourcePredicate = FragmentType::get(
        reduce.getContext(), builder.getI1Type(), blockedSource.getShape(),
        blockedSource.getAxisMaps(), 2, blockedSource.getOwner());
    valid = builder.create<BroadcastOp>(reduce.getLoc(), sourcePredicate, valid);
    if (load.getValid()) {
      FailureOr<Value> scalar = scalarSource(load.getValid());
      Value original = builder.create<BroadcastOp>(reduce.getLoc(),
                                                    sourcePredicate, *scalar);
      valid = builder.create<BinaryOp>(reduce.getLoc(), sourcePredicate, valid,
                                       original, 11);
    }
    Value identity =
        reduce.getInputs()[reduce.getSourceCount() + component];
    Value fill = identity;
    if (identity.getType() != blockedSource)
      fill = builder.create<BroadcastOp>(reduce.getLoc(), blockedSource, identity);
    SmallVector<Value> coordinates(load.getCoordinates());
    coordinates[coordinateIndices[component]] = coordinate;
    blockedSources.push_back(builder.create<LoadOp>(
        reduce.getLoc(), blockedSource, load.getResource(), coordinates, valid,
        fill, load.getSourceAxes()));
  }

  SmallVector<Value> inputs(blockedSources);
  inputs.append(reduce.getInputs().drop_front(reduce.getSourceCount()).begin(),
                reduce.getInputs().drop_front(reduce.getSourceCount()).end());
  OperationState state(reduce.getLoc(), ReduceOp::getOperationName());
  state.addOperands(inputs);
  state.addTypes(reduce.getResultTypes());
  state.addAttribute("axes", reduce->getAttr("axes"));
  state.addAttribute("source_count", reduce->getAttr("source_count"));
  state.addAttribute("identity_count", reduce->getAttr("identity_count"));
  state.addAttribute("capture_count", reduce->getAttr("capture_count"));
  state.addRegion();
  Operation *raw = builder.create(state);
  auto replacement = cast<ReduceOp>(raw);
  replacement.getCombine().takeBody(reduce.getCombine());
  if (Attribute origin = reduce->getAttr(originAttr))
    replacement->setAttr(originAttr, origin);
  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  for (LoadOp load : sourceLoads)
    if (load->getBlock() && load.getResult().use_empty())
      load.erase();
  return success();
}

} // namespace

LogicalResult realizeReductionBlocking(ModuleOp module) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  SmallVector<ReduceOp> reductions;
  kernel.walk([&](ReduceOp reduce) { reductions.push_back(reduce); });
  for (ReduceOp reduce : reductions)
    if (reduce->getBlock() && failed(realizeReduce(reduce, kernel)))
      return failure();
  return success();
}

} // namespace intent::gpu

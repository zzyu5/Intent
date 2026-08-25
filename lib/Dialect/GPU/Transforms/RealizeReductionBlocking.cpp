#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

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
                                               ValueRange arguments) {
  if (region.empty() || region.getBlocks().size() != 1 ||
      region.front().getNumArguments() != arguments.size())
    return failure();
  auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
  if (!yield)
    return failure();
  IRMapping mapping;
  for (auto [argument, value] :
       llvm::zip(region.front().getArguments(), arguments))
    mapping.map(argument, value);
  for (Operation &operation : region.front().without_terminator())
    builder.clone(operation, mapping);
  SmallVector<Value> results;
  for (Value value : yield.getValues()) {
    Value mapped = mapping.lookupOrNull(value);
    if (!mapped)
      return failure();
    results.push_back(mapped);
  }
  return results;
}

LogicalResult realizeRuntimeReduce(
    ReduceOp reduce, ArrayRef<LoadOp> sourceLoads,
    ArrayRef<MakeRangeOp> sourceRanges, ArrayRef<unsigned> coordinateIndices,
    unsigned reductionAxis, func::FuncOp kernel) {
  if (sourceLoads.empty())
    return reduce.emitOpError("runtime reduction has no physical source loads");
  for (LoadOp load : sourceLoads) {
    auto source = cast<FragmentType>(load.getResult().getType());
    for (auto [axis, extent] : llvm::enumerate(source.getShape()))
      if (axis != reductionAxis &&
          !isCompileTimeExtent(cast<PhysicalExprAttr>(extent)))
        return reduce.emitOpError(
            "runtime reduction free axes must be physicalized before chunking");
  }
  for (Type result : reduce.getResultTypes())
    if (auto fragment = dyn_cast<FragmentType>(result))
      if (llvm::any_of(fragment.getShape(), [](Attribute extent) {
            return !isCompileTimeExtent(cast<PhysicalExprAttr>(extent));
          }))
        return reduce.emitOpError(
            "runtime reduction result still has an unphysicalized free axis");

  MakeRangeOp firstRange = sourceRanges.front();
  LoadOp firstLoad = sourceLoads.front();
  for (MakeRangeOp range : sourceRanges)
    if (range.getStart() != firstRange.getStart() ||
        range.getExtent() != firstRange.getExtent() ||
        range.getStep() != firstRange.getStep())
      return reduce.emitOpError(
          "runtime reduction components require one lockstep source range");
  auto firstType = cast<FragmentType>(firstLoad.getResult().getType());
  FailureOr<AxisMapAttr> firstMapping = axisMap(firstType, reductionAxis);
  if (failed(firstMapping))
    return reduce.emitOpError("runtime reduction axis lost source provenance");
  std::string name =
      ("REDUCE_CHUNK_" + Twine(firstMapping->getSourceId())).str();
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
  auto loop = builder.create<scf::ForOp>(
      location, firstRange.getStart(), stop, chunk.getResult(), identities,
      [&](OpBuilder &nested, Location nestedLocation, Value chunkStart,
          ValueRange carries) {
        SmallVector<Value> blockedSources;
        for (unsigned component = 0; component < sourceLoads.size(); ++component) {
          LoadOp load = sourceLoads[component];
          MakeRangeOp range = sourceRanges[component];
          auto sourceType = cast<FragmentType>(load.getResult().getType());
          FragmentType blockedSource =
              replaceExtent(sourceType, reductionAxis, chunkExtent);
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
          Value end = nested.create<BroadcastOp>(nestedLocation,
                                                 blockedCoordinate, stop);
          auto coordinatePredicate = FragmentType::get(
              reduce.getContext(), nested.getI1Type(),
              blockedCoordinate.getShape(), blockedCoordinate.getAxisMaps(), 2,
              blockedCoordinate.getOwner());
          Value valid = nested.create<CompareOp>(nestedLocation,
                                                 coordinatePredicate, coordinate,
                                                 end, 2);
          auto sourcePredicate = FragmentType::get(
              reduce.getContext(), nested.getI1Type(), blockedSource.getShape(),
              blockedSource.getAxisMaps(), 2, blockedSource.getOwner());
          valid = nested.create<BroadcastOp>(nestedLocation, sourcePredicate,
                                             valid);
          if (load.getValid()) {
            FailureOr<Value> scalar = scalarSource(load.getValid());
            if (failed(scalar)) {
              bodyFailed = true;
              return;
            }
            Value original = nested.create<BroadcastOp>(nestedLocation,
                                                        sourcePredicate, *scalar);
            valid = nested.create<BinaryOp>(nestedLocation, sourcePredicate,
                                            valid, original, 11);
          }
          Value identity = identities[component];
          Value fill = identity;
          if (fill.getType() != blockedSource)
            fill = nested.create<BroadcastOp>(nestedLocation, blockedSource,
                                              identity);
          SmallVector<Value> coordinates(load.getCoordinates());
          coordinates[coordinateIndices[component]] = coordinate;
          blockedSources.push_back(nested.create<LoadOp>(
              nestedLocation, blockedSource, load.getResource(), coordinates,
              valid, fill, load.getSourceAxes()));
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
        FailureOr<SmallVector<Value>> combined =
            inlinePureRegion(nested, reduce.getCombine(), combineArguments);
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
        "runtime reduction combine could not be materialized in the chunk loop");
  }

  for (auto [oldResult, newResult] :
       llvm::zip(reduce.getResults(), loop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  reduce.erase();
  for (LoadOp load : sourceLoads)
    if (load->getBlock() && load.getResult().use_empty())
      load.erase();
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
  SmallVector<LoadOp> sourceLoads;
  SmallVector<MakeRangeOp> sourceRanges;
  SmallVector<unsigned> coordinateIndices;
  PhysicalExprAttr sourceExtent;
  for (Value source : reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    auto load = source.getDefiningOp<LoadOp>();
    if (!fragment || reductionAxis < 0 ||
        reductionAxis >= static_cast<int64_t>(fragment.getShape().size()) ||
        !load)
      return unhandled("source is not a direct physical load on the reduction axis");
    FailureOr<AxisMapAttr> mapping = axisMap(fragment, reductionAxis);
    if (failed(mapping))
      return unhandled("reduction axis lost coordinate provenance");
    FailureOr<unsigned> coordinate =
        coordinateForSource(load.getCoordinates(), mapping->getSourceId());
    if (failed(coordinate))
      return unhandled("load coordinates do not cover the reduction axis");
    auto range = load.getCoordinates()[*coordinate].getDefiningOp<MakeRangeOp>();
    if (!range)
      return unhandled("reduction coordinate is not an explicit physical range");
    auto step = range.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!step || step.value() != 1)
      return unhandled("reduction blocking currently requires a unit-step range");
    if (load.getValid() && failed(scalarSource(load.getValid())))
      return unhandled("source has non-scalar residual validity");
    Value identity =
        reduce.getInputs()[reduce.getSourceCount() + sourceLoads.size()];
    if (load.getValid() && !sameScalarValue(load.getFill(), identity))
      return unhandled("invalid source fill is not the reduction identity");
    PhysicalExprAttr extent =
        cast<PhysicalExprAttr>(fragment.getShape()[reductionAxis]);
    if (sourceExtent && sourceExtent != extent)
      return unhandled("reduction components disagree on physical extent");
    sourceExtent = extent;
    sourceLoads.push_back(load);
    sourceRanges.push_back(range);
    coordinateIndices.push_back(*coordinate);
  }
  if (!sourceExtent)
    return unhandled("reduction has no physical source extent");
  if (!isCompileTimeExtent(sourceExtent))
    return realizeRuntimeReduce(reduce, sourceLoads, sourceRanges,
                                coordinateIndices, reductionAxis, kernel);

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

#include "Intent/Target/CuTile/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

using namespace mlir;

namespace intent::cutile {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_cutile.legalized";

bool isOne(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  return integer && integer.getInt() == 1;
}

std::optional<int64_t> constantValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto bound = value.getDefiningOp<gpu::RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<gpu::RangeOp>();
    if (!range)
      return std::nullopt;
    if (bound.getBound() == 0)
      return constantValue(range.getStart());
    if (bound.getBound() == 1)
      return constantValue(range.getStop());
    return constantValue(range.getStep());
  }
  if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    std::optional<int64_t> lhs = constantValue(binary.getLhs());
    std::optional<int64_t> rhs = constantValue(binary.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    switch (binary.getOperatorKind()) {
    case 0:
      return *lhs + *rhs;
    case 1:
      return *lhs - *rhs;
    case 2:
      return *lhs * *rhs;
    case 4:
      return *rhs == 0 ? std::nullopt
                       : std::optional<int64_t>(*lhs / *rhs);
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

bool isProvably(Value value, int64_t expected) {
  std::optional<int64_t> actual = constantValue(value);
  return actual && *actual == expected;
}

Value stripBroadcast(Value value) {
  while (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
    value = broadcast.getValue();
  return value;
}

bool isViewExtent(Value value, Value resource, unsigned axis) {
  value = stripBroadcast(value);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    if (!function)
      return false;
    DictionaryAttr attrs = function.getArgAttrDict(argument.getArgNumber());
    auto kind = attrs.getAs<StringAttr>(gpu::abiKindAttr);
    auto view = cast<gpu::ViewType>(resource.getType());
    auto name = attrs.getAs<StringAttr>(gpu::abiNameAttr);
    auto extent = cast<gpu::PhysicalExprAttr>(
        view.getLayout().getExtents()[axis]);
    return kind && kind.getValue() == "dimension" && name &&
           extent.getKind() ==
               static_cast<uint32_t>(gpu::PhysicalExprKind::Dimension) &&
           name.getValue() == extent.getSymbol().getValue();
  }
  if (auto dim = value.getDefiningOp<gpu::DimOp>())
    return dim.getView() == resource && dim.getAxis() == axis;
  if (auto expression = value.getDefiningOp<gpu::PhysicalExprOp>()) {
    auto view = cast<gpu::ViewType>(resource.getType());
    return expression.getExpression() ==
           cast<gpu::PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
  }
  if (auto bound = value.getDefiningOp<gpu::RangeBoundOp>()) {
    auto range = bound.getRange().getDefiningOp<gpu::RangeOp>();
    if (!range)
      return false;
    if (bound.getBound() == 0)
      return isViewExtent(range.getStart(), resource, axis);
    if (bound.getBound() == 1)
      return isViewExtent(range.getStop(), resource, axis);
    return isViewExtent(range.getStep(), resource, axis);
  }
  if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() == 0 && isProvably(binary.getLhs(), 0))
      return isViewExtent(binary.getRhs(), resource, axis);
    if (binary.getOperatorKind() == 0 && isProvably(binary.getRhs(), 0))
      return isViewExtent(binary.getLhs(), resource, axis);
    if (binary.getOperatorKind() == 1 && isProvably(binary.getRhs(), 0))
      return isViewExtent(binary.getLhs(), resource, axis);
    if ((binary.getOperatorKind() == 2 || binary.getOperatorKind() == 4) &&
        isProvably(binary.getRhs(), 1))
      return isViewExtent(binary.getLhs(), resource, axis);
    if (binary.getOperatorKind() == 2 && isProvably(binary.getLhs(), 1))
      return isViewExtent(binary.getRhs(), resource, axis);
  }
  return false;
}

bool derivesFromCoordinate(Value value, Value coordinate) {
  if (value == coordinate)
    return true;
  if (auto broadcast = value.getDefiningOp<gpu::BroadcastOp>())
    return derivesFromCoordinate(broadcast.getValue(), coordinate);
  return false;
}

bool isFullViewValidity(Value valid, Value resource, ValueRange coordinates,
                        ArrayRef<int64_t> sourceAxes) {
  if (!valid)
    return false;
  valid = stripBroadcast(valid);
  if (auto binary = valid.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() != 11)
      return false;
    return isFullViewValidity(binary.getLhs(), resource, coordinates, sourceAxes) &&
           isFullViewValidity(binary.getRhs(), resource, coordinates, sourceAxes);
  }
  auto compare = valid.getDefiningOp<gpu::CompareOp>();
  if (!compare || compare.getPredicate() != 2)
    return false;
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes))
    if (derivesFromCoordinate(compare.getLhs(), coordinate) &&
        isViewExtent(compare.getRhs(), resource, sourceAxis))
      return true;
  return false;
}

FailureOr<Value> tileIndex(OpBuilder &builder, Location location, Value start,
                           Value extent) {
  if (isProvably(start, 0))
    return builder.create<arith::ConstantIndexOp>(location, 0).getResult();
  if (auto binary = start.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() == 2) {
      if (binary.getLhs() == extent)
        return binary.getRhs();
      if (binary.getRhs() == extent)
        return binary.getLhs();
    }
    if (binary.getOperatorKind() == 0) {
      if (isProvably(binary.getLhs(), 0))
        return tileIndex(builder, location, binary.getRhs(), extent);
      if (isProvably(binary.getRhs(), 0))
        return tileIndex(builder, location, binary.getLhs(), extent);
    }
  }
  auto argument = dyn_cast<BlockArgument>(start);
  auto loop = argument ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp())
                       : scf::ForOp();
  if (loop && argument == loop.getInductionVar() &&
      isProvably(loop.getLowerBound(), 0) && loop.getStep() == extent)
    return Value(builder.create<gpu::BinaryOp>(
        location, builder.getIndexType(), start, extent, /*floor-div=*/4));
  return failure();
}

FailureOr<SmallVector<Value>> tileIndices(OpBuilder &builder, Operation *owner,
                                          Value resource,
                                          ValueRange coordinates,
                                          ArrayRef<int64_t> sourceAxes) {
  auto view = cast<gpu::ViewType>(resource.getType());
  SmallVector<Value> result(view.getRank());
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes)) {
    auto range = coordinate.getDefiningOp<gpu::MakeRangeOp>();
    if (!range || !isOne(range.getStep()))
      return failure();
    FailureOr<Value> index =
        tileIndex(builder, owner->getLoc(), range.getStart(), range.getExtent());
    if (failed(index))
      return failure();
    result[sourceAxis] = *index;
  }
  if (llvm::any_of(result, [](Value value) { return !value; }))
    return failure();
  return result;
}

FailureOr<SmallVector<Value>> orderedCoordinates(Operation *owner, Value resource,
                                                 ValueRange coordinates,
                                                 ArrayRef<int64_t> sourceAxes) {
  auto view = cast<gpu::ViewType>(resource.getType());
  SmallVector<Value> result(view.getRank());
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes))
    result[sourceAxis] = coordinate;
  if (llvm::any_of(result, [](Value value) { return !value; }))
    return owner->emitOpError("cuTile advanced access source axes are incomplete");
  return result;
}

bool samePhysicalDomain(gpu::FragmentType lhs, gpu::FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

bool canBroadcastTo(gpu::FragmentType source, gpu::FragmentType target) {
  for (auto [sourceIndex, sourceAttribute] :
       llvm::enumerate(source.getAxisMaps())) {
    auto sourceMap = cast<gpu::AxisMapAttr>(sourceAttribute);
    std::optional<unsigned> targetIndex;
    for (auto [index, targetAttribute] :
         llvm::enumerate(target.getAxisMaps())) {
      auto targetMap = cast<gpu::AxisMapAttr>(targetAttribute);
      if (sourceMap.getSourceId() != targetMap.getSourceId() ||
          sourceMap.getSourceAxis() != targetMap.getSourceAxis())
        continue;
      if (targetIndex)
        return false;
      targetIndex = index;
    }
    if (!targetIndex)
      return false;
    Attribute sourceExtent = source.getShape()[sourceIndex];
    Attribute targetExtent = target.getShape()[*targetIndex];
    auto constant = dyn_cast<gpu::PhysicalExprAttr>(sourceExtent);
    bool unit = constant &&
                constant.getKind() == static_cast<uint32_t>(
                                          gpu::PhysicalExprKind::Constant) &&
                constant.getValue() == 1;
    if (!unit && sourceExtent != targetExtent)
      return false;
  }
  return true;
}

FailureOr<SmallVector<Value>> materializeCoordinateDomains(
    OpBuilder &builder, Operation *owner, ValueRange coordinates,
    gpu::FragmentType target) {
  SmallVector<Value> results;
  results.reserve(coordinates.size());
  for (Value coordinate : coordinates) {
    auto source = dyn_cast<gpu::FragmentType>(coordinate.getType());
    if (!source || samePhysicalDomain(source, target)) {
      results.push_back(coordinate);
      continue;
    }
    if (source.getOwner() != target.getOwner() ||
        source.getShape().size() > target.getShape().size() ||
        !canBroadcastTo(source, target))
      return owner->emitOpError(
          "cuTile coordinate cannot adopt the selected physical domain");
    auto resultType = gpu::FragmentType::get(
        owner->getContext(), source.getElementType(), target.getShape(),
        target.getAxisMaps(), target.getValidity(), target.getOwner());
    auto broadcast = builder.create<gpu::BroadcastOp>(
        owner->getLoc(), resultType, coordinate);
    if (Operation *definition = coordinate.getDefiningOp())
      if (Attribute origin = definition->getAttr(gpu::originAttr))
        broadcast->setAttr(gpu::originAttr, origin);
    results.push_back(broadcast.getResult());
  }
  return results;
}

FailureOr<Value> scalarFill(Operation *owner, Value fill) {
  if (!fill)
    return Value();
  if (!isa<gpu::FragmentType>(fill.getType()))
    return fill;
  if (auto splat = fill.getDefiningOp<gpu::SplatOp>())
    return splat.getValue();
  return owner->emitOpError(
      "cuTile gather padding must be an explicit scalar or splat");
}

std::optional<uint64_t> nativeCombineKind(Region &region) {
  if (!llvm::hasSingleElement(region))
    return std::nullopt;
  Block &block = region.front();
  if (block.getNumArguments() != 2 ||
      std::distance(block.begin(), block.end()) != 2)
    return std::nullopt;
  auto binary = dyn_cast<gpu::BinaryOp>(block.front());
  auto yield = dyn_cast<gpu::YieldOp>(block.back());
  if (!binary || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != binary.getResult())
    return std::nullopt;
  if (!((binary.getLhs() == block.getArgument(0) &&
         binary.getRhs() == block.getArgument(1)) ||
        (binary.getLhs() == block.getArgument(1) &&
         binary.getRhs() == block.getArgument(0))))
    return std::nullopt;
  if (binary.getOperatorKind() == 0)
    return 0;
  if (binary.getOperatorKind() == 7 || binary.getOperatorKind() == 9)
    return 1;
  if (binary.getOperatorKind() == 8 || binary.getOperatorKind() == 10)
    return 2;
  return std::nullopt;
}

LogicalResult formNativeTiles(func::FuncOp kernel) {
  SmallVector<gpu::LoadOp> loads;
  SmallVector<gpu::GatherOp> gathers;
  SmallVector<gpu::ContractOp> contracts;
  SmallVector<gpu::ScaledContractOp> scaledContracts;
  SmallVector<gpu::ReduceOp> reductions;
  SmallVector<gpu::ScanOp> scans;
  SmallVector<gpu::StoreOp> stores;
  SmallVector<gpu::AtomicRMWOp> atomics;
  SmallVector<gpu::AssumeInBoundsOp> assumptions;
  kernel.walk([&](gpu::LoadOp op) { loads.push_back(op); });
  kernel.walk([&](gpu::GatherOp op) { gathers.push_back(op); });
  kernel.walk([&](gpu::ContractOp op) { contracts.push_back(op); });
  kernel.walk(
      [&](gpu::ScaledContractOp op) { scaledContracts.push_back(op); });
  kernel.walk([&](gpu::ReduceOp op) { reductions.push_back(op); });
  kernel.walk([&](gpu::ScanOp op) { scans.push_back(op); });
  kernel.walk([&](gpu::StoreOp op) { stores.push_back(op); });
  kernel.walk([&](gpu::AtomicRMWOp op) { atomics.push_back(op); });
  kernel.walk([&](gpu::AssumeInBoundsOp op) { assumptions.push_back(op); });

  for (gpu::LoadOp load : loads) {
    auto view = dyn_cast<gpu::ViewType>(load.getResource().getType());
    if (!view)
      return load.emitOpError(
          "cuTile native load requires an external view resource");
    OpBuilder builder(load);
    if (!isa<gpu::FragmentType>(load.getResult().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(
          load, load.getResource(), load.getCoordinates(), load.getSourceAxes());
      FailureOr<Value> fill = scalarFill(load, load.getFill());
      if (failed(indices) || failed(fill) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (load.getValid() && isa<gpu::FragmentType>(load.getValid().getType())))
        return load.emitOpError(
            "cuTile scalar load requires scalar indices and validity");
      auto replacement = builder.create<ScalarLoadOp>(
          load.getLoc(), load.getResult().getType(), load.getResource(), *indices,
          load.getValid(), *fill);
      if (Attribute origin = load->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      load.getResult().replaceAllUsesWith(replacement.getResult());
      load.erase();
      continue;
    }
    auto result = cast<gpu::FragmentType>(load.getResult().getType());
    FailureOr<SmallVector<Value>> indices = tileIndices(
        builder, load, load.getResource(), load.getCoordinates(), load.getSourceAxes());
    Value replacementResult;
    Operation *replacementOperation = nullptr;
    if (succeeded(indices) &&
        isFullViewValidity(load.getValid(), load.getResource(),
                           load.getCoordinates(), load.getSourceAxes())) {
      auto replacement = builder.create<TileLoadOp>(
          load.getLoc(), result, load.getResource(), *indices, load.getValid(),
          load.getFill());
      replacementResult = replacement.getResult();
      replacementOperation = replacement;
    } else {
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
          load, load.getResource(), load.getCoordinates(), load.getSourceAxes());
      FailureOr<Value> fill = scalarFill(load, load.getFill());
      if (failed(coordinates) || failed(fill))
        return failure();
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, load, *coordinates, result);
      if (failed(materialized))
        return failure();
      auto replacement = builder.create<GatherLoadOp>(
          load.getLoc(), result, load.getResource(), *materialized,
          load.getValid(), *fill, load.getSourceAxes());
      replacementResult = replacement.getResult();
      replacementOperation = replacement;
    }
    if (Attribute origin = load->getAttr(gpu::originAttr))
      replacementOperation->setAttr(gpu::originAttr, origin);
    load.getResult().replaceAllUsesWith(replacementResult);
    load.erase();
  }

  for (gpu::GatherOp gather : gathers) {
    auto source = dyn_cast<gpu::FragmentType>(gather.getSource().getType());
    if (!source || isa<gpu::FragmentType>(gather.getResult().getType()) ||
        llvm::any_of(gather.getCoordinates(), [](Value coordinate) {
          return isa<gpu::FragmentType>(coordinate.getType());
        }))
      return gather.emitOpError(
          "cuTile tile extraction requires scalar coordinates and result");
    if (gather.getCoordinates().size() != gather.getSourceAxes().size())
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    SmallVector<Value> coordinates(source.getShape().size());
    for (auto [coordinate, sourceAxis] :
         llvm::zip(gather.getCoordinates(), gather.getSourceAxes())) {
      if (sourceAxis < 0 ||
          sourceAxis >= static_cast<int64_t>(coordinates.size()) ||
          coordinates[sourceAxis])
        return gather.emitOpError(
            "cuTile tile extraction source axes are not a permutation");
      coordinates[sourceAxis] = coordinate;
    }
    if (llvm::any_of(coordinates, [](Value coordinate) { return !coordinate; }))
      return gather.emitOpError("cuTile tile extraction source axes are incomplete");
    OpBuilder builder(gather);
    auto replacement = builder.create<ExtractScalarOp>(
        gather.getLoc(), gather.getResult().getType(), gather.getSource(),
        coordinates, gather.getValid(), gather.getFill());
    if (Attribute origin = gather->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    gather.getResult().replaceAllUsesWith(replacement.getResult());
    gather.erase();
  }

  for (gpu::ReduceOp reduce : reductions) {
    std::optional<uint64_t> kind = nativeCombineKind(reduce.getCombine());
    if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
        reduce.getCaptureCount() != 0 || reduce.getAxes().size() != 1 ||
        reduce.getNumResults() != 1 || !kind)
      return reduce.emitOpError(
          "cuTile native reduce requires one source/identity/axis and builtin add/max/min combine");
    auto source = dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
    if (!source)
      return reduce.emitOpError("cuTile native reduce source must be a tile");
    OpBuilder builder(reduce);
    auto replacement = builder.create<ReduceOp>(
        reduce.getLoc(), reduce.getResultTypes().front(), reduce.getInputs().front(),
        reduce.getAxes().front(), *kind);
    reduce.getResults().front().replaceAllUsesWith(replacement.getResult());
    reduce.erase();
  }

  for (gpu::ScanOp scan : scans) {
    std::optional<uint64_t> kind = nativeCombineKind(scan.getCombine());
    if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
        scan.getCaptureCount() != 0 || scan.getNumResults() != 1 || !kind ||
        *kind != 0 || !scan.getInclusive() || scan.getReverse())
      return scan.emitOpError(
          "cuTile native scan requires one source/identity, inclusive forward additive combine");
    auto source = dyn_cast<gpu::FragmentType>(scan.getInputs().front().getType());
    if (!source)
      return scan.emitOpError("cuTile native scan source must be a tile");
    OpBuilder builder(scan);
    auto replacement = builder.create<ScanOp>(
        scan.getLoc(), cast<gpu::FragmentType>(scan.getResultTypes().front()),
        scan.getInputs().front(), scan.getAxis(), *kind, false);
    scan.getResults().front().replaceAllUsesWith(replacement.getResult());
    scan.erase();
  }

  for (gpu::ContractOp contract : contracts) {
    if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      return contract.emitOpError(
          "cuTile native MMA requires [M,K] x [K,N] physical axes");
    OpBuilder builder(contract);
    auto replacement = builder.create<MMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getRhs(), contract.getAccumulator());
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    contract.getResult().replaceAllUsesWith(replacement.getResult());
    contract.erase();
  }

  for (gpu::ScaledContractOp contract : scaledContracts) {
    if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1, 2} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{0, 1} ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty() || contract.getLhsFormat() != 1 ||
        contract.getRhsFormat() != 1 || contract.getLhsGroupSize() != 32 ||
        contract.getRhsGroupSize() != 32)
      return contract.emitOpError(
          "cuTile scaled MMA requires E4M3/E8M0 group-32 adjacent reduction axes");
    OpBuilder builder(contract);
    auto replacement = builder.create<ScaledMMAOp>(
        contract.getLoc(), contract.getResult().getType(), contract.getLhs(),
        contract.getLhsScale(), contract.getRhs(), contract.getRhsScale(),
        contract.getAccumulator(), contract.getLhsGroupSize());
    if (Attribute origin = contract->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    contract.getResult().replaceAllUsesWith(replacement.getResult());
    contract.erase();
  }

  for (gpu::AtomicRMWOp atomic : atomics) {
    auto view = dyn_cast<gpu::ViewType>(atomic.getResource().getType());
    if (!view)
      return atomic.emitOpError(
          "cuTile atomic RMW requires an external view resource");
    if (atomic.getValid())
      return atomic.emitOpError(
          "cuTile atomic RMW has no native arbitrary-validity form");
    FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
        atomic, atomic.getResource(), atomic.getCoordinates(),
        atomic.getSourceAxes());
    if (failed(coordinates))
      return failure();
    OpBuilder builder(atomic);
    if (auto target = dyn_cast<gpu::FragmentType>(atomic.getValue().getType())) {
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, atomic, *coordinates, target);
      if (failed(materialized))
        return failure();
      coordinates = std::move(*materialized);
    }
    auto replacement = builder.create<AtomicRMWOp>(
        atomic.getLoc(), atomic.getResult().getType(), atomic.getResource(),
        *coordinates, atomic.getValue(), atomic.getKind(), atomic.getOrdering(),
        atomic.getSharing());
    if (Attribute origin = atomic->getAttr(gpu::originAttr))
      replacement->setAttr(gpu::originAttr, origin);
    atomic.getResult().replaceAllUsesWith(replacement.getResult());
    atomic.erase();
  }

  for (gpu::StoreOp store : stores) {
    auto view = dyn_cast<gpu::ViewType>(store.getResource().getType());
    if (!view || store.getCollision() != 0)
      return store.emitOpError(
          "cuTile native store requires an external unique-write view");
    OpBuilder builder(store);
    if (!isa<gpu::FragmentType>(store.getValue().getType())) {
      FailureOr<SmallVector<Value>> indices = orderedCoordinates(
          store, store.getResource(), store.getCoordinates(), store.getSourceAxes());
      if (failed(indices) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (store.getValid() && isa<gpu::FragmentType>(store.getValid().getType())))
        return store.emitOpError(
            "cuTile scalar store requires scalar indices and validity");
      auto replacement = builder.create<ScalarStoreOp>(
          store.getLoc(), store.getResource(), *indices, store.getValue(),
          store.getValid());
      if (Attribute origin = store->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      store.erase();
      continue;
    }
    FailureOr<SmallVector<Value>> indices = tileIndices(
        builder, store, store.getResource(), store.getCoordinates(), store.getSourceAxes());
    Operation *replacementOperation = nullptr;
    if (succeeded(indices) &&
        isFullViewValidity(store.getValid(), store.getResource(),
                           store.getCoordinates(), store.getSourceAxes())) {
      replacementOperation = builder.create<TileStoreOp>(
          store.getLoc(), store.getResource(), *indices, store.getValue());
    } else {
      FailureOr<SmallVector<Value>> coordinates = orderedCoordinates(
          store, store.getResource(), store.getCoordinates(), store.getSourceAxes());
      if (failed(coordinates))
        return failure();
      auto target = cast<gpu::FragmentType>(store.getValue().getType());
      FailureOr<SmallVector<Value>> materialized =
          materializeCoordinateDomains(builder, store, *coordinates, target);
      if (failed(materialized))
        return failure();
      replacementOperation = builder.create<ScatterStoreOp>(
          store.getLoc(), store.getResource(), *materialized, store.getValue(),
          store.getValid(), store.getSourceAxes());
    }
    if (Attribute origin = store->getAttr(gpu::originAttr))
      replacementOperation->setAttr(gpu::originAttr, origin);
    store.erase();
  }
  for (gpu::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  return success();
}

bool isCuTileScalarType(Type type) {
  if (type.isIndex() ||
      isa<Float16Type, BFloat16Type, Float32Type, Float8E4M3FNType,
          Float8E5M2Type>(type))
    return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 1 || integer.getWidth() == 8 ||
                     integer.getWidth() == 16 || integer.getWidth() == 32 ||
                     integer.getWidth() == 64);
}

LogicalResult verifyKernel(func::FuncOp kernel) {
  auto space = kernel->getAttrOfType<ArrayAttr>(gpu::programSpaceAttr);
  if (!space || space.size() != 1)
    return kernel.emitError(
        "cuTile provider currently requires one explicit linear program space");
  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<gpu::LoadOp, gpu::StoreOp, gpu::ContractOp>(operation)) {
      operation->emitOpError(
          "was not converted to an explicit cuTile tile/MMA form");
      return WalkResult::interrupt();
    }
    for (Type type : operation->getResultTypes()) {
      if (auto fragment = dyn_cast<gpu::FragmentType>(type)) {
        if (!isCuTileScalarType(fragment.getElementType())) {
          operation->emitOpError("contains a fragment dtype outside cuTile");
          return WalkResult::interrupt();
        }
      } else if (!isa<gpu::ViewType, gpu::RangeType, gpu::RecordType>(type) &&
                 !isCuTileScalarType(type)) {
        operation->emitOpError("has a result type outside the cuTile surface");
        return WalkResult::interrupt();
      }
    }
    if (isa<TileLoadOp, TileStoreOp, ScalarLoadOp, ScalarStoreOp, GatherLoadOp,
            ScatterStoreOp, AtomicRMWOp, ExtractScalarOp, MMAOp, ScaledMMAOp,
            ReduceOp, ScanOp, gpu::ParameterOp,
            gpu::PhysicalExprOp, gpu::ProgramIdOp, gpu::DelinearizeOp,
            gpu::DimOp, gpu::RangeOp, gpu::RangeBoundOp, gpu::MakeRangeOp,
            gpu::SplatOp, gpu::BroadcastOp, gpu::UnaryOp, gpu::BinaryOp,
            gpu::CompareOp, gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
            gpu::ReshapeOp, gpu::TransposeOp, gpu::JoinOp, gpu::MakeRecordOp,
            gpu::ExtractOp, arith::ConstantOp,
            scf::ForOp, scf::YieldOp, func::FuncOp, func::ReturnOp>(operation))
      return WalkResult::advance();
    operation->emitOpError("is outside the closed cuTile provider surface");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

LogicalResult verifyCuTileProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) ? failure() : verifyKernel(*kernel);
}

LogicalResult legalizeGPUProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(formNativeTiles(*kernel)) ||
      failed(verifyCuTileProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::cutile

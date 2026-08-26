#include "PassDetail.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <numeric>
#include <optional>

using namespace mlir;

namespace intent::tilelang {
namespace {

gpu::PhysicalExprAttr constantExtent(MLIRContext *context, int64_t value) {
  return gpu::PhysicalExprAttr::get(
      context, static_cast<uint32_t>(gpu::PhysicalExprKind::Constant), value,
      StringAttr::get(context), ArrayAttr::get(context, {}));
}

bool isZero(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt() == 0;
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
      return floating.getValue().isZero();
  }
  if (auto splat = value.getDefiningOp<gpu::SplatOp>())
    return isZero(splat.getValue());
  return false;
}

FailureOr<Value> scalarSplat(Value value) {
  if (!value)
    return failure();
  if (!isa<gpu::FragmentType>(value.getType()))
    return value;
  if (auto splat = value.getDefiningOp<gpu::SplatOp>())
    return splat.getValue();
  return failure();
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
    auto name = attrs.getAs<StringAttr>(gpu::abiNameAttr);
    auto view = cast<gpu::ViewType>(resource.getType());
    auto extent =
        cast<gpu::PhysicalExprAttr>(view.getLayout().getExtents()[axis]);
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
    return true;
  valid = stripBroadcast(valid);
  if (auto binary = valid.getDefiningOp<gpu::BinaryOp>()) {
    if (binary.getOperatorKind() != 11)
      return false;
    return isFullViewValidity(binary.getLhs(), resource, coordinates,
                              sourceAxes) &&
           isFullViewValidity(binary.getRhs(), resource, coordinates,
                              sourceAxes);
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

FailureOr<SmallVector<Value>> accessOffsets(Operation *owner,
                                            ValueRange coordinates,
                                            ArrayRef<int64_t> sourceAxes) {
  auto view = cast<gpu::ViewType>(owner->getOperand(0).getType());
  SmallVector<Value> offsets(view.getRank());
  for (auto [coordinate, sourceAxis] : llvm::zip(coordinates, sourceAxes)) {
    if (sourceAxis < 0 || sourceAxis >= static_cast<int64_t>(view.getRank()))
      return owner->emitOpError(
          "TileLang access source-axis mapping is outside the external view");
    auto range = coordinate.getDefiningOp<gpu::MakeRangeOp>();
    if (range)
      offsets[sourceAxis] = range.getStart();
    else if (!isa<gpu::FragmentType>(coordinate.getType()))
      offsets[sourceAxis] = coordinate;
    else
      return failure();
  }
  if (llvm::any_of(offsets, [](Value value) { return !value; }))
    return owner->emitOpError(
        "TileLang bulk copy requires one explicit offset per source axis");
  return offsets;
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

Operation *allocationAnchor(Operation *operation) {
  Operation *anchor = operation;
  for (Operation *parent = operation->getParentOp(); isa_and_nonnull<scf::ForOp>(parent);
       parent = parent->getParentOp())
    anchor = parent;
  return anchor;
}

class Bufferizer {
public:
  explicit Bufferizer(func::FuncOp kernel) : kernel(kernel) {}

  LogicalResult run() {
    SmallVector<gpu::LoadOp> loads;
    SmallVector<gpu::ContractOp> contracts;
    SmallVector<gpu::ReduceOp> reductions;
    SmallVector<gpu::ScanOp> scans;
    SmallVector<gpu::StoreOp> stores;
    kernel.walk([&](gpu::LoadOp op) { loads.push_back(op); });
    kernel.walk([&](gpu::ContractOp op) {
      contracts.push_back(op);
      directContractOperands.insert(op.getLhs());
      directContractOperands.insert(op.getRhs());
    });
    kernel.walk([&](gpu::ReduceOp op) { reductions.push_back(op); });
    kernel.walk([&](gpu::ScanOp op) { scans.push_back(op); });
    kernel.walk([&](gpu::StoreOp op) { stores.push_back(op); });

    for (gpu::LoadOp load : loads)
      if (failed(lowerLoad(load, directContractOperands.contains(load.getResult())
                                     ? 0u
                                     : 1u)))
        return failure();
    for (gpu::ContractOp contract : contracts)
      if (failed(lowerContract(contract)))
        return failure();
    for (gpu::ReduceOp reduce : reductions)
      if (failed(lowerReduce(reduce)))
        return failure();
    for (gpu::ScanOp scan : scans)
      if (failed(lowerScan(scan)))
        return failure();
    for (gpu::StoreOp store : stores)
      if (failed(lowerStore(store)))
        return failure();
    eraseLoweredOperations();
    if (failed(dropFragmentLoopCarries()) || failed(bufferizeScalarLoopCarries()))
      return failure();
    eraseLoweredOperations();
    eraseDeadPureValues();
    eraseLoweredOperations();
    eraseDeadPureValues();
    return success();
  }

private:
  struct LoopDrops {
    scf::ForOp loop;
    SmallVector<unsigned> indices;
  };

  FailureOr<Value> allocateFor(Value value, unsigned space, Operation *before,
                               ArrayAttr shape = {}) {
    auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
    if (!fragment)
      return failure();
    auto type = BufferType::get(kernel.getContext(), fragment.getElementType(),
                                shape ? shape : fragment.getShape(), space);
    OpBuilder builder(allocationAnchor(before));
    return builder.create<AllocOp>(before->getLoc(), type).getResult();
  }

  FailureOr<ParallelOp> createParallel(Operation *before,
                                       gpu::FragmentType fragment) {
    OpBuilder builder(before);
    OperationState state(before->getLoc(), ParallelOp::getOperationName());
    state.addAttribute("shape", fragment.getShape());
    state.addRegion();
    auto parallel = cast<ParallelOp>(builder.create(state));
    auto *body = new Block();
    for (size_t axis = 0; axis < fragment.getShape().size(); ++axis)
      body->addArgument(builder.getIndexType(), before->getLoc());
    parallel.getBody().push_back(body);
    return parallel;
  }

  FailureOr<SmallVector<Value>> indicesFor(
      OpBuilder &builder, gpu::FragmentType source,
      gpu::FragmentType target, ValueRange targetIndices, Operation *owner) {
    SmallVector<Value> result(source.getShape().size());
    for (Attribute sourceAttribute : source.getAxisMaps()) {
      auto sourceMap = cast<gpu::AxisMapAttr>(sourceAttribute);
      std::optional<unsigned> targetAxis;
      for (Attribute targetAttribute : target.getAxisMaps()) {
        auto targetMap = cast<gpu::AxisMapAttr>(targetAttribute);
        if (targetMap.getSourceId() == sourceMap.getSourceId() &&
            targetMap.getSourceAxis() == sourceMap.getSourceAxis()) {
          if (targetAxis)
            return owner->emitOpError(
                "TileLang scalarization found an ambiguous coordinate axis");
          targetAxis = targetMap.getFragmentAxis();
        }
      }
      if (targetAxis) {
        result[sourceMap.getFragmentAxis()] = targetIndices[*targetAxis];
        continue;
      }
      auto extent = cast<gpu::PhysicalExprAttr>(
          source.getShape()[sourceMap.getFragmentAxis()]);
      if (extent.getKind() !=
              static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
          extent.getValue() != 1)
        return owner->emitOpError(
            "TileLang scalarization cannot recover a missing non-unit physical axis");
      result[sourceMap.getFragmentAxis()] =
          builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
    }
    if (llvm::any_of(result, [](Value value) { return !value; }))
      return owner->emitOpError(
          "TileLang scalarization requires one typed map per fragment axis");
    return result;
  }

  FailureOr<SmallVector<unsigned>> copyInAxisOrder(
      gpu::LoadOp load, gpu::FragmentType fragment) {
    auto view = cast<gpu::ViewType>(load.getResource().getType());
    if (load.getCoordinates().size() != view.getRank() ||
        load.getSourceAxes().size() != view.getRank() ||
        fragment.getShape().size() != view.getRank())
      return failure();
    SmallVector<unsigned> bufferToFragment(view.getRank());
    SmallVector<bool> assigned(view.getRank(), false);
    for (auto [coordinate, viewAxis] :
         llvm::zip(load.getCoordinates(), load.getSourceAxes())) {
      auto coordinateType = dyn_cast<gpu::FragmentType>(coordinate.getType());
      if (!coordinateType || coordinateType.getShape().size() != 1 ||
          viewAxis < 0 || viewAxis >= static_cast<int64_t>(view.getRank()))
        return failure();
      auto coordinateMap =
          cast<gpu::AxisMapAttr>(coordinateType.getAxisMaps()[0]);
      std::optional<unsigned> fragmentAxis;
      for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
        auto mapping = cast<gpu::AxisMapAttr>(attribute);
        if (mapping.getSourceId() == coordinateMap.getSourceId() &&
            mapping.getSourceAxis() == coordinateMap.getSourceAxis()) {
          if (fragmentAxis)
            return failure();
          fragmentAxis = axis;
        }
      }
      if (!fragmentAxis || assigned[viewAxis])
        return failure();
      bufferToFragment[viewAxis] = *fragmentAxis;
      assigned[viewAxis] = true;
    }
    if (llvm::any_of(assigned, [](bool value) { return !value; }))
      return failure();
    return bufferToFragment;
  }

  SmallVector<unsigned> identityAxisOrder(unsigned rank) {
    SmallVector<unsigned> result(rank);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }

  FailureOr<Value> loadBufferElement(OpBuilder &builder, Value buffer,
                                     gpu::FragmentType source,
                                     gpu::FragmentType target,
                                     ValueRange targetIndices,
                                     Operation *owner) {
    FailureOr<SmallVector<Value>> indices =
        indicesFor(builder, source, target, targetIndices, owner);
    if (failed(indices))
      return failure();
    return builder
        .create<BufferLoadOp>(owner->getLoc(), source.getElementType(), buffer,
                              *indices)
        .getResult();
  }

  Value findBuffer(Value value) const {
    auto fragment = fragmentBuffers.find(value);
    if (fragment != fragmentBuffers.end())
      return fragment->second;
    auto shared = sharedBuffers.find(value);
    return shared == sharedBuffers.end() ? Value() : shared->second;
  }

  FailureOr<Value> scalarize(Value value, gpu::FragmentType target,
                             ValueRange targetIndices, OpBuilder &builder,
                             Operation *owner,
                             DenseMap<Value, Value> &memo) {
    if (!isa<gpu::FragmentType>(value.getType()))
      return value;
    if (Value cached = memo.lookup(value))
      return cached;
    auto source = cast<gpu::FragmentType>(value.getType());
    if (Value buffer = findBuffer(value)) {
      FailureOr<Value> loaded = loadBufferElement(
          builder, buffer, source, target, targetIndices, owner);
      if (succeeded(loaded))
        memo[value] = *loaded;
      return loaded;
    }
    Operation *producer = value.getDefiningOp();
    if (!producer)
      return owner->emitOpError(
          "TileLang scalarization encountered an unbufferized fragment argument");
    auto recurse = [&](Value operand) {
      return scalarize(operand, target, targetIndices, builder, owner, memo);
    };
    Value result;
    if (auto range = dyn_cast<gpu::MakeRangeOp>(producer)) {
      FailureOr<unsigned> targetAxis = failure();
      for (Attribute targetAttribute : target.getAxisMaps()) {
        auto mapping = cast<gpu::AxisMapAttr>(targetAttribute);
        if (mapping.getSourceId() == range.getSourceId() &&
            mapping.getSourceAxis() == range.getSourceAxis()) {
          if (succeeded(targetAxis))
            return owner->emitOpError(
                "range provenance maps to multiple TileLang loop axes");
          targetAxis = mapping.getFragmentAxis();
        }
      }
      if (failed(targetAxis))
        return owner->emitOpError(
            "range provenance is absent from the TileLang parallel shape");
      Value scaled = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getIndexType(), targetIndices[*targetAxis],
          range.getStep(), /*multiply=*/2);
      result = builder.create<gpu::BinaryOp>(
          owner->getLoc(), builder.getIndexType(), range.getStart(), scaled,
          /*add=*/0);
    } else if (auto splat = dyn_cast<gpu::SplatOp>(producer)) {
      result = splat.getValue();
    } else if (auto broadcast = dyn_cast<gpu::BroadcastOp>(producer)) {
      FailureOr<Value> input = recurse(broadcast.getValue());
      if (failed(input))
        return failure();
      result = *input;
    } else if (isa<gpu::ReshapeOp, gpu::TransposeOp>(producer)) {
      FailureOr<Value> input = recurse(producer->getOperand(0));
      if (failed(input))
        return failure();
      result = *input;
    } else if (auto join = dyn_cast<gpu::JoinOp>(producer)) {
      if (join.getAxis() >= targetIndices.size())
        return owner->emitOpError(
            "TileLang join axis is absent from the selected parallel shape");
      FailureOr<Value> lhs = recurse(join.getLhs());
      FailureOr<Value> rhs = recurse(join.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      Value zero =
          builder.create<arith::ConstantIndexOp>(owner->getLoc(), 0);
      Value first = builder.create<gpu::CompareOp>(
          owner->getLoc(), builder.getI1Type(), targetIndices[join.getAxis()],
          zero, /*equal=*/0);
      result = builder.create<gpu::SelectOp>(
          owner->getLoc(), source.getElementType(), first, *lhs, *rhs);
    } else if (auto unary = dyn_cast<gpu::UnaryOp>(producer)) {
      FailureOr<Value> input = recurse(unary.getInput());
      if (failed(input))
        return failure();
      result = builder.create<gpu::UnaryOp>(owner->getLoc(),
                                            source.getElementType(), *input,
                                            unary.getOperatorKind());
    } else if (auto binary = dyn_cast<gpu::BinaryOp>(producer)) {
      FailureOr<Value> lhs = recurse(binary.getLhs());
      FailureOr<Value> rhs = recurse(binary.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      result = builder.create<gpu::BinaryOp>(
          owner->getLoc(), source.getElementType(), *lhs, *rhs,
          binary.getOperatorKind());
    } else if (auto compare = dyn_cast<gpu::CompareOp>(producer)) {
      FailureOr<Value> lhs = recurse(compare.getLhs());
      FailureOr<Value> rhs = recurse(compare.getRhs());
      if (failed(lhs) || failed(rhs))
        return failure();
      result = builder.create<gpu::CompareOp>(
          owner->getLoc(), builder.getI1Type(), *lhs, *rhs,
          compare.getPredicate());
    } else if (auto select = dyn_cast<gpu::SelectOp>(producer)) {
      FailureOr<Value> condition = recurse(select.getCondition());
      FailureOr<Value> trueValue = recurse(select.getTrueValue());
      FailureOr<Value> falseValue = recurse(select.getFalseValue());
      if (failed(condition) || failed(trueValue) || failed(falseValue))
        return failure();
      result = builder.create<gpu::SelectOp>(
          owner->getLoc(), source.getElementType(), *condition, *trueValue,
          *falseValue);
    } else if (auto cast = dyn_cast<gpu::CastOp>(producer)) {
      FailureOr<Value> input = recurse(cast.getValue());
      if (failed(input))
        return failure();
      result = builder.create<gpu::CastOp>(owner->getLoc(),
                                           source.getElementType(), *input);
    } else if (auto bitcast = dyn_cast<gpu::BitcastOp>(producer)) {
      FailureOr<Value> input = recurse(bitcast.getValue());
      if (failed(input))
        return failure();
      result = builder.create<gpu::BitcastOp>(owner->getLoc(),
                                              source.getElementType(), *input);
    } else {
      return owner->emitOpError(
          "TileLang bufferization cannot scalarize this physical fragment producer");
    }
    memo[value] = result;
    return result;
  }

  LogicalResult lowerLoad(gpu::LoadOp load, unsigned space) {
    auto fragment = dyn_cast<gpu::FragmentType>(load.getResult().getType());
    auto view = dyn_cast<gpu::ViewType>(load.getResource().getType());
    if (!view)
      return load.emitOpError(
          "TileLang load bufferization requires an external view resource");
    if (!fragment) {
      FailureOr<SmallVector<Value>> indices =
          accessOffsets(load, load.getCoordinates(), load.getSourceAxes());
      if (failed(indices) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (load.getValid() && isa<gpu::FragmentType>(load.getValid().getType())) ||
          (load.getFill() && isa<gpu::FragmentType>(load.getFill().getType())))
        return load.emitOpError(
            "TileLang scalar view load requires scalar indices, validity and fill");
      OpBuilder builder(load);
      Value replacement = builder.create<ViewLoadOp>(
          load.getLoc(), load.getResult().getType(), load.getResource(), *indices,
          load.getValid(), load.getFill());
      load.getResult().replaceAllUsesWith(replacement);
      lowered.insert(load);
      return success();
    }
    FailureOr<SmallVector<Value>> offsets =
        accessOffsets(load, load.getCoordinates(), load.getSourceAxes());
    FailureOr<Value> fill = scalarSplat(load.getFill());
    FailureOr<SmallVector<unsigned>> copyOrder =
        copyInAxisOrder(load, fragment);
    bool directCopy = succeeded(offsets) && succeeded(copyOrder) &&
                      isFullViewValidity(load.getValid(), load.getResource(),
                                         load.getCoordinates(),
                                         load.getSourceAxes()) &&
                      (!load.getFill() || (succeeded(fill) && isZero(*fill)));
    SmallVector<Attribute> allocationShape;
    if (directCopy)
      for (unsigned fragmentAxis : *copyOrder)
        allocationShape.push_back(fragment.getShape()[fragmentAxis]);
    FailureOr<Value> allocation = allocateFor(
        load.getResult(), space, load,
        allocationShape.empty()
            ? ArrayAttr()
            : ArrayAttr::get(kernel.getContext(), allocationShape));
    if (failed(allocation))
      return failure();
    Value destination = *allocation;
    if (directCopy) {
      OpBuilder builder(load);
      builder.create<CopyInOp>(load.getLoc(), load.getResource(), *offsets,
                               destination);
      if (space == 0)
        sharedBufferAxes[load.getResult()] = *copyOrder;
    } else {
      FailureOr<ParallelOp> parallel = createParallel(load, fragment);
      if (failed(parallel))
        return failure();
      Block &body = parallel->getBody().front();
      OpBuilder builder = OpBuilder::atBlockBegin(&body);
      DenseMap<Value, Value> memo;
      SmallVector<Value> indices(view.getRank());
      for (auto [coordinate, sourceAxis] :
           llvm::zip(load.getCoordinates(), load.getSourceAxes())) {
        FailureOr<Value> scalar = scalarize(
            coordinate, fragment, body.getArguments(), builder, load, memo);
        if (failed(scalar))
          return failure();
        indices[sourceAxis] = *scalar;
      }
      if (llvm::any_of(indices, [](Value value) { return !value; }))
        return load.emitOpError(
            "TileLang view load source-axis mapping is incomplete");
      Value valid;
      Value padding;
      if (load.getValid()) {
        FailureOr<Value> scalar = scalarize(
            load.getValid(), fragment, body.getArguments(), builder, load, memo);
        FailureOr<Value> scalarFill = scalarize(
            load.getFill(), fragment, body.getArguments(), builder, load, memo);
        if (failed(scalar) || failed(scalarFill))
          return failure();
        valid = *scalar;
        padding = *scalarFill;
      }
      Value loaded = builder.create<ViewLoadOp>(
          load.getLoc(), fragment.getElementType(), load.getResource(), indices,
          valid, padding);
      builder.create<BufferStoreOp>(load.getLoc(), destination,
                                    body.getArguments(), loaded);
      builder.create<YieldOp>(load.getLoc());
      if (space == 0)
        sharedBufferAxes[load.getResult()] =
            identityAxisOrder(fragment.getShape().size());
    }
    (space == 0 ? sharedBuffers : fragmentBuffers)[load.getResult()] =
        destination;
    lowered.insert(load);
    return success();
  }

  FailureOr<Value> materialize(Value value, unsigned space, Operation *owner) {
    auto fragment = dyn_cast<gpu::FragmentType>(value.getType());
    if (!fragment)
      return failure();
    DenseMap<Value, Value> &selected =
        space == 0 ? sharedBuffers : fragmentBuffers;
    if (Value existing = selected.lookup(value))
      return existing;
    FailureOr<Value> allocation = allocateFor(value, space, owner);
    if (failed(allocation))
      return failure();
    FailureOr<ParallelOp> parallel = createParallel(owner, fragment);
    if (failed(parallel))
      return failure();
    Block &body = parallel->getBody().front();
    OpBuilder builder = OpBuilder::atBlockBegin(&body);
    DenseMap<Value, Value> memo;
    FailureOr<Value> scalar = scalarize(value, fragment, body.getArguments(),
                                        builder, owner, memo);
    if (failed(scalar))
      return failure();
    builder.create<BufferStoreOp>(owner->getLoc(), *allocation,
                                  body.getArguments(), *scalar);
    builder.create<YieldOp>(owner->getLoc());
    selected[value] = *allocation;
    if (space == 0)
      sharedBufferAxes[value] = identityAxisOrder(fragment.getShape().size());
    if (Operation *producer = value.getDefiningOp())
      if (!isa<gpu::LoadOp>(producer))
        lowered.insert(producer);
    return *allocation;
  }

  FailureOr<Value> contractAccumulator(gpu::ContractOp contract) {
    Value accumulator = contract.getAccumulator();
    if (Value existing = fragmentBuffers.lookup(accumulator))
      return existing;
    auto argument = dyn_cast<BlockArgument>(accumulator);
    auto loop = argument
                    ? dyn_cast_or_null<scf::ForOp>(
                          argument.getOwner()->getParentOp())
                    : scf::ForOp();
    if (loop && argument != loop.getInductionVar()) {
      unsigned index = argument.getArgNumber() - 1;
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      if (index >= loop.getInitArgs().size() ||
          yield.getOperand(index) != contract.getResult())
        return contract.emitOpError(
            "TileLang contract accumulator is not the loop-carried physical result");
      FailureOr<Value> initial = scalarSplat(loop.getInitArgs()[index]);
      if (failed(initial))
        return contract.emitOpError(
            "TileLang native GEMM requires a scalar-filled physical accumulator");
      FailureOr<Value> allocation =
          allocateFor(contract.getResult(), 1, loop);
      if (failed(allocation))
        return failure();
      OpBuilder builder(loop);
      builder.create<FillOp>(contract.getLoc(), *allocation, *initial);
      fragmentBuffers[accumulator] = *allocation;
      fragmentBuffers[contract.getResult()] = *allocation;
      fragmentBuffers[loop.getResult(index)] = *allocation;
      loopDrops[loop.getOperation()].push_back(index);
      return *allocation;
    }
    FailureOr<Value> initial = scalarSplat(accumulator);
    if (succeeded(initial)) {
      FailureOr<Value> allocation =
          allocateFor(contract.getResult(), 1, contract);
      if (failed(allocation))
        return failure();
      OpBuilder builder(contract);
      builder.create<FillOp>(contract.getLoc(), *allocation, *initial);
      fragmentBuffers[accumulator] = *allocation;
      fragmentBuffers[contract.getResult()] = *allocation;
      return *allocation;
    }
    return materialize(accumulator, 1, contract);
  }

  LogicalResult lowerContract(gpu::ContractOp contract) {
    if (contract.getLhsReductionAxes() != ArrayRef<int64_t>{1} ||
        contract.getRhsReductionAxes() != ArrayRef<int64_t>{0} ||
        !contract.getLhsBatchAxes().empty() ||
        !contract.getRhsBatchAxes().empty())
      return contract.emitOpError(
          "TileLang native GEMM requires [M,K] x [K,N] physical axes");
    FailureOr<Value> lhs = materialize(contract.getLhs(), 0, contract);
    FailureOr<Value> rhs = materialize(contract.getRhs(), 0, contract);
    FailureOr<Value> accumulator = contractAccumulator(contract);
    if (failed(lhs) || failed(rhs) || failed(accumulator))
      return failure();
    auto transpose = [&](Value operand) -> FailureOr<bool> {
      auto fragment = cast<gpu::FragmentType>(operand.getType());
      SmallVector<unsigned> order = sharedBufferAxes.lookup(operand);
      if (order.empty())
        order = identityAxisOrder(fragment.getShape().size());
      if (order.size() == 2 && order[0] == 0 && order[1] == 1)
        return false;
      if (order.size() == 2 && order[0] == 1 && order[1] == 0)
        return true;
      return contract.emitOpError(
          "TileLang GEMM operand has no native two-axis storage form");
    };
    FailureOr<bool> transposeLhs = transpose(contract.getLhs());
    FailureOr<bool> transposeRhs = transpose(contract.getRhs());
    if (failed(transposeLhs) || failed(transposeRhs))
      return failure();
    OpBuilder builder(contract);
    builder.create<GemmOp>(contract.getLoc(), *lhs, *rhs, *accumulator,
                           *transposeLhs, *transposeRhs);
    fragmentBuffers[contract.getResult()] = *accumulator;
    lowered.insert(contract);
    return success();
  }

  SmallVector<Attribute> reducedShape(gpu::FragmentType source,
                                      unsigned axis) {
    SmallVector<Attribute> result;
    for (auto [index, extent] : llvm::enumerate(source.getShape()))
      if (index != axis)
        result.push_back(extent);
    if (result.empty())
      result.push_back(constantExtent(kernel.getContext(), 1));
    return result;
  }

  LogicalResult lowerReduce(gpu::ReduceOp reduce) {
    std::optional<uint64_t> kind = nativeCombineKind(reduce.getCombine());
    if (reduce.getSourceCount() != 1 || reduce.getIdentityCount() != 1 ||
        reduce.getCaptureCount() != 0 || reduce.getAxes().size() != 1 ||
        reduce.getNumResults() != 1 || !kind)
      return reduce.emitOpError(
          "TileLang native reduce requires one source/identity/axis and builtin add/max/min combine");
    auto sourceType =
        dyn_cast<gpu::FragmentType>(reduce.getInputs().front().getType());
    if (!sourceType)
      return reduce.emitOpError("TileLang native reduce source must be a fragment");
    FailureOr<Value> source =
        materialize(reduce.getInputs().front(), 1, reduce);
    FailureOr<Value> identity =
        scalarSplat(reduce.getInputs()[reduce.getSourceCount()]);
    if (failed(source) || failed(identity))
      return reduce.emitOpError(
          "TileLang native reduce identity must be an explicit scalar/splat");
    auto destinationType = BufferType::get(
        kernel.getContext(), sourceType.getElementType(),
        ArrayAttr::get(kernel.getContext(),
                       reducedShape(sourceType, reduce.getAxes().front())),
        1);
    OpBuilder allocationBuilder(allocationAnchor(reduce));
    Value destination =
        allocationBuilder.create<AllocOp>(reduce.getLoc(), destinationType);
    OpBuilder builder(reduce);
    builder.create<FillOp>(reduce.getLoc(), destination, *identity);
    builder.create<ReduceOp>(reduce.getLoc(), *source, destination,
                             reduce.getAxes().front(), *kind);
    Type resultType = reduce.getResultTypes().front();
    if (auto fragment = dyn_cast<gpu::FragmentType>(resultType)) {
      fragmentBuffers[reduce.getResult(0)] = destination;
    } else {
      SmallVector<Value> indices(destinationType.getShape().size());
      for (Value &index : indices)
        index = builder.create<arith::ConstantIndexOp>(reduce.getLoc(), 0);
      Value loaded = builder.create<BufferLoadOp>(reduce.getLoc(), resultType,
                                                  destination, indices);
      reduce.getResult(0).replaceAllUsesWith(loaded);
    }
    lowered.insert(reduce);
    return success();
  }

  LogicalResult lowerScan(gpu::ScanOp scan) {
    std::optional<uint64_t> kind = nativeCombineKind(scan.getCombine());
    if (scan.getSourceCount() != 1 || scan.getIdentityCount() != 1 ||
        scan.getCaptureCount() != 0 || scan.getNumResults() != 1 || !kind ||
        (*kind != 0 && *kind != 1) || !scan.getInclusive())
      return scan.emitOpError(
          "TileLang native scan requires one inclusive builtin add/max component");
    auto sourceType =
        dyn_cast<gpu::FragmentType>(scan.getInputs().front().getType());
    if (!sourceType)
      return scan.emitOpError("TileLang native scan source must be a fragment");
    FailureOr<Value> source = materialize(scan.getInputs().front(), 1, scan);
    if (failed(source))
      return failure();
    auto destinationType = BufferType::get(
        kernel.getContext(), sourceType.getElementType(), sourceType.getShape(),
        1);
    OpBuilder allocationBuilder(allocationAnchor(scan));
    Value destination =
        allocationBuilder.create<AllocOp>(scan.getLoc(), destinationType);
    OpBuilder builder(scan);
    builder.create<ScanOp>(scan.getLoc(), *source, destination, scan.getAxis(),
                           *kind, scan.getReverse());
    fragmentBuffers[scan.getResult(0)] = destination;
    lowered.insert(scan);
    return success();
  }

  LogicalResult lowerStore(gpu::StoreOp store) {
    if (store.getCollision() != 0)
      return store.emitOpError(
          "TileLang unique-store lowering cannot realize a collision operation");
    auto fragment = dyn_cast<gpu::FragmentType>(store.getValue().getType());
    auto view = dyn_cast<gpu::ViewType>(store.getResource().getType());
    if (!view)
      return store.emitOpError(
          "TileLang store bufferization requires an external view");
    if (!fragment) {
      FailureOr<SmallVector<Value>> indices =
          accessOffsets(store, store.getCoordinates(), store.getSourceAxes());
      if (failed(indices) ||
          llvm::any_of(*indices, [](Value value) {
            return isa<gpu::FragmentType>(value.getType());
          }) ||
          (store.getValid() && isa<gpu::FragmentType>(store.getValid().getType())))
        return store.emitOpError(
            "TileLang scalar view store requires scalar indices and validity");
      OpBuilder builder(store);
      builder.create<ViewStoreOp>(store.getLoc(), store.getResource(), *indices,
                                  store.getValue(), store.getValid());
      lowered.insert(store);
      return success();
    }
    FailureOr<SmallVector<Value>> offsets =
        accessOffsets(store, store.getCoordinates(), store.getSourceAxes());
    if (succeeded(offsets) &&
        isFullViewValidity(store.getValid(), store.getResource(),
                           store.getCoordinates(), store.getSourceAxes())) {
      OpBuilder builder(store);
      if (auto cast = store.getValue().getDefiningOp<gpu::CastOp>()) {
        auto input = dyn_cast<gpu::FragmentType>(cast.getValue().getType());
        auto result = dyn_cast<gpu::FragmentType>(cast.getResult().getType());
        Value source = findBuffer(cast.getValue());
        if (input && result && source && input.getShape() == result.getShape() &&
            input.getAxisMaps() == result.getAxisMaps() &&
            input.getValidity() == result.getValidity() &&
            input.getOwner() == result.getOwner()) {
          builder.create<CastCopyOutOp>(store.getLoc(), source,
                                        store.getResource(), *offsets);
          lowered.insert(cast);
          lowered.insert(store);
          return success();
        }
      }
      FailureOr<Value> source = materialize(store.getValue(), 1, store);
      if (failed(source))
        return failure();
      builder.create<CopyOutOp>(store.getLoc(), *source, store.getResource(),
                                *offsets);
    } else {
      FailureOr<Value> source = materialize(store.getValue(), 1, store);
      if (failed(source))
        return failure();
      FailureOr<ParallelOp> parallel = createParallel(store, fragment);
      if (failed(parallel))
        return failure();
      Block &body = parallel->getBody().front();
      OpBuilder builder = OpBuilder::atBlockBegin(&body);
      DenseMap<Value, Value> memo;
      FailureOr<Value> value = loadBufferElement(
          builder, *source, fragment, fragment, body.getArguments(), store);
      if (failed(value))
        return failure();
      SmallVector<Value> indices(view.getRank());
      for (auto [coordinate, sourceAxis] :
           llvm::zip(store.getCoordinates(), store.getSourceAxes())) {
        FailureOr<Value> scalar = scalarize(
            coordinate, fragment, body.getArguments(), builder, store, memo);
        if (failed(scalar))
          return failure();
        indices[sourceAxis] = *scalar;
      }
      if (llvm::any_of(indices, [](Value value) { return !value; }))
        return store.emitOpError(
            "TileLang view store source-axis mapping is incomplete");
      Value valid;
      if (store.getValid()) {
        FailureOr<Value> scalar = scalarize(
            store.getValid(), fragment, body.getArguments(), builder, store,
            memo);
        if (failed(scalar))
          return failure();
        valid = *scalar;
      }
      builder.create<ViewStoreOp>(store.getLoc(), store.getResource(), indices,
                                   *value, valid);
      builder.create<YieldOp>(store.getLoc());
    }
    lowered.insert(store);
    return success();
  }

  LogicalResult dropFragmentLoopCarries() {
    SmallVector<std::pair<scf::ForOp, SmallVector<unsigned>>> work;
    for (auto &item : loopDrops) {
      auto loop = dyn_cast<scf::ForOp>(item.first);
      if (!loop)
        return failure();
      llvm::sort(item.second);
      item.second.erase(std::unique(item.second.begin(), item.second.end()),
                        item.second.end());
      work.push_back({loop, item.second});
    }
    auto depth = [](Operation *operation) {
      unsigned result = 0;
      for (Operation *parent = operation->getParentOp(); parent;
           parent = parent->getParentOp())
        ++result;
      return result;
    };
    llvm::sort(work, [&](const auto &lhs, const auto &rhs) {
      return depth(lhs.first) > depth(rhs.first);
    });
    for (auto &entry : work) {
      scf::ForOp loop = entry.first;
      llvm::DenseSet<unsigned> dropped(entry.second.begin(), entry.second.end());
      SmallVector<Value> initial;
      for (auto [index, value] : llvm::enumerate(loop.getInitArgs()))
        if (!dropped.contains(index))
          initial.push_back(value);
      OpBuilder builder(loop);
      auto replacement = builder.create<scf::ForOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep(), initial);
      if (Attribute origin = loop->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      Block *oldBody = loop.getBody();
      Block *newBody = replacement.getBody();
      loop.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
      unsigned next = 0;
      for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
        if (dropped.contains(index))
          continue;
        argument.replaceAllUsesWith(replacement.getRegionIterArgs()[next++]);
      }
      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      SmallVector<Value> yielded;
      for (auto [index, value] : llvm::enumerate(oldYield.getOperands()))
        if (!dropped.contains(index))
          yielded.push_back(value);
      replacement.getBody()->getTerminator()->setOperands(yielded);
      Operation *newTerminator = newBody->getTerminator();
      for (Operation &operation : llvm::make_early_inc_range(
               oldBody->without_terminator()))
        if (!lowered.contains(&operation))
          operation.moveBefore(newTerminator);
      next = 0;
      for (auto [index, result] : llvm::enumerate(loop.getResults())) {
        if (dropped.contains(index)) {
          if (!result.use_empty())
            return loop.emitOpError(
                "dropped TileLang fragment loop result still has executable uses");
          continue;
        }
        result.replaceAllUsesWith(replacement.getResult(next++));
      }
      loop.walk([&](Operation *operation) { lowered.erase(operation); });
      loop.erase();
    }
    return success();
  }

  LogicalResult bufferizeScalarLoopCarries() {
    SmallVector<scf::ForOp> loops;
    kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
      if (loop.getNumRegionIterArgs())
        loops.push_back(loop);
    });
    for (scf::ForOp loop : loops) {
      if (!loop || loop.getNumRegionIterArgs() == 0)
        continue;
      for (Value argument : loop.getRegionIterArgs())
        if (!isa<IntegerType, IndexType, FloatType>(argument.getType()))
          return loop.emitOpError(
              "TileLang loop carries a non-scalar value without explicit bufferization");
      SmallVector<Value> buffers;
      OpBuilder before(loop);
      for (Value initial : loop.getInitArgs()) {
        auto type = BufferType::get(
            kernel.getContext(), initial.getType(),
            before.getArrayAttr({constantExtent(kernel.getContext(), 1)}), 1);
        Value buffer = before.create<AllocOp>(loop.getLoc(), type);
        before.create<FillOp>(loop.getLoc(), buffer, initial);
        buffers.push_back(buffer);
      }
      auto replacement = before.create<scf::ForOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep());
      if (Attribute origin = loop->getAttr(gpu::originAttr))
        replacement->setAttr(gpu::originAttr, origin);
      Block *oldBody = loop.getBody();
      Block *newBody = replacement.getBody();
      loop.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newBody);
      Value zero =
          bodyBuilder.create<arith::ConstantIndexOp>(loop.getLoc(), 0);
      for (auto [argument, buffer] :
           llvm::zip(loop.getRegionIterArgs(), buffers)) {
        Value loaded = bodyBuilder.create<BufferLoadOp>(
            loop.getLoc(), argument.getType(), buffer, ValueRange{zero});
        argument.replaceAllUsesWith(loaded);
      }
      Operation *newTerminator = newBody->getTerminator();
      for (Operation &operation : llvm::make_early_inc_range(
               oldBody->without_terminator()))
        operation.moveBefore(newTerminator);
      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      OpBuilder yieldBuilder(newTerminator);
      for (auto [buffer, value] : llvm::zip(buffers, oldYield.getOperands()))
        yieldBuilder.create<BufferStoreOp>(loop.getLoc(), buffer,
                                           ValueRange{zero}, value);
      OpBuilder after(replacement);
      after.setInsertionPointAfter(replacement);
      Value resultZero =
          after.create<arith::ConstantIndexOp>(loop.getLoc(), 0);
      for (auto [result, buffer] : llvm::zip(loop.getResults(), buffers)) {
        Value loaded = after.create<BufferLoadOp>(
            loop.getLoc(), result.getType(), buffer, ValueRange{resultZero});
        result.replaceAllUsesWith(loaded);
      }
      loop.walk([&](Operation *operation) { lowered.erase(operation); });
      loop.erase();
    }
    return success();
  }

  void eraseLoweredOperations() {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> work;
      kernel.walk<WalkOrder::PostOrder>([&](Operation *operation) {
        if (lowered.contains(operation) &&
            llvm::all_of(operation->getResults(),
                         [](Value result) { return result.use_empty(); }))
          work.push_back(operation);
      });
      for (Operation *operation : work) {
        lowered.erase(operation);
        operation->erase();
        changed = true;
      }
    }
  }

  void eraseDeadPureValues() {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> dead;
      kernel.walk([&](Operation *operation) {
        StringRef dialect = operation->getName().getDialectNamespace();
        if (dialect != "intent_gpu" || operation->getNumResults() == 0 ||
            !llvm::all_of(operation->getResults(),
                          [](Value result) { return result.use_empty(); }) ||
            !isMemoryEffectFree(operation) ||
            isa<gpu::ParameterOp, gpu::ProgramIdOp>(operation))
          return;
        dead.push_back(operation);
      });
      for (Operation *operation : llvm::reverse(dead))
        if (operation->getBlock() &&
            llvm::all_of(operation->getResults(),
                         [](Value result) { return result.use_empty(); })) {
          operation->erase();
          changed = true;
        }
    }
  }

  func::FuncOp kernel;
  DenseMap<Value, Value> sharedBuffers;
  DenseMap<Value, SmallVector<unsigned>> sharedBufferAxes;
  DenseMap<Value, Value> fragmentBuffers;
  llvm::DenseSet<Value> directContractOperands;
  DenseMap<Operation *, SmallVector<unsigned>> loopDrops;
  llvm::DenseSet<Operation *> lowered;
};

} // namespace

LogicalResult bufferizeGPUProgram(func::FuncOp kernel) {
  return Bufferizer(kernel).run();
}

} // namespace intent::tilelang

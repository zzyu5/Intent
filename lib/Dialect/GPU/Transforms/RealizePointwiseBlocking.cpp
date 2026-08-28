#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include <limits>
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

constexpr uint64_t staticAxisMask = uint64_t{1} << 63;

bool isStaticAxis(uint64_t axis) { return (axis & staticAxisMask) != 0; }

uint64_t staticAxis(uint64_t sourceId) {
  return sourceId | staticAxisMask;
}

PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {}) {
  return PhysicalExprAttr::get(context, static_cast<uint32_t>(kind), value,
                               StringAttr::get(context, symbol),
                               ArrayAttr::get(context, {}));
}

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs) {
  return PhysicalExprAttr::get(
      context, static_cast<uint32_t>(kind), 0, StringAttr::get(context),
      ArrayAttr::get(context, {lhs, rhs}));
}

bool hasCompileTimeExtent(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() ||
         value.getDefiningOp<ParameterOp>() ||
         value.getDefiningOp<PhysicalExprOp>();
}

FailureOr<Value> dimensionArgument(func::FuncOp kernel, uint64_t dimension) {
  for (BlockArgument argument : kernel.getArguments()) {
    auto attributes = kernel.getArgAttrDict(argument.getArgNumber());
    auto kind = attributes.getAs<StringAttr>(abiKindAttr);
    auto identity = attributes.getAs<IntegerAttr>(dimensionAttr);
    if (kind && kind.getValue() == "dimension" && identity &&
        identity.getInt() == static_cast<int64_t>(dimension))
      return Value(argument);
  }
  return failure();
}

FailureOr<ParameterOp> blockingParameter(func::FuncOp kernel,
                                         MakeRangeOp range) {
  auto fragment = dyn_cast<FragmentType>(range.getResult().getType());
  auto extent = fragment && fragment.getShape().size() == 1
                    ? dyn_cast<PhysicalExprAttr>(fragment.getShape()[0])
                    : PhysicalExprAttr();
  std::string name;
  if (range->hasAttr(sourceSubregionAttr))
    name = ("FRAGMENT_S" + Twine(range.getSourceId())).str();
  else if (extent &&
      extent.getKind() ==
          static_cast<uint32_t>(PhysicalExprKind::Parameter))
    name = extent.getSymbol().getValue().str();
  else if (range->hasAttr(worksetCoordinateRangeAttr)) {
    auto dimension = range->getAttrOfType<IntegerAttr>(sourceDimensionAttr);
    if (!dimension)
      return failure();
    name = succeeded(dimensionArgument(kernel, dimension.getInt()))
               ? ("FRAGMENT_D" + Twine(dimension.getInt())).str()
               : ("FRAGMENT_S" + Twine(range.getSourceId())).str();
  }
  else if (range.getExtent().getDefiningOp<arith::ConstantIndexOp>())
    name = ("FRAGMENT_S" + Twine(range.getSourceId())).str();
  else if (auto dimension =
               range->getAttrOfType<IntegerAttr>(sourceDimensionAttr))
    name = ("FRAGMENT_D" + Twine(dimension.getInt())).str();
  else
    name = ("FRAGMENT_D" + Twine(range.getSourceId())).str();
  ParameterOp result;
  kernel.walk([&](ParameterOp parameter) {
    if (!result && parameter.getParameter().getName() == name)
      result = parameter;
  });
  return result ? FailureOr<ParameterOp>(result)
                : FailureOr<ParameterOp>(failure());
}

FailureOr<uint64_t> parameterDimension(ParameterOp parameter) {
  StringRef name = parameter.getParameter().getName().getValue();
  bool fixed = name.consume_front("FRAGMENT_S");
  if (!fixed && !name.consume_front("FRAGMENT_D"))
    return failure();
  uint64_t dimension = 0;
  if (name.getAsInteger(10, dimension))
    return failure();
  return fixed ? FailureOr<uint64_t>(staticAxis(dimension))
               : FailureOr<uint64_t>(dimension);
}

PhysicalExprAttr fragmentExtent(ParameterOp parameter) {
  ParameterAttr schema = parameter.getParameter();
  StringRef name = schema.getName().getValue();
  if (name.starts_with("FRAGMENT_S") && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    name);
}

struct ProductConstraint {
  int64_t constant = 1;
  StringAttr parameter;
  unsigned parameterCount = 0;
};

PhysicalExprAttr replaceParameter(PhysicalExprAttr current, StringAttr name,
                                  PhysicalExprAttr replacement) {
  auto kind = static_cast<PhysicalExprKind>(current.getKind());
  if (kind == PhysicalExprKind::Parameter && current.getSymbol() == name)
    return replacement;
  SmallVector<Attribute> operands;
  bool changed = false;
  for (Attribute operand : current.getOperands()) {
    auto rewritten = replaceParameter(cast<PhysicalExprAttr>(operand), name,
                                      replacement);
    operands.push_back(rewritten);
    changed |= rewritten != operand;
  }
  if (!changed)
    return current;
  return PhysicalExprAttr::get(current.getContext(), current.getKind(),
                               current.getValue(), current.getSymbol(),
                               ArrayAttr::get(current.getContext(), operands));
}

Attribute replaceParameter(Attribute current, StringAttr name,
                           PhysicalExprAttr replacement) {
  if (auto expression = dyn_cast<PhysicalExprAttr>(current))
    return replaceParameter(expression, name, replacement);
  if (auto array = dyn_cast<ArrayAttr>(current)) {
    SmallVector<Attribute> values;
    bool changed = false;
    for (Attribute value : array) {
      Attribute rewritten = replaceParameter(value, name, replacement);
      values.push_back(rewritten);
      changed |= rewritten != value;
    }
    return changed ? Attribute(ArrayAttr::get(current.getContext(), values))
                   : current;
  }
  if (auto dictionary = dyn_cast<DictionaryAttr>(current)) {
    SmallVector<NamedAttribute> values;
    bool changed = false;
    for (NamedAttribute value : dictionary) {
      Attribute rewritten =
          replaceParameter(value.getValue(), name, replacement);
      values.emplace_back(value.getName(), rewritten);
      changed |= rewritten != value.getValue();
    }
    return changed
               ? Attribute(DictionaryAttr::get(current.getContext(), values))
               : current;
  }
  return current;
}

void replaceParameterAttributes(Operation *operation, StringAttr name,
                                PhysicalExprAttr replacement) {
  SmallVector<NamedAttribute> attributes;
  bool changed = false;
  for (NamedAttribute attribute : operation->getAttrs()) {
    Attribute rewritten =
        replaceParameter(attribute.getValue(), name, replacement);
    attributes.emplace_back(attribute.getName(), rewritten);
    changed |= rewritten != attribute.getValue();
  }
  if (changed)
    operation->setAttrs(DictionaryAttr::get(operation->getContext(), attributes));
}

bool collectProductConstraint(PhysicalExprAttr extent,
                              ProductConstraint &constraint) {
  auto kind = static_cast<PhysicalExprKind>(extent.getKind());
  if (kind == PhysicalExprKind::Constant) {
    if (extent.getValue() <= 0)
      return false;
    if (constraint.constant >
        std::numeric_limits<int64_t>::max() / extent.getValue())
      return false;
    constraint.constant *= extent.getValue();
    return true;
  }
  if (kind == PhysicalExprKind::Parameter) {
    ++constraint.parameterCount;
    if (constraint.parameter && constraint.parameter != extent.getSymbol())
      return false;
    constraint.parameter = extent.getSymbol();
    return true;
  }
  if (kind != PhysicalExprKind::Multiply || extent.getOperands().size() != 2)
    return false;
  return collectProductConstraint(
             cast<PhysicalExprAttr>(extent.getOperands()[0]), constraint) &&
         collectProductConstraint(
             cast<PhysicalExprAttr>(extent.getOperands()[1]), constraint);
}

bool collectProductConstraint(FragmentType fragment,
                              ProductConstraint &constraint) {
  for (Attribute extent : fragment.getShape())
    if (!collectProductConstraint(cast<PhysicalExprAttr>(extent), constraint))
      return false;
  return true;
}

LogicalResult bindStructurallyRequiredStaticFragments(func::FuncOp kernel) {
  llvm::MapVector<ParameterOp, int64_t> required;
  SmallVector<ReshapeOp> reshapes;
  kernel.walk([&](ReshapeOp reshape) { reshapes.push_back(reshape); });
  for (ReshapeOp reshape : reshapes) {
    auto source = dyn_cast<FragmentType>(reshape.getValue().getType());
    auto result = dyn_cast<FragmentType>(reshape.getResult().getType());
    if (!source || !result)
      continue;
    ProductConstraint sourceProduct;
    ProductConstraint resultProduct;
    if (!collectProductConstraint(source, sourceProduct) ||
        !collectProductConstraint(result, resultProduct) ||
        sourceProduct.parameterCount != 1 ||
        resultProduct.parameterCount != 0 ||
        resultProduct.constant % sourceProduct.constant != 0)
      continue;
    StringRef name = sourceProduct.parameter.getValue();
    if (!name.starts_with("FRAGMENT_S"))
      continue;
    ParameterOp parameter;
    unsigned matches = 0;
    kernel.walk([&](ParameterOp candidate) {
      if (candidate.getParameter().getName() == sourceProduct.parameter) {
        parameter = candidate;
        ++matches;
      }
    });
    if (matches == 0)
      continue;
    if (matches != 1)
      return reshape.emitOpError(
          "reshape structural extent names a non-unique physical parameter");
    int64_t candidate = resultProduct.constant / sourceProduct.constant;
    if (!llvm::is_contained(
            parameter.getParameter().getCandidates().asArrayRef(), candidate))
      return reshape.emitOpError(
          "reshape requires a static fragment extent outside its legal domain");
    auto found = required.find(parameter);
    if (found != required.end() && found->second != candidate)
      return parameter.emitOpError(
          "one static fragment has incompatible structural extent requirements");
    required[parameter] = candidate;
  }

  for (auto [parameter, candidate] : required) {
    StringAttr parameterName = parameter.getParameter().getName();
    StringRef suffix = parameter.getParameter().getName().getValue();
    if (!suffix.consume_front("FRAGMENT_S"))
      continue;
    uint64_t sourceId = 0;
    if (suffix.getAsInteger(10, sourceId))
      return failure();
    PhysicalExprAttr fixedExtent =
        expression(kernel.getContext(), PhysicalExprKind::Constant, candidate);
    replaceParameterAttributes(kernel.getOperation(), parameterName,
                               fixedExtent);
    kernel.walk([&](Operation *operation) {
      replaceParameterAttributes(operation, parameterName, fixedExtent);
    });
    SmallVector<Value> fragmentRoots;
    kernel.walk([&](Operation *operation) {
      fragmentRoots.append(operation->getResults().begin(),
                           operation->getResults().end());
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          fragmentRoots.append(block.getArguments().begin(),
                               block.getArguments().end());
    });
    for (Value root : fragmentRoots)
      retargetSourceExtent(root, sourceId, fixedExtent);
    SmallVector<MakeRangeOp> ranges;
    kernel.walk([&](MakeRangeOp range) {
      if (range.getSourceId() == sourceId)
        ranges.push_back(range);
    });
    for (MakeRangeOp range : ranges) {
      retargetSourceExtent(range.getResult(), sourceId, fixedExtent);
      OpBuilder builder(range);
      Value fixed =
          builder.create<arith::ConstantIndexOp>(range.getLoc(), candidate);
      range->setOperand(1, fixed);
    }
    OpBuilder builder(parameter);
    Value fixed =
        builder.create<arith::ConstantIndexOp>(parameter.getLoc(), candidate);
    parameter.getResult().replaceAllUsesWith(fixed);
    parameter.erase();
  }
  return success();
}

LogicalResult requireFullDimensionCoverage(func::FuncOp kernel, Value source,
                                           uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  if (!extent || extent.getKind() ==
                     static_cast<uint32_t>(PhysicalExprKind::Constant))
    return success();
  if (extent.getKind() !=
      static_cast<uint32_t>(PhysicalExprKind::Parameter))
    return failure();
  ParameterOp parameter;
  kernel.walk([&](ParameterOp candidate) {
    if (candidate.getParameter().getName() == extent.getSymbol())
      parameter = candidate;
  });
  FailureOr<uint64_t> dimension =
      parameter ? parameterDimension(parameter)
                : FailureOr<uint64_t>(failure());
  if (failed(dimension) || failed(dimensionArgument(kernel, *dimension)))
    return failure();
  if (auto covered =
          parameter->getAttrOfType<IntegerAttr>(coverageDimensionAttr)) {
    if (covered.getInt() != static_cast<int64_t>(*dimension))
      return parameter.emitOpError(
          "one physical parameter covers multiple logical dimensions");
    return bindFullCoverageDimension(kernel, *dimension,
                                     parameter.getResult());
  }
  static constexpr int64_t candidates[] = {
      64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  ParameterAttr schema = parameter.getParameter();
  parameter->setAttr(
      "parameter",
      ParameterAttr::get(kernel.getContext(), schema.getName(), schema.getRole(),
                         DenseI64ArrayAttr::get(kernel.getContext(), candidates)));
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), *dimension));
  return bindFullCoverageDimension(kernel, *dimension, parameter.getResult());
}

FragmentType predicateType(FragmentType source) {
  return FragmentType::get(source.getContext(), IntegerType::get(source.getContext(), 1),
                           source.getShape(), source.getAxisMaps(),
                           source.getValidity(),
                           source.getOwner());
}

FailureOr<Value> broadcastTo(OpBuilder &builder, Location location, Value value,
                             FragmentType target) {
  if (value.getType() == target)
    return value;
  if (!isa<IntegerType, FloatType, IndexType, FragmentType>(value.getType()))
    return failure();
  Type element = value.getType();
  if (auto fragment = dyn_cast<FragmentType>(element))
    element = fragment.getElementType();
  if (element != target.getElementType())
    return failure();
  return Value(builder.create<BroadcastOp>(location, target, value));
}

FailureOr<Value> zeroFill(OpBuilder &builder, Location location,
                          FragmentType target) {
  Type element = target.getElementType();
  TypedAttr zero;
  if (auto integer = dyn_cast<IntegerType>(element))
    zero = builder.getIntegerAttr(integer, 0);
  else if (auto floating = dyn_cast<FloatType>(element))
    zero = builder.getFloatAttr(floating, 0.0);
  else if (isa<IndexType>(element))
    zero = builder.getIndexAttr(0);
  else
    return failure();
  FailureOr<Value> scalar =
      materializeScalarConstant(builder, location, zero, element);
  if (failed(scalar))
    return failure();
  return Value(builder.create<BroadcastOp>(location, target, *scalar));
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges);

bool hasTailPredicate(
    Value coordinate,
    const llvm::DenseMap<Value, Value> &rangePredicates) {
  llvm::SmallPtrSet<Operation *, 8> ranges;
  collectCoordinateRanges(coordinate, ranges);
  return llvm::any_of(ranges, [&](Operation *operation) {
    auto range = dyn_cast<MakeRangeOp>(operation);
    return range && rangePredicates.contains(range.getResult());
  });
}

FailureOr<Value> accessValidity(OpBuilder &builder, Location location,
                                ValueRange coordinates,
                                llvm::DenseMap<Value, Value> &rangePredicates,
                                FragmentType valueType, Value existing) {
  FragmentType target = predicateType(valueType);
  Value result;
  if (existing) {
    FailureOr<Value> broadcast = broadcastTo(builder, location, existing, target);
    if (failed(broadcast))
      return failure();
    result = *broadcast;
  }
  for (Value coordinate : coordinates) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    for (Operation *operation : ranges) {
      auto range = dyn_cast<MakeRangeOp>(operation);
      if (!range)
        continue;
      auto found = rangePredicates.find(range.getResult());
      if (found == rangePredicates.end())
        continue;
      FailureOr<Value> broadcast =
          broadcastTo(builder, location, found->second, target);
      if (failed(broadcast))
        return failure();
      result = result ? Value(builder.create<BinaryOp>(
                            location, target, result, *broadcast, 11))
                      : *broadcast;
    }
  }
  return result ? FailureOr<Value>(result) : FailureOr<Value>(failure());
}

LogicalResult addTailValidity(func::FuncOp kernel,
                              llvm::DenseMap<Value, Value> &rangePredicates,
                              bool includeStores) {
  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });

  for (LoadOp load : loads) {
    bool affected = llvm::any_of(load.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      continue;
    auto valueType = dyn_cast<FragmentType>(load.getResult().getType());
    if (!valueType)
      return load.emitOpError(
          "pointwise blocked load must produce a physical fragment");
    OpBuilder builder(load);
    FailureOr<Value> valid = accessValidity(
        builder, load.getLoc(), load.getCoordinates(), rangePredicates, valueType,
        load.getValid());
    if (failed(valid))
      return load.emitOpError("could not form pointwise tail validity");
    Value fill;
    if (load.getFill()) {
      FailureOr<Value> broadcast =
          broadcastTo(builder, load.getLoc(), load.getFill(), valueType);
      if (failed(broadcast))
        return load.emitOpError("could not broadcast the existing load fill");
      fill = *broadcast;
    } else {
      FailureOr<Value> zero = zeroFill(builder, load.getLoc(), valueType);
      if (failed(zero))
        return load.emitOpError("pointwise load element type has no zero fill");
      fill = *zero;
    }
    auto replacement = builder.create<LoadOp>(
        load.getLoc(), valueType, load.getResource(), load.getCoordinates(),
        *valid, fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    load.getResult().replaceAllUsesWith(replacement.getResult());
    load.erase();
  }

  if (!includeStores)
    return success();
  for (StoreOp store : stores) {
    bool affected = llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      continue;
    auto valueType = dyn_cast<FragmentType>(store.getValue().getType());
    if (!valueType)
      return store.emitOpError(
          "pointwise blocked store must consume a physical fragment");
    OpBuilder builder(store);
    FailureOr<Value> valid = accessValidity(
        builder, store.getLoc(), store.getCoordinates(), rangePredicates,
        valueType, store.getValid());
    if (failed(valid))
      return store.emitOpError("could not form pointwise store validity");
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        store.getValue(), *valid, store.getSourceAxes(), store.getCollision());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

void collectProducerRanges(Value value, uint64_t sourceId,
                           llvm::SmallPtrSetImpl<Operation *> &ranges,
                           llvm::SmallPtrSetImpl<Operation *> &visited) {
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    auto argument = dyn_cast<BlockArgument>(value);
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!argument || !fragment)
      return;
    FailureOr<unsigned> sourceAxis = failure();
    for (auto [axis, attribute] : llvm::enumerate(fragment.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      if (mapping.getSourceId() != sourceId)
        continue;
      if (succeeded(sourceAxis))
        return;
      sourceAxis = axis;
    }
    if (failed(sourceAxis))
      return;
    Operation *parent = argument.getOwner()->getParentOp();
    Value outer;
    if (auto fold = dyn_cast_or_null<RegionFoldOp>(parent)) {
      if (argument.getOwner() == &fold.getSummarize().front() &&
          argument.getArgNumber() < fold.getSourceCount())
        outer = fold.getInputs()[argument.getArgNumber()];
    } else if (auto scan = dyn_cast_or_null<RegionScanOp>(parent)) {
      if (argument.getOwner() == &scan.getSummarize().front() &&
          argument.getArgNumber() < scan.getSourceCount())
        outer = scan.getInputs()[argument.getArgNumber()];
    }
    auto outerFragment = outer ? dyn_cast<FragmentType>(outer.getType())
                               : FragmentType();
    if (!outerFragment || *sourceAxis >= outerFragment.getAxisMaps().size())
      return;
    auto outerMapping =
        cast<AxisMapAttr>(outerFragment.getAxisMaps()[*sourceAxis]);
    collectProducerRanges(outer, outerMapping.getSourceId(), ranges, visited);
    return;
  }
  if (!visited.insert(producer).second)
    return;
  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    if (range.getSourceId() == sourceId)
      ranges.insert(producer);
    return;
  }
  for (Value operand : producer->getOperands())
    collectProducerRanges(operand, sourceId, ranges, visited);
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

FragmentType replaceSourceExtent(FragmentType source, uint64_t sourceId,
                                 PhysicalExprAttr extent) {
  FailureOr<unsigned> axis = sourceAxis(source, sourceId);
  if (failed(axis))
    return source;
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[*axis] = extent;
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

bool containsSource(Value value, uint64_t sourceId) {
  auto fragment = dyn_cast<FragmentType>(value.getType());
  return fragment && succeeded(sourceAxis(fragment, sourceId));
}

bool isReplayablePointwiseProducer(Operation *operation) {
  return isa<LoadOp, GatherOp, UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp,
             BitcastOp, BroadcastOp, SplatOp, ReshapeOp, TransposeOp, JoinOp,
             ContractOp, ScaledContractOp, SparseContractOp, ReduceOp>(operation);
}

bool isReplayablePointwiseValueGraph(
    Value value, uint64_t sourceId,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (!containsSource(value, sourceId))
    return true;
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return false;
  if (!visited.insert(producer).second)
    return true;
  if (auto range = dyn_cast<MakeRangeOp>(producer))
    return range.getSourceId() == sourceId;
  if (!isReplayablePointwiseProducer(producer) ||
      (producer->getNumRegions() != 0 && !isa<ReduceOp>(producer)) ||
      producer->getNumResults() != 1)
    return false;
  return llvm::all_of(producer->getOperands(), [&](Value operand) {
    return isReplayablePointwiseValueGraph(operand, sourceId, visited);
  });
}

FailureOr<Value> replayPointwiseValue(OpBuilder &builder, Value value,
                                      uint64_t sourceId,
                                      PhysicalExprAttr blockedExtent,
                                      Value blockedRange, Value blockedValidity,
                                      IRMapping &mapping) {
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  if (!containsSource(value, sourceId))
    return value;
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return failure();
  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    if (range.getSourceId() != sourceId)
      return failure();
    mapping.map(value, blockedRange);
    return blockedRange;
  }
  if (!isReplayablePointwiseProducer(producer) ||
      (producer->getNumRegions() != 0 && !isa<ReduceOp>(producer)) ||
      producer->getNumResults() != 1)
    return failure();
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement = replayPointwiseValue(
        builder, operand, sourceId, blockedExtent, blockedRange,
        blockedValidity, mapping);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  auto result = dyn_cast<FragmentType>(producer->getResult(0).getType());
  if (!result)
    return failure();
  FragmentType resultType = replaceSourceExtent(result, sourceId, blockedExtent);
  auto mapped = [&](Value operand) {
    if (!operand)
      return Value();
    Value replacement = mapping.lookupOrNull(operand);
    return replacement ? replacement : operand;
  };
  auto accessValidity = [&](Value existing) -> FailureOr<Value> {
    Value valid = mapped(existing);
    if (!blockedValidity)
      return valid;
    FailureOr<Value> projected = broadcastTo(
        builder, producer->getLoc(), blockedValidity, predicateType(resultType));
    if (failed(projected))
      return failure();
    if (!valid)
      return *projected;
    FailureOr<Value> original =
        broadcastTo(builder, producer->getLoc(), valid, predicateType(resultType));
    if (failed(original))
      return failure();
    return Value(builder.create<BinaryOp>(producer->getLoc(),
                                          predicateType(resultType), *original,
                                          *projected, /*and=*/11));
  };
  Value replayed;
  if (auto load = dyn_cast<LoadOp>(producer)) {
    FailureOr<Value> valid = accessValidity(load.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(load.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = zeroFill(builder, producer->getLoc(), resultType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : load.getCoordinates())
      coordinates.push_back(mapped(coordinate));
    auto clone = builder.create<LoadOp>(
        producer->getLoc(), resultType, mapped(load.getResource()), coordinates,
        *valid, fill, load.getSourceAxes());
    if (Attribute origin = load->getAttr(originAttr))
      clone->setAttr(originAttr, origin);
    replayed = clone.getResult();
  } else if (auto gather = dyn_cast<GatherOp>(producer)) {
    FailureOr<Value> valid = accessValidity(gather.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(gather.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = zeroFill(builder, producer->getLoc(), resultType);
      if (failed(zero))
        return failure();
      fill = *zero;
    }
    SmallVector<Value> coordinates;
    for (Value coordinate : gather.getCoordinates())
      coordinates.push_back(mapped(coordinate));
    auto clone = builder.create<GatherOp>(
        producer->getLoc(), resultType, mapped(gather.getSource()), coordinates,
        *valid, fill, gather.getSourceAxes());
    if (Attribute origin = gather->getAttr(originAttr))
      clone->setAttr(originAttr, origin);
    replayed = clone.getResult();
  } else if (auto reshape = dyn_cast<ReshapeOp>(producer);
             reshape && !containsSource(reshape.getValue(), sourceId)) {
    replayed = builder.create<BroadcastOp>(producer->getLoc(), resultType,
                                           mapped(reshape.getValue()));
  } else {
    Operation *clone = builder.clone(*producer, mapping);
    clone->getResult(0).setType(resultType);
    replayed = clone->getResult(0);
  }
  if (!mapping.lookupOrNull(value))
    mapping.map(value, replayed);
  return replayed;
}

LogicalResult replayBlockedStorePayloads(func::FuncOp kernel,
                                         MakeRangeOp original, Value blocked,
                                         PhysicalExprAttr blockedExtent) {
  SmallVector<StoreOp> stores;
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  IRMapping mapping;
  mapping.map(original.getResult(), blocked);
  for (StoreOp store : stores) {
    bool usesRange = false;
    for (Value coordinate : store.getCoordinates()) {
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectCoordinateRanges(coordinate, ranges);
      usesRange |= ranges.contains(original.getOperation());
    }
    if (!usesRange || !containsSource(store.getValue(), original.getSourceId()))
      continue;
    OpBuilder builder(store);
    FailureOr<Value> payload = replayPointwiseValue(
        builder, store.getValue(), original.getSourceId(), blockedExtent,
        blocked, Value(), mapping);
    if (failed(payload)) {
      auto existing = dyn_cast<FragmentType>(store.getValue().getType());
      FailureOr<unsigned> axis =
          existing ? sourceAxis(existing, original.getSourceId())
                   : FailureOr<unsigned>(failure());
      if (succeeded(axis) && existing.getShape()[*axis] == blockedExtent)
        continue;
      return store.emitOpError(
          "pointwise blocked store payload cannot be replayed at the selected physical range");
    }
    store.getValueMutable().set(*payload);
  }
  return success();
}

bool reductionTypeConsumesSource(Type type, ArrayRef<int64_t> axes,
                                 uint64_t sourceId) {
  if (auto record = dyn_cast<RecordType>(type)) {
    return llvm::any_of(record.getFieldTypes(), [&](Attribute field) {
      return reductionTypeConsumesSource(cast<TypeAttr>(field).getValue(), axes,
                                         sourceId);
    });
  }
  auto fragment = dyn_cast<FragmentType>(type);
  if (!fragment)
    return false;
  for (int64_t axis : axes) {
    if (axis < 0 || axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
      continue;
    if (cast<AxisMapAttr>(fragment.getAxisMaps()[axis]).getSourceId() == sourceId)
      return true;
  }
  return false;
}

bool reductionConsumesSource(gpu::ReduceOp reduce, uint64_t sourceId) {
  for (Value source :
       reduce.getInputs().take_front(reduce.getSourceCount())) {
    if (reductionTypeConsumesSource(source.getType(), reduce.getAxes(), sourceId))
      return true;
  }
  return false;
}

bool reductionConsumesDimension(gpu::ReduceOp reduce, int64_t dimension) {
  for (Value source :
       reduce.getInputs().take_front(reduce.getSourceCount())) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment)
      continue;
    for (int64_t axis : reduce.getAxes()) {
      if (axis < 0 || axis >= static_cast<int64_t>(fragment.getAxisMaps().size()))
        continue;
      uint64_t sourceId =
          cast<AxisMapAttr>(fragment.getAxisMaps()[axis]).getSourceId();
      llvm::SmallPtrSet<Operation *, 8> ranges;
      llvm::SmallPtrSet<Operation *, 32> visited;
      collectProducerRanges(source, sourceId, ranges, visited);
      if (llvm::any_of(ranges, [&](Operation *operation) {
            auto range = dyn_cast<MakeRangeOp>(operation);
            auto sourceDimension =
                range ? range->getAttrOfType<IntegerAttr>(sourceDimensionAttr)
                      : IntegerAttr();
            return sourceDimension && sourceDimension.getInt() == dimension;
          }))
        return true;
    }
  }
  return false;
}

bool hasReductionDependency(Value value, uint64_t sourceId,
                            std::optional<int64_t> sourceDimension,
                            llvm::SmallPtrSetImpl<Operation *> &visited) {
  Operation *producer = value.getDefiningOp();
  if (!producer || !visited.insert(producer).second)
    return false;
  if (auto reduce = dyn_cast<ReduceOp>(producer))
    if (reductionConsumesSource(reduce, sourceId) ||
        (sourceDimension &&
         reductionConsumesDimension(reduce, *sourceDimension)))
      return true;
  if (auto loop = dyn_cast<scf::ForOp>(producer)) {
    auto traversal =
        loop->getAttrOfType<IntegerAttr>(reductionTraversalSourceAttr);
    if (traversal && static_cast<uint64_t>(traversal.getInt()) == sourceId)
      return true;
    auto result = dyn_cast<OpResult>(value);
    auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (!result || !yield || result.getResultNumber() >= loop.getInitArgs().size())
      return false;
    unsigned index = result.getResultNumber();
    if (hasReductionDependency(loop.getInitArgs()[index], sourceId,
                               sourceDimension, visited) ||
        hasReductionDependency(yield.getResults()[index], sourceId,
                               sourceDimension, visited))
      return true;
  }
  return llvm::any_of(producer->getOperands(), [&](Value operand) {
    return hasReductionDependency(operand, sourceId, sourceDimension, visited);
  });
}

bool storeUsesRange(StoreOp store, MakeRangeOp range) {
  return llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    return ranges.contains(range.getOperation());
  });
}

std::optional<int64_t> storeAxisForRange(StoreOp store, MakeRangeOp range) {
  std::optional<int64_t> result;
  for (auto [coordinateIndex, coordinate] :
       llvm::enumerate(store.getCoordinates())) {
    llvm::SmallPtrSet<Operation *, 8> ranges;
    collectCoordinateRanges(coordinate, ranges);
    if (!ranges.contains(range.getOperation()))
      continue;
    int64_t sourceAxis = store.getSourceAxes()[coordinateIndex];
    if (!result || sourceAxis > *result)
      result = sourceAxis;
  }
  return result;
}

LogicalResult realizeReusePointwiseTraversal(func::FuncOp kernel,
                                             MakeRangeOp range) {
  SmallVector<StoreOp> stores;
  kernel.walk([&](StoreOp store) {
    if (storeUsesRange(store, range))
      stores.push_back(store);
  });
  if (stores.empty())
    return range.emitOpError(
        "reuse-sensitive pointwise traversal has no write effect");

  std::string name = ("POINTWISE_CHUNK_" + Twine(range.getSourceId())).str();
  ParameterOp chunk;
  kernel.walk([&](ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() == name)
      chunk = parameter;
  });
  if (!chunk) {
    static constexpr int64_t candidates[] = {16, 32, 64, 128, 256, 512,
                                             1024, 2048, 4096};
    OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), entry.getStringAttr(name),
        static_cast<uint32_t>(ParameterRole::OwnershipN),
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    chunk = entry.create<ParameterOp>(range.getLoc(), entry.getIndexType(), schema);
    if (auto dimension =
            range->getAttrOfType<IntegerAttr>(sourceDimensionAttr))
      chunk->setAttr(dimensionAttr, dimension);
  }

  auto originalType = dyn_cast<FragmentType>(range.getResult().getType());
  if (!originalType || originalType.getShape().size() != 1)
    return range.emitOpError(
        "reuse-sensitive pointwise traversal requires one physical source axis");
  PhysicalExprAttr chunkExtent = expression(
      kernel.getContext(), PhysicalExprKind::Parameter, 0,
      chunk.getParameter().getName().getValue());
  auto blockedType = FragmentType::get(
      kernel.getContext(), originalType.getElementType(),
      ArrayAttr::get(kernel.getContext(), {chunkExtent}),
      originalType.getAxisMaps(), originalType.getValidity(),
      originalType.getOwner());
  // The original range remains the authority for the full reduction
  // traversal.  The internal writeback loop below owns a distinct blocked
  // range and replays only the store-side value graph against it.

  OpBuilder builder(stores.front());
  Value distance = builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), range.getExtent(), range.getStep(),
      /*multiply=*/2);
  Value stop = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                        range.getStart(), distance, /*add=*/0);
  Value loopStep = builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), chunk.getResult(), range.getStep(),
      /*multiply=*/2);
  bool bodyFailed = false;
  std::string failureReason;
  auto loop = builder.create<scf::ForOp>(
      range.getLoc(), range.getStart(), stop, loopStep, ValueRange{},
      [&](OpBuilder &nested, Location location, Value tileStart, ValueRange) {
        Value blocked = nested.create<MakeRangeOp>(
            location, blockedType, tileStart, chunk.getResult(), range.getStep(),
            range.getSourceId(), range.getSourceAxis());
        for (StringRef attribute : {sourceSubregionAttr, sourceDimensionAttr})
          if (Attribute value = range->getAttr(attribute))
            blocked.getDefiningOp()->setAttr(attribute, value);
        Value end = nested.create<BroadcastOp>(location, blockedType, stop);
        Value tail = nested.create<CompareOp>(
            location, predicateType(blockedType), blocked, end,
            /*less-than=*/2);
        IRMapping mapping;
        mapping.map(range.getResult(), blocked);
        for (StoreOp store : stores) {
          FailureOr<Value> payload = replayPointwiseValue(
              nested, store.getValue(), range.getSourceId(), chunkExtent,
              blocked, tail, mapping);
          if (failed(payload)) {
            bodyFailed = true;
            failureReason = "write payload cannot be replayed in the internal tile loop";
            return;
          }
          SmallVector<Value> coordinates;
          for (Value coordinate : store.getCoordinates()) {
            FailureOr<Value> replayed = replayPointwiseValue(
                nested, coordinate, range.getSourceId(), chunkExtent, blocked,
                tail, mapping);
            if (failed(replayed)) {
              bodyFailed = true;
              failureReason =
                  "write coordinate cannot be replayed in the internal tile loop";
              return;
            }
            coordinates.push_back(*replayed);
          }
          auto payloadType = dyn_cast<FragmentType>((*payload).getType());
          if (!payloadType) {
            bodyFailed = true;
            failureReason = "write payload lost its physical fragment schema";
            return;
          }
          FailureOr<Value> valid = broadcastTo(nested, location, tail,
                                               predicateType(payloadType));
          if (failed(valid)) {
            bodyFailed = true;
            failureReason = "tile tail cannot be projected to the write payload";
            return;
          }
          if (store.getValid()) {
            FailureOr<Value> existing = replayPointwiseValue(
                nested, store.getValid(), range.getSourceId(), chunkExtent,
                blocked, tail, mapping);
            if (failed(existing)) {
              bodyFailed = true;
              failureReason = "write validity cannot be replayed in the tile loop";
              return;
            }
            FailureOr<Value> projected = broadcastTo(
                nested, location, *existing, predicateType(payloadType));
            if (failed(projected)) {
              bodyFailed = true;
              failureReason = "write validity cannot be projected to the payload";
              return;
            }
            valid = Value(nested.create<BinaryOp>(
                location, predicateType(payloadType), *valid, *projected,
                /*and=*/11));
          }
          auto replacement = nested.create<StoreOp>(
              location, store.getResource(), coordinates, *payload, *valid,
              store.getSourceAxes(), store.getCollision());
          if (Attribute origin = store->getAttr(originAttr))
            replacement->setAttr(originAttr, origin);
        }
        if (!bodyFailed)
          nested.create<scf::YieldOp>(location);
      });
  if (bodyFailed) {
    loop.erase();
    return range.emitOpError(
               "reuse-sensitive pointwise traversal could not be materialized: ")
           << failureReason;
  }
  for (StoreOp store : stores)
    store.erase();
  return success();
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto fragment = dyn_cast<FragmentType>(coordinate.getType());
  if (!fragment)
    return;
  for (Attribute attribute : fragment.getAxisMaps()) {
    llvm::SmallPtrSet<Operation *, 32> visited;
    collectProducerRanges(coordinate, cast<AxisMapAttr>(attribute).getSourceId(),
                          ranges, visited);
  }
}

void collectStoreRanges(Value value, llvm::SmallPtrSetImpl<Operation *> &ranges,
                        llvm::SmallPtrSetImpl<Operation *> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<CastOp, BroadcastOp>(user)) {
      collectStoreRanges(user->getResult(0), ranges, visited);
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getValue() != value)
      continue;
    for (Value coordinate : store.getCoordinates())
      collectCoordinateRanges(coordinate, ranges);
  }
}

bool hasOnlyDirectStoreUsers(Value value,
                             llvm::SmallPtrSetImpl<Operation *> &visited) {
  bool foundStore = false;
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<CastOp, BroadcastOp>(user)) {
      if (!hasOnlyDirectStoreUsers(user->getResult(0), visited))
        return false;
      foundStore = true;
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getValue() != value || store.getCollision() != 0)
      return false;
    foundStore = true;
  }
  return foundStore;
}

HistogramOp histogramSource(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (auto histogram = dyn_cast<HistogramOp>(producer))
      return histogram;
    if (!isa<BroadcastOp, CastOp>(producer) || producer->getNumOperands() != 1 ||
        producer->getNumResults() != 1)
      return HistogramOp();
    value = producer->getOperand(0);
  }
  return HistogramOp();
}

LogicalResult realizeDistributedHistograms(func::FuncOp kernel) {
  SmallVector<StoreOp> stores;
  kernel.walk([&](StoreOp store) {
    if (histogramSource(store.getValue()))
      stores.push_back(store);
  });
  for (StoreOp store : stores) {
    auto resource = dyn_cast<ViewType>(store.getResource().getType());
    auto value = dyn_cast<FragmentType>(store.getValue().getType());
    if (!resource || !value || resource.getElementType() != value.getElementType())
      return store.emitOpError(
          "distributed histogram requires an output view with the count element type");
    OpBuilder builder(store);
    auto atomic = builder.create<AtomicRMWOp>(
        store.getLoc(), store.getValue().getType(), store.getResource(),
        store.getCoordinates(), store.getValue(), store.getValid(),
        /*kind=*/1, /*ordering=*/0, /*sharing=*/1, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      atomic->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

bool sameFragmentSchema(FragmentType lhs, FragmentType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getAxisMaps() == rhs.getAxisMaps() &&
         lhs.getValidity() == rhs.getValidity() &&
         lhs.getOwner() == rhs.getOwner();
}

LogicalResult alignExplicitBroadcastOperands(func::FuncOp kernel) {
  auto align = [&](Operation *operation, unsigned operandIndex,
                   FragmentType targetShape) -> LogicalResult {
    Value value = operation->getOperand(operandIndex);
    auto source = dyn_cast<FragmentType>(value.getType());
    if (!source || sameFragmentSchema(source, targetShape))
      return success();
    if (source.getShape().size() > targetShape.getShape().size())
      return failure();
    for (Attribute sourceMapping : source.getAxisMaps()) {
      auto sourceAxis = cast<AxisMapAttr>(sourceMapping);
      if (!llvm::any_of(targetShape.getAxisMaps(), [&](Attribute targetMapping) {
            auto targetAxis = cast<AxisMapAttr>(targetMapping);
            return sourceAxis.getSourceId() == targetAxis.getSourceId() &&
                   sourceAxis.getSourceAxis() == targetAxis.getSourceAxis();
          }))
        return failure();
    }
    Operation *definition = value.getDefiningOp();
    auto target = FragmentType::get(
        kernel.getContext(), source.getElementType(), targetShape.getShape(),
        targetShape.getAxisMaps(), targetShape.getValidity(),
        targetShape.getOwner());
    OpBuilder builder(operation);
    Value replacement;
    if (auto broadcast = dyn_cast<BroadcastOp>(definition))
      replacement = builder.create<BroadcastOp>(operation->getLoc(), target,
                                                broadcast.getValue());
    else if (auto splat = dyn_cast<SplatOp>(definition))
      replacement = builder.create<SplatOp>(
          operation->getLoc(), target, splat.getValue());
    else
      replacement =
          builder.create<BroadcastOp>(operation->getLoc(), target, value);
    if (Attribute origin = definition->getAttr(originAttr))
      replacement.getDefiningOp()->setAttr(originAttr, origin);
    operation->setOperand(operandIndex, replacement);
    return success();
  };

  WalkResult result = kernel.walk([&](Operation *operation) {
    FragmentType target;
    if (operation->getNumResults() == 1)
      target = dyn_cast<FragmentType>(operation->getResult(0).getType());
    if (!target)
      return WalkResult::advance();
    if (!isa<UnaryOp, BinaryOp, CompareOp, SelectOp, CastOp, BitcastOp>(
            operation))
      return WalkResult::advance();
    for (unsigned index = 0; index < operation->getNumOperands(); ++index)
      if (failed(align(operation, index, target)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool isCartesianPointwiseValueOp(Operation *operation) {
  return isa<SplatOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
             CastOp, BitcastOp, LoadOp, GatherOp, RandomBitsOp>(operation);
}

bool supportsCartesianPointwiseValueGraph(func::FuncOp kernel) {
  bool supported = true;
  kernel.walk([&](Operation *operation) {
    if (!supported || isa<MakeRangeOp>(operation))
      return WalkResult::advance();
    if (llvm::none_of(operation->getResultTypes(),
                      [](Type type) { return isa<FragmentType>(type); }))
      return WalkResult::advance();
    if (!isCartesianPointwiseValueOp(operation)) {
      supported = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return supported;
}

LogicalResult rankLiftPointwiseValueGraph(
    func::FuncOp kernel, ArrayRef<MakeRangeOp> liftedRanges) {
  if (liftedRanges.empty())
    return success();

  SmallVector<std::pair<PhysicalExprAttr, AxisMapAttr>> liftedAxes;
  for (MakeRangeOp range : llvm::reverse(liftedRanges)) {
    auto fragment = cast<FragmentType>(range.getResult().getType());
    liftedAxes.emplace_back(
        cast<PhysicalExprAttr>(fragment.getShape()[0]),
        cast<AxisMapAttr>(fragment.getAxisMaps()[0]));
  }
  auto liftedType = [&](FragmentType original) {
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    auto appendAxis = [&](PhysicalExprAttr extent, AxisMapAttr mapping) {
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mappings.size()));
    };
    for (auto [extent, mapping] : liftedAxes) {
      bool present = llvm::any_of(original.getAxisMaps(), [&](Attribute attribute) {
        auto axis = cast<AxisMapAttr>(attribute);
        return axis.getSourceId() == mapping.getSourceId() &&
               axis.getSourceAxis() == mapping.getSourceAxis();
      });
      if (!present)
        appendAxis(extent, mapping);
    }
    for (auto [extent, mapping] :
         llvm::zip(original.getShape(), original.getAxisMaps()))
      appendAxis(cast<PhysicalExprAttr>(extent), cast<AxisMapAttr>(mapping));
    return FragmentType::get(kernel.getContext(), original.getElementType(),
                             ArrayAttr::get(kernel.getContext(), shape),
                             ArrayAttr::get(kernel.getContext(), mappings),
                             original.getValidity(), original.getOwner());
  };

  kernel.walk([&](Operation *operation) {
    if (isa<MakeRangeOp>(operation))
      return;
    for (Value result : operation->getResults())
      if (auto fragment = dyn_cast<FragmentType>(result.getType()))
        result.setType(liftedType(fragment));
  });
  return success();
}

} // namespace

static LogicalResult realizePointwiseBlockingImpl(ModuleOp module,
                                                   bool ownershipOnly) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;

  if (ownershipOnly) {
    SmallVector<WorksetCoordinateOp> pointwiseCoordinates;
    kernel.walk([&](WorksetCoordinateOp coordinate) {
      if (coordinate->hasAttr(pointwiseWorksetAttr))
        pointwiseCoordinates.push_back(coordinate);
    });
    SmallVector<MakeRangeOp> existingRanges;
    kernel.walk([&](MakeRangeOp range) { existingRanges.push_back(range); });
    SmallVector<WorksetCoordinateOp> lifted;
    if (pointwiseCoordinates.size() == 1 && existingRanges.empty()) {
      lifted.push_back(pointwiseCoordinates.front());
    } else if (pointwiseCoordinates.size() > 2) {
      lifted.append(pointwiseCoordinates.end() - 2,
                    pointwiseCoordinates.end());
    } else if (pointwiseCoordinates.size() == 2 && existingRanges.empty()) {
      lifted.append(pointwiseCoordinates.begin(), pointwiseCoordinates.end());
    }
    if (!supportsCartesianPointwiseValueGraph(kernel))
      lifted.clear();

    SmallVector<MakeRangeOp> liftedRanges;
    for (WorksetCoordinateOp coordinate : lifted) {
      OpBuilder builder(coordinate);
      builder.setInsertionPointAfter(coordinate);
      Value extent = builder.create<arith::ConstantIndexOp>(coordinate.getLoc(), 1);
      Value step = builder.create<arith::ConstantIndexOp>(coordinate.getLoc(), 1);
      PhysicalExprAttr unit = expression(module.getContext(),
                                         PhysicalExprKind::Constant, 1);
      auto type = FragmentType::get(
          module.getContext(), builder.getIndexType(), builder.getArrayAttr({unit}),
          builder.getArrayAttr({AxisMapAttr::get(
              module.getContext(), coordinate.getSourceId(),
              coordinate.getSourceAxis(), /*fragmentAxis=*/0)}),
          /*validity=*/1, /*owner=*/1);
      auto range = builder.create<MakeRangeOp>(
          coordinate.getLoc(), type, coordinate.getResult(), extent, step,
          coordinate.getSourceId(), coordinate.getSourceAxis());
      range->setAttr(worksetCoordinateRangeAttr, builder.getUnitAttr());
      if (Attribute dimension = coordinate->getAttr(sourceDimensionAttr))
        range->setAttr(sourceDimensionAttr, dimension);
      coordinate.getResult().replaceAllUsesExcept(range.getResult(), range);
      liftedRanges.push_back(range);
    }
    if (failed(rankLiftPointwiseValueGraph(kernel, liftedRanges)))
      return kernel.emitError(
          "failed to rank-lift a legal Cartesian pointwise value graph");
  }

  SmallVector<MakeRangeOp> dynamicRanges;
  SmallVector<MakeRangeOp> allRanges;
  llvm::DenseMap<Value, Value> fixedRangePredicates;
  kernel.walk([&](MakeRangeOp range) {
    allRanges.push_back(range);
    if (!hasCompileTimeExtent(range.getExtent())) {
      dynamicRanges.push_back(range);
      return;
    }
    if (!ownershipOnly ||
        !range.getExtent().getDefiningOp<arith::ConstantIndexOp>())
      return;
    if (range->hasAttr(worksetCoordinateRangeAttr)) {
      dynamicRanges.push_back(range);
      return;
    }
    auto fragment = cast<FragmentType>(range.getResult().getType());
    auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
    if (extent.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        extent.getValue() > 1)
      dynamicRanges.push_back(range);
  });
  // Classify ranges consumed by structured operations before considering any
  // pointwise reblocking.  Their logical extent belongs to the structured
  // traversal; replacing it with a pointwise fragment extent would truncate
  // fold/scan/reduction/contract semantics before the owning pass sees them.
  llvm::SmallPtrSet<Operation *, 32> internalTraversalRanges;
  llvm::SmallPtrSet<Operation *, 32> structuredTraversalRanges;
  auto collectAxisInto = [&](Value source, uint64_t axis,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis >= fragment.getAxisMaps().size())
      return;
    uint64_t sourceId =
        cast<AxisMapAttr>(fragment.getAxisMaps()[axis]).getSourceId();
    llvm::SmallPtrSet<Operation *, 32> visited;
    collectProducerRanges(source, sourceId, ranges, visited);
  };
  auto collectAllAxesInto = [&](Value value,
                                llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment)
      return;
    for (Attribute attribute : fragment.getAxisMaps()) {
      uint64_t sourceId = cast<AxisMapAttr>(attribute).getSourceId();
      llvm::SmallPtrSet<Operation *, 32> visited;
      collectProducerRanges(value, sourceId, ranges, visited);
    }
  };
  auto collectStructuredRegionRanges = [&](Operation *structured,
                                           ValueRange sources, uint64_t axis) {
    llvm::SmallDenseSet<uint64_t> segmentSources;
    for (Value source : sources) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment || axis >= fragment.getAxisMaps().size())
        continue;
      segmentSources.insert(
          cast<AxisMapAttr>(fragment.getAxisMaps()[axis]).getSourceId());
    }
    for (Region &region : structured->getRegions())
      region.walk([&](MakeRangeOp range) {
        if (segmentSources.contains(range.getSourceId()))
          structuredTraversalRanges.insert(range.getOperation());
      });
  };
  kernel.walk([&](RegionFoldOp fold) {
    ValueRange sources = fold.getInputs().take_front(fold.getSourceCount());
    for (Value source : sources)
      collectAxisInto(source, fold.getAxis(), structuredTraversalRanges);
    collectStructuredRegionRanges(fold, sources, fold.getAxis());
  });
  kernel.walk([&](RegionScanOp scan) {
    ValueRange sources = scan.getInputs().take_front(scan.getSourceCount());
    for (Value source : sources)
      collectAxisInto(source, scan.getAxis(), structuredTraversalRanges);
    collectStructuredRegionRanges(scan, sources, scan.getAxis());
  });
  kernel.walk([&](ReduceOp reduce) {
    for (Value source : reduce.getInputs().take_front(reduce.getSourceCount()))
      for (int64_t axis : reduce.getAxes())
        if (axis >= 0)
          collectAxisInto(source, static_cast<uint64_t>(axis),
                          structuredTraversalRanges);
  });
  bool scanCoverageFailed = false;
  kernel.walk([&](ScanOp scan) {
    for (Value source : scan.getInputs().take_front(scan.getSourceCount()))
      collectAxisInto(source, scan.getAxis(), structuredTraversalRanges);
    for (Value source : scan.getInputs().take_front(scan.getSourceCount()))
      scanCoverageFailed |=
          failed(requireFullDimensionCoverage(kernel, source, scan.getAxis()));
    for (Value result : scan.getResults()) {
      llvm::SmallPtrSet<Operation *, 16> visited;
      collectStoreRanges(result, internalTraversalRanges, visited);
    }
  });
  if (scanCoverageFailed)
    return kernel.emitError(
        "dynamic scan axis has no launch-visible full-coverage realization");
  kernel.walk([&](HistogramOp histogram) {
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(histogram.getResult(), internalTraversalRanges, visited);
  });
  kernel.walk([&](ContractOp contract) {
    llvm::SmallPtrSet<Operation *, 16> storeUsers;
    if (hasOnlyDirectStoreUsers(contract.getResult(), storeUsers)) {
      collectAllAxesInto(contract.getLhs(), structuredTraversalRanges);
      collectAllAxesInto(contract.getRhs(), structuredTraversalRanges);
      collectAllAxesInto(contract.getResult(), structuredTraversalRanges);
      llvm::SmallPtrSet<Operation *, 16> visited;
      collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
      return;
    }
    for (int64_t axis : contract.getLhsReductionAxes())
      if (axis >= 0)
        collectAxisInto(contract.getLhs(), static_cast<uint64_t>(axis),
                        structuredTraversalRanges);
    for (int64_t axis : contract.getRhsReductionAxes())
      if (axis >= 0)
        collectAxisInto(contract.getRhs(), static_cast<uint64_t>(axis),
                        structuredTraversalRanges);
  });
  kernel.walk([&](ScaledContractOp contract) {
    llvm::SmallPtrSet<Operation *, 16> storeUsers;
    if (hasOnlyDirectStoreUsers(contract.getResult(), storeUsers)) {
      collectAllAxesInto(contract.getLhs(), structuredTraversalRanges);
      collectAllAxesInto(contract.getLhsScale(), structuredTraversalRanges);
      collectAllAxesInto(contract.getRhs(), structuredTraversalRanges);
      collectAllAxesInto(contract.getRhsScale(), structuredTraversalRanges);
      collectAllAxesInto(contract.getResult(), structuredTraversalRanges);
      llvm::SmallPtrSet<Operation *, 16> visited;
      collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
      return;
    }
    for (int64_t axis : contract.getLhsReductionAxes())
      if (axis >= 0) {
        collectAxisInto(contract.getLhs(), static_cast<uint64_t>(axis),
                        structuredTraversalRanges);
        collectAxisInto(contract.getLhsScale(), static_cast<uint64_t>(axis),
                        structuredTraversalRanges);
      }
    for (int64_t axis : contract.getRhsReductionAxes())
      if (axis >= 0) {
        collectAxisInto(contract.getRhs(), static_cast<uint64_t>(axis),
                        structuredTraversalRanges);
        collectAxisInto(contract.getRhsScale(), static_cast<uint64_t>(axis),
                        structuredTraversalRanges);
      }
  });
  llvm::SmallPtrSet<Operation *, 16> reuseTraversalRanges;
  SmallVector<StoreOp> candidateStores;
  kernel.walk([&](StoreOp store) { candidateStores.push_back(store); });
  for (StoreOp store : candidateStores) {
    SmallVector<std::pair<MakeRangeOp, int64_t>> candidates;
    int64_t innermostSourceAxis = -1;
    for (MakeRangeOp range : allRanges) {
      std::optional<int64_t> sourceAxis = storeAxisForRange(store, range);
      if (!sourceAxis)
        continue;
      llvm::SmallPtrSet<Operation *, 32> visited;
      std::optional<int64_t> sourceDimension;
      if (auto dimension =
              range->getAttrOfType<IntegerAttr>(sourceDimensionAttr))
        sourceDimension = dimension.getInt();
      bool dependency = hasReductionDependency(store.getValue(),
                                               range.getSourceId(),
                                               sourceDimension, visited);
      if (!dependency)
        continue;
      candidates.emplace_back(range, *sourceAxis);
      innermostSourceAxis = std::max(innermostSourceAxis, *sourceAxis);
    }
    for (auto [range, sourceAxis] : candidates) {
      internalTraversalRanges.insert(range.getOperation());
      if (sourceAxis != innermostSourceAxis)
        continue;
      reuseTraversalRanges.insert(range.getOperation());
    }
  }
  if (!ownershipOnly)
    for (MakeRangeOp range : allRanges)
      if (reuseTraversalRanges.contains(range.getOperation()) &&
          !llvm::is_contained(dynamicRanges, range))
        dynamicRanges.push_back(range);
  for (Operation *operation : structuredTraversalRanges) {
    auto range = dyn_cast<MakeRangeOp>(operation);
    bool writeOwnership = range && llvm::any_of(candidateStores, [&](StoreOp store) {
      return storeUsesRange(store, range);
    });
    if (!writeOwnership)
      internalTraversalRanges.insert(operation);
  }
  if (ownershipOnly)
    llvm::erase_if(dynamicRanges, [](MakeRangeOp range) {
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto extent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      return range->hasAttr(sourceSubregionAttr) &&
             extent.getKind() ==
                 static_cast<uint32_t>(PhysicalExprKind::Constant);
    });
  if (!ownershipOnly) {
    SmallVector<MakeRangeOp> unresolved;
    for (MakeRangeOp range : dynamicRanges) {
      if (internalTraversalRanges.contains(range.getOperation())) {
        unresolved.push_back(range);
        continue;
      }
      FailureOr<ParameterOp> parameter = blockingParameter(kernel, range);
      if (failed(parameter) ||
          parameter->getParameter().getRole() !=
              static_cast<uint32_t>(ParameterRole::ScanChunk)) {
        unresolved.push_back(range);
        continue;
      }
      // Region fold/scan owns this logical traversal.  Its physical segment
      // parameter is the loop step, not a replacement for the logical source
      // extent; replacing the extent would silently truncate the program to
      // one segment before structured lowering sees it.
    }
    dynamicRanges = std::move(unresolved);
    unresolved.clear();
    for (MakeRangeOp range : dynamicRanges) {
      if (reuseTraversalRanges.contains(range.getOperation())) {
        unresolved.push_back(range);
        continue;
      }
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      if (physicalExtent.getKind() !=
          static_cast<uint32_t>(PhysicalExprKind::Constant)) {
        unresolved.push_back(range);
        continue;
      }
      OpBuilder builder(range);
      Value extent = builder.create<arith::ConstantIndexOp>(
          range.getLoc(), physicalExtent.getValue());
      Value exactDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getExtent(),
          range.getStep(), 2);
      Value exactEnd = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getStart(),
          exactDistance, 0);
      auto blocked = builder.create<MakeRangeOp>(
          range.getLoc(), fragment, range.getStart(), extent, range.getStep(),
          range.getSourceId(), range.getSourceAxis());
      for (StringRef attribute : {sourceSubregionAttr, sourceDimensionAttr})
        if (Attribute value = range->getAttr(attribute))
          blocked->setAttr(attribute, value);
      Value endFragment =
          builder.create<BroadcastOp>(range.getLoc(), fragment, exactEnd);
      Value valid = builder.create<CompareOp>(
          range.getLoc(), predicateType(fragment), blocked.getResult(),
          endFragment, 2);
      range.getResult().replaceAllUsesWith(blocked.getResult());
      fixedRangePredicates[blocked.getResult()] = valid;
      range.erase();
    }
    // A fixed physical extent immediately requires executable tail validity.
    // Keeping the predicate only in a side map across later rewrites leaves it
    // without an IR use and lets dead-value cleanup invalidate the fact.
    if (!fixedRangePredicates.empty()) {
      if (failed(addTailValidity(kernel, fixedRangePredicates,
                                 /*includeStores=*/true)))
        return failure();
      fixedRangePredicates.clear();
    }
    dynamicRanges = std::move(unresolved);
  }
  if (!ownershipOnly)
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return structuredTraversalRanges.contains(range.getOperation()) &&
             !reuseTraversalRanges.contains(range.getOperation());
    });
  if (!ownershipOnly) {
    SmallVector<MakeRangeOp> realized;
    for (MakeRangeOp range : dynamicRanges) {
      if (!reuseTraversalRanges.contains(range.getOperation()))
        continue;
      bool replayable = true;
      for (StoreOp store : candidateStores) {
        if (!storeUsesRange(store, range))
          continue;
        llvm::SmallPtrSet<Operation *, 32> visited;
        replayable &= isReplayablePointwiseValueGraph(
            store.getValue(), range.getSourceId(), visited);
      }
      // A value graph containing loop-carried or otherwise non-replayable
      // structure must remain one full fragment.  Tiling the writeback axis
      // would require cloning author control flow, which is not a pointwise
      // realization.  The ordinary internal-range path below binds an exact
      // full-coverage extent instead.
      if (!replayable)
        continue;
      if (failed(realizeReusePointwiseTraversal(kernel, range)))
        return failure();
      realized.push_back(range);
    }
    eraseDeadPhysicalValues(kernel);
    llvm::erase_if(dynamicRanges, [&](MakeRangeOp range) {
      return llvm::is_contained(realized, range);
    });
    llvm::erase_if(dynamicRanges, [](MakeRangeOp range) {
      return !range->getBlock() || range.getResult().use_empty();
    });
  }
  if (dynamicRanges.empty()) {
    if (!ownershipOnly &&
        failed(addTailValidity(kernel, fixedRangePredicates,
                               /*includeStores=*/true)))
      return failure();
    if (!ownershipOnly && failed(realizeDistributedHistograms(kernel)))
      return failure();
    if (failed(alignExplicitBroadcastOperands(kernel)))
      return failure();
    eraseDeadPhysicalValues(kernel);
    return success();
  }

  SmallVector<DelinearizeOp> mappings;
  kernel.walk([&](DelinearizeOp mapping) { mappings.push_back(mapping); });
  if (mappings.size() != 1)
    return kernel.emitError(
        "dynamic pointwise blocking requires one explicit execution workset");
  DelinearizeOp mapping = mappings.front();

  struct WriteEffectFacts {
    SmallVector<Value> coordinates;
    SmallVector<Value> payloads;
  };
  SmallVector<WriteEffectFacts> writeEffects;
  llvm::SmallDenseSet<uint64_t> ownershipSources;
  auto collectOwnership = [&](ValueRange coordinates, ValueRange payloads) {
    WriteEffectFacts effect;
    effect.coordinates.append(coordinates.begin(), coordinates.end());
    effect.payloads.append(payloads.begin(), payloads.end());
    writeEffects.push_back(std::move(effect));
    for (Value coordinate : coordinates) {
      auto fragment = dyn_cast<FragmentType>(coordinate.getType());
      if (!fragment)
        continue;
      for (Attribute attribute : fragment.getAxisMaps()) {
        ownershipSources.insert(cast<AxisMapAttr>(attribute).getSourceId());
      }
    }
  };
  kernel.walk([&](StoreOp store) {
    collectOwnership(store.getCoordinates(), ValueRange(store.getValue()));
  });
  kernel.walk([&](AtomicStoreOp store) {
    collectOwnership(store.getCoordinates(), ValueRange(store.getValue()));
  });
  kernel.walk([&](AtomicRMWOp store) {
    collectOwnership(store.getCoordinates(), ValueRange(store.getValue()));
  });
  kernel.walk([&](AtomicCompareExchangeOp store) {
    SmallVector<Value> payloads{store.getExpected(), store.getDesired()};
    collectOwnership(store.getCoordinates(), payloads);
  });
  kernel.walk([&](ScatterReduceOp scatter) {
    collectOwnership(scatter.getCoordinates(), ValueRange(scatter.getValue()));
  });
  kernel.walk([&](HistogramOp histogram) {
    auto fragment = cast<FragmentType>(histogram.getValues().getType());
    for (Attribute attribute : fragment.getAxisMaps())
      ownershipSources.insert(cast<AxisMapAttr>(attribute).getSourceId());
  });

  if (auto roles =
          mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr)) {
    SmallVector<int64_t> updatedRoles(roles.asArrayRef());
    kernel.walk([&](WorksetCoordinateOp coordinate) {
      auto axis = coordinate->getAttrOfType<IntegerAttr>(worksetAxisAttr);
      if (!axis || axis.getInt() < 0 ||
          static_cast<size_t>(axis.getInt()) >= updatedRoles.size())
        return;
      auto crossesDataLoad = [&](Value root) {
        SmallVector<std::pair<Value, bool>> worklist{{root, false}};
        llvm::SmallDenseSet<Value> beforeLoad;
        llvm::SmallDenseSet<Value> afterLoad;
        while (!worklist.empty()) {
          auto [value, crossed] = worklist.pop_back_val();
          auto &visited = crossed ? afterLoad : beforeLoad;
          if (!visited.insert(value).second)
            continue;
          if (value == coordinate.getResult())
            return crossed;
          Operation *definition = value.getDefiningOp();
          if (!definition)
            continue;
          bool nextCrossed = crossed || isa<LoadOp, GatherOp>(definition);
          for (Value operand : definition->getOperands())
            worklist.emplace_back(operand, nextCrossed);
        }
        return false;
      };
      bool indirect = llvm::any_of(writeEffects, [&](const auto &effect) {
        return llvm::any_of(effect.coordinates, crossesDataLoad);
      });
      updatedRoles[axis.getInt()] = static_cast<int64_t>(
          indirect ? CoordinateRole::IndirectTraversal
                   : CoordinateRole::Workset);
    });
    mapping->setAttr(coordinateRolesAttr,
                     DenseI64ArrayAttr::get(module.getContext(), updatedRoles));
  }

  llvm::MapVector<uint64_t, SmallVector<MakeRangeOp>> axes;
  llvm::DenseMap<uint64_t, ParameterOp> parameters;
  llvm::SmallDenseSet<uint64_t> ownershipDimensions;
  llvm::SmallDenseSet<uint64_t> internalDimensions;
  auto dependsOnSource = [&](ValueRange values, uint64_t sourceId) {
    for (Value value : values) {
      auto fragment = dyn_cast<FragmentType>(value.getType());
      if (!fragment || failed(sourceAxis(fragment, sourceId)))
        continue;
      llvm::SmallPtrSet<Operation *, 8> ranges;
      llvm::SmallPtrSet<Operation *, 32> visited;
      collectProducerRanges(value, sourceId, ranges, visited);
      if (!ranges.empty())
        return true;
    }
    return false;
  };
  for (MakeRangeOp range : dynamicRanges) {
    uint64_t sourceId = range.getSourceId();
    if (!ownershipSources.contains(sourceId)) {
      internalTraversalRanges.insert(range.getOperation());
      continue;
    }
    bool requiredByEveryEffect = llvm::all_of(writeEffects, [&](const auto &effect) {
      return dependsOnSource(effect.coordinates, sourceId) ||
             dependsOnSource(effect.payloads, sourceId);
    });
    if (!requiredByEveryEffect)
      internalTraversalRanges.insert(range.getOperation());
  }
  if (!ownershipOnly)
    for (MakeRangeOp range : dynamicRanges) {
      if (!internalTraversalRanges.contains(range.getOperation()) ||
          structuredTraversalRanges.contains(range.getOperation()))
        continue;
      if (failed(requireFullDimensionCoverage(kernel, range.getResult(), 0)))
        return range.emitOpError(
                   "internal pointwise traversal has no exact full-coverage realization")
               << "; source_id=" << range.getSourceId()
               << ", fragment=" << range.getResult().getType();
    }
  for (MakeRangeOp range : dynamicRanges) {
    if (ownershipOnly &&
        internalTraversalRanges.contains(range.getOperation())) {
      FailureOr<ParameterOp> internalParameter = blockingParameter(kernel, range);
      FailureOr<uint64_t> internalDimension =
          succeeded(internalParameter)
              ? parameterDimension(*internalParameter)
              : FailureOr<uint64_t>(failure());
      if (succeeded(internalDimension)) {
        internalDimensions.insert(*internalDimension);
      } else if (auto dimension =
                     range->getAttrOfType<IntegerAttr>(sourceDimensionAttr)) {
        internalDimensions.insert(dimension.getInt());
      }
      continue;
    }
    if (ownershipOnly &&
        !ownershipSources.contains(range.getSourceId()))
      continue;
    FailureOr<ParameterOp> parameter = blockingParameter(kernel, range);
    if (failed(parameter) && ownershipOnly &&
        ownershipSources.contains(range.getSourceId()) &&
        !internalTraversalRanges.contains(range.getOperation())) {
      auto logicalExtent = range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      const bool dynamicSubregion = range->hasAttr(sourceSubregionAttr);
      if (dynamicSubregion ||
          (logicalExtent && physicalExtent.getKind() ==
                                static_cast<uint32_t>(PhysicalExprKind::Constant))) {
        SmallVector<int64_t> candidates;
        if (dynamicSubregion) {
          candidates.assign({1, 2, 4, 8, 16, 32, 64, 128, 256});
        } else {
          for (int64_t candidate : {8, 16, 32, 64, 128, 256, 512, 1024,
                                    2048, 4096})
            if (candidate <= logicalExtent.value())
              candidates.push_back(candidate);
          if (candidates.empty() || candidates.back() != logicalExtent.value())
            candidates.push_back(logicalExtent.value());
        }
        OpBuilder builder(&kernel.getBody().front(),
                          kernel.getBody().front().begin());
        auto schema = ParameterAttr::get(
            module.getContext(),
            builder.getStringAttr(
                ("FRAGMENT_S" + Twine(range.getSourceId())).str()),
            static_cast<uint32_t>(ParameterRole::OwnershipN),
            DenseI64ArrayAttr::get(module.getContext(), candidates));
        parameter = builder.create<ParameterOp>(
            range.getLoc(), builder.getIndexType(), schema);
        if (auto dimension =
                range->getAttrOfType<IntegerAttr>(sourceDimensionAttr))
          (*parameter)->setAttr(dimensionAttr, dimension);
        retargetSourceExtent(range.getResult(), range.getSourceId(),
                             fragmentExtent(*parameter));
      }
    }
    if (succeeded(parameter))
      retargetSourceExtent(range.getResult(), range.getSourceId(),
                           fragmentExtent(*parameter));
    FailureOr<uint64_t> dimensionId =
        failed(parameter) ? FailureOr<uint64_t>(failure())
                          : parameterDimension(*parameter);
    if (failed(parameter) || failed(dimensionId)) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no canonical blocking dimension");
      diagnostic << "; source_id=" << range.getSourceId() << ", fragment="
                 << range.getResult().getType();
      if (succeeded(parameter))
        diagnostic << ", parameter="
                   << parameter->getParameter().getName().getValue();
      return failure();
    }
    auto found = parameters.find(*dimensionId);
    if (found != parameters.end() && found->second != *parameter)
      return range.emitOpError(
          "one logical dimension has multiple blocking parameters");
    parameters[*dimensionId] = *parameter;
    axes[*dimensionId].push_back(range);
    if (internalTraversalRanges.contains(range.getOperation()))
      internalDimensions.insert(*dimensionId);
    if (ownershipSources.contains(range.getSourceId()) &&
        !internalTraversalRanges.contains(range.getOperation()))
      ownershipDimensions.insert(*dimensionId);
  }
  if (!ownershipOnly)
    ownershipDimensions.clear();

  if (ownershipOnly) {
    SmallVector<uint64_t> nonLiftableDimensions;
    for (uint64_t dimensionId : ownershipDimensions) {
      bool losesPromotedAxis = false;
      for (MakeRangeOp range : axes.lookup(dimensionId)) {
        uint64_t sourceId = range.getSourceId();
        kernel.walk([&](BroadcastOp broadcast) {
          auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
          auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
          if (!source || !target || failed(sourceAxis(source, sourceId)) ||
              succeeded(sourceAxis(target, sourceId)))
            return;
          losesPromotedAxis = true;
        });
        if (losesPromotedAxis)
          break;
      }
      if (losesPromotedAxis)
        nonLiftableDimensions.push_back(dimensionId);
    }
    // A BroadcastOp whose result does not carry a promoted source axis can
    // only expand a singleton value along its existing target axes.  Keeping
    // a non-unit candidate for that dimension would create an invalid
    // equal-rank broadcast and leave provider autotune to discover the error.
    // Until the physical value graph contains an explicit rank-lifted result,
    // keep that workset dimension as a scalar program coordinate.
    for (uint64_t dimensionId : nonLiftableDimensions)
      ownershipDimensions.erase(dimensionId);
  }

  if (ownershipOnly) {
    llvm::SmallDenseSet<uint64_t> internalOwnershipDimensions;
    for (uint64_t dimensionId : ownershipDimensions) {
      if (isStaticAxis(dimensionId))
        continue;
      if (llvm::any_of(axes.lookup(dimensionId), [](MakeRangeOp range) {
            return !range->hasAttr(worksetCoordinateRangeAttr);
          }))
        internalOwnershipDimensions.insert(dimensionId);
    }
    if (!internalOwnershipDimensions.empty()) {
      SmallVector<uint64_t> scalarWorksetDimensions;
      for (uint64_t dimensionId : ownershipDimensions) {
        if (isStaticAxis(dimensionId) ||
            internalOwnershipDimensions.contains(dimensionId))
          continue;
        if (llvm::all_of(axes.lookup(dimensionId), [](MakeRangeOp range) {
              return range->hasAttr(worksetCoordinateRangeAttr);
            }))
          scalarWorksetDimensions.push_back(dimensionId);
      }
      for (uint64_t dimensionId : scalarWorksetDimensions)
        ownershipDimensions.erase(dimensionId);
    }
  }

  // Static logical axes are local tensor structure, not independent runtime
  // workset dimensions.  They therefore do not consume the two dynamic
  // ownership axes used for pointwise lane promotion.  Counting them against
  // that budget silently scalarizes an adjacent dynamic axis whenever a
  // pointwise value has a fixed innermost vector dimension.
  unsigned dynamicOwnershipCount = llvm::count_if(
      ownershipDimensions,
      [](uint64_t dimensionId) { return !isStaticAxis(dimensionId); });
  if (ownershipOnly && dynamicOwnershipCount > 2) {
    SmallVector<std::pair<int64_t, uint64_t>> ranked;
    for (uint64_t dimensionId : ownershipDimensions) {
      if (isStaticAxis(dimensionId))
        continue;
      int64_t sourceAxis = -1;
      for (MakeRangeOp range : axes.lookup(dimensionId)) {
        kernel.walk([&](StoreOp store) {
          for (auto [coordinateIndex, coordinate] :
               llvm::enumerate(store.getCoordinates())) {
            if (coordinateIndex >= store.getSourceAxes().size())
              continue;
            llvm::SmallPtrSet<Operation *, 8> ranges;
            llvm::SmallPtrSet<Operation *, 32> visited;
            collectProducerRanges(coordinate, range.getSourceId(), ranges,
                                  visited);
            if (!ranges.empty())
              sourceAxis = std::max(sourceAxis,
                                    store.getSourceAxes()[coordinateIndex]);
          }
        });
      }
      ranked.emplace_back(sourceAxis, dimensionId);
    }
    llvm::sort(ranked, [](const auto &lhs, const auto &rhs) {
      return lhs.first != rhs.first ? lhs.first > rhs.first
                                    : lhs.second > rhs.second;
    });
    llvm::SmallDenseSet<uint64_t> selected;
    for (uint64_t dimensionId : ownershipDimensions)
      if (isStaticAxis(dimensionId))
        selected.insert(dimensionId);
    for (auto [_, dimension] : ArrayRef(ranked).take_front(2))
      selected.insert(dimension);
    ownershipDimensions = std::move(selected);
  }

  if (ownershipOnly) {
    auto unitExtent = expression(module.getContext(), PhysicalExprKind::Constant, 1);
    for (auto [dimensionId, ranges] : axes) {
      if (ownershipDimensions.contains(dimensionId))
        continue;
      for (MakeRangeOp range : ranges) {
        if (!range->hasAttr(worksetCoordinateRangeAttr) ||
            internalTraversalRanges.contains(range.getOperation()))
          continue;
        // An unselected workset coordinate remains a scalar program-grid
        // coordinate.  KIR construction provisionally gives every dynamic ABI
        // dimension a fragment parameter; once ownership selects the actual
        // lane axes, that provisional parameter must not remain in live value
        // types or the provider tuning surface.
        retargetSourceExtent(range.getResult(), range.getSourceId(), unitExtent);
      }
    }
  }

  for (uint64_t dimensionId : ownershipDimensions) {
    ParameterOp parameter = parameters.lookup(dimensionId);
    if (!parameter || isStaticAxis(dimensionId) ||
        parameter.getParameter().getRole() ==
            static_cast<uint32_t>(ParameterRole::ScanChunk))
      continue;
    parameter->removeAttr(coverageDimensionAttr);
    auto schema = ParameterAttr::get(
        module.getContext(), parameter.getParameter().getName(),
        static_cast<uint32_t>(ParameterRole::OwnershipN),
        DenseI64ArrayAttr::get(module.getContext(),
                               {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                1024, 2048, 4096, 8192, 16384, 32768,
                                65536}));
    parameter->setAttr("parameter", schema);
  }

  if (ownershipOnly) {
    llvm::MapVector<uint64_t, SmallVector<MakeRangeOp>> selectedAxes;
    SmallVector<MakeRangeOp> selectedRanges;
    for (auto [dimensionId, ranges] : axes) {
      if (!ownershipDimensions.contains(dimensionId))
        continue;
      for (MakeRangeOp range : ranges) {
        if (internalTraversalRanges.contains(range.getOperation()))
          continue;
        selectedAxes[dimensionId].push_back(range);
        selectedRanges.push_back(range);
      }
    }
    axes = std::move(selectedAxes);
    dynamicRanges = std::move(selectedRanges);
    if (dynamicRanges.empty())
      return success();
  }

  OpBuilder mappingBuilder(mapping);
  SmallVector<Value> runtimeExtents(mapping.getExtents());
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  SmallVector<Type> coordinateTypes(mapping.getResultTypes());
  llvm::DenseMap<uint64_t, Value> tileCoordinates;
  llvm::DenseMap<uint64_t, Value> logicalDimensions;
  llvm::DenseMap<uint64_t, unsigned> mappedAxes;
  auto mappingDimension = [&](Attribute attribute) -> FailureOr<uint64_t> {
    FailureOr<uint64_t> blocked = blockedDimension(attribute);
    if (succeeded(blocked))
      return blocked;
    auto physical = dyn_cast<PhysicalExprAttr>(attribute);
    if (!physical ||
        physical.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Dimension))
      return failure();
    StringRef name = physical.getSymbol().getValue();
    if (!name.consume_front("D"))
      return failure();
    uint64_t dimension = 0;
    return name.getAsInteger(10, dimension)
               ? FailureOr<uint64_t>(failure())
               : FailureOr<uint64_t>(dimension);
  };
  std::optional<unsigned> reusableUnitAxis;
  for (auto [axis, extent] : llvm::enumerate(mapping.getLaunchExtents())) {
    FailureOr<uint64_t> dimension = mappingDimension(extent);
    if (succeeded(dimension) && axis < mapping.getCoordinates().size()) {
      tileCoordinates[*dimension] = mapping.getCoordinates()[axis];
      mappedAxes[*dimension] = axis;
    }
    auto physical = dyn_cast<PhysicalExprAttr>(extent);
    if (!reusableUnitAxis && axis < mapping.getCoordinates().size() &&
        mapping.getCoordinates()[axis].use_empty() && physical &&
        physical.getKind() ==
            static_cast<uint32_t>(PhysicalExprKind::Constant) &&
        physical.getValue() == 1)
      reusableUnitAxis = axis;
  }
  bool mappingChanged = false;
  SmallVector<std::pair<uint64_t, unsigned>> reusedCoordinates;
  SmallVector<uint64_t> appendedCoordinates;
  for (auto [dimensionId, ranges] : axes) {
    MakeRangeOp range = ranges.front();
    ParameterOp parameter = parameters.lookup(dimensionId);
    FailureOr<Value> dimension = failure();
    arith::ConstantIndexOp staticExtent;
    std::optional<uint64_t> sourceDimension;
    if (auto identity =
            range->getAttrOfType<IntegerAttr>(sourceDimensionAttr))
      sourceDimension = identity.getInt();
    if (isStaticAxis(dimensionId)) {
      staticExtent = range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
      if (staticExtent)
        dimension = Value(mappingBuilder.create<arith::ConstantIndexOp>(
            mapping.getLoc(), staticExtent.value()));
      else if (sourceDimension)
        dimension = dimensionArgument(kernel, *sourceDimension);
    } else {
      dimension = dimensionArgument(kernel, dimensionId);
    }
    if (!parameter) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no launch-visible dimension or blocking parameter");
      diagnostic << "; dimension=" << dimensionId << ", fragment="
                 << range.getResult().getType();
      return failure();
    }
    if (!ownershipDimensions.contains(dimensionId))
      continue;
    if (failed(dimension)) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic ownership range has no launch-visible logical dimension");
      diagnostic << "; dimension=" << dimensionId << ", fragment="
                 << range.getResult().getType() << ", parameter="
                 << parameter.getParameter().getName().getValue();
      return failure();
    }
    logicalDimensions[dimensionId] = *dimension;
    Value one = mappingBuilder.create<arith::ConstantIndexOp>(mapping.getLoc(), 1);
    Value adjusted = mappingBuilder.create<BinaryOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), *dimension,
        mappingBuilder.create<BinaryOp>(mapping.getLoc(),
                                        mappingBuilder.getIndexType(),
                                        parameter.getResult(), one, 1),
        0);
    Value tiles = mappingBuilder.create<BinaryOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), adjusted,
        parameter.getResult(), 4);
    PhysicalExprAttr logical;
    if (isStaticAxis(dimensionId) && staticExtent) {
      logical = expression(module.getContext(), PhysicalExprKind::Constant,
                           staticExtent.value());
    } else {
      uint64_t logicalDimension =
          isStaticAxis(dimensionId) && sourceDimension ? *sourceDimension
                                                       : dimensionId;
      if (isStaticAxis(dimensionId) && !sourceDimension)
        return range.emitOpError(
            "range-local ownership axis lost its logical dimension extent");
      logical = expression(module.getContext(), PhysicalExprKind::Dimension, 0,
                           ("D" + Twine(logicalDimension)).str());
    }
    PhysicalExprAttr tile = expression(
        module.getContext(), PhysicalExprKind::Parameter, 0,
        parameter.getParameter().getName().getValue());
    PhysicalExprAttr launch = binaryExpression(
        module.getContext(), PhysicalExprKind::CeilDiv, logical, tile);
    auto mapped = mappedAxes.find(dimensionId);
    if (mapped != mappedAxes.end() || reusableUnitAxis) {
      unsigned axis = mapped != mappedAxes.end() ? mapped->second
                                                 : *reusableUnitAxis;
      runtimeExtents[axis] = tiles;
      launchExtents[axis] = launch;
      reusedCoordinates.emplace_back(dimensionId, axis);
      if (mapped == mappedAxes.end())
        reusableUnitAxis.reset();
    } else {
      runtimeExtents.push_back(tiles);
      coordinateTypes.push_back(mappingBuilder.getIndexType());
      launchExtents.push_back(launch);
      appendedCoordinates.push_back(dimensionId);
    }
    mappingChanged = true;
  }

  if (mappingChanged) {
    auto replacement = mappingBuilder.create<DelinearizeOp>(
        mapping.getLoc(), coordinateTypes, mapping.getLinear(), runtimeExtents,
        mappingBuilder.getArrayAttr(launchExtents));
    SmallVector<int64_t> coordinateRoles(
        replacement.getNumResults(),
        static_cast<int64_t>(CoordinateRole::Unspecified));
    if (auto existing =
            mapping->getAttrOfType<DenseI64ArrayAttr>(coordinateRolesAttr))
      if (existing.size() == mapping.getNumResults())
        llvm::copy(existing.asArrayRef(), coordinateRoles.begin());
    auto ownershipRole = [&](uint64_t dimensionId) {
      return llvm::any_of(axes.lookup(dimensionId), [](MakeRangeOp range) {
               return range->hasAttr(worksetCoordinateRangeAttr);
             })
                 ? CoordinateRole::TiledWorkset
                 : CoordinateRole::PointwiseOwnership;
    };
    for (auto [dimensionId, axis] : reusedCoordinates) {
      if (coordinateRoles[axis] !=
          static_cast<int64_t>(CoordinateRole::IndirectTraversal))
        coordinateRoles[axis] =
            static_cast<int64_t>(ownershipRole(dimensionId));
    }
    unsigned appendedAxis = mapping.getNumResults();
    for (uint64_t _ : appendedCoordinates) {
      (void)_;
      coordinateRoles[appendedAxis++] =
          static_cast<int64_t>(ownershipRole(_));
    }
    replacement->setAttr(
        coordinateRolesAttr,
        DenseI64ArrayAttr::get(module.getContext(), coordinateRoles));
    for (auto [axis, pair] : llvm::enumerate(llvm::zip(
             mapping.getCoordinates(),
             replacement.getCoordinates().take_front(
                 mapping.getCoordinates().size())))) {
      Value oldCoordinate = std::get<0>(pair);
      Value newCoordinate = std::get<1>(pair);
      FailureOr<uint64_t> dimension =
          axis < mapping.getLaunchExtents().size()
              ? mappingDimension(mapping.getLaunchExtents()[axis])
              : FailureOr<uint64_t>(failure());
      if (succeeded(dimension) && ownershipDimensions.contains(*dimension)) {
        ParameterOp parameter = parameters.lookup(*dimension);
        if (!parameter)
          return kernel.emitError(
              "pointwise ownership mapping lost its blocking parameter");
        newCoordinate = mappingBuilder.create<BinaryOp>(
            mapping.getLoc(), mappingBuilder.getIndexType(), newCoordinate,
            parameter.getResult(), 2);
        oldCoordinate.replaceAllUsesWith(newCoordinate);
        PhysicalExprAttr extent = expression(
            module.getContext(), PhysicalExprKind::Parameter, 0,
            parameter.getParameter().getName().getValue());
        llvm::SmallDenseSet<uint64_t> propagatedSources;
        for (MakeRangeOp range : axes.lookup(*dimension))
          if (propagatedSources.insert(range.getSourceId()).second)
            retargetSourceExtent(newCoordinate, range.getSourceId(), extent);
        continue;
      }
      oldCoordinate.replaceAllUsesWith(newCoordinate);
    }
    for (auto [dimensionId, axis] : reusedCoordinates)
      tileCoordinates[dimensionId] = replacement.getCoordinates()[axis];
    unsigned extra = mapping.getCoordinates().size();
    for (uint64_t dimensionId : appendedCoordinates)
      tileCoordinates[dimensionId] = replacement.getCoordinates()[extra++];
    mapping.erase();

    PhysicalExprAttr total = cast<PhysicalExprAttr>(launchExtents.front());
    for (Attribute extent : llvm::drop_begin(launchExtents))
      total = binaryExpression(module.getContext(), PhysicalExprKind::Multiply,
                               total, cast<PhysicalExprAttr>(extent));
    kernel->setAttr(programSpaceAttr,
                    ArrayAttr::get(module.getContext(), {total}));
  }

  llvm::DenseMap<Value, Value> rangePredicates =
      std::move(fixedRangePredicates);
  for (MakeRangeOp range : dynamicRanges) {
    OpBuilder builder(range);
    FailureOr<ParameterOp> parameter = blockingParameter(kernel, range);
    FailureOr<uint64_t> dimensionId =
        failed(parameter) ? FailureOr<uint64_t>(failure())
                          : parameterDimension(*parameter);
    if (failed(parameter) || failed(dimensionId))
      return range.emitOpError(
                 "dynamic pointwise range lost its canonical blocking dimension")
             << "; source_id=" << range.getSourceId()
             << ", fragment=" << range.getResult().getType()
             << ", source_dimension="
             << range->getAttr(sourceDimensionAttr);
    Value tileCoordinate = tileCoordinates.lookup(*dimensionId);
    if (!tileCoordinate && internalDimensions.contains(*dimensionId))
      tileCoordinate = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
    if (!tileCoordinate)
      return range.emitOpError(
                 "dynamic pointwise range has no physical tile coordinate")
             << "; dimension=" << *dimensionId
             << ", source_id=" << range.getSourceId()
             << ", ownership=" << ownershipDimensions.contains(*dimensionId)
             << ", internal=" << internalDimensions.contains(*dimensionId)
             << ", parameter_name="
             << (*parameter).getParameter().getName().getValue()
             << ", parameter_role=" << (*parameter).getParameter().getRole()
             << ", has_coverage="
             << (*parameter)->hasAttr(coverageDimensionAttr)
             << ", launch_extents=" << mapping->getAttr("launch_extents");
    Value tileOffset = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), tileCoordinate,
        parameter->getResult(), 2);
    Value start;
    Value end;
    if (range->hasAttr(worksetCoordinateRangeAttr)) {
      start = range.getStart();
      Value dimension = logicalDimensions.lookup(*dimensionId);
      if (!dimension && !isStaticAxis(*dimensionId)) {
        FailureOr<Value> runtimeDimension =
            dimensionArgument(kernel, *dimensionId);
        if (succeeded(runtimeDimension))
          dimension = *runtimeDimension;
      }
      if (!dimension)
        return range.emitOpError(
                   "workset coordinate range lost its logical dimension extent")
               << "; dimension=" << *dimensionId << ", source_id="
               << range.getSourceId() << ", parameter="
               << parameter->getParameter().getName().getValue()
               << ", source_dimension="
               << range->getAttr(sourceDimensionAttr);
      Value remaining = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), dimension, tileOffset, 1);
      Value remainingDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), remaining, range.getStep(), 2);
      end = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                     start, remainingDistance, 0);
    } else {
      Value scaledOffset = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), tileOffset, range.getStep(), 2);
      start = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getStart(), scaledOffset,
          0);
      Value logicalDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getExtent(),
          range.getStep(), 2);
      end = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                     range.getStart(), logicalDistance, 0);
    }
    auto sourceType = cast<FragmentType>(range.getResult().getType());
    PhysicalExprAttr tileExtent = fragmentExtent(*parameter);
    auto blockedType = FragmentType::get(
        module.getContext(), sourceType.getElementType(),
        builder.getArrayAttr({tileExtent}), sourceType.getAxisMaps(),
        sourceType.getValidity(),
        sourceType.getOwner());
    Value blocked = builder.create<MakeRangeOp>(
        range.getLoc(), blockedType, start, parameter->getResult(),
        range.getStep(), range.getSourceId(), range.getSourceAxis());
    for (StringRef attribute : {sourceSubregionAttr, sourceDimensionAttr})
      if (Attribute value = range->getAttr(attribute))
        blocked.getDefiningOp()->setAttr(attribute, value);
    if (failed(replayBlockedStorePayloads(kernel, range, blocked, tileExtent)))
      return failure();
    Value endFragment = builder.create<BroadcastOp>(range.getLoc(), blockedType, end);
    Value valid = builder.create<CompareOp>(range.getLoc(), predicateType(blockedType),
                                            blocked, endFragment, 2);
    range.getResult().replaceAllUsesWith(blocked);
    rangePredicates[blocked] = valid;
    range.erase();
  }

  if (failed(addTailValidity(kernel, rangePredicates,
                             /*includeStores=*/true)))
    return failure();
  if (failed(realizeDistributedHistograms(kernel)))
    return failure();
  if (failed(alignExplicitBroadcastOperands(kernel)))
    return failure();
  if (failed(bindStructurallyRequiredStaticFragments(kernel)))
    return failure();
  eraseDeadPhysicalValues(kernel);
  return success();
}

LogicalResult realizePointwiseOwnership(ModuleOp module) {
  return realizePointwiseBlockingImpl(module, /*ownershipOnly=*/true);
}

LogicalResult realizePointwiseBlocking(ModuleOp module) {
  return realizePointwiseBlockingImpl(module, /*ownershipOnly=*/false);
}

} // namespace intent::gpu

#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"

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

#include <functional>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::gpu {
namespace {

Attribute dimensionAxisKey(MLIRContext *context, uint64_t dimension) {
  return IntegerAttr::get(IntegerType::get(context, 64), dimension);
}

Attribute sourceAxisKey(MLIRContext *context, PhysicalSourceAxis source) {
  return PhysicalSourceAttr::get(context, source.sourceId, source.sourceAxis,
                                 source.derived);
}

bool isSourceAxisKey(Attribute axis) { return isa<PhysicalSourceAttr>(axis); }

FailureOr<uint64_t> axisDimension(Attribute axis) {
  auto dimension = dyn_cast<IntegerAttr>(axis);
  return dimension && dimension.getInt() > 0
             ? FailureOr<uint64_t>(dimension.getInt())
             : FailureOr<uint64_t>(failure());
}

FailureOr<PhysicalSourceAxis> axisSource(Attribute axis) {
  auto source = dyn_cast<PhysicalSourceAttr>(axis);
  return source ? FailureOr<PhysicalSourceAxis>(PhysicalSourceAxis{
                      source.getSourceId(), source.getSourceAxis(),
                      source.getDerived()})
                : FailureOr<PhysicalSourceAxis>(failure());
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

FailureOr<uint64_t> rangeDimension(MakeRangeOp range) {
  FailureOr<int64_t> dimension = querySourceDimension(
      range.getResult().getType(),
      PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()});
  return succeeded(dimension) && *dimension > 0
             ? FailureOr<uint64_t>(*dimension)
             : FailureOr<uint64_t>(failure());
}

FailureOr<Attribute> parameterAxis(ParameterOp parameter) {
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (!binding.isExact())
    return failure();
  if (binding.source)
    return sourceAxisKey(parameter.getContext(), *binding.source);
  if (binding.dimension)
    return dimensionAxisKey(parameter.getContext(), *binding.dimension);
  return failure();
}

PhysicalExprAttr fragmentExtent(ParameterOp parameter) {
  ParameterAttr schema = parameter.getParameter();
  PhysicalParameterBinding binding = queryParameterBinding(parameter);
  if (binding.source && schema.getCandidates().size() == 1)
    return expression(parameter.getContext(), PhysicalExprKind::Constant,
                      schema.getCandidates()[0]);
  return expression(parameter.getContext(), PhysicalExprKind::Parameter, 0,
                    schema.getName().getValue());
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
    FailureOr<ParameterOp> declaration =
        queryParameterBySymbol(kernel, sourceProduct.parameter);
    if (failed(declaration))
      continue;
    ParameterOp parameter = *declaration;
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.source)
      continue;
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
    PhysicalParameterBinding binding = queryParameterBinding(parameter);
    if (!binding.isExact() || !binding.source)
      return parameter.emitOpError(
          "structural fragment parameter lost its typed source-axis binding");
    PhysicalSourceAxis source = *binding.source;
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
      retargetSourceExtent(root, source, fixedExtent);
    SmallVector<MakeRangeOp> ranges;
    kernel.walk([&](MakeRangeOp range) {
      if (sourceAxisIdentity(range) == source)
        ranges.push_back(range);
    });
    for (MakeRangeOp range : ranges) {
      retargetSourceExtent(range.getResult(), source, fixedExtent);
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
  return realizeFullCoverageDimension(kernel, source, axis);
}

LogicalResult requireScanFullCoverage(func::FuncOp kernel, Value source,
                                      uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  if (extent && extent.getKind() ==
                    static_cast<uint32_t>(PhysicalExprKind::Constant))
    return success();
  if (!extent || extent.getKind() !=
                     static_cast<uint32_t>(PhysicalExprKind::Dimension))
    return requireFullDimensionCoverage(kernel, source, axis);
  int64_t dimension = extent.getValue();
  if (dimension <= 0 ||
      failed(dimensionArgument(kernel, dimension)))
    return failure();
  std::string parameterName = ("FULL_D" + Twine(dimension)).str();
  ParameterOp parameter;
  kernel.walk([&](ParameterOp candidate) {
    if (!parameter &&
        candidate.getParameter().getName().getValue() == parameterName)
      parameter = candidate;
  });
  if (!parameter) {
    static constexpr int64_t candidates[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    OpBuilder builder(&kernel.getBody().front(),
                      kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), builder.getStringAttr(parameterName),
        static_cast<uint32_t>(ParameterRole::ScanChunk),
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    parameter = builder.create<ParameterOp>(source.getLoc(),
                                            builder.getIndexType(), schema);
  }
  parameter->setAttr(
      dimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
  parameter->setAttr(
      coverageDimensionAttr,
      IntegerAttr::get(IntegerType::get(kernel.getContext(), 64), dimension));
  PhysicalExprAttr covered = fragmentExtent(parameter);
  auto sourceMapping =
      cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact sourceRanges = analysis.axisRanges(source, axis);
  if (sourceRanges.state == PhysicalFactState::Unknown ||
      sourceRanges.roots.empty()) {
    PhysicalReplayFact replay = analysis.replayability(
        source, sourceAxisIdentity(sourceMapping),
        PhysicalReplayScope::ValueGraph, /*allowAccesses=*/false);
    if (!sourceRanges.roots.empty() || !replay.isReplayable()) {
      InFlightDiagnostic diagnostic = kernel.emitError(
          "scan source has no exact full-coverage range authority");
      diagnostic << "; source_type=" << source.getType();
      for (Operation *blocker : sourceRanges.blockers)
        diagnostic << ", blocker=" << blocker->getName();
      return failure();
    }
  }
  for (MakeRangeOp range : sourceRanges.roots) {
    FailureOr<uint64_t> rangeIdentity = rangeDimension(range);
    if (failed(rangeIdentity) ||
        *rangeIdentity != static_cast<uint64_t>(dimension) ||
        range->hasAttr(sourceSubregionAttr))
      return range.emitOpError(
          "scan source range does not match its full-coverage dimension");
    retargetDimensionExtent(range.getResult(), dimension, covered);
  }
  retargetDimensionExtent(source, dimension, covered);
  if (failed(
          bindFullCoverageDimension(kernel, dimension, parameter.getResult())))
    return kernel.emitError(
        "scan source could not bind its exact full-coverage parameter");
  return success();
}

LogicalResult requireStructuredReductionFullCoverage(func::FuncOp kernel,
                                                      Value source,
                                                      uint64_t axis) {
  auto fragment = dyn_cast<FragmentType>(source.getType());
  if (!fragment || axis >= fragment.getShape().size())
    return failure();
  auto extent = dyn_cast<PhysicalExprAttr>(fragment.getShape()[axis]);
  if (extent && extent.getKind() ==
                    static_cast<uint32_t>(PhysicalExprKind::Constant))
    return success();
  return realizeFullCoverageDimension(kernel, source, axis);
}

FragmentType predicateType(FragmentType source) {
  return FragmentType::get(source.getContext(), IntegerType::get(source.getContext(), 1),
                           source.getShape(), source.getAxisMaps(),
                           source.getValidity(),
                           source.getOwner());
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges);
void collectProducerRanges(Value value, PhysicalSourceAxis source,
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
    FailureOr<Value> broadcast = materializeBroadcastToFragment(builder, location, existing, target);
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
          materializeBroadcastToFragment(builder, location, found->second, target);
      if (failed(broadcast))
        return failure();
      result = result ? Value(builder.create<BinaryOp>(
                            location, target, result, *broadcast,
                            BinaryOperator::LogicalAnd))
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
  SmallVector<HistogramOp> histograms;
  kernel.walk([&](LoadOp load) { loads.push_back(load); });
  kernel.walk([&](StoreOp store) { stores.push_back(store); });
  kernel.walk([&](HistogramOp histogram) { histograms.push_back(histogram); });

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
          materializeBroadcastToFragment(builder, load.getLoc(), load.getFill(), valueType);
      if (failed(broadcast))
        return load.emitOpError("could not broadcast the existing load fill");
      fill = *broadcast;
    } else {
      FailureOr<Value> zero = materializeZeroFragment(builder, load.getLoc(), valueType);
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

  for (HistogramOp histogram : histograms) {
    auto valueType = dyn_cast<FragmentType>(histogram.getValues().getType());
    if (!valueType)
      return histogram.emitOpError(
          "pointwise blocked histogram must consume a physical fragment");
    FragmentType validType = predicateType(valueType);
    OpBuilder builder(histogram);
    FailureOr<Value> existing = materializeBroadcastToFragment(
        builder, histogram.getLoc(), histogram.getValid(), validType);
    if (failed(existing))
      return histogram.emitOpError(
          "could not broadcast the existing histogram validity");
    Value valid = *existing;
    bool affected = false;
    for (auto [rangeValue, predicate] : rangePredicates) {
      auto range = rangeValue.getDefiningOp<MakeRangeOp>();
      if (!range)
        continue;
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectProducerRanges(
          histogram.getValues(),
          PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()},
          ranges);
      if (!ranges.contains(range.getOperation()))
        continue;
      FailureOr<Value> broadcast =
          materializeBroadcastToFragment(builder, histogram.getLoc(), predicate, validType);
      if (failed(broadcast))
        return histogram.emitOpError(
            "could not project pointwise tail validity onto histogram values");
      valid = builder.create<BinaryOp>(histogram.getLoc(), validType, valid,
                                       *broadcast, BinaryOperator::LogicalAnd);
      affected = true;
    }
    if (!affected)
      continue;
    auto replacement = builder.create<HistogramOp>(
        histogram.getLoc(), histogram.getResult().getType(),
        histogram.getValues(), histogram.getBins(), valid);
    if (Attribute origin = histogram->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    histogram.getResult().replaceAllUsesWith(replacement.getResult());
    histogram.erase();
  }

  if (!includeStores)
    return success();
  for (StoreOp store : stores) {
    bool affected = llvm::any_of(store.getCoordinates(), [&](Value coordinate) {
      return hasTailPredicate(coordinate, rangePredicates);
    });
    if (!affected)
      continue;
    OpBuilder builder(store);
    Value payload = store.getValue();
    auto valueType = dyn_cast<FragmentType>(payload.getType());
    if (!valueType) {
      if (!isa<IntegerType, FloatType, IndexType>(payload.getType()))
        return store.emitOpError(
            "pointwise blocked store has no projectable value schema");
      SmallVector<Attribute> shape;
      SmallVector<Attribute> mappings;
      std::optional<uint64_t> owner;
      for (Value coordinate : store.getCoordinates()) {
        auto fragment = dyn_cast<FragmentType>(coordinate.getType());
        if (!fragment)
          continue;
        if (owner && *owner != fragment.getOwner())
          return store.emitOpError(
              "pointwise store coordinates have conflicting ownership");
        owner = fragment.getOwner();
        for (auto [extent, attribute] :
             llvm::zip(fragment.getShape(), fragment.getAxisMaps())) {
          auto mapping = cast<AxisMapAttr>(attribute);
          auto found = llvm::find_if(mappings, [&](Attribute existing) {
            auto axis = cast<AxisMapAttr>(existing);
            return sourceAxisIdentity(axis) == sourceAxisIdentity(mapping);
          });
          if (found != mappings.end())
            continue;
          shape.push_back(extent);
          mappings.push_back(AxisMapAttr::get(
              store.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
              mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
        }
      }
      if (shape.empty())
        return store.emitOpError(
            "pointwise blocked store has no coordinate fragment authority");
      auto target = FragmentType::get(
          store.getContext(), payload.getType(),
          ArrayAttr::get(store.getContext(), shape),
          ArrayAttr::get(store.getContext(), mappings), /*validity=*/1,
          owner.value_or(1));
      FailureOr<Value> projected = projectPhysicalValueToSchema(
          builder, store.getLoc(), payload, target);
      if (failed(projected))
        return store.emitOpError(
            "pointwise store value cannot be projected to its coordinate schema");
      payload = *projected;
      valueType = target;
    }
    FailureOr<Value> valid = accessValidity(
        builder, store.getLoc(), store.getCoordinates(), rangePredicates,
        valueType, store.getValid());
    if (failed(valid))
      return store.emitOpError("could not form pointwise store validity");
    auto replacement = builder.create<StoreOp>(
        store.getLoc(), store.getResource(), store.getCoordinates(),
        payload, *valid, store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      replacement->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

void collectProducerRanges(Value value, PhysicalSourceAxis source,
                           llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return;
  if (!queryFragmentAxis(value.getType(), source).isExact())
    return;
  PhysicalProgramAnalysis analysis(kernel);
  PhysicalRangeFact fact = analysis.sourceRanges(value, source);
  if (fact.state == PhysicalFactState::Unknown)
    return;
  for (MakeRangeOp range : fact.roots)
    if (sourceAxisIdentity(range) == source)
      ranges.insert(range.getOperation());
}

FragmentType replaceSourceExtent(FragmentType source, PhysicalSourceAxis logical,
                                 PhysicalExprAttr extent) {
  PhysicalAxisProjection axis = queryFragmentAxis(source, logical);
  if (!axis.isExact())
    return source;
  SmallVector<Attribute> shape(source.getShape().begin(), source.getShape().end());
  shape[axis.fragmentAxis] = extent;
  return FragmentType::get(source.getContext(), source.getElementType(),
                           ArrayAttr::get(source.getContext(), shape),
                           source.getAxisMaps(), source.getValidity(),
                           source.getOwner());
}

bool containsSource(Value value, PhysicalSourceAxis source) {
  return queryFragmentAxis(value.getType(), source).isExact();
}

bool containsTraversal(Value value, PhysicalSourceAxis source,
                       int64_t dimension) {
  return llvm::any_of(queryFragmentAxes(value.getType(), source),
                      [&](const PhysicalAxisProjection &projection) {
                        return projection.dimensionId == dimension;
                      });
}

FailureOr<FragmentType> coordinateValueSchema(Type element,
                                              ValueRange coordinates) {
  SmallVector<Attribute> shape;
  SmallVector<Attribute> mappings;
  std::optional<uint64_t> owner;
  for (Value coordinate : coordinates) {
    auto fragment = dyn_cast<FragmentType>(coordinate.getType());
    if (!fragment)
      continue;
    if (owner && *owner != fragment.getOwner())
      return failure();
    owner = fragment.getOwner();
    for (auto [extent, attribute] :
         llvm::zip(fragment.getShape(), fragment.getAxisMaps())) {
      auto mapping = cast<AxisMapAttr>(attribute);
      auto found = llvm::find_if(mappings, [&](Attribute existing) {
        return sourceAxisIdentity(cast<AxisMapAttr>(existing)) ==
               sourceAxisIdentity(mapping);
      });
      if (found != mappings.end()) {
        unsigned axis = std::distance(mappings.begin(), found);
        if (shape[axis] != extent)
          return failure();
        continue;
      }
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          element.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    }
  }
  if (shape.empty())
    return failure();
  return FragmentType::get(element.getContext(), element,
                           ArrayAttr::get(element.getContext(), shape),
                           ArrayAttr::get(element.getContext(), mappings),
                           /*validity=*/1, owner.value_or(1));
}

FailureOr<Value> replayPointwiseValue(OpBuilder &builder, Value value,
                                      PhysicalSourceAxis source,
                                      PhysicalExprAttr blockedExtent,
                                      Value blockedRange, Value blockedValidity,
                                      Operation *insertionAnchor,
                                      IRMapping &mapping) {
  if (Value replacement = mapping.lookupOrNull(value))
    return replacement;
  auto kernel = value.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return failure();
  auto blocked = blockedRange.getDefiningOp<MakeRangeOp>();
  FailureOr<int64_t> blockedDimension =
      blocked ? queryRangeDimension(blocked) : FailureOr<int64_t>(failure());
  if (failed(blockedDimension))
    return insertionAnchor->emitOpError(
        "pointwise replay has no exact blocked-dimension authority");
  if (!containsTraversal(value, source, *blockedDimension)) {
    DominanceInfo dominance(kernel);
    if (dominance.dominates(value, insertionAnchor))
      return value;
  }
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return insertionAnchor->emitOpError(
        "pointwise replay cannot rematerialize a non-dominating block argument");
  if (producer->getNumResults() != 1)
    return producer->emitOpError(
        "pointwise replay requires a single-result producer");
  if (auto range = dyn_cast<MakeRangeOp>(producer)) {
    if (sourceAxisIdentity(range) == source) {
      mapping.map(value, blockedRange);
      return blockedRange;
    }
    FailureOr<int64_t> rangeDimension = queryRangeDimension(range);
    auto originalType = dyn_cast<FragmentType>(range.getResult().getType());
    if (!blocked || failed(rangeDimension) || failed(blockedDimension) ||
        *rangeDimension != *blockedDimension || !originalType ||
        originalType.getShape().size() != 1) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "pointwise replay reached a range without an exact shared dimension relation");
      if (succeeded(rangeDimension))
        diagnostic << "; source_dimension=" << *rangeDimension;
      if (succeeded(blockedDimension))
        diagnostic << "; blocked_dimension=" << *blockedDimension;
      return failure();
    }
    auto projectedType = FragmentType::get(
        originalType.getContext(), originalType.getElementType(),
        ArrayAttr::get(originalType.getContext(), {blockedExtent}),
        originalType.getAxisMaps(), originalType.getValidity(),
        originalType.getOwner());
    Value projected = builder.create<MakeRangeOp>(
        range.getLoc(), projectedType, blocked.getStart(), blocked.getExtent(),
        blocked.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    if (Attribute subregion = range->getAttr(sourceSubregionAttr))
      projected.getDefiningOp()->setAttr(sourceSubregionAttr, subregion);
    mapping.map(value, projected);
    return projected;
  }
  PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
      value, source, PhysicalReplayScope::ValueGraph,
      /*allowAccesses=*/true, insertionAnchor, *blockedDimension);
  if (!replay.isReplayable()) {
    InFlightDiagnostic diagnostic = producer->emitOpError(
        "pointwise value has no exact insertion-point replay fact");
    for (Operation *blocker : replay.blockers)
      diagnostic << "; blocker=" << blocker->getName();
    return failure();
  }
  for (Value operand : producer->getOperands()) {
    FailureOr<Value> replacement = replayPointwiseValue(
        builder, operand, source, blockedExtent, blockedRange,
        blockedValidity, insertionAnchor, mapping);
    if (failed(replacement))
      return failure();
    if (*replacement != operand && !mapping.lookupOrNull(operand))
      mapping.map(operand, *replacement);
  }
  auto result = dyn_cast<FragmentType>(producer->getResult(0).getType());
  FragmentType resultType =
      result ? replaceSourceExtent(result, source, blockedExtent)
             : FragmentType();
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
    FailureOr<Value> projected = materializeBroadcastToFragment(
        builder, producer->getLoc(), blockedValidity, predicateType(resultType));
    if (failed(projected))
      return failure();
    if (!valid)
      return *projected;
    FailureOr<Value> original =
        materializeBroadcastToFragment(builder, producer->getLoc(), valid, predicateType(resultType));
    if (failed(original))
      return failure();
    return Value(builder.create<BinaryOp>(producer->getLoc(),
                                          predicateType(resultType), *original,
                                          *projected,
                                          BinaryOperator::LogicalAnd));
  };
  Value replayed;
  if (auto load = dyn_cast<LoadOp>(producer)) {
    if (!resultType)
      return failure();
    FailureOr<Value> valid = accessValidity(load.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(load.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, producer->getLoc(), resultType);
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
    if (!resultType)
      return failure();
    FailureOr<Value> valid = accessValidity(gather.getValid());
    if (failed(valid))
      return failure();
    Value fill = mapped(gather.getFill());
    if (*valid && !fill) {
      FailureOr<Value> zero = materializeZeroFragment(builder, producer->getLoc(), resultType);
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
             reshape && !containsSource(reshape.getValue(), source)) {
    if (!resultType)
      return failure();
    replayed = builder.create<BroadcastOp>(producer->getLoc(), resultType,
                                           mapped(reshape.getValue()));
  } else {
    Operation *clone = builder.clone(*producer, mapping);
    if (resultType)
      clone->getResult(0).setType(resultType);
    replayed = clone->getResult(0);
  }
  if (!mapping.lookupOrNull(value))
    mapping.map(value, replayed);
  return replayed;
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

  PhysicalSourceAxis logicalSource{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
  std::string name = ("POINTWISE_CHUNK_S" + Twine(logicalSource.sourceId) +
                      "_A" + Twine(logicalSource.sourceAxis))
                         .str();
  ParameterOp chunk;
  bool ambiguousChunk = false;
  kernel.walk([&](ParameterOp parameter) {
    auto source =
        parameter->getAttrOfType<PhysicalSourceAttr>(parameterSourceAttr);
    if (!parameter->hasAttr(pointwiseChunkAttr) || !source ||
        !(PhysicalSourceAxis{source.getSourceId(), source.getSourceAxis(),
                             source.getDerived()} == logicalSource))
      return;
    if (chunk && chunk != parameter)
      ambiguousChunk = true;
    else
      chunk = parameter;
  });
  if (ambiguousChunk)
    return range.emitOpError(
        "pointwise traversal has multiple parameters for one source axis");
  if (!chunk) {
    static constexpr int64_t candidates[] = {16, 32, 64, 128, 256, 512,
                                             1024, 2048, 4096};
    OpBuilder entry(&kernel.getBody().front(), kernel.getBody().front().begin());
    auto schema = ParameterAttr::get(
        kernel.getContext(), entry.getStringAttr(name),
        static_cast<uint32_t>(ParameterRole::OwnershipN),
        DenseI64ArrayAttr::get(kernel.getContext(), candidates));
    chunk = entry.create<ParameterOp>(range.getLoc(), entry.getIndexType(), schema);
    chunk->setAttr(parameterSourceAttr,
                   PhysicalSourceAttr::get(kernel.getContext(),
                                           logicalSource.sourceId,
                                           logicalSource.sourceAxis,
                                           logicalSource.derived));
    chunk->setAttr(pointwiseChunkAttr, entry.getUnitAttr());
    if (FailureOr<uint64_t> dimension = rangeDimension(range);
        succeeded(dimension))
      chunk->setAttr(dimensionAttr,
                     entry.getI64IntegerAttr(*dimension));
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

  FailureOr<int64_t> traversalDimension = queryRangeDimension(range);
  if (failed(traversalDimension))
    return range.emitOpError(
        "reuse-sensitive pointwise traversal has no exact logical dimension");
  llvm::MapVector<Block *, SmallVector<StoreOp>> storesByBlock;
  for (StoreOp store : stores)
    storesByBlock[store->getBlock()].push_back(store);
  SmallVector<SmallVector<StoreOp>> storeGroups;
  PhysicalProgramAnalysis replayAnalysis(kernel);
  auto canReplayAt = [&](StoreOp store, Operation *anchor) {
    auto replayable = [&](Value value) {
      return !value ||
             replayAnalysis
                 .replayability(value, logicalSource,
                                PhysicalReplayScope::ValueGraph,
                                /*allowAccesses=*/true, anchor,
                                *traversalDimension)
                 .isReplayable();
    };
    if (!replayable(store.getValue()) || !replayable(store.getValid()))
      return false;
    return llvm::all_of(store.getCoordinates(), replayable);
  };
  for (auto &entry : storesByBlock) {
    SmallVector<StoreOp> &blockStores = entry.second;
    SmallVector<bool> assigned(blockStores.size(), false);
    for (unsigned first = 0; first < blockStores.size(); ++first) {
      if (assigned[first])
        continue;
      SmallVector<StoreOp> group{blockStores[first]};
      assigned[first] = true;
      Operation *anchor = blockStores[first].getOperation();
      for (unsigned next = first + 1; next < blockStores.size(); ++next) {
        if (assigned[next] || !canReplayAt(blockStores[next], anchor))
          continue;
        group.push_back(blockStores[next]);
        assigned[next] = true;
      }
      storeGroups.push_back(std::move(group));
    }
  }
  SmallVector<scf::ForOp> materializedLoops;
  for (SmallVector<StoreOp> &group : storeGroups) {
  OpBuilder builder(group.front());
  Operation *loopInsertionAnchor = group.front().getOperation();
  Value distance = builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), range.getExtent(), range.getStep(),
      BinaryOperator::Multiply);
  Value stop = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                        range.getStart(), distance,
                                        BinaryOperator::Add);
  Value loopStep = builder.create<BinaryOp>(
      range.getLoc(), builder.getIndexType(), chunk.getResult(), range.getStep(),
      BinaryOperator::Multiply);
  bool bodyFailed = false;
  std::string failureReason;
  auto loop = builder.create<scf::ForOp>(
      range.getLoc(), range.getStart(), stop, loopStep, ValueRange{},
      [&](OpBuilder &nested, Location location, Value tileStart, ValueRange) {
        Value blocked = nested.create<MakeRangeOp>(
            location, blockedType, tileStart, chunk.getResult(), range.getStep(),
            range.getLogicalStart(), range.getLogicalStop(), range.getSourceId(),
            range.getSourceAxis(), range.getDerived());
        if (Attribute value = range->getAttr(sourceSubregionAttr))
          blocked.getDefiningOp()->setAttr(sourceSubregionAttr, value);
        Value end = nested.create<BroadcastOp>(location, blockedType, stop);
        auto tailComparison = nested.create<CompareOp>(
            location, predicateType(blockedType), blocked, end,
            ComparePredicate::Lt);
        tailComparison->setAttr(physicalTailAttr, nested.getUnitAttr());
        Value tail = tailComparison.getResult();
        IRMapping mapping;
        mapping.map(range.getResult(), blocked);
        for (StoreOp store : group) {
          FailureOr<Value> payload = replayPointwiseValue(
              nested, store.getValue(), logicalSource, chunkExtent,
              blocked, tail, loopInsertionAnchor, mapping);
          if (failed(payload)) {
            bodyFailed = true;
            failureReason = "write payload cannot be replayed in the internal tile loop";
            return;
          }
          SmallVector<Value> coordinates;
          for (Value coordinate : store.getCoordinates()) {
            FailureOr<Value> replayed = replayPointwiseValue(
                nested, coordinate, logicalSource, chunkExtent, blocked,
                tail, loopInsertionAnchor, mapping);
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
            FailureOr<FragmentType> schema =
                coordinateValueSchema((*payload).getType(), coordinates);
            if (failed(schema)) {
              bodyFailed = true;
              failureReason =
                  "write coordinates have no exact payload fragment schema";
              return;
            }
            FailureOr<Value> projected = projectPhysicalValueToSchema(
                nested, location, *payload, *schema);
            if (failed(projected)) {
              bodyFailed = true;
              failureReason =
                  "write payload cannot adopt its coordinate fragment schema";
              return;
            }
            payload = *projected;
            payloadType = *schema;
          }
          FailureOr<Value> valid = materializeBroadcastToFragment(nested, location, tail,
                                               predicateType(payloadType));
          if (failed(valid)) {
            bodyFailed = true;
            failureReason = "tile tail cannot be projected to the write payload";
            return;
          }
          if (store.getValid()) {
            FailureOr<Value> existing = replayPointwiseValue(
                nested, store.getValid(), logicalSource, chunkExtent,
                blocked, tail, loopInsertionAnchor, mapping);
            if (failed(existing)) {
              bodyFailed = true;
              failureReason = "write validity cannot be replayed in the tile loop";
              return;
            }
            FailureOr<Value> projected = materializeBroadcastToFragment(
                nested, location, *existing, predicateType(payloadType));
            if (failed(projected)) {
              bodyFailed = true;
              failureReason = "write validity cannot be projected to the payload";
              return;
            }
            valid = Value(nested.create<BinaryOp>(
                location, predicateType(payloadType), *valid, *projected,
                BinaryOperator::LogicalAnd));
          }
          auto replacement = nested.create<StoreOp>(
              location, store.getResource(), coordinates, *payload, *valid,
              store.getSourceAxes());
          if (Attribute origin = store->getAttr(originAttr))
            replacement->setAttr(originAttr, origin);
        }
        if (!bodyFailed)
          nested.create<scf::YieldOp>(location);
      });
  if (bodyFailed) {
    loop.erase();
    for (scf::ForOp materialized : materializedLoops)
      materialized.erase();
    return range.emitOpError(
               "reuse-sensitive pointwise traversal could not be materialized: ")
           << failureReason;
  }
  materializedLoops.push_back(loop);
  }
  for (StoreOp store : stores)
    store.erase();
  return success();
}

void collectCoordinateRanges(Value coordinate,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
  auto kernel = coordinate.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!kernel)
    return;
  PhysicalRangeFact fact = PhysicalProgramAnalysis(kernel).sourceRanges(coordinate);
  if (fact.state == PhysicalFactState::Unknown)
    return;
  for (MakeRangeOp range : fact.roots)
    ranges.insert(range.getOperation());
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
        AtomicRMWKind::Add, AtomicOrdering::Relaxed,
        AtomicSharingDomain::KernelInvocation,
        store.getSourceAxes());
    if (Attribute origin = store->getAttr(originAttr))
      atomic->setAttr(originAttr, origin);
    store.erase();
  }
  return success();
}

LogicalResult alignContractAccumulatorTypes(func::FuncOp kernel) {
  auto align = [](Operation *owner, Value accumulator,
                  Value result) -> LogicalResult {
    if (accumulator.getType() == result.getType())
      return success();
    auto source = dyn_cast<FragmentType>(accumulator.getType());
    auto target = dyn_cast<FragmentType>(result.getType());
    if (!source || !target || source.getElementType() != target.getElementType() ||
        source.getOwner() != target.getOwner() ||
        source.getAxisMaps() != target.getAxisMaps())
      return owner->emitOpError(
          "pointwise ownership cannot preserve the contract accumulator relation");
    accumulator.setType(target);
    return success();
  };
  WalkResult result = kernel.walk([&](Operation *operation) {
    Value accumulator;
    Value output;
    if (auto contract = dyn_cast<ContractOp>(operation)) {
      accumulator = contract.getAccumulator();
      output = contract.getResult();
    } else if (auto contract = dyn_cast<ScaledContractOp>(operation)) {
      accumulator = contract.getAccumulator();
      output = contract.getResult();
    } else if (auto contract = dyn_cast<SparseContractOp>(operation)) {
      accumulator = contract.getAccumulator();
      output = contract.getResult();
    } else {
      return WalkResult::advance();
    }
    return failed(align(operation, accumulator, output))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool isCartesianPointwiseValueOp(Operation *operation) {
  return isa<SplatOp, BroadcastOp, UnaryOp, BinaryOp, CompareOp, SelectOp,
             CastOp, BitcastOp, LoadOp, GatherOp, RandomBitsOp>(operation);
}

bool supportsCartesianPointwiseValueGraph(
    ArrayRef<WorksetCoordinateOp> coordinates) {
  llvm::SmallDenseSet<Value> dependent;
  SmallVector<Value> worklist;
  for (WorksetCoordinateOp coordinate : coordinates) {
    dependent.insert(coordinate.getResult());
    worklist.push_back(coordinate.getResult());
  }
  llvm::SmallPtrSet<Operation *, 32> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (user->getNumResults() == 0) {
        // A terminal pointwise store can consume the promoted fragment.  A
        // control-flow terminator or effectful region cannot: promoting the
        // predicate/carry would change scalar program structure into a lane
        // program without a physical control-flow realization.
        if (!isa<StoreOp>(user) || user->getNumRegions() != 0)
          return false;
        continue;
      }
      if (!isCartesianPointwiseValueOp(user) || user->getNumRegions() != 0)
        return false;
      for (Value result : user->getResults()) {
        Type type = result.getType();
        if (!isa<IntegerType, FloatType, IndexType, FragmentType>(type))
          return false;
        if (dependent.insert(result).second)
          worklist.push_back(result);
      }
    }
  }
  return true;
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
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    };
    for (auto [extent, mapping] : liftedAxes) {
      bool present = llvm::any_of(original.getAxisMaps(), [&](Attribute attribute) {
        auto axis = cast<AxisMapAttr>(attribute);
        return sourceAxisIdentity(axis) == sourceAxisIdentity(mapping);
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

  auto scalarLiftedType = [&](Type element) {
    SmallVector<Attribute> shape;
    SmallVector<Attribute> mappings;
    for (auto [extent, mapping] : liftedAxes) {
      shape.push_back(extent);
      mappings.push_back(AxisMapAttr::get(
          kernel.getContext(), mapping.getSourceId(), mapping.getSourceAxis(),
          mapping.getDimensionId(), mappings.size(), mapping.getDerived()));
    }
    return FragmentType::get(
        kernel.getContext(), element, ArrayAttr::get(kernel.getContext(), shape),
        ArrayAttr::get(kernel.getContext(), mappings), /*validity=*/1,
        /*owner=*/1);
  };
  auto carriesLiftedAxis = [&](FragmentType fragment) {
    return llvm::any_of(fragment.getAxisMaps(), [&](Attribute attribute) {
      auto axis = cast<AxisMapAttr>(attribute);
      return llvm::any_of(liftedAxes, [&](const auto &lifted) {
        return sourceAxisIdentity(axis) == sourceAxisIdentity(lifted.second);
      });
    });
  };

  llvm::SmallDenseSet<Value> liftedValues;
  for (MakeRangeOp range : liftedRanges)
    liftedValues.insert(range.getResult());
  WalkResult result = kernel.walk([&](Operation *operation) {
    if (isa<MakeRangeOp>(operation))
      return WalkResult::advance();
    bool dependsOnLiftedRange =
        llvm::any_of(operation->getOperands(), [&](Value operand) {
          if (liftedValues.contains(operand))
            return true;
          auto fragment = dyn_cast<FragmentType>(operand.getType());
          return fragment && carriesLiftedAxis(fragment);
        });
    if (!dependsOnLiftedRange)
      return WalkResult::advance();
    if (operation->getNumResults() == 0)
      return WalkResult::advance();
    if (!isCartesianPointwiseValueOp(operation) ||
        operation->getNumRegions() != 0)
      return WalkResult::interrupt();
    for (Value value : operation->getResults()) {
      Type type = value.getType();
      if (auto fragment = dyn_cast<FragmentType>(type))
        value.setType(liftedType(fragment));
      else if (isa<IntegerType, FloatType, IndexType>(type))
        value.setType(scalarLiftedType(type));
      else
        return WalkResult::interrupt();
      liftedValues.insert(value);
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace

static LogicalResult realizePointwiseBlockingImpl(ModuleOp module,
                                                   bool ownershipOnly) {
  FailureOr<func::FuncOp> physicalKernel = getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  func::FuncOp kernel = *physicalKernel;
  auto finalizeValueRelations = [&]() -> LogicalResult {
    if (failed(alignStructuredCaptureRelations(kernel)) ||
        failed(alignReductionIdentityRelations(kernel)) ||
        failed(alignAggregateValueRelations(kernel)) ||
        failed(alignPointwiseValueRelations(kernel)) ||
        failed(alignReductionYieldRelations(kernel)) ||
        failed(alignAccessValueRelations(kernel)))
      return failure();
    eraseDeadPhysicalValues(kernel);
    return success();
  };

  if (ownershipOnly) {
    SmallVector<WorksetCoordinateOp> pointwiseCoordinates;
    kernel.walk([&](WorksetCoordinateOp coordinate) {
      pointwiseCoordinates.push_back(coordinate);
    });
    SmallVector<MakeRangeOp> existingRanges;
    kernel.walk([&](MakeRangeOp range) { existingRanges.push_back(range); });
    SmallVector<WorksetCoordinateOp> lifted;
    if (existingRanges.empty() &&
        supportsCartesianPointwiseValueGraph(pointwiseCoordinates))
      lifted.append(pointwiseCoordinates.begin(), pointwiseCoordinates.end());

    SmallVector<MakeRangeOp> liftedRanges;
    for (WorksetCoordinateOp coordinate : lifted) {
      OpBuilder builder(coordinate);
      builder.setInsertionPointAfter(coordinate);
      Value extent = builder.create<arith::ConstantIndexOp>(coordinate.getLoc(), 1);
      Value step = builder.create<arith::ConstantIndexOp>(coordinate.getLoc(), 1);
      PhysicalExprAttr unit = expression(module.getContext(),
                                         PhysicalExprKind::Constant, 1);
      int64_t dimension = coordinate.getDimensionId();
      if (dimension <= 0)
        return coordinate.emitOpError(
            "workset coordinate has no logical dimension identity");
      auto type = FragmentType::get(
          module.getContext(), builder.getIndexType(), builder.getArrayAttr({unit}),
          builder.getArrayAttr({AxisMapAttr::get(
              module.getContext(), coordinate.getSourceId(),
              coordinate.getSourceAxis(), dimension,
              /*fragmentAxis=*/0, /*derived=*/false)}),
          /*validity=*/1, /*owner=*/1);
      Value logicalStop = builder.create<BinaryOp>(
          coordinate.getLoc(), builder.getIndexType(), coordinate.getResult(),
          extent, BinaryOperator::Add);
      auto range = builder.create<MakeRangeOp>(
          coordinate.getLoc(), type, coordinate.getResult(), extent, step,
          coordinate.getResult(), logicalStop,
          coordinate.getSourceId(), coordinate.getSourceAxis(),
          /*derived=*/false);
      range->setAttr(worksetCoordinateRangeAttr, builder.getUnitAttr());
      Operation *logicalStopProducer = logicalStop.getDefiningOp();
      coordinate.getResult().replaceUsesWithIf(
          range.getResult(), [&](OpOperand &use) {
            return use.getOwner() != range.getOperation() &&
                   use.getOwner() != logicalStopProducer;
          });
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
    // Helper-local ranges already describe one physical fold/scan slice.  The
    // enclosing structured operation owns their segment relation; pointwise
    // blocking may only revisit them after the helper has been materialized
    // into the executable loop.
    if (range->getParentOfType<RegionFoldOp>() ||
        range->getParentOfType<RegionScanOp>())
      return;
    auto parameter = range.getExtent().getDefiningOp<ParameterOp>();
    if (!hasCompileTimeExtent(range.getExtent()) ||
        (parameter && parameter->hasAttr(coverageDimensionAttr))) {
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
  llvm::SmallDenseSet<uint64_t> scanSegmentDimensions;
  llvm::SmallDenseSet<PhysicalSourceAxis> scanSegmentSources;
  auto collectAxisInto = [&](Value source, uint64_t axis,
                             llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(source.getType());
    if (!fragment || axis >= fragment.getAxisMaps().size())
      return;
    auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
    collectProducerRanges(
        source,
        PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
        ranges);
  };
  std::function<void(Value, llvm::SmallPtrSetImpl<Operation *> &)>
      collectAllAxesInto = [&](Value value,
                               llvm::SmallPtrSetImpl<Operation *> &ranges) {
    auto fragment = dyn_cast<FragmentType>(value.getType());
    if (!fragment) {
      if (auto record = value.getDefiningOp<MakeRecordOp>())
        for (Value field : record.getFields())
          collectAllAxesInto(field, ranges);
      return;
    }
    PhysicalRangeFact all = PhysicalProgramAnalysis(kernel).sourceRanges(value);
    for (MakeRangeOp range : all.roots)
      ranges.insert(range.getOperation());
    for (Attribute attribute : fragment.getAxisMaps()) {
      auto mapping = cast<AxisMapAttr>(attribute);
      collectProducerRanges(
          value,
          PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
          ranges);
    }
  };
  auto collectStructuredRegionRanges = [&](Operation *structured,
                                           ValueRange sources, uint64_t axis) {
    SmallVector<PhysicalSourceAxis> segmentSources;
    for (Value source : sources) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment || axis >= fragment.getAxisMaps().size())
        continue;
      auto mapping = cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
      PhysicalSourceAxis physical{mapping.getSourceId(),
                                  mapping.getSourceAxis(),
                                  mapping.getDerived()};
      if (!llvm::is_contained(segmentSources, physical))
        segmentSources.push_back(physical);
    }
    for (Region &region : structured->getRegions())
      region.walk([&](MakeRangeOp range) {
        if (llvm::is_contained(
                segmentSources,
                PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()}))
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
    for (Value source : sources) {
      collectAxisInto(source, scan.getAxis(), structuredTraversalRanges);
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (fragment && scan.getAxis() < fragment.getAxisMaps().size()) {
        auto mapping =
            cast<AxisMapAttr>(fragment.getAxisMaps()[scan.getAxis()]);
        int64_t dimension = mapping.getDimensionId();
        if (dimension > 0)
          scanSegmentDimensions.insert(static_cast<uint64_t>(dimension));
        scanSegmentSources.insert(
            {mapping.getSourceId(), mapping.getSourceAxis(),
             mapping.getDerived()});
      }
    }
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
          failed(requireScanFullCoverage(kernel, source, scan.getAxis()));
    for (Value result : scan.getResults()) {
      llvm::SmallPtrSet<Operation *, 16> visited;
      collectStoreRanges(result, internalTraversalRanges, visited);
    }
  });
  if (scanCoverageFailed)
    return kernel.emitError(
        "dynamic scan axis has no launch-visible full-coverage realization");
  SmallVector<std::pair<Value, uint64_t>> structuredReductionSources;
  kernel.walk([&](ReduceOp reduce) {
    for (Value source :
         reduce.getInputs().take_front(reduce.getSourceCount())) {
      auto fragment = dyn_cast<FragmentType>(source.getType());
      if (!fragment)
        continue;
      for (int64_t axis : reduce.getAxes()) {
        if (axis < 0 || axis >= static_cast<int64_t>(fragment.getShape().size()))
          continue;
        auto mapping =
            cast<AxisMapAttr>(fragment.getAxisMaps()[axis]);
        PhysicalReplayFact replay = PhysicalProgramAnalysis(kernel).replayability(
            source,
            PhysicalSourceAxis{mapping.getSourceId(), mapping.getSourceAxis(),
                           mapping.getDerived()},
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true);
        bool structuredBlocker = llvm::any_of(
            replay.blockers, [](Operation *blocker) {
              return isa<ScanOp, scf::ForOp>(blocker);
            });
        if (structuredBlocker)
          structuredReductionSources.emplace_back(source,
                                                   static_cast<uint64_t>(axis));
      }
    }
  });
  for (auto [source, axis] : structuredReductionSources)
    if (failed(requireStructuredReductionFullCoverage(kernel, source, axis)))
      return kernel.emitError(
          "structured reduction source has no exact full-coverage realization");
  kernel.walk([&](HistogramOp histogram) {
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(histogram.getResult(), internalTraversalRanges, visited);
  });
  kernel.walk([&](ContractOp contract) {
    collectAllAxesInto(contract.getLhs(), structuredTraversalRanges);
    collectAllAxesInto(contract.getRhs(), structuredTraversalRanges);
    collectAllAxesInto(contract.getResult(), structuredTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  kernel.walk([&](ScaledContractOp contract) {
    collectAllAxesInto(contract.getLhs(), structuredTraversalRanges);
    collectAllAxesInto(contract.getLhsScale(), structuredTraversalRanges);
    collectAllAxesInto(contract.getRhs(), structuredTraversalRanges);
    collectAllAxesInto(contract.getRhsScale(), structuredTraversalRanges);
    collectAllAxesInto(contract.getResult(), structuredTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  kernel.walk([&](SparseContractOp contract) {
    collectAllAxesInto(contract.getCompressed(), structuredTraversalRanges);
    collectAllAxesInto(contract.getMetadata(), structuredTraversalRanges);
    collectAllAxesInto(contract.getRhs(), structuredTraversalRanges);
    collectAllAxesInto(contract.getResult(), structuredTraversalRanges);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectStoreRanges(contract.getResult(), internalTraversalRanges, visited);
  });
  llvm::SmallDenseSet<PhysicalSourceAxis> structuredSources;
  for (Operation *operation : structuredTraversalRanges)
    if (auto range = dyn_cast<MakeRangeOp>(operation))
      structuredSources.insert({range.getSourceId(), range.getSourceAxis()});
  llvm::SmallPtrSet<Operation *, 16> reuseTraversalRanges;
  SmallVector<StoreOp> candidateStores;
  kernel.walk([&](StoreOp store) { candidateStores.push_back(store); });
  for (StoreOp store : candidateStores) {
    SmallVector<std::pair<MakeRangeOp, int64_t>> candidates;
    int64_t innermostSourceAxis = -1;
    for (MakeRangeOp range : allRanges) {
      PhysicalSourceAxis logicalSource{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
      if (structuredSources.contains(logicalSource))
        continue;
      std::optional<int64_t> sourceAxis = storeAxisForRange(store, range);
      if (!sourceAxis)
        continue;
      std::optional<int64_t> sourceDimension;
      if (FailureOr<uint64_t> dimension = rangeDimension(range);
          succeeded(dimension))
        sourceDimension = *dimension;
      PhysicalReductionDependencyFact dependency =
          PhysicalProgramAnalysis(kernel).reductionDependency(
              store.getValue(), logicalSource, sourceDimension);
      if (!dependency.isExact() || !dependency.depends)
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
      FailureOr<ParameterOp> parameter = queryBlockingParameter(kernel, range);
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
      auto blocked = builder.create<MakeRangeOp>(
          range.getLoc(), fragment, range.getStart(), extent, range.getStep(),
          range.getLogicalStart(), range.getLogicalStop(), range.getSourceId(),
          range.getSourceAxis(), range.getDerived());
      if (Attribute value = range->getAttr(sourceSubregionAttr))
        blocked->setAttr(sourceSubregionAttr, value);
      Value endFragment =
          builder.create<BroadcastOp>(range.getLoc(), fragment,
                                      range.getLogicalStop());
      auto validComparison = builder.create<CompareOp>(
          range.getLoc(), predicateType(fragment), blocked.getResult(),
          endFragment, ComparePredicate::Lt);
      validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
      Value valid = validComparison.getResult();
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
      SmallVector<StoreOp> currentStores;
      kernel.walk([&](StoreOp store) {
        if (storeUsesRange(store, range))
          currentStores.push_back(store);
      });
      if (currentStores.empty())
        continue;
      FailureOr<uint64_t> traversalDimension = rangeDimension(range);
      for (StoreOp store : currentStores) {
        PhysicalProgramAnalysis analysis(kernel);
        PhysicalReplayFact replay = analysis.replayability(
            store.getValue(),
            PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()},
            PhysicalReplayScope::ValueGraph, /*allowAccesses=*/true,
            store.getOperation(),
            succeeded(traversalDimension)
                ? std::optional<int64_t>(*traversalDimension)
                : std::nullopt);
        replayable &= replay.isReplayable();
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
    return finalizeValueRelations();
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
  llvm::SmallDenseSet<PhysicalSourceAxis> ownershipSources;
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
        auto mapping = cast<AxisMapAttr>(attribute);
        ownershipSources.insert(
            {mapping.getSourceId(), mapping.getSourceAxis(),
             mapping.getDerived()});
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
    for (Attribute attribute : fragment.getAxisMaps()) {
      auto mapping = cast<AxisMapAttr>(attribute);
      ownershipSources.insert(
          {mapping.getSourceId(), mapping.getSourceAxis(),
           mapping.getDerived()});
    }
  });
  // Structured traversal ownership is attached to the exact range operations
  // above, not erased source-wide here.  One immutable source axis may have a
  // query ownership projection and an independent fold/scan segment
  // projection; conflating them would discard a real physical decision.

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

  llvm::MapVector<Attribute, SmallVector<MakeRangeOp>> axes;
  llvm::DenseMap<Attribute, ParameterOp> parameters;
  llvm::SmallDenseSet<Attribute> ownershipAxes;
  llvm::SmallDenseSet<Attribute> internalAxes;
  auto hasPointwiseOwnership = [&](MakeRangeOp range) {
    FailureOr<uint64_t> dimension = rangeDimension(range);
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    return ownershipSources.contains(source) &&
           !scanSegmentSources.contains(source) &&
           (failed(dimension) ||
            !scanSegmentDimensions.contains(*dimension));
  };
  auto dependsOnSource = [&](ValueRange values, PhysicalSourceAxis source) {
    for (Value value : values) {
      if (!queryFragmentAxis(value.getType(), source).isExact())
        continue;
      llvm::SmallPtrSet<Operation *, 8> ranges;
      collectProducerRanges(value, source, ranges);
      if (!ranges.empty())
        return true;
    }
    return false;
  };
  auto dependsOnHistogramSource = [&](ValueRange values,
                                      PhysicalSourceAxis source) {
    return llvm::any_of(values, [&](Value value) {
      HistogramOp histogram = histogramSource(value);
      return histogram && containsSource(histogram.getValues(), source);
    });
  };
  llvm::DenseMap<PhysicalSourceAxis, uint64_t> sourceDimensions;
  llvm::MapVector<uint64_t, SmallVector<PhysicalSourceAxis>> dimensionSources;
  for (MakeRangeOp range : dynamicRanges) {
    FailureOr<uint64_t> dimension = rangeDimension(range);
    if (failed(dimension) || !hasPointwiseOwnership(range))
      continue;
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    uint64_t dimensionId = *dimension;
    sourceDimensions[source] = dimensionId;
    if (!llvm::is_contained(dimensionSources[dimensionId], source))
      dimensionSources[dimensionId].push_back(source);
  }
  auto effectDependsOn = [&](const WriteEffectFacts &effect,
                             PhysicalSourceAxis source) {
    return dependsOnSource(effect.coordinates, source) ||
           dependsOnSource(effect.payloads, source) ||
           dependsOnHistogramSource(effect.payloads, source);
  };
  llvm::SmallDenseSet<uint64_t> jointOwnershipDimensions;
  for (auto [dimensionId, sources] : dimensionSources) {
    if (sources.size() < 2)
      continue;
    bool onePreservedSourcePerEffect =
        llvm::all_of(writeEffects, [&](const WriteEffectFacts &effect) {
          return llvm::count_if(sources, [&](PhysicalSourceAxis source) {
                   return effectDependsOn(effect, source);
                 }) == 1;
        });
    if (onePreservedSourcePerEffect)
      jointOwnershipDimensions.insert(dimensionId);
  }
  for (MakeRangeOp range : dynamicRanges) {
    PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
    if (!hasPointwiseOwnership(range)) {
      internalTraversalRanges.insert(range.getOperation());
      continue;
    }
    bool requiredByEveryEffect =
        llvm::all_of(writeEffects, [&](const WriteEffectFacts &effect) {
          return effectDependsOn(effect, source);
        });
    auto dimension = sourceDimensions.find(source);
    bool jointlyOwned =
        dimension != sourceDimensions.end() &&
        jointOwnershipDimensions.contains(dimension->second);
    if (!requiredByEveryEffect && !jointlyOwned)
      internalTraversalRanges.insert(range.getOperation());
  }
  if (!ownershipOnly)
    for (MakeRangeOp range : dynamicRanges) {
      if (!internalTraversalRanges.contains(range.getOperation()) ||
          structuredTraversalRanges.contains(range.getOperation()) ||
          reuseTraversalRanges.contains(range.getOperation()))
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
      FailureOr<ParameterOp> internalParameter = queryBlockingParameter(kernel, range);
      FailureOr<Attribute> internalAxis =
          succeeded(internalParameter)
              ? parameterAxis(*internalParameter)
              : FailureOr<Attribute>(failure());
      if (succeeded(internalAxis)) {
        internalAxes.insert(*internalAxis);
      } else if (FailureOr<uint64_t> dimension = rangeDimension(range);
                 succeeded(dimension))
        internalAxes.insert(dimensionAxisKey(module.getContext(), *dimension));
      continue;
    }
    if (!ownershipOnly &&
        internalTraversalRanges.contains(range.getOperation()) &&
        !reuseTraversalRanges.contains(range.getOperation()))
      continue;
    if (ownershipOnly && !hasPointwiseOwnership(range))
      continue;
    if (!ownershipOnly &&
        !hasPointwiseOwnership(range) &&
        !internalTraversalRanges.contains(range.getOperation()) &&
        !reuseTraversalRanges.contains(range.getOperation()))
      continue;
    FailureOr<ParameterOp> parameter = queryBlockingParameter(kernel, range);
    bool requiresBlockingParameter =
        (ownershipOnly && hasPointwiseOwnership(range) &&
         !internalTraversalRanges.contains(range.getOperation())) ||
        (!ownershipOnly &&
         ((hasPointwiseOwnership(range) &&
           !internalTraversalRanges.contains(range.getOperation())) ||
          reuseTraversalRanges.contains(range.getOperation())));
    if (failed(parameter) && requiresBlockingParameter) {
      auto logicalExtent = range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
      auto fragment = cast<FragmentType>(range.getResult().getType());
      auto physicalExtent = cast<PhysicalExprAttr>(fragment.getShape()[0]);
      const bool dynamicSubregion = range->hasAttr(sourceSubregionAttr);
      FailureOr<uint64_t> sourceDimension = rangeDimension(range);
      const bool launchVisibleDimension =
          succeeded(sourceDimension) &&
          succeeded(dimensionArgument(kernel, *sourceDimension));
      if (dynamicSubregion ||
          launchVisibleDimension ||
          (logicalExtent && physicalExtent.getKind() ==
                                static_cast<uint32_t>(PhysicalExprKind::Constant))) {
        SmallVector<int64_t> candidates;
        if (dynamicSubregion) {
          candidates.assign({1, 2, 4, 8, 16, 32, 64, 128, 256});
        } else if (launchVisibleDimension) {
          candidates.assign({8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                             4096});
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
        PhysicalSourceAxis source{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
        auto schema = ParameterAttr::get(
            module.getContext(),
            builder.getStringAttr(launchVisibleDimension
                                      ? ("FRAGMENT_D" +
                                         Twine(*sourceDimension))
                                            .str()
                                      : ("FRAGMENT_S" + Twine(source.sourceId) +
                                         "_A" + Twine(source.sourceAxis))
                                            .str()),
            static_cast<uint32_t>(ParameterRole::OwnershipN),
            DenseI64ArrayAttr::get(module.getContext(), candidates));
        parameter = builder.create<ParameterOp>(
            range.getLoc(), builder.getIndexType(), schema);
        if (succeeded(sourceDimension))
          (*parameter)->setAttr(dimensionAttr,
                                builder.getI64IntegerAttr(*sourceDimension));
        if (!launchVisibleDimension)
          (*parameter)->setAttr(
              parameterSourceAttr,
              PhysicalSourceAttr::get(module.getContext(), source.sourceId,
                                      source.sourceAxis, source.derived));
      }
    }
    FailureOr<Attribute> axis =
        failed(parameter) ? FailureOr<Attribute>(failure())
                          : parameterAxis(*parameter);
    if (failed(parameter) || failed(axis)) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no canonical blocking dimension");
      diagnostic << "; source_id=" << range.getSourceId() << ", fragment="
                 << range.getResult().getType();
      if (succeeded(parameter))
        diagnostic << ", parameter="
                   << parameter->getParameter().getName().getValue();
      return failure();
    }
    if (FailureOr<PhysicalSourceAxis> source = axisSource(*axis);
        succeeded(source))
      retargetSourceExtent(range.getResult(), *source,
                           fragmentExtent(*parameter));
    else if (FailureOr<uint64_t> dimension = axisDimension(*axis);
             succeeded(dimension))
      retargetDimensionExtent(range.getResult(), *dimension,
                              fragmentExtent(*parameter));
    else
      return range.emitOpError("blocking parameter has no typed axis binding");
    auto found = parameters.find(*axis);
    if (found != parameters.end() && found->second != *parameter)
      return range.emitOpError("one physical axis has multiple blocking parameters")
             << "; axis=" << *axis << ", previous="
             << found->second.getParameter().getName().getValue()
             << ", current=" << parameter->getParameter().getName().getValue()
             << ", source_id=" << range.getSourceId();
    parameters[*axis] = *parameter;
    axes[*axis].push_back(range);
    if (internalTraversalRanges.contains(range.getOperation()))
      internalAxes.insert(*axis);
    if (hasPointwiseOwnership(range) &&
        !internalTraversalRanges.contains(range.getOperation()))
      ownershipAxes.insert(*axis);
  }
  if (!ownershipOnly)
    ownershipAxes.clear();

  if (ownershipOnly) {
    SmallVector<Attribute> nonLiftableAxes;
    for (Attribute axis : ownershipAxes) {
      bool losesPromotedAxis = false;
      for (MakeRangeOp range : axes.lookup(axis)) {
        PhysicalSourceAxis sourceAxis{range.getSourceId(), range.getSourceAxis(),
                              range.getDerived()};
        kernel.walk([&](BroadcastOp broadcast) {
          auto source = dyn_cast<FragmentType>(broadcast.getValue().getType());
          auto target = dyn_cast<FragmentType>(broadcast.getResult().getType());
          if (!source || !target ||
              queryFragmentAxes(source, sourceAxis).empty() ||
              !queryFragmentAxes(target, sourceAxis).empty())
            return;
          losesPromotedAxis = true;
        });
        if (losesPromotedAxis)
          break;
      }
      if (losesPromotedAxis)
        nonLiftableAxes.push_back(axis);
    }
    // A BroadcastOp whose result does not carry a promoted source axis can
    // only expand a singleton value along its existing target axes.  Keeping
    // a non-unit candidate for that dimension would create an invalid
    // equal-rank broadcast and leave provider autotune to discover the error.
    // Until the physical value graph contains an explicit rank-lifted result,
    // keep that workset dimension as a scalar program coordinate.
    for (Attribute axis : nonLiftableAxes)
      ownershipAxes.erase(axis);
  }

  if (ownershipOnly) {
    llvm::SmallDenseSet<Attribute> internalOwnershipAxes;
    for (Attribute axis : ownershipAxes) {
      if (isSourceAxisKey(axis))
        continue;
      if (llvm::any_of(axes.lookup(axis), [](MakeRangeOp range) {
            return !range->hasAttr(worksetCoordinateRangeAttr);
          }))
        internalOwnershipAxes.insert(axis);
    }
    if (!internalOwnershipAxes.empty()) {
      SmallVector<Attribute> scalarWorksetAxes;
      for (Attribute axis : ownershipAxes) {
        if (isSourceAxisKey(axis) || internalOwnershipAxes.contains(axis))
          continue;
        if (llvm::all_of(axes.lookup(axis), [](MakeRangeOp range) {
              return range->hasAttr(worksetCoordinateRangeAttr);
            }))
          scalarWorksetAxes.push_back(axis);
      }
      for (Attribute axis : scalarWorksetAxes)
        ownershipAxes.erase(axis);
    }
  }

  if (ownershipOnly) {
    auto unitExtent = expression(module.getContext(), PhysicalExprKind::Constant, 1);
    for (auto [axis, ranges] : axes) {
      if (ownershipAxes.contains(axis))
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
        retargetSourceExtent(
            range.getResult(),
            PhysicalSourceAxis{range.getSourceId(), range.getSourceAxis(),
                           range.getDerived()},
            unitExtent);
      }
    }
  }

  for (Attribute axis : ownershipAxes) {
    ParameterOp parameter = parameters.lookup(axis);
    if (!parameter || isSourceAxisKey(axis) ||
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
    llvm::MapVector<Attribute, SmallVector<MakeRangeOp>> selectedAxes;
    SmallVector<MakeRangeOp> selectedRanges;
    for (auto [axis, ranges] : axes) {
      if (!ownershipAxes.contains(axis))
        continue;
      for (MakeRangeOp range : ranges) {
        if (internalTraversalRanges.contains(range.getOperation()))
          continue;
        selectedAxes[axis].push_back(range);
        selectedRanges.push_back(range);
      }
    }
    axes = std::move(selectedAxes);
    dynamicRanges = std::move(selectedRanges);
    if (dynamicRanges.empty())
      return finalizeValueRelations();
  }

  OpBuilder mappingBuilder(mapping);
  SmallVector<Value> runtimeExtents(mapping.getExtents());
  SmallVector<Attribute> launchExtents(mapping.getLaunchExtents().begin(),
                                       mapping.getLaunchExtents().end());
  SmallVector<Type> coordinateTypes(mapping.getResultTypes());
  llvm::DenseMap<Attribute, Value> tileCoordinates;
  llvm::DenseMap<Attribute, Value> logicalDimensions;
  llvm::DenseMap<Attribute, unsigned> mappedAxes;
  auto mappingAxis = [&](Attribute attribute) -> FailureOr<Attribute> {
    FailureOr<uint64_t> blocked = blockedDimension(attribute);
    if (succeeded(blocked))
      return dimensionAxisKey(module.getContext(), *blocked);
    auto physical = dyn_cast<PhysicalExprAttr>(attribute);
    if (!physical ||
        physical.getKind() !=
            static_cast<uint32_t>(PhysicalExprKind::Dimension))
      return failure();
    return physical.getValue() > 0
               ? FailureOr<Attribute>(dimensionAxisKey(module.getContext(),
                                                       physical.getValue()))
               : FailureOr<Attribute>(failure());
  };
  std::optional<unsigned> reusableUnitAxis;
  for (auto [axis, extent] : llvm::enumerate(mapping.getLaunchExtents())) {
    FailureOr<Attribute> mapped = mappingAxis(extent);
    if (succeeded(mapped) && axis < mapping.getCoordinates().size()) {
      tileCoordinates[*mapped] = mapping.getCoordinates()[axis];
      mappedAxes[*mapped] = axis;
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
  SmallVector<std::pair<Attribute, unsigned>> reusedCoordinates;
  SmallVector<Attribute> appendedCoordinates;
  for (auto [axisKey, ranges] : axes) {
    MakeRangeOp range = ranges.front();
    ParameterOp parameter = parameters.lookup(axisKey);
    Value dimension;
    arith::ConstantIndexOp staticExtent;
    FailureOr<uint64_t> sourceDimension = rangeDimension(range);
    if (isSourceAxisKey(axisKey)) {
      staticExtent = range.getExtent().getDefiningOp<arith::ConstantIndexOp>();
      if (staticExtent)
        dimension = mappingBuilder.create<arith::ConstantIndexOp>(
            mapping.getLoc(), staticExtent.value());
      else if (succeeded(sourceDimension)) {
        FailureOr<Value> argument =
            dimensionArgument(kernel, *sourceDimension);
        if (succeeded(argument))
          dimension = *argument;
      }
    } else {
      FailureOr<uint64_t> dimensionId = axisDimension(axisKey);
      if (succeeded(dimensionId)) {
        FailureOr<Value> argument = dimensionArgument(kernel, *dimensionId);
        if (succeeded(argument))
          dimension = *argument;
      }
    }
    if (!parameter) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic pointwise range has no launch-visible dimension or blocking parameter");
      diagnostic << "; axis=" << axisKey << ", fragment="
                 << range.getResult().getType();
      return failure();
    }
    if (!ownershipAxes.contains(axisKey))
      continue;
    if (!dimension) {
      InFlightDiagnostic diagnostic = range.emitOpError(
          "dynamic ownership range has no launch-visible logical dimension");
      diagnostic << "; axis=" << axisKey << ", fragment="
                 << range.getResult().getType() << ", parameter="
                 << parameter.getParameter().getName().getValue();
      return failure();
    }
    logicalDimensions[axisKey] = dimension;
    Value one = mappingBuilder.create<arith::ConstantIndexOp>(mapping.getLoc(), 1);
    Value adjusted = mappingBuilder.create<BinaryOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), dimension,
        mappingBuilder.create<BinaryOp>(mapping.getLoc(),
                                        mappingBuilder.getIndexType(),
                                        parameter.getResult(), one,
                                        BinaryOperator::Subtract),
        BinaryOperator::Add);
    Value tiles = mappingBuilder.create<BinaryOp>(
        mapping.getLoc(), mappingBuilder.getIndexType(), adjusted,
        parameter.getResult(), BinaryOperator::FloorDivide);
    PhysicalExprAttr logical;
    if (isSourceAxisKey(axisKey) && staticExtent) {
      logical = expression(module.getContext(), PhysicalExprKind::Constant,
                           staticExtent.value());
    } else {
      FailureOr<uint64_t> dimensionId = axisDimension(axisKey);
      uint64_t logicalDimension =
          isSourceAxisKey(axisKey) && succeeded(sourceDimension)
              ? *sourceDimension
              : succeeded(dimensionId) ? *dimensionId : 0;
      if (logicalDimension == 0)
        return range.emitOpError(
            "range-local ownership axis lost its logical dimension extent");
      logical = expression(module.getContext(), PhysicalExprKind::Dimension,
                           logicalDimension,
                           ("D" + Twine(logicalDimension)).str());
    }
    PhysicalExprAttr tile = expression(
        module.getContext(), PhysicalExprKind::Parameter, 0,
        parameter.getParameter().getName().getValue());
    PhysicalExprAttr launch = binaryExpression(
        module.getContext(), PhysicalExprKind::CeilDiv, logical, tile);
    auto mapped = mappedAxes.find(axisKey);
    if (mapped != mappedAxes.end() || reusableUnitAxis) {
      unsigned axis = mapped != mappedAxes.end() ? mapped->second
                                                 : *reusableUnitAxis;
      runtimeExtents[axis] = tiles;
      launchExtents[axis] = launch;
      reusedCoordinates.emplace_back(axisKey, axis);
      if (mapped == mappedAxes.end())
        reusableUnitAxis.reset();
    } else {
      runtimeExtents.push_back(tiles);
      coordinateTypes.push_back(mappingBuilder.getIndexType());
      launchExtents.push_back(launch);
      appendedCoordinates.push_back(axisKey);
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
    auto ownershipRole = [&](Attribute axis) {
      return llvm::any_of(axes.lookup(axis), [](MakeRangeOp range) {
               return range->hasAttr(worksetCoordinateRangeAttr);
             })
                 ? CoordinateRole::TiledWorkset
                 : CoordinateRole::PointwiseOwnership;
    };
    for (auto [axisKey, axis] : reusedCoordinates) {
      if (coordinateRoles[axis] !=
          static_cast<int64_t>(CoordinateRole::IndirectTraversal))
        coordinateRoles[axis] =
            static_cast<int64_t>(ownershipRole(axisKey));
    }
    unsigned appendedAxis = mapping.getNumResults();
    for (Attribute axisKey : appendedCoordinates) {
      coordinateRoles[appendedAxis++] =
          static_cast<int64_t>(ownershipRole(axisKey));
    }
    replacement->setAttr(
        coordinateRolesAttr,
        DenseI64ArrayAttr::get(module.getContext(), coordinateRoles));
    for (StringRef attribute : {executionGroupAttr, segmentOffsetAttr})
      if (Attribute value = mapping->getAttr(attribute))
        replacement->setAttr(attribute, value);
    for (auto [axis, pair] : llvm::enumerate(llvm::zip(
             mapping.getCoordinates(),
             replacement.getCoordinates().take_front(
                 mapping.getCoordinates().size())))) {
      Value oldCoordinate = std::get<0>(pair);
      Value newCoordinate = std::get<1>(pair);
      FailureOr<Attribute> axisKey =
          axis < mapping.getLaunchExtents().size()
              ? mappingAxis(mapping.getLaunchExtents()[axis])
              : FailureOr<Attribute>(failure());
      if (succeeded(axisKey) && ownershipAxes.contains(*axisKey)) {
        ParameterOp parameter = parameters.lookup(*axisKey);
        if (!parameter)
          return kernel.emitError(
              "pointwise ownership mapping lost its blocking parameter");
        newCoordinate = mappingBuilder.create<BinaryOp>(
            mapping.getLoc(), mappingBuilder.getIndexType(), newCoordinate,
            parameter.getResult(), BinaryOperator::Multiply);
        oldCoordinate.replaceAllUsesWith(newCoordinate);
        continue;
      }
      oldCoordinate.replaceAllUsesWith(newCoordinate);
    }
    for (auto [axisKey, axis] : reusedCoordinates)
      tileCoordinates[axisKey] = replacement.getCoordinates()[axis];
    unsigned extra = mapping.getCoordinates().size();
    for (Attribute axisKey : appendedCoordinates)
      tileCoordinates[axisKey] = replacement.getCoordinates()[extra++];
    PhysicalExprAttr total = cast<PhysicalExprAttr>(launchExtents.front());
    for (Attribute extent : llvm::drop_begin(launchExtents))
      total = binaryExpression(module.getContext(), PhysicalExprKind::Multiply,
                               total, cast<PhysicalExprAttr>(extent));
    replacement->setAttr(segmentLengthAttr, total);
    mapping.erase();
    mapping = replacement;
    kernel->setAttr(programSpaceAttr,
                    ArrayAttr::get(module.getContext(), {total}));
  }

  llvm::DenseMap<Value, Value> rangePredicates =
      std::move(fixedRangePredicates);
  for (MakeRangeOp range : dynamicRanges) {
    OpBuilder builder(range);
    FailureOr<ParameterOp> parameter = queryBlockingParameter(kernel, range);
    FailureOr<Attribute> axisKey =
        failed(parameter) ? FailureOr<Attribute>(failure())
                          : parameterAxis(*parameter);
    if (failed(parameter) || failed(axisKey)) {
      if (!range->hasAttr(sourceSubregionAttr) &&
          succeeded(requireFullDimensionCoverage(kernel, range.getResult(), 0)))
        continue;
      return range.emitOpError(
                 "dynamic pointwise range lost its canonical blocking dimension")
             << "; source_id=" << range.getSourceId()
             << ", fragment=" << range.getResult().getType();
    }
    Value tileCoordinate = tileCoordinates.lookup(*axisKey);
    if (!tileCoordinate && internalAxes.contains(*axisKey))
      tileCoordinate = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
    if (!tileCoordinate && !range->hasAttr(sourceSubregionAttr)) {
      if (failed(requireFullDimensionCoverage(kernel, range.getResult(), 0)))
        return range.emitOpError(
            "program-local range has no exact full-coverage realization");
      tileCoordinate = builder.create<arith::ConstantIndexOp>(range.getLoc(), 0);
    }
    if (!tileCoordinate)
      return range.emitOpError(
                 "dynamic pointwise range has no physical tile coordinate")
             << "; axis=" << *axisKey
             << ", source_id=" << range.getSourceId()
             << ", ownership=" << ownershipAxes.contains(*axisKey)
             << ", internal=" << internalAxes.contains(*axisKey)
             << ", parameter_name="
             << (*parameter).getParameter().getName().getValue()
             << ", parameter_role=" << (*parameter).getParameter().getRole()
             << ", has_coverage="
             << (*parameter)->hasAttr(coverageDimensionAttr)
             << ", launch_extents=" << mapping->getAttr("launch_extents");
    Value tileOffset = builder.create<BinaryOp>(
        range.getLoc(), builder.getIndexType(), tileCoordinate,
        parameter->getResult(), BinaryOperator::Multiply);
    Value start;
    Value end;
    if (range->hasAttr(worksetCoordinateRangeAttr)) {
      start = range.getStart();
      Value dimension = logicalDimensions.lookup(*axisKey);
      if (!dimension) {
        FailureOr<uint64_t> logicalDimension = axisDimension(*axisKey);
        if (failed(logicalDimension))
          logicalDimension = rangeDimension(range);
        if (succeeded(logicalDimension)) {
          FailureOr<Value> runtimeDimension =
              dimensionArgument(kernel, *logicalDimension);
          if (succeeded(runtimeDimension))
            dimension = *runtimeDimension;
        }
      }
      if (!dimension)
        return range.emitOpError(
                   "workset coordinate range lost its logical dimension extent")
               << "; axis=" << *axisKey << ", source_id="
               << range.getSourceId() << ", parameter="
               << parameter->getParameter().getName().getValue();
      Value remaining = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), dimension, tileOffset,
          BinaryOperator::Subtract);
      Value remainingDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), remaining, range.getStep(),
          BinaryOperator::Multiply);
      end = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                     start, remainingDistance,
                                     BinaryOperator::Add);
    } else {
      Value scaledOffset = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), tileOffset, range.getStep(),
          BinaryOperator::Multiply);
      start = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getStart(), scaledOffset,
          BinaryOperator::Add);
      Value logicalDistance = builder.create<BinaryOp>(
          range.getLoc(), builder.getIndexType(), range.getExtent(),
          range.getStep(), BinaryOperator::Multiply);
      end = builder.create<BinaryOp>(range.getLoc(), builder.getIndexType(),
                                     range.getStart(), logicalDistance,
                                     BinaryOperator::Add);
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
        range.getStep(), range.getLogicalStart(), range.getLogicalStop(),
        range.getSourceId(), range.getSourceAxis(), range.getDerived());
    if (Attribute value = range->getAttr(sourceSubregionAttr))
      blocked.getDefiningOp()->setAttr(sourceSubregionAttr, value);
    Value endFragment = builder.create<BroadcastOp>(range.getLoc(), blockedType, end);
    auto validComparison = builder.create<CompareOp>(
        range.getLoc(), predicateType(blockedType), blocked, endFragment,
        ComparePredicate::Lt);
    validComparison->setAttr(physicalTailAttr, builder.getUnitAttr());
    Value valid = validComparison.getResult();
    range.getResult().replaceAllUsesWith(blocked);
    rangePredicates[blocked] = valid;
    range.erase();
  }

  if (failed(addTailValidity(kernel, rangePredicates,
                             /*includeStores=*/true)))
    return failure();
  if (failed(realizeDistributedHistograms(kernel)))
    return failure();
  if (failed(alignContractAccumulatorTypes(kernel)))
    return failure();
  if (failed(bindStructurallyRequiredStaticFragments(kernel)))
    return failure();
  return finalizeValueRelations();
}

LogicalResult realizePointwiseOwnership(ModuleOp module) {
  return realizePointwiseBlockingImpl(module, /*ownershipOnly=*/true);
}

LogicalResult realizePointwiseBlocking(ModuleOp module) {
  return realizePointwiseBlockingImpl(module, /*ownershipOnly=*/false);
}

} // namespace intent::gpu

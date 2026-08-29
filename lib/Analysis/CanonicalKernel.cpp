#include "Intent/Analysis/CanonicalKernel.h"

#include "Intent/Dialect/Intent/IR/IntentOps.h"
#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;

namespace intent {
namespace {

void appendOrigins(CoordinateProvenance &destination,
                   const CoordinateProvenance &source) {
  destination.known = destination.known && source.known;
  for (const CoordinateOrigin &origin : source.origins)
    if (!llvm::is_contained(destination.origins, origin))
      destination.origins.push_back(origin);
}

CoordinateProvenance knownWithoutCoordinates() {
  CoordinateProvenance result;
  result.known = true;
  return result;
}

Value absoluteCoordinateSource(Value source) {
  while (auto subregion = source.getDefiningOp<SubregionOp>()) {
    if (subregion.getInputs().empty())
      break;
    source = subregion.getInputs().front();
  }
  return source;
}

DenseI64ArrayAttr dimensionIDs(RankedTensorType tensor) {
  auto encoding = dyn_cast_or_null<TensorShapeAttr>(tensor.getEncoding());
  return encoding ? encoding.getDimensions() : DenseI64ArrayAttr();
}

std::optional<int64_t> constantInteger(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getInt();
  return std::nullopt;
}

struct IndexAccessKey {
  IndexRelationAttr relation;
  SmallVector<Value, 4> operands;

  bool operator==(const IndexAccessKey &other) const {
    return relation == other.relation && operands == other.operands;
  }
};

struct BufferInitializationState {
  DenseSet<Value> complete;
  DenseMap<Value, SmallVector<IndexAccessKey, 2>> points;
};

FailureOr<IndexAccessKey> accessKey(Operation *operation) {
  auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
  auto terms = relation ? relation.getTerms() : ArrayAttr();
  if (!relation || !terms)
    return failure();
  IndexAccessKey key{relation, {}};
  for (Attribute attribute : terms) {
    auto term = dyn_cast<IndexTermAttr>(attribute);
    if (!term)
      return failure();
    for (int64_t position : term.getOperandPositions().asArrayRef()) {
      if (position == -1)
        continue;
      if (position < 0 || position >= operation->getNumOperands())
        return failure();
      key.operands.push_back(operation->getOperand(position));
    }
  }
  return key;
}

bool isFullBufferStore(Operation *operation) {
  if (!isa_and_nonnull<BufferStoreOp>(operation))
    return false;
  auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
  auto terms = relation ? relation.getTerms() : ArrayAttr();
  if (!terms || terms.empty())
    return false;
  return llvm::all_of(terms, [](Attribute attribute) {
    auto term = dyn_cast<IndexTermAttr>(attribute);
    return term && term.getKind() == 0;
  });
}

bool definitelyWritesCoordinate(Block &block, Value buffer,
                                BlockArgument coordinate) {
  for (Operation &operation : block) {
    if (auto store = dyn_cast<BufferStoreOp>(operation);
        store && store.getInputs().front() == buffer) {
      FailureOr<IndexAccessKey> key = accessKey(&operation);
      if (succeeded(key) && key->operands.size() == 1 &&
          key->operands.front() == coordinate)
        return true;
    }
    if (isa<IfOp>(operation) && operation.getNumRegions() == 2 &&
        definitelyWritesCoordinate(operation.getRegion(0).front(), buffer,
                                   coordinate) &&
        definitelyWritesCoordinate(operation.getRegion(1).front(), buffer,
                                   coordinate))
      return true;
  }
  return false;
}

bool loopCoversBuffer(Operation *operation, Value buffer) {
  auto loop = dyn_cast_or_null<ForOp>(operation);
  if (!loop || loop.getInputs().empty() || operation->getNumRegions() != 1)
    return false;
  auto bufferType = dyn_cast<BufferType>(buffer.getType());
  auto tensor = bufferType
                    ? dyn_cast<RankedTensorType>(bufferType.getTensor())
                    : RankedTensorType();
  if (!tensor || tensor.getRank() != 1)
    return false;
  auto domain = loop.getInputs().front().getDefiningOp<DomainOp>();
  if (!domain || (domain.getBounds().size() != 2 &&
                  domain.getBounds().size() != 3) ||
      constantInteger(domain.getBounds().front()) != 0)
    return false;
  if (domain.getBounds().size() == 3 &&
      constantInteger(domain.getBounds()[2]) != 1)
    return false;
  Value stop = domain.getBounds()[1];
  if (tensor.hasStaticShape()) {
    if (constantInteger(stop) != tensor.getDimSize(0))
      return false;
  } else {
    Operation *definition = buffer.getDefiningOp();
    auto shape = definition
                     ? definition->getAttrOfType<ShapeRelationAttr>("shape")
                     : ShapeRelationAttr();
    auto entry = shape && !shape.getAxes().empty()
                     ? dyn_cast<ShapeExprAttr>(shape.getAxes()[0])
                     : ShapeExprAttr();
    if (!entry || entry.getKind() != 1 || entry.getPayload() < 0 ||
        entry.getPayload() >= definition->getNumOperands() ||
        definition->getOperand(entry.getPayload()) != stop)
      return false;
  }
  Block &body = loop.getBody().front();
  return body.getNumArguments() > 0 &&
         definitelyWritesCoordinate(body, buffer, body.getArgument(0));
}

LogicalResult collectDomainValues(Value source,
                                  SmallVectorImpl<Value> &domains) {
  if (isa_and_nonnull<DomainOp>(source.getDefiningOp())) {
    domains.push_back(source);
    return success();
  }
  auto product = source.getDefiningOp<DomainProductOp>();
  if (!product)
    return failure();
  for (Value component : product.getDomains())
    if (failed(collectDomainValues(component, domains)))
      return failure();
  return success();
}

bool hasObservableWrite(Operation *operation) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect());
  });
}

LogicalResult collectLogicalWorkset(
    ParallelOp parallel, ArrayRef<Value> parentDomains,
    ArrayRef<BlockArgument> parentCoordinates,
    SmallVectorImpl<LogicalWorksetFact> &worksets) {
  LogicalWorksetFact current;
  current.parallel = parallel;
  current.domains.append(parentDomains.begin(), parentDomains.end());
  current.coordinates.append(parentCoordinates.begin(),
                             parentCoordinates.end());
  SmallVector<Value> localDomains;
  if (failed(collectDomainValues(parallel.getSource(), localDomains)) ||
      localDomains.empty())
    return parallel.emitOpError(
        "parallel workset requires a finite canonical domain product");
  Block &body = parallel.getBody().front();
  if (body.getNumArguments() != localDomains.size())
    return parallel.emitOpError(
        "parallel coordinates do not match the canonical domain product");
  current.domains.append(localDomains.begin(), localDomains.end());
  current.coordinates.append(body.getArguments().begin(),
                             body.getArguments().end());

  SmallVector<ParallelOp> children;
  bool directWrite = false;
  for (Operation &nested : body.without_terminator()) {
    if (auto child = dyn_cast<ParallelOp>(nested)) {
      children.push_back(child);
      continue;
    }
    directWrite |= hasObservableWrite(&nested);
    nested.walk([&](Operation *candidate) {
      if (candidate != &nested)
        directWrite |= hasObservableWrite(candidate);
    });
  }
  if (!children.empty()) {
    if (directWrite)
      return parallel.emitOpError(
          "parallel region mixes direct writes with nested independent worksets");
    for (ParallelOp child : children)
      if (failed(collectLogicalWorkset(child, current.domains,
                                       current.coordinates, worksets)))
        return failure();
    return success();
  }

  current.state = CanonicalFactState::Exact;
  current.body = &body;
  worksets.push_back(std::move(current));
  return success();
}

void intersectInitialization(BufferInitializationState &destination,
                             const BufferInitializationState &other) {
  SmallVector<Value> remove;
  for (Value buffer : destination.complete)
    if (!other.complete.contains(buffer))
      remove.push_back(buffer);
  for (Value buffer : remove)
    destination.complete.erase(buffer);
  for (auto &entry : destination.points) {
    auto otherEntry = other.points.find(entry.first);
    if (otherEntry == other.points.end()) {
      entry.second.clear();
      continue;
    }
    llvm::erase_if(entry.second, [&](const IndexAccessKey &key) {
      return !llvm::is_contained(otherEntry->second, key);
    });
  }
}

LogicalResult verifyBufferInitializationRegion(
    Region &region, BufferInitializationState &state) {
  if (!llvm::hasSingleElement(region))
    return failure();
  for (Operation &operation : region.front()) {
    if (auto allocation = dyn_cast<BufferOp>(operation)) {
      Value buffer = operation.getResult(0);
      if (allocation.getInitialOperand())
        state.complete.insert(buffer);
      continue;
    }
    if (auto load = dyn_cast<BufferLoadOp>(operation)) {
      Value buffer = load.getInputs().front();
      FailureOr<IndexAccessKey> key = accessKey(&operation);
      bool initialized = state.complete.contains(buffer);
      if (!initialized && succeeded(key))
        initialized = llvm::is_contained(state.points[buffer], *key);
      if (!initialized)
        return operation.emitOpError(
            "logical buffer read is not dominated by a write of the same element set");
      continue;
    }
    if (auto store = dyn_cast<BufferStoreOp>(operation)) {
      Value buffer = store.getInputs().front();
      if (isFullBufferStore(&operation)) {
        state.complete.insert(buffer);
      } else if (FailureOr<IndexAccessKey> key = accessKey(&operation);
                 succeeded(key) &&
                 !llvm::is_contained(state.points[buffer], *key)) {
        state.points[buffer].push_back(*key);
      }
      continue;
    }
    if (isa<IfOp>(operation)) {
      BufferInitializationState thenState = state;
      BufferInitializationState elseState = state;
      if (failed(verifyBufferInitializationRegion(operation.getRegion(0),
                                                  thenState)) ||
          failed(verifyBufferInitializationRegion(operation.getRegion(1),
                                                  elseState)))
        return failure();
      intersectInitialization(thenState, elseState);
      state = std::move(thenState);
      continue;
    }
    if (isa<ForOp, ParallelOp, WhileOp>(operation)) {
      for (Region &nested : operation.getRegions()) {
        BufferInitializationState nestedState = state;
        if (failed(verifyBufferInitializationRegion(nested, nestedState)))
          return failure();
      }
      if (isa<ForOp>(operation)) {
        SmallVector<Value> buffers;
        for (auto &entry : state.points)
          buffers.push_back(entry.first);
        operation.getRegion(0).walk([&](Operation *nested) {
          if (isa<BufferStoreOp>(nested) &&
              !llvm::is_contained(buffers, nested->getOperand(0)))
            buffers.push_back(nested->getOperand(0));
        });
        for (Value buffer : buffers)
          if (loopCoversBuffer(&operation, buffer))
            state.complete.insert(buffer);
      }
      continue;
    }
    for (Region &nested : operation.getRegions()) {
      BufferInitializationState nestedState = state;
      if (failed(verifyBufferInitializationRegion(nested, nestedState)))
        return failure();
    }
  }
  return success();
}

} // namespace

CanonicalKernelAnalysis::CanonicalKernelAnalysis(ModuleOp module)
    : module(module) {}

CoordinateProvenance
CanonicalKernelAnalysis::coordinateProvenance(Value value) {
  if (!value)
    return {};
  if (auto iterator = coordinateCache.find(value);
      iterator != coordinateCache.end())
    return iterator->second;
  if (coordinateActive.lookup(value))
    return {};
  coordinateActive[value] = true;
  CoordinateProvenance result = computeCoordinateProvenance(value);
  coordinateActive.erase(value);
  coordinateCache.try_emplace(value, result);
  return result;
}

CoordinateProvenance
CanonicalKernelAnalysis::computeCoordinateProvenance(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return blockArgumentProvenance(argument);
  return resultProvenance(cast<OpResult>(value));
}

CoordinateProvenance CanonicalKernelAnalysis::blockArgumentProvenance(
    BlockArgument argument) {
  Operation *owner = argument.getOwner()->getParentOp();
  if (!owner)
    return knownWithoutCoordinates();
  if (isa<ParallelOp, ForOp>(owner)) {
    auto domain = dyn_cast<DomainType>(owner->getOperand(0).getType());
    auto region = dyn_cast<RegionType>(owner->getOperand(0).getType());
    unsigned rank = domain ? domain.getRank() : region ? region.getRank() : 0;
    if (argument.getArgNumber() < rank) {
      CoordinateProvenance result = knownWithoutCoordinates();
      result.origins.push_back(
          {absoluteCoordinateSource(owner->getOperand(0)),
           argument.getArgNumber()});
      return result;
    }
    unsigned carry = argument.getArgNumber() - rank + 1;
    return carry < owner->getNumOperands()
               ? coordinateProvenance(owner->getOperand(carry))
               : CoordinateProvenance();
  }
  if (isa<RegionFoldOp, RegionScanOp>(owner)) {
    auto sourceCount = owner->getAttrOfType<IntegerAttr>("source_count");
    if (sourceCount && argument.getArgNumber() < sourceCount.getInt())
      return coordinateProvenance(
          owner->getOperand(argument.getArgNumber()));
  }
  return knownWithoutCoordinates();
}

CoordinateProvenance
CanonicalKernelAnalysis::resultProvenance(OpResult result) {
  Operation *operation = result.getOwner();
  if (isa<IndicesOp>(operation)) {
    CoordinateProvenance provenance = knownWithoutCoordinates();
    auto axis = operation->getAttrOfType<IntegerAttr>("tensor_axis");
    auto rank = cast<RankedTensorType>(operation->getResult(0).getType()).getRank();
    if (axis)
      provenance.origins.push_back(
          {absoluteCoordinateSource(operation->getOperand(0)),
           static_cast<unsigned>(axis.getInt())});
    else if (rank == 1)
      provenance.origins.push_back(
          {absoluteCoordinateSource(operation->getOperand(0)), 0});
    else
      provenance.known = false;
    return provenance;
  }
  if (isa<ConstantOp, DimOp, RegionEndOp, RandomBitsOp, ContractOp,
          ScaledContractOp, SparseContractOp, HistogramOp, ViewLoadOp,
          BufferLoadOp, AtomicLoadOp, AtomicStoreOp, AtomicRMWOp,
          AtomicCompareExchangeOp>(operation))
    return knownWithoutCoordinates();

  if (isa<ExtractOp>(operation)) {
    Value product = operation->getOperand(0);
    auto field = operation->getAttrOfType<IntegerAttr>("field");
    if (field) {
      if (Operation *definition = product.getDefiningOp();
          definition && isa<MakeTupleOp, MakeRecordOp>(definition) &&
          field.getInt() >= 0 &&
          field.getInt() < definition->getNumOperands())
        return coordinateProvenance(definition->getOperand(field.getInt()));
    }
    return coordinateProvenance(product);
  }

  if (isa<GatherOp, ReshapeOp, BroadcastOp, TransposeOp, CastOp, BitcastOp,
          UnaryOp, FullOp>(operation))
    return coordinateProvenance(operation->getOperand(0));

  if (isa<BinaryOp, CompareOp, JoinOp, SelectOp, MaskOp, MakeTupleOp,
          MakeRecordOp>(operation)) {
    CoordinateProvenance combined = knownWithoutCoordinates();
    for (Value operand : operation->getOperands())
      appendOrigins(combined, coordinateProvenance(operand));
    return combined;
  }

  return {};
}

SmallVector<ShapeAxisFact, 4>
CanonicalKernelAnalysis::shapeFacts(Value value) const {
  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor) {
    if (auto view = dyn_cast<ViewType>(value.getType()))
      tensor = dyn_cast<RankedTensorType>(view.getTensor());
    else if (auto buffer = dyn_cast<BufferType>(value.getType()))
      tensor = dyn_cast<RankedTensorType>(buffer.getTensor());
  }
  if (!tensor)
    return {};
  DenseI64ArrayAttr ids = dimensionIDs(tensor);
  SmallVector<ShapeAxisFact, 4> facts;
  facts.reserve(tensor.getRank());
  Operation *definition = value.getDefiningOp();
  auto relation = definition
                      ? definition->getAttrOfType<ShapeRelationAttr>("shape")
                      : ShapeRelationAttr();
  for (unsigned axis = 0; axis < tensor.getRank(); ++axis) {
    ShapeAxisFact fact;
    fact.dimensionIdentity = ids ? ids[axis] : 0;
    if (!tensor.isDynamicDim(axis)) {
      fact.kind = ShapeAxisKind::Static;
      fact.staticExtent = tensor.getDimSize(axis);
    } else if (!definition) {
      fact.kind = ShapeAxisKind::ABI;
    } else if (relation && axis < relation.getAxes().size()) {
      auto entry = dyn_cast<ShapeExprAttr>(relation.getAxes()[axis]);
      if (entry && entry.getKind() == 1) {
        int64_t operand = entry.getPayload();
        if (operand >= 0 && operand < definition->getNumOperands()) {
          fact.kind = ShapeAxisKind::SSAExtent;
          fact.extent = definition->getOperand(operand);
        }
      } else if (entry && entry.getKind() == 2) {
        fact.kind = ShapeAxisKind::Inferred;
      }
    }
    facts.push_back(fact);
  }
  return facts;
}

FailureOr<IndexRelationFact>
CanonicalKernelAnalysis::indexRelation(Operation *operation) {
  auto relation = operation->getAttrOfType<IndexRelationAttr>("index");
  auto terms = relation ? relation.getTerms() : ArrayAttr();
  if (!relation || !terms ||
      operation->getNumOperands() == 0)
    return failure();
  IndexRelationFact result;
  result.source = operation->getOperand(0);
  result.sourceRank = relation.getSourceRank();
  result.resultDimensionIdentities.append(
      relation.getResultDimensions().asArrayRef().begin(),
      relation.getResultDimensions().asArrayRef().end());
  unsigned sourceAxis = 0;
  for (Attribute attribute : terms) {
    auto term = dyn_cast<IndexTermAttr>(attribute);
    if (!term)
      return failure();
    IndexTermFact fact;
    fact.kind = term.getKind();
    if (fact.kind != 1)
      fact.sourceAxis = sourceAxis++;
    for (int64_t operand : term.getOperandPositions().asArrayRef()) {
      if (operand == -1)
        continue;
      if (operand < 0 || operand >= operation->getNumOperands())
        return failure();
      fact.operands.push_back(operation->getOperand(operand));
    }
    for (int64_t value : term.getStaticValues().asArrayRef())
      fact.staticValues.push_back(
          value == std::numeric_limits<int64_t>::min()
              ? std::nullopt
              : std::optional<int64_t>(value));
    fact.coordinate = knownWithoutCoordinates();
    if (fact.sourceAxis && (fact.kind == 0 || fact.kind == 5))
      fact.coordinate.origins.push_back(
          {absoluteCoordinateSource(result.source), *fact.sourceAxis});
    else if (fact.kind == 4 && !fact.operands.empty())
      fact.coordinate.origins.push_back(
          {absoluteCoordinateSource(fact.operands.front()), 0});
    else if (fact.kind == 3 && !fact.operands.empty())
      fact.coordinate = coordinateProvenance(fact.operands.front());
    result.terms.push_back(std::move(fact));
  }
  return result;
}

FailureOr<SmallVector<LogicalWorksetFact, 4>>
CanonicalKernelAnalysis::logicalWorksets(func::FuncOp function) const {
  SmallVector<LogicalWorksetFact, 4> worksets;
  for (Operation &operation : function.getBody().front())
    if (auto parallel = dyn_cast<ParallelOp>(operation))
      if (failed(collectLogicalWorkset(parallel, {}, {}, worksets)))
        return failure();
  if (worksets.empty()) {
    LogicalWorksetFact singleton;
    singleton.state = CanonicalFactState::Exact;
    singleton.body = &function.getBody().front();
    singleton.singleton = true;
    worksets.push_back(std::move(singleton));
  }
  return worksets;
}

LogicalBufferFact
CanonicalKernelAnalysis::logicalBuffer(Operation *operation) const {
  LogicalBufferFact fact;
  auto buffer = dyn_cast_or_null<BufferOp>(operation);
  if (!buffer)
    return fact;
  auto type = dyn_cast<BufferType>(buffer.getResult().getType());
  if (!type || type.getOriginId() == 0)
    return fact;
  fact.state = CanonicalFactState::Exact;
  fact.instanceIdentity = type.getOriginId();
  fact.hasFullInitialValue = static_cast<bool>(buffer.getInitialOperand());
  for (Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<ForOp, WhileOp>(parent)) {
      fact.scope = LogicalBufferScope::IterationPrivate;
      break;
    }
    if (isa<func::FuncOp>(parent))
      break;
  }
  return fact;
}

RegionSegmentFact
CanonicalKernelAnalysis::regionSegment(Operation *operation) const {
  RegionSegmentFact fact;
  uint64_t axis = 0;
  ValueRange sources;
  if (auto fold = dyn_cast_or_null<RegionFoldOp>(operation)) {
    axis = fold.getAxis();
    sources = fold.getInputs().take_front(fold.getSourceCount());
  } else if (auto scan = dyn_cast_or_null<RegionScanOp>(operation)) {
    axis = scan.getAxis();
    sources = scan.getInputs().take_front(scan.getSourceCount());
  } else {
    return fact;
  }
  int64_t dimension = 0;
  for (Value source : sources) {
    auto tensor = dyn_cast<RankedTensorType>(source.getType());
    DenseI64ArrayAttr ids = tensor ? dimensionIDs(tensor)
                                   : DenseI64ArrayAttr();
    if (!tensor || axis >= static_cast<uint64_t>(tensor.getRank()) || !ids ||
        ids[axis] <= 0 || (dimension != 0 && ids[axis] != dimension))
      return fact;
    dimension = ids[axis];
  }
  auto node = operation->getAttrOfType<IntegerAttr>("intent.node");
  if (!node || node.getInt() < 0 || dimension <= 0)
    return fact;
  fact.state = CanonicalFactState::Exact;
  fact.dimensionIdentity = dimension;
  fact.operationIdentity = node.getInt();
  return fact;
}

LogicalResult CanonicalKernelAnalysis::verify() {
  LogicalResult result = success();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (failed(logicalWorksets(function)))
      return failure();
    BufferInitializationState state;
    if (failed(verifyBufferInitializationRegion(function.getBody(), state)))
      return failure();
    WalkResult constructionFacts = function.walk([&](Operation *operation) {
      if (isa<BufferOp>(operation) && !logicalBuffer(operation).isExact()) {
        operation->emitOpError(
            "canonical buffer has no exact lexical allocation fact");
        return WalkResult::interrupt();
      }
      if (isa<RegionFoldOp, RegionScanOp>(operation) &&
          !regionSegment(operation).isExact()) {
        operation->emitOpError(
            "canonical region operation has no exact segment source relation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (constructionFacts.wasInterrupted())
      return failure();
  }
  module.walk([&](Operation *operation) {
    if (failed(result))
      return WalkResult::interrupt();
    if (isa<IndicesOp>(operation)) {
      CoordinateProvenance provenance =
          coordinateProvenance(operation->getResult(0));
      if (!provenance.known || provenance.origins.size() != 1) {
        operation->emitOpError(
            "canonical coordinate provenance is not exact at its source");
        result = failure();
        return WalkResult::interrupt();
      }
    }
    if (operation->hasAttr("index") && failed(indexRelation(operation))) {
      operation->emitOpError(
          "canonical index relation analysis could not consume the typed relation");
      result = failure();
      return WalkResult::interrupt();
    }
    for (Value value : operation->getResults())
      (void)shapeFacts(value);
    return WalkResult::advance();
  });
  return result;
}

} // namespace intent

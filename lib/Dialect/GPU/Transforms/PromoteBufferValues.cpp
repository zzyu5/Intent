#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace intent::gpu {
namespace {

StoreOp fullInitialization(BufferOp buffer, func::FuncOp kernel) {
  auto storage = buffer.getResult().getType();
  PhysicalProgramAnalysis analysis(kernel);
  DominanceInfo dominance(kernel);
  for (Operation *user : buffer.getResult().getUsers()) {
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store->getBlock() != buffer->getBlock() ||
        store.getCoordinates().size() != 1 ||
        store.getSourceAxes() != ArrayRef<int64_t>{0})
      continue;
    auto value = dyn_cast<FragmentType>(store.getValue().getType());
    auto range = store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
    if (!value || !range || value.getShape() != storage.getShape() ||
        value.getOwner() != storage.getOwner() ||
        value.getAxisMaps() != range.getResult().getType().getAxisMaps() ||
        !matchPattern(range.getStart(), m_Zero()) ||
        !matchPattern(range.getLogicalStart(), m_Zero()) ||
        !isUnitStepRange(range) ||
        queryLaunchExpression(range.getExtent()) != storage.getShape()[0] ||
        queryLaunchExpression(range.getLogicalStop()) != storage.getShape()[0] ||
        !analysis.isTailPredicate(store.getValid(),
                                  {{range, range.getLogicalStop()}}))
      continue;
    if (llvm::all_of(buffer.getResult().getUsers(), [&](Operation *other) {
          return other == store || dominance.properlyDominates(store, other);
        }))
      return store;
  }
  return {};
}

bool collectStateRegions(BufferOp buffer, StoreOp initial,
                         llvm::DenseSet<Operation *> &writtenRegions) {
  for (Operation *user : buffer.getResult().getUsers()) {
    if (user == initial)
      continue;
    bool writes = false;
    if (auto load = dyn_cast<LoadOp>(user)) {
      if (load.getResource() != buffer.getResult() ||
          load.getCoordinates().size() != 1 ||
          load.getSourceAxes() != ArrayRef<int64_t>{0})
        return false;
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      if (store.getResource() != buffer.getResult() ||
          store.getCoordinates().size() != 1 ||
          store.getSourceAxes() != ArrayRef<int64_t>{0})
        return false;
      auto value = dyn_cast<FragmentType>(store.getValue().getType());
      if (value) {
        auto range = store.getCoordinates().front().getDefiningOp<MakeRangeOp>();
        if (!range || value.getShape().size() != 1 || !isUnitStepRange(range) ||
            range.getResult().getType().getShape() != value.getShape() ||
            range.getResult().getType().getAxisMaps() != value.getAxisMaps() ||
            queryLaunchExpression(range.getExtent()) != value.getShape()[0])
          return false;
      } else if (!store.getCoordinates().front().getType().isIndex() ||
                 store.getValue().getType() !=
                     buffer.getResult().getType().getElementType() ||
                 (store.getValid() && !store.getValid().getType().isInteger(1))) {
        return false;
      }
      writes = true;
    } else {
      return false;
    }
    for (Operation *parent = user->getParentOp(); parent != buffer->getParentOp();
         parent = parent->getParentOp()) {
      if (!isa<scf::ForOp, scf::IfOp>(parent) ||
          (writes && parent->hasAttr(reductionSourcesAttr)))
        return false;
      if (writes)
        writtenRegions.insert(parent);
    }
  }
  return true;
}

bool hasNestedScalarIndexedReads(BufferOp buffer) {
  bool nestedScalarRead = false;
  for (Operation *user : buffer.getResult().getUsers()) {
    auto loop = user->getParentOfType<scf::ForOp>();
    if (!loop)
      continue;
    Value coordinate;
    if (auto load = dyn_cast<LoadOp>(user)) {
      if (isa<FragmentType>(load.getType()))
        return false;
      coordinate = load.getCoordinates().front();
      if (loop->getParentOfType<scf::ForOp>() &&
          !coordinate.getDefiningOp<arith::ConstantOp>())
        nestedScalarRead = true;
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      if (isa<FragmentType>(store.getValue().getType()))
        return false;
    }
  }
  return nestedScalarRead;
}

class BufferStatePromotion {
public:
  BufferStatePromotion(BufferOp buffer, StoreOp initial,
                       const llvm::DenseSet<Operation *> &writtenRegions)
      : buffer(buffer), initial(initial), writtenRegions(writtenRegions),
        indices(initial.getCoordinates().front()),
        stateType(cast<FragmentType>(initial.getValue().getType())) {}

  void run() {
    Value state = initial.getValue();
    auto begin = std::next(initial->getIterator());
    rewrite(*initial->getBlock(), begin, state);
    initial.erase();
    buffer.erase();
  }

private:
  FragmentType withElement(Type element) {
    return FragmentType::get(stateType.getContext(), element, stateType.getShape(),
                            stateType.getAxisMaps(), stateType.getValidity(),
                            stateType.getOwner());
  }

  void rewrite(Block &block, Block::iterator begin, Value &state) {
    for (auto iterator = begin; iterator != block.end();) {
      Operation *operation = &*iterator++;
      if (auto load = dyn_cast<LoadOp>(operation);
          load && load.getResource() == buffer.getResult()) {
        OpBuilder builder(load);
        Value value = builder.create<GatherOp>(
            load.getLoc(), load.getResult().getType(), state,
            load.getCoordinates(), load.getValid(), load.getFill(),
            load.getSourceAxes());
        if (Attribute origin = load->getAttr(originAttr))
          value.getDefiningOp()->setAttr(originAttr, origin);
        load.getResult().replaceAllUsesWith(value);
        load.erase();
      } else if (auto store = dyn_cast<StoreOp>(operation);
                 store && store.getResource() == buffer.getResult()) {
        OpBuilder builder(store);
        Location location = store.getLoc();
        auto predicate = withElement(builder.getI1Type());
        Value valid, value;
        if (auto range =
                store.getCoordinates().front().getDefiningOp<MakeRangeOp>()) {
          Value start = builder.create<SplatOp>(location, indices.getType(),
                                                range.getStart());
          Value offsets = builder.create<BinaryOp>(
              location, indices.getType(), indices, start, BinaryOperator::Subtract);
          Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
          Value zeroIndices = builder.create<SplatOp>(location, indices.getType(),
                                                      zero);
          Value extent = builder.create<SplatOp>(location, indices.getType(),
                                                 range.getExtent());
          Value nonNegative = builder.create<CompareOp>(
              location, predicate, offsets, zeroIndices, ComparePredicate::Ge);
          Value inRange = builder.create<CompareOp>(
              location, predicate, offsets, extent, ComparePredicate::Lt);
          valid = builder.create<BinaryOp>(location, predicate, nonNegative,
                                           inRange, BinaryOperator::LogicalAnd);
          if (store.getValid()) {
            Value scalarFalse = builder.create<arith::ConstantIntOp>(location, 0, 1);
            Value falseValues = builder.create<SplatOp>(location, predicate,
                                                        scalarFalse);
            Value selected = builder.create<GatherOp>(
                location, predicate, store.getValid(), ValueRange{offsets}, valid,
                falseValues, ArrayRef<int64_t>{0});
            valid = builder.create<BinaryOp>(location, predicate, valid, selected,
                                             BinaryOperator::LogicalAnd);
          }
          Value scalarZero = builder.create<arith::ConstantOp>(
              location, stateType.getElementType(),
              builder.getZeroAttr(stateType.getElementType()));
          Value zeroValues = builder.create<SplatOp>(location, stateType,
                                                      scalarZero);
          value = builder.create<GatherOp>(
              location, stateType, store.getValue(), ValueRange{offsets}, valid,
              zeroValues, ArrayRef<int64_t>{0});
        } else {
          Value index = builder.create<SplatOp>(
              location, indices.getType(), store.getCoordinates().front());
          valid = builder.create<CompareOp>(location, predicate, indices,
                                            index, ComparePredicate::Eq);
          if (store.getValid()) {
            Value active = builder.create<SplatOp>(location, predicate,
                                                   store.getValid());
            valid = builder.create<BinaryOp>(location, predicate, valid, active,
                                             BinaryOperator::LogicalAnd);
          }
          value = builder.create<SplatOp>(location, stateType, store.getValue());
        }
        state = builder.create<SelectOp>(location, stateType, valid, value, state);
        if (Attribute origin = store->getAttr(originAttr))
          state.getDefiningOp()->setAttr(originAttr, origin);
        store.erase();
      } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        if (!writtenRegions.contains(loop)) {
          Value unchanged = state;
          rewrite(*loop.getBody(), loop.getBody()->begin(), unchanged);
          continue;
        }
        OpBuilder builder(loop);
        SmallVector<Value> initials(loop.getInitArgs());
        initials.push_back(state);
        auto replacement = builder.create<scf::ForOp>(
            loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
            loop.getStep(), initials);
        replacement->setAttrs(loop->getAttrs());
        replacement->removeAttr(independentIterationAttr);
        for (auto [oldArgument, argument] :
             llvm::zip(loop.getBody()->getArguments(),
                       replacement.getBody()->getArguments()))
          oldArgument.replaceAllUsesWith(argument);
        auto *body = replacement.getBody();
        if (!body->empty())
          body->getTerminator()->erase();
        body->getOperations().splice(body->end(), loop.getBody()->getOperations());
        Value carried = replacement.getRegionIterArgs().back();
        rewrite(*body, body->begin(), carried);
        auto yield = body->getTerminator();
        yield->insertOperands(yield->getNumOperands(), carried);
        for (auto [oldResult, result] :
             llvm::zip(loop.getResults(), replacement.getResults()))
          oldResult.replaceAllUsesWith(result);
        state = replacement.getResults().back();
        loop.erase();
      } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
        if (!writtenRegions.contains(branch)) {
          for (Region &region : branch->getRegions())
            if (!region.empty()) {
              Value unchanged = state;
              rewrite(region.front(), region.front().begin(), unchanged);
            }
          continue;
        }
        OpBuilder builder(branch);
        SmallVector<Type> types(branch.getResultTypes());
        types.push_back(stateType);
        auto replacement = builder.create<scf::IfOp>(
            branch.getLoc(), types, branch.getCondition(), true);
        replacement->setAttrs(branch->getAttrs());
        for (auto [oldRegion, region] :
             llvm::zip(branch->getRegions(), replacement->getRegions())) {
          Block &body = region.front();
          if (!body.empty())
            body.getTerminator()->erase();
          if (!oldRegion.empty())
            body.getOperations().splice(body.end(),
                                        oldRegion.front().getOperations());
          else {
            builder.setInsertionPointToEnd(&body);
            builder.create<scf::YieldOp>(branch.getLoc());
          }
          Value carried = state;
          rewrite(body, body.begin(), carried);
          auto yield = body.getTerminator();
          yield->insertOperands(yield->getNumOperands(), carried);
        }
        for (auto [oldResult, result] :
             llvm::zip(branch.getResults(), replacement.getResults()))
          oldResult.replaceAllUsesWith(result);
        state = replacement.getResults().back();
        branch.erase();
      }
    }
  }

  BufferOp buffer;
  StoreOp initial;
  const llvm::DenseSet<Operation *> &writtenRegions;
  Value indices;
  FragmentType stateType;
};

} // namespace

LogicalResult promoteBufferValues(ModuleOp module) {
  auto kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  auto capabilities = (*kernel)->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
  SmallVector<BufferOp> buffers;
  for (BufferOp buffer : (*kernel).getOps<BufferOp>())
    buffers.push_back(buffer);
  llvm::DenseSet<int64_t> removedEffects;
  bool changed = false;
  // Limit promoted words by the maximum CTA thread count. Native lowering still
  // owns the register layout and allocation of these fragments.
  int64_t remainingWords = capabilities.getMaxThreadsPerBlock();
  for (BufferOp buffer : buffers) {
    auto type = buffer.getResult().getType();
    if (type.getScope().getValue() != BufferScope::ProgramPrivate ||
        type.getWorkspace() || type.getShape().size() != 1)
      continue;
    auto extent =
        constantPhysicalExpression(cast<PhysicalExprAttr>(type.getShape()[0]));
    unsigned words = type.getElementType().isIndex()
                         ? 2
                         : std::max(1u,
                                    (type.getElementType().getIntOrFloatBitWidth() + 31) / 32);
    if (!extent || *extent <= 0 || *extent > remainingWords / words)
      continue;
    StoreOp initial = fullInitialization(buffer, *kernel);
    llvm::DenseSet<Operation *> writtenRegions;
    if (initial && collectStateRegions(buffer, initial, writtenRegions)) {
      // A scalar recurrence repeatedly indexing its state has no vector reuse.
      // Promoting it would carry and dynamically extract a complete tile through
      // every inner iteration. Keep the existing buffer accesses in that case.
      if (*extent > 1 && hasNestedScalarIndexedReads(buffer))
        continue;
      for (Operation *user : buffer.getResult().getUsers())
        if (isa<StoreOp>(user))
          if (auto origin = user->getAttrOfType<IntegerAttr>(originAttr))
            removedEffects.insert(origin.getInt());
      BufferStatePromotion(buffer, initial, writtenRegions).run();
      remainingWords -= *extent * words;
      changed = true;
    }
  }
  if (!changed)
    return success();
  // These private writes now exist as SSA state transitions. Other observable
  // effects, including another physical slice of the same origin, stay required.
  (*kernel).walk([&](Operation *operation) {
    if (isa<StoreOp, ScatterReduceOp, AtomicStoreOp, AtomicRMWOp,
            AtomicCompareExchangeOp>(operation))
      if (auto origin = operation->getAttrOfType<IntegerAttr>(originAttr))
        removedEffects.erase(origin.getInt());
  });
  SmallVector<Attribute> effects;
  for (Attribute origin : (*kernel)->getAttrOfType<ArrayAttr>(effectOriginsAttr))
    if (!removedEffects.contains(cast<IntegerAttr>(origin).getInt()))
      effects.push_back(origin);
  (*kernel)->setAttr(effectOriginsAttr,
                     ArrayAttr::get(module.getContext(), effects));
  return realizeAccessComposition(module);
}

} // namespace intent::gpu

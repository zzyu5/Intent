#include "ImplementationInputs.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Utilities.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::cpu {
namespace {

bool sameRepresentation(const InputRequirement &first, const InputRequirement &second) {
  return first.elementType == second.elementType && first.panelAxis == second.panelAxis &&
      first.panelSize == second.panelSize && first.alignment == second.alignment && first.reuse == second.reuse;
}

memref::AllocOp allocateRepresentation(OpBuilder &b, Location loc, Value source,
    const InputRequirement &requirement, SmallVectorImpl<Value> &sourceSizes, bool transposed = false) {
  auto type = cast<MemRefType>(source.getType());
  auto sourceAxis = [&](unsigned axis) { return transposed ? 1 - axis : axis; };
  for (int64_t axis = 0; axis < type.getRank(); ++axis)
    sourceSizes.push_back(b.create<memref::DimOp>(loc, source, sourceAxis(axis)));
  Value panel = index(b, loc, requirement.panelSize);
  Value count = b.create<arith::CeilDivSIOp>(loc, sourceSizes[requirement.panelAxis], panel);
  int64_t staticExtent = type.getDimSize(sourceAxis(requirement.panelAxis));
  SmallVector<int64_t> shape{ShapedType::isDynamic(staticExtent) ? ShapedType::kDynamic
      : static_cast<int64_t>(llvm::divideCeil(static_cast<uint64_t>(staticExtent),
                                            static_cast<uint64_t>(requirement.panelSize)))};
  SmallVector<Value> sizes{count};
  for (int64_t axis = 0; axis < type.getRank(); ++axis) {
    if (axis == requirement.panelAxis) continue;
    shape.push_back(type.getDimSize(sourceAxis(axis)));
    sizes.push_back(sourceSizes[axis]);
  }
  shape.push_back(requirement.panelSize);
  SmallVector<Value> dynamic;
  for (auto [axis, size] : llvm::enumerate(sizes))
    if (ShapedType::isDynamic(shape[axis])) dynamic.push_back(size);
  auto storage = b.create<memref::AllocOp>(loc, MemRefType::get(shape, requirement.elementType), dynamic);
  storage.setAlignment(requirement.alignment);
  return storage;
}

bool panelAligned(OpFoldResult offset, int64_t panel) {
  if (panel == 1) return true;
  llvm::DenseMap<Value, bool> known;
  std::function<bool(OpFoldResult)> aligned = [&](OpFoldResult value) {
    if (auto constant = getConstantIntValue(value)) return *constant % panel == 0;
    if (!llvm::isPowerOf2_64(panel)) return false;
    Value dynamic = cast<Value>(value);
    auto found = known.find(dynamic);
    if (found != known.end()) return found->second;
    bool result = false;
    if (auto argument = dyn_cast<BlockArgument>(dynamic)) {
      if (auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
          loop && argument == loop.getInductionVar())
        result = aligned(loop.getLowerBound()) && aligned(loop.getStep());
    } else if (auto product = dynamic.getDefiningOp<arith::MulIOp>()) {
      result = aligned(product.getLhs()) || aligned(product.getRhs());
    } else if (auto subtract = dynamic.getDefiningOp<arith::SubIOp>()) {
      auto remainder = subtract.getRhs().getDefiningOp<arith::RemSIOp>();
      auto divisor = remainder ? getConstantIntValue(remainder.getRhs()) : std::nullopt;
      result = remainder && remainder.getLhs() == subtract.getLhs() && divisor &&
          *divisor > 0 && *divisor % panel == 0;
      result |= aligned(subtract.getLhs()) && aligned(subtract.getRhs());
    } else if (auto select = dynamic.getDefiningOp<arith::SelectOp>()) {
      result = aligned(select.getTrueValue()) && aligned(select.getFalseValue());
    } else if (Operation *operation = dynamic.getDefiningOp();
        isa_and_nonnull<arith::AddIOp, arith::MinSIOp, arith::MaxSIOp>(operation)) {
      result = llvm::all_of(operation->getOperands(), [&](Value operand) { return aligned(operand); });
    }
    known[dynamic] = result;
    return result;
  };
  return aligned(offset);
}

}

std::optional<ConsumerWindow> consumerWindow(Value source, const InputRequirement &requirement,
                                            Operation *consumer) {
  if (requirement.reuse != InputReuse::Consumers || requirement.panelAxis >= 2 ||
      requirement.panelSize <= 0 || requirement.windowAlignment <= 0) return std::nullopt;
  auto view = source.getDefiningOp<memref::SubViewOp>();
  bool transposed = false;
  if (!view && source.getDefiningOp<memref::AllocOp>()) {
    StorageAnalysis storage(consumer->getParentOfType<func::FuncOp>());
    auto swap = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, source.getContext());
    for (Operation *user : source.getUsers()) {
      auto producer = dyn_cast<linalg::GenericOp>(user);
      if (!producer || producer.getNumResults() || producer.getInputs().size() != 1 ||
          producer.getOutputs().size() != 1 || producer.getOutputs()[0] != source ||
          producer.getIteratorTypesArray() != SmallVector<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel}) continue;
      Block &body = producer.getRegion().front();
      auto maps = producer.getIndexingMapsArray();
      if (body.getOperations().size() != 1 || body.getTerminator()->getOperand(0) != body.getArgument(0) ||
          !maps[0].isPermutation() || !maps[1].isPermutation() ||
          maps[0].compose(inversePermutation(maps[1])) != swap ||
          !storage.unchangedBetween(source, producer, consumer)) continue;
      view = producer.getInputs()[0].getDefiningOp<memref::SubViewOp>();
      if (view) { transposed = true; break; }
    }
  }
  if (!view || view.getType().getRank() != 2 || view.getSourceType().getRank() != 2 ||
      llvm::any_of(view.getMixedStrides(), [](OpFoldResult stride) { return getConstantIntValue(stride) != 1; }))
    return std::nullopt;
  auto full = [&](unsigned axis) {
    return getConstantIntValue(view.getMixedOffsets()[axis]) == 0 &&
        haveEqualExtents(ValueBoundsConstraintSet::Variable(view.getSource(), axis),
                         ValueBoundsConstraintSet::Variable(view.getMixedSizes()[axis]));
  };
  unsigned axis;
  if (full(1)) axis = 0;
  else if (full(0)) axis = 1;
  else return std::nullopt;
  unsigned panelAxis = transposed ? 1 - requirement.panelAxis : requirement.panelAxis;
  if (axis == panelAxis && getConstantIntValue(view.getMixedSizes()[axis]) != 1 &&
      !panelAligned(view.getMixedOffsets()[axis], requirement.windowAlignment)) return std::nullopt;
  return ConsumerWindow{view, axis, transposed};
}

bool hasIndependentWindowCoordinates(memref::SubViewOp window, Operation *scope, Value groupCoordinate) {
  llvm::SmallPtrSet<Operation *, 16> checked;
  std::function<bool(Value)> independent = [&](Value value) {
    if (value == groupCoordinate) return true;
    Operation *owner = value.getParentRegion()->getParentOp();
    if (owner != scope && !scope->isAncestor(owner)) return true;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto inner = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      return inner && inner != scope && scope->isAncestor(inner) && argument == inner.getInductionVar();
    }
    Operation *definition = value.getDefiningOp();
    if (!definition || definition->getNumRegions() || !isMemoryEffectFree(definition)) return false;
    if (!checked.insert(definition).second) return true;
    return llvm::all_of(definition->getOperands(), independent);
  };
  // Inner traversals still identify actual source rows. A window that advances
  // directly with this consumer loop does not provide reuse across its iterations.
  auto invariant = [&](OpFoldResult value) { return isa<Attribute>(value) || independent(cast<Value>(value)); };
  return llvm::all_of(window.getMixedOffsets(), invariant) && llvm::all_of(window.getMixedSizes(), invariant);
}

FailureOr<SmallVector<InputSupply>> ImplementationInputs::prepare(linalg::GenericOp operation,
    ArrayRef<InputRequirement> requirements, const Implementation &implementation) {
  if (auto reason = checkInputRequirements(operation, requirements))
    return operation.emitError(*reason), failure();
  SmallVector<InputSupply> supplies;
  for (auto requirement : requirements) {
    Value source = operation.getInputs()[requirement.operand];
    if (requirement.reuse == InputReuse::Group) continue;
    if (auto supply = prepareWindow(source, requirement, operation)) supplies.push_back(*supply);
    else {
      Operation *scope = consumerScope(source, operation, requirement, implementation);
      if (auto loop = dyn_cast<scf::ForOp>(scope)) guardLoop(loop);
      supplies.push_back(materialize(source, requirement, scope));
    }
  }
  return supplies;
}

void ImplementationInputs::guardLoop(scf::ForOp loop) {
  if (llvm::is_contained(guardedLoops, loop.getOperation())) return;
  OpBuilder b(loop);
  Value nonempty = b.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
      loop.getLowerBound(), loop.getUpperBound());
  auto guard = b.create<scf::IfOp>(loop.getLoc(), nonempty, false);
  loop->moveBefore(guard.thenBlock()->getTerminator());
  guardedLoops.push_back(loop);
}

std::optional<InputSupply> ImplementationInputs::prepareWindow(Value source,
    const InputRequirement &requirement, linalg::GenericOp operation) {
  auto window = consumerWindow(source, requirement, operation);
  if (!window) return std::nullopt;
  auto view = window->view;
  Value base = view.getSource();
  StorageAnalysis storage(function);
  auto interface = function->getAttrOfType<EntryRequirementsAttr>(entryRequirementsAttr);
  if (!interface || !interface.getDisjointOutputs() || !storage.isReadOnly(base)) return std::nullopt;
  unsigned axis = window->axis;
  unsigned logicalAxis = window->transposed ? 1 - axis : axis;
  OpBuilder b(operation);
  auto sizes = view.getMixedSizes(), offsets = view.getMixedOffsets();

  DominanceInfo dominance(function);
  scf::ForOp scope;
  for (Operation *parent = operation->getParentOp(); parent != function; parent = parent->getParentOp()) {
    if (isa<scf::IfOp>(parent)) continue;
    auto candidate = dyn_cast<scf::ForOp>(parent);
    if (!candidate || candidate.getNumResults()) break;
    auto step = getConstantIntValue(candidate.getStep());
    if (!step || *step <= 0 || !candidate->isAncestor(view) || !dominance.dominates(base, candidate)) continue;
    if (!hasIndependentWindowCoordinates(view, candidate)) continue;
    if (storage.preserves(candidate, base)) scope = candidate;
  }
  if (!scope) return std::nullopt;

  PreparedWindow *prepared = nullptr;
  for (auto &candidate : windows)
    if (candidate.source == base && candidate.scope == scope && candidate.axis == axis &&
        candidate.transposed == window->transposed && sameRepresentation(candidate.requirement, requirement)) {
      prepared = &candidate;
      break;
    }
  Location loc = operation.getLoc();
  if (!prepared) {
    guardLoop(scope);
    b.setInsertionPoint(scope);
    SmallVector<Value> sourceSizes;
    auto storage = allocateRepresentation(b, loc, base, requirement, sourceSizes, window->transposed);
    auto initialized = b.create<memref::AllocOp>(loc, MemRefType::get({ShapedType::kDynamic}, b.getI1Type()),
        ValueRange{sourceSizes[logicalAxis]});
    Value zero = index(b, loc, 0), clear = b.create<arith::ConstantIntOp>(loc, 0, 1);
    loop(b, loc, zero, sourceSizes[logicalAxis], 1, [&](Value row) {
      b.create<memref::StoreOp>(loc, clear, initialized, row);
    });
    b.setInsertionPointAfter(scope);
    b.create<memref::DeallocOp>(loc, initialized);
    b.create<memref::DeallocOp>(loc, storage);
    windows.push_back({base, requirement, axis, window->transposed, scope, storage, initialized});
    prepared = &windows.back();
  }
  b.setInsertionPoint(operation);
  Value zero = index(b, loc, 0), panel = index(b, loc, requirement.panelSize);
  Value begin = getValueOrCreateConstantIndexOp(b, loc, offsets[axis]);
  Value count = getValueOrCreateConstantIndexOp(b, loc, sizes[axis]);
  Value width = getValueOrCreateConstantIndexOp(b, loc, sizes[1 - axis]);
  Value clear = b.create<arith::ConstantIntOp>(loc, 0, 1), ready = b.create<arith::ConstantIntOp>(loc, 1, 1);
  // A marker covers one complete source slice, including only its valid tail.
  // Fill at the original consumer so guards never introduce extra input reads.
  loop(b, loc, zero, count, 1, [&](Value row) {
    Value coordinate = add(b, loc, begin, row);
    Value initialized = b.create<memref::LoadOp>(loc, prepared->initialized, coordinate);
    Value needed = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, initialized, clear);
    auto fill = b.create<scf::IfOp>(loc, needed, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(fill.thenBlock());
    auto copy = [&](Value other, Value ordinal, Value storageRow, Value lane) {
      SmallVector<Value> logical(2);
      logical[axis] = row;
      logical[1 - axis] = other;
      Value value = b.create<memref::LoadOp>(loc, view, logical);
      if (value.getType() != requirement.elementType) value = b.create<arith::ExtFOp>(loc, requirement.elementType, value);
      b.create<memref::StoreOp>(loc, value, prepared->storage, ValueRange{ordinal, storageRow, lane});
    };
    if (logicalAxis == requirement.panelAxis) {
      Value ordinal = b.create<arith::DivSIOp>(loc, coordinate, panel);
      Value lane = b.create<arith::RemSIOp>(loc, coordinate, panel);
      loop(b, loc, zero, width, 1, [&](Value other) { copy(other, ordinal, other, lane); });
    } else {
      auto copyPanel = [&](Value ordinal, Value lanes) {
        Value start = multiply(b, loc, ordinal, panel);
        loop(b, loc, zero, lanes, 1, [&](Value lane) { copy(add(b, loc, start, lane), ordinal, coordinate, lane); });
      };
      Value full = b.create<arith::DivSIOp>(loc, width, panel), tail = b.create<arith::RemSIOp>(loc, width, panel);
      loop(b, loc, zero, full, 1, [&](Value ordinal) { copyPanel(ordinal, panel); });
      copyPanel(full, tail);
    }
    b.create<memref::StoreOp>(loc, ready, prepared->initialized, coordinate);
  });
  SmallVector<Value> begins(2, zero);
  begins[logicalAxis] = b.create<arith::SubIOp>(loc, zero, begin);
  return InputSupply{requirement.operand, requirement.panelAxis, requirement.panelSize, prepared->storage, std::move(begins)};
}

Operation *ImplementationInputs::consumerScope(Value source, linalg::GenericOp operation,
    const InputRequirement &requirement, const Implementation &implementation) {
  if (auto branch = dyn_cast<scf::IfOp>(operation->getParentOp()); branch && !branch.getElseRegion().empty()) {
    StorageAnalysis storage(function);
    if (storage.isReadOnly(source) && storage.preserves(branch, source) &&
        DominanceInfo(function).dominates(source, branch)) {
      auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
      auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
      Block *other = operation->getParentRegion() == &branch.getThenRegion() ? branch.elseBlock() : branch.thenBlock();
      // Both successors must unconditionally require the same read-only
      // snapshot. Enclosing guards, including empty-work guards, stay intact.
      for (auto candidate : other->getOps<linalg::GenericOp>()) {
        if (!isMatrixContraction(candidate) || candidate->hasAttr("intent_cpu.microtile") ||
            candidate->getAttrOfType<ImplementationAttr>("intent_cpu.implementation") != binding) continue;
        for (auto requested : implementation.inputRequirements(candidate, configuration, binding))
          if (requested.operand < candidate.getInputs().size() &&
              candidate.getInputs()[requested.operand] == source && sameRepresentation(requirement, requested)) return branch;
      }
    }
  }
  auto loop = dyn_cast<scf::ForOp>(operation->getParentOp());
  if (!loop || loop.getNumResults()) return operation;
  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0 || !DominanceInfo(function).dominates(source, loop)) return operation;
  StorageAnalysis storage(function);
  return storage.preserves(loop, source) ? loop.getOperation() : operation;
}

FailureOr<InputSupply> ImplementationInputs::prepareCaptured(linalg::GenericOp operation,
    memref::LoadOp input, const InputRequirement &requirement, Operation *scope) {
  if (input->getBlock() != &operation.getRegion().front() ||
      (scope != operation && !scope->isAncestor(operation)) || requirement.reuse != InputReuse::Consumers)
    return operation.emitError("captured input supply requires its consumer load and enclosing scope"), failure();
  Value source = input.getMemref();
  DominanceInfo dominance(function);
  StorageAnalysis storage(function);
  if (!dominance.dominates(source, scope) ||
      (scope != operation &&
       (!storage.isReadOnly(source) || !storage.preserves(scope, source))))
    return operation.emitError("captured input supply cannot preserve its scoped read snapshot"), failure();
  if (auto reason = checkInputRequirement(source, input.getResult(), requirement))
    return operation.emitError(*reason), failure();
  return materialize(source, requirement, scope);
}

InputSupply ImplementationInputs::materialize(Value source, const InputRequirement &requirement, Operation *scope) {
  StorageAnalysis analysis(function);
  DominanceInfo dominance(function);
  auto type = cast<MemRefType>(source.getType());
  memref::AllocOp storage;
  for (auto &previous : prepared) {
    Operation *consumer = previous.allocation->getBlock()->findAncestorOpInBlock(*scope);
    if (previous.source != source || previous.requirement.panelAxis != requirement.panelAxis ||
        previous.requirement.elementType != requirement.elementType ||
        previous.requirement.panelSize != requirement.panelSize ||
        previous.requirement.alignment < requirement.alignment ||
        !consumer || !dominance.dominates(previous.allocation.getOperation(), scope) ||
        !previous.allocation->isBeforeInBlock(consumer) ||
        (previous.allocation->getBlock() != scope->getBlock() && !analysis.isReadOnly(source)) ||
        !analysis.readStable(source, previous.allocation, consumer)) continue;
    auto lifetime = analysis.lifetime(previous.allocation);
    if (!lifetime || !lifetime->aliases.complete) continue;
    storage = previous.allocation;
    if (lifetime->end->isBeforeInBlock(consumer)) lifetime->end->moveAfter(consumer);
    break;
  }
  if (!storage) {
    OpBuilder b(scope);
    Location loc = scope->getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1), panel = index(b, loc, requirement.panelSize);
    SmallVector<Value> sourceSizes;
    storage = allocateRepresentation(b, loc, source, requirement, sourceSizes);
    Value extent = sourceSizes[requirement.panelAxis];

    auto copyPanel = [&](Value ordinal, Value width) {
      SmallVector<Value> logical(type.getRank()), physical{ordinal};
      Value begin = multiply(b, loc, ordinal, panel);
      std::function<void(unsigned)> axes = [&](unsigned axis) {
        if (axis == static_cast<unsigned>(type.getRank())) {
          loop(b, loc, zero, width, 1, [&](Value lane) {
            logical[requirement.panelAxis] = add(b, loc, begin, lane);
            physical.push_back(lane);
            Value value = b.create<memref::LoadOp>(loc, source, logical);
            if (value.getType() != requirement.elementType)
              value = b.create<arith::ExtFOp>(loc, requirement.elementType, value);
            b.create<memref::StoreOp>(loc, value, storage, physical);
            physical.pop_back();
          });
        } else if (axis == requirement.panelAxis) {
          axes(axis + 1);
        } else {
          loop(b, loc, zero, sourceSizes[axis], 1, [&](Value coordinate) {
            logical[axis] = coordinate;
            physical.push_back(coordinate);
            axes(axis + 1);
            physical.pop_back();
          });
        }
      };
      axes(0);
    };
    Value full = b.create<arith::DivSIOp>(loc, extent, panel);
    Value tail = b.create<arith::RemSIOp>(loc, extent, panel);
    if (scope->getParentOfType<scf::ForOp>() || scope->getParentOfType<scf::ParallelOp>() ||
        scope->getParentOfType<TasksOp>() || scope->getParentOfType<TaskDispatchOp>()) {
      loop(b, loc, zero, full, 1, [&](Value ordinal) { copyPanel(ordinal, panel); });
    } else {
      auto parallel = b.create<scf::ParallelOp>(loc, ValueRange{zero}, ValueRange{full}, ValueRange{one});
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(parallel.getBody());
      copyPanel(parallel.getInductionVars()[0], panel);
    }
    auto hasTail = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, tail, zero);
    auto remainder = b.create<scf::IfOp>(loc, hasTail, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(remainder.thenBlock());
      copyPanel(full, tail);
    }
    b.setInsertionPointAfter(scope);
    b.create<memref::DeallocOp>(loc, storage);
    prepared.push_back({source, requirement, storage});
  }
  OpBuilder b(scope);
  SmallVector<Value> begins(type.getRank(), index(b, scope->getLoc(), 0));
  return {requirement.operand, requirement.panelAxis, requirement.panelSize, storage, std::move(begins)};
}

FailureOr<SmallVector<InputSupply>> ImplementationInputs::prepareGroup(OpBuilder &b,
    linalg::GenericOp operation, const ContractionTile &tile, ConfigurationAttr configuration,
    ArrayRef<InputRequirement> requirements) {
  SmallVector<InputSupply> supplies;
  for (auto requirement : requirements) {
    if (requirement.reuse != InputReuse::Group) continue;
    auto loc = operation.getLoc();
    Value source = operation.getInputs()[requirement.operand];
    auto sourceType = cast<MemRefType>(source.getType());
    auto map = operation.getIndexingMapsArray()[requirement.operand];
    if (sourceType.getRank() != 2 || !map.isProjectedPermutation())
      return operation.emitError("grouped input supply requires a projected matrix input"), failure();
    SmallVector<Value> starts{tile.mBegin, tile.nBegin, tile.kBegin};
    SmallVector<Value> counts{tile.mCount, tile.nCount, tile.depth};
    SmallVector<int64_t> capacities{configuration.getTileM(), configuration.getTileN(), configuration.getTileK()};
    SmallVector<OpFoldResult> offsets, sizes;
    SmallVector<Value> begins;
    unsigned otherAxis = 1 - requirement.panelAxis;
    for (AffineExpr expression : map.getResults()) {
      auto axis = cast<AffineDimExpr>(expression).getPosition();
      offsets.push_back(starts[axis]);
      sizes.push_back(counts[axis]);
      begins.push_back(starts[axis]);
    }
    auto other = cast<AffineDimExpr>(map.getResult(otherAxis)).getPosition();
    int64_t capacity = capacities[other];
    // A valid tile window cannot exceed its current source dimension. Keep the
    // panel pitch/alignment unchanged; only remove unused rows of its storage.
    // Empty dimensions retain the existing allocation form, which remains under
    // the zero-work loop guard; this optimization does not introduce zero slots.
    if (auto bound = constantDimensionUpperBound(source, otherAxis); bound && *bound > 0)
      capacity = std::min(capacity, *bound);
    auto storage = b.create<memref::AllocaOp>(loc,
        MemRefType::get({1, capacity, requirement.panelSize}, requirement.elementType));
    storage.setAlignment(requirement.alignment);
    SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
    Value window = b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);
    SmallVector<OpFoldResult> packedOffsets(3, b.getIndexAttr(0));
    SmallVector<OpFoldResult> packedSizes{b.getIndexAttr(1), sizes[otherAxis], sizes[requirement.panelAxis]};
    SmallVector<OpFoldResult> packedStrides(3, b.getIndexAttr(1));
    auto resultType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
        {ShapedType::kDynamic, ShapedType::kDynamic}, storage.getType(), packedOffsets, packedSizes, packedStrides));
    Value destination = b.create<memref::SubViewOp>(loc, resultType, storage, packedOffsets, packedSizes, packedStrides);
    if (requirement.panelAxis == 1 && sourceType.getElementType() == requirement.elementType) {
      b.create<memref::CopyOp>(loc, window, destination);
    } else {
      auto zero = index(b, loc, 0);
      loop(b, loc, zero, cast<Value>(sizes[0]), 1, [&](Value row) {
        loop(b, loc, zero, cast<Value>(sizes[1]), 1, [&](Value column) {
          Value value = b.create<memref::LoadOp>(loc, window, ValueRange{row, column});
          if (value.getType() != requirement.elementType)
            value = b.create<arith::ExtFOp>(loc, requirement.elementType, value);
          SmallVector<Value> coordinates = requirement.panelAxis == 1 ? SmallVector<Value>{row, column}
                                                                     : SmallVector<Value>{column, row};
          b.create<memref::StoreOp>(loc, value, destination, coordinates);
        });
      });
    }
    supplies.push_back({requirement.operand, requirement.panelAxis, requirement.panelSize, storage, std::move(begins)});
  }
  return supplies;
}

}

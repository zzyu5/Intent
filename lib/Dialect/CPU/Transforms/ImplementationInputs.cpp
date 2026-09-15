#include "ImplementationInputs.h"
#include "Utilities.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
namespace intent::cpu {
namespace {

LogicalResult validateRepresentation(Operation *operation, Value source, Value element,
                                     const InputRequirement &requirement) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || requirement.panelSize <= 0 || requirement.alignment <= 0 ||
      !llvm::isPowerOf2_64(requirement.alignment))
    return operation->emitError("implementation has an invalid input representation requirement");
  if (!requirement.elementType || !requirement.elementType.isIntOrIndexOrFloat())
    return operation->emitError("implementation input requires an explicit scalar representation type");
  if (requirement.elementType != type.getElementType()) {
    if (element.use_empty() || !llvm::all_of(element.getUsers(), [&](Operation *user) {
          auto widen = dyn_cast<arith::ExtFOp>(user);
          return widen && widen.getType() == requirement.elementType;
        }))
      return operation->emitError("input representation must preserve the consumer's explicit floating extension");
  }
  int64_t bits = requirement.elementType.isIndex() ? 64 : requirement.elementType.getIntOrFloatBitWidth();
  if (requirement.panelAxis >= static_cast<unsigned>(type.getRank()) || requirement.alignment < (bits + 7) / 8)
    return operation->emitError("implementation input panel does not match its typed source");
  return success();
}

bool sameRepresentation(const InputRequirement &first, const InputRequirement &second) {
  return first.elementType == second.elementType && first.panelAxis == second.panelAxis &&
      first.panelSize == second.panelSize && first.alignment == second.alignment && first.reuse == second.reuse;
}

}

FailureOr<SmallVector<InputSupply>> ImplementationInputs::prepare(linalg::GenericOp operation,
    ArrayRef<InputRequirement> requirements, const Implementation &implementation) {
  SmallVector<InputSupply> supplies;
  SmallVector<unsigned> operands;
  for (auto requirement : requirements) {
    if (requirement.operand >= operation.getInputs().size() ||
        llvm::is_contained(operands, requirement.operand))
      return operation.emitError("implementation has an invalid or repeated input representation requirement"), failure();
    operands.push_back(requirement.operand);
    Value source = operation.getInputs()[requirement.operand];
    if (failed(validateRepresentation(operation, source,
        operation.getRegion().front().getArgument(requirement.operand), requirement))) return failure();
    if (requirement.reuse == InputReuse::Group) continue;
    supplies.push_back(materialize(source, requirement, consumerScope(source, operation, requirement, implementation)));
  }
  return supplies;
}

Operation *ImplementationInputs::consumerScope(Value source, linalg::GenericOp operation,
    const InputRequirement &requirement, const Implementation &implementation) {
  if (auto branch = dyn_cast<scf::IfOp>(operation->getParentOp()); branch && !branch.getElseRegion().empty()) {
    PhysicalProgramAnalysis physical(function);
    if (physical.isReadOnly(source) && DominanceInfo(function).dominates(source, branch)) {
      auto binding = operation->getAttrOfType<ImplementationAttr>("intent_cpu.implementation");
      auto configuration = function->getAttrOfType<ConfigurationAttr>("intent_cpu.configuration");
      Block *other = operation->getParentRegion() == &branch.getThenRegion() ? branch.elseBlock() : branch.thenBlock();
      // Both successors must unconditionally require the same read-only
      // snapshot. Enclosing guards, including empty-work guards, stay intact.
      for (auto candidate : other->getOps<linalg::GenericOp>()) {
        if (!isMatrixContraction(candidate) || candidate->hasAttr("intent_cpu.microtile") ||
            candidate->getAttrOfType<ImplementationAttr>("intent_cpu.implementation") != binding) continue;
        for (auto requested : implementation.inputs(candidate, configuration, binding))
          if (requested.operand < candidate.getInputs().size() &&
              candidate.getInputs()[requested.operand] == source && sameRepresentation(requirement, requested)) return branch;
      }
    }
  }
  auto loop = dyn_cast<scf::ForOp>(operation->getParentOp());
  if (!loop || loop.getNumResults()) return operation;
  auto step = getConstantIntValue(loop.getStep());
  if (!step || *step <= 0 || !DominanceInfo(function).dominates(source, loop)) return operation;
  auto effects = getEffectsRecursively(loop);
  if (!effects) return operation;
  bool known = true;
  loop->walk([&](Operation *nested) {
    if (!isa<MemoryEffectOpInterface>(nested) &&
        !nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) known = false;
  });
  if (!known) return operation;
  PhysicalProgramAnalysis physical(function);
  Value sourceRoot = physical.storageRoot(source);
  auto fresh = [](Value value) {
    return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(value.getDefiningOp());
  };
  for (auto &effect : *effects) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
    Value memory = effect.getValue();
    if (!memory || !isa<MemRefType>(memory.getType()) ||
        !isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect())) return operation;
    Value root = physical.storageRoot(memory);
    if (root == sourceRoot || (!fresh(root) && !fresh(sourceRoot))) return operation;
  }
  if (!llvm::is_contained(guardedLoops, loop.getOperation())) {
    OpBuilder b(loop);
    Value nonempty = b.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
        loop.getLowerBound(), loop.getUpperBound());
    auto guard = b.create<scf::IfOp>(loop.getLoc(), nonempty, false);
    loop->moveBefore(guard.thenBlock()->getTerminator());
    guardedLoops.push_back(loop);
  }
  return loop;
}

FailureOr<InputSupply> ImplementationInputs::prepareCaptured(linalg::GenericOp operation,
    memref::LoadOp input, const InputRequirement &requirement, Operation *scope) {
  if (input->getBlock() != &operation.getRegion().front() ||
      (scope != operation && !scope->isAncestor(operation)) || requirement.reuse != InputReuse::Consumers)
    return operation.emitError("captured input supply requires its consumer load and enclosing scope"), failure();
  Value source = input.getMemref();
  DominanceInfo dominance(function);
  PhysicalProgramAnalysis analysis(function);
  if (!dominance.dominates(source, scope) || (scope != operation && !analysis.isReadOnly(source)))
    return operation.emitError("captured input supply cannot preserve its scoped read snapshot"), failure();
  if (failed(validateRepresentation(operation, source, input.getResult(), requirement))) return failure();
  return materialize(source, requirement, scope);
}

InputSupply ImplementationInputs::materialize(Value source, const InputRequirement &requirement, Operation *scope) {
  PhysicalProgramAnalysis analysis(function);
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
        !analysis.mayReadAt(source, previous.allocation, consumer)) continue;
    storage = previous.allocation;
    if (previous.end->isBeforeInBlock(consumer)) previous.end->moveAfter(consumer);
    break;
  }
  if (!storage) {
    OpBuilder b(scope);
    Location loc = scope->getLoc();
    Value zero = index(b, loc, 0), one = index(b, loc, 1), panel = index(b, loc, requirement.panelSize);
    SmallVector<Value> sourceSizes;
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      sourceSizes.push_back(b.create<memref::DimOp>(loc, source, axis));
    Value extent = sourceSizes[requirement.panelAxis];
    Value count = b.create<arith::CeilDivSIOp>(loc, extent, panel);
    int64_t staticExtent = type.getDimSize(requirement.panelAxis);
    SmallVector<int64_t> shape{ShapedType::isDynamic(staticExtent) ? ShapedType::kDynamic
        : static_cast<int64_t>(llvm::divideCeil(static_cast<uint64_t>(staticExtent),
                                              static_cast<uint64_t>(requirement.panelSize)))};
    SmallVector<Value> sizes{count};
    for (int64_t axis = 0; axis < type.getRank(); ++axis) {
      if (axis == requirement.panelAxis) continue;
      shape.push_back(type.getDimSize(axis));
      sizes.push_back(sourceSizes[axis]);
    }
    shape.push_back(requirement.panelSize);
    sizes.push_back(panel);
    SmallVector<Value> dynamic;
    for (auto [axis, size] : llvm::enumerate(sizes))
      if (ShapedType::isDynamic(shape[axis])) dynamic.push_back(size);
    storage = b.create<memref::AllocOp>(loc, MemRefType::get(shape, requirement.elementType), dynamic);
    storage.setAlignment(requirement.alignment);

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
    auto end = b.create<memref::DeallocOp>(loc, storage);
    prepared.push_back({source, requirement, storage, end});
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
    auto storage = b.create<memref::AllocaOp>(loc,
        MemRefType::get({1, capacities[other], requirement.panelSize}, requirement.elementType));
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

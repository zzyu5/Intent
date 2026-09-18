#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::gpu {
namespace {

bool variesWithIteration(Value value, scf::ForOp loop,
                        llvm::DenseMap<Value, bool> &known) {
  if (loop.isDefinedOutsideOfLoop(value))
    return false;
  if (auto found = known.find(value); found != known.end())
    return found->second;
  bool varies = true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto nested = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    if (nested && nested != loop && argument == nested.getInductionVar())
      varies = variesWithIteration(nested.getLowerBound(), loop, known) ||
               variesWithIteration(nested.getUpperBound(), loop, known) ||
               variesWithIteration(nested.getStep(), loop, known);
  } else if (Operation *producer = value.getDefiningOp();
             producer && !producer->getNumRegions() &&
             canPredicateValueOperation(producer) &&
             !isa<LoadOp, GatherOp>(producer)) {
    varies = llvm::any_of(producer->getOperands(), [&](Value operand) {
      return variesWithIteration(operand, loop, known);
    });
  }
  known[value] = varies;
  return varies;
}

bool canVectorizeIterations(Block &block, scf::ForOp loop,
                           llvm::DenseMap<Value, bool> &varying) {
  auto data = [](Type type) {
    return isa<IntegerType, IndexType, FloatType, FragmentType>(type);
  };
  for (Operation &operation : block.without_terminator()) {
    if (!llvm::all_of(operation.getResultTypes(), data) ||
        llvm::any_of(operation.getOperandTypes(), [](Type type) {
          return isa<RecordType>(type);
        }))
      return false;
    // A range still describes one shared column domain. A lane-dependent
    // extent needs a different traversal, not a tensor-valued range bound.
    if (isa<MakeRangeOp>(operation) &&
        llvm::any_of(operation.getOperands(), [&](Value operand) {
          return variesWithIteration(operand, loop, varying);
        }))
      return false;
    // Gathering from a lane-expanded source would also need a new source-axis
    // coordinate. Keep this transform to reads from an unchanged SSA source.
    if (auto gather = dyn_cast<GatherOp>(operation))
      if (!loop.isDefinedOutsideOfLoop(gather.getSource()))
        return false;
    if (auto nested = dyn_cast<scf::ForOp>(operation)) {
      if (variesWithIteration(nested.getLowerBound(), loop, varying) ||
          variesWithIteration(nested.getUpperBound(), loop, varying) ||
          variesWithIteration(nested.getStep(), loop, varying) ||
          !canVectorizeIterations(*nested.getBody(), loop, varying))
        return false;
    } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
      if (!canVectorizeIterations(*branch.thenBlock(), loop, varying) ||
          (!branch.getElseRegion().empty() &&
           !canVectorizeIterations(*branch.elseBlock(), loop, varying)))
        return false;
    } else if (auto store = dyn_cast<StoreOp>(operation)) {
      // Compiler-created retained storage may be reused sequentially by the
      // original points. Its allocation has not been widened to lane slices.
      if (auto buffer = dyn_cast<BufferType>(store.getResource().getType());
          buffer && buffer.getWorkspace())
        return false;
      if (!data(store.getValue().getType()) ||
          !llvm::all_of(store.getCoordinates(), [&](Value value) {
            return data(value.getType());
          }))
        return false;
    } else if (operation.getNumRegions() ||
               !canPredicateValueOperation(&operation)) {
      return false;
    }
  }
  return true;
}

bool isOutsideIterationRange(Value coordinate, scf::ForOp loop) {
  if (!coordinate.getType().isIndex() ||
      !loop.isDefinedOutsideOfLoop(coordinate))
    return false;
  if (samePhysicalScalarExpression(coordinate, loop.getUpperBound()))
    return true;
  if (auto index = coordinate.getDefiningOp<arith::ConstantIndexOp>()) {
    auto lower = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto upper = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if ((lower && index.value() < lower.value()) ||
        (upper && index.value() >= upper.value()))
      return true;
  }
  auto successor = loop.getLowerBound().getDefiningOp<BinaryOp>();
  if (!successor || successor.getOperatorKind() != BinaryOperator::Add)
    return false;
  auto induction = dyn_cast<BlockArgument>(coordinate);
  auto owner = induction
                   ? dyn_cast<scf::ForOp>(induction.getOwner()->getParentOp())
                   : scf::ForOp();
  auto step = owner ? owner.getStep().getDefiningOp<arith::ConstantIndexOp>()
                   : arith::ConstantIndexOp();
  if (!owner || coordinate != owner.getInductionVar() || !step ||
      step.value() != 1 || !owner->isProperAncestor(loop))
    return false;
  // An active unit-step induction value is strictly below its signed upper
  // bound, so adding one is representable even at the last iteration.
  for (auto [base, offset] :
       {std::pair{successor.getLhs(), successor.getRhs()},
        std::pair{successor.getRhs(), successor.getLhs()}})
    if (base == coordinate)
      if (auto one = offset.getDefiningOp<arith::ConstantIndexOp>())
        if (one.value() == 1)
          return true;
  return false;
}

bool hasIndependentBufferUpdates(scf::ForOp loop, func::FuncOp kernel) {
  llvm::DenseMap<Value, unsigned> writtenAxes;
  bool independent = true;
  loop.walk([&](StoreOp store) {
    auto buffer = store.getResource().getDefiningOp<BufferOp>();
    auto type = dyn_cast<BufferType>(store.getResource().getType());
    if (!buffer || buffer->getBlock() != &kernel.front() || !type ||
        type.getWorkspace() ||
        type.getScope().getValue() != BufferScope::ProgramPrivate ||
        type.getLifetime().getValue() != BufferLifetime::Program) {
      independent = false;
      return;
    }
    std::optional<unsigned> selected;
    for (auto [axis, coordinate] :
         llvm::zip(store.getSourceAxes(), store.getCoordinates())) {
      if (coordinate == loop.getInductionVar() && !selected)
        selected = axis;
      else if (!loop.isDefinedOutsideOfLoop(coordinate))
        independent = false;
    }
    if (!selected) {
      independent = false;
      return;
    }
    auto [previous, inserted] =
        writtenAxes.try_emplace(store.getResource(), *selected);
    independent &= inserted || previous->second == *selected;
  });
  if (!independent || writtenAxes.empty())
    return false;

  // Fresh logical buffers cannot alias other resources. Each written buffer
  // uses a distinct slice for every iteration. Reads must stay in that slice
  // or outside the written range, without changing ordered arithmetic.
  loop.walk([&](LoadOp load) {
    auto written = writtenAxes.find(load.getResource());
    if (written == writtenAxes.end())
      return;
    bool independentRead = false;
    for (auto [axis, coordinate] :
         llvm::zip(load.getSourceAxes(), load.getCoordinates()))
      if (axis == written->second)
        independentRead = coordinate == loop.getInductionVar() ||
                          isOutsideIterationRange(coordinate, loop);
    independent &= independentRead;
  });
  return independent;
}

LogicalResult vectorizeIterations(func::FuncOp kernel,
                                 uint64_t &source, int64_t &dimension) {
  SmallVector<scf::ForOp> loops;
  // Lift the inner independent axis first. Its fragment becomes the suffix
  // when an enclosing independent loop is subsequently widened.
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
    llvm::DenseMap<Value, bool> varying;
    if (loop.getNumResults() || !step || step.value() != 1 ||
        !canVectorizeIterations(*loop.getBody(), loop, varying) ||
        (!loop->hasAttr(independentIterationAttr) &&
         !hasIndependentBufferUpdates(loop, kernel)))
      continue;
    uint32_t elementBitWidth = 0;
    loop.walk([&](StoreOp store) {
      Type type = store.getValue().getType();
      if (auto fragment = dyn_cast<FragmentType>(type))
        type = fragment.getElementType();
      elementBitWidth = std::max(
          elementBitWidth, type.isIndex() ? 64u : type.getIntOrFloatBitWidth());
    });
    if (!elementBitWidth)
      continue;
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    SmallVector<int64_t> candidates;
    for (int64_t width = 1; width <= capabilities.getMaxThreadsPerBlock();
         width *= 2)
      candidates.push_back(width);
    auto name = ("ITERATION_" + Twine(source)).str();
    ParameterOp width = getOrCreatePhysicalParameter(
        kernel, name, ParameterRole::OwnershipN, ParameterCategory::Pointwise,
        elementBitWidth, candidates);
    if (!width)
      return failure();
    OpBuilder builder(loop);
    // This is a fresh one-dimensional iteration domain. Load/store sourceAxes
    // continue to map its coordinates to the original resource axes.
    auto ordinal = AxisMapAttr::get(kernel.getContext(), source++, 0,
                                   dimension++, 0, false);
    width->setAttr(parameterSourceAttr,
                   PhysicalSourceAttr::get(kernel.getContext(),
                                           ordinal.getSourceId(), 0, false));
    width->setAttr(pointwiseChunkAttr, builder.getUnitAttr());
    auto extent = PhysicalExprAttr::get(
        kernel.getContext(), static_cast<uint32_t>(PhysicalExprKind::Parameter),
        0, builder.getStringAttr(name), builder.getArrayAttr({}));
    auto shape = FragmentType::get(
        kernel.getContext(), builder.getIndexType(), builder.getArrayAttr({extent}),
        builder.getArrayAttr({ordinal}), 1, /*owner=*/1);
    auto boolean = FragmentType::get(
        kernel.getContext(), builder.getI1Type(), shape.getShape(),
        shape.getAxisMaps(), shape.getValidity(), shape.getOwner());
    Location location = loop.getLoc();
    Type index = builder.getIndexType();
    Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
    Value one = builder.create<arith::ConstantIndexOp>(location, 1);
    Value nonempty = builder.create<CompareOp>(
        location, builder.getI1Type(), loop.getLowerBound(), loop.getUpperBound(),
        ComparePredicate::Lt);
    Value span = builder.create<BinaryOp>(
        location, index, loop.getUpperBound(), loop.getLowerBound(),
        BinaryOperator::Subtract);
    Value last = builder.create<BinaryOp>(location, index, span, one,
                                         BinaryOperator::Subtract);
    last = builder.create<SelectOp>(location, index, nonempty, last, zero);
    Value chunks = builder.create<BinaryOp>(
        location, index, last, width.getResult(), BinaryOperator::FloorDivide);
    Value present = builder.create<CastOp>(location, index, nonempty);
    chunks = builder.create<BinaryOp>(location, index, chunks, present,
                                     BinaryOperator::Add);
    auto blocked = builder.create<scf::ForOp>(location, zero, chunks, one);
    blocked->setAttrs(loop->getAttrs());
    builder.setInsertionPoint(blocked.getBody()->getTerminator());
    Value offset = builder.create<BinaryOp>(
        location, index, blocked.getInductionVar(), width.getResult(),
        BinaryOperator::Multiply);
    Value start = builder.create<BinaryOp>(location, index, loop.getLowerBound(),
                                          offset, BinaryOperator::Add);
    Value members = builder.create<MakeRangeOp>(
        loop.getLoc(), shape, start, width.getResult(), loop.getStep(),
        loop.getLowerBound(), loop.getUpperBound(),
        ordinal.getSourceId(), 0, false);
    // Count chunks and guard lane ordinals so a padded final lane cannot wrap
    // past a large logical upper bound and accidentally become active again.
    Value remaining = builder.create<BinaryOp>(location, index, span, offset,
                                              BinaryOperator::Subtract);
    Value base = builder.create<BroadcastOp>(location, shape, start);
    Value lanes = builder.create<BinaryOp>(location, shape, members, base,
                                          BinaryOperator::Subtract);
    Value end = builder.create<BroadcastOp>(location, shape, remaining);
    Value active = builder.create<CompareOp>(loop.getLoc(), boolean, lanes, end,
                                            ComparePredicate::Lt);
    IRMapping values;
    values.map(loop.getInductionVar(), members);
    for (Operation &operation : loop.getBody()->without_terminator())
      clonePredicatedScalarOperation(builder, &operation, values, active, shape);
    loop.erase();
  }
  return success();
}

} // namespace

LogicalResult vectorizeBufferLoops(ModuleOp module) {
  auto physical = getPhysicalKernel(module);
  if (failed(physical))
    return failure();
  func::FuncOp kernel = *physical;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  if (!llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      }))
    return success();
  auto [source, dimension] = nextPhysicalAxisIdentities(kernel);
  if (failed(vectorizeIterations(kernel, source, dimension)))
    return failure();
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::gpu

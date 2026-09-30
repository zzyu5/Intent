#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "Intent/Dialect/GPU/Transforms/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Transforms/Predication.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace intent::gpu {
namespace {

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
    } else if (isa<ReshapeOp, TransposeOp>(operation)) {
      continue;
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      if (!llvm::all_of(reduce.getCombine().front().without_terminator(),
                        [](Operation &nested) {
                          return canPredicateValueOperation(&nested) &&
                                 !isa<LoadOp, GatherOp>(nested);
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

bool hasIndependentUpdates(scf::ForOp loop, func::FuncOp kernel,
                           SmallVectorImpl<Value> &guardedViews,
                           SmallVectorImpl<std::pair<Value, Value>> &disjointViews,
                           llvm::DenseMap<Value, bool> &varying) {
  llvm::DenseMap<Value, unsigned> writtenAxes;
  bool independent = true;
  loop.walk([&](StoreOp store) {
    auto buffer = store.getResource().getDefiningOp<BufferOp>();
    auto type = dyn_cast<BufferType>(store.getResource().getType());
    auto view = dyn_cast<ViewType>(store.getResource().getType());
    bool privateBuffer = buffer && buffer->getBlock() == &kernel.front() &&
        type && !type.getWorkspace() &&
        type.getScope().getValue() == BufferScope::ProgramPrivate &&
        type.getLifetime().getValue() == BufferLifetime::Program;
    if (!privateBuffer && (!view || !view.getLayout().getHasStrides() ||
                          view.getLayout().getStrides().size() != view.getRank())) {
      independent = false;
      return;
    }
    if (view && !llvm::is_contained(guardedViews, store.getResource()))
      guardedViews.push_back(store.getResource());
    std::optional<unsigned> selected;
    for (auto [axis, coordinate] :
         llvm::zip(store.getSourceAxes(), store.getCoordinates())) {
      if (coordinate == loop.getInductionVar() && !selected)
        selected = axis;
      else if (variesWithIteration(coordinate, loop, varying))
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

  // Distinct external resources may alias even when each layout is injective.
  // Guard every write/read and write/write pair using the actual launch views.
  SmallVector<Value> accessedViews(guardedViews.begin(), guardedViews.end());
  loop.walk([&](LoadOp load) {
    if (isa<ViewType>(load.getResource().getType()) &&
        !llvm::is_contained(accessedViews, load.getResource()))
      accessedViews.push_back(load.getResource());
  });
  for (Value view : accessedViews) {
    auto argument = dyn_cast<BlockArgument>(view);
    if (!argument || argument.getOwner() != &kernel.front())
      return false;
  }
  for (Value written : guardedViews)
    for (Value other : accessedViews) {
      if (written == other)
        continue;
      auto pair = std::pair{written, other};
      if (cast<BlockArgument>(written).getArgNumber() >
          cast<BlockArgument>(other).getArgNumber())
        std::swap(pair.first, pair.second);
      if (!llvm::is_contained(disjointViews, pair))
        disjointViews.push_back(pair);
    }

  // Fresh buffers are disjoint, and external views need the layout guard below.
  // Each iteration owns one slice. Reads must stay in that slice or outside
  // the written range, without changing ordered arithmetic.
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
                                 uint64_t &source, int64_t &dimension,
                                 bool singleInstance) {
  SmallVector<scf::ForOp> loops;
  // Lift the inner independent axis first. Its fragment becomes the suffix
  // when an enclosing independent loop is subsequently widened.
  kernel.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    // Declared independent iterations remain independent within each instance.
    // Inferred mutable-buffer updates retain the single-instance restriction.
    if (!singleInstance && !loop->hasAttr(independentIterationAttr))
      continue;
    auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
    llvm::DenseMap<Value, bool> varying;
    SmallVector<Value> guardedViews;
    SmallVector<std::pair<Value, Value>> disjointViews;
    if (loop.getNumResults() || !step || step.value() != 1 ||
        !canVectorizeIterations(*loop.getBody(), loop, varying) ||
        (!loop->hasAttr(independentIterationAttr) &&
         !hasIndependentUpdates(loop, kernel, guardedViews, disjointViews, varying)))
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
    if (!guardedViews.empty()) {
      SmallVector<Value> injective;
      for (Value view : guardedViews) {
        auto condition = materializeNonOverlappingView(kernel, view);
        if (failed(condition))
          break;
        injective.push_back(*condition);
      }
      if (injective.size() != guardedViews.size())
        continue;
      OpBuilder builder(loop);
      Value safe = injective.front();
      auto require = [&](Value condition) {
        safe = builder.create<BinaryOp>(loop.getLoc(), builder.getI1Type(),
                                        safe, condition, BinaryOperator::LogicalAnd);
      };
      for (Value condition : ArrayRef<Value>(injective).drop_front())
        require(condition);
      for (auto [lhs, rhs] : disjointViews) {
        Value overlap;
        for (ViewOverlapOp fact : kernel.getOps<ViewOverlapOp>())
          if ((fact.getLhs() == lhs && fact.getRhs() == rhs) ||
              (fact.getLhs() == rhs && fact.getRhs() == lhs)) {
            overlap = fact.getResult();
            break;
          }
        if (!overlap) {
          OpBuilder entry(&kernel.front(), kernel.front().begin());
          overlap = entry.create<ViewOverlapOp>(loop.getLoc(), entry.getI1Type(),
                                               lhs, rhs);
        }
        Value zero = builder.create<arith::ConstantIntOp>(loop.getLoc(), 0, 1);
        require(builder.create<CompareOp>(loop.getLoc(), builder.getI1Type(),
                                          overlap, zero, ComparePredicate::Eq));
      }
      auto branch = builder.create<scf::IfOp>(loop.getLoc(), safe, true);
      builder.setInsertionPointToStart(branch.thenBlock());
      auto selected = cast<scf::ForOp>(builder.clone(*loop));
      loop->moveBefore(branch.elseBlock()->getTerminator());
      loop = selected;
    }
    auto capabilities = kernel->getAttrOfType<CapabilitiesAttr>(capabilitiesAttr);
    int64_t maximumWidth = capabilities.getMaxThreadsPerBlock();
    auto lower = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto upper = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if (lower && upper) {
      __int128 length = static_cast<__int128>(upper.value()) - lower.value();
      int64_t covered = 1;
      while (covered < length && covered <= maximumWidth / 2)
        covered *= 2;
      maximumWidth = covered;
    }
    SmallVector<int64_t> candidates;
    for (int64_t width = 1; width <= maximumWidth;
         width *= 2)
      candidates.push_back(width);
    bool completeChunks = lower && upper && upper.value() > lower.value() &&
        llvm::all_of(candidates, [&](int64_t width) {
          return (static_cast<__int128>(upper.value()) - lower.value()) % width == 0;
        });
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
    Value active;
    if (completeChunks) {
      active = builder.create<arith::ConstantIntOp>(location, 1, 1);
    } else {
      Value remaining = builder.create<BinaryOp>(location, index, span, offset,
                                                BinaryOperator::Subtract);
      Value base = builder.create<BroadcastOp>(location, shape, start);
      Value lanes = builder.create<BinaryOp>(location, shape, members, base,
                                            BinaryOperator::Subtract);
      Value end = builder.create<BroadcastOp>(location, shape, remaining);
      active = builder.create<CompareOp>(loop.getLoc(), boolean, lanes, end,
                                        ComparePredicate::Lt);
    }
    IRMapping values;
    values.map(loop.getInductionVar(), members);
    for (Operation &operation : loop.getBody()->without_terminator())
      clonePredicatedScalarOperation(builder, &operation, values, active, shape,
                                     /*nonemptyIterations=*/true);
    loop.erase();
  }
  return success();
}

void assignIterationRoles(func::FuncOp kernel) {
  llvm::DenseMap<Operation *, unsigned> roles;
  kernel.walk([&](StoreOp store) {
    auto value = dyn_cast<FragmentType>(store.getValue().getType());
    if (!value || value.getShape().size() != 2)
      return;
    SmallVector<std::pair<int64_t, ParameterOp>, 2> axes;
    for (auto [coordinate, resourceAxis] :
         llvm::zip(store.getCoordinates(), store.getSourceAxes())) {
      auto type = dyn_cast<FragmentType>(coordinate.getType());
      if (!type)
        continue;
      if (type.getShape().size() != 1)
        return;
      auto extent = cast<PhysicalExprAttr>(type.getShape()[0]);
      if (extent.getKind() != static_cast<uint32_t>(PhysicalExprKind::Parameter))
        return;
      auto parameter = queryParameterBySymbol(kernel, extent.getSymbol());
      if (failed(parameter) || !(*parameter)->hasAttr(pointwiseChunkAttr) ||
          parameter->getParameter().getCategory() !=
              static_cast<uint32_t>(ParameterCategory::Pointwise) ||
          parameter->getParameter().getCandidates().size() < 2)
        return;
      auto source = sourceAxisIdentity(cast<AxisMapAttr>(type.getAxisMaps()[0]));
      auto binding = queryParameterBinding(*parameter);
      auto projection = queryFragmentAxis(value, source);
      if (!binding.isExact() || !binding.source || !(*binding.source == source) ||
          !projection.isExact() ||
          value.getShape()[projection.fragmentAxis] != extent)
        return;
      axes.emplace_back(resourceAxis, *parameter);
    }
    if (axes.size() != 2 || axes[0].second == axes[1].second ||
        axes[0].first == axes[1].first)
      return;
    if (axes[0].first > axes[1].first)
      std::swap(axes[0], axes[1]);
    roles[axes[0].second] |= 1;
    roles[axes[1].second] |= 2;
  });
  // Use the resource axis order, not the fragment prefix order introduced by
  // nested-loop lifting. Independent one-dimensional stages keep their roles.
  for (auto [operation, mask] : roles) {
    if (mask == 3)
      continue;
    auto parameter = cast<ParameterOp>(operation);
    auto schema = parameter.getParameter();
    auto role = mask == 1 ? ParameterRole::OwnershipM : ParameterRole::OwnershipN;
    parameter->setAttr("parameter", ParameterAttr::get(
        kernel.getContext(), schema.getName(), static_cast<uint32_t>(role),
        schema.getCategory(), schema.getElementBitWidth(), schema.getCandidates()));
  }
}

} // namespace

LogicalResult vectorizeBufferLoops(ModuleOp module) {
  auto physical = getPhysicalKernel(module);
  if (failed(physical))
    return failure();
  func::FuncOp kernel = *physical;
  auto space = kernel->getAttrOfType<ArrayAttr>(programSpaceAttr);
  bool singleInstance = llvm::all_of(space, [](Attribute attribute) {
        auto extent = cast<PhysicalExprAttr>(attribute);
        return extent.getKind() == static_cast<uint32_t>(PhysicalExprKind::Constant) &&
               extent.getValue() == 1;
      });
  auto [source, dimension] = nextPhysicalAxisIdentities(kernel);
  if (failed(vectorizeIterations(kernel, source, dimension, singleInstance)))
    return failure();
  assignIterationRoles(kernel);
  eraseDeadPhysicalValues(kernel);
  return success();
}

} // namespace intent::gpu

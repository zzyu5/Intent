#include "Legalization.h"
#include "llvm/ADT/DenseSet.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/ResourceAlias.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/UniformValues.h"
#include "Intent/Dialect/GPU/Analysis/ValueSchema.h"
#include "Intent/Dialect/GPU/Transforms/Value/ValueMaterialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

using namespace mlir;

namespace intent::triton::detail {

using ViewAccessModes = llvm::DenseMap<Value, unsigned>;

ViewAccessModes readViewAccessModes(Operation *owner) {
  ViewAccessModes modes;
  owner->walk([&](gpu::AccessOpInterface access) {
    Value resource = access.getAccessResource();
    if (!isa<gpu::ViewType>(resource.getType())) return;
    // This transformation realizes ordered ordinary loads/stores. An access
    // schema does not authorize treating an atomic or pure gather as either.
    if (access.getAccessKind() == gpu::AccessKind::Load) modes[resource] |= 1;
    else if (access.getAccessKind() == gpu::AccessKind::Store) modes[resource] |= 2;
  });
  return modes;
}

bool hasOrderedViewDependencies(func::FuncOp kernel) {
  gpu::ResourceAliasAnalysis aliases;
  auto modes = readViewAccessModes(kernel);
  return llvm::any_of(modes, [&](auto read) {
    return (read.second & 1) && llvm::any_of(modes, [&](auto write) {
      return (write.second & 2) &&
             !aliases.alias(read.first, write.first).isNo();
    });
  });
}

LogicalResult legalizeOrderedViewDependencies(func::FuncOp kernel) {
  if (!hasOrderedViewDependencies(kernel))
    return success();
  gpu::ResourceAliasAnalysis aliases;
  using Accesses = ViewAccessModes;
  auto accesses = readViewAccessModes;
  auto conflicts = [&](const Accesses &pending, const Accesses &current) {
    for (auto [resource, mode] : pending)
      for (auto [other, otherMode] : current)
        if (((mode | otherMode) & 2) &&
            !aliases.alias(resource, other).isNo())
          return true;
    return false;
  };
  llvm::DenseMap<Value, Value> nonOverlappingViews;
  auto disjointLoopAccesses = [&](Block &body, const Accesses &bodyAccesses) {
    llvm::DenseMap<Value, Value> conditions;
    auto loop = dyn_cast<scf::ForOp>(body.getParentOp());
    if (!loop || !loop.getUpperBound().getType().isIndex())
      return conditions;
    auto stepBounds = gpu::queryPositiveExtentBounds(
        gpu::queryLaunchExpression(loop.getStep()), kernel);
    if (!stepBounds)
      return conditions;
    int64_t maximumStep = stepBounds->second;
    bool modeledEffects = true;
    body.walk([&](Operation *operation) {
      if (!isa<gpu::LoadOp, gpu::StoreOp, scf::ForOp, scf::IfOp,
               scf::WhileOp>(operation) && !isMemoryEffectFree(operation))
        modeledEffects = false;
    });
    if (!modeledEffects)
      return conditions;
    llvm::DenseMap<Value, bool> varying;
    Value safeRange;
    for (auto [resource, mode] : bodyAccesses) {
      auto argument = dyn_cast<BlockArgument>(resource);
      auto view = dyn_cast<gpu::ViewType>(resource.getType());
      if (!(mode & 2) || !argument || argument.getOwner() != &kernel.front() || !view)
        continue;
      gpu::StoreOp selected;
      unsigned stores = 0;
      body.walk([&](gpu::StoreOp store) {
        if (store.getResource() == resource) {
          selected = store;
          ++stores;
        }
      });
      if (stores != 1 || selected->getBlock() != &body)
        continue;
      auto chunkAxis = [&](gpu::AccessOpInterface access) -> std::optional<int64_t> {
        std::optional<int64_t> axis;
        for (auto [coordinate, sourceAxis] :
             llvm::zip(access.getAccessCoordinates(), access.getAccessSourceAxes())) {
          Value root = coordinate;
          while (true) {
            if (auto broadcast = root.getDefiningOp<gpu::BroadcastOp>())
              root = broadcast.getValue();
            else if (auto reshape = root.getDefiningOp<gpu::ReshapeOp>())
              root = reshape.getValue();
            else
              break;
          }
          auto range = root.getDefiningOp<gpu::MakeRangeOp>();
          if (root == loop.getInductionVar() ||
              (range && range.getStart() == loop.getInductionVar() &&
               gpu::isUnitStepRange(range) &&
               gpu::samePhysicalScalarExpression(range.getExtent(),
                                                  loop.getStep()))) {
            if (axis)
              return std::nullopt;
            axis = sourceAxis;
          } else if (gpu::variesWithIteration(root, loop, varying)) {
            return std::nullopt;
          }
        }
        return axis;
      };
      auto storeAxis = chunkAxis(cast<gpu::AccessOpInterface>(selected.getOperation()));
      if (!storeAxis)
        continue;
      bool disjoint = true;
      body.walk([&](gpu::LoadOp load) {
        auto access = cast<gpu::AccessOpInterface>(load.getOperation());
        if (access.getAccessResource() == resource && chunkAxis(access) != storeAxis)
          disjoint = false;
      });
      if (!disjoint)
        continue;
      Value &layout = nonOverlappingViews[resource];
      if (!layout) {
        auto proof = gpu::materializeNonOverlappingView(kernel, resource);
        if (failed(proof))
          continue;
        layout = *proof;
      }
      OpBuilder builder(loop);
      if (!safeRange) {
        // Include the final padded chunk and the terminating IV update in
        // the no-wrap proof; access masks can only narrow each chunk.
        Value limit = builder.create<arith::ConstantIndexOp>(
            loop.getLoc(), std::numeric_limits<int64_t>::max() - maximumStep);
        safeRange = builder.create<gpu::CompareOp>(
            loop.getLoc(), builder.getI1Type(), loop.getUpperBound(), limit,
            ComparePredicate::Le);
      }
      conditions[resource] = builder.create<gpu::BinaryOp>(
          loop.getLoc(), builder.getI1Type(), layout, safeRange,
          BinaryOperator::LogicalAnd);
    }
    return conditions;
  };
  const llvm::DenseMap<Value, Value> noDisjointIterations;
  std::map<std::pair<unsigned, unsigned>, Value> overlapFacts;
  auto insertBarrier = [&](OpBuilder &builder, Location location,
                           Accesses &pending, const Accesses &current,
                           const llvm::DenseMap<Value, Value> &disjointIterations) {
    SmallVector<std::pair<unsigned, unsigned>> pairs;
    Value condition;
    auto externalArgument = [&](Value value) -> BlockArgument {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument || argument.getOwner() != &kernel.front())
        return {};
      auto binding = gpu::getArgumentBinding(argument);
      return binding && binding.getKind() == gpu::ArgumentKind::Public &&
                     isa<gpu::ViewType>(argument.getType())
                 ? argument : BlockArgument();
    };
    for (auto [resource, mode] : pending)
      for (auto [other, otherMode] : current) {
        if (!((mode | otherMode) & 2) || aliases.alias(resource, other).isNo())
          continue;
        if (resource == other)
          if (auto proof = disjointIterations.find(resource);
              proof != disjointIterations.end()) {
            Value zero = builder.create<arith::ConstantIntOp>(location, 0, 1);
            Value needed = builder.create<gpu::CompareOp>(
                location, builder.getI1Type(), proof->second, zero,
                ComparePredicate::Eq);
            condition = condition
                            ? Value(builder.create<gpu::BinaryOp>(
                                  location, builder.getI1Type(), condition,
                                  needed, BinaryOperator::LogicalOr))
                            : needed;
            continue;
          }
        auto left = externalArgument(resource);
        auto right = externalArgument(other);
        if (resource == other || !left || !right) {
          builder.create<CtaBarrierOp>(location);
          pending.clear();
          return;
        }
        unsigned lhs = left.getArgNumber(), rhs = right.getArgNumber();
        pairs.emplace_back(std::min(lhs, rhs), std::max(lhs, rhs));
      }
    llvm::sort(pairs);
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    for (auto pair : pairs) {
      Value &overlap = overlapFacts[pair];
      if (!overlap) {
        OpBuilder entry(&kernel.front(), kernel.front().begin());
        overlap = entry.create<gpu::ViewOverlapOp>(
            location, entry.getI1Type(), kernel.getArgument(pair.first),
            kernel.getArgument(pair.second));
      }
      condition = condition
                      ? Value(builder.create<gpu::BinaryOp>(
                            location, builder.getI1Type(), condition, overlap,
                            BinaryOperator::LogicalOr))
                      : overlap;
    }
    auto guard = builder.create<scf::IfOp>(location, condition, false);
    OpBuilder nested = guard.getThenBodyBuilder();
    nested.create<CtaBarrierOp>(location);
    // A false guard performs no synchronization: keep all outstanding accesses
    // so a later operation cannot lose an unrelated dependency.
  };
  llvm::DenseSet<Value> visiting;
  std::function<bool(Value)> uniform = [&](Value value) {
    if (isa<gpu::FragmentType, gpu::RecordType>(value.getType()))
      return false;
    if (!visiting.insert(value).second)
      return true;
    bool result = false;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parent)) {
        result = true;
      } else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        result = uniform(loop.getLowerBound()) && uniform(loop.getUpperBound()) &&
                 uniform(loop.getStep());
        if (result && argument.getArgNumber()) {
          unsigned index = argument.getArgNumber() - 1;
          result = uniform(loop.getInitArgs()[index]) &&
                   uniform(loop.getBody()->getTerminator()->getOperand(index));
        }
      } else if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
        auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        unsigned index = argument.getArgNumber();
        if (argument.getOwner() == &loop.getBefore().front())
          result = uniform(loop.getInits()[index]) &&
                   uniform(loop.getAfter().front().getTerminator()->getOperand(index));
        else
          result = uniform(condition.getCondition()) && uniform(condition.getArgs()[index]);
      }
    } else if (Operation *producer = value.getDefiningOp()) {
      if (auto loop = dyn_cast<scf::ForOp>(producer)) {
        auto index = cast<OpResult>(value).getResultNumber();
        result = uniform(loop.getRegionIterArgs()[index]);
      } else if (auto loop = dyn_cast<scf::WhileOp>(producer)) {
        auto condition = cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        unsigned index = cast<OpResult>(value).getResultNumber();
        result = uniform(condition.getCondition()) && uniform(condition.getArgs()[index]);
      } else if (auto branch = dyn_cast<scf::IfOp>(producer)) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        result = uniform(branch.getCondition()) &&
                 uniform(branch.thenBlock()->getTerminator()->getOperand(index)) &&
                 uniform(branch.elseBlock()->getTerminator()->getOperand(index));
      } else if (isa<gpu::ReduceOp, gpu::GatherOp, gpu::DimOp, gpu::ParameterOp,
              gpu::PhysicalExprOp, gpu::ProgramIdOp, gpu::ViewOverlapOp,
              arith::ConstantOp>(producer)) {
        result = true;
      } else if (isa<gpu::LoadOp, gpu::UnaryOp, gpu::BinaryOp, gpu::CompareOp,
                     gpu::SelectOp, gpu::CastOp, gpu::BitcastOp,
                     gpu::WorksetCoordinateOp, gpu::DelinearizeOp>(producer)) {
        result = llvm::all_of(producer->getOperands(), uniform);
      }
    }
    visiting.erase(value);
    return result;
  };
  std::function<LogicalResult(Block &, bool, bool, Accesses &)> synchronize =
      [&](Block &block, bool loopBody, bool uniformControl,
          Accesses &pending) -> LogicalResult {
    Accesses bodyAccesses;
    for (Operation &operation : block)
      for (auto [resource, mode] : accesses(&operation))
        bodyAccesses[resource] |= mode;
    auto disjointIterations = loopBody ? disjointLoopAccesses(block, bodyAccesses)
                                      : llvm::DenseMap<Value, Value>();
    for (Operation &operation : llvm::make_early_inc_range(block.without_terminator())) {
      if (isa<CtaBarrierOp>(operation)) {
        pending.clear();
        continue;
      }
      Accesses current = accesses(&operation);
      if (conflicts(pending, current)) {
        if (!uniformControl)
          return operation.emitOpError("ordered view dependency requires uniform CTA control before synchronization");
        OpBuilder builder(&operation);
        insertBarrier(builder, operation.getLoc(), pending, current,
                      noDisjointIterations);
      }
      if (operation.getNumRegions()) {
        bool nestedUniform = uniformControl;
        if (auto branch = dyn_cast<scf::IfOp>(operation))
          nestedUniform &= uniform(branch.getCondition());
        else if (auto loop = dyn_cast<scf::ForOp>(operation))
          nestedUniform &= uniform(loop.getInductionVar());
        else if (auto loop = dyn_cast<scf::WhileOp>(operation))
          nestedUniform &= uniform(
              cast<scf::ConditionOp>(loop.getBefore().front().getTerminator()).getCondition());
        for (Region &region : operation.getRegions())
          for (Block &nested : region) {
            Accesses outstanding;
            if (failed(synchronize(nested, isa<scf::ForOp, scf::WhileOp>(operation),
                                   nestedUniform, outstanding)))
              return failure();
            for (auto [resource, mode] : outstanding)
              pending[resource] |= mode;
          }
      } else {
        for (auto [resource, mode] : current)
          pending[resource] |= mode;
      }
    }
    if (loopBody && conflicts(pending, bodyAccesses)) {
      if (!uniformControl)
        return block.getTerminator()->emitOpError("loop-carried view dependency requires uniform CTA control before synchronization");
      OpBuilder builder(block.getTerminator());
      insertBarrier(builder, block.getTerminator()->getLoc(), pending,
                    bodyAccesses, disjointIterations);
    }
    return success();
  };
  Accesses pending;
  return synchronize(kernel.front(), false, true, pending);
}

// Query all loop-carried roots in one address-graph traversal. Nested loop
// results keep their init/yield dependency; IV variation alone is pipeline legal.
bool addressDependsOnLoopCarry(ValueRange operands, scf::ForOp loop) {
  SmallVector<Value> pending(operands);
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (llvm::is_contained(loop.getRegionIterArgs(), value))
      return true;
    if (!visited.insert(value).second)
      continue;
    Operation *definition = value.getDefiningOp();
    if (!definition || !loop->isProperAncestor(definition))
      continue;
    if (auto nested = dyn_cast<scf::ForOp>(definition)) {
      auto result = dyn_cast<OpResult>(value);
      auto yield = dyn_cast<scf::YieldOp>(nested.getBody()->getTerminator());
      if (result && yield && result.getResultNumber() < nested.getInitArgs().size()) {
        unsigned index = result.getResultNumber();
        pending.push_back(nested.getInitArgs()[index]);
        pending.push_back(yield.getOperand(index));
      }
    }
    llvm::append_range(pending, definition->getOperands());
  }
  return false;
}

SmallVector<scf::ForOp> findLoadPipelineLoops(func::FuncOp kernel) {
  SmallVector<scf::ForOp> loops;
  kernel.walk([&](scf::ForOp loop) {
    bool vectorLoad = false;
    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (isa<gpu::ContractOp, gpu::ScaledContractOp, gpu::SparseContractOp>(
              operation) ||
          (operation.getNumRegions() &&
           !isa<gpu::ReduceOp, gpu::ScanOp>(operation)))
        return;
      if (auto access = dyn_cast<gpu::AccessOpInterface>(&operation);
          access && (access.getAccessKind() == gpu::AccessKind::Load ||
                     access.getAccessKind() == gpu::AccessKind::Store)) {
        if (!isa<gpu::ViewType>(access.getAccessResource().getType())) return;
        SmallVector<Value> addressing{access.getAccessResource()};
        llvm::append_range(addressing, access.getAccessCoordinates());
        if (access.getAccessValidity()) addressing.push_back(access.getAccessValidity());
        if (access.getAccessFill()) addressing.push_back(access.getAccessFill());
        if (addressDependsOnLoopCarry(addressing, loop))
          return;
        vectorLoad |= access.getAccessKind() == gpu::AccessKind::Load &&
                      isa<gpu::FragmentType>(access.getAccessValueType());
      } else if (!isMemoryEffectFree(&operation)) {
        return;
      }
    }
    if (vectorLoad)
      loops.push_back(loop);
  });
  return loops;
}

void selectOrderedLoadUnrolling(func::FuncOp kernel) {
  UniformValueAnalysis constants(gpu::describeUniformValue);
  auto integer = [&](Value value) {
    return dyn_cast_or_null<IntegerAttr>(constants.evaluate(value));
  };
  kernel.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(loopStagesAttr) || loop.getNumResults() != 1 ||
        !isa<FloatType>(gpu::uniformElementType(loop.getResult(0).getType())))
      return;
    auto lower = integer(loop.getLowerBound());
    auto step = integer(loop.getStep());
    if (!lower || !lower.getValue().isZero() || !step ||
        !step.getValue().isOne() ||
        !gpu::queryNonNegativeIndexUpperBound(loop.getUpperBound()))
      return;
    int64_t factor = 4;
    if (auto upper = integer(loop.getUpperBound())) {
      if (upper.getInt() <= 1)
        return;
      factor = std::min<int64_t>(factor, upper.getInt());
    }

    unsigned loads = 0;
    bool product = false;
    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (operation.getNumRegions() ||
          isa<gpu::ContractOp, gpu::ScaledContractOp, gpu::SparseContractOp,
              gpu::HistogramOp>(operation))
        return;
      if (auto access = dyn_cast<gpu::AccessOpInterface>(&operation);
          access && access.getAccessKind() == gpu::AccessKind::Load) {
        SmallVector<Value> dependencies{access.getAccessResource()};
        llvm::append_range(dependencies, access.getAccessCoordinates());
        if (access.getAccessValidity()) dependencies.push_back(access.getAccessValidity());
        if (access.getAccessFill()) dependencies.push_back(access.getAccessFill());
        if (addressDependsOnLoopCarry(dependencies, loop))
          return;
        ++loads;
      } else if (!isMemoryEffectFree(&operation)) {
        return;
      }
      if (auto binary = dyn_cast<gpu::BinaryOp>(operation))
        product |= binary.getOperatorKind() == BinaryOperator::Multiply &&
                   isa<FloatType>(
                       gpu::uniformElementType(binary.getResult().getType()));
    }
    if (loads == 0 || loads > 2 || !product)
      return;
    // Native unrolling preserves the accumulator chain and handles the tail;
    // independent reads from later iterations can overlap the current update.
    loop->setAttr(loopUnrollFactorAttr,
                  IntegerAttr::get(IntegerType::get(kernel.getContext(), 32),
                                   factor));
  });
}

} // namespace intent::triton::detail

#include "Intent/Target/CuTile/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace intent::cutile {
namespace {

bool isSpecializationExpression(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return false;
  if (isa<arith::ConstantOp, gpu::ParameterOp, gpu::PhysicalExprOp, gpu::DimOp>(
          operation))
    return true;
  if (!isa<gpu::BinaryOp, gpu::CompareOp, gpu::CastOp>(operation))
    return false;
  return llvm::all_of(operation->getOperands(), isSpecializationExpression);
}

bool positiveExtent(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto integer = dyn_cast<IntegerAttr>(constant.getValue());
    return integer && integer.getInt() > 0;
  }
  auto parameter = value.getDefiningOp<gpu::ParameterOp>();
  return parameter && !parameter.getParameter().getCandidates().empty() &&
         llvm::all_of(parameter.getParameter().getCandidates().asArrayRef(),
                      [](int64_t candidate) { return candidate > 0; });
}

bool safeScalarGuardOperation(Operation *operation) {
  if (isa<arith::ConstantOp, gpu::CompareOp, gpu::DimOp,
          gpu::ParameterOp>(operation))
    return true;
  if (auto cast = dyn_cast<gpu::CastOp>(operation))
    return cast.getValue().getType().isIntOrIndex() &&
           cast.getResult().getType().isIntOrIndex();
  auto binary = dyn_cast<gpu::BinaryOp>(operation);
  if (!binary || !binary.getResult().getType().isIntOrIndex())
    return false;
  switch (binary.getOperatorKind()) {
  case BinaryOperator::Add:
  case BinaryOperator::Subtract:
  case BinaryOperator::Multiply:
  case BinaryOperator::Minimum:
  case BinaryOperator::Maximum:
  case BinaryOperator::MinimumNum:
  case BinaryOperator::MaximumNum:
  case BinaryOperator::LogicalAnd:
  case BinaryOperator::LogicalOr:
  case BinaryOperator::BitwiseAnd:
  case BinaryOperator::BitwiseOr:
    return true;
  case BinaryOperator::FloorDivide:
  case BinaryOperator::Remainder:
    return positiveExtent(binary.getRhs());
  default:
    return false;
  }
}

bool collectInvariantGuard(Value value, scf::ForOp loop,
                           llvm::SmallPtrSetImpl<Operation *> &visited,
                           SmallVectorImpl<Operation *> &operations) {
  Operation *definition = value.getDefiningOp();
  Operation *owner = definition
                         ? definition
                         : cast<BlockArgument>(value).getOwner()->getParentOp();
  if (owner != loop.getOperation() && !loop->isAncestor(owner))
    return true;
  // Existing external SSA reads can be reused, but never move a read across
  // iterations or speculate a potentially invalid scalar operation.
  if (!definition || !safeScalarGuardOperation(definition))
    return false;
  if (!visited.insert(definition).second)
    return true;
  for (Value operand : definition->getOperands())
    if (!collectInvariantGuard(operand, loop, visited, operations))
      return false;
  operations.push_back(definition);
  return true;
}

void unswitchNativeAccessGuard(scf::ForOp loop) {
  scf::IfOp selected;
  SmallVector<Operation *> guardOperations;
  loop.walk([&](scf::IfOp conditional) {
    if (selected || conditional->getParentOfType<scf::ForOp>() != loop ||
        conditional.getNumResults() == 0 ||
        isSpecializationExpression(conditional.getCondition()))
      return;
    bool tileLoad = false;
    bool gatherLoad = false;
    conditional.getThenRegion().walk([&](TileLoadOp) { tileLoad = true; });
    conditional.getElseRegion().walk([&](GatherLoadOp) { gatherLoad = true; });
    if (!tileLoad || !gatherLoad)
      return;
    llvm::SmallPtrSet<Operation *, 16> visited;
    SmallVector<Operation *> operations;
    if (!collectInvariantGuard(conditional.getCondition(), loop, visited,
                               operations))
      return;
    selected = conditional;
    guardOperations = std::move(operations);
  });
  if (!selected)
    return;

  // One program-uniform access decision, hence at most two loop versions.
  // Specialization-only choices stay for the provider compiler to fold.
  OpBuilder builder(loop);
  IRMapping guardMapping;
  for (Operation *operation : guardOperations)
    builder.clone(*operation, guardMapping);
  Value condition = guardMapping.lookupOrDefault(selected.getCondition());
  auto version = builder.create<scf::IfOp>(
      loop.getLoc(), loop.getResultTypes(), condition,
      /*withElseRegion=*/true);
  for (bool takeThen : {true, false}) {
    Region &region =
        takeThen ? version.getThenRegion() : version.getElseRegion();
    Block &block = region.front();
    if (!block.empty() && isa<scf::YieldOp>(block.back()))
      block.back().erase();
    OpBuilder nested(&block, block.end());
    IRMapping mapping;
    auto copied = cast<scf::ForOp>(nested.clone(*loop.getOperation(), mapping));
    auto choice =
        mapping.lookup(selected.getResult(0)).getDefiningOp<scf::IfOp>();
    Block &chosen =
        (takeThen ? choice.getThenRegion() : choice.getElseRegion()).front();
    auto yield = cast<scf::YieldOp>(chosen.getTerminator());
    for (Operation &operation :
         llvm::make_early_inc_range(chosen.without_terminator()))
      operation.moveBefore(choice);
    choice.getResults().replaceAllUsesWith(yield.getResults());
    choice.erase();
    nested.create<scf::YieldOp>(loop.getLoc(), copied.getResults());
  }
  loop.getResults().replaceAllUsesWith(version.getResults());
  loop.erase();
}

} // namespace

LogicalResult refineMMALoops(ModuleOp module) {
  auto physicalKernel = gpu::getPhysicalKernel(module);
  if (failed(physicalKernel))
    return failure();
  SmallVector<scf::ForOp> loops;
  physicalKernel->walk<WalkOrder::PostOrder>(
      [&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
      auto original = dyn_cast<gpu::FragmentType>(argument.getType());
      if (!original || original.getShape().size() != 3 || !argument.hasOneUse())
        continue;
      auto unit = dyn_cast<gpu::PhysicalExprAttr>(original.getShape()[0]);
      if (!unit ||
          unit.getKind() !=
              static_cast<uint32_t>(gpu::PhysicalExprKind::Constant) ||
          unit.getValue() != 1)
        continue;
      auto incoming = dyn_cast<gpu::ReshapeOp>(*argument.getUsers().begin());
      auto outgoing = yield.getOperand(index).getDefiningOp<gpu::ReshapeOp>();
      if (!incoming || !outgoing || incoming->getBlock() != loop.getBody() ||
          outgoing->getBlock() != loop.getBody() ||
          !outgoing.getResult().hasOneUse())
        continue;
      auto matrix = dyn_cast<gpu::FragmentType>(incoming.getResult().getType());
      auto mma = outgoing.getValue().getDefiningOp<MMAOp>();
      if (!matrix || matrix.getShape().size() != 2 || !mma ||
          mma.getAccumulator() != incoming.getResult() ||
          mma.getResult().getType() != matrix ||
          matrix.getShape().getValue() !=
              original.getShape().getValue().drop_front())
        continue;
      // Project only at the loop boundaries; the native MMA carries a matrix.
      OpBuilder before(loop);
      auto init = before.create<gpu::ReshapeOp>(
          loop.getLoc(), matrix, loop.getInitArgs()[index],
          incoming.getReassociation());
      loop.getInitArgsMutable()[index].assign(init.getResult());
      argument.setType(matrix);
      incoming.getResult().replaceAllUsesWith(argument);
      incoming.erase();
      yield->setOperand(index, outgoing.getValue());
      auto restoreRelation = outgoing.getReassociation();
      outgoing.erase();
      Value result = loop.getResult(index);
      result.setType(matrix);
      OpBuilder after(loop);
      after.setInsertionPointAfter(loop);
      auto restored = after.create<gpu::ReshapeOp>(
          loop.getLoc(), original, result, restoreRelation);
      result.replaceAllUsesExcept(restored.getResult(), restored.getOperation());
    }
    bool hasMMA = false;
    loop.getBody()->walk([&](MMAOp mma) {
      hasMMA |= mma->getParentOfType<scf::ForOp>() == loop;
    });
    if (hasMMA)
      unswitchNativeAccessGuard(loop);
  }
  gpu::eraseDeadPhysicalValues(*physicalKernel);
  return success();
}

} // namespace intent::cutile

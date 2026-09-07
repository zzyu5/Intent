#include "Intent/Target/CuTile/Transforms/Passes.h"

#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace intent::cutile {

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
  }
  return success();
}

} // namespace intent::cutile

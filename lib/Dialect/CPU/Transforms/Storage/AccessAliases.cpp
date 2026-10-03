#include "AccessAliases.h"
#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace intent::cpu {

LogicalResult foldPrivateAccessAliases(func::FuncOp function) {
  llvm::SmallSetVector<Operation *, 32> accesses;
  {
    StorageAnalysis storage(function);
    function.walk([&](Operation *operation) {
      Value memory;
      if (auto load = dyn_cast<memref::LoadOp>(operation))
        memory = load.getMemref();
      else if (auto store = dyn_cast<memref::StoreOp>(operation))
        memory = store.getMemref();
      if (!memory) return;
      Value origin = storage.uniqueOrigin(memory);
      if (!origin || !origin.getDefiningOp<memref::AllocOp>() || memory == origin)
        return;

      // This is a private-intermediate optimization, not an ABI descriptor
      // normalization. Keep external views and vector access layout decisions
      // intact. Native patterns own offset/stride/rank-reduction composition.
      accesses.insert(operation);
      while (Operation *view = memory.getDefiningOp()) {
        if (!isa<memref::SubViewOp, memref::ExpandShapeOp,
                 memref::CollapseShapeOp, memref::CastOp>(view))
          break;
        accesses.insert(view);
        memory = view->getOperand(0);
      }
    });
  }
  if (accesses.empty()) return success();

  RewritePatternSet patterns(function.getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  // Native alias folding may create affine.apply. Close that implementation
  // detail here; CPU physical programs continue to contain ordinary arithmetic.
  populateAffineToStdConversionPatterns(patterns);
  GreedyRewriteConfig config;
  config.scope = &function.getBody();
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  if (failed(applyOpPatternsGreedily(accesses.getArrayRef(),
                                    std::move(patterns), config)))
    return function.emitError("private access alias folding did not converge");
  return success();
}

} // namespace intent::cpu

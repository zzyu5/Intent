#include "Intent/Target/CuTile/Emission/Translate.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "Intent/Transforms/Passes.h"
#include "Support/Model.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult emitCuTileSource(ModuleOp module, llvm::raw_ostream &output) {
  if (failed(verifyKernelModule(module)) || failed(verify(module)))
    return failure();
  SmallVector<intent::plan::RealizationOp> realizations(
      module.getOps<intent::plan::RealizationOp>());
  SmallVector<intent::plan::SearchSpaceOp> searchSpaces(
      module.getOps<intent::plan::SearchSpaceOp>());
  if (realizations.size() != 1 || searchSpaces.size() > 1)
    return module.emitError(
        "cuTile emission requires one realization and at most one search space");
  if (failed(plan::verifyCuTileRealization(realizations.front())))
    return failure();
  intent::plan::SearchSpaceOp searchSpace =
      searchSpaces.empty() ? intent::plan::SearchSpaceOp() : searchSpaces.front();
  if (searchSpace &&
      (searchSpace.getEntry() != realizations.front().getEntry() ||
       searchSpace.getTarget() != realizations.front().getTarget()))
    return searchSpace.emitOpError("does not match the resolved realization");
  if (searchSpace && failed(plan::verifyCuTileSearchSpace(searchSpace)))
    return failure();
  FailureOr<intent::target::KernelModel> kernel =
      intent::target::analyzeKernel(module);
  if (failed(kernel))
    return failure();
  return emission::emitRealizedKernelSource(
      std::move(*kernel), realizations.front(), searchSpace, output);
}

} // namespace intent::cutile

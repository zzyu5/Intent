#include "Intent/Target/Triton/Emission/Translate.h"

#include "Emitter.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "Intent/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::triton {

LogicalResult emitTritonSource(ModuleOp module, llvm::raw_ostream &output) {
  if (failed(verifyKernelModule(module)) || failed(verify(module)))
    return failure();
  if (!module.getOps<intent::plan::SearchSpaceOp>().empty())
    return module.emitError(
        "Triton emission consumes a resolved realization, not a search space");
  SmallVector<intent::plan::RealizationOp> realizations(
      module.getOps<intent::plan::RealizationOp>());
  if (realizations.size() != 1)
    return module.emitError(
        "Triton emission requires exactly one resolved realization");
  if (failed(plan::verifyTritonRealization(realizations.front())))
    return failure();
  return emitRealizedKernelSource(module, realizations.front(), output);
}

} // namespace intent::triton

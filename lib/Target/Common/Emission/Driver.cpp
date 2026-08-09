#include "Intent/Target/Common/Emission/Driver.h"

#include "Intent/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::target {

LogicalResult emitTargetSource(ModuleOp module, llvm::raw_ostream &output,
                               const EmissionTarget &target) {
  if (!target.emitKernelSource)
    return module.emitError("target emission has an incomplete implementation");
  if (failed(verifyKernelModule(module)) || failed(verify(module)))
    return failure();

  SmallVector<plan::RealizationOp> realizations(
      module.getOps<plan::RealizationOp>());
  SmallVector<plan::SearchSpaceOp> searchSpaces(
      module.getOps<plan::SearchSpaceOp>());
  if (realizations.size() != 1 || searchSpaces.size() > 1)
    return module.emitError()
           << target.displayName
           << " emission requires one realization and at most one search space";
  if (failed(plan::verifyGpuRealization(realizations.front())))
    return failure();

  plan::SearchSpaceOp searchSpace =
      searchSpaces.empty() ? plan::SearchSpaceOp() : searchSpaces.front();
  if (searchSpace &&
      (searchSpace.getEntry() != realizations.front().getEntry() ||
       searchSpace.getTarget() != realizations.front().getTarget()))
    return searchSpace.emitOpError("does not match the resolved realization");
  if (searchSpace && failed(plan::verifyGpuSearchSpace(searchSpace)))
    return failure();

  FailureOr<KernelModel> kernel = analyzeKernel(module);
  if (failed(kernel))
    return failure();
  return target.emitKernelSource(std::move(*kernel), realizations.front(),
                                 searchSpace, output);
}

} // namespace intent::target

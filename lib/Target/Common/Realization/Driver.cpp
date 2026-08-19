#include "Intent/Target/Common/Realization/Driver.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Transforms/Passes.h"

using namespace mlir;

namespace intent::target {

LogicalResult realizeTarget(ModuleOp module, TargetRealization realize) {
  if (!module.getOps<plan::ProgramOp>().empty() ||
      !module.getOps<plan::SearchSpaceOp>().empty())
    return module.emitError("module already contains a physical program");
  if (failed(verifyKernelModule(module)))
    return failure();

  FailureOr<KernelModel> kernel = analyzeKernel(module);
  if (failed(kernel))
    return failure();
  return realize(*kernel);
}

} // namespace intent::target

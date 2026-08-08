#include "Intent/Target/CuTile/Realization/Realize.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Transforms/Passes.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult realizeKernel(ModuleOp module, const TargetOptions &targetOptions) {
  if (targetOptions.architecture.empty() || targetOptions.device < 0)
    return module.emitError("cuTile realization requires a complete target");
  if (!module.getOps<intent::plan::RealizationOp>().empty() ||
      !module.getOps<intent::plan::SearchSpaceOp>().empty())
    return module.emitError("module already contains realization state");
  if (failed(verifyKernelModule(module)))
    return failure();

  FailureOr<intent::target::KernelModel> kernel =
      intent::target::analyzeKernel(module);
  if (failed(kernel))
    return failure();
  realization::OperationFacts facts(*kernel);
  if (failed(realization::analyzeOperations(facts)))
    return failure();
  FailureOr<realization::PolicyDecision> policy =
      realization::decidePolicy(facts);
  if (failed(policy))
    return failure();
  return realization::emitPlan(module, targetOptions, facts, *policy);
}

} // namespace intent::cutile

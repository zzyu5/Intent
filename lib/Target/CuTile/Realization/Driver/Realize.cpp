#include "Intent/Target/CuTile/Realization/Realize.h"

#include "Intent/Target/Common/Realization/Driver.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult realizeKernel(ModuleOp module, const TargetOptions &targetOptions) {
  if (targetOptions.architecture.empty() || targetOptions.device < 0)
    return module.emitError("cuTile realization requires a complete target");
  return intent::target::realizeTarget(
      module, [&](intent::target::KernelModel &kernel) {
        realization::OperationFacts facts(kernel);
        if (failed(realization::analyzeOperations(facts)))
          return failure();
        FailureOr<realization::PolicyDecision> policy =
            realization::decidePolicy(facts);
        if (failed(policy))
          return failure();
        return realization::emitPlan(module, targetOptions, facts, *policy);
      });
}

} // namespace intent::cutile

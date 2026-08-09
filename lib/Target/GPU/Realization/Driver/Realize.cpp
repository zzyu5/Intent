#include "Intent/Target/GPU/Realization/Realize.h"

#include "Intent/Target/Common/Realization/Driver.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult realizeKernel(ModuleOp module,
                            const DeviceCapabilities &device) {
  if (device.device < 0 || device.computeUnits <= 0 ||
      device.sharedMemoryPerUnit <= 0 || device.registersPerUnit <= 0)
    return module.emitError(
        "GPU realization requires positive algorithm-visible capacities");
  return intent::target::realizeTarget(
      module, [&](intent::target::KernelModel &kernel) {
        realization::KernelFacts facts(kernel);
        if (failed(realization::analyzeOperations(facts)))
          return failure();
        return realization::emitMachinePlan(module, device, facts);
      });
}

} // namespace intent::gpu

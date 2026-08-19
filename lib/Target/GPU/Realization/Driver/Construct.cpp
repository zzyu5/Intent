#include "Intent/Target/GPU/Realization/PhysicalProgram.h"

#include "Intent/Target/Common/Realization/Driver.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult constructPhysicalProgram(ModuleOp module,
                                       const DeviceCapabilities &device) {
  if (device.device < 0 || device.computeUnits <= 0 ||
      device.sharedMemoryPerUnit <= 0 || device.registersPerUnit <= 0)
    return module.emitError(
        "GPU physical-program construction requires positive algorithm-visible capacities");
  return intent::target::realizeTarget(
      module, [&](intent::target::KernelModel &kernel) {
        realization::KernelFacts facts(kernel);
        if (failed(realization::analyzeOperations(facts)))
          return failure();
        return realization::buildPhysicalProgram(module, device, facts);
      });
}

} // namespace intent::gpu

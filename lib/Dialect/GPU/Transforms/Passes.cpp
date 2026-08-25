#include "Intent/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult runSharedGPUPasses(ModuleOp module) {
  if (failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeAccessComposition(module)) ||
      failed(verifyGPUProgram(module)) ||
      failed(realizeContractionBlocking(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeReductionBlocking(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  return verifyGPUProgram(module);
}

} // namespace intent::gpu

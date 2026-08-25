#include "Intent/Conversion/KIRToGPU/KIRToGPU.h"

#include "Intent/Transforms/Passes.h"

using namespace mlir;

namespace intent {

LogicalResult lowerCanonicalKIRToGPU(ModuleOp module) {
  if (failed(verifyKernelModule(module)))
    return failure();
  return module.emitError(
      "canonical Intent KIR verified; executable GPU Program construction "
      "is not implemented for this typed boundary");
}

} // namespace intent

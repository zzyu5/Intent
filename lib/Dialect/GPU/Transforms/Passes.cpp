#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult runSharedGPUPasses(ModuleOp module) {
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  if (failed(realizeAccessComposition(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  // Normalize multi-axis reductions while their complete logical source
  // traversals are still intact.  This separates structured consumers from
  // pointwise consumers before ownership blocking rewrites shared ranges.
  if (failed(decomposeMultiAxisReductions(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  if (failed(realizePointwiseOwnership(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  // Establish every ownership/internal physical range while the structured
  // operations still expose which logical axes they consume.  Later
  // structured passes lower those already-physical slices into loops and
  // primitives; they must not be asked to reconstruct range provenance.
  if (failed(realizePointwiseBlocking(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  if (failed(realizeRegionFolds(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  if (failed(realizeRegionScans(module)))
    return failure();
  if (failed(mlir::verify(module.getOperation())))
    return failure();
  if (failed(realizeContractionBlocking(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  if (failed(realizeReductionBlocking(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  // Structured realization replays source slices and may create new gathers.
  // Compose those typed access relations before provider legalization just as
  // we do for the access graph constructed directly from KIR.
  if (failed(realizeAccessComposition(module)) ||
      failed(mlir::verify(module.getOperation())))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  eraseUnusedPhysicalParameters(*kernel);
  return verifyGPUProgram(module);
}

} // namespace intent::gpu

#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult completeGPUProgramConstruction(ModuleOp module) {
  auto verifyStructured = [&] {
    return verifyGPUProgramStage(module,
                                 GPUProgramStage::StructuredConstruction);
  };
  if (failed(verifyStructured()))
    return failure();
  if (failed(realizeAccessComposition(module)) ||
      failed(verifyStructured()))
    return failure();
  // Normalize multi-axis reductions while their complete logical source
  // traversals are still intact.  This separates structured consumers from
  // pointwise consumers before ownership blocking rewrites shared ranges.
  if (failed(decomposeMultiAxisReductions(module)) ||
      failed(verifyStructured()))
    return failure();
  if (failed(realizePointwiseOwnership(module)) ||
      failed(verifyStructured()))
    return failure();
  // Establish every ownership/internal physical range while the structured
  // operations still expose which logical axes they consume.  Later
  // structured passes lower those already-physical slices into loops and
  // primitives; they must not be asked to reconstruct range provenance.
  if (failed(realizePointwiseBlocking(module)) ||
      failed(verifyStructured()))
    return failure();
  if (failed(realizeRegionFolds(module)) ||
      failed(verifyStructured()))
    return failure();
  if (failed(realizeRegionScans(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel) || failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAggregateValueRelations(*kernel)))
    return failure();
  if (failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeContractionBlocking(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeReductionBlocking(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  // Structured realization replays source slices and may create new gathers.
  // Compose those typed access relations before provider legalization just as
  // we do for the access graph constructed directly from KIR.
  if (failed(realizeAccessComposition(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  return success();
}

LogicalResult runSharedGPUPasses(ModuleOp module) {
  if (failed(verifyGPUProgram(module)))
    return failure();
  if (failed(refineProgramMapping(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  eraseUnusedPhysicalParameters(*kernel);
  return verifyGPUProgram(module);
}

} // namespace intent::gpu

#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::gpu {

LogicalResult completeGPUProgramConstruction(ModuleOp module) {
  // Region fold/scan are complete executable structured operations: their
  // source slices, helper regions, carry and result assembly are verified by
  // the operations themselves.  Subsequent realization rewrites one complete
  // program into another; it does not finish a construction-time shell.
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  // Access composition is the final construction step for indexed reads whose
  // minimal fragment seed is not itself their logical source extent.  Complete
  // those exact relations before applying the executable-program verifier.
  if (failed(realizeAccessComposition(module)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  // Normalize multi-axis reductions while their complete logical source
  // traversals are still intact.  This separates structured consumers from
  // pointwise consumers before ownership blocking rewrites shared ranges.
  if (failed(decomposeMultiAxisReductions(module)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizePointwiseOwnership(module)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  // Establish every ownership/internal physical range while the structured
  // operations still expose which logical axes they consume.  Later
  // structured passes lower those already-physical slices into loops and
  // primitives; they must not be asked to reconstruct range provenance.
  if (failed(realizePointwiseBlocking(module)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  // A pointwise producer graph can feed a compatible max/normalization/moment
  // summary over one traversal.  Co-realize that typed algebra before the
  // individual structured passes split its shared producer into sibling loops.
  if (failed(realizeOnlineReductions(module)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignReductionResultRelations(*kernel)) ||
      failed(alignReductionIdentityRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignReductionYieldRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeRegionFolds(module)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeRegionScans(module)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeReductionBlocking(module)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignReductionResultRelations(*kernel)) ||
      failed(alignReductionIdentityRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignReductionYieldRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  if (failed(realizeContractionBlocking(module)) ||
      failed(alignAggregateValueRelations(*kernel)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(verifyGPUProgram(module)))
    return failure();
  // Structured realization replays source slices and may create new gathers.
  // Compose those typed access relations before provider legalization just as
  // we do for the access graph constructed directly from KIR.
  if (failed(realizeAccessComposition(module)) ||
      failed(alignAccessResultRelations(*kernel)) ||
      failed(alignPointwiseValueRelations(*kernel)) ||
      failed(alignAccessValueRelations(*kernel)) ||
      failed(refreshReshapeRelations(*kernel)) ||
      failed(alignContractValueRelations(*kernel)) ||
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
  if (failed(materializeSharedConfigTuples(*kernel)) ||
      failed(verifySharedConfigTuples(*kernel)))
    return failure();
  return verifyGPUProgram(module);
}

} // namespace intent::gpu

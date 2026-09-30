#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace intent::gpu {

class TuningProfiles;
void registerGPUPasses();

mlir::LogicalResult verifyGPUProgram(mlir::ModuleOp module);
mlir::LogicalResult realizeAccessComposition(mlir::ModuleOp module);
mlir::LogicalResult simplifyMaskedAccessCoordinates(mlir::ModuleOp module);
mlir::LogicalResult predicateScalarControl(mlir::ModuleOp module);
mlir::LogicalResult realizeScanConsumerTraversals(mlir::ModuleOp module);
mlir::LogicalResult materializeRetainedValues(mlir::ModuleOp module);
mlir::LogicalResult vectorizeBufferLoops(mlir::ModuleOp module);
mlir::LogicalResult promoteBufferValues(mlir::ModuleOp module);
mlir::LogicalResult schedulePrivateStores(mlir::ModuleOp module);
/// Realizes region operations and closes their inlined value/access relations
/// before returning to the pipeline's executable-program verification boundary.
mlir::LogicalResult realizeRegionFolds(mlir::ModuleOp module);
mlir::LogicalResult realizeRegionScans(mlir::ModuleOp module);
mlir::LogicalResult realizeContractionBlocking(mlir::ModuleOp module);
mlir::LogicalResult materializeProgramBuffers(mlir::ModuleOp module);
mlir::LogicalResult lowerInvocationWorkspaces(mlir::ModuleOp module);
mlir::LogicalResult orientLoopContractions(mlir::ModuleOp module);
mlir::LogicalResult realizeVectorContractions(mlir::ModuleOp module);
mlir::LogicalResult normalizeContractionSources(mlir::ModuleOp module);
mlir::LogicalResult decomposeMultiAxisReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeOnlineReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeReductionBlocking(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseOwnership(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseBlocking(mlir::ModuleOp module);
mlir::LogicalResult refineProgramMapping(mlir::ModuleOp module);
mlir::LogicalResult eliminateCommonValues(mlir::ModuleOp module);
mlir::LogicalResult simplifyRangePredicates(mlir::ModuleOp module);
mlir::LogicalResult fuseIndependentReductions(mlir::ModuleOp module);
mlir::LogicalResult fuseIndependentTraversals(mlir::ModuleOp module);
mlir::LogicalResult completeGPUProgramConstruction(mlir::ModuleOp module);
mlir::LogicalResult runSharedGPUPasses(mlir::ModuleOp module,
                                      const TuningProfiles &profiles);

} // namespace intent::gpu

#endif

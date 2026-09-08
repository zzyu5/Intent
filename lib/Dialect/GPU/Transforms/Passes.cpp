#include "Intent/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;

namespace intent::gpu {
namespace {

// Each group owns its relation repairs. Analyses are recreated by its rewrites;
// the executable-program verifier runs only after the complete group.
LogicalResult closeReductionValueRelations(func::FuncOp kernel) {
  if (failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult normalizeStructuredSources(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeVectorContractions(module)) ||
      failed(decomposeMultiAxisReductions(module)) ||
      failed(refreshReshapeRelations(kernel)))
    return failure();
  return success();
}

LogicalResult formPointwiseOwnership(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizePointwiseOwnership(module)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return closeReductionValueRelations(kernel);
}

LogicalResult formPointwiseBlocking(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizePointwiseBlocking(module)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return closeReductionValueRelations(kernel);
}

LogicalResult coRealizeOnlineReductions(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeOnlineReductions(module)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult closeValueAccessRelations(func::FuncOp kernel) {
  if (failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult realizeRegionFoldGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeRegionFolds(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult realizeRegionScanGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeRegionScans(module)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult realizeReductionGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeReductionBlocking(module)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignReductionResultRelations(kernel)) ||
      failed(alignReductionIdentityRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignReductionYieldRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult realizeContractionGroup(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeContractionBlocking(module)) ||
      failed(alignAggregateValueRelations(kernel)) ||
      failed(alignAccessResultRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(refreshReshapeRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)) ||
      failed(alignContractValueRelations(kernel)) ||
      failed(alignAccessValueRelations(kernel)) ||
      failed(alignPointwiseValueRelations(kernel)))
    return failure();
  return success();
}

LogicalResult composeRealizedAccesses(ModuleOp module, func::FuncOp kernel) {
  if (failed(realizeAccessComposition(module)) ||
      failed(orientLoopContractions(module)) ||
      failed(alignAggregateValueRelations(kernel)))
    return failure();
  return closeValueAccessRelations(kernel);
}

LogicalResult refineMapping(ModuleOp module, func::FuncOp) {
  return refineProgramMapping(module);
}

LogicalResult simplifyValues(ModuleOp module, func::FuncOp) {
  return eliminateCommonValues(module);
}

LogicalResult closeSharedConfigurations(func::FuncOp kernel,
                                        const TuningProfiles &profiles) {
  eraseUnusedPhysicalParameters(kernel);
  if (failed(materializeSharedConfigTuples(kernel, profiles)))
    return failure();
  return verifySharedConfigTuples(kernel);
}

struct TransformationGroup {
  StringRef name;
  LogicalResult (*run)(ModuleOp, func::FuncOp);
};

} // namespace

LogicalResult completeGPUProgramConstruction(ModuleOp module) {
  // Construction closes the initial indexed access graph. Ownership/blocking
  // and structured realization are subsequent transformations of this program.
  if (failed(realizeAccessComposition(module)))
    return failure();
  return verifyGPUProgram(module);
}

LogicalResult runSharedGPUPasses(ModuleOp module, const TuningProfiles &profiles) {
  if (failed(verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  const TransformationGroup groups[] = {
      {"normalize-structured-sources", normalizeStructuredSources},
      {"form-pointwise-ownership", formPointwiseOwnership},
      {"form-pointwise-blocking", formPointwiseBlocking},
      {"co-realize-online-reductions", coRealizeOnlineReductions},
      {"realize-region-folds", realizeRegionFoldGroup},
      {"realize-region-scans", realizeRegionScanGroup},
      {"realize-reductions", realizeReductionGroup},
      {"realize-contractions", realizeContractionGroup},
      {"compose-realized-accesses", composeRealizedAccesses},
      {"refine-program-mapping", refineMapping},
      {"eliminate-common-values", simplifyValues},
  };
  for (const TransformationGroup &group : groups) {
    if (failed(group.run(module, *kernel)))
      return module.emitError()
             << "shared GPU transformation failed: " << group.name;
    if (failed(verifyGPUProgram(module)))
      return module.emitError()
             << "shared GPU postcondition failed: " << group.name;
  }
  if (failed(closeSharedConfigurations(*kernel, profiles)))
    return failure();
  return verifyGPUProgram(module);
}

} // namespace intent::gpu

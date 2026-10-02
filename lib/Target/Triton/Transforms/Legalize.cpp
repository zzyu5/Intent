#include "Legalization.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Contraction.h"
#include "Intent/Dialect/GPU/Transforms/ExecutionGroups.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Dialect/GPU/Transforms/ValueMaterialization.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::triton {

LogicalResult prepareTritonMemory(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)) ||
      failed(gpu::lowerExecutionGroups(module)))
    return failure();
  auto kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  detail::foldIntegerScanTails(*kernel);
  if (failed(gpu::materializeProgramBuffers(module)) ||
      failed(detail::materializeOversizedGathers(*kernel)) ||
      failed(detail::legalizeLargeScalarGathers(*kernel)) ||
      failed(detail::legalizeMaskedGather(*kernel)) ||
      failed(detail::legalizeExpandingGathers(*kernel)) ||
      failed(detail::legalizeScatterAdd(*kernel)) ||
      failed(gpu::contraction::normalizeMatrixContractShapes(*kernel)))
    return failure();
  return gpu::verifyGPUProgram(module);
}

LogicalResult formTritonProgram(ModuleOp module) {
  auto profiles = gpu::TuningProfiles::from(module);
  if (failed(profiles)) return failure();
  auto kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  bool requiresCtaSynchronization =
      llvm::any_of(kernel->getArgumentTypes(), [](Type type) {
        return isa<gpu::BufferType>(type);
      }) || detail::hasOrderedViewDependencies(*kernel);
  auto localOptions = declareProviderOptions(
      *kernel, *profiles, requiresCtaSynchronization,
      detail::findLoadPipelineLoops(*kernel));
  if (failed(localOptions) || failed(gpu::verifyGPUProgram(module)) ||
      failed(gpu::lowerInvocationWorkspaces(module)))
    return failure();
  if (failed(detail::legalizeOrderedViewDependencies(*kernel)) ||
      failed(detail::legalizeSplitGatherPairs(*kernel)))
    return failure();
  detail::selectOrderedLoadUnrolling(*kernel);
  auto descriptors =
      detail::materializeTensorDescriptorForms(*kernel, *localOptions);
  if (failed(descriptors) ||
      failed(detail::materializeBlockPointerForms(*kernel)))
    return failure();
  detail::orientPointerLoads(*kernel);
  if (failed(materializeLegalConfigs(*kernel, *descriptors, *localOptions)))
    return failure();
  detail::selectContractForms(*kernel);
  // Native extensions now belong to the current program; the shared surface
  // verifier is no longer applicable.
  return mlir::verify(module);
}

LogicalResult finalizeTritonProgram(ModuleOp module) {
  auto kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  gpu::foldExactConstantDivisions(*kernel);
  detail::canonicalizeBroadcastProjections(*kernel);
  SmallVector<gpu::AssumeInBoundsOp> assumptions;
  kernel->walk([&](gpu::AssumeInBoundsOp assumption) {
    assumptions.push_back(assumption);
  });
  for (gpu::AssumeInBoundsOp assumption : assumptions)
    assumption.erase();
  if (failed(gpu::eliminateCommonValues(module)) ||
      failed(detail::legalizeCollectiveCallbacks(*kernel)) ||
      failed(materializeDeferredResourceBounds(*kernel)))
    return failure();
  detail::sinkSelectProducers(*kernel);
  if (failed(verifyTritonProgram(module)))
    return failure();
  (*kernel)->setAttr(detail::legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::triton

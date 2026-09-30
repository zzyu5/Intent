#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace intent::dsa {

LogicalResult runProgramTransforms(ModuleOp module) {
  if (failed(verifyProgram(module))) return failure();
  auto function = *module.getOps<func::FuncOp>().begin();
  if (failed(realizeCollectiveGatherSupply(function)))
    return module.emitError("DSA collective gather supply realization failed");
  if (failed(verifyProgram(module)))
    return module.emitError("DSA collective gather supply postcondition failed");
  if (failed(realizeMatrixSupply(function)))
    return module.emitError("DSA matrix supply realization failed");
  if (failed(verifyProgram(module)))
    return module.emitError("DSA matrix supply postcondition failed");
  if (normalizeLinearIndices(function)) {
    PassManager cleanup(module.getContext());
    cleanup.addPass(createCanonicalizerPass());
    cleanup.addPass(createCSEPass());
    if (failed(cleanup.run(module))) return failure();
  }
  if (failed(verifyProgram(module)))
    return module.emitError("DSA index normalization postcondition failed");
  return success();
}

} // namespace intent::dsa

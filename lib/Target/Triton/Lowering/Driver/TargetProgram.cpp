#include "Intent/Target/Triton/Lowering/TargetProgram.h"

#include "Intent/Target/Common/Lowering/Driver.h"
#include "Intent/Target/Triton/Lowering/Passes.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::triton {

LogicalResult materializeTritonProgram(ModuleOp module) {
  return intent::target::runTargetMaterializationPipeline(
      module, intent::target::MaterializationTarget{
                  "triton", "Triton", lowering::addProviderPasses,
                  lowering::verifyProviderProgram,
                  lowering::materializeProgramSource});
}

LogicalResult translateTritonProgram(ModuleOp module,
                                     llvm::raw_ostream &output) {
  return intent::target::translateTargetProgram(module, "triton", output);
}

} // namespace intent::triton

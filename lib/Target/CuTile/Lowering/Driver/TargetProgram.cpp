#include "Intent/Target/CuTile/Lowering/TargetProgram.h"

#include "Intent/Target/Common/Lowering/Driver.h"
#include "Intent/Target/CuTile/Lowering/Passes.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult materializeCuTileProgram(ModuleOp module) {
  return intent::target::runTargetMaterializationPipeline(
      module, intent::target::MaterializationTarget{
                  "cutile", "cuTile", lowering::addProviderPasses,
                  lowering::verifyProviderProgram,
                  lowering::materializeProgramSource});
}

LogicalResult translateCuTileProgram(ModuleOp module,
                                     llvm::raw_ostream &output) {
  return intent::target::translateTargetProgram(module, "cutile", output);
}

} // namespace intent::cutile

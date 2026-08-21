#include "Intent/Target/TileLang/Lowering/TargetProgram.h"

#include "Intent/Target/Common/Lowering/Driver.h"
#include "Intent/Target/TileLang/Lowering/Passes.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::tilelang {

LogicalResult materializeTileLangProgram(ModuleOp module) {
  return intent::target::runTargetMaterializationPipeline(
      module, intent::target::MaterializationTarget{
                  "tilelang", "TileLang", lowering::addProviderPasses,
                  lowering::verifyProviderProgram,
                  lowering::materializeProgramSource});
}

LogicalResult translateTileLangProgram(ModuleOp module,
                                       llvm::raw_ostream &output) {
  return intent::target::translateTargetProgram(module, "tilelang", output);
}

} // namespace intent::tilelang

#include "Intent/Target/CuTile/Emission/Translate.h"

#include "Intent/Target/Common/Emission/Driver.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::cutile {

LogicalResult emitCuTileSource(ModuleOp module, llvm::raw_ostream &output) {
  return intent::target::emitTargetSource(
      module, output, intent::target::EmissionTarget{
                          "cuTile", emission::emitRealizedKernelSource});
}

} // namespace intent::cutile

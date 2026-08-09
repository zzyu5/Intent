#include "Intent/Target/TileLang/Emission/Translate.h"

#include "Intent/Target/Common/Emission/Driver.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::tilelang {

LogicalResult emitTileLangSource(ModuleOp module, llvm::raw_ostream &output) {
  return intent::target::emitTargetSource(
      module, output, intent::target::EmissionTarget{
                          "TileLang", emission::emitRealizedKernelSource});
}

} // namespace intent::tilelang

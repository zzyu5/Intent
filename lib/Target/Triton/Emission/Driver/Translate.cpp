#include "Intent/Target/Triton/Emission/Translate.h"

#include "Intent/Target/Common/Emission/Driver.h"
#include "Support/Model.h"

using namespace mlir;

namespace intent::triton {

LogicalResult emitTritonSource(ModuleOp module, llvm::raw_ostream &output) {
  return intent::target::emitTargetSource(
      module, output, intent::target::EmissionTarget{
                          "Triton", emission::emitRealizedKernelSource});
}

} // namespace intent::triton

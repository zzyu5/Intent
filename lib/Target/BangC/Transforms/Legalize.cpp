#include "PassDetail.h"

using namespace mlir;
namespace intent::bangc {

LogicalResult legalizeProgram(ModuleOp module, StringRef architecture) {
  if (architecture != "mtp_372")
    return module.emitError("BANG C currently has a bound implementation profile for mtp_372");
  if (failed(dsa::verifyProgram(module))) return failure();
  module->setAttr("bangc.architecture", StringAttr::get(module.getContext(), architecture));
  struct Group {
    StringRef name;
    LogicalResult (*run)(ModuleOp);
  };
  const Group groups[] = {
      {"native-computations", realizeNativeComputations},
      {"native-workspace", realizeNativeWorkspace},
      {"native-implementations", selectNativeImplementations},
      {"local-composition", composeLocalProgram},
      {"supply-synchronization", scheduleProgramSupply},
      {"storage-binding", bindProgramStorage},
  };
  for (const Group &group : groups) {
    if (failed(group.run(module)))
      return module.emitError() << "BANG C transformation failed: " << group.name;
    if (failed(dsa::verifyProgram(module)))
      return module.emitError() << "BANG C postcondition failed: " << group.name;
  }
  return verifyProgram(module);
}

} // namespace intent::bangc

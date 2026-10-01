#include "Intent/Compiler/Backend.h"
#include "Intent/Compiler/Passes.h"
#include "Intent/Dialect/DSA/Transforms/Passes.h"
#include "Intent/Target/BangC/Passes.h"

using namespace mlir;

namespace intent::compiler {

ArrayRef<Backend> dsaBackends() {
  static const Backend adapters[] = {
      {Provider::BangC, "bangc", true, DSABackend{}, nullptr,
       [] { bangc::registerBangCPasses(); bangc::registerBangCPipelines(); },
       [](OpPassManager &manager, const Request &request) {
         bangc::buildBangCPipeline(manager, request.dsa.architecture);
       },
       bangc::serializeProgram}};
  return adapters;
}

void DSABackend::buildConstruction(OpPassManager &manager, const Request &request) const {
  ConstructDSAOptions options;
  options.tile = request.dsa.tile;
  options.tileM = request.dsa.tileM;
  options.tileN = request.dsa.tileN;
  options.tileK = request.dsa.tileK;
  options.regionTile = request.dsa.regionTile;
  options.tasks = request.dsa.tasks;
  options.localBytes = request.dsa.localBytes;
  options.shapes = request.dsa.shapes;
  options.strides = request.dsa.strides;
  manager.addPass(createConstructDSA(options));
}

void DSABackend::buildShared(OpPassManager &manager, const Request &, StringRef) const {
  dsa::buildDSAPipeline(manager);
}

LogicalResult DSABackend::verifySharedInput(ModuleOp module, const Request &request, StringRef) const {
  if (failed(dsa::verifyProgram(module))) return failure();
  if (module->hasAttr("bangc.architecture"))
    return module.emitError("provider-specific BANG C input cannot be resumed as shared DSA IR");
  if (request.dsa.architecture != "mtp_372")
    return module.emitError("shared DSA input requires the supported mtp_372 implementation profile");
  return success();
}

} // namespace intent::compiler

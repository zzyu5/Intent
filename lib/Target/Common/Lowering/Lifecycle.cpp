#include "Intent/Target/Common/Lowering/Lifecycle.h"

using namespace mlir;

namespace intent::target {

LogicalResult materializeSource(TargetProgramMaterializer &emitter) {
  if (failed(emitter.prepare()))
    return failure();
  emitter.emitImports();
  if (failed(emitter.emitHelpers()))
    return failure();
  if (failed(emitter.emitKernelHeader()))
    return failure();

  OperationHandlerRegistry registry;
  if (failed(emitter.registerOperationHandlers(registry)))
    return failure();
  emitter.setOperationRegistry(&registry);
  LogicalResult result =
      traverseKernel(emitter.entry(), registry, emitter.stage());
  emitter.setOperationRegistry(nullptr);
  if (failed(result))
    return failure();

  emitter.stream() << "\n\n";
  return emitter.emitWrapper();
}

} // namespace intent::target

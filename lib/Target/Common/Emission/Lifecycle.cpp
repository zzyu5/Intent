#include "Intent/Target/Common/Emission/Lifecycle.h"

using namespace mlir;

namespace intent::target {

LogicalResult emitSource(TargetSourceEmitter &emitter) {
  if (failed(emitter.prepare()))
    return failure();
  emitter.emitImports();
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

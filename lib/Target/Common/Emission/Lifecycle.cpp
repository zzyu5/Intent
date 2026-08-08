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
  if (failed(emitter.registerOperationHandlers(registry)) ||
      failed(traverseKernel(emitter.entry(), registry, emitter.stage())))
    return failure();

  emitter.stream() << "\n\n";
  return emitter.emitWrapper();
}

} // namespace intent::target

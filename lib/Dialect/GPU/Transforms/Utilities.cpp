#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu {

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  SmallVector<Operation *> operations;
  kernel.walk([&](Operation *operation) { operations.push_back(operation); });
  for (Operation *operation : llvm::reverse(operations)) {
    if (operation == kernel.getOperation() || operation->getNumRegions() != 0 ||
        !operation->getNumResults() ||
        !llvm::all_of(operation->getResults(),
                      [](Value value) { return value.use_empty(); }))
      continue;
    if (isMemoryEffectFree(operation))
      operation->erase();
  }
}

} // namespace intent::gpu

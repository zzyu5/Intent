#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/GPUOps.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

namespace intent::gpu {

void eraseDeadPhysicalValues(func::FuncOp kernel) {
  SmallVector<Operation *> operations;
  kernel.walk([&](Operation *operation) { operations.push_back(operation); });
  for (Operation *operation : llvm::reverse(operations)) {
    // Delinearize is the explicit execution-workset authority.  A singleton
    // workset may not consume its coordinate until a later blocking pass, so
    // ordinary SSA liveness cannot remove it between shared realizations.
    if (isa<DelinearizeOp>(operation))
      continue;
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

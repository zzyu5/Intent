#include "Intent/Dialect/GPU/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {
namespace {

void eliminateInBlock(Block &block) {
  llvm::DenseMap<OperationName, SmallVector<Operation *>> available;
  for (Operation &operation : llvm::make_early_inc_range(block)) {
    for (Region &region : operation.getRegions())
      for (Block &nested : region)
        eliminateInBlock(nested);
    // Parameter declarations also have symbolic type/attribute users. They are
    // retained and cleaned up by eraseUnusedPhysicalParameters, not SSA DCE.
    if (isa<ParameterOp, DelinearizeOp>(operation) ||
        operation.getNumRegions() != 0 ||
        operation.getNumResults() == 0 || !isMemoryEffectFree(&operation) ||
        !isPhysicalReplayNode(&operation, PhysicalReplayScope::ValueGraph,
                              /*allowAccesses=*/false))
      continue;
    auto &candidates = available[operation.getName()];
    Operation *equivalent = nullptr;
    for (Operation *candidate : candidates)
      if (OperationEquivalence::isEquivalentTo(
              candidate, &operation, OperationEquivalence::exactValueMatch,
              nullptr, OperationEquivalence::IgnoreLocations)) {
        equivalent = candidate;
        break;
      }
    if (!equivalent) {
      candidates.push_back(&operation);
      continue;
    }
    operation.replaceAllUsesWith(equivalent->getResults());
    operation.erase();
  }
}

} // namespace

LogicalResult eliminateCommonValues(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  for (Block &block : kernel->getBody())
    eliminateInBlock(block);
  return success();
}

} // namespace intent::gpu

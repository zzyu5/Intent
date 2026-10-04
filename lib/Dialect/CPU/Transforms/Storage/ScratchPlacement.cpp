#include "ScratchStorage.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace intent::cpu::detail {
namespace {

bool invariantCapacity(const ScratchAllocation &scratch, scf::ForOp loop,
                       DominanceInfo &dominance) {
  using Bounds = ValueBoundsConstraintSet;
  auto zero = IntegerAttr::get(IndexType::get(loop.getContext()), 0);
  for (Value operand : scratch.operation->getOperands()) {
    if (!dominance.properlyDominates(operand, loop) ||
        !Bounds::compare(Bounds::Variable(operand), Bounds::GE,
                         Bounds::Variable(zero)))
      return false;
  }
  return true;
}

} // namespace

bool placeScratchAllocations(func::FuncOp function, int64_t byteLimit,
                             ScratchRepresentation representation) {
  SmallVector<Operation *> allocations;
  function.walk([&](Operation *operation) {
    if (isa<memref::AllocOp, memref::AllocaOp>(operation))
      allocations.push_back(operation);
  });
  for (Operation *operation : llvm::reverse(allocations)) {
    // A direct serial-loop boundary is the only motion edge. In particular,
    // If, task, parallel and automatic-allocation scopes retain their owners.
    auto loop = dyn_cast<scf::ForOp>(operation->getParentOp());
    if (!loop) continue;
    StorageAnalysis storage(function);
    auto scratch = scratchAllocation(operation, storage);
    if (!scratch) continue;
    DominanceInfo dominance(function);
    if (invariantCapacity(*scratch, loop, dominance)) {
      operation->moveBefore(loop);
      if (scratch->release) scratch->release->moveAfter(loop);
      return true;
    }
    if (representation != ScratchRepresentation::LinearCapacity) continue;
    auto capacity = scratchCapacity(*scratch, byteLimit);
    if (!capacity) continue;
    OpBuilder before(loop);
    Value backing = createScratchBacking(before, *scratch, *capacity,
                                          scratchAlignment(*scratch));
    OpBuilder at(operation);
    Value descriptor = scratchDescriptor(at, *scratch, backing);
    scratch->memory.replaceAllUsesWith(descriptor);
    if (scratch->release) {
      OpBuilder after(loop);
      after.setInsertionPointAfter(loop);
      after.create<memref::DeallocOp>(operation->getLoc(), backing);
      scratch->release.erase();
    }
    operation->erase();
    return true;
  }
  return false;
}

} // namespace intent::cpu::detail

#include "ScratchStorage.h"
#include "Intent/Dialect/CPU/Analysis/ExtentRelations.h"
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace intent::cpu::detail {
namespace {

bool sameDescriptor(const ScratchAllocation &first,
                    const ScratchAllocation &second) {
  if (first.type != second.type) return false;
  for (int64_t axis = 0; axis < first.type.getRank(); ++axis)
    if (!haveEqualExtents(ValueBoundsConstraintSet::Variable(first.memory, axis),
                          ValueBoundsConstraintSet::Variable(second.memory, axis)))
      return false;
  return true;
}

SmallVector<intent::QuantFormat> storageInterpretations(
    const ScratchAllocation &scratch) {
  SmallVector<intent::QuantFormat> formats;
  auto require = [&](Value memory, intent::QuantFormat format) {
    if (llvm::is_contained(scratch.aliases.values, memory) &&
        !llvm::is_contained(formats, format))
      formats.push_back(format);
  };
  for (Operation *user : scratch.aliases.users) {
    if (auto quantize = dyn_cast<QuantizeOp>(user))
      require(quantize.getOutput(), quantize.getFormat());
    else if (auto dot = dyn_cast<QuantizedDotOp>(user)) {
      require(dot.getLhs(), dot.getLhsFormat());
      require(dot.getRhs(), dot.getRhsFormat());
    }
  }
  return formats;
}

bool sameStorageInterpretation(const ScratchAllocation &first,
                              const ScratchAllocation &second) {
  auto firstFormats = storageInterpretations(first);
  auto secondFormats = storageInterpretations(second);
  // Encoded storage interpretation belongs to its allocation identity even
  // when two uses have disjoint lifetimes and identical memref descriptors.
  return firstFormats.size() <= 1 && firstFormats == secondFormats;
}

void alignScratch(const ScratchAllocation &scratch, int64_t alignment) {
  if (!alignment) return;
  if (scratch.stack) cast<memref::AllocaOp>(scratch.operation).setAlignment(alignment);
  else cast<memref::AllocOp>(scratch.operation).setAlignment(alignment);
}

} // namespace

bool reuseScratchSlots(func::FuncOp function, int64_t byteLimit,
                       ScratchRepresentation representation) {
  StorageAnalysis storage(function);
  SmallVector<ScratchAllocation> allocations;
  function.walk([&](Operation *operation) {
    if (auto scratch = scratchAllocation(operation, storage))
      allocations.push_back(std::move(*scratch));
  });
  for (auto [number, first] : llvm::enumerate(allocations)) {
    for (ScratchAllocation &second : MutableArrayRef(allocations).drop_front(number + 1)) {
      if (first.operation->getBlock() != second.operation->getBlock() ||
          first.stack != second.stack ||
          first.type.getElementType() != second.type.getElementType() ||
          first.type.getMemorySpace() != second.type.getMemorySpace() ||
          !sameStorageInterpretation(first, second)) continue;
      Operation *end = first.release ? first.release.getOperation() : first.lastUse;
      if (!end->isBeforeInBlock(second.operation)) continue;
      int64_t alignment = std::max(scratchAlignment(first), scratchAlignment(second));
      if (sameDescriptor(first, second)) {
        alignScratch(first, alignment);
        second.memory.replaceAllUsesWith(first.memory);
        if (first.release) {
          first.release->moveBefore(second.release);
          second.release->erase();
        }
        second.operation->erase();
        return true;
      }
      if (representation != ScratchRepresentation::LinearCapacity) continue;
      auto firstCapacity = scratchCapacity(first, byteLimit);
      auto secondCapacity = scratchCapacity(second, byteLimit);
      if (!firstCapacity || !secondCapacity) continue;
      OpBuilder atFirst(first.operation);
      Value backing = createScratchBacking(atFirst, first,
          std::max(*firstCapacity, *secondCapacity), alignment);
      Value firstView = scratchDescriptor(atFirst, first, backing);
      OpBuilder atSecond(second.operation);
      Value secondView = scratchDescriptor(atSecond, second, backing);
      first.memory.replaceAllUsesWith(firstView);
      second.memory.replaceAllUsesWith(secondView);
      if (first.release) {
        OpBuilder atEnd(second.release);
        atEnd.create<memref::DeallocOp>(second.release.getLoc(), backing);
        first.release.erase();
        second.release->erase();
      }
      first.operation->erase();
      second.operation->erase();
      return true;
    }
  }
  return false;
}

} // namespace intent::cpu::detail

namespace intent::cpu {

LogicalResult reuseScratchStorage(func::FuncOp function,
                                 ScratchRepresentation representation) {
  if (function.isExternal()) return success();
  int64_t capacityBudget = 0;
  if (representation == ScratchRepresentation::LinearCapacity) {
    auto capabilities = function->getParentOfType<ModuleOp>()->getAttrOfType<CapabilitiesAttr>(
        "intent_cpu.capabilities");
    if (!capabilities) return function.emitError("scratch capacity expansion requires CPU capabilities");
    capacityBudget = capabilities.getPrivateBytes();
  }
  // Coalesce while the original short lifetimes are still explicit, then lift
  // their shared allocation. Each mutation invalidates all collected facts.
  while (detail::reuseScratchSlots(function, capacityBudget, representation) ||
         detail::placeScratchAllocations(function, capacityBudget, representation)) {}
  return success();
}

} // namespace intent::cpu

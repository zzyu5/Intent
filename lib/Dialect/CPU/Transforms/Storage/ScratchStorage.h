#ifndef INTENT_CPU_TRANSFORMS_STORAGE_SCRATCHSTORAGE_H
#define INTENT_CPU_TRANSFORMS_STORAGE_SCRATCHSTORAGE_H

#include "Intent/Dialect/CPU/Analysis/Storage.h"
#include "Intent/Dialect/CPU/Transforms/Storage/Storage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::cpu::detail {

// Ephemeral facts about one current allocation and its closed local uses.
struct ScratchAllocation {
  mlir::Operation *operation;
  mlir::Value memory;
  mlir::MemRefType type;
  mlir::Operation *lastUse;
  mlir::memref::DeallocOp release;
  StorageAliasFacts aliases;
  bool stack;
};

std::optional<ScratchAllocation> scratchAllocation(
    mlir::Operation *operation, StorageAnalysis &storage);
std::optional<int64_t> scratchCapacity(const ScratchAllocation &scratch,
                                      int64_t byteLimit);
int64_t scratchAlignment(const ScratchAllocation &scratch);
mlir::Value createScratchBacking(mlir::OpBuilder &builder,
    const ScratchAllocation &scratch, int64_t capacity, int64_t alignment);
// Keep the original logical shape and dense strides even when backing capacity
// is larger or has a different rank. No initialization or read is moved.
mlir::Value scratchDescriptor(mlir::OpBuilder &builder,
    const ScratchAllocation &scratch, mlir::Value backing);

bool reuseScratchSlots(mlir::func::FuncOp function, int64_t byteLimit,
                       ScratchRepresentation representation);
bool placeScratchAllocations(mlir::func::FuncOp function, int64_t byteLimit,
                             ScratchRepresentation representation);

} // namespace intent::cpu::detail
#endif

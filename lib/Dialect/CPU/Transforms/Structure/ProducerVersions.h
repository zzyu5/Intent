#ifndef INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERVERSIONS_H
#define INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERVERSIONS_H

#include "ProducerReuse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cpu {

struct ProducerVersionUse {
  mlir::Operation *operation;
  unsigned input;
};

// All observations after one complete write and before its next complete write
// or release. Every observation must be an exact, transportable input slot.
// These facts are valid only until the first rewrite of this lexical sequence.
struct ProducerVersion {
  llvm::SmallVector<ProducerVersionUse> uses;
};

bool completelyWritesBuffer(mlir::Operation *operation, mlir::Value buffer);

mlir::FailureOr<ProducerVersion> queryProducerVersion(
    mlir::linalg::GenericOp producer, StorageAnalysis &storage);

// Select an existing complete copy as the representative of a stored version.
// Retarget the defining write and its reads; do not replay any computation.
bool forwardProducerVersionCopies(mlir::func::FuncOp function);

// In addition to ordinary replay, permit a source's current element to be read
// immediately before its own injective pointwise overwrite. The source must be
// the exact output descriptor at the exact output coordinates, not an alias.
bool canReplayProducerVersionAt(const ProducerReplay &payload,
    mlir::linalg::GenericOp producer, mlir::Operation *consumer,
    mlir::AffineMap coordinates, StorageAnalysis &storage);

} // namespace intent::cpu

#endif

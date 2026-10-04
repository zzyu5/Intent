#ifndef INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERVERSIONS_H
#define INTENT_CPU_TRANSFORMS_STRUCTURE_PRODUCERVERSIONS_H

#include "ProducerReuse.h"

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

mlir::FailureOr<ProducerVersion> queryProducerVersion(
    mlir::linalg::GenericOp producer, StorageAnalysis &storage);

// In addition to ordinary replay, permit a source's current element to be read
// immediately before its own injective pointwise overwrite. The source must be
// the exact output descriptor at the exact output coordinates, not an alias.
bool canReplayProducerVersionAt(const ProducerReplay &payload,
    mlir::linalg::GenericOp producer, mlir::Operation *consumer,
    mlir::AffineMap coordinates, StorageAnalysis &storage);

} // namespace intent::cpu

#endif

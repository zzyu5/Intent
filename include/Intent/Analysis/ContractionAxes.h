#ifndef INTENT_ANALYSIS_CONTRACTIONAXES_H
#define INTENT_ANALYSIS_CONTRACTIONAXES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <string>

namespace intent {

enum class ContractionOperand { Lhs, Rhs };

struct ContractionAxisPair {
  unsigned lhs, rhs;
};

struct ContractionResultAxis {
  ContractionOperand operand;
  unsigned axis;
};

// Structural contraction semantics only: no values, extents, provenance or
// execution policy. Result positions follow the lhs non-reduction axes, then
// the rhs axes that are neither reduced nor paired batches. Repeated source
// identities do not merge these positions. Recompute when axis lists change.
struct ContractionAxes {
  static std::optional<ContractionAxes> get(
      unsigned lhsRank, unsigned rhsRank,
      llvm::ArrayRef<int64_t> lhsReduction,
      llvm::ArrayRef<int64_t> rhsReduction,
      llvm::ArrayRef<int64_t> lhsBatch, llvm::ArrayRef<int64_t> rhsBatch,
      std::string *failureReason = nullptr);

  llvm::SmallVector<ContractionAxisPair> reduction, batch;
  llvm::SmallVector<unsigned> lhsFree, rhsFree;
  llvm::SmallVector<ContractionResultAxis> results;
  // Reduction axes have no result position; both members of a batch pair map
  // to the same position, whose representative is the lhs member.
  llvm::SmallVector<std::optional<unsigned>> lhsResultAxes, rhsResultAxes;

  // Shape/dtype/capability checks remain with each consumer. This predicate
  // only recognizes the existing [..., M, K] x [..., K, N] axis convention.
  bool hasCanonicalMatrixAxes() const;
};

} // namespace intent

#endif

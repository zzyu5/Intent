#include "Intent/Analysis/ContractionAxes.h"
#include "llvm/ADT/SmallBitVector.h"

namespace intent {

std::optional<ContractionAxes> ContractionAxes::get(
    unsigned lhsRank, unsigned rhsRank, llvm::ArrayRef<int64_t> lhsReduction,
    llvm::ArrayRef<int64_t> rhsReduction, llvm::ArrayRef<int64_t> lhsBatch,
    llvm::ArrayRef<int64_t> rhsBatch, std::string *failureReason) {
  auto reject = [&](const std::string &reason) -> std::optional<ContractionAxes> {
    if (failureReason) *failureReason = reason;
    return std::nullopt;
  };
  if (lhsReduction.empty())
    return reject("requires at least one reduction pair");
  if (lhsReduction.size() != rhsReduction.size())
    return reject("lhs/rhs reduction pair counts disagree");
  if (lhsBatch.size() != rhsBatch.size())
    return reject("lhs/rhs batch pair counts disagree");

  ContractionAxes result;
  llvm::SmallBitVector lhsUsed(lhsRank), rhsUsed(rhsRank);
  auto appendPairs = [&](llvm::ArrayRef<int64_t> left,
                         llvm::ArrayRef<int64_t> right,
                         llvm::SmallVectorImpl<ContractionAxisPair> &pairs,
                         const char *role) -> std::optional<std::string> {
    for (unsigned index = 0; index < left.size(); ++index) {
      if (left[index] < 0 || static_cast<uint64_t>(left[index]) >= lhsRank)
        return std::string("lhs ") + role + " axis is out of range: " +
               std::to_string(left[index]);
      if (right[index] < 0 || static_cast<uint64_t>(right[index]) >= rhsRank)
        return std::string("rhs ") + role + " axis is out of range: " +
               std::to_string(right[index]);
      unsigned lhs = left[index], rhs = right[index];
      if (lhsUsed.test(lhs) || rhsUsed.test(rhs))
        return std::string(role) + " axis is repeated or overlaps another pair";
      lhsUsed.set(lhs);
      rhsUsed.set(rhs);
      pairs.push_back({lhs, rhs});
    }
    return std::nullopt;
  };
  if (auto error = appendPairs(lhsReduction, rhsReduction, result.reduction,
                               "reduction"))
    return reject(*error);
  if (auto error = appendPairs(lhsBatch, rhsBatch, result.batch, "batch"))
    return reject(*error);

  result.lhsResultAxes.resize(lhsRank);
  result.rhsResultAxes.resize(rhsRank);
  llvm::SmallBitVector lhsReduced(lhsRank);
  for (const auto &pair : result.reduction) lhsReduced.set(pair.lhs);
  for (unsigned axis = 0; axis < lhsRank; ++axis) {
    if (!lhsUsed.test(axis)) result.lhsFree.push_back(axis);
    if (lhsReduced.test(axis)) continue;
    result.lhsResultAxes[axis] = result.results.size();
    result.results.push_back({ContractionOperand::Lhs, axis});
  }
  for (const auto &pair : result.batch)
    result.rhsResultAxes[pair.rhs] = result.lhsResultAxes[pair.lhs];
  for (unsigned axis = 0; axis < rhsRank; ++axis) {
    if (rhsUsed.test(axis)) continue;
    result.rhsFree.push_back(axis);
    result.rhsResultAxes[axis] = result.results.size();
    result.results.push_back({ContractionOperand::Rhs, axis});
  }
  return result;
}

bool ContractionAxes::hasCanonicalMatrixAxes() const {
  unsigned rank = lhsResultAxes.size();
  if (rank < 2 || rhsResultAxes.size() != rank || reduction.size() != 1 ||
      batch.size() != rank - 2 || reduction.front().lhs != rank - 1 ||
      reduction.front().rhs != rank - 2)
    return false;
  for (unsigned axis = 0; axis < batch.size(); ++axis)
    if (batch[axis].lhs != axis || batch[axis].rhs != axis)
      return false;
  return true;
}

} // namespace intent

#ifndef INTENT_ANALYSIS_INTEGERRELATIONS_H
#define INTENT_ANALYSIS_INTEGERRELATIONS_H

#include "Intent/Analysis/UniformValues.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent {

using IntegerDifference = std::optional<int64_t>;
using IntegerDifferenceQuery = llvm::function_ref<IntegerDifference(mlir::Value)>;

// Fold one scalar integer operation's variation coefficient from facts supplied
// by its consumer. The supported value domain is i64 or an explicitly bound
// 64-bit index. Arithmetic values retain their modulo semantics; coefficient
// arithmetic itself is checked and must fit int64_t. Unknown is never zero.
// This query proves neither value bounds nor control/memory independence.
IntegerDifference foldIntegerDifference(
    const UniformExpression &expression, IntegerDifferenceQuery difference,
    llvm::function_ref<std::optional<int64_t>(mlir::Value)> constant,
    unsigned indexBitWidth);

} // namespace intent
#endif

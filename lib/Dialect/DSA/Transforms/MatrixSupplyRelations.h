#ifndef INTENT_DSA_TRANSFORMS_MATRIX_SUPPLY_RELATIONS_H
#define INTENT_DSA_TRANSFORMS_MATRIX_SUPPLY_RELATIONS_H

#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace intent::dsa::detail {

// Prove the complete current workset and its actual memory coordinates. The
// query neither constructs IR nor depends on a particular folded index tree.
bool hasCompleteMatrixWorkset(mlir::scf::ForOp work,
    mlir::scf::ForOp reduction, MatMulOp matrix, LoadTileOp lhs, LoadTileOp rhs);

} // namespace intent::dsa::detail

#endif

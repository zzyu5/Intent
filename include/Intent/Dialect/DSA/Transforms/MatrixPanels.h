#ifndef INTENT_DIALECT_DSA_TRANSFORMS_MATRIXPANELS_H
#define INTENT_DIALECT_DSA_TRANSFORMS_MATRIXPANELS_H

#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>

namespace intent::dsa {

struct MatrixPanelStorage {
  int64_t nram, wram;
};

// Same-dtype matrix input/packing storage, including an alternate LHS panel.
// Failure includes unsupported native geometry and unrepresentable sizes.
std::optional<MatrixPanelStorage>
matrixPanelStorage(mlir::Type element, int64_t rows, int64_t columns,
                   int64_t depth);

// Call only while forming one ordinary contraction. availableDepth is a
// complete source panel; selection never authorizes merging ordered updates.
// Existing allocations are conservatively retained alongside the new panels.
int64_t selectMatrixPanelDepth(mlir::func::FuncOp function,
                              ConfigurationAttr config, mlir::Type element,
                              int64_t rows, int64_t columns,
                              int64_t availableDepth, int64_t baselineDepth,
                              unsigned products = 1);

struct StreamedMatrixPanel {
  int64_t depth, sliceDepth, sliceColumns;
  bool pipeline;
};

// A complete WRAM RHS prepared through bounded NRAM slices. Its consumers
// retain one or two full-K LHS slots; both SRAM supply slots are reserved.
std::optional<StreamedMatrixPanel>
selectStreamedMatrixPanel(mlir::func::FuncOp function, ConfigurationAttr config,
                         mlir::Type element, int64_t rows, int64_t columns,
                         int64_t depth);

} // namespace intent::dsa
#endif

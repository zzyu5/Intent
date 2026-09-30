#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace intent::weft_provider {

// A current-IR descriptor proof, consumed while lowering the defining view.
// A missing source axis inserts a unit dimension; unreferenced source axes
// must also be units. No allocation, memory read, or layout conversion occurs.
struct AxisView {
  mlir::Value source;
  llvm::SmallVector<std::optional<unsigned>> sourceAxes;
};

bool isAxisView(mlir::Operation *operation);
mlir::FailureOr<AxisView> queryAxisView(mlir::Value value);

// The native task ABI represents contiguous storage, not arbitrary memref
// descriptors. Keep pure view definitions inside the task and capture their
// storage roots and scalar operands before taking provider analysis snapshots.
mlir::LogicalResult reifyTaskViewCaptures(mlir::func::FuncOp function);

} // namespace intent::weft_provider

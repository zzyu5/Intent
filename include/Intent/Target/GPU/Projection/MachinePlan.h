#ifndef INTENT_TARGET_GPU_PROJECTION_MACHINEPLAN_H
#define INTENT_TARGET_GPU_PROJECTION_MACHINEPLAN_H

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"

namespace intent::gpu {

struct MachinePlanIndex {
  intent::plan::DeviceOp device;
  intent::plan::ProgramOp program;
  llvm::SmallVector<intent::plan::AxisOp> axes;
  llvm::SmallVector<intent::plan::StorageOp> storage;
  llvm::SmallVector<intent::plan::PaddingOp> paddings;
  llvm::SmallVector<intent::plan::TransferOp> transfers;
  llvm::SmallVector<intent::plan::ReductionOp> reductions;
  llvm::SmallVector<intent::plan::PointwiseOp> pointwise;
  llvm::SmallVector<intent::plan::ContractOp> contracts;
  llvm::SmallVector<intent::plan::StreamOp> streams;
  llvm::SmallVector<intent::plan::RaggedOp> ragged;
  llvm::SmallVector<intent::plan::StageOp> stages;
  llvm::SmallVector<intent::plan::AtomicOp> atomics;
};

mlir::FailureOr<MachinePlanIndex>
indexMachinePlan(intent::plan::RealizationOp realization);

mlir::FailureOr<intent::plan::AutotuneOp>
indexMachineSearch(intent::plan::SearchSpaceOp searchSpace);

mlir::StringAttr stringAttr(mlir::OpBuilder &builder, llvm::StringRef value);
mlir::ArrayAttr stringArrayAttr(mlir::OpBuilder &builder,
                                llvm::ArrayRef<llvm::StringRef> values);

} // namespace intent::gpu

#endif

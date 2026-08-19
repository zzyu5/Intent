#ifndef INTENT_DIALECT_PLAN_IR_PLANOPS_H
#define INTENT_DIALECT_PLAN_IR_PLANOPS_H

#include "Intent/Dialect/Plan/IR/PlanDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Intent/Dialect/Plan/IR/PlanOps.h.inc"

namespace intent::plan {

mlir::LogicalResult verifyPaddingFields(mlir::Operation *operation,
                                        int64_t value,
                                        llvm::ArrayRef<int64_t> tensorAxes,
                                        llvm::ArrayRef<int64_t> domainNodes,
                                        llvm::StringRef fill);
mlir::FailureOr<mlir::func::FuncOp> getPhysicalEntry(ProgramOp program);
mlir::LogicalResult verifyGpuProgram(ProgramOp program);
mlir::LogicalResult verifyGpuSearchSpace(SearchSpaceOp searchSpace);

} // namespace intent::plan

#endif

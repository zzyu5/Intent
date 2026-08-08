#ifndef INTENT_LIB_TARGET_TRITON_REALIZATION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_TRITON_REALIZATION_SUPPORT_MODEL_H

#include "Intent/Target/Common/Realization/Model.h"
#include "Intent/Target/Triton/Config/Target.h"
#include "mlir/IR/Builders.h"

namespace intent::triton::realization {

using OperationFacts = intent::target::OperationFacts;
using AxisDecision = intent::target::AxisDecision;

struct PolicyDecision : intent::target::ScheduleDecision {};

mlir::LogicalResult analyzeOperations(OperationFacts &facts);

mlir::FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts);

mlir::LogicalResult emitPlan(mlir::ModuleOp module,
                             const intent::triton::TargetOptions &target,
                             const OperationFacts &facts,
                             const PolicyDecision &policy);

void emitAutotuneSpace(mlir::ModuleOp module, mlir::func::FuncOp entry,
                       const PolicyDecision &policy, mlir::OpBuilder &builder);

} // namespace intent::triton::realization

#endif

#ifndef INTENT_LIB_TARGET_CUTILE_REALIZATION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_CUTILE_REALIZATION_SUPPORT_MODEL_H

#include "Intent/Target/Common/Realization/Model.h"
#include "Intent/Target/CuTile/Config/Target.h"
#include "mlir/IR/Builders.h"

namespace intent::cutile::realization {

using OperationFacts = intent::target::OperationFacts;
using AxisDecision = intent::target::AxisDecision;

struct PolicyDecision : intent::target::ScheduleDecision {
  int64_t groupSize = 1;
  int64_t fixedOccupancy = 0;
};

mlir::LogicalResult analyzeOperations(OperationFacts &facts);

mlir::FailureOr<PolicyDecision> decidePolicy(const OperationFacts &facts);

mlir::LogicalResult emitPlan(mlir::ModuleOp module,
                             const intent::cutile::TargetOptions &target,
                             const OperationFacts &facts,
                             const PolicyDecision &policy);

mlir::LogicalResult
emitAutotuneSpace(mlir::ModuleOp module, mlir::func::FuncOp entry,
                  const intent::cutile::TargetOptions &target,
                  const PolicyDecision &policy, mlir::OpBuilder &builder);

} // namespace intent::cutile::realization

#endif

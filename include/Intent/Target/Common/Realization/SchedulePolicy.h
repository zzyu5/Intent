#ifndef INTENT_TARGET_COMMON_REALIZATION_SCHEDULEPOLICY_H
#define INTENT_TARGET_COMMON_REALIZATION_SCHEDULEPOLICY_H

#include "Intent/Target/Common/Realization/Model.h"
#include "mlir/Support/LLVM.h"

namespace intent::target {

mlir::FailureOr<ScheduleDecision>
decideGpuSchedule(const KernelFacts &facts);

} // namespace intent::target

#endif

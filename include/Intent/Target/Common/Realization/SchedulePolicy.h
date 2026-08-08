#ifndef INTENT_TARGET_COMMON_REALIZATION_SCHEDULEPOLICY_H
#define INTENT_TARGET_COMMON_REALIZATION_SCHEDULEPOLICY_H

#include "Intent/Target/Common/Realization/Model.h"
#include "mlir/Support/LLVM.h"

namespace intent::target {

struct ScheduleRules {
  llvm::StringRef targetName;
  llvm::StringRef raggedMemberTile;
  llvm::StringRef streamProgramTile;
  llvm::StringRef streamTile;
  llvm::ArrayRef<llvm::StringRef> programTiles;
  llvm::StringRef firstReductionTile;
  llvm::StringRef extraReductionTile;
  llvm::StringRef vectorTile;
  llvm::StringRef persistentMapping;
};

mlir::FailureOr<ScheduleDecision>
decideSchedule(const KernelFacts &facts, const ScheduleRules &rules);

} // namespace intent::target

#endif

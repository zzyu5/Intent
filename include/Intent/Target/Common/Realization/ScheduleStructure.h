#ifndef INTENT_TARGET_COMMON_REALIZATION_SCHEDULESTRUCTURE_H
#define INTENT_TARGET_COMMON_REALIZATION_SCHEDULESTRUCTURE_H

#include "Intent/Target/Common/Realization/KernelFacts.h"

namespace intent::target {

struct ScheduleStructure {
  llvm::SmallVector<mlir::Operation *> programDomains;
  llvm::DenseSet<mlir::Operation *> tiledProgramDomains;
  llvm::SmallVector<mlir::Operation *> vectorDomains;
  llvm::SmallVector<mlir::Operation *> streamedReductionDomains;
  mlir::Operation *programRoot = nullptr;
};

mlir::FailureOr<ScheduleStructure>
analyzeScheduleStructure(const KernelFacts &facts);

mlir::FailureOr<std::string>
sourceDimensionSymbol(mlir::Operation &domain, const KernelFacts &facts);

} // namespace intent::target

#endif

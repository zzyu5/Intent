#ifndef INTENT_TARGET_COMMON_REALIZATION_SCHEDULESTRUCTURE_H
#define INTENT_TARGET_COMMON_REALIZATION_SCHEDULESTRUCTURE_H

#include "Intent/Target/Common/Realization/KernelFacts.h"

namespace intent::target {

struct RaggedOwnership {
  mlir::Operation *relation;
  mlir::Operation *outerDomain;
  llvm::SmallVector<mlir::Operation *> memberDomains;
};

struct ContractionStage {
  mlir::Operation *contraction;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
};

struct ScheduleStructure {
  llvm::SmallVector<mlir::Operation *> programDomains;
  llvm::DenseSet<mlir::Operation *> tiledProgramDomains;
  llvm::SmallVector<mlir::Operation *> vectorDomains;
  llvm::SmallVector<mlir::Operation *> contractionDomains;
  llvm::SmallVector<mlir::Operation *> orderedStreamDomains;
  llvm::SmallVector<RaggedOwnership> raggedOwnerships;
  llvm::SmallVector<mlir::Operation *> scatterWrites;
  mlir::Operation *programRoot = nullptr;
};

mlir::FailureOr<ScheduleStructure>
analyzeScheduleStructure(const KernelFacts &facts);

mlir::FailureOr<std::string>
sourceDimensionSymbol(mlir::Operation &domain, const KernelFacts &facts);

mlir::FailureOr<llvm::SmallVector<ContractionStage>>
analyzeContractionPipeline(const KernelFacts &facts);

} // namespace intent::target

#endif

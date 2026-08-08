#ifndef INTENT_TARGET_COMMON_REALIZATION_MODEL_H
#define INTENT_TARGET_COMMON_REALIZATION_MODEL_H

#include "Intent/Target/Common/Realization/KernelFacts.h"
#include "Intent/Target/Common/Realization/ScheduleStructure.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace intent::target {

struct OperationFacts {
  explicit OperationFacts(KernelModel &kernel) : semantics(kernel) {}

  KernelFacts semantics;
  llvm::DenseMap<mlir::Operation *, std::string> primitiveRoles;
};

struct AxisDecision {
  mlir::Operation *domain;
  int64_t sourceAxis;
  std::string role;
  std::string tile;
};

struct ScheduleDecision {
  mlir::Operation *programRoot = nullptr;
  llvm::SmallVector<mlir::Operation *> stateStreams;
  llvm::SmallVector<mlir::Operation *> raggedRelations;
  llvm::SmallVector<ContractionStage> stages;
  llvm::SmallVector<AxisDecision> axes;
  std::string traversal;
  std::string mapping;
  llvm::SmallVector<int64_t> workerAxes;
  llvm::SmallVector<std::string> autotuneKeys;
  llvm::SmallVector<std::string> autotuneParameters;
  bool usesAutotuner = false;
};

} // namespace intent::target

#endif

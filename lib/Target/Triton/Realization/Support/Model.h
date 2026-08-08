#ifndef INTENT_LIB_TARGET_TRITON_REALIZATION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_TRITON_REALIZATION_SUPPORT_MODEL_H

#include "Intent/Target/Common/Realization/KernelFacts.h"
#include "Intent/Target/Triton/Config/Target.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

#include <string>

namespace intent::triton::realization {

struct OperationFacts {
  explicit OperationFacts(intent::target::KernelModel &kernel)
      : semantics(kernel) {}

  intent::target::KernelFacts semantics;
  llvm::DenseMap<mlir::Operation *, std::string> primitiveLowerings;
};

struct AxisDecision {
  mlir::Operation *domain;
  int64_t sourceAxis;
  std::string role;
  std::string tile;
};

struct PolicyDecision {
  mlir::Operation *programRoot;
  llvm::SmallVector<AxisDecision> axes;
  std::string traversal;
  std::string mapping;
  llvm::SmallVector<int64_t> workerAxes;
  llvm::SmallVector<std::string> autotuneKeys;
  bool usesAutotuner;
};

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

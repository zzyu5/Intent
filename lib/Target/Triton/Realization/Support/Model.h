#ifndef INTENT_LIB_TARGET_TRITON_REALIZATION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_TRITON_REALIZATION_SUPPORT_MODEL_H

#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Triton/Config/Target.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

#include <string>

namespace intent::triton::realization {

struct OperationFacts {
  explicit OperationFacts(intent::target::KernelModel &kernel)
      : kernel(kernel) {}

  intent::target::KernelModel &kernel;
  llvm::DenseMap<mlir::Operation *, mlir::Value> domainSources;
  llvm::DenseMap<mlir::Operation *, int64_t> domainSourceAxes;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> partitionDomains;
  llvm::SmallVector<mlir::Operation *> parallels;
  llvm::DenseMap<mlir::Operation *, std::string> primitiveLowerings;
  llvm::DenseMap<mlir::Operation *, std::string> boundaryFills;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      boundaryDomains;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *>> valueDomains;
  llvm::DenseSet<mlir::Operation *> vectorDomains;
  llvm::DenseSet<mlir::Operation *> streamedReductionDomains;
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

bool proveMaskedLaneNeutrality(mlir::Value loaded);

void emitAutotuneSpace(mlir::ModuleOp module, mlir::func::FuncOp entry,
                       const PolicyDecision &policy, mlir::OpBuilder &builder);

mlir::FailureOr<mlir::Operation *>
resolveDomain(mlir::Value indexedValue, const OperationFacts &facts,
              mlir::Operation &consumer);

} // namespace intent::triton::realization

#endif

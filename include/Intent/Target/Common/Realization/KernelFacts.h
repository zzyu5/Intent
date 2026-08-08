#ifndef INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H
#define INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H

#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace intent::target {

struct KernelFacts {
  explicit KernelFacts(KernelModel &kernel) : kernel(kernel) {}

  KernelModel &kernel;
  llvm::DenseMap<mlir::Operation *, mlir::Value> domainSources;
  llvm::DenseMap<mlir::Operation *, int64_t> domainSourceAxes;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> partitionDomains;
  llvm::SmallVector<mlir::Operation *> parallels;
  llvm::DenseMap<mlir::Operation *, std::string> boundaryFills;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      boundaryDomains;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *>> valueDomains;
  llvm::DenseSet<mlir::Operation *> vectorDomains;
  llvm::DenseSet<mlir::Operation *> streamedReductionDomains;
};

mlir::LogicalResult analyzeKernelFacts(KernelFacts &facts);

mlir::FailureOr<mlir::Operation *>
resolveDomain(mlir::Value indexedValue, const KernelFacts &facts,
              mlir::Operation &consumer);

bool proveMaskedLaneNeutrality(mlir::Value loaded);

} // namespace intent::target

#endif

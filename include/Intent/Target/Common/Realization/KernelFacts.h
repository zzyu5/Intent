#ifndef INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H
#define INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H

#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace intent::target {

struct LogicalAxis {
  mlir::Operation *domain = nullptr;
  std::string extent;

  bool operator==(const LogicalAxis &other) const {
    return domain == other.domain && extent == other.extent;
  }

  bool operator!=(const LogicalAxis &other) const { return !(*this == other); }
};

struct StateStreamFact {
  mlir::Operation *axisDomain;
  int64_t stateCount;
  std::string tile;
  mlir::Block *body;
  llvm::SmallVector<mlir::Value> initialState;
  llvm::SmallVector<mlir::Value> bodyState;
  llvm::SmallVector<mlir::Value> yieldedState;
};

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
  llvm::DenseMap<mlir::Value, llvm::SmallVector<LogicalAxis>> valueAxes;
  llvm::StringMap<LogicalAxis> axisLabels;
  llvm::DenseSet<mlir::Operation *> vectorDomains;
  llvm::DenseSet<mlir::Operation *> contractionDomains;
  llvm::DenseSet<mlir::Operation *> orderedStreamDomains;
  llvm::DenseMap<mlir::Operation *, StateStreamFact> stateStreams;
};

mlir::LogicalResult analyzeKernelFacts(KernelFacts &facts);

mlir::FailureOr<mlir::Operation *>
resolveDomain(mlir::Value indexedValue, const KernelFacts &facts,
              mlir::Operation &consumer);

bool proveMaskedLaneNeutrality(mlir::Value loaded);

} // namespace intent::target

#endif

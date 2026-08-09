#ifndef INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H
#define INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H

#include "Intent/Target/Common/Analysis/Kernel.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>
#include <optional>

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

struct RaggedRelationFact {
  mlir::Operation *relation = nullptr;
  mlir::Operation *outerSource = nullptr;
  mlir::Operation *outerDomain = nullptr;
  mlir::Value offsets;
  mlir::Value indices;
  llvm::SmallVector<mlir::Operation *> memberDomains;
};

struct RaggedMemberFact {
  mlir::Operation *relation = nullptr;
  mlir::Operation *domain = nullptr;
  mlir::Value selector;
};

struct ContractionFact {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<LogicalAxis> lhsAxes;
  llvm::SmallVector<LogicalAxis> rhsAxes;
  llvm::SmallVector<LogicalAxis> resultAxes;
  llvm::SmallVector<unsigned> lhsReductionAxes;
  llvm::SmallVector<unsigned> rhsReductionAxes;
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
  llvm::DenseMap<mlir::Operation *, RaggedRelationFact> raggedRelations;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> raggedOuterRelations;
  llvm::DenseMap<mlir::Operation *, RaggedMemberFact> raggedMembers;
  llvm::DenseMap<mlir::Value, mlir::Operation *> memberValues;
  llvm::DenseSet<mlir::Operation *> wholeViewLoads;
  llvm::DenseSet<mlir::Operation *> scatterWrites;
  llvm::DenseMap<mlir::Operation *, ContractionFact> contractions;
};

mlir::LogicalResult analyzeKernelFacts(KernelFacts &facts);

mlir::FailureOr<mlir::Operation *>
resolveDomain(mlir::Value indexedValue, const KernelFacts &facts,
              mlir::Operation &consumer);

std::optional<std::string> inferMaskedLaneFill(mlir::Value loaded);

} // namespace intent::target

#endif

#ifndef INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H
#define INTENT_TARGET_COMMON_REALIZATION_KERNELFACTS_H

#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Analysis/LogicalBuffer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>
#include <optional>
#include <utility>

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
  int64_t stateCount;
  mlir::Block *body;
  llvm::SmallVector<mlir::Value> initialState;
};

struct RaggedRelationFact {
  mlir::Operation *relation = nullptr;
  mlir::Operation *outerSource = nullptr;
  mlir::Operation *memberSource = nullptr;
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
  llvm::SmallVector<unsigned> lhsBatchAxes;
  llvm::SmallVector<unsigned> rhsBatchAxes;
};

struct SparseContractionFact {
  mlir::Operation *operation = nullptr;
  mlir::Operation *rowDomain = nullptr;
  mlir::Operation *columnDomain = nullptr;
  mlir::Operation *reductionDomain = nullptr;
};

struct LogicalBufferFact {
  mlir::Operation *owner = nullptr;
  LogicalBufferInfo info;
  bool hasDynamicAccess = false;
  bool hasUnstructuredDynamicAccess = false;
};

struct ScanFact {
  mlir::Operation *axis = nullptr;
  llvm::SmallVector<int64_t> producers;
  llvm::SmallVector<int64_t> materializedValues;
  bool scalarConsumers = false;
};

struct AccessRangeFact {
  mlir::Operation *transfer = nullptr;
  mlir::Operation *axis = nullptr;
  unsigned sourceAxis = 0;
  int64_t divisor = 1;
  int64_t offset = 0;
};

struct CountPartitionFact {
  mlir::Operation *partition = nullptr;
  mlir::Operation *iteration = nullptr;
  mlir::Operation *domain = nullptr;
  mlir::Value count;
  mlir::Value partArgument;
  mlir::Value regionArgument;
};

enum class TensorIndexingKind {
  none,
  structured,
  compact,
  dataDependent,
};

struct KernelFacts {
  explicit KernelFacts(KernelModel &kernel) : kernel(kernel) {}

  KernelModel &kernel;
  llvm::DenseMap<mlir::Operation *, mlir::Value> domainSources;
  llvm::DenseMap<mlir::Operation *, int64_t> domainSourceAxes;
  llvm::DenseMap<mlir::Operation *, int64_t> staticDomainExtents;
  llvm::DenseMap<mlir::Operation *, std::pair<int64_t, int64_t>>
      staticDomainBounds;
  llvm::DenseSet<mlir::Operation *> runtimeSequentialDomains;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> partitionDomains;
  llvm::DenseMap<mlir::Operation *, int64_t> partitionFixedExtents;
  llvm::DenseMap<mlir::Operation *, mlir::Value> partitionCounts;
  llvm::DenseMap<mlir::Value, mlir::Operation *> partitionPartArguments;
  llvm::SmallVector<CountPartitionFact> countPartitions;
  llvm::SmallVector<mlir::Operation *> parallels;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      parallelDomains;
  llvm::DenseMap<mlir::Operation *, std::string> boundaryFills;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      boundaryDomains;
  llvm::DenseMap<mlir::Operation *, TensorIndexingKind> tensorIndexing;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<LogicalAxis>> valueAxes;
  llvm::DenseMap<mlir::Value, LogicalAxis> regionArgumentAxes;
  llvm::StringMap<LogicalAxis> axisLabels;
  llvm::DenseSet<mlir::Operation *> vectorDomains;
  llvm::DenseSet<mlir::Operation *> reductionDomains;
  llvm::DenseSet<mlir::Operation *> contractionDomains;
  llvm::DenseSet<mlir::Operation *> scaledStreamDomains;
  llvm::DenseSet<mlir::Operation *> orderedDomains;
  llvm::DenseSet<mlir::Operation *> serialLoopDomains;
  llvm::DenseMap<mlir::Operation *, ScanFact> scans;
  llvm::DenseMap<mlir::Operation *, int64_t> orderedStreamFixedExtents;
  llvm::DenseMap<mlir::Operation *, StateStreamFact> stateStreams;
  llvm::DenseMap<mlir::Operation *, RaggedRelationFact> raggedRelations;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> raggedOuterRelations;
  llvm::DenseMap<mlir::Operation *, RaggedMemberFact> raggedMembers;
  llvm::DenseMap<mlir::Value, mlir::Operation *> memberValues;
  llvm::DenseSet<mlir::Operation *> wholeViewLoads;
  llvm::DenseSet<mlir::Operation *> scatterWrites;
  llvm::DenseMap<mlir::Operation *, ContractionFact> contractions;
  llvm::DenseMap<mlir::Operation *, SparseContractionFact> sparseContractions;
  llvm::DenseMap<mlir::Operation *, LogicalBufferFact> logicalBuffers;
  llvm::SmallVector<AccessRangeFact> accessRanges;
};

mlir::LogicalResult analyzeKernelFacts(KernelFacts &facts);

mlir::FailureOr<mlir::Operation *>
resolveDomain(mlir::Value indexedValue, const KernelFacts &facts,
              mlir::Operation &consumer);

bool hasNonnegativeIntegerOperands(mlir::Operation &operation,
                                   const KernelFacts &facts);

TensorIndexingKind tensorIndexingKind(mlir::Operation &operation,
                                      const KernelFacts &facts);

std::optional<std::string> inferMaskedLaneFill(mlir::Value loaded);

std::optional<std::string> inferValuePadding(
    mlir::Value value, const KernelFacts &facts,
    const llvm::DenseMap<mlir::Value, std::string> &assumedPadding);

} // namespace intent::target

#endif

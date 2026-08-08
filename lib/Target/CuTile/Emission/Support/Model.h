#ifndef INTENT_LIB_TARGET_CUTILE_EMISSION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_CUTILE_EMISSION_SUPPORT_MODEL_H

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace intent::cutile::emission {

struct RealizationIndex {
  plan::TargetOp target;
  plan::ProgramOp program;
  plan::LaunchOp launch;
  llvm::DenseMap<int64_t, plan::AxisOp> axes;
  llvm::StringMap<plan::AxisOp> axesByRole;
  llvm::DenseMap<int64_t, plan::StorageOp> storage;
  llvm::DenseMap<int64_t, plan::LayoutOp> layouts;
  llvm::DenseMap<int64_t, plan::ReductionOp> reductions;
  llvm::DenseMap<int64_t, plan::PointwiseOp> pointwise;
  llvm::DenseMap<int64_t, plan::ContractOp> contracts;
  llvm::DenseMap<int64_t, plan::BoundaryOp> boundaries;
};

struct SearchIndex {
  plan::AutotuneOp autotune;
  llvm::SmallVector<plan::ConfigOp> configs;
};

struct ABIView {
  const intent::target::ABIArgument *argument;
  intent::ViewType view;
  mlir::RankedTensorType tensor;
  llvm::SmallVector<std::string> shape;
};

class SourceEmitter {
public:
  SourceEmitter(intent::target::KernelModel kernel,
                intent::plan::RealizationOp realization,
                intent::plan::SearchSpaceOp searchSpace,
                RealizationIndex planIndex, SearchIndex searchIndex,
                llvm::raw_ostream &output);

  mlir::LogicalResult emit();
  mlir::LogicalResult emitConstant(mlir::Operation &operation);
  mlir::LogicalResult enterParallel(mlir::Operation &operation);
  mlir::LogicalResult leaveParallel(mlir::Operation &operation);
  mlir::LogicalResult emitLoad(mlir::Operation &operation);
  mlir::LogicalResult emitReduction(mlir::Operation &operation);
  mlir::LogicalResult emitBroadcast(mlir::Operation &operation);
  mlir::LogicalResult emitUnary(mlir::Operation &operation);
  mlir::LogicalResult emitBinary(mlir::Operation &operation);
  mlir::LogicalResult emitCast(mlir::Operation &operation);
  mlir::LogicalResult emitContract(mlir::Operation &operation);
  mlir::LogicalResult emitStore(mlir::Operation &operation);

private:
  mlir::LogicalResult indexABI();
  mlir::LogicalResult resolvePhysicalBindings();
  void emitImports();
  mlir::LogicalResult emitKernelHeader();
  mlir::LogicalResult emitWrapper();

  mlir::FailureOr<llvm::StringRef> lookupValue(mlir::Operation &consumer,
                                                unsigned operandIndex);
  mlir::FailureOr<ABIView *> lookupView(mlir::Value value,
                                       mlir::Operation &consumer);
  mlir::FailureOr<mlir::Operation *>
  resolveDomain(mlir::Value indexedValue, mlir::Operation &consumer);
  mlir::FailureOr<plan::AxisOp> resolveAxis(mlir::Value indexedValue,
                                           mlir::Operation &consumer);
  mlir::FailureOr<std::string> dimensionName(mlir::Operation &domain);
  mlir::FailureOr<std::string>
  indexTuple(mlir::Operation &operation, bool reductionLoop);
  mlir::FailureOr<std::string> tileShape(mlir::Operation &operation);
  std::string dtypeName(mlir::Type type, mlir::Operation &consumer);
  std::string makeResultName(mlir::Operation &operation, unsigned index);
  std::string makeRegionArgumentName(mlir::Operation &operation,
                                     unsigned argumentIndex);
  std::string uniqueName(llvm::StringRef candidate, int64_t node);
  void line(llvm::StringRef text);

  intent::target::KernelModel kernel;
  intent::plan::RealizationOp realization;
  intent::plan::SearchSpaceOp searchSpace;
  RealizationIndex planIndex;
  SearchIndex searchIndex;
  llvm::raw_ostream &output;
  llvm::SmallVector<ABIView> views;
  llvm::DenseMap<mlir::Value, unsigned> viewPositions;
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  llvm::DenseMap<mlir::Value, mlir::Operation *> deferredLoads;
  llvm::StringSet<> usedNames;
  llvm::StringMap<std::string> dimensionOwners;
  llvm::StringMap<std::string> roleDimensions;
  llvm::SmallVector<std::string> dimensionOrder;
  mlir::Operation *programRoot = nullptr;
  mlir::Operation *vectorDomain = nullptr;
  ABIView *fixedOutput = nullptr;
  std::string kernelName;
  std::string programIndex;
  std::string vectorIndex;
  unsigned indentation = 1;
};

mlir::LogicalResult
registerEmissionHandlers(intent::target::OperationHandlerRegistry &registry,
                         SourceEmitter &emitter);

mlir::FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization);

mlir::FailureOr<SearchIndex>
indexSearchSpace(intent::plan::SearchSpaceOp searchSpace);

mlir::LogicalResult emitRealizedKernelSource(
    intent::target::KernelModel kernel,
    intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, llvm::raw_ostream &output);

} // namespace intent::cutile::emission

#endif

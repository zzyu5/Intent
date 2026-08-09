#ifndef INTENT_LIB_TARGET_TRITON_EMISSION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_TRITON_EMISSION_SUPPORT_MODEL_H

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Emission/Lifecycle.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace intent::triton::emission {

struct RealizationIndex {
  plan::TargetOp target;
  plan::ProgramOp program;
  llvm::DenseMap<int64_t, plan::AxisOp> axes;
  llvm::StringMap<plan::AxisOp> axesByRole;
  llvm::DenseMap<int64_t, plan::StorageOp> storage;
  llvm::DenseMap<int64_t, plan::ReductionOp> reductions;
  llvm::DenseMap<int64_t, plan::PointwiseOp> pointwise;
  llvm::DenseMap<int64_t, plan::ContractOp> contracts;
  llvm::DenseMap<int64_t, plan::StreamOp> streams;
  llvm::DenseMap<int64_t, plan::BoundaryOp> boundaries;
  llvm::SmallVector<plan::RaggedOp> ragged;
  llvm::SmallVector<plan::StageOp> stages;
  llvm::DenseMap<int64_t, plan::AtomicOp> atomics;
};

struct SearchIndex {
  plan::AutotuneOp autotune;
};

struct ABIView {
  const intent::target::ABIArgument *argument;
  intent::ViewType view;
  mlir::RankedTensorType tensor;
  std::string pointer;
  llvm::SmallVector<std::string> strides;
  llvm::SmallVector<std::string> shape;
};

struct ABIScalar {
  const intent::target::ABIArgument *argument;
  mlir::Type type;
  std::string name;
};

class SourceEmitter : public intent::target::TargetSourceEmitter {
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
  mlir::LogicalResult emitIndices(mlir::Operation &operation);
  mlir::LogicalResult emitReduction(mlir::Operation &operation);
  mlir::LogicalResult emitBroadcast(mlir::Operation &operation);
  mlir::LogicalResult emitUnary(mlir::Operation &operation);
  mlir::LogicalResult emitBinary(mlir::Operation &operation);
  mlir::LogicalResult emitMask(mlir::Operation &operation);
  mlir::LogicalResult emitCast(mlir::Operation &operation);
  mlir::LogicalResult emitFull(mlir::Operation &operation);
  mlir::LogicalResult emitZeros(mlir::Operation &operation);
  mlir::LogicalResult emitGather(mlir::Operation &operation);
  mlir::LogicalResult emitMembers(mlir::Operation &operation);
  mlir::LogicalResult enterStateStream(mlir::Operation &operation);
  mlir::LogicalResult leaveStateStream(mlir::Operation &operation);
  mlir::LogicalResult emitContract(mlir::Operation &operation);
  mlir::LogicalResult emitStore(mlir::Operation &operation);
  mlir::LogicalResult emitUniqueStore(mlir::Operation &operation);
  mlir::LogicalResult emitAtomic(mlir::Operation &operation);
  bool selectOperation(mlir::Operation &operation);

private:
  mlir::LogicalResult prepare() override;
  mlir::LogicalResult registerOperationHandlers(
      intent::target::OperationHandlerRegistry &registry) override;
  mlir::func::FuncOp entry() const override { return kernel.entry; }
  llvm::StringRef stage() const override { return "Triton target emission"; }
  llvm::raw_ostream &stream() override { return output; }

  mlir::LogicalResult indexABI();
  mlir::LogicalResult resolvePhysicalBindings();
  void emitImports() override;
  mlir::LogicalResult emitKernelHeader() override;
  mlir::LogicalResult emitWrapper() override;

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
  indexExpression(plan::AxisOp axis, bool store,
                  unsigned tensorAxis, unsigned tensorRank,
                  mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  emitPointerExpression(mlir::Operation &operation, ABIView &view, bool store);
  mlir::FailureOr<std::string>
  emitMaskExpression(mlir::Operation &operation, bool store);
  mlir::FailureOr<std::string>
  emitTensorShape(mlir::Operation &operation, unsigned resultIndex);
  mlir::FailureOr<unsigned> emittedTensorRank(mlir::Operation &operation,
                                             bool store);
  std::string broadcastIndex(llvm::StringRef base, unsigned axis,
                             unsigned rank);
  mlir::LogicalResult prepareRaggedMetadata();
  mlir::LogicalResult prepareRaggedStages();
  void collectStageValue(mlir::Value value, unsigned stage,
                         const llvm::DenseSet<mlir::Value> &inputs,
                         llvm::DenseSet<mlir::Value> &visited);
  void stageLine(unsigned stage, llvm::StringRef text,
                 unsigned indent = 1);
  void bindResult(mlir::Operation &operation, unsigned index,
                  llvm::StringRef name);
  bool hasTraversal(llvm::StringRef traversal);
  bool usesRaggedOwnership();
  bool usesRaggedOrderedTraversal();
  bool usesStagedEmission();

  std::string makeResultName(mlir::Operation &operation, unsigned index);
  std::string makeRegionArgumentName(mlir::Operation &operation,
                                     unsigned argumentIndex);
  std::string programGrid(llvm::StringRef extent);
  std::string uniqueName(llvm::StringRef candidate, int64_t node);
  void line(llvm::StringRef text);

  intent::target::KernelModel kernel;
  intent::plan::RealizationOp realization;
  intent::plan::SearchSpaceOp searchSpace;
  RealizationIndex planIndex;
  SearchIndex searchIndex;
  llvm::raw_ostream &output;
  llvm::SmallVector<ABIView> views;
  llvm::SmallVector<ABIScalar> scalars;
  llvm::DenseMap<mlir::Value, unsigned> viewPositions;
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  llvm::DenseMap<mlir::Value, mlir::Operation *> deferredLoads;
  llvm::StringSet<> usedNames;
  llvm::StringMap<std::string> dimensionOwners;
  llvm::StringMap<std::string> roleDimensions;
  llvm::StringMap<std::string> regionTiles;
  llvm::SmallVector<std::string> dimensionOrder;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      streamCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned>>
      operationStages;
  llvm::DenseMap<mlir::Value, unsigned> stageOutputOwners;
  llvm::DenseMap<mlir::Value, std::string> workspaceNames;
  llvm::SmallVector<std::string> stageBodies;
  llvm::SmallVector<unsigned> activeStages;
  llvm::DenseMap<unsigned, std::string> stageFeatureDimensions;
  llvm::DenseMap<unsigned, std::string> stageReductionDimensions;
  mlir::Operation *raggedRelation = nullptr;
  mlir::Operation *raggedOuter = nullptr;
  mlir::Operation *raggedMember = nullptr;
  mlir::Operation *membersOperation = nullptr;
  ABIView *raggedOffsets = nullptr;
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

} // namespace intent::triton::emission

#endif

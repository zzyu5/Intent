#ifndef INTENT_LIB_TARGET_TILELANG_EMISSION_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_TILELANG_EMISSION_SUPPORT_MODEL_H

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Emission/Lifecycle.h"
#include "Intent/Target/Common/Emission/SurfacePlan.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace intent::tilelang::plan {
using TargetOp = target::emission::TargetBinding;
using AxisOp = target::emission::AxisBinding;
using RegionBindingOp = intent::plan::RegionBindingOp;
using ProgramOp = target::emission::ProgramBinding;
using BlockExtentOp = target::emission::BlockExtentBinding;
using BufferOp = target::emission::BufferBinding;
using PaddingOp = target::emission::PaddingBinding;
using ReductionOp = target::emission::ReductionBinding;
using ScanOp = target::emission::ScanBinding;
using PointwiseOp = target::emission::PointwiseBinding;
using ContractOp = target::emission::ContractBinding;
using SparseContractOp = intent::plan::SparseContractOp;
using StreamOp = target::emission::StreamBinding;
using RaggedOp = target::emission::RaggedBinding;
using StageOp = target::emission::StageBinding;
using StageAxisOp = target::emission::StageAxisBinding;
using StreamAxisOp = intent::plan::StreamAxisOp;
using StreamBindingOp = intent::plan::StreamBindingOp;
using BoundaryOp = target::emission::BoundaryBinding;
using AutotuneOp = target::emission::AutotuneBinding;
} // namespace intent::tilelang::plan

namespace intent::tilelang::emission {

struct RealizationIndex {
  plan::TargetOp target;
  plan::ProgramOp program;
  llvm::StringMap<plan::BlockExtentOp> blockExtents;
  llvm::DenseMap<int64_t, plan::AxisOp> axes;
  llvm::DenseMap<int64_t, plan::RegionBindingOp> regionBindings;
  llvm::StringMap<plan::AxisOp> axesByRole;
  llvm::DenseMap<int64_t, plan::PaddingOp> paddings;
  llvm::DenseMap<int64_t, plan::BufferOp> buffers;
  llvm::DenseMap<int64_t, plan::ReductionOp> reductions;
  llvm::DenseMap<int64_t, plan::ScanOp> scans;
  llvm::DenseMap<int64_t, plan::PointwiseOp> pointwise;
  llvm::DenseMap<int64_t, plan::ContractOp> contracts;
  llvm::DenseMap<int64_t, plan::SparseContractOp> sparseContracts;
  llvm::DenseMap<int64_t, plan::StreamOp> streams;
  llvm::DenseMap<int64_t, plan::StreamBindingOp> streamBindings;
  llvm::DenseMap<int64_t, plan::BoundaryOp> boundaries;
  llvm::SmallVector<plan::RaggedOp, 0> ragged;
  llvm::SmallVector<plan::StageOp, 0> stages;
  llvm::DenseMap<int64_t, llvm::StringMap<plan::StageAxisOp>> stageAxes;
  llvm::SmallVector<plan::StreamAxisOp> streamAxes;
  target::emission::PhysicalComponents components;
  bool requiresSymmetricProgramTiles = false;
};

struct SearchIndex {
  plan::AutotuneOp autotune;
};

struct ABIView {
  const intent::target::ABIArgument *argument;
  intent::ViewType view;
  mlir::RankedTensorType tensor;
  llvm::SmallVector<std::string> shape;
};

struct ABIScalar {
  const intent::target::ABIArgument *argument;
  mlir::Type type;
  std::string name;
};

struct RaggedRuntime {
  plan::RaggedOp binding;
  mlir::Operation *relation = nullptr;
  mlir::Operation *outer = nullptr;
  llvm::SmallVector<mlir::Operation *> members;
  llvm::SmallVector<mlir::Operation *> ownedMembers;
  ABIView *offsets = nullptr;
  ABIView *indices = nullptr;
  std::string memberDimension;
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
  mlir::LogicalResult emitDimension(mlir::Operation &operation);
  mlir::LogicalResult emitExtract(mlir::Operation &operation);
  mlir::LogicalResult emitAssumeInBounds(mlir::Operation &operation);
  mlir::LogicalResult enterParallel(mlir::Operation &operation);
  mlir::LogicalResult leaveParallel(mlir::Operation &operation);
  mlir::LogicalResult enterFor(mlir::Operation &operation);
  mlir::LogicalResult leaveFor(mlir::Operation &operation);
  mlir::LogicalResult enterIf(mlir::Operation &operation);
  mlir::LogicalResult leaveIf(mlir::Operation &operation);
  mlir::LogicalResult enterWhile(mlir::Operation &operation);
  mlir::LogicalResult leaveWhile(mlir::Operation &operation);
  mlir::LogicalResult emitCondition(mlir::Operation &operation);
  mlir::LogicalResult emitYield(mlir::Operation &operation);
  mlir::LogicalResult emitBuffer(mlir::Operation &operation);
  mlir::LogicalResult emitBufferLoad(mlir::Operation &operation);
  mlir::LogicalResult emitBufferStore(mlir::Operation &operation);
  mlir::LogicalResult emitLoad(mlir::Operation &operation);
  mlir::LogicalResult emitIndices(mlir::Operation &operation);
  mlir::LogicalResult emitRandom(mlir::Operation &operation);
  mlir::LogicalResult emitReduction(mlir::Operation &operation);
  mlir::LogicalResult emitScan(mlir::Operation &operation);
  mlir::LogicalResult emitBroadcast(mlir::Operation &operation);
  mlir::LogicalResult emitUnary(mlir::Operation &operation);
  mlir::LogicalResult emitBinary(mlir::Operation &operation);
  mlir::LogicalResult emitMask(mlir::Operation &operation);
  mlir::LogicalResult emitSelect(mlir::Operation &operation);
  mlir::LogicalResult emitCast(mlir::Operation &operation);
  mlir::LogicalResult emitReshape(mlir::Operation &operation);
  mlir::LogicalResult emitTranspose(mlir::Operation &operation);
  mlir::LogicalResult emitFull(mlir::Operation &operation);
  mlir::LogicalResult emitZeros(mlir::Operation &operation);
  mlir::LogicalResult emitGather(mlir::Operation &operation);
  mlir::LogicalResult emitMembers(mlir::Operation &operation);
  mlir::LogicalResult enterStateStream(mlir::Operation &operation);
  mlir::LogicalResult leaveStateStream(mlir::Operation &operation);
  mlir::LogicalResult emitContract(mlir::Operation &operation);
  mlir::LogicalResult emitScaledContract(mlir::Operation &operation);
  mlir::LogicalResult emitSparseContract(mlir::Operation &operation);
  mlir::LogicalResult emitStore(mlir::Operation &operation);
  mlir::LogicalResult emitUniqueStore(mlir::Operation &operation);
  mlir::LogicalResult emitAtomic(mlir::Operation &operation);
  bool selectOperation(mlir::Operation &operation);

private:
  mlir::LogicalResult prepare() override;
  mlir::LogicalResult registerOperationHandlers(
      intent::target::OperationHandlerRegistry &registry) override;
  mlir::func::FuncOp entry() const override { return kernel.entry; }
  llvm::StringRef stage() const override { return "TileLang target emission"; }
  llvm::raw_ostream &stream() override { return output; }

  mlir::LogicalResult indexABI();
  mlir::LogicalResult resolvePhysicalBindings();
  mlir::LogicalResult preparePrivateWorkspaces();
  mlir::LogicalResult emitConditional(mlir::Operation &operation, bool mask);
  mlir::LogicalResult replayScanProducers(const plan::ScanOp &binding,
                                          llvm::StringRef logicalIndex);
  mlir::LogicalResult replayContractProducers(
      llvm::ArrayRef<mlir::Operation *> producers);
  mlir::FailureOr<std::string>
  scanWorkspaceIndex(const plan::ScanOp &binding,
                     llvm::StringRef logicalIndex,
                     mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  scanMaterializedIndex(mlir::Value value, llvm::StringRef logicalIndex,
                        mlir::Operation &consumer);
  void emitImports() override;
  mlir::LogicalResult emitHelpers() override;
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
  bool usesMatrixContraction() const;
  std::string addressIndex(llvm::StringRef expression) const;
  std::string logicalExtent(llvm::StringRef extent) const;
  std::string physicalExtent(llvm::StringRef logicalExtent) const;
  mlir::FailureOr<std::string>
  transferPhysicalExtentFill(mlir::Operation &operation);
  mlir::FailureOr<std::string> accessIndices(mlir::Operation &operation);
  mlir::FailureOr<bool>
  isTraversalRegionArgument(mlir::Value value, mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  structuredIndexExpression(mlir::Value value, mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  privateWorkspaceIndex(mlir::Operation &operation);
  mlir::FailureOr<std::string>
  elementAccessIndices(mlir::Operation &operation,
                       llvm::ArrayRef<std::string> tileIndices);
  mlir::FailureOr<std::string>
  elementBoundsPredicate(mlir::Operation &operation,
                         llvm::ArrayRef<std::string> tileIndices,
                         bool physicalOnly = false,
                         bool includePhysical = true);
  mlir::FailureOr<std::string>
  scalarTransferPredicate(mlir::Operation &operation,
                          const plan::BoundaryOp &boundary);
  mlir::FailureOr<std::string>
  wholeTileBoundsPredicate(mlir::Operation &operation,
                           llvm::ArrayRef<std::string> tileExtents);
  mlir::FailureOr<std::string>
  tileFitsViewPredicate(mlir::Operation &operation,
                        llvm::ArrayRef<std::string> tileExtents);
  mlir::FailureOr<std::string>
  tileBoundsPredicate(mlir::Operation &operation,
                      llvm::ArrayRef<std::string> tileExtents,
                      bool includeBase);
  mlir::FailureOr<std::string>
  elementValidityPredicate(llvm::ArrayRef<int64_t> tensorAxes,
                           llvm::ArrayRef<int64_t> domainNodes,
                           llvm::ArrayRef<std::string> elementIndices,
                           mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  padElementExpression(mlir::Value value, llvm::StringRef expression,
                       llvm::ArrayRef<std::string> elementIndices,
                       mlir::Operation &consumer);
  mlir::FailureOr<llvm::SmallVector<std::string>>
  tensorExtents(mlir::Operation &operation, unsigned resultIndex,
                bool physical = true);
  mlir::FailureOr<std::string> tensorShape(mlir::Operation &operation,
                                           unsigned resultIndex);
  mlir::FailureOr<std::string> tensorElement(mlir::Value value,
                                             llvm::ArrayRef<std::string> indices,
                                             mlir::Operation &consumer);
  mlir::FailureOr<std::string> allocateResult(mlir::Operation &operation,
                                              unsigned resultIndex,
                                              llvm::StringRef space);
  std::string dtypeName(mlir::Type type, mlir::Operation &consumer);
  void bindResult(mlir::Operation &operation, unsigned index,
                  llvm::StringRef name);

  mlir::LogicalResult prepareRaggedMetadata();
  mlir::LogicalResult prepareRaggedStages();
  void stageLine(unsigned stage, llvm::StringRef text,
                 unsigned indent = 3);
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
  llvm::SmallVector<ABIScalar> scalars;
  llvm::DenseMap<mlir::Value, unsigned> viewPositions;
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  llvm::DenseMap<mlir::Value, std::string> assumedIndexNames;
  llvm::DenseMap<mlir::Value, mlir::Operation *> deferredLoads;
  llvm::DenseMap<mlir::Operation *, target::emission::DeferredContractReplay>
      deferredContractReplays;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
      deferredContractProducerOwners;
  mlir::Operation *activeDeferredContract = nullptr;
  llvm::StringSet<> usedNames;
  llvm::StringMap<std::string> dimensionOwners;
  llvm::StringMap<std::string> roleDimensions;
  llvm::DenseMap<int64_t, std::string> axisDimensions;
  llvm::StringMap<std::string> regionTiles;
  llvm::DenseMap<int64_t, std::string> axisIndices;
  llvm::DenseMap<int64_t, std::string> programBlocks;
  llvm::SmallVector<std::string> dimensionOrder;
  llvm::SmallVector<std::pair<std::string, std::string>> blockExtentConstants;
  llvm::StringSet<> exactBulkExtents;
  llvm::DenseMap<mlir::Operation *, std::string> streamOuterAxisIndices;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      streamCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      loopCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      whileCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>> ifResults;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<std::string>> scalarBuffers;
  llvm::DenseMap<mlir::Value, std::string> localBuffers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned>>
      operationStages;
  llvm::DenseMap<mlir::Value, unsigned> stageOutputOwners;
  llvm::DenseMap<mlir::Value, std::string> workspaceNames;
  llvm::DenseMap<mlir::Value, plan::ScanOp> scanResults;
  llvm::DenseMap<mlir::Value, plan::ScanOp> scanMaterializedValues;
  llvm::DenseMap<int64_t, std::string> scanExtents;
  llvm::DenseMap<mlir::Operation *, int64_t> scanProducerOwners;
  int64_t activeScanReplay = -1;
  llvm::SmallVector<mlir::Operation *> privateWorkspaceBuffers;
  llvm::SmallVector<std::string> stageBodies;
  llvm::SmallVector<unsigned> activeStages;
  llvm::DenseMap<unsigned, std::string> stageFeatureDimensions;
  llvm::DenseMap<unsigned, std::string> stageMemberDimensions;
  llvm::DenseMap<unsigned, std::string> stageReductionDimensions;
  llvm::DenseMap<unsigned, std::string> stageFeatureTiles;
  llvm::DenseMap<unsigned, std::string> stageMemberTiles;
  llvm::DenseMap<unsigned, std::string> stageReductionTiles;
  llvm::DenseMap<unsigned, int64_t> stageFeatureWorkers;
  llvm::DenseMap<unsigned, int64_t> stageMemberWorkers;
  llvm::SmallVector<RaggedRuntime, 0> raggedRuntimes;
  llvm::DenseMap<int64_t, unsigned> raggedRuntimeByRelation;
  llvm::DenseMap<int64_t, llvm::SmallVector<unsigned>> raggedRuntimesByAxis;
  llvm::DenseMap<unsigned, unsigned> stageRaggedRuntime;
  mlir::Operation *programRoot = nullptr;
  mlir::Operation *vectorDomain = nullptr;
  ABIView *fixedOutput = nullptr;
  bool tuneRowLaunch = false;
  std::string kernelName;
  unsigned indentation = 3;
};

mlir::LogicalResult
registerEmissionHandlers(intent::target::OperationHandlerRegistry &registry,
                         SourceEmitter &emitter);

mlir::FailureOr<RealizationIndex>
indexRealization(intent::plan::RealizationOp realization,
                 const intent::target::KernelModel &kernel);

mlir::FailureOr<SearchIndex>
indexSearchSpace(intent::plan::SearchSpaceOp searchSpace);

mlir::LogicalResult emitRealizedKernelSource(
    intent::target::KernelModel kernel,
    intent::plan::RealizationOp realization,
    intent::plan::SearchSpaceOp searchSpace, llvm::raw_ostream &output);

} // namespace intent::tilelang::emission

#endif

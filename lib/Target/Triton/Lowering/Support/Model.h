#ifndef INTENT_LIB_TARGET_TRITON_LOWERING_SUPPORT_MODEL_H
#define INTENT_LIB_TARGET_TRITON_LOWERING_SUPPORT_MODEL_H

#include "Intent/Dialect/Intent/IR/IntentTypes.h"
#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Common/Analysis/Kernel.h"
#include "Intent/Target/Common/Lowering/Lifecycle.h"
#include "Intent/Target/Common/Lowering/ProgramAnalysis.h"
#include "Intent/Target/Common/Traversal/OperationRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace intent::triton::plan {
using TargetOp = target::lowering::TargetBinding;
using AxisOp = target::lowering::AxisBinding;
using RegionBindingOp = intent::plan::RegionBindingOp;
using PartitionBindingOp = intent::plan::PartitionBindingOp;
using ProgramOp = target::lowering::ProgramBinding;
using BlockExtentOp = target::lowering::BlockExtentBinding;
using BufferOp = target::lowering::BufferBinding;
using PaddingOp = target::lowering::PaddingBinding;
using ReductionOp = target::lowering::ReductionBinding;
using ScanOp = target::lowering::ScanBinding;
using PointwiseOp = target::lowering::PointwiseBinding;
using ContractOp = target::lowering::ContractBinding;
using SparseContractOp = intent::plan::SparseContractOp;
using StreamOp = target::lowering::StreamBinding;
using RaggedOp = target::lowering::RaggedBinding;
using StreamAxisOp = intent::plan::StreamAxisOp;
using StreamBindingOp = intent::plan::StreamBindingOp;
using BoundaryOp = target::lowering::BoundaryBinding;
using AutotuneOp = target::lowering::AutotuneBinding;
} // namespace intent::triton::plan

namespace intent::triton::lowering {

struct PhysicalProgramIndex {
  plan::TargetOp target;
  plan::ProgramOp program;
  llvm::StringMap<plan::BlockExtentOp> blockExtents;
  llvm::DenseMap<int64_t, plan::AxisOp> axes;
  llvm::DenseMap<int64_t, plan::RegionBindingOp> regionBindings;
  llvm::SmallVector<plan::PartitionBindingOp> partitionBindings;
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
  llvm::SmallVector<plan::StreamAxisOp> streamAxes;
  target::lowering::PhysicalComponents components;
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
  llvm::SmallVector<bool> dynamicStrides;
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
};

class ProgramMaterializer : public intent::target::TargetProgramMaterializer {
public:
  ProgramMaterializer(intent::target::KernelModel kernel,
                intent::plan::ProgramOp physicalProgram,
                intent::plan::SearchSpaceOp searchSpace,
                PhysicalProgramIndex planIndex, SearchIndex searchIndex,
                llvm::raw_ostream &output);

  mlir::LogicalResult materialize();

  mlir::LogicalResult emitConstant(mlir::Operation &operation);
  mlir::LogicalResult emitDimension(mlir::Operation &operation);
  mlir::LogicalResult emitExtract(mlir::Operation &operation);
  mlir::LogicalResult emitRegionEnd(mlir::Operation &operation);
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
  mlir::LogicalResult emitBitcast(mlir::Operation &operation);
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
  mlir::LogicalResult emitStore(mlir::Operation &operation);
  mlir::LogicalResult emitUniqueStore(mlir::Operation &operation);
  mlir::LogicalResult emitAtomic(mlir::Operation &operation);
  mlir::LogicalResult emitAtomicCas(mlir::Operation &operation);
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
  mlir::LogicalResult preparePrivateWorkspaces();
  mlir::LogicalResult emitConditional(mlir::Operation &operation, bool mask);
  mlir::LogicalResult replayScanProducers(const plan::ScanOp &binding,
                                          llvm::StringRef offsets);
  mlir::LogicalResult replayContractProducers(
      llvm::ArrayRef<mlir::Operation *> producers);
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
  mlir::FailureOr<plan::AxisOp>
  packedScalarAxis(mlir::Value indexedValue, mlir::Operation &consumer);
  mlir::FailureOr<llvm::SmallVector<unsigned>>
  packedScalarInsertions(mlir::Operation &operation);
  mlir::FailureOr<std::string> dimensionName(mlir::Operation &domain);
  std::string addressIndex(llvm::StringRef expression) const;
  std::string physicalExtent(llvm::StringRef logicalExtent) const;
  bool usesScaledContraction() const;
  mlir::FailureOr<std::string> physicalAxisTile(plan::AxisOp axis);
  mlir::FailureOr<std::string>
  transferPhysicalExtentFill(mlir::Operation &operation);
  mlir::FailureOr<std::string>
  indexExpression(plan::AxisOp axis, bool store,
                  unsigned tensorAxis, unsigned tensorRank,
                  mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  emitPointerExpression(mlir::Operation &operation, ABIView &view, bool store);
  mlir::FailureOr<std::string>
  privateWorkspacePointer(mlir::Operation &operation);
  mlir::FailureOr<std::string>
  scanWorkspacePointer(mlir::Value result, const plan::ScanOp &binding,
                       llvm::StringRef logicalIndex,
                       mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  scanMaterializedPointer(mlir::Value value, llvm::StringRef logicalIndex,
                          mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  emitMaskExpression(mlir::Operation &operation, bool store);
  mlir::FailureOr<std::string>
  emitValidityExpression(llvm::ArrayRef<int64_t> tensorAxes,
                         llvm::ArrayRef<int64_t> domainNodes,
                         mlir::Value value, mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  padExpression(mlir::Value value, llvm::StringRef expression,
                mlir::Operation &consumer);
  mlir::FailureOr<std::string>
  emitTensorShape(mlir::Operation &operation, unsigned resultIndex);
  mlir::FailureOr<unsigned> emittedTensorRank(mlir::Operation &operation,
                                             bool store);
  std::string broadcastIndex(llvm::StringRef base, unsigned axis,
                             unsigned rank);
  mlir::LogicalResult prepareRaggedMetadata();
  mlir::LogicalResult emitProgramBindings();
  void bindResult(mlir::Operation &operation, unsigned index,
                  llvm::StringRef name);
  std::string makeResultName(mlir::Operation &operation, unsigned index);
  std::string makeRegionArgumentName(mlir::Operation &operation,
                                     unsigned argumentIndex);
  std::string programGrid(llvm::StringRef extent);
  std::string uniqueName(llvm::StringRef candidate, int64_t node);
  void line(llvm::StringRef text);

  intent::target::KernelModel kernel;
  intent::plan::ProgramOp physicalProgram;
  intent::plan::SearchSpaceOp searchSpace;
  PhysicalProgramIndex planIndex;
  SearchIndex searchIndex;
  llvm::raw_ostream &output;
  llvm::SmallVector<ABIView> views;
  llvm::SmallVector<ABIScalar> scalars;
  llvm::DenseMap<mlir::Value, unsigned> viewPositions;
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  llvm::DenseMap<mlir::Value, mlir::Operation *> deferredLoads;
  llvm::DenseMap<mlir::Operation *, target::lowering::DeferredContractReplay>
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
  llvm::DenseMap<
      mlir::Operation *,
      llvm::SmallVector<std::pair<int64_t, std::optional<std::string>>>>
      streamAxisRestores;
  llvm::DenseMap<int64_t, std::string> activeTraversalEnds;
  llvm::DenseMap<
      mlir::Operation *,
      llvm::SmallVector<std::pair<int64_t, std::optional<std::string>>>>
      streamEndRestores;
  llvm::DenseMap<int64_t, std::string> programBlocks;
  llvm::SmallVector<std::string> dimensionOrder;
  bool requiresPreallocatedOutputs = false;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      streamCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      loopCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>>
      whileCarriers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<std::string>> ifResults;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<std::string>> scalarBuffers;
  llvm::DenseMap<mlir::Value, std::string> vectorBuffers;
  llvm::DenseMap<mlir::Value, std::string> workspaceNames;
  llvm::DenseMap<mlir::Value, plan::ScanOp> scanResults;
  llvm::DenseMap<mlir::Value, plan::ScanOp> scanMaterializedValues;
  llvm::DenseMap<int64_t, std::string> scanExtents;
  llvm::DenseMap<mlir::Operation *, int64_t> scanProducerOwners;
  int64_t activeScanReplay = -1;
  llvm::SmallVector<mlir::Operation *> privateWorkspaceBuffers;
  llvm::SmallVector<RaggedRuntime, 0> raggedRuntimes;
  llvm::DenseMap<int64_t, unsigned> raggedRuntimeByRelation;
  llvm::DenseMap<int64_t, llvm::SmallVector<unsigned>> raggedRuntimesByAxis;
  mlir::Operation *programRoot = nullptr;
  mlir::Operation *vectorDomain = nullptr;
  ABIView *fixedOutput = nullptr;
  std::string kernelName;
  std::string programIndex;
  std::string vectorIndex;
  bool programBindingsEmitted = false;
  bool configureRowVector = false;
  unsigned indentation = 1;
};

mlir::LogicalResult
registerEmissionHandlers(intent::target::OperationHandlerRegistry &registry,
                         ProgramMaterializer &emitter);

mlir::FailureOr<PhysicalProgramIndex>
indexPhysicalProgram(intent::plan::ProgramOp physicalProgram,
                 const intent::target::KernelModel &kernel);

mlir::FailureOr<SearchIndex>
indexSearchSpace(intent::plan::SearchSpaceOp searchSpace);

mlir::LogicalResult materializeProgramSource(
    intent::target::KernelModel kernel,
    intent::plan::ProgramOp physicalProgram,
    intent::plan::SearchSpaceOp searchSpace, llvm::raw_ostream &output);

} // namespace intent::triton::lowering

#endif

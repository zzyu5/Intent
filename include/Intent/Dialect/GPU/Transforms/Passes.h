#ifndef INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_PASSES_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <utility>

namespace intent::gpu {

struct ReplayMaterializationOptions {
  PhysicalReplayScope scope = PhysicalReplayScope::ValueGraph;
  bool allowAccesses = true;
  llvm::ArrayRef<MakeRangeOp> traversalRanges;
  std::optional<unsigned> fragmentAxis;
  mlir::Value segmentTail;
  AxisMapAttr segmentMapping;
  bool materializeZeroFill = false;
};

mlir::FailureOr<mlir::func::FuncOp>
getPhysicalKernel(mlir::ModuleOp module);
std::pair<uint64_t, int64_t>
nextPhysicalAxisIdentities(mlir::func::FuncOp kernel);
ParameterOp getOrCreatePhysicalParameter(
    mlir::func::FuncOp kernel, llvm::StringRef name, ParameterRole role,
    ParameterCategory category, uint32_t elementBitWidth,
    llvm::ArrayRef<int64_t> candidates);
mlir::FailureOr<uint64_t> blockedDimension(mlir::Attribute attribute);
bool hasBlockedDimension(mlir::func::FuncOp kernel, uint64_t dimension);
void retargetSourceExtent(mlir::Value root, PhysicalSourceAxis source,
                          PhysicalExprAttr extent);
void retargetDimensionExtent(mlir::Value root, int64_t dimensionId,
                             PhysicalExprAttr extent);
mlir::LogicalResult
alignStructuredCaptureRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignPointwiseValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignAccessResultRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignAggregateValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignContractValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignAccessValueRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
refreshReshapeRelations(mlir::func::FuncOp kernel);
mlir::FailureOr<mlir::Value>
projectPhysicalValueToSchema(mlir::OpBuilder &builder, mlir::Location location,
                             mlir::Value value, mlir::Type target);
mlir::LogicalResult
alignReductionResultRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignReductionIdentityRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult
alignReductionYieldRelations(mlir::func::FuncOp kernel);
mlir::LogicalResult bindFullCoverageDimension(mlir::func::FuncOp kernel,
                                              uint64_t dimension,
                                              mlir::Value physicalExtent);
mlir::LogicalResult realizeFullCoverageDimension(mlir::func::FuncOp kernel,
                                                 mlir::Value source,
                                                 unsigned fragmentAxis);
mlir::FailureOr<mlir::Value>
resolveLogicalRangeEnd(mlir::func::FuncOp kernel, MakeRangeOp range);
mlir::FailureOr<mlir::Value>
materializeScalarConstant(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Attribute value, mlir::Type resultType);
mlir::FailureOr<mlir::Value>
materializeBroadcastToFragment(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value value,
                               FragmentType target);
mlir::FailureOr<mlir::Value>
materializeZeroFragment(mlir::OpBuilder &builder, mlir::Location location,
                        FragmentType target);
mlir::Value stripAdditiveProjection(mlir::Value value, bool singleUse = false);
bool isLiteralZeroProjection(mlir::Value value);
mlir::FailureOr<mlir::Value> materializeReplayedValue(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    mlir::IRMapping &mapping,
    ReplayMaterializationOptions options = {});
mlir::FailureOr<mlir::Value>
projectPredicateToFragmentAxis(mlir::OpBuilder &builder, mlir::Location location,
                               mlir::Value predicate, FragmentType target,
                               unsigned fragmentAxis);
mlir::FailureOr<mlir::Value>
projectPredicateToFragment(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate, FragmentType target,
                           PhysicalSourceAxis source);
mlir::FailureOr<mlir::Value>
projectPredicateToFragment(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value predicate, FragmentType target,
                           int64_t dimensionId);
mlir::FailureOr<mlir::Value>
materializeValidityConjunction(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value lhs,
                               mlir::Value rhs, FragmentType valueType);
/// Replaces the tail portion of an existing validity predicate with the tail
/// selected by the current physical rewrite while preserving scalar author
/// predicates.  Shape-dependent residual predicates fail instead of being
/// silently discarded or guessed from surrounding structure.
mlir::FailureOr<mlir::Value> materializeRetargetedValidity(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value original,
    llvm::ArrayRef<std::pair<MakeRangeOp, mlir::Value>> originalTailRanges,
    mlir::Value physicalTail, FragmentType target);
mlir::LogicalResult verifyGPUProgram(mlir::ModuleOp module);
void eraseDeadPhysicalValues(mlir::func::FuncOp kernel);
void eraseUnusedPhysicalParameters(mlir::func::FuncOp kernel);
mlir::LogicalResult replacePhysicalParameter(mlir::func::FuncOp kernel,
                                             ParameterOp previous,
                                             ParameterOp replacement);
mlir::LogicalResult realizeAccessComposition(mlir::ModuleOp module);
mlir::LogicalResult simplifyMaskedAccessCoordinates(mlir::ModuleOp module);
mlir::LogicalResult predicateScalarControl(mlir::ModuleOp module);
bool canPredicateScalarBlock(mlir::Block &block);
bool canPredicateValueOperation(mlir::Operation *operation);
void clonePredicatedScalarOperation(mlir::OpBuilder &builder,
                                   mlir::Operation *operation,
                                   mlir::IRMapping &mapping,
                                   mlir::Value predicate,
                                   FragmentType shape = {});
mlir::LogicalResult realizeScanConsumerTraversals(mlir::ModuleOp module);
mlir::Value createInvocationWorkspace(mlir::func::FuncOp kernel,
                                      mlir::Location location,
                                      mlir::Type elementType,
                                      mlir::ArrayAttr shape, uint64_t owner);
mlir::LogicalResult materializeRetainedValues(mlir::ModuleOp module);
mlir::LogicalResult vectorizeBufferLoops(mlir::ModuleOp module);
mlir::FailureOr<mlir::Value> materializeRetainedSlice(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    unsigned axis, PhysicalExprAttr blockedExtent, mlir::Value coordinates,
    mlir::Operation *insertionAnchor);
mlir::LogicalResult realizeRegionFolds(mlir::ModuleOp module);
mlir::LogicalResult realizeRegionScans(mlir::ModuleOp module);
mlir::LogicalResult realizeContractionBlocking(mlir::ModuleOp module);
mlir::LogicalResult orientLoopContractions(mlir::ModuleOp module);
mlir::LogicalResult realizeVectorContractions(mlir::ModuleOp module);
mlir::LogicalResult composeContractResultReshapes(mlir::ModuleOp module);
mlir::LogicalResult decomposeMultiAxisReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeOnlineReductions(mlir::ModuleOp module);
mlir::LogicalResult realizeReductionBlocking(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseOwnership(mlir::ModuleOp module);
mlir::LogicalResult realizePointwiseBlocking(mlir::ModuleOp module);
mlir::LogicalResult refineProgramMapping(mlir::ModuleOp module);
mlir::LogicalResult eliminateCommonValues(mlir::ModuleOp module);
mlir::LogicalResult fuseIndependentTraversals(mlir::ModuleOp module);
mlir::LogicalResult materializeSharedConfigTuples(mlir::func::FuncOp kernel,
                                                 const TuningProfiles &profiles);
mlir::LogicalResult verifySharedConfigTuples(mlir::func::FuncOp kernel);
mlir::LogicalResult completeGPUProgramConstruction(mlir::ModuleOp module);
mlir::LogicalResult runSharedGPUPasses(mlir::ModuleOp module,
                                      const TuningProfiles &profiles);

} // namespace intent::gpu

#endif

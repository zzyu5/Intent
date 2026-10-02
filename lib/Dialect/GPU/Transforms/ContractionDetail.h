#ifndef INTENT_GPU_CONTRACTION_DETAIL_H
#define INTENT_GPU_CONTRACTION_DETAIL_H

#include "Intent/Dialect/GPU/Transforms/Contraction.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace intent::gpu::contraction {
using namespace mlir;

inline constexpr int64_t contractionReductionCandidates[] = {
    4, 8, 16, 32, 64, 128, 256};

struct StorePath {
  SmallVector<Operation *> operations;
  StoreOp store;
};

enum class ProgramMappingScope { SameBlock, Dominating };

struct ProgramSegment {
  ExecutionGroupOp mapping;
  ArrayAttr space;
  PhysicalExprAttr offset, length;

  bool isComplete() const {
    return mapping && space && space.size() == 1 && offset && length &&
           offset.getKind() == PhysicalExprKind::Constant &&
           offset.getValue() == 0 && space[0] == length;
  }
};

ProgramSegment queryContractionProgramSegment(func::FuncOp kernel,
                                              Operation *operation,
                                              ProgramMappingScope scope);

FailureOr<bool> rewriteContractionSnapshot(
    func::FuncOp kernel,
    llvm::function_ref<FailureOr<bool>(ContractOp)> rewrite);
LogicalResult prepareRetainedContractionReads(func::FuncOp kernel);
void pruneContractionProgramCoordinates(func::FuncOp kernel);

// These internal interfaces operate on current physical values. They separate
// source normalization, relation queries, replay, projection and traversal.
PhysicalExprAttr expression(MLIRContext *context, PhysicalExprKind kind,
                            int64_t value = 0, StringRef symbol = {},
                            ArrayRef<Attribute> operands = {});

PhysicalExprAttr parameterExpression(MLIRContext *context, StringRef name);

PhysicalExprAttr binaryExpression(MLIRContext *context, PhysicalExprKind kind,
                                  PhysicalExprAttr lhs,
                                  PhysicalExprAttr rhs);

bool isCompileTimeExtent(PhysicalExprAttr expression);

bool isFullCoverageExtent(Operation *origin, Attribute attribute);

bool reductionUsesOwnershipExtent(Operation *origin, Value operand,
                                  ArrayRef<int64_t> reductionAxes);

void fuseContractionAdds(func::FuncOp kernel);

bool hasFragmentSchema(ContractOp contract);

bool hasFragmentSchema(ScaledContractOp contract);

bool hasFragmentSchema(SparseContractOp contract);

bool requiresPhysicalRealization(ContractOp contract);

bool requiresPhysicalRealization(ScaledContractOp contract);

Value binary(OpBuilder &builder, Location location, Type result, Value lhs,
             Value rhs, BinaryOperator kind);

Value compare(OpBuilder &builder, Location location, Type result, Value lhs,
              Value rhs, ComparePredicate predicate);

void inheritRangeAuthority(Value derived, MakeRangeOp source);

MakeRangeOp sourceRange(Value value);

bool containsSource(Value value, PhysicalSourceAxis source);

FailureOr<unsigned> mappingAxisForScalar(Value value, ExecutionGroupOp mapping);

LogicalResult refineOwnershipParameter(func::FuncOp kernel, MakeRangeOp range,
                                       FailureOr<unsigned> mappingAxis,
                                       ParameterAttr replacement);

LogicalResult collectPairedReductionRanges(
    ContractOp contract, SmallVectorImpl<MakeRangeOp> &lhsRanges,
    SmallVectorImpl<MakeRangeOp> &rhsRanges);

bool hasExplicitPairedReductionRanges(ContractOp contract);

FailureOr<MakeRangeOp> producerRange(Value value, PhysicalSourceAxis source);

FailureOr<Value> replaySourceValue(OpBuilder &builder, Location location,
                                   Value value,
                                   PhysicalExprAttr blockedExtent,
                                   ArrayRef<MakeRangeOp> roots,
                                   Value replacement,
                                   IRMapping &mapping,
                                   Operation *insertionAnchor);

FailureOr<Value> buildRangeTailPredicate(OpBuilder &builder, Location location,
                                         Value range,
                                         MakeRangeOp authority);

LogicalResult appendTailValidity(Location location, Value source,
                                 ArrayRef<MakeRangeOp> ranges,
                                 Value tailPredicate,
                                 IRMapping &mapping);

FailureOr<Value> replaySourceValue(OpBuilder &builder, Location location,
                                   func::FuncOp kernel, Value value,
                                   PhysicalSourceAxis source,
                                   PhysicalExprAttr blockedExtent,
                                   MakeRangeOp root, Value replacement,
                                   IRMapping &mapping,
                                   Operation *insertionAnchor = nullptr);

bool isIntegerConstant(Value value, int64_t expected);

bool isTailPredicate(Value value,
                     ArrayRef<std::pair<MakeRangeOp, Value>> ranges);

FailureOr<ParameterAttr> parameterForExtent(func::FuncOp kernel,
                                          PhysicalExprAttr extent);

LogicalResult markNativeCoverage(func::FuncOp kernel, Value source,
                                 ArrayRef<int64_t> axes);

LogicalResult markNativeCoverage(func::FuncOp kernel, ContractOp contract);

LogicalResult markNativeCoverage(func::FuncOp kernel,
                                 ScaledContractOp contract);

bool isZeroPastLogicalEnd(Value value, MakeRangeOp range);

LogicalResult neutralizeFullCoverageOperand(ContractOp contract,
                                            OpOperand &operand,
                                            ArrayRef<int64_t> axes,
                                            func::FuncOp kernel);

FragmentType transposeLastTwo(FragmentType source);

LogicalResult normalizeMatrixContractForms(func::FuncOp kernel);

FragmentType fragmentType(MLIRContext *context, Type element,
                          ArrayRef<PhysicalExprAttr> shape,
                          ArrayRef<AxisMapAttr> sourceMappings,
                          uint64_t owner);

FailureOr<Value> scalarSource(Value value);

bool isZeroScalar(Value value);

Value broadcastAxis(OpBuilder &builder, Location location, FragmentType result,
                    Value value, unsigned targetAxis);

Value rangeBoundsValidity(OpBuilder &builder, Location location,
                          FragmentType indexType,
                          FragmentType predicateType, Value coordinate,
                          Value logicalStop);

FailureOr<Value> retargetFill(OpBuilder &builder, Location location,
                              Value original, FragmentType result);

bool collectStorePaths(Value value, SmallVector<Operation *> operations,
                       SmallVectorImpl<StorePath> &paths,
                       llvm::SmallPtrSetImpl<Operation *> &visited,
                       Value root = {});

FailureOr<Value> materializeResultCapture(
    OpBuilder &builder, func::FuncOp kernel, Value value, FragmentType tile,
    Value rows, Value columns, Operation *insertionAnchor);

FailureOr<Value> materializeStorePath(
    OpBuilder &builder, func::FuncOp kernel, StorePath &path, Value original,
    Value blocked, Value rows, Value columns, Operation *insertionAnchor);

LoadOp matrixOperandLoad(Value value);

FailureOr<unsigned> accessCoordinatePosition(LoadOp load,
                                             AxisMapAttr mapping,
                                             Value operand);

FailureOr<unsigned> directRankOneAccessPosition(ValueRange coordinates,
                                                Type valueType,
                                                unsigned valueAxis);

bool freeAxesNeedRealization(ContractOp contract, func::FuncOp kernel);

bool freeAxesReadyForReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel);

bool reductionAxesNeedTraversal(ContractOp contract, func::FuncOp kernel);

bool hasCompleteStorePath(ContractOp contract);

bool outputCoordinatesNeedRealization(ContractOp contract);

FragmentType eraseFragmentAxis(FragmentType source, unsigned erasedAxis);

SmallVector<int64_t> eraseAxis(ArrayRef<int64_t> axes, unsigned erasedAxis);

FailureOr<bool> collapseMultiReductionContract(ContractOp contract);

LogicalResult decomposeMultiReductionContract(ContractOp contract);

scf::ForOp enclosingRegionContractionSegment(Operation *operation);

FailureOr<ParameterAttr>
regionContractionParameter(func::FuncOp kernel, PhysicalExprAttr extent);

FailureOr<bool> realizeSegmentNativeReduction(ContractOp contract,
                                              func::FuncOp kernel);

FailureOr<bool> realizeStructuredNativeReduction(
    ContractOp contract, func::FuncOp kernel,
    SmallVectorImpl<ContractOp> &replayed);

LogicalResult realizeReductionTraversal(ContractOp contract,
                                        func::FuncOp kernel,
                                        SmallVectorImpl<ContractOp> &replayed);

bool canReplayContractionReads(ContractOp contract);

bool fullReductionNeedsTraversal(ContractOp contract);

FailureOr<bool> realizeFullResultTraversal(
    ContractOp contract, func::FuncOp kernel,
    SmallVectorImpl<ContractOp> &pending);

LogicalResult realizeSparseReductionTraversal(SparseContractOp contract,
                                              func::FuncOp kernel);

LogicalResult realizeContract(ContractOp contract, func::FuncOp kernel,
                              SmallVectorImpl<ContractOp> &pending);

LogicalResult realizeScaledContract(ScaledContractOp contract,
                                    func::FuncOp kernel);

FailureOr<bool> projectContractResult(ContractOp contract);

} // namespace intent::gpu::contraction

#endif

#ifndef INTENT_DIALECT_GPU_TRANSFORMS_VALUEMATERIALIZATION_H
#define INTENT_DIALECT_GPU_TRANSFORMS_VALUEMATERIALIZATION_H

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/Transforms/ValueRelations.h"
#include "mlir/IR/IRMapping.h"

namespace intent::gpu {

// Rewrites consume current coordinate/replay facts; they do not choose tiling.
struct ReplayMaterializationOptions {
  PhysicalReplayScope scope = PhysicalReplayScope::ValueGraph;
  bool allowAccesses = true;
  llvm::ArrayRef<MakeRangeOp> traversalRanges;
  std::optional<unsigned> fragmentAxis;
  mlir::Value segmentTail;
  AxisMapAttr segmentMapping;
  bool materializeZeroFill = false;
};

mlir::LogicalResult scalarizeElementwiseCallback(mlir::Region &source,
                                                 mlir::Region &target);
void foldExactConstantDivisions(mlir::func::FuncOp kernel);
void foldScalarIntegerValues(mlir::func::FuncOp kernel);
mlir::FailureOr<mlir::Value>
materializeNonOverlappingView(mlir::func::FuncOp kernel, mlir::Value resource);
mlir::FailureOr<mlir::Value>
materializeScalarConstant(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Attribute value, mlir::Type resultType);
mlir::FailureOr<mlir::Value>
projectPhysicalValueToSchema(mlir::OpBuilder &builder, mlir::Location location,
                             mlir::Value value, mlir::Type target,
                             ValueTypeChangeCallback changed = {});
mlir::FailureOr<mlir::Value>
materializeZeroValue(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Type target);
mlir::FailureOr<mlir::Value>
materializeZeroFragment(mlir::OpBuilder &builder, mlir::Location location,
                        FragmentType target);
mlir::FailureOr<mlir::Value> materializeReplayedValue(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value value,
    PhysicalSourceAxis source, PhysicalExprAttr blockedExtent,
    mlir::IRMapping &mapping, ReplayMaterializationOptions options = {});
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
// Replaces proven tail predicates while retaining author scalar predicates.
mlir::FailureOr<mlir::Value> materializeRetargetedValidity(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value original,
    llvm::ArrayRef<std::pair<MakeRangeOp, mlir::Value>> originalTailRanges,
    mlir::Value physicalTail, FragmentType target);

} // namespace intent::gpu

#endif

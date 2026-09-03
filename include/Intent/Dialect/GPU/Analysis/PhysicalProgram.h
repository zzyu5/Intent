#ifndef INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>

namespace intent::gpu {

/// A logical source axis carried by the current executable GPU program.
struct PhysicalSourceAxis {
  uint64_t sourceId = 0;
  uint64_t sourceAxis = 0;
  bool derived = false;

  bool operator==(const PhysicalSourceAxis &other) const {
    return sourceId == other.sourceId && sourceAxis == other.sourceAxis &&
           derived == other.derived;
  }
};

PhysicalSourceAxis sourceAxisIdentity(AxisMapAttr mapping);
PhysicalSourceAxis sourceAxisIdentity(MakeRangeOp range);

enum class PhysicalFactState { Exact, Unknown, Ambiguous };

enum class PhysicalReplayScope {
  /// Scalar/coordinate expressions and access coordinates.  Structured
  /// reductions and contractions are not mechanically replayable here.
  Coordinate,
  /// A complete pointwise value graph, including first-class structured
  /// operations whose result type can be retargeted by the caller.
  ValueGraph,
};

bool isPhysicalReplayNode(mlir::Operation *operation,
                          PhysicalReplayScope scope,
                          bool allowAccesses);

struct PhysicalAxisProjection {
  PhysicalFactState state = PhysicalFactState::Unknown;
  PhysicalSourceAxis source;
  int64_t dimensionId = 0;
  unsigned fragmentAxis = 0;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

struct PhysicalDimensionProjection {
  PhysicalFactState state = PhysicalFactState::Unknown;
  int64_t dimensionId = 0;
  unsigned fragmentAxis = 0;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

PhysicalAxisProjection queryFragmentAxis(mlir::Type type,
                                         PhysicalSourceAxis source);
llvm::SmallVector<PhysicalAxisProjection, 2>
queryFragmentAxes(mlir::Type type, PhysicalSourceAxis source);
llvm::SmallVector<PhysicalAxisProjection, 2>
queryRangeProjections(mlir::Type type, MakeRangeOp range);
PhysicalDimensionProjection queryFragmentDimension(mlir::Type type,
                                                   int64_t dimensionId);
llvm::SmallVector<PhysicalDimensionProjection, 2>
queryFragmentDimensions(mlir::Type type, int64_t dimensionId);
mlir::FailureOr<int64_t>
querySourceDimension(mlir::Type type, PhysicalSourceAxis source);
PhysicalAxisProjection
queryCoordinateIndex(mlir::ValueRange coordinates, PhysicalSourceAxis source);
mlir::FailureOr<unsigned>
queryCoordinatePosition(mlir::ValueRange coordinates,
                        PhysicalSourceAxis source);
mlir::FailureOr<AxisMapAttr> queryAxisMap(mlir::Type type,
                                         unsigned fragmentAxis);
mlir::FailureOr<int64_t> queryRangeDimension(MakeRangeOp range);
bool samePhysicalScalarExpression(mlir::Value lhs, mlir::Value rhs);

/// Returns the single typed binary operation implemented by a two-argument
/// combine region. Physical broadcast projections inserted while aligning
/// helper arguments do not change that semantic operation and are ignored.
/// Arbitrary arithmetic, casts, reshapes, or captures remain unknown.
std::optional<BinaryOperator>
queryBinaryCombineKind(mlir::Region &region);

/// All current-IR range roots that carry one requested source axis.
struct PhysicalRangeFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<MakeRangeOp, 2> roots;
  llvm::SmallVector<mlir::Operation *, 2> accesses;
  llvm::SmallVector<mlir::Operation *, 2> blockers;
  bool unitStep = false;

  bool isExact() const { return state == PhysicalFactState::Exact; }
  bool isUnique() const { return isExact() && roots.size() == 1; }
};

/// Whether one current fragment axis has been materialized over its exact
/// physical range.  Construction may legally seed a dynamic logical axis with
/// a scalar fragment; that scalar type is not evidence that the traversal has
/// already been blocked.  Consumers use this fact instead of comparing shape
/// attributes or range operands independently.
struct PhysicalAxisRealizationFact {
  enum class ExtentAuthority {
    None,
    /// The current axis is materialized by one exact physical range whose SSA
    /// extent agrees with the fragment extent.
    Range,
    /// A verified value relation explicitly selects or preserves this physical
    /// extent.  This includes reshape reassociation and an extent-preserving
    /// projection of an already authoritative input.
    Structural,
  };

  PhysicalFactState state = PhysicalFactState::Unknown;
  PhysicalSourceAxis source;
  int64_t dimensionId = 0;
  unsigned fragmentAxis = 0;
  llvm::SmallVector<MakeRangeOp, 2> roots;
  llvm::SmallVector<mlir::Operation *, 2> blockers;
  bool constructionScalarSeed = false;
  bool physicalized = false;
  ExtentAuthority extentAuthority = ExtentAuthority::None;

  bool isExact() const { return state == PhysicalFactState::Exact; }
  bool hasExtentAuthority() const {
    return isExact() && extentAuthority != ExtentAuthority::None;
  }
};

/// Exact fragment axes whose current coordinate provenance reaches one of a
/// selected set of physical range roots.  This distinguishes repeated uses of
/// one logical source axis in different result positions without inventing a
/// second source identity.
struct PhysicalRangeAxisFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<unsigned, 2> fragmentAxes;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

enum class PhysicalLockstepState { Exact, Unknown, Inconsistent };

/// One current physical traversal shared by several structured sources.  The
/// selected MakeRange operation is only a canonical SSA carrier after all
/// source ranges have been proven equivalent.
struct PhysicalLockstepTraversalFact {
  PhysicalLockstepState state = PhysicalLockstepState::Unknown;
  MakeRangeOp authority;
  llvm::SmallVector<mlir::Operation *, 4> blockers;

  bool isExact() const { return state == PhysicalLockstepState::Exact; }
};

bool sameLogicalRange(MakeRangeOp lhs, MakeRangeOp rhs);
bool isUnitStepRange(MakeRangeOp range);
/// The launch-visible parent dimension of a narrowed logical subregion.  The
/// subregion keeps its own extent identity in RangeType/AxisMapAttr; this
/// relation only supplies the program-space upper bound used for ownership.
mlir::FailureOr<int64_t> querySubregionParentDimension(MakeRangeOp range);
mlir::FailureOr<MakeRangeOp> queryExactLogicalRange(
    const PhysicalRangeFact &fact);

/// Whether a value can be mechanically rebuilt after replacing one source
/// range.  This is a fact about the current graph; it never performs cloning.
struct PhysicalReplayFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  bool crossesAccess = false;
  bool crossesStructuredProgram = false;
  llvm::SmallVector<mlir::Operation *, 2> accesses;
  llvm::SmallVector<mlir::Operation *, 2> contractions;
  llvm::SmallVector<mlir::Operation *, 2> structuredPrograms;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isReplayable() const { return state == PhysicalFactState::Exact; }
};

/// Exact dependence of one current physical value on a structured reduction
/// traversal. Unknown is conservative and carries the operations that prevent
/// the relation from being established.
struct PhysicalReductionDependencyFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  bool depends = false;
  bool throughStructuredReduction = false;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

/// Current physical realizations of every non-batch, non-reduction operand
/// axis of one contraction operation.  This is the single authority used by
/// ownership and contraction transformations; neither consumer re-derives
/// free axes from result shape or nearby stores.
struct PhysicalContractFreeAxis {
  mlir::Value operand;
  unsigned operandAxis = 0;
  PhysicalAxisRealizationFact realization;
  PhysicalRangeFact ranges;
};

struct PhysicalContractFreeAxisFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<PhysicalContractFreeAxis, 4> axes;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
  bool needsRealization() const {
    return !isExact() || llvm::any_of(axes, [](const auto &axis) {
             return !axis.realization.physicalized;
           });
  }
};

/// Typed logical relation carried by one physical parameter declaration.  The
/// parameter name remains only its compile-time symbol; consumers must not
/// recover a dimension or source axis from that spelling.
struct PhysicalParameterBinding {
  PhysicalFactState state = PhysicalFactState::Unknown;
  std::optional<int64_t> dimension;
  std::optional<PhysicalSourceAxis> source;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

PhysicalParameterBinding queryParameterBinding(ParameterOp parameter);
mlir::FailureOr<ParameterOp>
queryParameterBySymbol(mlir::func::FuncOp kernel, mlir::StringAttr symbol);
mlir::FailureOr<ParameterOp>
queryBlockingParameter(mlir::func::FuncOp kernel, MakeRangeOp range);

/// Exact current-IR access relation, or an explicit conservative result.
struct PhysicalAccessFootprint {
  PhysicalFactState state = PhysicalFactState::Unknown;
  PhysicalFactState rangeState = PhysicalFactState::Unknown;
  mlir::Value resource;
  llvm::SmallVector<mlir::Value, 4> coordinates;
  llvm::SmallVector<int64_t, 4> sourceAxes;
  llvm::SmallVector<MakeRangeOp, 4> ranges;
  llvm::SmallVector<mlir::Operation *, 4> blockers;
  mlir::Value validity;
  mlir::Value fill;
};

/// Current-IR initialization and resource-use legality for one physical
/// mutable buffer.  Exact means every read is dominated either by a direct
/// initializing write or by a proven full-domain loop/branch initialization.
struct PhysicalBufferDataflowFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  llvm::SmallVector<mlir::Operation *, 4> blockers;

  bool isExact() const { return state == PhysicalFactState::Exact; }
};

/// Recomputable facts derived only from the current executable GPU IR.
/// A transformation constructs this object after the preceding mutation,
/// queries all plans it needs, then discards it before rewriting the IR.
class PhysicalProgramAnalysis {
public:
  explicit PhysicalProgramAnalysis(mlir::func::FuncOp kernel);

  mlir::FailureOr<unsigned> fragmentAxis(mlir::Type type,
                                         PhysicalSourceAxis source) const;
  mlir::FailureOr<unsigned>
  coordinateIndex(mlir::ValueRange coordinates,
                  PhysicalSourceAxis source) const;
  PhysicalRangeFact sourceRanges(
      mlir::Value value,
      std::optional<PhysicalSourceAxis> source = std::nullopt);
  /// Exact current-program ranges for a typed logical source axis.  This is
  /// used only when a replayable pure value carries the axis in its type but
  /// has no producer edge to the coordinate value.
  PhysicalRangeFact programRanges(PhysicalSourceAxis source);
  /// Exact range roots that make one concrete fragment axis vary.  Unlike a
  /// source-id query, this preserves repeated occurrences of the same logical
  /// source in Cartesian/indexed values.
  PhysicalRangeFact axisRanges(mlir::Value value, unsigned fragmentAxis);
  PhysicalAxisRealizationFact axisRealization(mlir::Value value,
                                               unsigned fragmentAxis);
  PhysicalRangeAxisFact rangeAxes(mlir::Value value,
                                  llvm::ArrayRef<MakeRangeOp> roots);
  PhysicalLockstepTraversalFact
  lockstepRanges(llvm::ArrayRef<MakeRangeOp> ranges);
  PhysicalLockstepTraversalFact
  lockstepTraversal(mlir::ValueRange sources,
                    llvm::ArrayRef<unsigned> fragmentAxes);
  PhysicalReplayFact replayability(
      mlir::Value value,
      std::optional<PhysicalSourceAxis> source = std::nullopt,
      PhysicalReplayScope scope = PhysicalReplayScope::Coordinate,
      bool allowAccesses = true,
      mlir::Operation *insertionAnchor = nullptr,
      std::optional<int64_t> sourceDimension = std::nullopt);
  PhysicalReductionDependencyFact reductionDependency(
      mlir::Value value, PhysicalSourceAxis source,
      std::optional<int64_t> sourceDimension = std::nullopt);
  PhysicalContractFreeAxisFact contractFreeAxes(mlir::Operation *contract);
  PhysicalAccessFootprint footprint(mlir::Operation *access);
  PhysicalBufferDataflowFact bufferDataflow(BufferOp buffer);

  /// Recognizes a predicate composed only from exact range-end comparisons,
  /// predicate-preserving shape operations and boolean conjunction.  This is
  /// a pure query; it never invents or rewrites validity.
  bool isTailPredicate(
      mlir::Value value,
      llvm::ArrayRef<std::pair<MakeRangeOp, mlir::Value>> ranges) const;

private:
  void collectRanges(mlir::Value value,
                     std::optional<PhysicalSourceAxis> source,
                     PhysicalRangeFact &result,
                     llvm::SmallPtrSetImpl<mlir::Operation *> &visited);
  void collectAxisRanges(mlir::Value value, unsigned fragmentAxis,
                         PhysicalRangeFact &result,
                         llvm::SmallPtrSetImpl<mlir::Operation *> &visited);
  void analyzeReplay(mlir::Value value,
                     std::optional<PhysicalSourceAxis> source,
                     PhysicalReplayScope scope, bool allowAccesses,
                     mlir::Operation *insertionAnchor,
                     std::optional<int64_t> sourceDimension,
                     mlir::DominanceInfo *dominance,
                     PhysicalReplayFact &result,
                     llvm::SmallPtrSetImpl<mlir::Operation *> &visited);
  llvm::SmallVector<mlir::Value, 2>
  structuredSourcesForArgument(mlir::BlockArgument argument) const;
  bool carriesSource(mlir::Type type, PhysicalSourceAxis source) const;

  mlir::func::FuncOp kernel;
  llvm::DenseMap<mlir::Value, PhysicalRangeFact> unrestrictedRangeCache;
};

} // namespace intent::gpu

namespace llvm {
template <> struct DenseMapInfo<intent::gpu::PhysicalSourceAxis> {
  static inline intent::gpu::PhysicalSourceAxis getEmptyKey() {
    return {DenseMapInfo<uint64_t>::getEmptyKey(),
            DenseMapInfo<uint64_t>::getEmptyKey(), false};
  }
  static inline intent::gpu::PhysicalSourceAxis getTombstoneKey() {
    return {DenseMapInfo<uint64_t>::getTombstoneKey(),
            DenseMapInfo<uint64_t>::getTombstoneKey(), false};
  }
  static unsigned getHashValue(const intent::gpu::PhysicalSourceAxis &value) {
    return static_cast<unsigned>(
        hash_combine(value.sourceId, value.sourceAxis, value.derived));
  }
  static bool isEqual(const intent::gpu::PhysicalSourceAxis &lhs,
                      const intent::gpu::PhysicalSourceAxis &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif

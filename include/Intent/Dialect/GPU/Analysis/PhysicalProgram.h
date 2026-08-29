#ifndef INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPROGRAM_H
#define INTENT_DIALECT_GPU_ANALYSIS_PHYSICALPROGRAM_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace intent::gpu {

/// A logical source axis carried by the current executable GPU program.
struct PhysicalSourceAxis {
  uint64_t sourceId = 0;
  uint64_t sourceAxis = 0;

  bool operator==(const PhysicalSourceAxis &other) const {
    return sourceId == other.sourceId && sourceAxis == other.sourceAxis;
  }
};

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

mlir::FailureOr<unsigned> queryFragmentAxis(mlir::Type type,
                                            uint64_t sourceId);
PhysicalAxisProjection queryFragmentAxis(mlir::Type type,
                                         PhysicalSourceAxis source);
llvm::SmallVector<PhysicalAxisProjection, 2>
queryFragmentAxes(mlir::Type type, PhysicalSourceAxis source);
PhysicalDimensionProjection queryFragmentDimension(mlir::Type type,
                                                   int64_t dimensionId);
mlir::FailureOr<int64_t>
querySourceDimension(mlir::Type type, PhysicalSourceAxis source);
PhysicalAxisProjection queryUniqueSourceAxis(mlir::Type type,
                                             uint64_t sourceId);
mlir::FailureOr<unsigned>
queryCoordinateIndex(mlir::ValueRange coordinates, uint64_t sourceId);
PhysicalAxisProjection
queryCoordinateIndex(mlir::ValueRange coordinates, PhysicalSourceAxis source);

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

/// Whether a value can be mechanically rebuilt after replacing one source
/// range.  This is a fact about the current graph; it never performs cloning.
struct PhysicalReplayFact {
  PhysicalFactState state = PhysicalFactState::Unknown;
  bool crossesAccess = false;
  llvm::SmallVector<mlir::Operation *, 2> blockers;

  bool isReplayable() const { return state == PhysicalFactState::Exact; }
};

/// Exact current-IR access relation, or an explicit conservative result.
struct PhysicalAccessFootprint {
  PhysicalFactState state = PhysicalFactState::Unknown;
  mlir::Value resource;
  llvm::SmallVector<mlir::Value, 4> coordinates;
  llvm::SmallVector<int64_t, 4> sourceAxes;
  llvm::SmallVector<MakeRangeOp, 4> ranges;
  mlir::Value validity;
  mlir::Value fill;
};

/// Recomputable facts derived only from the current executable GPU IR.
/// A transformation constructs this object after the preceding mutation,
/// queries all plans it needs, then discards it before rewriting the IR.
class PhysicalProgramAnalysis {
public:
  explicit PhysicalProgramAnalysis(mlir::func::FuncOp kernel);

  mlir::FailureOr<unsigned> fragmentAxis(mlir::Type type,
                                         PhysicalSourceAxis source) const;
  mlir::FailureOr<unsigned> fragmentAxis(mlir::Type type,
                                         uint64_t sourceId) const;
  mlir::FailureOr<unsigned>
  coordinateIndex(mlir::ValueRange coordinates,
                  PhysicalSourceAxis source) const;
  mlir::FailureOr<unsigned> coordinateIndex(mlir::ValueRange coordinates,
                                            uint64_t sourceId) const;
  PhysicalRangeFact sourceRanges(
      mlir::Value value,
      std::optional<PhysicalSourceAxis> source = std::nullopt);
  /// Exact range roots that make one concrete fragment axis vary.  Unlike a
  /// source-id query, this preserves repeated occurrences of the same logical
  /// source in Cartesian/indexed values.
  PhysicalRangeFact axisRanges(mlir::Value value, unsigned fragmentAxis);
  PhysicalReplayFact replayability(
      mlir::Value value,
      std::optional<PhysicalSourceAxis> source = std::nullopt,
      PhysicalReplayScope scope = PhysicalReplayScope::Coordinate,
      bool allowAccesses = true);
  PhysicalAccessFootprint footprint(mlir::Operation *access);

  /// Recognizes a predicate composed only from exact range-end comparisons,
  /// predicate-preserving shape operations and boolean conjunction.  This is
  /// a pure query; it never invents or rewrites validity.
  bool isTailPredicate(
      mlir::Value value,
      llvm::ArrayRef<std::pair<MakeRangeOp, mlir::Value>> ranges) const;

  /// Explicitly drops cached current-IR facts.  Callers normally discard the
  /// analysis object instead; this exists for transformations with phased
  /// query/rewrite/query structure.
  void invalidate();

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
                     PhysicalReplayFact &result,
                     llvm::SmallPtrSetImpl<mlir::Operation *> &visited);
  llvm::SmallVector<mlir::Value, 2>
  structuredSourcesForArgument(mlir::BlockArgument argument) const;
  bool carriesSource(mlir::Type type, PhysicalSourceAxis source) const;

  mlir::func::FuncOp kernel;
  llvm::DenseMap<mlir::Value, PhysicalRangeFact> unrestrictedRangeCache;
};

} // namespace intent::gpu

#endif

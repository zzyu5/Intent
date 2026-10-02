#ifndef INTENT_ANALYSIS_CANONICALKERNEL_H
#define INTENT_ANALYSIS_CANONICALKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace intent {

struct CoordinateOrigin {
  mlir::Value source;
  unsigned axis = 0;

  bool operator==(const CoordinateOrigin &other) const {
    return source == other.source && axis == other.axis;
  }
};

struct CoordinateProvenance {
  bool known = false;
  llvm::SmallVector<CoordinateOrigin, 2> origins;
};

struct IndexTermFact {
  int64_t kind = 0;
  std::optional<unsigned> sourceAxis;
  // Preserve start/stop/step slots: a static or absent slot has an empty Value.
  llvm::SmallVector<mlir::Value, 3> operands;
  llvm::SmallVector<std::optional<int64_t>, 3> staticValues;
  CoordinateProvenance coordinate;
  // Logical result axes contributed by this term. All tensor indices share
  // the advanced-index block; scalar indices contribute no logical axis.
  llvm::SmallVector<unsigned, 4> resultAxes;
  // Each axis of a tensor index maps into the shared advanced-index block.
  llvm::SmallVector<unsigned, 4> indexAxes;
  // Static indexing or a dominating explicit assume_in_bounds proves this
  // coordinate lies within the logical resource axis.
  bool inBounds = false;
};

struct IndexRelationFact {
  mlir::Value source;
  unsigned sourceRank = 0;
  llvm::SmallVector<int64_t, 4> resultDimensionIdentities;
  llvm::SmallVector<IndexTermFact, 4> terms;
  unsigned advancedRank = 0;
  std::optional<unsigned> advancedStart;
};

/// One operand's logical axes projected into an operation result. An absent
/// result position denotes an eliminated axis. This is a coordinate relation,
/// not a proof that values are equal or that an operation can be replayed.
struct TensorOperandProjection {
  unsigned operandNumber = 0;
  llvm::SmallVector<std::optional<unsigned>, 4> resultAxes;
};

/// An extent explicitly supplied by the defining shape relation or tensor type.
/// Unknown extents have neither member; no other equal-sized value is searched.
struct TensorExtentFact {
  std::optional<int64_t> constant;
  mlir::Value value;
  // A domain/subregion extent, without synthesizing a DimOp in immutable KIR.
  mlir::Value domain;
};

enum class CanonicalFactState { Exact, Unknown, Ambiguous };

/// One canonical independent instance domain and the connected program slice
/// executed for each instance.  This is semantic input to physical program
/// construction, not a GPU grid decision.
struct LogicalWorksetFact {
  CanonicalFactState state = CanonicalFactState::Unknown;
  mlir::Operation *parallel = nullptr;
  mlir::Block *body = nullptr;
  bool singleton = false;
  llvm::SmallVector<mlir::Value, 4> domains;
  llvm::SmallVector<mlir::BlockArgument, 4> coordinates;

  bool isExact() const { return state == CanonicalFactState::Exact; }
};

enum class LogicalBufferScope { ProgramPrivate, IterationPrivate };

/// Lexical allocation semantics of one canonical logical buffer.
struct LogicalBufferFact {
  CanonicalFactState state = CanonicalFactState::Unknown;
  LogicalBufferScope scope = LogicalBufferScope::ProgramPrivate;
  uint64_t instanceIdentity = 0;
  bool hasFullInitialValue = false;

  bool isExact() const { return state == CanonicalFactState::Exact; }
};

/// Canonical source relation shared by a region-fold/scan segment decision.
struct RegionSegmentFact {
  CanonicalFactState state = CanonicalFactState::Unknown;
  int64_t dimensionIdentity = 0;
  int64_t operationIdentity = -1;

  bool isExact() const { return state == CanonicalFactState::Exact; }
};

/// Immutable, recomputable facts derived only from canonical Intent KIR.
/// This analysis does not select physical structure and is never executable
/// authority.
class CanonicalKernelAnalysis {
public:
  explicit CanonicalKernelAnalysis(mlir::ModuleOp module);

  mlir::LogicalResult verify();

  CoordinateProvenance coordinateProvenance(mlir::Value value);
  CoordinateProvenance axisProvenance(mlir::Value value, unsigned axis);
  mlir::FailureOr<llvm::SmallVector<TensorOperandProjection, 3>>
  operandProjections(mlir::OpResult result) const;
  TensorExtentFact tensorExtent(mlir::Value value, unsigned axis);
  mlir::FailureOr<IndexRelationFact>
  indexRelation(mlir::Operation *operation);
  mlir::FailureOr<llvm::SmallVector<LogicalWorksetFact, 4>>
  logicalWorksets(mlir::func::FuncOp function) const;
  LogicalBufferFact logicalBuffer(mlir::Operation *operation) const;
  RegionSegmentFact regionSegment(mlir::Operation *operation) const;

private:
  CoordinateProvenance computeCoordinateProvenance(mlir::Value value);
  CoordinateProvenance blockArgumentProvenance(mlir::BlockArgument argument);
  CoordinateProvenance resultProvenance(mlir::OpResult result);
  CoordinateProvenance computeAxisProvenance(mlir::Value value, unsigned axis);

  mlir::ModuleOp module;
  llvm::DenseMap<mlir::Value, CoordinateProvenance> coordinateCache;
  llvm::DenseMap<mlir::Value, bool> coordinateActive;
  llvm::DenseMap<std::pair<mlir::Value, unsigned>, CoordinateProvenance> axisCache;
  llvm::DenseMap<std::pair<mlir::Value, unsigned>, bool> axisActive;
};

} // namespace intent

#endif

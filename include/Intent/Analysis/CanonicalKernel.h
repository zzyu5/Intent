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

enum class ShapeAxisKind { Static, ABI, SSAExtent, Inferred, Derived };

struct ShapeAxisFact {
  ShapeAxisKind kind = ShapeAxisKind::Derived;
  int64_t dimensionIdentity = 0;
  std::optional<int64_t> staticExtent;
  mlir::Value extent;
};

struct IndexTermFact {
  int64_t kind = 0;
  std::optional<unsigned> sourceAxis;
  llvm::SmallVector<mlir::Value, 3> operands;
  llvm::SmallVector<std::optional<int64_t>, 3> staticValues;
  CoordinateProvenance coordinate;
};

struct IndexRelationFact {
  mlir::Value source;
  unsigned sourceRank = 0;
  llvm::SmallVector<int64_t, 4> resultDimensionIdentities;
  llvm::SmallVector<IndexTermFact, 4> terms;
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
  llvm::SmallVector<ShapeAxisFact, 4> shapeFacts(mlir::Value value) const;
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

  mlir::ModuleOp module;
  llvm::DenseMap<mlir::Value, CoordinateProvenance> coordinateCache;
  llvm::DenseMap<mlir::Value, bool> coordinateActive;
};

} // namespace intent

#endif

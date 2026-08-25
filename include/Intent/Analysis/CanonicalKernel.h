#ifndef INTENT_ANALYSIS_CANONICALKERNEL_H
#define INTENT_ANALYSIS_CANONICALKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
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

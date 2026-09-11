#ifndef INTENT_DIALECT_CPU_ANALYSIS_REGIONPREDICATES_H
#define INTENT_DIALECT_CPU_ANALYSIS_REGIONPREDICATES_H

#include "Intent/Analysis/RegionSemantics.h"
#include "Intent/Dialect/CPU/IR/RegionProgram.h"
#include "mlir/IR/OpDefinition.h"

namespace intent::cpu {

struct CoordinateSequence {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  bool beginsAtZero;
  bool nonnegative;
};
struct RegionPredicatePlan {
  mlir::Value predicate;
  unsigned source, capture;
  UniformPredicate comparison;
  std::optional<unsigned> validityField;
  mlir::Operation *validityReduction;
  bool identityWhenFalse;
};

std::optional<CoordinateSequence> coordinateSequence(mlir::Value memory, mlir::Operation *at);
std::optional<RegionPredicatePlan> analyzeRegionPredicate(RegionProgram program);

}
#endif

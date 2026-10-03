#ifndef INTENT_GPU_TRANSFORMS_CONFIGURATIONPOLICY_H
#define INTENT_GPU_TRANSFORMS_CONFIGURATIONPOLICY_H

#include "Intent/Dialect/GPU/Analysis/Resources.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace intent::gpu::configuration {

enum class TuningClass {
  Pointwise, PointwiseReduction, StatefulReduction, OnlineMoment, Reduction,
  MultiAxisReduction, RegionReduction, RegionContraction, Scan, Contraction,
  PersistentContraction, Histogram, Execution
};

struct TuningProfile {
  int64_t ownershipM, ownershipN, reduction, reductionOuter, scan;
  int64_t traversalWorkers, traversalGroup;
};

struct ParameterClassification {
  TuningClass kind;
  unsigned width;
  bool reductionRow;
  bool pointwiseTraversal;
};

struct CorrelatedProfileParameters {
  ParameterAttr pointwise, reduction, contraction;
};

struct ReductionProfileParameters {
  ParameterAttr chunk;
  llvm::SmallVector<ParameterAttr> rows;
};

struct ContractionFreeExtent {
  llvm::SmallVector<PhysicalExprAttr> extents;
  llvm::SmallVector<ParameterAttr> parameters;
  llvm::SmallVector<ParameterAttr> profileParameters;
  ParameterRole role;
};

struct FullResultContraction {
  ParameterAttr parameter;
  PhysicalExprAttr otherExtent;
};

// Recomputed for one unchanged current program. These facts select profile
// projections; the resulting configuration set is published only in the IR.
struct ConfigurationFacts {
  llvm::SmallVector<ParameterAttr> parameters;
  llvm::DenseMap<ParameterAttr, ParameterClassification> classifications;
  bool hasTwoAxisPointwiseOwnership = false;
  bool hasFixedPointwiseLocal = false;
  bool pointwiseOnlyProgram = true;
  bool hasContraction = false;
  bool smallRegionRows = false;
  bool multipleRegionMatrixAccumulators = false;
  llvm::DenseMap<ParameterAttr, int64_t> pointwiseLocalMultiplicity;
  llvm::SmallVector<ParameterAttr> pointwiseRowAxes;
  llvm::MapVector<unsigned, llvm::SmallVector<ParameterAttr>> rowGroups;
  llvm::SmallVector<CorrelatedProfileParameters> correlatedProfiles;
  llvm::SmallVector<ReductionProfileParameters> reductionProfiles;
  llvm::SmallVector<mlir::Attribute> indirectRowGroups;
  llvm::SmallVector<ContractionFreeExtent> freeExtents;
  llvm::SmallVector<FullResultContraction> fullResultContractions;
};

ConfigurationFacts analyzeConfigurationPolicy(
    mlir::func::FuncOp kernel, llvm::ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources);

using ProfileLookup =
    llvm::function_ref<const TuningProfile &(ParameterAttr)>;
using ProfileBindingConsumer =
    llvm::function_ref<void(mlir::NamedAttrList &, ProfileLookup, bool)>;
mlir::LogicalResult projectConfigurationProfiles(
    mlir::func::FuncOp kernel, const ConfigurationFacts &facts,
    const TuningProfiles &tables, ProfileBindingConsumer consume);
int64_t requestedValue(const TuningProfile &profile, ParameterRole role);
int64_t selectCandidate(llvm::ArrayRef<int64_t> candidates, int64_t requested);

void bindContractionFreeExtents(
    llvm::ArrayRef<ContractionFreeExtent> groups, mlir::NamedAttrList &bindings,
    ProfileLookup profileFor, mlir::Builder &builder, bool splitInnerAxis);
mlir::LogicalResult bindTraversalFragmentFootprints(
    mlir::func::FuncOp kernel, llvm::ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources,
    mlir::NamedAttrList &bindings, mlir::Builder &builder);
void appendFullResultContractionTuples(
    mlir::func::FuncOp kernel,
    llvm::ArrayRef<FullResultContraction> contractions,
    const mlir::NamedAttrList &bindings, llvm::ArrayRef<ParameterAttr> parameters,
    const FragmentResourceAnalysis &resources, ProfileLookup profileFor,
    mlir::Builder &builder,
    llvm::function_ref<void(mlir::DictionaryAttr)> append);

} // namespace intent::gpu::configuration

#endif

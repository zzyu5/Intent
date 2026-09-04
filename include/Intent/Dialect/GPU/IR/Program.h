#ifndef INTENT_DIALECT_GPU_IR_PROGRAM_H
#define INTENT_DIALECT_GPU_IR_PROGRAM_H

#include "llvm/ADT/StringRef.h"

namespace intent::gpu {

inline constexpr llvm::StringLiteral kernelAttr = "intent_gpu.kernel";
inline constexpr llvm::StringLiteral capabilitiesAttr =
    "intent_gpu.capabilities";
inline constexpr llvm::StringLiteral programSpaceAttr =
    "intent_gpu.program_space";
inline constexpr llvm::StringLiteral gridRankAttr = "intent_gpu.grid_rank";
inline constexpr llvm::StringLiteral coordinateRolesAttr =
    "intent_gpu.coordinate_roles";
inline constexpr llvm::StringLiteral originAttr = "intent_gpu.origin";
inline constexpr llvm::StringLiteral effectOriginsAttr =
    "intent_gpu.effect_origins";
inline constexpr llvm::StringLiteral abiKindAttr = "intent_gpu.abi_kind";
inline constexpr llvm::StringLiteral abiNameAttr = "intent_gpu.abi_name";
inline constexpr llvm::StringLiteral dimensionAttr = "intent_gpu.dimension";
inline constexpr llvm::StringLiteral sourceABIAttr = "intent_gpu.source_abi";
inline constexpr llvm::StringLiteral sourceAxisAttr = "intent_gpu.source_axis";
inline constexpr llvm::StringLiteral sourceSubregionAttr =
    "intent_gpu.source_subregion";
inline constexpr llvm::StringLiteral sourceSubregionBoundAttr =
    "intent_gpu.source_subregion_bound";
inline constexpr llvm::StringLiteral worksetCoordinateRangeAttr =
    "intent_gpu.workset_coordinate_range";
inline constexpr llvm::StringLiteral worksetAxisAttr =
    "intent_gpu.workset_axis";
inline constexpr llvm::StringLiteral executionGroupAttr =
    "intent_gpu.execution_group";
inline constexpr llvm::StringLiteral segmentOffsetAttr =
    "intent_gpu.segment_offset";
inline constexpr llvm::StringLiteral segmentLengthAttr =
    "intent_gpu.segment_length";
inline constexpr llvm::StringLiteral coverageDimensionAttr =
    "intent_gpu.coverage_dimension";
inline constexpr llvm::StringLiteral reductionSourcesAttr =
    "intent_gpu.reduction_sources";
inline constexpr llvm::StringLiteral parameterSourceAttr =
    "intent_gpu.parameter_source";
inline constexpr llvm::StringLiteral parameterGroupAttr =
    "intent_gpu.parameter_group";
inline constexpr llvm::StringLiteral pointwiseChunkAttr =
    "intent_gpu.pointwise_chunk";
inline constexpr llvm::StringLiteral pointwiseLocalAttr =
    "intent_gpu.pointwise_local";
inline constexpr llvm::StringLiteral physicalTailAttr =
    "intent_gpu.physical_tail";
inline constexpr llvm::StringLiteral programBoundedOriginAttr =
    "intent_gpu.program_bounded_origin";
inline constexpr llvm::StringLiteral tritonConfigsAttr =
    "intent_gpu.triton.configs";
inline constexpr llvm::StringLiteral tileLangConfigsAttr =
    "intent_gpu.tilelang.configs";
inline constexpr llvm::StringLiteral sharedConfigTuplesAttr =
    "intent_gpu.shared.config_tuples";

enum class PhysicalExprKind : uint32_t {
  Constant = 0,
  Parameter = 1,
  Dimension = 2,
  ScalarABI = 3,
  Add = 4,
  Multiply = 5,
  CeilDiv = 6,
  Minimum = 7,
  Select = 8,
  Subtract = 9,
  FloorDiv = 10,
  Maximum = 11,
  NextPowerOfTwo = 12,
};

enum class ParameterRole : uint32_t {
  OwnershipM = 0,
  OwnershipN = 1,
  Reduction = 2,
  ScanChunk = 3,
  ProviderWarps = 4,
  ProviderStages = 5,
  ProviderCTAs = 6,
  ProviderThreads = 7,
  TraversalWorkers = 8,
  TraversalGroup = 9,
  ResidentWorkers = 10,
  FullCoverage = 11,
  ReductionOuter = 12,
  ReductionInner = 13,
};

enum class ParameterCategory : uint32_t {
  Pointwise = 0,
  Reduction = 1,
  Scan = 2,
  Contraction = 3,
  Execution = 4,
  Coverage = 5,
  Provider = 6,
  RegionReduction = 7,
  RegionContraction = 8,
  PersistentContraction = 9,
  Histogram = 10,
};

enum class CoordinateRole : int64_t {
  Unspecified = -1,
  Workset = 0,
  PointwiseOwnership = 1,
  TiledWorkset = 2,
  ContractionM = 3,
  ContractionN = 4,
  TraversalWorker = 5,
  IndirectTraversal = 6,
};

} // namespace intent::gpu

#endif

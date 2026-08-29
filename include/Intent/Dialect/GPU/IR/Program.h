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
inline constexpr llvm::StringLiteral reductionTraversalSourceAttr =
    "intent_gpu.reduction_traversal_source";
inline constexpr llvm::StringLiteral parameterSourceAttr =
    "intent_gpu.parameter_source";
inline constexpr llvm::StringLiteral pointwiseChunkAttr =
    "intent_gpu.pointwise_chunk";
inline constexpr llvm::StringLiteral tritonConfigsAttr =
    "intent_gpu.triton.configs";

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

#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_CONFIGURATIONS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_CONFIGURATIONS_H
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace intent::cutile {
inline constexpr llvm::StringLiteral accessFormParameter = "CUTILE_ACCESS_FORM";
inline constexpr llvm::StringLiteral occupancyParameter = "CUTILE_OCCUPANCY";
inline constexpr llvm::StringLiteral loadPolicyParameter = "CUTILE_LOAD_POLICY";
inline constexpr llvm::StringLiteral ctasParameter = "CUTILE_CTAS";
inline constexpr llvm::StringLiteral workerWarpsParameter = "CUTILE_WORKER_WARPS";
inline constexpr int64_t nativeAccessForm = 1;
inline constexpr int64_t gatherAccessForm = 2;
inline constexpr int64_t nativeNoTMAForm = 3;


bool isLegalAccessForm(int64_t value);
bool isLegalOccupancy(int64_t value);
bool isLegalWorkerWarps(int64_t value);
bool isLegalCTAs(int64_t value);
bool isCuTileProviderRole(gpu::ParameterRole role);
mlir::FailureOr<gpu::ParameterRefAttr> declareProviderParameter(
    mlir::func::FuncOp kernel, const gpu::TuningProfiles &profiles,
    llvm::StringRef family, llvm::StringRef name, gpu::ParameterRole role,
    bool (*isLegal)(int64_t));
mlir::LogicalResult materializeClosedConfigs(mlir::func::FuncOp kernel);
mlir::LogicalResult finalizeConfigurationRequirements(mlir::func::FuncOp kernel);
mlir::LogicalResult verifyClosedConfigs(mlir::func::FuncOp kernel);
} // namespace intent::cutile
#endif

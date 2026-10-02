#ifndef INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONS_H
#define INTENT_TARGET_TRITON_TRANSFORMS_CONFIGURATIONS_H
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>
namespace intent::triton {
inline constexpr int64_t maxTritonTensorElements = 1048576;

struct TritonLocalOptions {
  int64_t warps;
  int64_t stages;
  int64_t ctas;
};


std::optional<int64_t> evaluateCompileTimeExpression(
    gpu::PhysicalExprAttr expression, mlir::DictionaryAttr bindings = {});
bool isTritonExpression(gpu::PhysicalExprAttr expression);
bool isTritonFragmentExtent(mlir::Attribute attribute);
std::optional<int64_t> descriptorElementBytes(mlir::Type type);
mlir::FailureOr<llvm::SmallVector<TritonLocalOptions>> declareProviderOptions(
    mlir::func::FuncOp kernel, const gpu::TuningProfiles &profiles,
    bool requiresCtaSynchronization, llvm::ArrayRef<mlir::scf::ForOp> loadPipelineLoops);
mlir::LogicalResult materializeLegalConfigs(
    mlir::func::FuncOp kernel, TensorDescriptorChoiceOp descriptorChoice,
    llvm::ArrayRef<TritonLocalOptions> localOptions);
mlir::LogicalResult finalizeConfigurationRequirements(mlir::func::FuncOp kernel);
} // namespace intent::triton
#endif

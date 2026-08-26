#include "Intent/Target/TileLang/Transforms/Passes.h"

#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Transforms/Passes.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"
#include "PassDetail.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace intent::tilelang {
namespace {

constexpr llvm::StringLiteral legalizedAttr = "intent_tilelang.legalized";

gpu::ParameterOp getOrCreateParameter(func::FuncOp kernel, StringRef name,
                                      gpu::ParameterRole role,
                                      ArrayRef<int64_t> candidates) {
  gpu::ParameterOp existing;
  bool duplicate = false;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getParameter().getName().getValue() != name)
      return;
    if (existing)
      duplicate = true;
    else
      existing = parameter;
  });
  if (duplicate) {
    kernel.emitError("duplicates a TileLang physical parameter name");
    return {};
  }
  auto expectedCandidates =
      DenseI64ArrayAttr::get(kernel.getContext(), candidates);
  if (existing) {
    gpu::ParameterAttr schema = existing.getParameter();
    if (schema.getRole() != static_cast<uint32_t>(role) ||
        schema.getCandidates() != expectedCandidates) {
      existing.emitOpError(
          "physical parameter name is reused with a different role or candidate domain");
      return {};
    }
    return existing;
  }
  OpBuilder builder(&kernel.getBody().front(), kernel.getBody().front().begin());
  auto schema = gpu::ParameterAttr::get(
      kernel.getContext(), builder.getStringAttr(name),
      static_cast<uint32_t>(role), expectedCandidates);
  return builder.create<gpu::ParameterOp>(kernel.getLoc(), builder.getIndexType(),
                                          schema);
}

} // namespace

LogicalResult materializeLaunchConfiguration(func::FuncOp kernel) {
  gpu::ParameterOp threads = getOrCreateParameter(
      kernel, "THREADS", gpu::ParameterRole::ProviderThreads, {128, 256});
  if (!threads)
    return failure();
  SmallVector<LaunchConfigOp> configs;
  kernel.getBody().front().walk(
      [&](LaunchConfigOp config) { configs.push_back(config); });
  if (configs.size() > 1)
    return kernel.emitError(
        "TileLang provider program has duplicate launch configurations");
  if (!configs.empty())
    return configs.front().getThreads() == threads.getResult()
               ? success()
               : configs.front().emitOpError(
                     "uses a conflicting TileLang thread parameter");
  OpBuilder builder(threads);
  builder.setInsertionPointAfter(threads);
  builder.create<LaunchConfigOp>(kernel.getLoc(), threads.getResult());
  return success();
}

LogicalResult verifyTileLangProgram(ModuleOp module) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  return failed(kernel) || failed(mlir::verify(module))
             ? failure()
             : verifyTileLangKernel(*kernel);
}

LogicalResult legalizeGPUProgram(ModuleOp module) {
  if (failed(gpu::verifyGPUProgram(module)))
    return failure();
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel) || failed(materializeLaunchConfiguration(*kernel)) ||
      failed(bufferizeGPUProgram(*kernel)) || failed(formPipelines(*kernel)) ||
      failed(verifyTileLangProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::tilelang

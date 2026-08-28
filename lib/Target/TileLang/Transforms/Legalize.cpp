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

SmallVector<int64_t> extentCandidates(func::FuncOp kernel, Attribute attribute) {
  auto extent = dyn_cast<gpu::PhysicalExprAttr>(attribute);
  if (!extent)
    return {};
  auto kind = static_cast<gpu::PhysicalExprKind>(extent.getKind());
  if (kind == gpu::PhysicalExprKind::Constant)
    return {extent.getValue()};
  if (kind != gpu::PhysicalExprKind::Parameter)
    return {};
  SmallVector<int64_t> result;
  kernel.walk([&](gpu::ParameterOp parameter) {
    if (parameter.getParameter().getName() == extent.getSymbol()) {
      ArrayRef<int64_t> candidates =
          parameter.getParameter().getCandidates().asArrayRef();
      result.assign(candidates.begin(), candidates.end());
    }
  });
  return result;
}

bool hasMmaWarpPartition(int64_t m, int64_t n, int64_t threads) {
  if (threads <= 0 || threads % 32 != 0 || m % 16 != 0 || n % 8 != 0)
    return false;
  int64_t warps = threads / 32;
  for (int64_t mWarps = 1; mWarps <= warps; ++mWarps) {
    if (warps % mWarps != 0)
      continue;
    int64_t nWarps = warps / mWarps;
    if (m % (mWarps * 16) == 0 && n % (nWarps * 8) == 0)
      return true;
  }
  return false;
}

bool supportsThreads(func::FuncOp kernel, int64_t threads) {
  bool sawGemm = false;
  bool supported = true;
  kernel.walk([&](GemmOp gemm) {
    sawGemm = true;
    auto lhs = gemm.getLhs().getType();
    auto rhs = gemm.getRhs().getType();
    Attribute mExtent = lhs.getShape()[gemm.getTransposeLhs() ? 1 : 0];
    Attribute nExtent = rhs.getShape()[gemm.getTransposeRhs() ? 0 : 1];
    SmallVector<int64_t> mCandidates = extentCandidates(kernel, mExtent);
    SmallVector<int64_t> nCandidates = extentCandidates(kernel, nExtent);
    if (mCandidates.empty() || nCandidates.empty()) {
      supported = false;
      return;
    }
    for (int64_t m : mCandidates)
      for (int64_t n : nCandidates)
        if (!hasMmaWarpPartition(m, n, threads)) {
          supported = false;
          return;
        }
  });
  return !sawGemm || supported;
}

} // namespace

LogicalResult materializeLaunchConfiguration(func::FuncOp kernel) {
  SmallVector<int64_t> candidates;
  for (int64_t threads : {128, 256})
    if (supportsThreads(kernel, threads))
      candidates.push_back(threads);
  if (candidates.empty())
    for (int64_t threads : {64, 32})
      if (supportsThreads(kernel, threads)) {
        candidates.push_back(threads);
        break;
      }
  if (candidates.empty())
    return kernel.emitError(
        "TileLang provider found no legal thread count for every native GEMM shape");
  gpu::ParameterOp threads = getOrCreateParameter(
      kernel, "THREADS", gpu::ParameterRole::ProviderThreads, candidates);
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
  if (failed(kernel) || failed(bufferizeGPUProgram(*kernel)) ||
      failed(materializeLaunchConfiguration(*kernel)) ||
      failed(formPipelines(*kernel)) ||
      failed(verifyTileLangProgram(module)))
    return failure();
  (*kernel)->setAttr(legalizedAttr, UnitAttr::get(module.getContext()));
  return success();
}

} // namespace intent::tilelang

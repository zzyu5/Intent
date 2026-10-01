#include "Intent/Target/TileLang/Transforms/Passes.h"
#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Dialect/GPU/IR/GPUDialect.h"
#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
namespace intent::tilelang {

#define GEN_PASS_DEF_TILELANGFORMMEMORY
#define GEN_PASS_DEF_TILELANGCONFIGUREPROGRAM
#define GEN_PASS_DEF_TILELANGFINALIZEPROGRAM
#include "Intent/Target/TileLang/Transforms/Passes.h.inc"

namespace {
LogicalResult finishGroup(ModuleOp module, StringRef name, LogicalResult result) {
  if (failed(result))
    return module.emitError() << "TileLang transformation failed: " << name;
  // Native operations own their local invariants. The full TileLang surface is
  // checked after configuration and cleanup; the shared verifier rejects it.
  if (failed(mlir::verify(module)))
    return module.emitError() << "TileLang structural postcondition failed: " << name;
  return success();
}

struct FormMemoryPass : impl::TileLangFormMemoryBase<FormMemoryPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), formNativeMemory(module))))
      signalPassFailure();
  }
};

struct ConfigureProgramPass : impl::TileLangConfigureProgramBase<ConfigureProgramPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), configureNativeProgram(module))))
      signalPassFailure();
  }
};

struct FinalizeProgramPass : impl::TileLangFinalizeProgramBase<FinalizeProgramPass> {
  void runOnOperation() final {
    auto module = getOperation();
    if (failed(finishGroup(module, getArgument(), finalizeNativeProgram(module))))
      signalPassFailure();
  }
};
} // namespace

void buildTileLangPipeline(OpPassManager &manager) {
  manager.addPass(createTileLangFormMemory());
  manager.addPass(createTileLangConfigureProgram());
  manager.addPass(createTileLangFinalizeProgram());
}

void registerTileLangPipelines() {
  PassPipelineRegistration<>("intent-tilelang", "Legalize a shared GPU program to TileLang",
      [](OpPassManager &manager) { buildTileLangPipeline(manager); });
}

} // namespace intent::tilelang

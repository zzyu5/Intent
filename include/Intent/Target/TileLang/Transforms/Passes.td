#ifndef INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_TD
#define INTENT_TARGET_TILELANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class IntentTileLangPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "::intent::IntentDialect",
    "::intent::gpu::IntentGPUDialect", "::intent::tilelang::IntentTileLangDialect",
    "::mlir::arith::ArithDialect", "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def TileLangFormMemory : IntentTileLangPass<"intent-tilelang-form-memory",
    "Form TileLang buffers and complete native accesses from the shared GPU program">;
def TileLangConfigureProgram : IntentTileLangPass<"intent-tilelang-configure-program",
    "Bind TileLang launch, pipeline and legal configuration facts from IR profiles">;
def TileLangFinalizeProgram : IntentTileLangPass<"intent-tilelang-finalize-program",
    "Simplify the native TileLang program and verify its complete surface">;

#endif

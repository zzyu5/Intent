include "mlir/Pass/PassBase.td"

class IntentCuTileTransformPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = ["::intent::gpu::IntentGPUDialect",
      "::intent::cutile::IntentCuTileDialect", "::mlir::arith::ArithDialect",
      "::mlir::cf::ControlFlowDialect", "::mlir::func::FuncDialect",
      "::mlir::scf::SCFDialect"];
}

def PrepareProgramPass : IntentCuTileTransformPass<
    "intent-cutile-prepare-program",
    "Prepare cuTile matrix shapes, buffers and workspace accesses">;

def NativeProgramPass : IntentCuTileTransformPass<
    "intent-cutile-native-program",
    "Form native cuTile accesses and computations from the current GPU program">;

def FinalizeProgramPass : IntentCuTileTransformPass<
    "intent-cutile-finalize-program",
    "Close cuTile configurations, loops and array views and verify the surface">;

#ifndef INTENT_TARGET_MOJO_TRANSFORMS_PASSES_TD
#define INTENT_TARGET_MOJO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

class MojoTransform<string argument, string summaryText> : Pass<argument, "mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = [
    "intent::cpu::IntentCPUDialect", "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect", "mlir::math::MathDialect", "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect", "mlir::vector::VectorDialect"
  ];
}

def MojoMaterializeProgram : MojoTransform<"intent-mojo-materialize-program",
    "Expand selected CPU implementations and materialize native task dispatch">;
def MojoFusePrivateComputations : MojoTransform<"intent-mojo-fuse-private-computations",
    "Replay private computations under coordinate and storage lifetime proofs">;
def MojoVectorizeProgram : MojoTransform<"intent-mojo-vectorize-program",
    "Materialize vector traversals using coordinated implementation bindings">;
def MojoFinalizeProgram : MojoTransform<"intent-mojo-finalize-program",
    "Legalize the Mojo surface and remove equivalent finalized candidates">;

#endif

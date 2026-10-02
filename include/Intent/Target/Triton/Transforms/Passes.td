include "mlir/Pass/PassBase.td"

class IntentTritonTransformPass<string passArgument, string summaryText>
    : Pass<passArgument, "::mlir::ModuleOp"> {
  let summary = summaryText;
  let dependentDialects = ["::intent::gpu::IntentGPUDialect",
      "::intent::triton::IntentTritonDialect", "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::scf::SCFDialect"];
}

def ProgramGridPass : IntentTritonTransformPass<
    "intent-triton-program-grid",
    "Legalize the Triton program grid">;

def PrepareMemoryPass : IntentTritonTransformPass<
    "intent-triton-prepare-memory",
    "Prepare Triton buffers, indexed accesses and matrix shapes">;

def NativeFormsPass : IntentTritonTransformPass<
    "intent-triton-native-forms",
    "Form native Triton accesses, supply and complete configurations">;

def FinalizeProgramPass : IntentTritonTransformPass<
    "intent-triton-finalize-program",
    "Close Triton callbacks and resource constraints and verify the surface">;

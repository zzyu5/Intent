include "mlir/Pass/PassBase.td"

def WeftPrepareTaskViews : Pass<"intent-weft-prepare-task-views", "mlir::ModuleOp"> {
  let summary = "Reify typed task views and close the captures for Weft conversion";
  let dependentDialects = ["intent::cpu::IntentCPUDialect",
      "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
      "mlir::memref::MemRefDialect"];
}

def WeftConvertProgram : Pass<"intent-weft-convert-program", "mlir::ModuleOp"> {
  let summary = "Convert CPU tasks into an explicit host and Canonical Weft program";
  let dependentDialects = ["intent::cpu::IntentCPUDialect",
      "intent::weft_provider::IntentWeftDialect", "weft::kernel::WEFTKernelDialect",
      "mlir::arith::ArithDialect", "mlir::func::FuncDialect", "mlir::linalg::LinalgDialect",
      "mlir::math::MathDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"];
}

def WeftVerifyProgram : Pass<"intent-weft-verify-program", "mlir::ModuleOp"> {
  let summary = "Verify the complete Weft host, device and task interface";
  let dependentDialects = ["intent::cpu::IntentCPUDialect",
      "intent::weft_provider::IntentWeftDialect", "weft::kernel::WEFTKernelDialect"];
}

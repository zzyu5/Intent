add_mlir_library(MLIRIntentGPUAnalysis
  PhysicalProgram.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
)

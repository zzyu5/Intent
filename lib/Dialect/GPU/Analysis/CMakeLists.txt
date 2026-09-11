add_mlir_library(MLIRIntentGPUAnalysis
  PhysicalProgram.cpp
  UniformValues.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentAnalysis
)

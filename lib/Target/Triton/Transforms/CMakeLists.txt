add_mlir_library(MLIRIntentTritonTransforms
  Legalize.cpp
  ProgramGrid.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
)

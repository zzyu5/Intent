add_mlir_library(MLIRIntentTritonTransforms
  Legalize.cpp
  Configurations.cpp
  ProgramGrid.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTritonDialect
)

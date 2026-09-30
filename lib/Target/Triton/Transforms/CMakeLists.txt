add_mlir_library(MLIRIntentTritonTransforms
  Legalize.cpp
  Passes.cpp
  AccessForms.cpp
  Collectives.cpp
  Supply.cpp
  Values.cpp
  Verify.cpp
  Configurations.cpp
  ProgramGrid.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTritonDialect
)

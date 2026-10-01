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

  DEPENDS
  IntentTritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTritonDialect
)

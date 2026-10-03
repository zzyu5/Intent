add_mlir_library(MLIRIntentTritonTransforms
  Legalize.cpp
  Passes.cpp
  AccessForms.cpp
  Collectives.cpp
  Supply.cpp
  Values.cpp
  Configurations.cpp
  ConfigurationFacts.cpp
  ProgramGrid.cpp

  DEPENDS
  IntentTritonTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTritonDialect
  MLIRIntentTritonAnalysis
  MLIRIntentTritonSerialization
)

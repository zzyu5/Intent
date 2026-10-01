add_mlir_library(MLIRIntentTileLangTransforms
  Bufferize.cpp
  FormPipeline.cpp
  Legalize.cpp
  Passes.cpp
  Verify.cpp

  DEPENDS
  IntentTileLangPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentDialect
  MLIRIntentGPUTransforms
  MLIRIntentTileLangDialect
)

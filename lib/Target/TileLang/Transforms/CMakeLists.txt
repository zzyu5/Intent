add_mlir_library(MLIRIntentTileLangTransforms
  Bufferize.cpp
  FormPipeline.cpp
  Legalize.cpp
  Verify.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentTileLangDialect
)

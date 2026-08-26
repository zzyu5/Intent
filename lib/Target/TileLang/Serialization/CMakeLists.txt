add_mlir_library(MLIRIntentTileLangSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentTileLangDialect
)

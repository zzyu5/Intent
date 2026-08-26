add_mlir_library(MLIRIntentCuTileSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentCuTileDialect
  MLIRIntentGPUDialect
)

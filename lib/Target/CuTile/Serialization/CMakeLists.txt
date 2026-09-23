add_mlir_library(MLIRIntentCuTileSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentCuTileDialect
  MLIRIntentGPUDialect
)

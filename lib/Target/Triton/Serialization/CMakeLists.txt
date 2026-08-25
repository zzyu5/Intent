add_mlir_library(MLIRIntentTritonSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
)

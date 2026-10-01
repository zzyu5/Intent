add_mlir_library(MLIRIntentTritonSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentGPUSerialization
  MLIRIntentTritonDialect
)

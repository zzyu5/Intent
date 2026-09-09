add_mlir_library(MLIRIntentMojoSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
)

add_mlir_library(MLIRIntentMojoSerialization
  Serializer.cpp

  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRIntentNativeABI
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
)

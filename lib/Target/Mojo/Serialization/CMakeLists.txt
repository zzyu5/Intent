add_mlir_library(MLIRIntentMojoScalarSerialization
  Scalar.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC MLIRIntentScalarSerialization MLIRIntentSourceSerialization
)

add_mlir_library(MLIRIntentMojoSerialization
  Serializer.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRIntentNativeABI
  MLIRIntentMojoScalarSerialization
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
)

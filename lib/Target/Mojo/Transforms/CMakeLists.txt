add_mlir_library(MLIRIntentMojoTransforms
  Legalize.cpp
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

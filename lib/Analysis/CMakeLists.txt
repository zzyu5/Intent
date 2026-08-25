add_mlir_library(MLIRIntentAnalysis
  CanonicalKernel.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
)

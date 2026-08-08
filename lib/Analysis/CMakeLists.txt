add_mlir_library(MLIRIntentAnalysis
  StableSoftmax.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRSupport
)

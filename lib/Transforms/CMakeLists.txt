add_mlir_library(MLIRIntentTransforms
  VerifyKernelIR.cpp

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)

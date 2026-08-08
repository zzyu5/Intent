add_mlir_library(MLIRIntentTransforms
  VerifyKernelIR.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)

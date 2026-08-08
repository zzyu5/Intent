add_mlir_library(MLIRIntentTransforms
  RealizeStableSoftmax.cpp
  VerifyKernelIR.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentAnalysis
  MLIRIntentPlanDialect
  MLIRPass
  MLIRSupport
)

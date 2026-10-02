add_mlir_library(MLIRIntentKIRToCPU
  KIRToCPU.cpp

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentScalarLowering
  MLIRIntentDialect
  MLIRIntentCPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)

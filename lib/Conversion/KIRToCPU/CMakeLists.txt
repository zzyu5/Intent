add_mlir_library(MLIRIntentKIRToCPU
  KIRToCPU.cpp

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)

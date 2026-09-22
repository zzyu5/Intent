add_mlir_conversion_library(MLIRIntentKIRToDSA KIRToDSA.cpp
  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentDialect MLIRIntentDSADialect
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
    MLIRIR MLIRSupport)

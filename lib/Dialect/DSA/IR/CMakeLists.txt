add_mlir_dialect_library(MLIRIntentDSADialect
  DSADialect.cpp DSAOps.cpp
  DEPENDS IntentDSAIncGen
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRIR MLIRSupport MLIRSideEffectInterfaces
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect)

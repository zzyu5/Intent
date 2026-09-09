add_mlir_library(MLIRIntentWeftTransforms
  Legalize.cpp
  LINK_LIBS PUBLIC
  IntentWeftCanonical
  MLIRIntentCPUAnalysis
  MLIRIntentCPUTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)

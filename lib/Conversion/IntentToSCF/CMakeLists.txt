add_mlir_conversion_library(MLIRIntentToSCF
  IntentToSCF.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
)

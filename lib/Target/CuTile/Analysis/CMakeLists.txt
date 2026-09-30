add_mlir_library(MLIRIntentCuTileAnalysis
  Tuning.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRIntentCuTileDialect
  MLIRIntentGPUAnalysis
)

add_mlir_library(MLIRIntentDSAAnalysis
  PhysicalProgram.cpp

  LINK_LIBS PUBLIC
  MLIRIntentDSADialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)

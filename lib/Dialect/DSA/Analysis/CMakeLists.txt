add_mlir_library(MLIRIntentDSAAnalysis
  PhysicalProgram.cpp
  Storage.cpp
  StorageQueries.cpp

  LINK_LIBS PUBLIC
  MLIRIntentDSADialect
  MLIRIntentAnalysis
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
)

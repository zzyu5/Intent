add_mlir_dialect_library(MLIRIntentWeftDialect
  WeftDialect.cpp
  Program.cpp
  DEPENDS IntentWeftIncGen
  LINK_LIBS PUBLIC
  IntentWeftCanonical
  MLIRIntentCPUAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)

add_mlir_dialect_library(MLIRIntentWeftDialect
  WeftDialect.cpp
  Program.cpp
  HostScalar.cpp
  DEPENDS IntentWeftIncGen
  LINK_LIBS PUBLIC
  IntentWeftCanonical
  MLIRIntentCPUAnalysis
  MLIRIntentScalarSerialization
  MLIRIntentSourceSerialization
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)

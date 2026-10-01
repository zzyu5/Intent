add_mlir_library(MLIRIntentWeftTransforms
  Legalize.cpp
  Implementations.cpp
  Quantization.cpp
  TaskInterface.cpp
  Views.cpp
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

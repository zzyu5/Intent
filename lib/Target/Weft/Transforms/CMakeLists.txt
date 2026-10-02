add_mlir_library(MLIRIntentWeftTransforms
  Legalize.cpp
  Passes.cpp
  Implementations.cpp
  Quantization.cpp
  Reductions.cpp
  TaskInterface.cpp
  TaskLowering.cpp
  Views.cpp
  DEPENDS IntentWeftTransformsIncGen
  LINK_LIBS PUBLIC
  IntentWeftCanonical
  MLIRIntentWeftDialect
  MLIRIntentCPUAnalysis
  MLIRIntentCPUTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
)

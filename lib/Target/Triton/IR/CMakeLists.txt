add_mlir_dialect_library(MLIRIntentTritonDialect
  TritonDialect.cpp
  TritonOps.cpp
  Configuration.cpp

  DEPENDS
  IntentTritonIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPUDialect
  MLIRIntentGPUAnalysis
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRViewLikeInterface
  MLIRSCFDialect
  MLIRSupport
)

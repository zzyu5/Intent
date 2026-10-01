add_mlir_dialect_library(MLIRIntentGPUDialect
  GPUDialect.cpp
  GPUOps.cpp
  AccessOpInterface.cpp

  DEPENDS
  IntentGPUIncGen
  IntentStructuredOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIntentDialect
  MLIRIntentStructuredOpInterface
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

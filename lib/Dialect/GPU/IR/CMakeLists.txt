add_mlir_dialect_library(MLIRIntentGPUDialect
  GPUDialect.cpp
  GPUOps.cpp

  DEPENDS
  IntentGPUIncGen

  LINK_LIBS PUBLIC
  MLIRIntentDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

add_mlir_dialect_library(MLIRIntentGPUDialect
  GPUDialect.cpp
  ProgramInterface.cpp
  PhysicalExpressions.cpp
  TypeVerification.cpp
  GPUOps.cpp
  Canonicalization.cpp
  AccessOpInterface.cpp

  DEPENDS
  IntentGPUIncGen
  IntentStructuredOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIntentDialect
  MLIRIntentStructuredOpInterface
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

add_mlir_dialect_library(MLIRIntentDialect
  IntentDialect.cpp
  IntentOps.cpp
  ControlFlow.cpp

  DEPENDS
  IntentIncGen
  IntentStructuredOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentStructuredOpInterface
  MLIRPresburger
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
)

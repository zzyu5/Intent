add_mlir_dialect_library(MLIRIntentDialect
  IntentDialect.cpp
  IntentOps.cpp

  DEPENDS
  IntentIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
)

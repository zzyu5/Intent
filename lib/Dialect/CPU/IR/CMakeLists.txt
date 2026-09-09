add_mlir_dialect_library(MLIRIntentCPUDialect
  CPUDialect.cpp
  CPUOps.cpp
  DEPENDS IntentCPUIncGen
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

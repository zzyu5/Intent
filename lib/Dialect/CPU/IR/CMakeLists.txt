add_mlir_dialect_library(MLIRIntentCPUDialect
  CPUDialect.cpp
  CPUOps.cpp
  RegionProgram.cpp
  DEPENDS IntentCPUIncGen
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

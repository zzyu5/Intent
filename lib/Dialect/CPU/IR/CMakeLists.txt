add_mlir_dialect_library(MLIRIntentCPUDialect
  CPUDialect.cpp
  CPUOps.cpp
  RegionProgram.cpp
  DEPENDS IntentCPUIncGen
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

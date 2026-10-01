add_mlir_dialect_library(MLIRIntentCPUDialect
  CPUDialect.cpp
  CPUOps.cpp
  CPUOpInterfaces.cpp
  DEPENDS IntentCPUIncGen
  LINK_LIBS PUBLIC
  MLIRIntentDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)

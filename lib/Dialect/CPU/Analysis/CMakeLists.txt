add_mlir_library(MLIRIntentCPUAnalysis
  PhysicalProgram.cpp
  LINK_LIBS PUBLIC
  MLIRIntentCPUDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRSideEffectInterfaces
)

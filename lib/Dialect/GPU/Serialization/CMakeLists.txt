add_mlir_library(MLIRIntentGPUSerialization
  Interface.cpp
  Python.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
)

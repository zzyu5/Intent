add_mlir_library(MLIRIntentGPUAnalysis
  PhysicalProgram.cpp
  IndexPredicates.cpp
  PhysicalParameters.cpp
  Resources.cpp
  ValueSchema.cpp
  UniformValues.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
  MLIRIntentAnalysis
)

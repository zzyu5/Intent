add_mlir_library(MLIRIntentGPUAnalysis
  PhysicalProgram.cpp
  ProgramInterface.cpp
  IndexPredicates.cpp
  IndexRelations.cpp
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

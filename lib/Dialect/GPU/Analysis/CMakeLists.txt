add_mlir_library(MLIRIntentGPUAnalysis
  PhysicalProgram.cpp
  ProgramInterface.cpp
  IndexPredicates.cpp
  IndexRelations.cpp
  PhysicalParameters.cpp
  ResourceAlias.cpp
  Resources.cpp
  ValueSchema.cpp
  UniformValues.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRControlFlowInterfaces
  MLIRViewLikeInterface
  MLIRIntentGPUDialect
  MLIRIntentAnalysis
)

add_mlir_library(MLIRIntentCPUAnalysis
  PhysicalProgram.cpp
  AxisRelations.cpp
  RegionPartition.cpp
  RegionPredicates.cpp
  LINK_LIBS PUBLIC
  MLIRIntentCPUDialect
  MLIRIntentAnalysis
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRSideEffectInterfaces
)

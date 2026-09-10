add_mlir_library(MLIRIntentCPUAnalysis
  PhysicalProgram.cpp
  AxisRelations.cpp
  RegionPartition.cpp
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

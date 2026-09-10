add_mlir_library(MLIRIntentCPUTransforms
  FuseStructuredComputations.cpp
  MaterializeStructuredComputations.cpp
  FuseIntermediateBuffers.cpp
  FuseReductionTraversals.cpp
  BlockContractions.cpp
  VectorizeLoops.cpp
  PartitionTasks.cpp
  Implementation.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRIntentCPUAnalysis
  MLIRIntentCPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

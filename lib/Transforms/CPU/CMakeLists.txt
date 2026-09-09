add_mlir_library(MLIRIntentCPUTransforms
  FuseIntermediateBuffers.cpp
  BlockContractions.cpp
  VectorizeLoops.cpp
  PartitionTasks.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

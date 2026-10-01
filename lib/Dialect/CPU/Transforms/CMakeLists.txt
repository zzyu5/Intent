add_mlir_library(MLIRIntentCPUTransforms
  FuseStructuredComputations.cpp
  IntegerSources.cpp
  ContiguousAccesses.cpp
  FoldUniformComputations.cpp
  ReusePreparedInputs.cpp
  GroupQuantizedDots.cpp
  GroupRegionComputations.cpp
  GroupWorksetComputations.cpp
  MaterializeStructuredComputations.cpp
  FuseIntermediateBuffers.cpp
  FuseReductionTraversals.cpp
  BlockContractions.cpp
  NormalizeContractions.cpp
  ContractionSources.cpp
  BlockStructuredComputations.cpp
  RealizeRegions.cpp
  VectorizeLoops.cpp
  VectorReductions.cpp
  PartitionTasks.cpp
  Implementation.cpp
  ImplementationInputs.cpp
  TuningProfiles.cpp
  Configurations.cpp
  FinalizedCandidates.cpp
  Passes.cpp

  DEPENDS IntentCPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIntentCPUAnalysis
  MLIRIntentAnalysis
  MLIRIntentCPUDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

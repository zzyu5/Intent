add_mlir_library(MLIRIntentGPUTransforms
  RealizeContractionBlocking.cpp
  RealizeAccessComposition.cpp
  RealizePointwiseBlocking.cpp
  RealizeOnlineReduction.cpp
  RealizeRegionFold.cpp
  RealizeReductionBlocking.cpp
  RefineProgramMapping.cpp
  MaterializeConfigTuples.cpp
  OnlineSummary.cpp
  Passes.cpp
  Utilities.cpp
  VerifyGPUProgram.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUAnalysis
  MLIRIntentGPUDialect
)

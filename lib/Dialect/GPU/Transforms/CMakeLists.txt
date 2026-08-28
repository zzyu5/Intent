add_mlir_library(MLIRIntentGPUTransforms
  RealizeContractionBlocking.cpp
  RealizeAccessComposition.cpp
  RealizePointwiseBlocking.cpp
  RealizeRegionFold.cpp
  RealizeReductionBlocking.cpp
  Passes.cpp
  Utilities.cpp
  VerifyGPUProgram.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentGPUDialect
)

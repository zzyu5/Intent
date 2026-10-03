add_mlir_library(MLIRIntentCuTileTransforms
  Legalize.cpp
  Control/Loops.cpp
  Passes.cpp
  NativeProgram.cpp
  NativeAccessAnalysis.cpp
  NativeAccess.cpp
  ComputeForms.cpp
  Configurations.cpp
  CollapseArrayViews.cpp
  RefineMMALoops.cpp

  DEPENDS
  IntentCuTileTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRIntentCuTileDialect
  MLIRIntentCuTileAnalysis
  MLIRIntentCuTileSerialization
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
)

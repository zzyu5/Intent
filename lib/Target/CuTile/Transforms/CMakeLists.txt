add_mlir_library(MLIRIntentCuTileTransforms
  Legalize.cpp
  Control/Loops.cpp
  Passes.cpp
  NativeProgram.cpp
  NativeRewrite.cpp
  Access/Accesses.cpp
  Access/AccessFormSelection.cpp
  Access/Coordinates.cpp
  Access/TilePlan.cpp
  Access/Bounds.cpp
  Access/TileIndices.cpp
  Access/Loads.cpp
  Access/Extraction.cpp
  Access/Atomics.cpp
  Access/Stores.cpp
  Compute/ComputeForms.cpp
  Configuration/Configurations.cpp
  Access/CollapseArrayViews.cpp
  Compute/RefineMMALoops.cpp

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

add_mlir_library(MLIRIntentCuTileTransforms
  Legalize.cpp
  Configurations.cpp
  CollapseArrayViews.cpp
  RefineMMALoops.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentCuTileDialect
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
)

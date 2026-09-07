add_mlir_library(MLIRIntentCuTileTransforms
  Legalize.cpp
  CollapseArrayViews.cpp
  RefineMMALoops.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentCuTileDialect
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
)

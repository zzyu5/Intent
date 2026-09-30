add_mlir_library(MLIRIntentCuTileTransforms
  Legalize.cpp
  Passes.cpp
  NativeProgram.cpp
  NativeAccessAnalysis.cpp
  NativeAccess.cpp
  ComputeForms.cpp
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

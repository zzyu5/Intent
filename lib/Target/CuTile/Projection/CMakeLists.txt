add_mlir_library(MLIRIntentCuTileProjection
  Project.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentCuTileDialect
  MLIRIntentGPUProjection
  MLIRIntentPlanDialect
  MLIRSupport
)

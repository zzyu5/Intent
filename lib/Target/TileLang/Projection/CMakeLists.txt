add_mlir_library(MLIRIntentTileLangProjection
  Project.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPUProjection
  MLIRIntentPlanDialect
  MLIRIntentTileLangDialect
  MLIRSupport
)

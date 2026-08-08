add_mlir_library(MLIRIntentTritonProjection
  Project.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPUProjection
  MLIRIntentPlanDialect
  MLIRIntentTritonDialect
  MLIRSupport
)

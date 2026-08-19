add_mlir_library(MLIRIntentGPUTransforms
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPURealization
  MLIRIntentPlanDialect
  MLIRPass
  MLIRSupport
)

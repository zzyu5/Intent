add_mlir_library(MLIRIntentGPUProjection
  MachinePlan.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentPlanDialect
  MLIRSupport
)

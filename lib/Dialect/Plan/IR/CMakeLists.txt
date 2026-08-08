add_mlir_dialect_library(MLIRIntentPlanDialect
  PlanDialect.cpp
  PlanOps.cpp

  DEPENDS
  IntentPlanIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRBytecodeOpInterface
  MLIRIR
  MLIRSupport
)

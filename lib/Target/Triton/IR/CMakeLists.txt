add_mlir_dialect_library(MLIRIntentTritonDialect
  TritonDialect.cpp
  TritonOps.cpp

  DEPENDS
  IntentTritonIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRSupport
)

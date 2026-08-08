add_mlir_dialect_library(MLIRIntentCuTileDialect
  CuTileDialect.cpp
  CuTileOps.cpp

  DEPENDS
  IntentCuTileIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentPlanDialect
  MLIRIntentTargetCommon
  MLIRSupport
)

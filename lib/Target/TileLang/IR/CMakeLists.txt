add_mlir_dialect_library(MLIRIntentTileLangDialect
  TileLangDialect.cpp
  TileLangOps.cpp

  DEPENDS
  IntentTileLangIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPUDialect
  MLIRSideEffectInterfaces
  MLIRSupport
)

add_mlir_library(MLIRIntentTritonEmission
  Emitter.cpp
  Translate.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRIntentTritonDialect
  MLIRSupport
)

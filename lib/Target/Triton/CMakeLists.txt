add_mlir_library(MLIRIntentTritonTarget
  StableSoftmax.cpp
  Translate.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentAnalysis
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRSupport
)

add_mlir_library(MLIRIntentTritonRealization
  Rowwise.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentDialect
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRIntentTritonDialect
  MLIRSupport
)

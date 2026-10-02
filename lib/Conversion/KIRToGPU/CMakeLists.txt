add_mlir_library(MLIRIntentKIRToGPU
  KIRToGPU.cpp
  ConstructionSchema.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRIntentAnalysis
  MLIRIntentGPUDialect
  MLIRIntentGPUTransforms
  MLIRIntentDialect
  MLIRIntentTransforms
)

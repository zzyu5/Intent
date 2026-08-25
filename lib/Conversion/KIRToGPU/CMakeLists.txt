add_mlir_library(MLIRIntentKIRToGPU
  KIRToGPU.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentTransforms
)

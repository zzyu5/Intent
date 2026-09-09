add_mlir_library(MLIRIntentMojoTransforms
  Legalize.cpp
  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRIR
)

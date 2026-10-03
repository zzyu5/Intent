add_mlir_library(MLIRIntentKIRToCPU
  KIRToCPU.cpp
  Construction.cpp
  Values.cpp
  Access.cpp
  Arithmetic.cpp
  Tensor.cpp
  Collectives.cpp
  Contractions.cpp
  ControlFlow.cpp

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentConversionSupport
  MLIRIntentDialect
  MLIRIntentCPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRIR
)

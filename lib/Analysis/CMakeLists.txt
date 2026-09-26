add_mlir_library(MLIRIntentAnalysis
  CanonicalKernel.cpp
  UniformValues.cpp
  RegionSemantics.cpp
  OnlineSummary.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRIR
  MLIRIntentDialect
)

add_mlir_library(MLIRIntentAnalysis
  CanonicalKernel.cpp
  ContractionAxes.cpp
  UniformValues.cpp
  IntegerRelations.cpp
  RegionSemantics.cpp
  OnlineSummary.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRIR
  MLIRIntentDialect
)

add_mlir_library(MLIRIntentAnalysis
  CanonicalKernel.cpp
  ContractionAxes.cpp
  ControlFlow.cpp
  UniformValues.cpp
  IntegerRelations.cpp
  RegionSemantics.cpp
  OnlineSummary.cpp
  ProductSchema.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRIntentDialect
  MLIRIntentStructuredOpInterface
)

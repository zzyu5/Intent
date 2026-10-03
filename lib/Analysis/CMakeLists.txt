add_mlir_library(MLIRIntentAnalysis
  CanonicalKernel.cpp
  BufferStorage.cpp
  ContractionAxes.cpp
  ControlFlow.cpp
  UniformValues.cpp
  IntegerRelations.cpp
  RegionSemantics.cpp
  OnlineSummary.cpp
  ProductSchema.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRAnalysis
  MLIRBufferizationTransforms
  MLIRBufferizationDialect
  MLIRCallInterfaces
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  MLIRArithDialect
  MLIRMathDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRIntentDialect
  MLIRIntentStructuredOpInterface
)

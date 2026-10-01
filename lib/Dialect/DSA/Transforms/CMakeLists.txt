add_mlir_library(MLIRIntentDSATransforms
  Passes.cpp
  MatrixSupply.cpp
  CollectiveGather.cpp
  IndexPrograms.cpp
  LocalValues.cpp

  DEPENDS
  IntentDSAPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentDSAAnalysis
  MLIRIntentDSADialect
  MLIRIntentDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
)

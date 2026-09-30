add_mlir_library(MLIRIntentDSATransforms
  Passes.cpp
  MatrixSupply.cpp
  CollectiveGather.cpp
  IndexPrograms.cpp
  LocalValues.cpp

  LINK_LIBS PUBLIC
  MLIRIntentAnalysis
  MLIRIntentDSAAnalysis
  MLIRIntentDSADialect
  MLIRPass
  MLIRTransforms
)

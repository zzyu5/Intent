add_mlir_library(MLIRIntentMojoTransforms
  Legalize.cpp
  Implementations.cpp
  MaterializeRegisterContractions.cpp
  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRLinalgDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRIR
)

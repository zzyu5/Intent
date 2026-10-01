add_mlir_library(MLIRIntentMojoTransforms
  Legalize.cpp
  Passes.cpp
  Implementations.cpp
  MaterializeRegisterContractions.cpp
  DEPENDS IntentMojoTransformsIncGen
  LINK_LIBS PUBLIC
  MLIRIntentCPUTransforms
  MLIRPass
  MLIRArithDialect
  MLIRArithTransforms
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

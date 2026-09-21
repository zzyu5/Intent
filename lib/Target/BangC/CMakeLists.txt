add_mlir_library(MLIRIntentBangC
  Transforms/Legalize.cpp Serialization/Serializer.cpp
  LINK_LIBS PUBLIC MLIRIntentDSADialect MLIRArithDialect MLIRFuncDialect
    MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect MLIRTransforms MLIRPass MLIRIR MLIRSupport)

add_mlir_library(MLIRIntentBangC
  Transforms/Legalize.cpp
  Transforms/Matrix.cpp
  Transforms/Numerics.cpp
  Transforms/IndexLowering.cpp
  Transforms/Supply.cpp
  Transforms/Storage.cpp
  Transforms/Verifier.cpp
  Serialization/Serializer.cpp
  LINK_LIBS PUBLIC MLIRIntentDSADialect MLIRIntentDSAAnalysis MLIRIntentDSATransforms MLIRArithDialect MLIRFuncDialect
    MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect MLIRTransforms MLIRPass MLIRIR MLIRSupport)

add_mlir_library(MLIRIntentBangC
  Transforms/Legalize.cpp
  Transforms/Matrix.cpp
  Transforms/Numerics.cpp
  Transforms/IndexLowering.cpp
  Transforms/Supply.cpp
  Transforms/InvariantSupply.cpp
  Transforms/Storage.cpp
  Transforms/Verifier.cpp
  Serialization/Serializer.cpp
  Serialization/Scalar.cpp
  DEPENDS IntentBangCPassesIncGen
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRIntentNativeABI MLIRIntentScalarSerialization MLIRIntentNativeSourceSerialization MLIRIntentDSADialect MLIRIntentDSAAnalysis MLIRIntentDSATransforms MLIRArithDialect MLIRFuncDialect
    MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect MLIRTransforms MLIRPass MLIRIR MLIRSupport)

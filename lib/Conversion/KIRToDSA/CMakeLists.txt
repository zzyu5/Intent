add_mlir_conversion_library(MLIRIntentKIRToDSA
  KIRToDSA.cpp
  Construction.cpp
  Values.cpp
  Tensors.cpp
  Access.cpp
  AccessCoordinates.cpp
  Contractions.cpp
  Collectives.cpp
  CollectiveConstruction.cpp
  Regions.cpp
  Worksets.cpp

  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentConversionSupport MLIRIntentDialect MLIRIntentDSADialect
    MLIRIntentDSAAnalysis
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
    MLIRIR MLIRSupport MLIRTransformUtils)

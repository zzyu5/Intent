add_mlir_conversion_library(MLIRIntentKIRToDSA
  KIRToDSA.cpp
  Construction.cpp
  Values.cpp
  Tensors.cpp
  Access.cpp
  Contractions.cpp
  Collectives.cpp
  Regions.cpp
  Worksets.cpp

  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentDialect MLIRIntentDSADialect
    MLIRArithDialect MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
    MLIRIR MLIRSupport)

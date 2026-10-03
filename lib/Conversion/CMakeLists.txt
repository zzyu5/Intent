add_mlir_library(MLIRIntentConversionSupport
  ScalarLowering.cpp
  LogicalShape.cpp
  LINK_LIBS PUBLIC MLIRIntentAnalysis MLIRIntentDialect MLIRArithDialect MLIRMathDialect MLIRIR)

add_subdirectory(KIRToGPU)
add_subdirectory(KIRToCPU)
add_subdirectory(KIRToDSA)

add_mlir_library(MLIRIntentScalarLowering
  ScalarLowering.cpp
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRArithDialect MLIRMathDialect MLIRIR)

add_subdirectory(KIRToGPU)
add_subdirectory(KIRToCPU)
add_subdirectory(KIRToDSA)

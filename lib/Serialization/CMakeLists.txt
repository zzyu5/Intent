add_mlir_library(MLIRIntentNativeABI
  NativeABI.cpp
  LINK_LIBS PUBLIC MLIRIntentDialect MLIRFuncDialect MLIRIR MLIRSupport
)

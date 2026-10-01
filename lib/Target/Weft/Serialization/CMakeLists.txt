add_mlir_library(MLIRIntentWeftSerialization
  Serializer.cpp
  HostSerializer.cpp
  TaskABI.cpp
  LINK_LIBS PUBLIC IntentWeftCanonical MLIRIntentWeftDialect MLIRIntentCPUDialect MLIRIntentNativeABI MLIRArithDialect
  MLIRFuncDialect MLIRMemRefDialect MLIRSCFDialect MLIRIR
)

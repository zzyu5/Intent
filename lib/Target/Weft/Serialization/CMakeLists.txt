add_mlir_library(MLIRIntentWeftSerialization
  Serializer.cpp
  HostSerializer.cpp
  LINK_LIBS PUBLIC IntentWeftCanonical MLIRIntentCPUDialect MLIRArithDialect
  MLIRFuncDialect MLIRMemRefDialect MLIRSCFDialect MLIRIR
)

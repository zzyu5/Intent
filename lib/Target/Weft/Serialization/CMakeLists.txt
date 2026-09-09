add_mlir_library(MLIRIntentWeftSerialization
  Serializer.cpp
  LINK_LIBS PUBLIC IntentWeftCanonical MLIRIR
)

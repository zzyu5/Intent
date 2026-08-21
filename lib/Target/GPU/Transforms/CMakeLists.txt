add_mlir_library(MLIRIntentGPUTransforms
  Access/BoundaryNeutralization.cpp
  Analysis/PhysicalProgram.cpp
  Execution/PersistentTraversal.cpp
  Passes.cpp
  Value/PrivateBufferResidency.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPURealization
  MLIRIntentPlanDialect
  MLIRPass
  MLIRSupport
)

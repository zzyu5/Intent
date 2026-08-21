add_mlir_library(MLIRIntentGPUTransforms
  Access/BoundaryNeutralization.cpp
  Access/AccessRanges.cpp
  Access/TransferRealization.cpp
  Analysis/PhysicalProgram.cpp
  Execution/PersistentTraversal.cpp
  Execution/AutomaticBlocking.cpp
  Passes.cpp
  Search/SearchSpace.cpp
  Structured/StageFormation.cpp
  Value/PrivateBufferResidency.cpp
  Value/ContractionRealization.cpp
  Value/ScanRealization.cpp
  Value/OperationRealization.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRIntentGPURealization
  MLIRIntentPlanDialect
  MLIRPass
  MLIRSupport
)

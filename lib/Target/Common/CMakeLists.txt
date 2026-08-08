add_mlir_library(MLIRIntentTargetCommon
  Analysis/IndexRelation.cpp
  Analysis/Kernel.cpp
  Emission/Driver.cpp
  Emission/Lifecycle.cpp
  Projection/Capabilities.cpp
  Realization/Driver.cpp
  Realization/KernelFacts.cpp
  Realization/Proofs.cpp
  Realization/SchedulePolicy.cpp
  Realization/ScheduleStructure.cpp
  Traversal/OperationRegistry.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRSupport
)

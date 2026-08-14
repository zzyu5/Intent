add_mlir_library(MLIRIntentTargetCommon
  Analysis/IndexRelation.cpp
  Analysis/Kernel.cpp
  Emission/Combiner.cpp
  Emission/Driver.cpp
  Emission/Lifecycle.cpp
  Realization/Driver.cpp
  Realization/KernelFacts.cpp
  Realization/Proofs.cpp
  Traversal/OperationRegistry.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRIntentPlanDialect
  MLIRIntentTransforms
  MLIRSupport
)

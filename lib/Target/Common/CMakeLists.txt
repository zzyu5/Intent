add_mlir_library(MLIRIntentTargetCommon
  Analysis/IndexRelation.cpp
  Analysis/Kernel.cpp
  Lowering/Combiner.cpp
  Lowering/Driver.cpp
  Lowering/Lifecycle.cpp
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
